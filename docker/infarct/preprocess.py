#!/usr/bin/env python3
"""Infarct Preprocessing — SynthSeg, the two derivations, and skull-strip.

Layer 1 of 3. Runs OUTSIDE the Redis rad_gpu_0 lock, which is why SynthSeg is
called from here rather than from inference.

Aneurysm cannot do that: its SynthSeg call (Stage C.5) is buried inside legacy
inference code, which runs inside the lock, and the SynthSeg container takes the
same lock (LAZY_GPU_SWAP=1). Redis SET NX is not reentrant, so C.5 blocked until
its own lock expired -- 292 wasted seconds per study, and the neck filter never
once applied. The fix there was a dispatcher-side prewarm with a byte-identical
payload so the in-lock call becomes a cache hit. This container is new code, so
it can simply make the call where no lock is held and skip that indirection --
along with the failure mode where a one-character difference in output_dir turns
the cache hit back into a deadlock.

Two derivations come out of the single SynthSeg run. Neither is recoverable from
the other, and they answer different questions:

  aparc+aseg (99 FreeSurfer labels)
    ├─ _synthseg33_native.nii.gz   ← already native space, order=0
    │     └─ apply_merge_rule(..., "infarct10") → 0-9
    │           → model channel 2, and the skull-strip mask (> 0)
    └─ post_process="dwi" → _DWI → resampleSynthSEG2original(..., "DWI")
          → david 1xx/2xx/3xx territories → the 34 clinical location names

The file named `_synthseg33_native.nii.gz` does NOT contain synthseg33. It is
the full aparc+aseg resampled to the input grid with order=0; the name is wrong
in the SynthSeg server. Verified: 99 labels, range 0..2035, 68 of them >= 1000.
Using real synthseg33 here would be silently destructive -- FreeSurfer 3 and 42
(the cerebral cortices) do not exist in the 33-label vocabulary, so every
cortical voxel would fall to background and the brain would lose 35% of its
volume with no error raised.

Both resampling paths use nearest-neighbour. That is not incidental: channel 2
is a categorical label map used as a prompt, and interpolating it invents values
that are not classes.

Input:  ADC / DWI0 / DWI1000 NIfTI + their DICOM directories
Output: process_dir with BET_{ADC,DWI1000}.nii.gz, SynthSeg_merged.nii.gz,
        SynthSEG.nii.gz, and the nnUNet directory skeleton
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys

logger = logging.getLogger("infarct.preprocess")

SYNTHSEG_URL = os.environ.get("SYNTHSEG_URL", "http://localhost:5005")
# SynthSeg on DWI is ~14s warm, but this call can queue behind another study's
# GPU work holding rad_gpu_0, and that wait is legitimate rather than a fault.
SYNTHSEG_TIMEOUT_S = int(os.environ.get("INFARCT_SYNTHSEG_TIMEOUT_S", "1800"))

SERIES = ("ADC", "DWI0", "DWI1000")


def _run_synthseg(dwi0_path: str, synthseg_dir: str, study_id: str) -> dict:
    """Call the SynthSeg service on DWI0 and return its output_paths map."""
    import requests

    os.makedirs(synthseg_dir, exist_ok=True)
    resp = requests.post(
        f"{SYNTHSEG_URL}/predict",
        json={
            "input_path": dwi0_path,
            "output_dir": synthseg_dir,
            "study_id": study_id,
            # "dwi" also emits _david and _wm; _DWI is the one the clinical
            # territory names are built from.
            "post_process": "dwi",
            # Required: the infarct10 merge needs the full parcellation, not the
            # faster seg33-only path.
            "run_parcellation": True,
            "force": False,
        },
        timeout=(10, SYNTHSEG_TIMEOUT_S),
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "ok":
        raise RuntimeError(f"SynthSeg failed: {body.get('error_msg') or body}")
    logger.info("[preprocess] synthseg done (cached=%s, %.0fs)",
                body.get("cached"), body.get("elapsed_time", 0.0))
    return body.get("output_paths") or {}


def _locate(output_paths: dict, synthseg_dir: str, suffix: str) -> str:
    """Find a SynthSeg output by suffix, preferring the response over guessing.

    The response keys are not documented anywhere we control, so a glob is kept
    as the fallback -- but a miss is raised rather than defaulted, because every
    consumer of these files is silent about receiving the wrong one.
    """
    for path in output_paths.values():
        if path.endswith(suffix):
            return path
    hits = sorted(glob.glob(os.path.join(synthseg_dir, f"*{suffix}")))
    if not hits:
        raise FileNotFoundError(
            f"SynthSeg produced no *{suffix} in {synthseg_dir}; "
            f"got {sorted(output_paths)} / {sorted(os.listdir(synthseg_dir))[:20]}"
        )
    return hits[0]


def _derive_infarct10(aparc_path: str, out_path: str) -> "object":
    """aparc+aseg -> the 10-class map the model was trained against."""
    import nibabel as nib
    import numpy as np
    from merge_rules import apply_merge_rule, unmapped_labels

    seg_nii = nib.load(aparc_path)
    seg = np.asanyarray(seg_nii.dataobj).astype(np.int32)

    # Guards against being handed the wrong vocabulary. infarct10 covers all of
    # aparc+aseg, so anything unmapped means this is not aparc+aseg -- most
    # likely synthseg33, whose missing cortex would vanish into background.
    missing = unmapped_labels(seg, "infarct10")
    if missing:
        raise ValueError(
            f"{aparc_path} has {len(missing)} labels outside the infarct10 rule "
            f"({dict(list(missing.items())[:5])}...). Expected aparc+aseg; a "
            f"synthseg33 volume would look exactly like this and would drop all "
            f"cortex to background."
        )

    merged = apply_merge_rule(seg, "infarct10").astype(np.uint8)
    out = nib.Nifti1Image(merged, seg_nii.affine, seg_nii.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, out_path)
    logger.info("[preprocess] infarct10 merged: %s (classes %s)",
                os.path.basename(out_path), np.unique(merged).tolist())
    return merged


def _derive_clinical_territories(path_nii: str, synthseg_dir: str,
                                 output_paths: dict) -> str:
    """david 1xx/2xx/3xx territories, for the 34 clinical location names.

    resampleSynthSEG2original reads three fixed filenames out of one directory,
    so the SynthSeg outputs are staged under the names it expects rather than
    changing a function that aneurysm and WMH both depend on.
    """
    import util_aneurysm

    resample_src = _locate(output_paths, synthseg_dir, "_resample.nii.gz")
    dwi_src = _locate(output_paths, synthseg_dir, "_DWI.nii.gz")

    shutil.copy(resample_src, os.path.join(path_nii, "DWI0_resample.nii.gz"))
    shutil.copy(dwi_src, os.path.join(path_nii, "DWI0_DWI.nii.gz"))

    util_aneurysm.resampleSynthSEG2original(path_nii, "DWI0", "DWI")

    produced = os.path.join(path_nii, "NEW_DWI0_DWI.nii.gz")
    if not os.path.isfile(produced):
        raise FileNotFoundError(f"resampleSynthSEG2original wrote no {produced}")
    return produced


def _apply_brain_mask(process_dir: str, merged) -> None:
    """Multiply DWI1000 and ADC by (SynthSeg_merged > 0), as in training.

    The model's ZScoreBrainNormalization derives its brain mask as `image > 0`,
    which is only meaningful because the training images were already skull
    stripped this way. Feeding it un-stripped images silently shifts the
    normalisation statistics.
    """
    import nibabel as nib
    import numpy as np

    mask = (merged > 0).astype(np.uint8)

    mask_nii = None
    for name in ("DWI1000", "ADC", "DWI0"):
        src = os.path.join(process_dir, f"{name}.nii.gz")
        nii = nib.load(src)
        arr = np.asanyarray(nii.dataobj)
        if arr.shape != mask.shape:
            raise ValueError(
                f"{name} is {arr.shape} but the brain mask is {mask.shape}; "
                f"SynthSeg ran on a different grid than this series"
            )
        out = nib.Nifti1Image(arr * mask, nii.affine, nii.header)
        nib.save(out, os.path.join(process_dir, f"BET_{name}.nii.gz"))
        mask_nii = nii

    out = nib.Nifti1Image(mask, mask_nii.affine, mask_nii.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, os.path.join(process_dir, "BET_mask.nii.gz"))
    logger.info("[preprocess] skull-strip applied (brain fraction %.4f)",
                float(mask.mean()))


def infarct_preprocess(
    study_id: str,
    adc_path: str,
    dwi0_path: str,
    dwi1000_path: str,
    dicom_dirs: dict,
    process_dir: str,
) -> bool:
    try:
        logger.info("[preprocess] %s start", study_id)

        srcs = {"ADC": adc_path, "DWI0": dwi0_path, "DWI1000": dwi1000_path}
        for name, path in srcs.items():
            if not path or not os.path.isfile(path):
                logger.error("[preprocess] Missing %s: %s", name, path)
                return False

        path_nnunet = os.path.join(process_dir, "nnUNet")
        path_nii = os.path.join(path_nnunet, "Image_nii")
        path_dcm = os.path.join(path_nnunet, "Dicom")
        path_norm = os.path.join(path_nnunet, "Normalized_Image")
        path_brain = os.path.join(path_nnunet, "Brain")
        path_result = os.path.join(path_nnunet, "Result")
        path_excel = os.path.join(path_nnunet, "excel")
        path_json_out = os.path.join(path_nnunet, "JSON")
        synthseg_dir = os.path.join(process_dir, "synthseg")

        for d in [process_dir, path_nnunet, path_nii, path_dcm, path_norm,
                  path_brain, path_result, path_excel, path_json_out, synthseg_dir]:
            os.makedirs(d, exist_ok=True)
            os.chmod(d, 0o775)  # group-writable — worker (gid=1001) needs access

        # Stale predictions from an earlier run would otherwise be picked up by
        # postprocess if inference failed to overwrite them.
        for stale in ("DeepInfarct_00001.nii.gz", "Prob.nii.gz", "Pred.nii.gz"):
            p = os.path.join(path_nnunet, stale)
            if os.path.isfile(p):
                os.remove(p)
                logger.info("[preprocess] Removed stale: %s", stale)

        for name, path in srcs.items():
            shutil.copy(path, os.path.join(process_dir, f"{name}.nii.gz"))
            shutil.copy(path, os.path.join(path_nii, f"{name}.nii.gz"))

        # Always re-copy: on a retrigger the rename_dicom tree may have changed.
        for name in SERIES:
            src = dicom_dirs.get(name)
            if not src or not os.path.isdir(src):
                logger.warning("[preprocess] no DICOM dir for %s (%s)", name, src)
                continue
            dst = os.path.join(path_dcm, name)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

        output_paths = _run_synthseg(
            os.path.join(process_dir, "DWI0.nii.gz"), synthseg_dir, study_id)

        aparc = _locate(output_paths, synthseg_dir, "_synthseg33_native.nii.gz")
        merged = _derive_infarct10(
            aparc, os.path.join(process_dir, "SynthSeg_merged.nii.gz"))

        territories = _derive_clinical_territories(
            path_nii, synthseg_dir, output_paths)
        shutil.copy(territories, os.path.join(process_dir, "SynthSEG.nii.gz"))
        shutil.copy(territories, os.path.join(path_nii, "SynthSEG.nii.gz"))

        _apply_brain_mask(process_dir, merged)

        required = [
            os.path.join(process_dir, f)
            for f in ("BET_DWI1000.nii.gz", "BET_ADC.nii.gz", "BET_mask.nii.gz",
                      "SynthSeg_merged.nii.gz", "SynthSEG.nii.gz")
        ]
        missing = [f for f in required if not os.path.isfile(f)]
        if missing:
            logger.error("[preprocess] missing outputs: %s", missing)
            return False

        logger.info("[preprocess] %s done", study_id)
        return True

    except Exception as exc:
        logger.error("[preprocess] %s failed: %s", study_id, exc, exc_info=True)
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Infarct Preprocessing (Layer 1/3)")
    parser.add_argument("--study_id", required=True)
    parser.add_argument("--adc_path", required=True)
    parser.add_argument("--dwi0_path", required=True)
    parser.add_argument("--dwi1000_path", required=True)
    parser.add_argument("--dicom_dirs", default="{}",
                        help='JSON: {"ADC": dir, "DWI0": dir, "DWI1000": dir}')
    parser.add_argument("--process_dir", required=True)
    args = parser.parse_args()

    ok = infarct_preprocess(
        study_id=args.study_id,
        adc_path=args.adc_path,
        dwi0_path=args.dwi0_path,
        dwi1000_path=args.dwi1000_path,
        dicom_dirs=json.loads(args.dicom_dirs),
        process_dir=args.process_dir,
    )
    sys.exit(0 if ok else 1)
