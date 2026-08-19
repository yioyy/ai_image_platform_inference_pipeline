#!/usr/bin/env python3
"""Infarct Inference — MUTP predictor, three channels, TTA.

Layer 2 of 3. Runs INSIDE the Redis rad_gpu_0 lock (the dispatcher holds it
around this phase only).

The import order at the top of this file is load-bearing. custom_predict imports
from `nnunetv2`, and the container already has an older nnunetv2 installed --
the generation that predates every mask-fusion architecture. If that one wins
the import, three things happen in sequence and none of them raise:

  1. ResidualEncoderUNet_DeepConcat does not exist, so the network is built as
     something else or fails to load;
  2. the built network has no `image_channels` attribute;
  3. custom_predict gates its per-channel resampling override on
     `hasattr(network, "image_channels")`, so the override never fires and
     nnUNet's default order=3 cubic runs over ALL channels -- including the
     anatomy channel, whose integer classes 0-9 become 1.18, 3.6, 7.99.

The model then produces a plausible-looking segmentation from a corrupted
anatomical prompt. _assert_vendored_nnunet turns that silent path into an
immediate, explicit failure.

Input:  BET_DWI1000 / BET_ADC / SynthSeg_merged (from preprocess)
Output: nnUNet/Prob.nii.gz (float32 softmax) and nnUNet/Pred.nii.gz
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

# Must precede any nnunetv2 import, including transitive ones.
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

logger = logging.getLogger("infarct.inference")

CASE = "DeepInfarct_00001"
# Channel order is from the trained plans, not a convention we may choose:
# 0 DWI1000 (ZScoreBrainNormalization), 1 ADC (ADCNormalization),
# 2 SynthSeg_merged (NoNormalization). The previous 2-channel model had ADC
# first; swapping these produces garbage silently.
CHANNELS = (
    ("0000", "BET_DWI1000.nii.gz"),
    ("0001", "BET_ADC.nii.gz"),
    ("0002", "SynthSeg_merged.nii.gz"),
)

# A ~24-slice volume with a [3,192,160] patch needs very few tiles, but this is
# a 3D network where the 2D model used 64. Keep it small; the GPU is shared.
BATCH_SIZE = int(os.environ.get("INFARCT_BATCH_SIZE", "2"))
# Threshold to binarise the probability map into connected components.
CONF_TH = float(os.environ.get("INFARCT_CONF_TH", "0.1"))
# A component is kept only if its peak probability clears this.
OBJ_TH = float(os.environ.get("INFARCT_OBJ_TH", "0.5"))
# Deliberately permissive: the smallest lesion worth recording is 6 voxels, and
# what actually shows in the viewer table is the platform's decision, not ours.
# Expressed in voxels rather than cm3 so it does not silently change meaning
# when the model's target spacing does.
MIN_VOXELS = int(os.environ.get("INFARCT_MIN_VOXELS", "6"))


def _assert_vendored_nnunet() -> None:
    """Fail now if the container's older nnunetv2 won the import race."""
    import nnunetv2

    loaded = os.path.abspath(os.path.dirname(nnunetv2.__file__))
    expected = os.path.abspath(os.path.join(_VENDOR, "nnunetv2"))
    if loaded != expected:
        raise RuntimeError(
            f"nnunetv2 resolved to {loaded}, not the vendored {expected}. The "
            f"vendored copy is the only one with ResidualEncoderUNet_DeepConcat; "
            f"the other would run without the per-channel resampling override "
            f"and cubic-interpolate the anatomy channel without complaint."
        )

    from nnunetv2.utilities.unet_v2 import ResidualEncoderUNet_DeepConcat  # noqa: F401
    logger.info("[inference] nnunetv2 <- %s", loaded)


def _stage_channels(process_dir: str, path_norm: str, path_brain: str) -> None:
    """Lay out the three channels under the names nnUNet expects."""
    os.makedirs(path_norm, exist_ok=True)
    os.makedirs(path_brain, exist_ok=True)

    for idx, src_name in CHANNELS:
        src = os.path.join(process_dir, src_name)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"channel {idx} source missing: {src}")
        shutil.copy(src, os.path.join(path_norm, f"{CASE}_{idx}.nii.gz"))

    mask = os.path.join(process_dir, "BET_mask.nii.gz")
    if not os.path.isfile(mask):
        raise FileNotFoundError(f"brain mask missing: {mask}")
    shutil.copy(mask, os.path.join(path_brain, f"{CASE}_0000.nii.gz"))

    logger.info("[inference] channels staged: %s",
                [f"{i}={n}" for i, n in CHANNELS])


def _verify_anatomy_channel(path_norm: str) -> None:
    """V7: the anatomy channel must still be the integer classes 0-9.

    Checked on the way in as well as being the acceptance criterion, because a
    corrupted prompt does not make the model fail -- it makes it wrong.
    """
    import nibabel as nib
    import numpy as np

    p = os.path.join(path_norm, f"{CASE}_0002.nii.gz")
    arr = np.asanyarray(nib.load(p).dataobj)
    vals = np.unique(arr)
    if not np.all(vals == vals.astype(np.int64)) or vals.min() < 0 or vals.max() > 9:
        raise ValueError(
            f"anatomy channel {p} carries {vals[:12]}... — expected integers in "
            f"0..9. Non-integer values mean it was interpolated somewhere."
        )
    logger.info("[inference] anatomy channel classes: %s", vals.astype(int).tolist())


def _write_prob(src_nii: str, dst: str) -> "object":
    """Copy the softmax output out as float32.

    write_probabilities in the fork inherits the header of the *input* image, so
    an int16-typed input quantises the probabilities to 1/65536 steps on the way
    to disk. That is cosmetic for a threshold at 0.5 but not for prob_max, which
    is what the report shows.
    """
    import nibabel as nib
    import numpy as np

    nii = nib.load(src_nii)
    arr = np.asanyarray(nii.dataobj).astype(np.float32)
    out = nib.Nifti1Image(arr, nii.affine, nii.header)
    out.set_data_dtype(np.float32)
    nib.save(out, dst)
    logger.info("[inference] Prob.nii.gz float32, range [%.4f, %.4f]",
                float(arr.min()), float(arr.max()))
    return out


def filter_infarct_by_volume(pred_prob_map, spacing, conf_th=CONF_TH,
                             min_voxels=MIN_VOXELS, obj_th=OBJ_TH):
    """Connected components of the probability map, filtered and renumbered.

    Ported from the reference gpu_infarct.py with two deliberate changes:

    - the size floor is a voxel count, not 0.3 cm3. The plan keeps every lesion
      down to 6 voxels and leaves the display threshold to the viewer, and a
      voxel count does not quietly change meaning when spacing does.
    - top_k is gone. It capped the result at 30 components ordered by peak
      probability, which is another silent truncation of the same kind.

    Renumbering happens after every filter, unlike the original, which assigned
    labels before applying obj_th and so left gaps in the sequence.
    """
    import numpy as np
    import pandas as pd
    from skimage.measure import label, regionprops_table

    pred_label = label(pred_prob_map > conf_th)
    if pred_label.max() == 0:
        return pd.DataFrame(columns=["ori_Pred_label", "Pred_volume", "Pred_max",
                                     "Pred_mean", "Pred_voxels", "Pred_label"]), \
               np.zeros_like(pred_label, dtype=np.uint8)

    props = regionprops_table(
        pred_label, pred_prob_map,
        properties=("label", "area", "intensity_max", "intensity_mean"))

    voxel_cm3 = float(spacing[0] * spacing[1] * spacing[2]) / 1000.0
    df = pd.DataFrame({
        "ori_Pred_label": props["label"],
        "Pred_voxels": props["area"],
        "Pred_volume": props["area"] * voxel_cm3,
        "Pred_max": props["intensity_max"],
        "Pred_mean": props["intensity_mean"],
    })
    n0 = len(df)

    df = df[df["Pred_voxels"] >= min_voxels]
    df = df[df["Pred_max"] >= obj_th]
    df = df.sort_values(by="Pred_max", ascending=False).reset_index(drop=True)
    df["Pred_label"] = np.arange(1, len(df) + 1)

    new_pred = np.zeros_like(pred_label, dtype=np.uint8)
    for _, row in df.iterrows():
        new_pred[pred_label == int(row["ori_Pred_label"])] = int(row["Pred_label"])

    logger.info("[inference] components %d -> %d (>=%d voxels, peak>=%.2f)",
                n0, len(df), min_voxels, obj_th)
    return df, new_pred


def infarct_inference(process_dir: str, model_dir: str, gpu_id: int = 0) -> bool:
    try:
        _assert_vendored_nnunet()

        import nibabel as nib
        import numpy as np
        import torch
        from mutp_engine.custom_predict import predict_from_raw_data

        path_nnunet = os.path.join(process_dir, "nnUNet")
        path_norm = os.path.join(path_nnunet, "Normalized_Image")
        path_brain = os.path.join(path_nnunet, "Brain")
        path_out = os.path.join(path_nnunet, "predict")
        os.makedirs(path_out, exist_ok=True)

        _stage_channels(process_dir, path_norm, path_brain)
        _verify_anatomy_channel(path_norm)

        logger.info("[inference] predict start (gpu=%d, TTA on, batch=%d)",
                    gpu_id, BATCH_SIZE)
        predict_from_raw_data(
            path_norm,
            path_brain,
            path_out,
            model_dir,
            use_folds=(0,),
            tile_step_size=0.25,
            use_gaussian=True,
            # The chosen model is the TTA variant; mirroring is what makes it so.
            use_mirroring=True,
            perform_everything_on_gpu=True,
            verbose=True,
            overwrite=True,
            checkpoint_name="checkpoint_best.pth",
            num_processes_preprocessing=2,
            num_processes_segmentation_export=3,
            desired_gpu_index=gpu_id,
            device=torch.device("cuda"),
            batch_size=BATCH_SIZE,
        )

        # custom_predict writes the foreground softmax as the primary NIfTI; it
        # never writes an argmax mask, so Pred is derived here.
        raw = os.path.join(path_out, f"{CASE}.nii.gz")
        if not os.path.isfile(raw):
            raise FileNotFoundError(f"predictor wrote no {raw}")

        prob_path = os.path.join(path_nnunet, "Prob.nii.gz")
        prob_nii = _write_prob(raw, prob_path)

        prob = np.asanyarray(prob_nii.dataobj)
        spacing = prob_nii.header.get_zooms()[:3]
        _, pred = filter_infarct_by_volume(prob, spacing)

        pred_nii = nib.Nifti1Image(pred, prob_nii.affine, prob_nii.header)
        pred_nii.set_data_dtype(np.uint8)
        nib.save(pred_nii, os.path.join(path_nnunet, "Pred.nii.gz"))

        path_result = os.path.join(path_nnunet, "Result")
        os.makedirs(path_result, exist_ok=True)
        for f in ("Prob.nii.gz", "Pred.nii.gz"):
            shutil.copy(os.path.join(path_nnunet, f),
                        os.path.join(path_result, f))

        logger.info("[inference] done — %d lesion(s)", int(pred.max()))
        return True

    except Exception as exc:
        logger.error("[inference] failed: %s", exc, exc_info=True)
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Infarct Inference (Layer 2/3)")
    parser.add_argument("--process_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    ok = infarct_inference(args.process_dir, args.model_dir, args.gpu_id)
    sys.exit(0 if ok else 1)
