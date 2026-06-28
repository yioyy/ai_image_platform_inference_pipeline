#!/usr/bin/env python3
"""CMB Preprocessing — FLIRT coregistration using pre-computed SynthSeg outputs.

Layer 1 of 3 in the CMB three-layer pipeline.

Pre-requisite: task_pipeline must have already run SynthSeg for SWAN + T1, producing:
  - {study_id}_SWAN_resample_synthseg33_1mm.nii.gz
  - {study_id}_T1*_resample_synthseg33_1mm.nii.gz
  - {study_id}_T1*_resample_CMB.nii.gz

Steps:
  1. Find pre-computed SynthSeg outputs (from task_pipeline) in process_dir
  2. FSL FLIRT rigid registration: T1_synthseg33 → SWAN_synthseg33 (-dof 6)
  3. FLIRT -applyxfm: T1_CMB_mask → SWAN space (nearest neighbour)
  4. Save coregistered CMB mask → process_dir

The coregistered CMB mask is a brain region map (29 labels) in SWAN native space,
used by inference.py as the brain mask for sliding window detection.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time

logger = logging.getLogger("cmb.preprocess")

FSL_FLIRT = os.environ.get("FSL_FLIRT", "/usr/local/fsl/bin/flirt")


def cmb_preprocess(
    study_id: str,
    swan_path: str,
    t1_path: str,
    process_dir: str,
    output_dir: str,
) -> bool:
    try:
        t0 = time.time()
        logger.info("[preprocess] %s start", study_id)

        for label, path in [("SWAN", swan_path), ("T1", t1_path)]:
            if not os.path.isfile(path):
                logger.error("Missing %s: %s", label, path)
                return False

        os.makedirs(process_dir, exist_ok=True)

        # Copy inputs to process dir（永遠覆蓋 — retrigger 時 rename_nifti 可能已更新）
        for src, name in [(swan_path, "SWAN.nii.gz"), (t1_path, "T1.nii.gz")]:
            dest = os.path.join(process_dir, name)
            shutil.copy(src, dest)

        # ── 1. Find pre-computed SynthSeg outputs (from task_pipeline) ──
        # RAD task_pipeline doesn't pre-compute SynthSeg (ws2030 does), so the
        # container runs main.py subprocess as a fallback. Phase B will replace
        # this with in-process import for warm-load benefit.
        import glob

        def _find_pre_computed(process_dir, pattern, exclude_patterns=("_from_", )):
            """Find pre-computed file, excluding FLIRT outputs and intermediates."""
            candidates = glob.glob(os.path.join(process_dir, pattern))
            filtered = [f for f in candidates
                        if not any(ex in os.path.basename(f) for ex in exclude_patterns)
                        and not os.path.basename(f).startswith("synthseg_")]
            return filtered

        swan_pre = _find_pre_computed(process_dir, "*SWAN*resample_synthseg33_1mm.nii.gz")
        t1_pre_33 = _find_pre_computed(process_dir, "*T1*_resample_synthseg33_1mm.nii.gz")
        t1_pre_cmb = _find_pre_computed(process_dir, "*T1*_resample_CMB.nii.gz")

        if not (swan_pre and t1_pre_33 and t1_pre_cmb):
            SYNTHSEG_MAIN = "/opt/shh_aiplatform-david/brain-parcellation/code_ai/pipeline/main.py"
            PY = "/opt/conda/envs/tf_2_14/bin/python"
            if not os.path.isfile(SYNTHSEG_MAIN):
                logger.error("[SynthSeg] main.py not found at %s", SYNTHSEG_MAIN)
                return False

            swan_in_proc = os.path.join(process_dir, "SWAN.nii.gz")
            t1_in_proc = os.path.join(process_dir, "T1.nii.gz")
            logger.info("[SynthSeg] pre-computed outputs missing — running main.py subprocess")
            t_ss = time.time()
            synthseg_env = {
                **os.environ,
                "PYTHONPATH": "/opt/shh_aiplatform-david/brain-parcellation:"
                              + os.environ.get("PYTHONPATH", ""),
            }
            result = subprocess.run(
                [PY, SYNTHSEG_MAIN,
                 "-i", swan_in_proc,
                 "--template", t1_in_proc,
                 "--output", process_dir,
                 "--all", "False",
                 "--CMB", "TRUE"],
                capture_output=True, text=True, env=synthseg_env, timeout=900,
            )
            if result.returncode != 0:
                logger.error("[SynthSeg] failed (rc=%d): %s",
                             result.returncode, result.stderr[-1000:])
                return False
            logger.info("[SynthSeg] done in %.0fs", time.time() - t_ss)

            swan_pre = _find_pre_computed(process_dir, "*SWAN*resample_synthseg33_1mm.nii.gz")
            t1_pre_33 = _find_pre_computed(process_dir, "*T1*_resample_synthseg33_1mm.nii.gz")
            t1_pre_cmb = _find_pre_computed(process_dir, "*T1*_resample_CMB.nii.gz")
            if not (swan_pre and t1_pre_33 and t1_pre_cmb):
                logger.error("[SynthSeg] ran but expected outputs still missing in %s", process_dir)
                return False

        swan_synthseg33 = swan_pre[0]
        t1_synthseg33 = t1_pre_33[0]
        t1_cmb_mask = t1_pre_cmb[0]
        logger.info("[SynthSeg] SWAN: %s", os.path.basename(swan_synthseg33))
        logger.info("[SynthSeg] T1: %s, %s",
                    os.path.basename(t1_synthseg33), os.path.basename(t1_cmb_mask))

        # ── 2. FLIRT rigid registration: T1_synthseg33 → SWAN_synthseg33 ──
        coreg_base = os.path.join(process_dir, "T1_to_SWAN_coreg")
        coreg_mat = coreg_base + ".mat"
        flirt_base_cmd = [
            FSL_FLIRT,
            "-in", t1_synthseg33,
            "-ref", swan_synthseg33,
            "-out", coreg_base,
            "-dof", "6",
            "-cost", "corratio",
            "-omat", coreg_mat,
            "-interp", "nearestneighbour",
        ]
        logger.info("FLIRT base: %s", " ".join(flirt_base_cmd))
        env = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"}
        result = subprocess.run(flirt_base_cmd, capture_output=True, text=True, env=env, timeout=120)
        if result.returncode != 0:
            logger.error("FLIRT base failed (rc=%d): %s", result.returncode, result.stderr[:500])
            return False

        if not os.path.isfile(coreg_mat):
            logger.error("FLIRT base: no .mat file produced")
            return False

        # ── 3. FLIRT apply: T1_CMB_mask → SWAN space ──
        cmb_coreg = os.path.join(process_dir, "CMB_mask_coregistered")
        flirt_apply_cmd = [
            FSL_FLIRT,
            "-in", t1_cmb_mask,
            "-ref", swan_synthseg33,
            "-out", cmb_coreg,
            "-init", coreg_mat,
            "-applyxfm",
            "-interp", "nearestneighbour",
        ]
        logger.info("FLIRT apply: %s", " ".join(flirt_apply_cmd))
        result = subprocess.run(flirt_apply_cmd, capture_output=True, text=True, env=env, timeout=120)
        if result.returncode != 0:
            logger.error("FLIRT apply failed (rc=%d): %s", result.returncode, result.stderr[:500])
            return False

        cmb_coreg_nii = cmb_coreg + ".nii.gz"
        if not os.path.isfile(cmb_coreg_nii):
            logger.error("FLIRT apply: no output file")
            return False

        elapsed = time.time() - t0
        logger.info("[preprocess] %s done (%.0fs). CMB mask: %s", study_id, elapsed, cmb_coreg_nii)
        return True

    except Exception as exc:
        logger.error("[preprocess] %s failed: %s", study_id, exc, exc_info=True)
        return False



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="CMB Preprocessing (Layer 1/3)")
    parser.add_argument("--study_id", required=True)
    parser.add_argument("--swan_path", required=True)
    parser.add_argument("--t1_path", required=True)
    parser.add_argument("--process_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    ok = cmb_preprocess(
        study_id=args.study_id, swan_path=args.swan_path,
        t1_path=args.t1_path, process_dir=args.process_dir,
        output_dir=args.output_dir,
    )
    sys.exit(0 if ok else 1)
