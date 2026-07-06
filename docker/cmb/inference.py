#!/usr/bin/env python3
"""CMB GPU Inference — ONNX sliding window detection + FP filtering.

Input:  SWAN + SynthSeg brain mask (pre-computed by task_pipeline)
Output: Pred_CMB.nii.gz + Pred_CMB.json

Uses cmb_inference_core.cmb_detect() — rewritten from scratch.
"""

import argparse
import glob
import logging
import os
import sys
import time

logger = logging.getLogger("cmb.inference")


def cmb_inference(
    study_id: str,
    swan_path: str,
    t1_path: str,
    output_dir: str,
    process_dir: str,
    log_dir: str,
    gpu_id: int = 0,
) -> bool:
    try:
        logger.info("[inference] %s start (GPU %d)", study_id, gpu_id)
        t0 = time.time()

        # Find FLIRT-coregistered CMB brain mask (T1 SynthSeg → SWAN space).
        # Produced by preprocess.py via FSL FLIRT → saved in process_dir.
        candidates = (
            glob.glob(os.path.join(process_dir, "CMB_mask_coregistered.nii.gz"))
            or glob.glob(os.path.join(process_dir, "synthseg_*CMB*.nii.gz"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No coregistered CMB mask in {process_dir}")
        synthseg_path = candidates[0]
        logger.info("SynthSeg CMB mask: %s", synthseg_path)

        # Output → process_dir（不嵌套 {study_id}/，不寫入 rename_nifti）
        output_nii = os.path.join(process_dir, "Pred_CMB.nii.gz")
        output_json = os.path.join(process_dir, "Pred_CMB.json")

        # Run detection
        from cmb_inference_core import cmb_detect
        result = cmb_detect(swan_path, synthseg_path, output_nii, output_json)

        elapsed = time.time() - t0
        if not result or not os.path.isfile(output_nii):
            logger.error("FAILED (%.0fs)", elapsed)
            return False

        logger.info("Done (%.0fs). Pred=%s", elapsed, output_nii)
        return True

    except Exception as exc:
        logger.error("Failed: %s", exc, exc_info=True)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--study_id", required=True)
    p.add_argument("--swan_path", required=True)
    p.add_argument("--t1_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--process_dir", required=True)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--gpu_id", type=int, default=0)
    a = p.parse_args()
    sys.exit(0 if cmb_inference(a.study_id, a.swan_path, a.t1_path, a.output_dir, a.process_dir, a.log_dir, a.gpu_id) else 1)
