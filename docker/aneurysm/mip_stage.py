#!/usr/bin/env python3
"""Aneurysm Stage D subprocess: Reslice + DICOM decompress + MIP.

Runs as separate Python process from main aneurysm inference. Rationale:
- Stage D allocates ~5-8 GB peak GPU (rotation buffers + activations for MIP)
- In-process would leave ~4-5 GB in PyTorch reserved pool, unavailable to next case
- Subprocess exit fully releases all GPU memory back to CUDA driver
- No cold-start penalty because Stage D uses NO warmup model
- Cost: ~2-3s subprocess launch overhead vs -4 GB VRAM after case

Isolates GPU memory pool churn from the main aneurysm inference process while
preserving in-process benefit for Stage A-C (which DO have warm models).
"""

from __future__ import annotations

import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import logging
import os
import shutil
import sys
import time

# mip_utils lives in the same /app dir when subprocess launched
from mip_utils import (
    reslice_nifti_pred_nobrain,
    decompress_dicom_with_gdcm,
    create_MIP_pred,
)

logger = logging.getLogger("aneurysm.mip_stage")


def mip_stage(process_dir: str, code_dir: str, gpu_id: int, dicom_dir: str) -> bool:
    try:
        path_nnunet = os.path.join(process_dir, "nnUNet")
        path_nii = os.path.join(path_nnunet, "Image_nii")
        path_dcm = os.path.join(path_nnunet, "Dicom")
        path_reslice = os.path.join(path_nnunet, "Image_reslice")
        for d in [path_nii, path_dcm, path_reslice]:
            os.makedirs(d, exist_ok=True)

        # Move nnUNet outputs into Stage D layout (some may already be copied)
        for src, dest_dir, name in [
            (os.path.join(path_nnunet, "Pred.nii.gz"), path_nii, "Pred.nii.gz"),
            (os.path.join(process_dir, "Vessel.nii.gz"), path_nii, "Vessel.nii.gz"),
            (os.path.join(process_dir, "Vessel_16.nii.gz"), path_nnunet, "Vessel_16.nii.gz"),
        ]:
            dest = os.path.join(dest_dir, name)
            if os.path.isfile(src) and not os.path.isfile(dest):
                shutil.copy(src, dest)

        if dicom_dir and os.path.isdir(dicom_dir):
            dcm_dest = os.path.join(path_dcm, "MRA_BRAIN")
            if not os.path.isdir(dcm_dest):
                shutil.copytree(dicom_dir, dcm_dest)

        t = time.time()
        reslice_nifti_pred_nobrain(path_nii, path_reslice)
        logger.info("[D] Reslice done (%.0fs)", time.time() - t)

        t = time.time()
        decompress_dicom_with_gdcm(path_dcm)
        logger.info("[D] DICOM decompress done (%.0fs)", time.time() - t)

        t = time.time()
        path_png = os.path.join(code_dir, "png") if code_dir else "/tmp/png"
        create_MIP_pred(path_dcm, path_reslice, path_png, gpu_id)
        logger.info("[D] MIP done (%.0fs)", time.time() - t)

        for name in ["MIP_Pitch_pred.nii.gz", "MIP_Yaw_pred.nii.gz"]:
            src = os.path.join(path_reslice, name)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(path_nnunet, name))

        return True
    except Exception as exc:
        logger.error("[mip_stage] failed: %s", exc, exc_info=True)
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--process_dir", required=True)
    p.add_argument("--code_dir", default="")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--dicom_dir", default="")
    a = p.parse_args()
    sys.exit(0 if mip_stage(a.process_dir, a.code_dir, a.gpu_id, a.dicom_dir) else 1)
