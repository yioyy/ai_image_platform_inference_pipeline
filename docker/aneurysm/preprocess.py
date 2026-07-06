#!/usr/bin/env python3
"""Aneurysm Preprocessing — minimal directory setup.

Layer 1 of 3 in the Aneurysm three-layer pipeline (CP5b).
CPU only — Aneurysm has no BET preprocessing (brain detection is Stage A of inference).

Input:  MRA_BRAIN NIfTI + DICOM directory
Output: process_dir with MRA_BRAIN.nii.gz + nnUNet directory structure
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

logger = logging.getLogger("aneurysm.preprocess")


def aneurysm_preprocess(
    study_id: str,
    mra_path: str,
    dicom_dir: str,
    process_dir: str,
) -> bool:
    try:
        logger.info("[preprocess] %s start", study_id)

        if not os.path.isfile(mra_path):
            logger.error("[preprocess] Missing MRA_BRAIN: %s", mra_path)
            return False

        # --- Create directory structure ---
        path_nnunet = os.path.join(process_dir, "nnUNet")
        path_nii = os.path.join(path_nnunet, "Image_nii")
        path_dcm = os.path.join(path_nnunet, "Dicom")
        path_reslice = os.path.join(path_nnunet, "Image_reslice")
        path_excel = os.path.join(path_nnunet, "excel")
        path_json_out = os.path.join(path_nnunet, "JSON")

        for d in [process_dir, path_nnunet, path_nii, path_dcm,
                  path_reslice, path_excel, path_json_out]:
            os.makedirs(d, exist_ok=True)
            os.chmod(d, 0o775)  # group-writable — worker (gid=1001) needs access

        # --- Remove stale predictions ---
        for stale in ["DeepAneurysm_00001.nii.gz"]:
            p = os.path.join(process_dir, stale)
            if os.path.isfile(p):
                os.remove(p)
                logger.info("[preprocess] Removed stale: %s", stale)

        # --- Copy MRA_BRAIN ---
        shutil.copy(mra_path, os.path.join(process_dir, "MRA_BRAIN.nii.gz"))
        shutil.copy(mra_path, os.path.join(path_nnunet, "MRA_BRAIN.nii.gz"))
        shutil.copy(mra_path, os.path.join(path_nii, "MRA_BRAIN.nii.gz"))

        # --- Copy DICOM（永遠重新複製 — retrigger 時 rename_dicom 可能已更新）---
        path_dcm_mra = os.path.join(path_dcm, "MRA_BRAIN")
        if os.path.isdir(dicom_dir):
            if os.path.isdir(path_dcm_mra):
                shutil.rmtree(path_dcm_mra)
            shutil.copytree(dicom_dir, path_dcm_mra)

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
    parser = argparse.ArgumentParser(description="Aneurysm Preprocessing (Layer 1/3)")
    parser.add_argument("--study_id", required=True)
    parser.add_argument("--mra_path", required=True)
    parser.add_argument("--dicom_dir", required=True)
    parser.add_argument("--process_dir", required=True)
    args = parser.parse_args()

    ok = aneurysm_preprocess(
        study_id=args.study_id, mra_path=args.mra_path,
        dicom_dir=args.dicom_dir, process_dir=args.process_dir,
    )
    sys.exit(0 if ok else 1)
