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


# Smallest through-plane coverage a TOF MRA may have and still be worth
# running. See the module docstring of the patch that introduced this for the
# measurement; briefly, legitimate volumes here start at 76.3 mm and the two
# truncated ones are 33.6 and 24.0, so 60 has 26 mm of margin on one side and
# 16 on the other. Env override for a site whose protocol is genuinely shorter.
MIN_TOF_SI_COVERAGE_MM = float(os.environ.get("ANEURYSM_MIN_TOF_COVERAGE_MM", "60"))


def _si_coverage_mm(path: str) -> float:
    """Extent of the volume along the patient's superior-inferior axis, in mm.

    Read off the affine rather than assuming axis 2 is the slice direction: a
    coronal or oblique acquisition would make the named axis mean nothing, and
    this number decides whether a study runs.
    """
    import nibabel as nib
    import numpy as np

    nii = nib.load(path)
    axis = int(np.argmax(np.abs(nii.affine[:3, :3][2, :])))
    return float(nii.shape[axis]) * float(nii.header.get_zooms()[axis])


def _study_uid_from_dicom_dir(dicom_dir: str) -> str:
    """StudyInstanceUID from any DICOM in a series directory.

    postprocess has _study_uid_from_mra, but that one takes the parent and
    looks for an MRA_BRAIN child; here the series directory itself is what the
    caller was given, and a failure this early must not depend on a naming
    convention holding.
    """
    try:
        import pydicom

        for name in sorted(os.listdir(dicom_dir)):
            ds = pydicom.dcmread(os.path.join(dicom_dir, name),
                                 stop_before_pixels=True)
            uid = str(getattr(ds, "StudyInstanceUID", "") or "")
            if uid:
                return uid
    except Exception as exc:
        logger.error("[preprocess] cannot read StudyInstanceUID from %s: %s",
                     dicom_dir, exc)
    return ""


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

        # A volume that does not reach the circle of Willis cannot answer the
        # question being asked of it. Refusing is the whole point: a prediction
        # computed from the top of the head is reported against a study whose
        # base was never imaged, and nothing downstream says so.
        try:
            coverage = _si_coverage_mm(mra_path)
        except Exception as exc:
            logger.error("[preprocess] cannot measure MRA_BRAIN coverage (%s): %s",
                         mra_path, exc)
            coverage = None
        if coverage is not None and coverage < MIN_TOF_SI_COVERAGE_MM:
            reason = (
                "MRA_BRAIN covers only %.1f mm head-to-foot, below the %.0f mm "
                "minimum: the circle of Willis is unlikely to be inside the "
                "volume. Usually a truncated series -- check whether the study "
                "also holds a longer Ax TOF MRA under a higher series number."
                % (coverage, MIN_TOF_SI_COVERAGE_MM))
            logger.error("[preprocess] %s: %s", study_id, reason)
            from postprocess import _notify_platform_failed

            _notify_platform_failed(
                _study_uid_from_dicom_dir(dicom_dir), "aneurysm_model", reason)
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
