#!/usr/bin/env python3
"""CMB Postprocessing — DICOM-SEG generation + delivery.

Input:  Pred_CMB.nii.gz + Pred_CMB.json (from inference, at process_dir)
Output: DICOM-SEG → Orthanc + platform_json → Laravel

Zero Chuan code dependencies. Uses David's code_ai modules only.
"""

import argparse
import glob
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

logger = logging.getLogger("cmb.postprocess")


def cmb_postprocess(
    study_id: str,
    output_dir: str,
    dicom_dir: str,
    process_dir: str,
    input_json: str = "",
    group_id: int = None,
) -> bool:
    try:
        logger.info("[postprocess] %s start", study_id)

        # Locate Pred_CMB.nii.gz (inference layer wrote to process_dir; may also exist
        # in output_dir from earlier runs).
        pred_nii_proc = os.path.join(process_dir, "Pred_CMB.nii.gz")
        pred_json_proc = os.path.join(process_dir, "Pred_CMB.json")
        pred_nii_out = os.path.join(output_dir, "Pred_CMB.nii.gz")
        pred_json_out = os.path.join(output_dir, "Pred_CMB.json")

        src_nii = pred_nii_proc if os.path.isfile(pred_nii_proc) else pred_nii_out
        if not os.path.isfile(src_nii):
            logger.error("Missing Pred_CMB.nii.gz in %s or %s", process_dir, output_dir)
            return False

        # Mirror inference outputs into output_dir so RAD downstream (build/cmb.py
        # subprocess) finds them at the expected RAD path.
        if not os.path.isfile(pred_nii_out) or os.path.getmtime(src_nii) > os.path.getmtime(pred_nii_out):
            shutil.copy(src_nii, pred_nii_out)

        # Container inference writes Pred_CMB.json as
        #   {"lesion_count": N, "lesions": [{index, cmb_prob, diameter_mm, ...}]}
        # but RAD build/cmb.py merge step (line 56 dict-spread) expects the inline
        # pipeline_cmb_tensorflow.py format — a flat LIST of records with field
        # names {label#, CMB_prob, pred_diameter, class_name, type_name, type}.
        # Translate before writing output_dir/Pred_CMB.json.
        src_json = pred_json_proc if os.path.isfile(pred_json_proc) else pred_json_out
        if os.path.isfile(src_json):
            with open(src_json) as f:
                raw = json.load(f)
            lesions = raw.get("lesions", raw) if isinstance(raw, dict) else raw
            translated = [{
                "label#": le.get("index", le.get("label#", 0)),
                "CMB_prob": le.get("cmb_prob", le.get("CMB_prob", 0)),
                "pred_diameter": le.get("diameter_mm", le.get("pred_diameter", 0)),
                "class_name": le.get("class", le.get("class_name", "CMB")),
                "type_name": le.get("type_name", ""),
                "type": le.get("type", ""),
            } for le in lesions]
            with open(pred_json_out, "w") as f:
                json.dump(translated, f)
            logger.info("Pred_CMB.json translated to inline records format (%d lesions)", len(translated))

        # ── 4. RAD DICOM-SEG via build/cmb.py subprocess ──────────────────
        # Replaces ws2030 NewReviewCMBPlatformJSONBuilder (incompatible RAD schema).
        # build/cmb.py composes `output_series_folder = --Output_folder / <ID>`,
        # so pass the PARENT of output_dir (rename_nifti) — mirroring inline
        # pipeline_cmb_tensorflow.py call. Produces Pred_CMB_rdx_cmb_pred_json.json
        # + Pred_CMB_A*.dcm under output_dir.
        t = time.time()
        BP_ROOT = "/opt/shh_aiplatform-david/brain-parcellation"
        PY = "/opt/conda/envs/tf_2_14/bin/python"
        rename_nifti_root = os.path.dirname(output_dir.rstrip("/"))
        build_cmd = [
            PY, f"{BP_ROOT}/code_ai/pipeline/dicomseg/build/cmb.py",
            "--ID", study_id,
            "--InputsDicomDir", dicom_dir,
            "--Inputs", pred_nii_out,
            "--Output_folder", rename_nifti_root,
        ]
        logger.info("RAD DICOM-SEG: %s", " ".join(build_cmd))
        env = {
            **os.environ,
            "PYTHONPATH": BP_ROOT + ":" + os.environ.get("PYTHONPATH", ""),
        }
        r = subprocess.run(build_cmd, capture_output=True, text=True, env=env, timeout=300)
        if r.returncode != 0:
            logger.error("build/cmb.py failed (rc=%d): %s", r.returncode, r.stderr[-1000:])
            return False
        rdx_json_path = pred_json_out.replace(".json", "_rdx_cmb_pred_json.json")
        if not os.path.isfile(rdx_json_path):
            logger.error("build/cmb.py did not produce %s", rdx_json_path)
            return False
        logger.info("RAD DICOM-SEG done (%.0fs)", time.time() - t)

        # ── 5. RAD upload to AI_INFERENCE_RESULT_PATH ─────────────────────
        # Replaces ws2030 deliver_results (Laravel API). Mirrors inline
        # pipeline_cmb_tensorflow.py copy_to_ai_result_path: writes
        # ai-inference-result/<study_uid>/cmb_model/<infer_id>/
        # {prediction.json, <series_uid>_<label>.dcm}.
        upload_dir = os.environ.get(
            "RADX_UPLOAD_DIR",
            os.environ.get("AI_INFERENCE_RESULT_PATH", "/home/david/ai-inference-result"),
        )
        with open(rdx_json_path) as f:
            rdx_data = json.load(f)
        study_uid = (rdx_data.get("input_study_instance_uid") or [""])[0]
        inference_id = str(rdx_data.get("inference_id") or "unknown")
        if not study_uid:
            logger.error("rdx JSON missing input_study_instance_uid: %s", rdx_json_path)
            return False

        target_dir = os.path.join(upload_dir, study_uid, "cmb_model", inference_id)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(rdx_json_path, os.path.join(target_dir, "prediction.json"))

        # Copy + rename DICOM-SEG files (Pred_CMB_<label>.dcm → <series_uid>_<label>.dcm).
        import pydicom
        for dcm_file in sorted(glob.glob(os.path.join(output_dir, "Pred_CMB_*.dcm"))):
            try:
                ds = pydicom.dcmread(dcm_file, stop_before_pixels=True)
                series_uid = str(ds.SeriesInstanceUID)
            except Exception as exc:
                logger.warning("Skip unreadable DICOM-SEG %s: %s", dcm_file, exc)
                continue
            label = os.path.basename(dcm_file).replace("Pred_CMB_", "").replace(".dcm", "")
            shutil.copy(dcm_file, os.path.join(target_dir, f"{series_uid}_{label}.dcm"))
        logger.info("RAD upload done -> %s", target_dir)

        # ── 6. Followup — 未來實作 ────────────────────────────────────────
        if input_json:
            logger.info("Followup skipped — future implementation")

        logger.info("[postprocess] %s done", study_id)
        return True

    except Exception as exc:
        logger.error("[postprocess] %s failed: %s", study_id, exc, exc_info=True)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--study_id", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dicom_dir", required=True)
    p.add_argument("--process_dir", required=True)
    p.add_argument("--input_json", default="")
    p.add_argument("--group_id", type=int, default=None,
                   help="CP1c (AP-085): per-group cleanup target; None=use env fallback")
    a = p.parse_args()
    sys.exit(0 if cmb_postprocess(a.study_id, a.output_dir, a.dicom_dir, a.process_dir, a.input_json, a.group_id) else 1)
