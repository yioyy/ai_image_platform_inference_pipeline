#!/usr/bin/env python3
"""Aneurysm Postprocessing — Excel stats, platform_json, DICOM-SEG, deliver.

Input (from inference):
  - nnUNet/Pred.nii.gz, Prob.nii.gz
  - Vessel.nii.gz, Vessel_16.nii.gz
  - MIP_Pitch.nii.gz, MIP_Yaw.nii.gz
  - SynthSEG.nii.gz (from task_pipeline)
  - Dicom/MRA_BRAIN/, Dicom/MIP_Pitch/, Dicom/MIP_Yaw/

Output:
  - Pred_Aneurysm.nii.gz → output_dir
  - Aneurysm_Pred_list.xlsx (per-lesion stats with vessel location)
  - DICOM-SEG → Orthanc
  - platform_json → Laravel

Zero Chuan code dependencies.
"""

import argparse
import logging
import os
import pathlib
import shutil
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

logger = logging.getLogger("aneurysm.postprocess")


def _notify_platform_complete(rdx_json_path: str, model_name: str) -> None:
    """POST inference completion to RAD backend (AI_APP_INFERENCE_COMPLETE).

    Mirrors inline pipeline_*_tensorflow.py upload_inference_complete so the
    platform flips the case from "running" to "done". Best-effort: failure is
    logged but does not fail postprocess.
    """
    import json as _json
    url = os.environ.get("AI_APP_INFERENCE_COMPLETE")
    if not url:
        logger.warning("AI_APP_INFERENCE_COMPLETE not set — skip platform notify")
        return
    try:
        with open(rdx_json_path) as f:
            data = _json.load(f)
        study_uid = (data.get("input_study_instance_uid") or [""])[0]
        inference_id = str(data.get("inference_id") or "")
        if not (study_uid and inference_id):
            logger.warning("notify: missing study_uid or inference_id in %s", rdx_json_path)
            return
        import requests
        # Backend schema InferenceSuccessRequest requires the literal `result: "success"`
        # field — without it the POST returns HTTP 400 and the case stays "running".
        r = requests.post(
            url,
            json={
                "studyInstanceUid": study_uid,
                "modelName": model_name,
                "result": "success",
                "inferenceId": inference_id,
            },
            timeout=30,
        )
        logger.info("[notify] %s -> %s (%d)", model_name, url, r.status_code)
    except Exception as exc:
        logger.warning("[notify] %s failed: %s", model_name, exc)

# Vessel 16-label → vessel name mapping
# Vessel 16-label → (location, sub_location) mapping
# Exact copy from old pipeline util_aneurysm.py decode_location()
def _decode_vessel_location(val):
    if val in (1, 3):   return 'ICA', ''
    elif val in (2, 4): return 'MCA', ''
    elif val in (5, 6): return 'ACA', '1'
    elif val == 7:      return 'ACA', ''
    elif val in (8, 9): return 'ICA', 'p'
    elif val in (10, 11): return 'PCA', ''
    elif val in (12, 13): return 'BA', 's'
    elif val == 14:     return 'BA', ''
    elif val in (15, 16): return 'VA', ''
    else:               return '', ''


def aneurysm_postprocess(
    study_id: str,
    process_dir: str,
    output_dir: str,
    group_id: int = 56,
    input_json: str = "",
) -> bool:
    try:
        t0 = time.time()
        logger.info("[postprocess] %s start", study_id)

        path_nnunet = os.path.join(process_dir, "nnUNet")
        path_nii = os.path.join(path_nnunet, "Image_nii")
        path_dcm = os.path.join(path_nnunet, "Dicom")
        path_excel = os.path.join(path_nnunet, "excel")
        path_json_out = os.path.join(path_nnunet, "JSON")

        for d in [path_excel, path_json_out, output_dir,
                  os.path.join(path_dcm, "Dicom-Seg")]:
            os.makedirs(d, exist_ok=True)

        # ── 1. Verify + load inference outputs ───────────────────────────
        pred_path = os.path.join(path_nnunet, "Pred.nii.gz")
        if not os.path.isfile(pred_path):
            logger.error("Missing Pred.nii.gz")
            return False

        pred_nii = nib.load(pred_path)
        pred_arr = np.asanyarray(pred_nii.dataobj)

        # ── 2. Copy outputs to output_dir ────────────────────────────────
        for src, name in [
            (pred_path, "Pred_Aneurysm.nii.gz"),
            (os.path.join(path_nnunet, "Prob.nii.gz"), "Prob_Aneurysm.nii.gz"),
            (os.path.join(process_dir, "Vessel.nii.gz"), "Pred_Aneurysm_Vessel.nii.gz"),
            (os.path.join(process_dir, "Vessel_16.nii.gz"), "Pred_Aneurysm_Vessel16.nii.gz"),
        ]:
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(output_dir, name))

        synthseg_src = os.path.join(path_nnunet, "SynthSEG.nii.gz")
        if os.path.isfile(synthseg_src):
            shutil.copy(synthseg_src, os.path.join(output_dir, "SynthSEG_Aneurysm.nii.gz"))
            # Also copy to Image_nii for dicomseg
            dest = os.path.join(path_nii, "NEW_MRA_BRAIN_synthseg33_1mm.nii.gz")
            if not os.path.isfile(dest):
                shutil.copy(synthseg_src, dest)

        # ── 3. Generate Excel (per-lesion stats) ─────────────────────────
        # Old pipeline does data_translate (flip_to_cnn) on pred and v16 before analysis
        from nifti_utils import flip_to_cnn
        pred_cnn = flip_to_cnn(pred_arr, pred_nii)
        v16_path = os.path.join(process_dir, "Vessel_16.nii.gz")
        v16_cnn = None
        if os.path.isfile(v16_path):
            v16_nii = nib.load(v16_path)
            v16_cnn = flip_to_cnn(np.asanyarray(v16_nii.dataobj), v16_nii)
        _generate_excel(pred_cnn, pred_nii, process_dir, path_nnunet, study_id, v16_cnn)

        # ── 4. RAD Platform JSON + DICOM-SEG via make_aneurysm_pred_json ──
        # Replaces ws2030 execute_dicomseg_platform_json (incompatible schema).
        # RAD wrappers write rdx_aneurysm_pred_json.json + rdx_vessel_dilated_json.json
        # under nnUNet/, and DICOM-SEG files under nnUNet/Dicom/Dicom-Seg/.
        n_lesions = len(np.unique(pred_arr)) - 1  # exclude background

        from code_ai.pipeline.dicomseg.build_aneurysm import main as make_aneurysm_pred_json
        from code_ai.pipeline.dicomseg.build_vessel_dilated import main as make_vessel_pred_json
        # NOTE: build_aneurysm.main 3rd arg is `model_id: str`, NOT group_id.
        # Earlier we passed group_id (int 56) which wrote model_id:56 into the rdx
        # JSON and broke backend validation. Use the default model_id UUID.
        make_aneurysm_pred_json(study_id, pathlib.Path(path_nnunet))
        make_vessel_pred_json(study_id, pathlib.Path(path_nnunet))
        logger.info("RAD platform JSON + DICOM-SEG done (%d lesions)", n_lesions)

        # ── 5. RAD upload to AI_INFERENCE_RESULT_PATH ──
        # Replaces ws2030 deliver_results (Laravel API).
        # Mirrors pipeline_aneurysm_tensorflow.py lines ~439-580 logic:
        # writes ai-inference-result/<study_uid>/{aneurysm_model,vessel_model}/<infer_id>/
        # RAD backend monitors this dir and handles Orthanc upload separately.
        import json as _json

        upload_dir = os.environ.get(
            "RADX_UPLOAD_DIR",
            os.environ.get("AI_INFERENCE_RESULT_PATH", "/home/david/ai-inference-result"),
        )
        path_dicomseg_n = os.path.join(path_dcm, "Dicom-Seg")
        aneurysm_json_file = os.path.join(path_nnunet, "rdx_aneurysm_pred_json.json")
        vessel_json_file = os.path.join(path_nnunet, "rdx_vessel_dilated_json.json")

        if not os.path.isfile(aneurysm_json_file):
            logger.error("RAD JSON not found: %s", aneurysm_json_file)
            return False

        with open(aneurysm_json_file, "r", encoding="utf-8") as f:
            aneurysm_data = _json.load(f)

        study_instance_uid = (aneurysm_data.get("input_study_instance_uid") or [""])[0]
        aneurysm_inference_id = str(aneurysm_data.get("inference_id") or "unknown")

        study_dir = os.path.join(upload_dir, study_instance_uid)
        aneurysm_model_root = os.path.join(study_dir, "aneurysm_model")
        aneurysm_infer_dir = os.path.join(aneurysm_model_root, aneurysm_inference_id)
        os.makedirs(aneurysm_infer_dir, exist_ok=True)
        shutil.copy(aneurysm_json_file, os.path.join(aneurysm_infer_dir, "prediction.json"))

        # MRA_BRAIN DICOM-SEG
        for det in aneurysm_data.get("detections", []) or []:
            label = str(det.get("label") or "")
            seg_series_uid = str(det.get("series_instance_uid") or "")
            if not label or not seg_series_uid:
                continue
            src = os.path.join(path_dicomseg_n, f"MRA_BRAIN_{label}.dcm")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(aneurysm_infer_dir, f"{seg_series_uid}_{label}.dcm"))

        # MIP Pitch/Yaw — series DICOM folder + DICOM-SEG flat
        for series_item in aneurysm_data.get("reformatted_series", []) or []:
            series_folder_uid = str(series_item.get("series_instance_uid") or "")
            if not series_folder_uid:
                continue
            sd = str(series_item.get("series_description") or "").lower()
            if sd == "mip_pitch":
                src_prefix = "MIP_Pitch"
            elif sd == "mip_yaw":
                src_prefix = "MIP_Yaw"
            else:
                continue
            series_dir = os.path.join(aneurysm_infer_dir, series_folder_uid)
            os.makedirs(series_dir, exist_ok=True)
            src_dicom_dir = os.path.join(path_nnunet, "Dicom", src_prefix)
            if os.path.isdir(src_dicom_dir):
                for fn in sorted(os.listdir(src_dicom_dir)):
                    src_fp = os.path.join(src_dicom_dir, fn)
                    if os.path.isfile(src_fp):
                        shutil.copy(src_fp, os.path.join(series_dir, fn))
            for det in series_item.get("detections", []) or []:
                label = str(det.get("label") or "")
                seg_series_uid = str(det.get("series_instance_uid") or "")
                if not label or not seg_series_uid:
                    continue
                src = os.path.join(path_dicomseg_n, f"{src_prefix}_{label}.dcm")
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(aneurysm_infer_dir, f"{seg_series_uid}_{label}.dcm"))

        # Vessel model — Vessel JSON + single DICOM-SEG (MRA_BRAIN_Vessel_A1.dcm)
        if os.path.isfile(vessel_json_file):
            with open(vessel_json_file, "r", encoding="utf-8") as f:
                vessel_data = _json.load(f)
            vessel_inference_id = str(vessel_data.get("inference_id") or "unknown")
            vessel_model_root = os.path.join(study_dir, "vessel_model")
            vessel_infer_dir = os.path.join(vessel_model_root, vessel_inference_id)
            os.makedirs(vessel_infer_dir, exist_ok=True)
            shutil.copy(vessel_json_file, os.path.join(vessel_infer_dir, "prediction.json"))
            vessel_src = os.path.join(path_dicomseg_n, "MRA_BRAIN_Vessel_A1.dcm")
            try:
                _vessel_det = (vessel_data.get("detections") or [{}])[0]
                vessel_seg_uid = str(_vessel_det.get("series_instance_uid") or "")
                vessel_label = str(_vessel_det.get("label") or "")
            except Exception:
                vessel_seg_uid = ""
                vessel_label = ""
            if os.path.exists(vessel_src) and vessel_seg_uid and vessel_label:
                # '<uid>_<label>.dcm' — seg-naming.md rule 1. The underscore is
                # not decoration: since RADAX-615 the platform pairs SEGs with
                # startsWith(uid + "_"), so a bare '<uid>.dcm' is skipped, and
                # skipped silently — the finding imports with no mask behind it.
                shutil.copy(
                    vessel_src,
                    os.path.join(vessel_infer_dir, f"{vessel_seg_uid}_{vessel_label}.dcm"),
                )
            elif os.path.exists(vessel_src):
                logger.error(
                    "vessel SEG not delivered: series_instance_uid=%r label=%r — "
                    "both are required to name the file, and a missing name means "
                    "the platform shows a finding with no mask",
                    vessel_seg_uid, vessel_label,
                )

        # Copy RAD JSONs to output_dir (rename_nifti — for legacy consumers)
        shutil.copy(
            aneurysm_json_file,
            os.path.join(output_dir, "Pred_Aneurysm_rdx_aneurysm_pred_json.json"),
        )
        if os.path.isfile(vessel_json_file):
            shutil.copy(
                vessel_json_file,
                os.path.join(output_dir, "Pred_Vessel_dilated_rdx_vessel_dilated_pred_json.json"),
            )
        logger.info("RAD upload done -> %s", aneurysm_infer_dir)

        # ── 6. Notify RAD backend (inference complete) ───────────────────
        # Inline pipeline_aneurysm_tensorflow.py calls upload_inference_complete
        # after rdx upload — container path must mirror it so the platform
        # flips the case from "running" to "done". Backend schema accepts
        # both aneurysm_model and vessel_model.
        _notify_platform_complete(aneurysm_json_file, "aneurysm_model")
        if os.path.isfile(vessel_json_file):
            _notify_platform_complete(vessel_json_file, "vessel_model")

        # ── 7. Followup — CP9 ────────────────────────────────────────────
        if input_json:
            logger.info("Followup skipped — CP9")

        logger.info("[postprocess] %s done (%.0fs)", study_id, time.time() - t0)
        return True

    except Exception as exc:
        logger.error("[postprocess] %s failed: %s", study_id, exc, exc_info=True)
        return False


def _generate_excel(pred_arr, pred_nii, process_dir, path_nnunet, study_id, v16_arr=None):
    """Generate Aneurysm_Pred_list.xlsx with per-lesion stats + vessel location.

    IMPORTANT: Use original label values from Pred.nii.gz directly — do NOT redo
    connected component labeling. Pred.nii.gz is already CC-labeled by gpu_aneurysm.py
    filter_aneurysm() with sequential labels (1, 2, 3...). Re-doing CC labeling after
    flip_to_cnn changes spatial order and causes label mismatch with DICOM-SEG.
    (Bug AP-040: lesion number vs mask mismatch)
    """
    # Use original labels from Pred.nii.gz (already CC-labeled by inference)
    unique_labels = np.unique(pred_arr)
    unique_labels = unique_labels[unique_labels > 0]
    n_lesions = len(unique_labels)

    # Load Prob for confidence scores
    prob_path = os.path.join(path_nnunet, "Prob.nii.gz")
    if os.path.isfile(prob_path):
        from nifti_utils import flip_to_cnn as _flip
        prob_nii = nib.load(prob_path)
        prob_arr = _flip(np.asanyarray(prob_nii.dataobj), prob_nii)
    else:
        prob_arr = None

    # v16_arr passed from caller (already in CNN space)

    # Spacing for diameter calculation (aneurysm size = max diameter in mm, not volume)
    spacing = pred_nii.header.get_zooms()[:3]

    row = {
        "PatientID": study_id.split("_")[0] if "_" in study_id else study_id,
        "StudyDate": study_id.split("_")[1] if len(study_id.split("_")) > 1 else "",
        "AccessionNumber": study_id.split("_")[3] if len(study_id.split("_")) > 3 else "",
        "Aneurysm_Number": n_lesions,
    }

    if n_lesions > 0:
        import cv2
        from collections import Counter
        from scipy.ndimage import binary_dilation

        for lbl in unique_labels:
            idx = int(lbl)  # label value = lesion index (1, 2, 3...)
            mask = pred_arr == lbl

            # Prob stats for this lesion
            if prob_arr is not None:
                prob_vals = prob_arr[mask]
                prob_max = round(float(np.max(prob_vals)), 2)
                prob_mean = round(float(np.mean(prob_vals)), 2)
            else:
                prob_max = 1.0
                prob_mean = 1.0

            # Aneurysm size: per-slice minEnclosingCircle, take max, then -0.5
            # (exact copy of old pipeline calculate_aneurysm_long_axis_make_pred)
            cluster_matrix = np.zeros(pred_arr.shape, dtype=np.uint8)
            cluster_matrix[mask] = 254
            z_slices = np.where(np.sum(cluster_matrix, axis=(0, 1)) > 0)[0]
            max_diameter_mm = 0.0
            for zs in z_slices:
                slice_img = cluster_matrix[:, :, zs].copy()
                ret, thresh = cv2.threshold(slice_img, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, 1, 2)
                if contours:
                    (_, _), radius = cv2.minEnclosingCircle(contours[0])
                    d_mm = 2 * radius * spacing[0]
                    if d_mm > max_diameter_mm:
                        max_diameter_mm = d_mm
            size_mm = round(max_diameter_mm, 1) - 0.5

            # Vessel location from 16-label map (old pipeline: overlap → Counter → decode_location)
            location = ""
            sub_location = ""
            if v16_arr is not None:
                overlap = mask * v16_arr
                non_zero = overlap[overlap != 0]
                if len(non_zero) > 0:
                    counter = Counter(non_zero.astype(int).tolist())
                    most_common_val = counter.most_common(1)[0][0]
                    location, sub_location = _decode_vessel_location(most_common_val)
                else:
                    # Dilation fallback (old pipeline: dilate until hitting vessel)
                    dilated = mask.copy()
                    for _ in range(50):  # max 50 iterations
                        dilated = binary_dilation(dilated)
                        overlap = dilated * v16_arr
                        non_zero = overlap[overlap != 0]
                        if len(non_zero) > 0:
                            counter = Counter(non_zero.astype(int).tolist())
                            most_common_val = counter.most_common(1)[0][0]
                            location, sub_location = _decode_vessel_location(most_common_val)
                            break

            row[f"{idx}_size"] = size_mm
            row[f"{idx}_Prob_max"] = prob_max
            row[f"{idx}_Prob_mean"] = prob_mean
            row[f"{idx}_Confirm"] = ""
            row[f"{idx}_type"] = "saccular"
            row[f"{idx}_Location"] = location
            row[f"{idx}_SubLocation"] = sub_location
            row[f"{idx}_Location4labels"] = ""
            row[f"{idx}_Location6labels"] = ""
            row[f"{idx}_Comment"] = ""

    excel_path = os.path.join(path_nnunet, "excel", "Aneurysm_Pred_list.xlsx")
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    pd.DataFrame([row]).to_excel(excel_path, index=False)
    logger.info("Excel: %s (%d aneurysms)", excel_path, n_lesions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--study_id", required=True)
    p.add_argument("--process_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--group_id", type=int, default=56)
    p.add_argument("--input_json", default="")
    p.add_argument("--code_dir", default="")
    p.add_argument("--json_dir", default="")
    a = p.parse_args()
    sys.exit(0 if aneurysm_postprocess(a.study_id, a.process_dir, a.output_dir, a.group_id, a.input_json) else 1)
