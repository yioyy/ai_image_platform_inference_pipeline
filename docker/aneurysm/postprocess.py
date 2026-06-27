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

        # ── 4. Platform JSON + DICOM-SEG ─────────────────────────────────
        # Always generate — 0 lesion also needs study_model record in Laravel
        # (old pipeline calls make_pred_json unconditionally)
        n_lesions = len(np.unique(pred_arr)) - 1  # exclude background
        from code_ai.pipeline.dicomseg.aneurysm import execute_dicomseg_platform_json
        execute_dicomseg_platform_json(_id=study_id, root_path=str(path_nnunet), group_id=group_id)
        logger.info("Platform JSON + DICOM-SEG done (%d lesions)", n_lesions)

        # ── 5. Deliver to Orthanc + Laravel ──────────────────────────────
        from code_ai.pipeline.deliver import deliver_results
        # dicomseg/aneurysm.py saves JSON as nnUNet/aneurysm_platform_json.json
        json_file = os.path.join(path_nnunet, "aneurysm_platform_json.json")
        # Also check legacy path
        if not os.path.isfile(json_file):
            json_file = os.path.join(path_json_out, f"{study_id}_platform_json.json")
        delivery = deliver_results(
            model_type="Aneurysm",
            study_id=study_id,
            platform_json_path=json_file if os.path.isfile(json_file) else "",
            dicom_seg_dir=os.path.join(path_dcm, "Dicom-Seg"),
            extra_orthanc_dirs=[
                os.path.join(path_dcm, d)
                for d in ["MRA_BRAIN", "MIP_Pitch", "MIP_Yaw"]
                if os.path.isdir(os.path.join(path_dcm, d))
            ],
            group_id=group_id,
        )
        if not delivery.success:
            logger.error("deliver failed: %s", delivery)
            return False
        if os.path.isfile(json_file):
            shutil.copy(json_file, os.path.join(output_dir, "Pred_Aneurysm_platform_json.json"))

        # ── 6. Followup — CP9 ────────────────────────────────────────────
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
