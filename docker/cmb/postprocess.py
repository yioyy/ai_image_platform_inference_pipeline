#!/usr/bin/env python3
"""CMB Postprocessing — DICOM-SEG generation + delivery.

Input:  Pred_CMB.nii.gz + Pred_CMB.json (from inference, at process_dir)
Output: DICOM-SEG → Orthanc + platform_json → Laravel

Zero Chuan code dependencies. Uses David's code_ai modules only.
"""

import argparse
import json
import logging
import os
import pathlib
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

        pred_nii = os.path.join(process_dir, "Pred_CMB.nii.gz")
        pred_json_path = os.path.join(process_dir, "Pred_CMB.json")

        if not os.path.isfile(pred_nii):
            logger.error("Missing Pred_CMB.nii.gz: %s", pred_nii)
            return False

        # ── 1. DICOM-SEG + platform_json (Python import, not subprocess) ──
        t = time.time()

        seg_output_dir = os.path.join(process_dir, study_id)
        os.makedirs(seg_output_dir, exist_ok=True)

        from code_ai.pipeline.dicomseg import utils
        from code_ai.pipeline.dicomseg.schema.enum import SeriesTypeEnum
        from code_ai.pipeline.dicomseg.cmb import NewReviewCMBPlatformJSONBuilder

        # Load prediction NIfTI
        new_nifti_array = utils.get_array_to_dcm_axcodes(pathlib.Path(pred_nii))
        pred_data_unique = np.unique(new_nifti_array)

        n_lesions = len(pred_data_unique[pred_data_unique > 0])
        logger.info("CMB lesions detected: %d", n_lesions)

        # Always load source DICOM — needed for platform JSON even with 0 lesions
        if not dicom_dir or not os.path.isdir(dicom_dir):
            logger.warning("No DICOM dir for DICOM-SEG: %s", dicom_dir)
        else:
            sorted_dcms, image, first_dcm, source_images = utils.load_and_sort_dicom_files(dicom_dir)

            # Create DICOM-SEG files (only when lesions > 0)
            result_list = []
            if n_lesions > 0:
                series_name = "Pred_CMB"
                pred_data_nonzero = pred_data_unique[pred_data_unique > 0]
                result_list = utils.create_dicom_seg_file(
                    pred_data_nonzero, new_nifti_array, series_name,
                    pathlib.Path(seg_output_dir), image, first_dcm, source_images,
                )
            else:
                logger.info("0 CMB detected — skip DICOM-SEG")

            # Load pred JSON and convert to builder-expected format
            pred_json = []
            if n_lesions > 0 and os.path.isfile(pred_json_path):
                with open(pred_json_path) as f:
                    raw = json.load(f)
                lesions = raw.get("lesions", raw) if isinstance(raw, dict) else raw
                for le in lesions:
                    pred_json.append({
                        "label#": le.get("index", le.get("label#", 0)),
                        "CMB_prob": le.get("cmb_prob", le.get("CMB_prob", 0)),
                        "pred_diameter": le.get("diameter_mm", le.get("pred_diameter", 0)),
                        "class_name": le.get("class", le.get("class_name", "CMB")),
                        "type_name": le.get("type_name", ""),
                        "type": le.get("type", ""),
                    })

            # Build platform_json (always — 0 lesion also needs study_model in Laravel)
            # CP1c (AP-085): explicit --group_id arg 優先（per-task）；fallback 到 env GROUP_ID（legacy）
            if group_id is None:
                group_id = int(os.environ.get("GROUP_ID_CMB", os.environ.get("GROUP_ID", "56")))
            dicom_seg_result_list = [{"series_type": SeriesTypeEnum.SWAN, "data": result_list}]
            pred_json_list = [{"series_type": SeriesTypeEnum.SWAN, "data": pred_json}]

            builder = NewReviewCMBPlatformJSONBuilder()
            platform_json_obj = (
                builder
                .set_series_type(SeriesTypeEnum.SWAN, source_images=source_images)
                .set_group_id(group_id)
                .build_sorted()
                .build_study(pred_json_list=pred_json_list)
                .build_mask(dicom_seg_result_list=dicom_seg_result_list, pred_json_list=pred_json_list)
                .build()
            )

            platform_json_path = os.path.join(process_dir, "Pred_CMB_platform_json.json")
            with open(platform_json_path, "w") as f:
                f.write(platform_json_obj.model_dump_json())
            logger.info("Platform JSON: %s (%d lesions)", platform_json_path, n_lesions)

        logger.info("DICOM-SEG done (%.0fs)", time.time() - t)

        # ── 2. Deliver to Orthanc + Laravel ──────────────────────────────
        from code_ai.pipeline.deliver import deliver_results

        platform_json = os.path.join(process_dir, "Pred_CMB_platform_json.json")

        delivery = deliver_results(
            model_type="CMB",
            study_id=study_id,
            platform_json_path=platform_json if os.path.isfile(platform_json) else "",
            dicom_seg_dir=seg_output_dir,
            group_id=group_id,
        )
        if not delivery.success:
            logger.error("deliver_results failed: %s", delivery)
            return False

        # ── 3. Followup — 未來實作 ────────────────────────────────────────
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
