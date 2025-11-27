"""
InfarctPipeline
================

產生梗塞統計用的 Excel，供後續 DICOM/JSON 流程使用。
資料來源：
- `Result/Pred.nii.gz`：整體梗塞遮罩
- `Result/Pred_synthseg.nii.gz`：套用 SynthSeg 後的分區遮罩
- `Image_nii/ADC.nii.gz`：計算 Mean ADC
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd


class InfarctPipeline:
    """依據 nnUNet 結果產生梗塞統計 Excel。"""

    def __init__(
        self,
        path_dcm: str,
        path_result: str,
        path_excel: str,
        patient_id: str,
        dataset_path: str,
        path_image_nii: str,
    ) -> None:
        self.path_dcm = path_dcm
        self.path_result = path_result
        self.path_excel = path_excel
        self.patient_id = patient_id
        self.dataset_path = dataset_path
        self.path_image_nii = path_image_nii

    def run_all(self) -> Dict[str, float]:
        """執行所有流程並返回摘要。"""
        os.makedirs(self.path_excel, exist_ok=True)
        summary = self._compute_summary()
        self._write_excel(summary)
        return summary

    def _compute_summary(self) -> Dict[str, any]:
        pred_path = os.path.join(self.path_result, "Pred.nii.gz")
        synthseg_path = os.path.join(self.path_result, "Pred_synthseg.nii.gz")
        adc_path = os.path.join(self.path_image_nii, "ADC.nii.gz")
        prob_path = self._locate_prob_nifti()

        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"找不到 Pred.nii.gz: {pred_path}")

        pred_img = nib.load(pred_path)
        pred_data = np.asanyarray(pred_img.dataobj)
        voxel_ml = float(np.prod(pred_img.header.get_zooms()) / 1000.0)
        mask = pred_data > 0
        total_voxels = int(mask.sum())
        total_volume_ml = round(total_voxels * voxel_ml, 1)

        if not os.path.exists(adc_path):
            raise FileNotFoundError(f"找不到 ADC.nii.gz: {adc_path}")
        adc_img = nib.load(adc_path)
        adc_data = np.asanyarray(adc_img.dataobj)
        mean_adc = float(np.mean(adc_data[mask])) if total_voxels > 0 else 0.0

        prob_data = None
        if prob_path and os.path.exists(prob_path):
            prob_img = nib.load(prob_path)
            prob_data = np.asanyarray(prob_img.dataobj)

        lesions: List[Dict[str, any]] = [
            self._stats_for_mask(
                location="Total",
                mask=mask,
                voxel_ml=voxel_ml,
                adc_data=adc_data,
                prob_data=prob_data,
            )
        ]
        lesions.extend(
            self._compute_region_stats(
                synthseg_path=synthseg_path,
                voxel_ml=voxel_ml,
                adc_data=adc_data,
                prob_data=prob_data,
            )
        )

        patient_info = self._parse_patient_info()

        return {
            "patient_info": patient_info,
            "total_volume_ml": total_volume_ml,
            "mean_adc": mean_adc,
            "lesions": lesions,
        }

    def _locate_prob_nifti(self) -> Optional[str]:
        candidates = [
            os.path.join(self.path_result, "Prob.nii.gz"),
            os.path.join(Path(self.path_result).parent, "Prob.nii.gz"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def _compute_region_stats(
        self,
        synthseg_path: str,
        voxel_ml: float,
        adc_data: np.ndarray,
        prob_data: Optional[np.ndarray],
    ) -> List[Dict[str, any]]:
        if not os.path.exists(synthseg_path):
            return []

        with open(self.dataset_path, "r", encoding="utf-8") as fp:
            dataset = json.load(fp)

        labels_config = dataset.get("Infarct_labels_OHIF", {})
        synth_img = nib.load(synthseg_path)
        synth_data = np.asanyarray(synth_img.dataobj).astype(np.int32)

        stats_map: Dict[str, Dict[str, any]] = {}
        for region, label_ids in labels_config.items():
            if region in {"Background", "CSF"}:
                continue
            mask = np.isin(synth_data, np.array(label_ids, dtype=np.int32))
            if not np.any(mask):
                continue
            stats = self._stats_for_mask(region, mask, voxel_ml, adc_data, prob_data)
            if stats["volume_ml"] >= 0.3:
                stats_map[region] = stats

        ordered = sorted(stats_map.values(), key=lambda item: item["volume_ml"], reverse=True)
        if len(ordered) <= 15:
            return ordered

        limited: List[Dict[str, any]] = []
        for item in ordered:
            if item["volume_ml"] >= 1 or len(limited) < 15:
                limited.append(item)
            if len(limited) == 15:
                break
        return limited

    def _stats_for_mask(
        self,
        location: str,
        mask: np.ndarray,
        voxel_ml: float,
        adc_data: np.ndarray,
        prob_data: Optional[np.ndarray],
    ) -> Dict[str, any]:
        voxel_count = int(mask.sum())
        volume_ml = round(voxel_count * voxel_ml, 1)

        if voxel_count > 0:
            mean_adc = float(np.mean(adc_data[mask]))
        else:
            mean_adc = 0.0

        if prob_data is not None and voxel_count > 0:
            prob_values = prob_data[mask]
            prob_max = float(np.max(prob_values))
            prob_mean = float(np.mean(prob_values))
        else:
            prob_max = 0.0
            prob_mean = 0.0

        return {
            "location": location,
            "volume_ml": volume_ml,
            "mean_adc": round(mean_adc),
            "prob_max": round(prob_max, 2),
            "prob_mean": round(prob_mean, 2),
            "type": "",
        }

    def _write_excel(self, summary: Dict[str, any]) -> None:
        patient_info = summary["patient_info"]
        lesions = summary["lesions"]
        row: Dict[str, any] = {
            "PatientID": patient_info["patient_id"],
            "StudyDate": patient_info["study_date"],
            "AccessionNumber": patient_info["accession_number"],
            "Lesion_Number": len(lesions),
        }

        for idx, lesion in enumerate(lesions, start=1):
            prefix = f"{idx}_"
            row[f"{prefix}size"] = lesion["volume_ml"]
            row[f"{prefix}Prob_max"] = lesion["prob_max"]
            row[f"{prefix}Prob_mean"] = lesion["prob_mean"]
            row[f"{prefix}Confirm"] = ""
            row[f"{prefix}type"] = lesion["type"]
            row[f"{prefix}Location"] = lesion["location"]
            row[f"{prefix}MeanADC"] = lesion["mean_adc"]

        df = pd.DataFrame([row])
        out_path = os.path.join(self.path_excel, "Infarct_Pred_list.xlsx")
        df.to_excel(out_path, index=False)

    def _parse_patient_info(self) -> Dict[str, str]:
        parts = self.patient_id.split("_")
        patient = parts[0] if parts else self.patient_id
        study_date = parts[1] if len(parts) > 1 else ""
        accession = parts[3] if len(parts) > 3 else ""

        return {
            "patient_id": patient,
            "study_date": study_date,
            "accession_number": accession,
        }


