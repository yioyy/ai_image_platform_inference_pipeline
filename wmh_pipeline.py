"""WMH Pipeline: compute lesion statistics and export Excel summary."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd


class WMHPipeline:
    """Compute WMH statistics and export Excel summary."""

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
        os.makedirs(self.path_excel, exist_ok=True)
        summary = self._compute_summary()
        self._write_excel(summary)
        return summary

    def _compute_summary(self) -> Dict[str, any]:
        pred_path = os.path.join(self.path_result, "Pred.nii.gz")
        synthseg_path = os.path.join(self.path_result, "Pred_synthseg.nii.gz")
        prob_path = os.path.join(self.path_result, "Prob.nii.gz")

        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"找不到 Pred.nii.gz: {pred_path}")

        pred_img = nib.load(pred_path)
        pred_data = np.asanyarray(pred_img.dataobj).astype(np.uint8)
        voxel_ml = float(np.prod(pred_img.header.get_zooms()) / 1000.0)
        mask = pred_data > 0
        total_voxels = int(mask.sum())
        total_volume_ml = round(total_voxels * voxel_ml, 1)

        prob_data = None
        if os.path.exists(prob_path):
            prob_img = nib.load(prob_path)
            prob_data = np.asanyarray(prob_img.dataobj).astype(np.float32)

        synthseg_data = None
        if os.path.exists(synthseg_path):
            synthseg_img = nib.load(synthseg_path)
            synthseg_data = np.asanyarray(synthseg_img.dataobj).astype(np.int32)

        with open(self.dataset_path, "r", encoding="utf-8") as fp:
            dataset = json.load(fp)
        labels_config = dataset.get("WMH_labels", {})

        lesions: List[Dict[str, any]] = []
        lesions.append(
            self._stats_for_mask(
                location="Total",
                mask=mask,
                voxel_ml=voxel_ml,
                prob_data=prob_data,
            )
        )

        if synthseg_data is not None:
            lesions.extend(
                self._compute_region_stats(
                    synthseg_data=synthseg_data,
                    mask=mask,
                    voxel_ml=voxel_ml,
                    prob_data=prob_data,
                    labels_config=labels_config,
                )
            )

        fazekas = self._compute_fazekas(mask, synthseg_data)
        patient_info = self._parse_patient_info()

        return {
            "patient_info": patient_info,
            "total_volume_ml": total_volume_ml,
            "lesions": lesions,
            "fazekas": fazekas,
        }

    def _compute_region_stats(
        self,
        synthseg_data: np.ndarray,
        mask: np.ndarray,
        voxel_ml: float,
        prob_data: Optional[np.ndarray],
        labels_config: Dict[str, List[int]],
    ) -> List[Dict[str, any]]:
        stats_map: Dict[str, Dict[str, any]] = {}
        for region, label_ids in labels_config.items():
            if region in {"Background"}:
                continue
            region_mask = np.isin(synthseg_data, np.array(label_ids, dtype=np.int32))
            region_mask &= mask
            if not np.any(region_mask):
                continue
            stats = self._stats_for_mask(region, region_mask, voxel_ml, None, prob_data)
            if stats["volume_ml"] >= 0.3:
                stats_map[region] = stats

        ordered = sorted(stats_map.values(), key=lambda item: item["volume_ml"], reverse=True)
        return ordered

    def _stats_for_mask(
        self,
        location: str,
        mask: np.ndarray,
        voxel_ml: float,
        prob_data_total: Optional[np.ndarray] = None,
        prob_data: Optional[np.ndarray] = None,
    ) -> Dict[str, any]:
        voxel_count = int(mask.sum())
        volume_ml = round(voxel_count * voxel_ml, 1)
        prob_source = prob_data if prob_data is not None else prob_data_total
        if prob_source is not None and voxel_count > 0:
            prob_values = prob_source[mask.astype(bool)]
            prob_max = float(np.max(prob_values))
            prob_mean = float(np.mean(prob_values))
        else:
            prob_max = 0.0
            prob_mean = 0.0

        return {
            "location": location,
            "volume_ml": volume_ml,
            "prob_max": round(prob_max, 4),
            "prob_mean": round(prob_mean, 4),
        }

    def _compute_fazekas(self, mask: np.ndarray, synthseg_data: Optional[np.ndarray]) -> int:
        if synthseg_data is None:
            return 0
        wm_mask = synthseg_data > 0
        total_brain = np.sum(wm_mask)
        if total_brain == 0:
            return 0
        percentage = (np.sum(mask) / total_brain) * 100.0
        if percentage == 0.0:
            return 0
        if percentage < 0.2923:
            return 1
        if percentage > 1.4825:
            return 3
        return 2

    def _write_excel(self, summary: Dict[str, any]) -> None:
        patient_info = summary["patient_info"]
        lesions = summary["lesions"]
        row: Dict[str, any] = {
            "PatientID": patient_info["patient_id"],
            "StudyDate": patient_info["study_date"],
            "AccessionNumber": patient_info["accession_number"],
            "Lesion_Number": len(lesions),
            "Fazekas": summary["fazekas"],
            "TotalVolume": summary["total_volume_ml"],
        }

        for idx, lesion in enumerate(lesions, start=1):
            prefix = f"{idx}_"
            row[f"{prefix}size"] = lesion["volume_ml"]
            row[f"{prefix}Prob_max"] = lesion["prob_max"]
            row[f"{prefix}Prob_mean"] = lesion["prob_mean"]
            row[f"{prefix}Location"] = lesion["location"]

        df = pd.DataFrame([row])
        out_path = os.path.join(self.path_excel, "WMH_Pred_list.xlsx")
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


