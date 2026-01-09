import json
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import pydicom

from code_ai.pipeline.dicomseg import utils


def _utc_iso_now_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def execute_rdx_platform_json(
    *,
    patient_id: str,
    path_root: pathlib.Path,
    model_id: str = "924d1538-597c-41d6-bc27-4b0b359111cf",
) -> pathlib.Path:
    """
    從「平台 JSON + 既有 DICOM-SEG」產生 RADAX WMH prediction json。
    """
    path_root = pathlib.Path(path_root)
    path_dcms = path_root.joinpath("Dicom")
    path_json_dir = path_root.joinpath("JSON")
    path_dcmseg = path_dcms.joinpath("Dicom-Seg")

    platform_json_path = path_json_dir.joinpath(f"{patient_id}_platform_json.json")
    with open(platform_json_path, "r", encoding="utf-8") as f:
        platform_payload: Dict[str, Any] = json.load(f)

    # 取 input series/study uid
    _, _, first_dcm, source_images = utils.load_and_sort_dicom_files(path_dcms.joinpath("T2FLAIR_AXI"))
    input_study_uid = str(getattr(first_dcm, "StudyInstanceUID", ""))
    input_series_uid = str(getattr(first_dcm, "SeriesInstanceUID", ""))

    # 讀取平台 instances（主要取 Total 的 instance；也支援多 lesion）
    mask_obj = platform_payload.get("mask") or {}
    models = (mask_obj.get("model") or []) if isinstance(mask_obj, dict) else []

    detections: List[Dict[str, Any]] = []
    for model in models:
        for series in (model.get("series") or []):
            series_type = str(series.get("series_type") or "")
            if series_type != "T2FLAIR_AXI":
                continue
            for inst in (series.get("instances") or []):
                mask_name = str(inst.get("mask_name") or "")
                if not mask_name:
                    continue
                seg_path = path_dcmseg.joinpath(f"{series_type}_{mask_name}.dcm")
                if not seg_path.exists():
                    continue
                seg_ds = pydicom.dcmread(str(seg_path), stop_before_pixels=True)
                detections.append(
                    {
                        "annotated_series_instance_uid": input_series_uid,
                        "series_instance_uid": str(getattr(seg_ds, "SeriesInstanceUID", "")),
                        "sop_instance_uid": str(getattr(seg_ds, "SOPInstanceUID", "")),
                        "label": mask_name,
                        "type": str(inst.get("type") or ""),
                        "location": str(inst.get("location") or ""),
                        "fazekas": str(inst.get("fazekas") or ""),
                        "volume": round(_safe_float(inst.get("volume")), 1),
                        "main_seg_slice": _safe_int(inst.get("main_seg_slice")),
                        "probability": _safe_float(inst.get("prob_max")),
                        "mask_index": _safe_int(inst.get("mask_index")),
                        "sub_location": "",
                    }
                )

    output_json: Dict[str, Any] = {
        "inference_id": str(uuid.uuid4()),
        "inference_timestamp": _utc_iso_now_ms(),
        "input_study_instance_uid": [input_study_uid] if input_study_uid else [],
        "input_series_instance_uid": [input_series_uid] if input_series_uid else [],
        "model_id": model_id,
        "patient_id": str(patient_id),
        "detections": detections,
    }

    out_path = path_root.joinpath("rdx_wmh_pred_json.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False)
    return out_path


def main(
    patient_id: str = "00144245",
    path_processID: pathlib.Path = pathlib.Path("./nnUNet"),
    model_id: str = "924d1538-597c-41d6-bc27-4b0b359111cf",
):
    execute_rdx_platform_json(patient_id=patient_id, path_root=path_processID, model_id=model_id)


if __name__ == "__main__":
    main()

