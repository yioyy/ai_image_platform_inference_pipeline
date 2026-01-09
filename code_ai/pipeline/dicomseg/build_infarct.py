import json
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pydicom

from code_ai.pipeline.dicomseg import utils


def _utc_iso_now_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _read_platform_instances(platform_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    讀取平台（aiteam）schema 的 mask instances，回傳扁平列表。
    每個元素會至少包含：
      - series_type (DWI1000 / ADC)
      - mask_name (A1 ...)
      - mask_index
      - location
      - mean_adc
      - volume
      - prob_max
      - main_seg_slice
      - type / sub_location（若不存在則給空字串）
    """
    instances: List[Dict[str, Any]] = []
    mask_obj = platform_json.get("mask") or {}
    models = (mask_obj.get("model") or []) if isinstance(mask_obj, dict) else []
    for model in models:
        for series in (model.get("series") or []):
            series_type = str(series.get("series_type") or "")
            for inst in (series.get("instances") or []):
                item = dict(inst)
                item["series_type"] = series_type
                item.setdefault("sub_location", "")
                item.setdefault("type", "")
                instances.append(item)
    return instances


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


def _load_series_uid(path_series_dir: pathlib.Path) -> Tuple[str, str]:
    """回傳 (StudyInstanceUID, SeriesInstanceUID)"""
    _, _, first_dcm, _ = utils.load_and_sort_dicom_files(path_series_dir)
    return str(getattr(first_dcm, "StudyInstanceUID", "")), str(getattr(first_dcm, "SeriesInstanceUID", ""))


def execute_rdx_platform_json(
    *,
    patient_id: str,
    path_root: pathlib.Path,
    model_id: str = "924d1538-597c-41d6-bc27-4b0b359111cf",
) -> pathlib.Path:
    """
    從「平台 JSON + 既有 DICOM-SEG」產生 RADAX Infarct prediction json。

    依需求：
    - `inference_id` / `inference_timestamp`
    - `input_study_instance_uid`（list）
    - `input_series_instance_uid`（list：DWI1000 + ADC）
    - `detections`：同一個 lesion 會各自對應到 DWI1000 與 ADC 的 DICOM-SEG（可在兩個序列上顯示）
    """
    path_root = pathlib.Path(path_root)
    path_dcms = path_root.joinpath("Dicom")
    path_json_dir = path_root.joinpath("JSON")
    path_dcmseg = path_dcms.joinpath("Dicom-Seg")

    # 平台 JSON：<id>_platform_json.json
    platform_json_path = path_json_dir.joinpath(f"{patient_id}_platform_json.json")
    with open(platform_json_path, "r", encoding="utf-8") as f:
        platform_payload: Dict[str, Any] = json.load(f)

    # 取 input series/study uid
    dwi_study_uid, dwi_series_uid = _load_series_uid(path_dcms.joinpath("DWI1000"))
    adc_study_uid, adc_series_uid = _load_series_uid(path_dcms.joinpath("ADC"))
    input_study_uid = dwi_study_uid or adc_study_uid
    input_series_uids = [uid for uid in [dwi_series_uid, adc_series_uid] if uid]

    # 讀取平台 instances（含 volume/mean_adc/prob）
    inst_list = _read_platform_instances(platform_payload)
    # 建立 key：(<series_type>, <mask_name>)
    inst_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for inst in inst_list:
        series_type = str(inst.get("series_type") or "")
        mask_name = str(inst.get("mask_name") or "")
        if series_type and mask_name:
            inst_map[(series_type, mask_name)] = inst

    detections: List[Dict[str, Any]] = []
    for mask_name in sorted({k[1] for k in inst_map.keys()}):
        for series_type, annotated_uid in (("DWI1000", dwi_series_uid), ("ADC", adc_series_uid)):
            inst = inst_map.get((series_type, mask_name))
            if not inst:
                continue
            seg_path = path_dcmseg.joinpath(f"{series_type}_{mask_name}.dcm")
            if not seg_path.exists():
                continue
            seg_ds = pydicom.dcmread(str(seg_path), stop_before_pixels=True)
            detections.append(
                {
                    "annotated_series_instance_uid": annotated_uid,
                    "series_instance_uid": str(getattr(seg_ds, "SeriesInstanceUID", "")),
                    "sop_instance_uid": str(getattr(seg_ds, "SOPInstanceUID", "")),
                    "label": mask_name,
                    "type": str(inst.get("type") or ""),
                    "location": str(inst.get("location") or ""),
                    "mean_adc": str(inst.get("mean_adc") or ""),
                    "volume": round(_safe_float(inst.get("volume")), 1),
                    "main_seg_slice": _safe_int(inst.get("main_seg_slice")),
                    "probability": _safe_float(inst.get("prob_max")),
                    "mask_index": _safe_int(inst.get("mask_index")),
                    "sub_location": str(inst.get("sub_location") or ""),
                }
            )

    output_json: Dict[str, Any] = {
        "inference_id": str(uuid.uuid4()),
        "inference_timestamp": _utc_iso_now_ms(),
        "input_study_instance_uid": [input_study_uid] if input_study_uid else [],
        "input_series_instance_uid": input_series_uids,
        "model_id": model_id,
        "patient_id": str(patient_id),
        "detections": detections,
    }

    out_path = path_root.joinpath("rdx_infarct_pred_json.json")
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

