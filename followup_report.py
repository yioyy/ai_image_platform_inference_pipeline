"""
Follow-up report generation (Platform JSON).

本模組目標：
- 讀取 baseline/followup 的平台 JSON 與 NIfTI（baseline Pred、followup Pred(reg)）
- 以 baseline label index 為主鍵做病灶配對
- 產出 Followup_<model>_platform_json.json（沿用 baseline JSON 結構，新增 followup 欄位）

重要假設：
- baseline Pred 的 label index 與 baseline JSON 的 mask_index 對應
- followup Pred(reg) 的 label index 與 followup JSON 的 mask_index 對應
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import nibabel as nib
import numpy as np


TransformDirection = str  # "F2B" | "B2F"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverlapMatch:
    baseline_label: int
    followup_label: Optional[int]
    overlap_voxels: int


@dataclass(frozen=True)
class SortedSlice:
    sop_instance_uid: str
    projection: float


def make_followup_json(
    *,
    baseline_pred_file: Path,
    baseline_json_file: Path,
    followup_json_file: Path,
    followup_reg_pred_file: Path,
    registration_mat_file: Path,
    output_json_file: Path,
    group_id: int = 56,
    transform_direction: TransformDirection = "F2B",
) -> Path:
    baseline_pred = _load_label_volume(baseline_pred_file)
    followup_reg_pred = _load_label_volume(followup_reg_pred_file)
    if baseline_pred.shape != followup_reg_pred.shape:
        raise ValueError(
            "baseline Pred 與 followup Pred(reg) shape 不一致："
            f"{baseline_pred.shape} vs {followup_reg_pred.shape}"
        )

    baseline_payload = _read_json(baseline_json_file)
    followup_payload = _read_json(followup_json_file)

    _attach_followup_uids(baseline_payload, followup_payload)
    _attach_followup_series_uid_to_mask(baseline_payload, followup_payload)

    matches = _match_by_overlap(baseline_pred, followup_reg_pred)
    baseline_to_followup = {m.baseline_label: m.followup_label for m in matches}
    used_followup_labels = {m.followup_label for m in matches if m.followup_label is not None}

    baseline_instances = _collect_mask_instances_by_index(baseline_payload)
    followup_instances = _collect_mask_instances_by_index(followup_payload)

    baseline_sorted = _get_sorted_slices(baseline_payload)
    followup_sorted = _get_sorted_slices(followup_payload)

    mat = _load_fsl_mat(registration_mat_file)

    _update_baseline_instances_with_followup(
        baseline_payload=baseline_payload,
        baseline_instances=baseline_instances,
        followup_instances=followup_instances,
        baseline_to_followup=baseline_to_followup,
        baseline_sorted=baseline_sorted,
        followup_sorted=followup_sorted,
        mat=mat,
        transform_direction=transform_direction,
    )

    _append_disappeared_followup_instances(
        baseline_payload=baseline_payload,
        followup_payload=followup_payload,
        followup_instances=followup_instances,
        used_followup_labels=used_followup_labels,
        baseline_sorted=baseline_sorted,
        followup_sorted=followup_sorted,
        mat=mat,
        transform_direction=transform_direction,
    )

    if group_id:
        _force_group_id(baseline_payload, group_id)

    output_json_file.parent.mkdir(parents=True, exist_ok=True)
    output_json_file.write_text(json.dumps(baseline_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_json_file


def _load_label_volume(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到 NIfTI 檔案：{path}")
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"NIfTI 維度必須為 3D：{path} (ndim={data.ndim})")
    return data.astype(np.int32)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 JSON 檔案：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _attach_followup_uids(baseline_payload: Dict[str, Any], followup_payload: Dict[str, Any]) -> None:
    baseline_study = baseline_payload.get("study") or {}
    followup_study = followup_payload.get("study") or {}

    followup_study_uid = str(followup_study.get("study_instance_uid", "") or "")
    followup_series_uid = ""
    series_list = followup_study.get("series") or []
    if isinstance(series_list, list) and series_list:
        followup_series_uid = str(series_list[0].get("series_instance_uid", "") or "")

    baseline_series_list = baseline_study.get("series") or []
    if not isinstance(baseline_series_list, list):
        return

    for idx, series in enumerate(baseline_series_list):
        if not isinstance(series, dict):
            continue
        series["followup_study_instance_uid"] = followup_study_uid if idx == 0 else ""
        series["followup_series_instance_uid"] = followup_series_uid if idx == 0 else ""


def _attach_followup_series_uid_to_mask(baseline_payload: Dict[str, Any], followup_payload: Dict[str, Any]) -> None:
    followup_study = followup_payload.get("study") or {}
    series_list = followup_study.get("series") or []
    followup_series_uid = ""
    if isinstance(series_list, list) and series_list:
        followup_series_uid = str(series_list[0].get("series_instance_uid", "") or "")

    for series in _iter_mask_series(baseline_payload):
        series["followup_series_instance_uid"] = followup_series_uid


def _force_group_id(payload: Dict[str, Any], group_id: int) -> None:
    study = payload.get("study")
    if isinstance(study, dict):
        study["group_id"] = group_id
    mask = payload.get("mask")
    if isinstance(mask, dict):
        mask["group_id"] = group_id


def _iter_mask_series(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    mask = payload.get("mask")
    if not isinstance(mask, dict):
        return []

    series = mask.get("series")
    if isinstance(series, list):
        return [s for s in series if isinstance(s, dict)]

    models = mask.get("model")
    if not isinstance(models, list):
        return []

    out: List[Dict[str, Any]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        series_list = model.get("series")
        if not isinstance(series_list, list):
            continue
        out.extend([s for s in series_list if isinstance(s, dict)])
    return out


def _collect_mask_instances_by_index(payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    for series in _iter_mask_series(payload):
        instances = series.get("instances")
        if not isinstance(instances, list):
            continue
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            idx = inst.get("mask_index")
            if idx is None:
                continue
            try:
                key = int(idx)
            except (TypeError, ValueError):
                continue
            mapping[key] = inst
    return mapping


def _get_sorted_slices(payload: Dict[str, Any]) -> List[SortedSlice]:
    sorted_req = payload.get("sorted")
    if not isinstance(sorted_req, dict):
        return []
    series = sorted_req.get("series")
    if not isinstance(series, list) or not series:
        return []
    first = series[0]
    if not isinstance(first, dict):
        return []
    inst = first.get("instance")
    if not isinstance(inst, list):
        return []
    out: List[SortedSlice] = []
    for item in inst:
        if not isinstance(item, dict):
            continue
        uid = str(item.get("sop_instance_uid", "") or "")
        if not uid:
            continue
        proj = item.get("projection")
        try:
            proj_f = float(proj)
        except (TypeError, ValueError):
            continue
        out.append(SortedSlice(sop_instance_uid=uid, projection=proj_f))
    return out


def _match_by_overlap(baseline: np.ndarray, followup: np.ndarray) -> List[OverlapMatch]:
    baseline_labels = [int(x) for x in np.unique(baseline) if int(x) > 0]
    matches: List[OverlapMatch] = []
    for label in baseline_labels:
        mask = baseline == label
        overlapped = followup[mask]
        overlapped = overlapped[overlapped > 0]
        if overlapped.size == 0:
            matches.append(OverlapMatch(baseline_label=label, followup_label=None, overlap_voxels=0))
            continue
        counts = np.bincount(overlapped.astype(np.int64))
        best = int(np.argmax(counts))
        matches.append(OverlapMatch(baseline_label=label, followup_label=best, overlap_voxels=int(counts[best])))
    return matches


def _update_baseline_instances_with_followup(
    *,
    baseline_payload: Dict[str, Any],
    baseline_instances: Dict[int, Dict[str, Any]],
    followup_instances: Dict[int, Dict[str, Any]],
    baseline_to_followup: Dict[int, Optional[int]],
    baseline_sorted: List[SortedSlice],
    followup_sorted: List[SortedSlice],
    mat: np.ndarray,
    transform_direction: TransformDirection,
) -> None:
    for baseline_label, inst in baseline_instances.items():
        followup_label = baseline_to_followup.get(baseline_label)
        if followup_label is None:
            inst["status"] = "new"
            inst["followup_mask_index"] = ""
            inst["followup_diameter"] = ""
            inst["followup_seg_series_instance_uid"] = ""
            inst["followup_seg_sop_instance_uid"] = ""
            inst["followup_dicom_sop_instance_uid"] = _map_sop_uid_by_z_ratio(
                source_uid=str(inst.get("dicom_sop_instance_uid", "") or ""),
                source_sorted=baseline_sorted,
                target_sorted=followup_sorted,
                mat=mat,
                transform_direction="B2F",
                input_mat_direction=transform_direction,
                context="new",
            )
            continue

        f_inst = followup_instances.get(int(followup_label))
        if not f_inst:
            inst["status"] = "new"
            inst["followup_mask_index"] = ""
            inst["followup_diameter"] = ""
            inst["followup_seg_series_instance_uid"] = ""
            inst["followup_seg_sop_instance_uid"] = ""
            inst["followup_dicom_sop_instance_uid"] = ""
            continue

        inst["followup_mask_index"] = f_inst.get("mask_index", "")
        inst["followup_diameter"] = f_inst.get("diameter", "")
        inst["followup_seg_series_instance_uid"] = f_inst.get("seg_series_instance_uid", "")
        inst["followup_seg_sop_instance_uid"] = f_inst.get("seg_sop_instance_uid", "")
        # 病灶有配對(重疊)到時：沿用「之前 JSON」的 followup_dicom_sop_instance_uid，
        # 避免每次重跑都覆寫關鍵張 UID。只有沒有舊值時才補上 followup JSON 的值。
        prev_followup_dicom_uid = str(inst.get("followup_dicom_sop_instance_uid", "") or "")
        if not prev_followup_dicom_uid:
            inst["followup_dicom_sop_instance_uid"] = f_inst.get("dicom_sop_instance_uid", "")

        base_d = _safe_float(inst.get("diameter"))
        foll_d = _safe_float(f_inst.get("diameter"))
        if base_d > foll_d:
            inst["status"] = "enlarging"
        elif base_d < foll_d:
            inst["status"] = "shrinking"
        else:
            inst["status"] = "stable"


def _append_disappeared_followup_instances(
    *,
    baseline_payload: Dict[str, Any],
    followup_payload: Dict[str, Any],
    followup_instances: Dict[int, Dict[str, Any]],
    used_followup_labels: set[Optional[int]],
    baseline_sorted: List[SortedSlice],
    followup_sorted: List[SortedSlice],
    mat: np.ndarray,
    transform_direction: TransformDirection,
) -> None:
    unused = [k for k in followup_instances.keys() if k not in used_followup_labels]
    if not unused:
        return

    # Append to the first mask series (simplest, avoid deep nesting).
    series_list = list(_iter_mask_series(baseline_payload))
    if not series_list:
        return
    target_series = series_list[0]
    instances = target_series.get("instances")
    if not isinstance(instances, list):
        return

    for k in sorted(unused):
        f_inst = followup_instances.get(k)
        if not f_inst:
            continue
        entry: Dict[str, Any] = {}
        entry["mask_index"] = ""
        entry["mask_name"] = ""
        entry["diameter"] = ""
        entry["type"] = f_inst.get("type", "")
        entry["location"] = f_inst.get("location", "")
        entry["sub_location"] = f_inst.get("sub_location", None)
        entry["prob_max"] = f_inst.get("prob_max", "")
        entry["checked"] = f_inst.get("checked", "1")
        entry["is_ai"] = f_inst.get("is_ai", "1")
        entry["seg_series_instance_uid"] = ""
        entry["seg_sop_instance_uid"] = ""

        entry["followup_mask_index"] = f_inst.get("mask_index", "")
        entry["followup_diameter"] = f_inst.get("diameter", "")
        entry["followup_seg_series_instance_uid"] = f_inst.get("seg_series_instance_uid", "")
        entry["followup_seg_sop_instance_uid"] = f_inst.get("seg_sop_instance_uid", "")
        entry["followup_dicom_sop_instance_uid"] = f_inst.get("dicom_sop_instance_uid", "")

        entry["status"] = "disappeared"
        entry["dicom_sop_instance_uid"] = _map_sop_uid_by_z_ratio(
            source_uid=str(f_inst.get("dicom_sop_instance_uid", "") or ""),
            source_sorted=followup_sorted,
            target_sorted=baseline_sorted,
            mat=mat,
            transform_direction="F2B",
            input_mat_direction=transform_direction,
            context="disappeared",
        )
        instances.append(entry)


def _safe_float(value: Any) -> float:
    if value in ("", None):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_fsl_mat(path: Path) -> np.ndarray:
    if not path.exists():
        return np.eye(4, dtype=np.float64)
    mat = np.loadtxt(str(path), dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"registration mat 必須為 4x4：{path} (shape={mat.shape})")
    return mat


def _map_sop_uid_by_z_ratio(
    *,
    source_uid: str,
    source_sorted: List[SortedSlice],
    target_sorted: List[SortedSlice],
    mat: np.ndarray,
    transform_direction: TransformDirection,
    input_mat_direction: TransformDirection,
    context: str = "",
) -> str:
    """
    依「slice projection (mm)」做近似的 SOP UID 映射。

    - transform_direction：這次要做的映射方向（F2B 或 B2F）
    - input_mat_direction：輸入的 mat 方向（預設 F2B：followup->baseline）
    """
    if not source_uid or not source_sorted or not target_sorted:
        return ""

    z_src: Optional[float] = None
    for s in source_sorted:
        if s.sop_instance_uid == source_uid:
            z_src = s.projection
            break
    if z_src is None:
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "followup_report.sop_map(%s): source uid not found in sorted list: %s",
                context or "n/a",
                source_uid,
            )
        return ""

    m = _select_transform(mat, want=transform_direction, input_direction=input_mat_direction)
    z_tgt = float(m[2, 2] * z_src + m[2, 3])

    # target_sorted 的 projection 本身已排序；用二分搜尋找最接近的 slice。
    projections = np.asarray([t.projection for t in target_sorted], dtype=np.float64)
    idx = int(np.searchsorted(projections, z_tgt))
    if idx <= 0:
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "followup_report.sop_map(%s): %s->%s z_src=%.6f z_tgt=%.6f out_idx=%d out_proj=%.6f out_uid=%s",
                context or "n/a",
                input_mat_direction,
                transform_direction,
                z_src,
                z_tgt,
                0,
                target_sorted[0].projection,
                target_sorted[0].sop_instance_uid,
            )
        return target_sorted[0].sop_instance_uid
    if idx >= len(target_sorted):
        if LOGGER.isEnabledFor(logging.DEBUG):
            last = len(target_sorted) - 1
            LOGGER.debug(
                "followup_report.sop_map(%s): %s->%s z_src=%.6f z_tgt=%.6f out_idx=%d out_proj=%.6f out_uid=%s",
                context or "n/a",
                input_mat_direction,
                transform_direction,
                z_src,
                z_tgt,
                last,
                target_sorted[last].projection,
                target_sorted[last].sop_instance_uid,
            )
        return target_sorted[-1].sop_instance_uid
    prev_i = idx - 1
    if abs(projections[idx] - z_tgt) < abs(projections[prev_i] - z_tgt):
        chosen = idx
    else:
        chosen = prev_i

    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "followup_report.sop_map(%s): %s->%s z_src=%.6f z_tgt=%.6f out_idx=%d out_proj=%.6f out_uid=%s",
            context or "n/a",
            input_mat_direction,
            transform_direction,
            z_src,
            z_tgt,
            chosen,
            target_sorted[chosen].projection,
            target_sorted[chosen].sop_instance_uid,
        )

    return target_sorted[chosen].sop_instance_uid


def _select_transform(mat: np.ndarray, *, want: TransformDirection, input_direction: TransformDirection) -> np.ndarray:
    if want == input_direction:
        return mat
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.eye(4, dtype=np.float64)


