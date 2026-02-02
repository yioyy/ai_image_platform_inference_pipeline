#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Follow-up v3 pipeline (NIfTI + multi-date).

輸入：
- follow_up_input.json（current + prior list）
- case 資料夾內的 Pred/SynthSEG NIfTI + rdx prediction.json

輸出：
- Deep_FollowUp/<current_case_id>/...（保留 v1 過程檔）
- followup_manifest.json（manifest + items）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np

from generate_followup_pred import generate_followup_pred
from pipeline_followup import _overwrite_pred_geometry_from_synthseg
from registration import run_registration_followup_to_baseline


LOGGER = logging.getLogger(__name__)

MODEL_BY_MODEL_ID = {
    "924d1538-597c-41d6-bc27-4b0b359111cf": "Aneurysm",
    "48c0cfa2-347b-4d32-aa74-a7b1e20dd2e6": "CMB",
}


@dataclass(frozen=True)
class StudyItem:
    patient_id: str
    study_date_raw: str
    study_date_key: str
    inference_id: str
    inference_timestamp: str
    input_study_instance_uid: str
    input_series_instance_uid: str
    case_dir: Path
    case_id: str
    pred_file: Path
    synthseg_file: Path
    pred_json_file: Path


@dataclass(frozen=True)
class Comparison:
    current: StudyItem
    prior: StudyItem


def pipeline_followup_v3(
    *,
    input_json: Path,
    path_process: Path,
    model: str = "Aneurysm",
    prediction_json_name: Optional[str] = None,
    path_followup_root: Optional[Path] = None,
    fsl_flirt_path: str = "/usr/local/fsl/bin/flirt",
) -> Path:
    start_time = time.time()
    LOGGER.info("Load input json: %s", input_json)
    payload = _read_json(input_json)
    current_study = payload.get("current_study") or {}
    prior_list = payload.get("prior_study_list") or []

    if not isinstance(prior_list, list):
        raise ValueError("prior_study_list 必須為 list")

    input_model_id = str(current_study.get("model_id") or "").strip()
    resolved_model = _resolve_model_from_model_id(input_model_id)
    for st in prior_list:
        if not isinstance(st, dict):
            continue
        prior_model_id = str(st.get("model_id") or "").strip()
        if not prior_model_id:
            continue
        prior_model = _resolve_model_from_model_id(prior_model_id)
        if prior_model != resolved_model:
            raise ValueError(
                "prior_study_list model_id 對應的 model 不一致："
                f"{prior_model_id} -> {prior_model} (expected {resolved_model})"
            )
    if model and model.strip() != resolved_model:
        LOGGER.info("Override model=%s -> %s (by model_id)", model, resolved_model)
    model = resolved_model
    if not prediction_json_name:
        if model == "CMB":
            prediction_json_name = "Pred_CMB_rdx_cmb_pred_json.json"
        else:
            prediction_json_name = f"Pred_{model}_rdx_aneurysm_pred_json.json"

    LOGGER.info("Resolve current study -> case folder")
    current_item = _build_study_item(
        study=current_study,
        path_process=path_process,
        model=model,
        prediction_json_name=prediction_json_name,
    )

    LOGGER.info("Resolve prior studies -> case folders (count=%d)", len(prior_list))
    prior_items = [
        _build_study_item(
            study=st,
            path_process=path_process,
            model=model,
            prediction_json_name=prediction_json_name,
        )
        for st in prior_list
    ]

    comparisons = _build_comparisons(current_item, prior_items)
    if not comparisons:
        raise ValueError("找不到可比較的日期組合（current/prior 皆為同日或無資料）")

    process_root = _build_process_root(
        path_process,
        current_item.case_id,
        model=model,
        path_followup_root=path_followup_root,
    )
    comparisons_root = process_root / "comparisons"
    outputs_root = process_root / "outputs"
    LOGGER.info("Create process root: %s", process_root)
    comparisons_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)

    outputs: List[Dict[str, Any]] = []
    for current_case_id, comps in _group_by_current(comparisons).items():
        current_item = comps[0].current
        LOGGER.info("Load current prediction json: %s", current_item.pred_json_file)
        current_payload = _read_json(current_item.pred_json_file)
        _ensure_study_date(current_payload, current_item.study_date_raw)

        follow_up_entries = list(current_payload.get("follow_up") or [])
        for comp in comps:
            comp_dir = comparisons_root / f"{comp.current.case_id}__VS__{comp.prior.case_id}"
            LOGGER.info(
                "Run comparison: current=%s prior=%s",
                comp.current.case_id,
                comp.prior.case_id,
            )
            comp_outputs = _run_single_comparison(
                comp=comp,
                comp_dir=comp_dir,
                fsl_flirt_path=fsl_flirt_path,
                model=model,
            )

            follow_up_entries = _replace_followup_entry(
                follow_up_entries=follow_up_entries,
                entry=comp_outputs["follow_up_entry"],
            )

        current_payload["follow_up"] = follow_up_entries
        output_dir = outputs_root / current_case_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "prediction-grouped.json"
        LOGGER.info("Write prediction output: %s", output_path)
        output_path.write_text(json.dumps(current_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(current_payload)

    manifest = {
        "manifest": {
            "inference_id": str(current_item.inference_id),
            "inference_timestamp": str(current_item.inference_timestamp),
            "model_id": str(current_study.get("model_id") or ""),
            "patient_id": str(current_item.patient_id),
        },
        "items": outputs,
    }

    manifest_path = outputs_root / "followup_manifest.json"
    LOGGER.info("Write manifest: %s", manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    spend_sec = int(time.time() - start_time)
    logging.info(f"[Done Follow-up v3 Pipeline!!! ] spend {spend_sec} sec")
    logging.info(f"!!! {current_item.case_id} followup finish.")
    return manifest_path


def _build_comparisons(current: StudyItem, prior_items: List[StudyItem]) -> List[Comparison]:
    comparisons: List[Comparison] = []
    for item in prior_items:
        if item.study_date_key == current.study_date_key:
            continue
        if item.study_date_key < current.study_date_key:
            comparisons.append(Comparison(current=current, prior=item))
        else:
            comparisons.append(Comparison(current=item, prior=current))
    return comparisons


def _group_by_current(comparisons: List[Comparison]) -> Dict[str, List[Comparison]]:
    grouped: Dict[str, List[Comparison]] = defaultdict(list)
    for comp in comparisons:
        grouped[comp.current.case_id].append(comp)
    for key in list(grouped.keys()):
        grouped[key] = sorted(grouped[key], key=lambda c: c.prior.study_date_key)
    return grouped


def _run_single_comparison(
    *,
    comp: Comparison,
    comp_dir: Path,
    fsl_flirt_path: str,
    model: str,
) -> Dict[str, Any]:
    baseline_dir = comp_dir / "baseline"
    followup_dir = comp_dir / "followup"
    registration_dir = comp_dir / "registration"
    result_dir = comp_dir / "result"
    dicom_dir = comp_dir / "dicom"
    LOGGER.info("Prepare comparison folders: %s", comp_dir)
    _ensure_dirs([baseline_dir, followup_dir, registration_dir, result_dir, dicom_dir])

    baseline_pred = baseline_dir / f"Pred_{model}.nii.gz"
    baseline_synthseg = baseline_dir / f"SynthSEG_{model}.nii.gz"
    followup_pred = followup_dir / f"Pred_{model}.nii.gz"
    followup_synthseg = followup_dir / f"SynthSEG_{model}.nii.gz"
    baseline_json = baseline_dir / comp.current.pred_json_file.name
    followup_json = followup_dir / comp.prior.pred_json_file.name

    LOGGER.info("Copy current files -> baseline")
    _copy_file(comp.current.pred_file, baseline_pred)
    _copy_file(comp.current.synthseg_file, baseline_synthseg)
    _copy_file(comp.current.pred_json_file, baseline_json)
    LOGGER.info("Copy prior files -> followup")
    _copy_file(comp.prior.pred_file, followup_pred)
    _copy_file(comp.prior.synthseg_file, followup_synthseg)
    _copy_file(comp.prior.pred_json_file, followup_json)

    LOGGER.info("Overwrite pred geometry from synthseg")
    _overwrite_pred_geometry_from_synthseg(pred_file=baseline_pred, synthseg_file=baseline_synthseg)
    _overwrite_pred_geometry_from_synthseg(pred_file=followup_pred, synthseg_file=followup_synthseg)

    mat_file = registration_dir / f"{comp.current.case_id}__VS__{comp.prior.case_id}.mat"
    synthseg_reg = registration_dir / f"SynthSEG_{model}_registration.nii.gz"
    pred_reg = registration_dir / f"Pred_{model}_registration.nii.gz"

    LOGGER.info("Run registration (followup -> baseline)")
    run_registration_followup_to_baseline(
        baseline_pred_file=baseline_pred,
        baseline_synthseg_file=baseline_synthseg,
        followup_synthseg_file=followup_synthseg,
        followup_pred_file=followup_pred,
        out_dir=registration_dir,
        mat_file=mat_file,
        out_synthseg_reg_file=synthseg_reg,
        out_pred_reg_file=pred_reg,
        fsl_flirt_path=fsl_flirt_path,
    )

    merged_pred = result_dir / f"Pred_{model}_followup.nii.gz"
    LOGGER.info("Generate followup pred")
    generate_followup_pred(
        baseline_pred_file=baseline_pred,
        followup_reg_pred_file=pred_reg,
        output_file=merged_pred,
    )

    LOGGER.info("Build follow_up entry")
    follow_up_entry = _build_followup_entry(
        current_item=comp.current,
        prior_item=comp.prior,
        current_pred_file=baseline_pred,
        prior_pred_reg_file=pred_reg,
        mat_file=mat_file,
    )
    return {"follow_up_entry": follow_up_entry}


def _build_followup_entry(
    *,
    current_item: StudyItem,
    prior_item: StudyItem,
    current_pred_file: Path,
    prior_pred_reg_file: Path,
    mat_file: Path,
) -> Dict[str, Any]:
    current_pred = _load_label_volume(current_pred_file)
    prior_pred_reg = _load_label_volume(prior_pred_reg_file)
    if current_pred.shape != prior_pred_reg.shape:
        raise ValueError(
            "current 與 prior(reg) shape 不一致："
            f"{current_pred.shape} vs {prior_pred_reg.shape}"
        )

    mat = _load_fsl_mat(mat_file)
    matches, used_prior_labels = _match_by_overlap(current_pred, prior_pred_reg)

    current_payload = _read_json(current_item.pred_json_file)
    prior_payload = _read_json(prior_item.pred_json_file)
    current_det = _collect_detections_by_index(current_payload)
    prior_det = _collect_detections_by_index(prior_payload)

    stable_entries = []
    new_entries = []
    regress_entries = []

    for current_label, prior_label in matches.items():
        current_info = current_det.get(current_label, {})
        current_slice = _safe_int(current_info.get("main_seg_slice"))
        if prior_label is None:
            prior_slice = _map_slice_index(
                current_slice, mat=mat, direction="B2F"
            )
            new_entries.append(
                {
                    "current_mask_index": current_label,
                    "current_main_seg_slice": current_slice,
                    "prior_main_seg_slice": prior_slice,
                }
            )
            continue

        prior_info = prior_det.get(prior_label, {})
        prior_slice = _safe_int(prior_info.get("main_seg_slice"))
        stable_entries.append(
            {
                "current_mask_index": current_label,
                "current_main_seg_slice": current_slice,
                "prior_mask_index": prior_label,
                "prior_main_seg_slice": prior_slice,
            }
        )

    for prior_label in sorted(prior_det.keys()):
        if prior_label in used_prior_labels:
            continue
        prior_info = prior_det.get(prior_label, {})
        prior_slice = _safe_int(prior_info.get("main_seg_slice"))
        current_slice = _map_slice_index(
            prior_slice, mat=mat, direction="F2B"
        )
        regress_entries.append(
            {
                "prior_mask_index": prior_label,
                "current_main_seg_slice": current_slice,
                "prior_main_seg_slice": prior_slice,
            }
        )

    LOGGER.info(
        "Build sorted mapping: current_slices=%d prior_slices=%d",
        current_pred.shape[2],
        prior_pred_reg.shape[2],
    )
    sorted_mapping, reverse_sorted_mapping = _build_sorted_mapping_one_based(
        current_slices=current_pred.shape[2],
        prior_slices=prior_pred_reg.shape[2],
        mat=mat,
    )

    return {
        "current_study_date": current_item.study_date_key,
        "prior_study_date": prior_item.study_date_key,
        "follow_up_timestamp": current_item.inference_timestamp,
        "prior_study_instance_uid": prior_item.input_study_instance_uid,
        "current_series_instance_uid": _guess_series_uid(current_payload),
        "prior_series_instance_uid": _guess_series_uid(prior_payload),
        "status": {
            "new": new_entries,
            "stable": stable_entries,
            "regress": regress_entries,
        },
        "sorted": sorted_mapping,
        "reverse-sorted": reverse_sorted_mapping,
    }


def _build_sorted_mapping_one_based(
    *, current_slices: int, prior_slices: int, mat: np.ndarray
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if current_slices <= 0 or prior_slices <= 0:
        return [], []

    inv_mat = _safe_inv(mat)
    current_candidates: List[List[int]] = [[] for _ in range(current_slices)]
    for current_idx in range(current_slices):
        z_tgt = float(inv_mat[2, 2] * current_idx + inv_mat[2, 3])
        prior_idx = _clamp_index(int(np.rint(z_tgt)), prior_slices)
        current_candidates[current_idx].append(prior_idx)
    empty_current = sum(1 for cands in current_candidates if not cands)

    current_choice: List[Optional[int]] = [_select_candidate(cands) for cands in current_candidates]
    current_choice_filled = _fill_empty_choices(current_choice)
    sorted_list = [
        {"current_slice": idx + 1, "prior_slice": prior_idx + 1}
        for idx, prior_idx in enumerate(current_choice_filled)
    ]

    prior_candidates: List[List[int]] = [[] for _ in range(prior_slices)]
    for prior_idx in range(prior_slices):
        z_tgt = float(mat[2, 2] * prior_idx + mat[2, 3])
        current_idx = _clamp_index(int(np.rint(z_tgt)), current_slices)
        prior_candidates[prior_idx].append(current_idx)
    empty_prior = sum(1 for cands in prior_candidates if not cands)

    prior_choice: List[Optional[int]] = [_select_candidate(cands) for cands in prior_candidates]
    prior_choice_filled = _fill_empty_choices(prior_choice)
    reverse_sorted_list = [
        {"prior_slice": idx + 1, "current_slice": current_idx + 1}
        for idx, current_idx in enumerate(prior_choice_filled)
    ]

    LOGGER.info(
        "sorted fill stats: empty_current=%d empty_prior=%d",
        empty_current,
        empty_prior,
    )
    return sorted_list, reverse_sorted_list


def _clamp_index(value: int, size: int) -> int:
    if value < 0:
        return 0
    if value >= size:
        return size - 1
    return value


def _select_candidate(cands: List[int]) -> Optional[int]:
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    sorted_cands = sorted(cands)
    median_val = float(np.median(sorted_cands))
    chosen = min(sorted_cands, key=lambda v: (abs(v - median_val), v))
    if len(sorted_cands) > 2:
        LOGGER.info(
            "sorted median pick: candidates=%s median=%.2f chosen=%d",
            sorted_cands,
            median_val,
            chosen,
        )
    return chosen


def _fill_empty_choices(values: List[Optional[int]]) -> List[int]:
    if not values:
        return []
    if all(v is None for v in values):
        return [0 for _ in values]

    filled: List[Optional[int]] = list(values)
    first_idx = next(i for i, v in enumerate(filled) if v is not None)
    for i in range(0, first_idx):
        filled[i] = filled[first_idx]

    last_idx = max(i for i, v in enumerate(filled) if v is not None)
    for i in range(last_idx + 1, len(filled)):
        filled[i] = filled[last_idx]

    prev_idx: List[int] = [-1] * len(filled)
    last = -1
    for i, val in enumerate(filled):
        if val is not None:
            last = i
        prev_idx[i] = last

    next_idx: List[int] = [-1] * len(filled)
    nxt = -1
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is not None:
            nxt = i
        next_idx[i] = nxt

    for i, val in enumerate(filled):
        if val is not None:
            continue
        prev_i = prev_idx[i]
        next_i = next_idx[i]
        if prev_i == -1 and next_i == -1:
            filled[i] = 0
            continue
        if prev_i == -1:
            filled[i] = filled[next_i]
            continue
        if next_i == -1:
            filled[i] = filled[prev_i]
            continue
        left_val = filled[prev_i]
        right_val = filled[next_i]
        if left_val is None or right_val is None:
            filled[i] = left_val if left_val is not None else right_val
            continue
        left_dist = i - prev_i
        right_dist = next_i - i
        filled[i] = left_val if left_dist <= right_dist else right_val

    return [int(v) if v is not None else 0 for v in filled]


def _map_slice_index(value: int, *, mat: np.ndarray, direction: str) -> int:
    if value < 0:
        return 0
    if direction not in {"F2B", "B2F"}:
        return value
    m = mat if direction == "F2B" else _safe_inv(mat)
    z_tgt = float(m[2, 2] * value + m[2, 3])
    return int(np.rint(z_tgt))


def _match_by_overlap(
    current_pred: np.ndarray, prior_pred_reg: np.ndarray
) -> Tuple[Dict[int, Optional[int]], set[int]]:
    current_labels = [int(x) for x in np.unique(current_pred) if int(x) > 0]
    mapping: Dict[int, Optional[int]] = {}
    used_prior: set[int] = set()
    for label in current_labels:
        mask = current_pred == label
        overlapped = prior_pred_reg[mask]
        overlapped = overlapped[overlapped > 0]
        if overlapped.size == 0:
            mapping[label] = None
            continue
        counts = np.bincount(overlapped.astype(np.int64))
        best = int(np.argmax(counts))
        mapping[label] = best
        used_prior.add(best)
    return mapping, used_prior


def _collect_detections_by_index(payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    detections = payload.get("detections") or []
    if not isinstance(detections, list):
        return mapping
    for det in detections:
        if not isinstance(det, dict):
            continue
        idx = det.get("mask_index")
        try:
            key = int(idx)
        except (TypeError, ValueError):
            continue
        mapping[key] = det
    return mapping


def _guess_series_uid(payload: Dict[str, Any]) -> str:
    dets = payload.get("detections") or []
    if isinstance(dets, list) and dets:
        first = dets[0]
        if isinstance(first, dict):
            return str(first.get("series_instance_uid", "") or "")
    series_list = payload.get("input_series_instance_uid") or []
    if isinstance(series_list, list) and series_list:
        return str(series_list[0] or "")
    return ""


def _replace_followup_entry(
    *, follow_up_entries: List[Dict[str, Any]], entry: Dict[str, Any]
) -> List[Dict[str, Any]]:
    prior_date = entry.get("prior_study_date")
    current_date = entry.get("current_study_date")
    out = []
    for item in follow_up_entries:
        if not isinstance(item, dict):
            continue
        if item.get("prior_study_date") == prior_date and item.get("current_study_date") == current_date:
            continue
        out.append(item)
    out.append(entry)
    return out


def _build_study_item(
    *,
    study: Dict[str, Any],
    path_process: Path,
    model: str,
    prediction_json_name: str,
) -> StudyItem:
    patient_id = str(study.get("patient_id") or "").strip()
    study_date_raw = str(study.get("study_date") or "").strip()
    if not patient_id or not study_date_raw:
        raise ValueError(f"study 缺少 patient_id / study_date: {study}")

    study_date_key = _normalize_date_key(study_date_raw)
    case_dir = _find_case_dir(path_process, patient_id, study_date_key)
    case_id = case_dir.name
    pred_file = case_dir / f"Pred_{model}.nii.gz"
    synthseg_file = case_dir / _default_synthseg_filename(model)
    pred_json_file = case_dir / prediction_json_name

    _ensure_file(pred_file, "Pred NIfTI")
    _ensure_file(synthseg_file, "SynthSEG NIfTI")
    _ensure_file(pred_json_file, "prediction json")

    return StudyItem(
        patient_id=patient_id,
        study_date_raw=study_date_raw,
        study_date_key=study_date_key,
        inference_id=str(study.get("inference_id") or ""),
        inference_timestamp=str(study.get("inference_timestamp") or ""),
        input_study_instance_uid=_first_uid(study.get("input_study_instance_uid")),
        input_series_instance_uid=_first_uid(study.get("input_series_instance_uid")),
        case_dir=case_dir,
        case_id=case_id,
        pred_file=pred_file,
        synthseg_file=synthseg_file,
        pred_json_file=pred_json_file,
    )


def _default_synthseg_filename(model: str) -> str:
    model_key = model.strip().lower()
    if model_key == "cmb":
        return "synthseg_SWAN_original_CMB_from_T1BRAVO_AXI_original_CMB.nii.gz"
    return f"SynthSEG_{model}.nii.gz"


def _build_process_root(
    path_process: Path,
    current_case_id: str,
    *,
    model: str,
    path_followup_root: Optional[Path],
) -> Path:
    if path_followup_root:
        root = Path(path_followup_root)
    else:
        base = Path(path_process)
        root = base.parent if base.name in {"Deep_Radax", "Deep_Aneurysm"} else base
    model_key = model.strip().lower()
    if model_key == "aneurysm":
        model_dir = "Deep_Aneurysm"
    elif model_key == "cmb":
        model_dir = "Deep_CMB"
    else:
        model_dir = f"Deep_{model}"
    return root / "Deep_FollowUp" / current_case_id / model_dir


def _find_case_dir(path_process: Path, patient_id: str, study_date_key: str) -> Path:
    prefix = f"{patient_id}_{study_date_key}"
    candidates = [p for p in path_process.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"找不到 case 資料夾：{prefix}_MR_* in {path_process}")
    if len(candidates) > 1:
        raise ValueError(f"同日找到多個 case 資料夾：{[p.name for p in candidates]}")
    return candidates[0]


def _normalize_date_key(date_str: str) -> str:
    clean = date_str.strip()
    if "-" in clean:
        parts = clean.split("-")
        if len(parts) == 3:
            return f"{parts[0]}{parts[1].rjust(2,'0')}{parts[2].rjust(2,'0')}"
    if len(clean) == 8 and clean.isdigit():
        return clean
    raise ValueError(f"無法解析 study_date：{date_str}")


def _resolve_model_from_model_id(model_id: str) -> str:
    if not model_id:
        raise ValueError("current_study.model_id 必填")
    model = MODEL_BY_MODEL_ID.get(model_id)
    if not model:
        raise ValueError(f"不支援的 model_id（無法判定 model）：{model_id}")
    return model


def _ensure_study_date(payload: Dict[str, Any], study_date_raw: str) -> None:
    if "study_date" not in payload:
        payload["study_date"] = study_date_raw


def _safe_inv(mat: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.eye(4, dtype=np.float64)


def _safe_int(value: Any) -> int:
    if value in ("", None):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 JSON：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _first_uid(val: Any) -> str:
    if isinstance(val, list) and val:
        return str(val[0] or "")
    return str(val or "")


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"找不到檔案：{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    dst.write_bytes(src.read_bytes())


def _ensure_file(path: Path, name: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(f"缺少檔案 ({name}): {path}")


def _load_label_volume(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到 NIfTI：{path}")
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"NIfTI 維度必須為 3D：{path} (ndim={data.ndim})")
    return data.astype(np.int32)


def _load_fsl_mat(path: Path) -> np.ndarray:
    if not path.exists():
        return np.eye(4, dtype=np.float64)
    mat = np.loadtxt(str(path), dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"registration mat 必須為 4x4：{path} (shape={mat.shape})")
    return mat


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Follow-up v3 pipeline (NIfTI + multi-date)")
    p.add_argument(
        "--input_json",
        default="/home/david/pipeline/chuan/example_input/follow_up_input.json",
    )
    p.add_argument(
        "--path_process",
        default="/home/david/pipeline/sean/rename_nifti/",
        help="case_id 根目錄（每個 case 資料夾內放 Pred/SynthSEG/JSON）",
    )
    p.add_argument(
        "--path_followup_root",
        default="/home/david/pipeline/chuan/process",
        help="Follow-up 輸出根目錄（會建立 Deep_FollowUp/<case_id>/...）",
    )
    p.add_argument("--model", default="Aneurysm")
    p.add_argument(
        "--prediction_json_name",
        default="",
        help="prediction JSON 檔名（不填則依 inference_id 自動推導）",
    )
    p.add_argument("--fsl_flirt_path", default="/usr/local/fsl/bin/flirt")
    p.add_argument("--path_log", default="/home/david/pipeline/chuan/log")
    return p


def main(argv: List[str]) -> int:
    args = _build_parser().parse_args(argv)

    if args.path_log:
        Path(args.path_log).mkdir(parents=True, exist_ok=True)
        localt = time.localtime(time.time())
        time_str_short = (
            str(localt.tm_year)
            + str(localt.tm_mon).rjust(2, "0")
            + str(localt.tm_mday).rjust(2, "0")
        )
        log_file = os.path.join(args.path_log, time_str_short + ".log")
        if not os.path.isfile(log_file):
            with open(log_file, "a+", encoding="utf-8") as f:
                f.write("")
        logging.basicConfig(
            level=logging.INFO,
            filename=log_file,
            filemode="a",
            format="%(asctime)s %(levelname)s %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    manifest = pipeline_followup_v3(
        input_json=Path(args.input_json),
        path_process=Path(args.path_process),
        model=args.model,
        prediction_json_name=args.prediction_json_name or None,
        path_followup_root=Path(args.path_followup_root) if args.path_followup_root else None,
        fsl_flirt_path=args.fsl_flirt_path,
    )
    print(f"[Done] manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
