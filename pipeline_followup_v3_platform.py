#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Follow-up v3 pipeline (Platform JSON + NIfTI).

輸入：
- 00130846_followup_input.json（ids + needFollowup）
- case 資料夾內的 Pred/SynthSEG NIfTI + Pred_<Model>_platform_json.json

輸出：
- Deep_FollowUp/<current_case_id>/result/Followup_<Model>_platform_json_new.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np

from generate_followup_pred import generate_followup_pred
from pipeline_followup import _overwrite_pred_geometry_from_synthseg
from registration import run_registration_followup_to_baseline


LOGGER = logging.getLogger(__name__)

MODEL_NAME_BY_TYPE = {
    1: "Aneurysm",
    2: "CMB",
    3: "Infarct",
    4: "WMH",
}

MODEL_TYPE_BY_NAME = {v.lower(): k for k, v in MODEL_NAME_BY_TYPE.items()}


@dataclass(frozen=True)
class StudyItem:
    patient_id: str
    study_date_raw: str
    study_date_key: str
    study_instance_uid: str
    model_types: List[int]
    case_dir: Path
    case_id: str


@dataclass(frozen=True)
class ModelStudyItem:
    base: StudyItem
    model_type: int
    model_name: str
    pred_file: Path
    synthseg_file: Path
    platform_json_file: Path


@dataclass(frozen=True)
class Comparison:
    current: ModelStudyItem
    prior: ModelStudyItem


@dataclass(frozen=True)
class SortedSliceItem:
    sop_instance_uid: str
    projection: float


def pipeline_followup_v3_platform(
    *,
    input_json: Path,
    path_process: Path,
    model: Optional[str] = None,
    model_type: Optional[int] = None,
    case_id: Optional[str] = None,
    platform_json_name: Optional[str] = None,
    path_followup_root: Optional[Path] = None,
    fsl_flirt_path: str = "/usr/local/fsl/bin/flirt",
) -> Tuple[List[Path], bool]:
    LOGGER.info("Load input json: %s", input_json)
    print(f"[FollowUp] Load input json: {input_json}")
    payload = _read_json(input_json)
    latest_date_key = _get_latest_date_key(payload)
    case_id_date_key = _case_id_to_date_key(case_id)
    LOGGER.info("Resolve studies (latest_date_key=%s, case_id=%s)", latest_date_key, case_id or "")
    print(f"[FollowUp] Resolve studies (latest_date_key={latest_date_key}, case_id={case_id or ''})")
    studies = _build_studies(
        payload,
        path_process,
        latest_date_key=latest_date_key,
        current_case_id=case_id,
        current_case_date_key=case_id_date_key,
    )
    if case_id and not any(item.case_id == case_id for item in studies):
        LOGGER.info("Inject case_id study for comparison: %s", case_id)
        print(f"[FollowUp] Inject case_id study for comparison: {case_id}")
        studies.append(_build_case_id_study(path_process, case_id, model, model_type))
    if not studies:
        raise ValueError("needFollowup 為空，無可比對的 study")

    if not case_id_date_key:
        raise ValueError("無法解析 case_id 的日期，請提供正確 case_id")

    if case_id and not any(item.case_id == case_id for item in studies):
        LOGGER.info("Inject case_id study for comparison: %s", case_id)
        print(f"[FollowUp] Inject case_id study for comparison: {case_id}")
        studies.append(_build_case_id_study(path_process, case_id, model, model_type))
    if not studies:
        raise ValueError("needFollowup 為空，無可比對的 study")

    input_study = next((item for item in studies if item.case_id == case_id), None)
    if not input_study:
        raise ValueError("找不到輸入 case_id 對應的 study")
    model_type_list = _resolve_model_types(studies, model=model, model_type=model_type)
    if not model_type_list:
        LOGGER.warning(
            "指定模型未出現在 needFollowup.models，跳過執行：model=%s model_type=%s",
            model or "",
            model_type or "",
        )
        print(
            f"[FollowUp] Skip: model not in needFollowup.models (model={model or ''} model_type={model_type or ''})"
        )
        return [], False

    LOGGER.info("Resolved model_type_list=%s", model_type_list)
    print(f"[FollowUp] Resolved model_type_list={model_type_list}")

    outputs: List[Path] = []
    targets: List[Tuple[StudyItem, List[StudyItem]]] = []
    older = [item for item in studies if item.study_date_key < case_id_date_key]
    newer = [item for item in studies if item.study_date_key > case_id_date_key]
    targets = [(input_study, older)]
    for item in newer:
        targets.append((item, [input_study]))

    for mt in model_type_list:
        model_name = _model_name_from_type(mt)
        for current_base, prior_bases in targets:
            LOGGER.info("Build model item: model_type=%s model_name=%s", mt, model_name)
            print(f"[FollowUp] Build model item: model_type={mt} model_name={model_name}")
            current = _build_model_item(current_base, mt, model_name, path_process, platform_json_name)
            prior_items = [
                _build_model_item(item, mt, model_name, path_process, platform_json_name)
                for item in prior_bases
                if mt in item.model_types and item.case_id != current_base.case_id
            ]
            LOGGER.info("Prior items count=%d", len(prior_items))
            print(f"[FollowUp] Prior items count={len(prior_items)}")
            output = _run_model_followup(
                current=current,
                priors=prior_items,
                path_followup_root=path_followup_root,
                fsl_flirt_path=fsl_flirt_path,
                output_date_key=current_base.study_date_key,
                output_root_case_id=input_study.case_id,
            )
            outputs.append(output)
    return outputs, True


def _run_model_followup(
    *,
    current: ModelStudyItem,
    priors: List[ModelStudyItem],
    path_followup_root: Optional[Path],
    fsl_flirt_path: str,
    output_date_key: str,
    output_root_case_id: str,
) -> Path:
    process_root = _build_process_root(output_root_case_id, path_followup_root)
    comparisons_root = process_root / "comparisons" / current.model_name
    result_root = process_root / "result"
    LOGGER.info("Process root: %s", process_root)
    LOGGER.info("Comparisons root: %s", comparisons_root)
    LOGGER.info("Result root: %s", result_root)
    print(f"[FollowUp] Process root: {process_root}")
    comparisons_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Load current platform json: %s", current.platform_json_file)
    print(f"[FollowUp] Load current json: {current.platform_json_file}")
    current_payload = _read_json(current.platform_json_file)
    current_model, current_mask_model = _find_models_by_type(current_payload, current.model_type)
    if current_model is None or current_mask_model is None:
        raise ValueError(f"找不到 current model_type={current.model_type} 的資料")

    _reset_followup(current_model, current_mask_model)
    _ensure_sorted_slice_list(current_payload, current.model_type)

    current_series_uid, current_series_type = _get_series_info(current_payload, current.model_type)

    sorted_slice_entries: List[Dict[str, Any]] = []
    priors_sorted = sorted(priors, key=lambda item: item.base.study_date_key, reverse=True)
    for prior in priors_sorted:
        comp_dir = comparisons_root / f"{current.base.case_id}__VS__{prior.base.case_id}"
        LOGGER.info("Run comparison: current=%s prior=%s", current.base.case_id, prior.base.case_id)
        print(f"[FollowUp] Run comparison: current={current.base.case_id} prior={prior.base.case_id}")
        comp_dir.mkdir(parents=True, exist_ok=True)
        comp_result = _run_single_comparison(
            comp=Comparison(current=current, prior=prior),
            comp_dir=comp_dir,
            fsl_flirt_path=fsl_flirt_path,
        )

        LOGGER.info("Load prior platform json: %s", prior.platform_json_file)
        print(f"[FollowUp] Load prior json: {prior.platform_json_file}")
        prior_payload = _read_json(prior.platform_json_file)
        _, prior_mask_model = _find_models_by_type(prior_payload, prior.model_type)
        if prior_mask_model is None:
            continue

        prior_series_uid, prior_series_type = _get_series_info(prior_payload, prior.model_type)
        _append_study_followup(
            current_model=current_model,
            current_series_uid=current_series_uid,
            prior_item=prior,
            prior_series_uid=prior_series_uid,
            prior_series_type=prior_series_type,
        )

        _apply_mask_followup(
            current_payload=current_payload,
            current_mask_model=current_mask_model,
            prior_item=prior,
            prior_payload=prior_payload,
            comp=comp_result,
        )

        sorted_slice_entries.append(
            _build_sorted_slice_entry(
                model_type=current.model_type,
                current_payload=current_payload,
                prior_payload=prior_payload,
                sorted_mapping=comp_result["sorted_mapping"],
                reverse_sorted_mapping=comp_result["reverse_sorted_mapping"],
            )
        )

    _replace_sorted_slice(current_payload, current.model_type, sorted_slice_entries)

    output_path = result_root / f"Followup_{current.model_name}_platform_json_new_{output_date_key}.json"
    LOGGER.info("Write output json: %s", output_path)
    print(f"[FollowUp] Write output json: {output_path}")
    output_path.write_text(json.dumps(current_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _run_single_comparison(
    *,
    comp: Comparison,
    comp_dir: Path,
    fsl_flirt_path: str,
) -> Dict[str, Any]:
    baseline_dir = comp_dir / "baseline"
    followup_dir = comp_dir / "followup"
    registration_dir = comp_dir / "registration"
    result_dir = comp_dir / "result"
    LOGGER.info("Prepare comparison folders: %s", comp_dir)
    print(f"[FollowUp] Prepare comparison folders: {comp_dir}")
    _ensure_dirs([baseline_dir, followup_dir, registration_dir, result_dir])

    baseline_pred = baseline_dir / f"Pred_{comp.current.model_name}.nii.gz"
    baseline_synthseg = baseline_dir / f"SynthSEG_{comp.current.model_name}.nii.gz"
    followup_pred = followup_dir / f"Pred_{comp.prior.model_name}.nii.gz"
    followup_synthseg = followup_dir / f"SynthSEG_{comp.prior.model_name}.nii.gz"

    LOGGER.info("Copy current files to baseline")
    print("[FollowUp] Copy current files to baseline")
    _copy_file(comp.current.pred_file, baseline_pred)
    _copy_file(comp.current.synthseg_file, baseline_synthseg)
    LOGGER.info("Copy prior files to followup")
    print("[FollowUp] Copy prior files to followup")
    _copy_file(comp.prior.pred_file, followup_pred)
    _copy_file(comp.prior.synthseg_file, followup_synthseg)

    LOGGER.info("Overwrite pred geometry from synthseg")
    print("[FollowUp] Overwrite pred geometry from synthseg")
    _overwrite_pred_geometry_from_synthseg(pred_file=baseline_pred, synthseg_file=baseline_synthseg)
    _overwrite_pred_geometry_from_synthseg(pred_file=followup_pred, synthseg_file=followup_synthseg)

    mat_file = registration_dir / f"{comp.current.base.case_id}__VS__{comp.prior.base.case_id}.mat"
    synthseg_reg = registration_dir / f"SynthSEG_{comp.current.model_name}_registration.nii.gz"
    pred_reg = registration_dir / f"Pred_{comp.current.model_name}_registration.nii.gz"

    LOGGER.info("Run registration (followup -> baseline)")
    print("[FollowUp] Run registration (followup -> baseline)")
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

    merged_pred = result_dir / f"Pred_{comp.current.model_name}_followup.nii.gz"
    LOGGER.info("Generate followup pred")
    print("[FollowUp] Generate followup pred")
    generate_followup_pred(
        baseline_pred_file=baseline_pred,
        followup_reg_pred_file=pred_reg,
        output_file=merged_pred,
    )

    LOGGER.info("Load label volumes")
    print("[FollowUp] Load label volumes")
    current_pred = _load_label_volume(baseline_pred)
    prior_pred_reg = _load_label_volume(pred_reg)
    if current_pred.shape != prior_pred_reg.shape:
        raise ValueError("current 與 prior(reg) shape 不一致")

    mat = _load_fsl_mat(mat_file)
    matches, used_prior_labels = _match_by_overlap(current_pred, prior_pred_reg)
    LOGGER.info(
        "Build sorted_slice mapping: current_slices=%d prior_slices=%d",
        current_pred.shape[2],
        prior_pred_reg.shape[2],
    )
    sorted_mapping, reverse_sorted_mapping = _build_sorted_mapping_one_based(
        current_slices=current_pred.shape[2],
        prior_slices=prior_pred_reg.shape[2],
        mat=mat,
    )
    if sorted_mapping and reverse_sorted_mapping:
        LOGGER.info(
            "sorted_slice sample: current 1->prior %s, prior 1->current %s",
            sorted_mapping[0].get("prior_slice"),
            reverse_sorted_mapping[0].get("current_slice"),
        )
    return {
        "mat": mat,
        "matches": matches,
        "used_prior_labels": used_prior_labels,
        "sorted_mapping": sorted_mapping,
        "reverse_sorted_mapping": reverse_sorted_mapping,
        "current_pred": current_pred,
        "prior_pred_reg": prior_pred_reg,
    }


def _apply_mask_followup(
    *,
    current_payload: Dict[str, Any],
    current_mask_model: Dict[str, Any],
    prior_item: ModelStudyItem,
    prior_payload: Dict[str, Any],
    comp: Dict[str, Any],
) -> None:
    prior_mask_model = _find_mask_model(prior_payload, prior_item.model_type)
    if prior_mask_model is None:
        return
    prior_instances_map: Dict[int, Dict[str, Any]] = {}

    current_series_type = _pick_series_type(current_payload, prior_item.model_type)
    current_series = _select_series_by_type(current_mask_model, current_series_type)
    if current_series is None:
        current_series = _get_first_series(current_mask_model)
    if current_series is None:
        return

    current_instances = current_series.get("instances")
    if not isinstance(current_instances, list):
        return

    current_series_uid = _safe_str(current_series.get("series_instance_uid")) if current_series else ""
    current_sorted = _get_sorted_slices(current_payload, current_series_uid)

    prior_series = _select_series_by_type(prior_mask_model, current_series_type)
    if prior_series is None:
        prior_series = _get_first_series(prior_mask_model)
    if isinstance(prior_series, dict):
        prior_series_uid = _safe_str(prior_series.get("series_instance_uid"))
    else:
        prior_series_uid = _get_series_uid_by_type(prior_payload, current_series_type)
    prior_instances_map = _collect_instances_by_index_for_series(prior_mask_model, prior_series_uid)
    prior_sorted = _get_sorted_slices(prior_payload, prior_series_uid)

    for inst in current_instances:
        if not isinstance(inst, dict):
            continue
        followup_list = inst.get("followup")
        if not isinstance(followup_list, list):
            followup_list = []
            inst["followup"] = followup_list

        current_label = _safe_int(inst.get("mask_index"))
        prior_label = comp["matches"].get(current_label)
        prior_inst = prior_instances_map.get(prior_label) if prior_label is not None else None
        entry = _build_followup_entry(
            current_inst=inst,
            prior_inst=prior_inst,
            prior_item=prior_item,
            prior_series_uid=prior_series_uid,
            status="stable" if prior_inst else "new",
            mat=comp["mat"],
            current_sorted=current_sorted,
            prior_sorted=prior_sorted,
        )
        _append_unique_instance_followup(followup_list, entry)

    _append_regress_instances(
        current_series=current_series,
        prior_item=prior_item,
        prior_series_uid=prior_series_uid,
        prior_instances=prior_instances_map,
        used_prior_labels=comp["used_prior_labels"],
        mat=comp["mat"],
        current_sorted=current_sorted,
        prior_sorted=prior_sorted,
    )


def _append_regress_instances(
    *,
    current_series: Dict[str, Any],
    prior_item: ModelStudyItem,
    prior_series_uid: str,
    prior_instances: Dict[int, Dict[str, Any]],
    used_prior_labels: set[int],
    mat: np.ndarray,
    current_sorted: List[SortedSliceItem],
    prior_sorted: List[SortedSliceItem],
) -> None:
    instances = current_series.get("instances")
    if not isinstance(instances, list):
        return

    for prior_label, prior_inst in prior_instances.items():
        if prior_label in used_prior_labels:
            continue
        if _has_regress_placeholder(instances, prior_item, prior_inst):
            continue
        placeholder = _build_regress_placeholder(
            prior_inst=prior_inst,
            prior_item=prior_item,
            prior_series_uid=prior_series_uid,
            mat=mat,
            current_sorted=current_sorted,
            prior_sorted=prior_sorted,
        )
        instances.append(placeholder)


def _has_regress_placeholder(
    instances: List[Dict[str, Any]],
    prior_item: ModelStudyItem,
    prior_inst: Dict[str, Any],
) -> bool:
    prior_mask_index = _safe_int(prior_inst.get("mask_index"))
    prior_study_uid = prior_item.base.study_instance_uid
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        followup_list = inst.get("followup")
        if not isinstance(followup_list, list):
            continue
        for entry in followup_list:
            if not isinstance(entry, dict):
                continue
            if _safe_str(entry.get("status")) != "regress":
                continue
            if _safe_str(entry.get("jump_study_instance_uid")) != prior_study_uid:
                continue
            if _safe_int(entry.get("mask_index")) != prior_mask_index:
                continue
            return True
    return False


def _build_followup_entry(
    *,
    current_inst: Dict[str, Any],
    prior_inst: Optional[Dict[str, Any]],
    prior_item: ModelStudyItem,
    prior_series_uid: str,
    status: str,
    mat: np.ndarray,
    current_sorted: List[SortedSliceItem],
    prior_sorted: List[SortedSliceItem],
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {}
    entry["status"] = status
    entry["study_date"] = prior_item.base.study_date_raw
    entry["mask_index"] = _safe_int(prior_inst.get("mask_index")) if prior_inst else 0
    entry["old_diameter"] = _safe_str(prior_inst.get("diameter")) if prior_inst else ""
    entry["old_volume"] = _safe_str(prior_inst.get("volume")) if prior_inst else ""
    entry["old_prob_max"] = _safe_str(prior_inst.get("prob_max")) if prior_inst else ""
    entry["jump_study_instance_uid"] = prior_item.base.study_instance_uid
    entry["jump_series_instance_uid"] = _safe_str(prior_series_uid) if prior_inst else ""
    entry["jump_sop_instance_uid"] = _safe_str(prior_inst.get("dicom_sop_instance_uid")) if prior_inst else ""
    entry["seg_series_instance_uid"] = _safe_str(prior_inst.get("seg_series_instance_uid")) if prior_inst else ""
    entry["seg_sop_instance_uid"] = _safe_str(prior_inst.get("seg_sop_instance_uid")) if prior_inst else ""
    entry["is_ai"] = _safe_str(current_inst.get("is_ai") or prior_inst.get("is_ai") if prior_inst else "1")
    if prior_inst:
        entry["main_seg_slice"] = _safe_int(prior_inst.get("main_seg_slice"))
    else:
        mapped_slice = _map_slice_index(_safe_int(current_inst.get("main_seg_slice")), mat=mat, direction="B2F")
        entry["main_seg_slice"] = mapped_slice

    if status == "new" and current_sorted and prior_sorted:
        mapped = _map_sop_uid_by_projection(
            source_uid=_safe_str(current_inst.get("dicom_sop_instance_uid")),
            source_sorted=current_sorted,
            target_sorted=prior_sorted,
            mat=mat,
            direction="B2F",
        )
        entry["jump_sop_instance_uid"] = mapped or entry["jump_sop_instance_uid"]
    return entry


def _build_regress_placeholder(
    *,
    prior_inst: Dict[str, Any],
    prior_item: ModelStudyItem,
    prior_series_uid: str,
    mat: np.ndarray,
    current_sorted: List[SortedSliceItem],
    prior_sorted: List[SortedSliceItem],
) -> Dict[str, Any]:
    placeholder: Dict[str, Any] = {
        "mask_index": "",
        "mask_name": "",
        "diameter": "",
        "type": "",
        "location": "",
        "sub_location": "",
        "prob_max": "",
        "checked": "",
        "is_ai": "",
        "seg_series_instance_uid": "",
        "seg_sop_instance_uid": "",
        "is_main_seg": "",
    }

    mapped_slice = _map_slice_index(
        _safe_int(prior_inst.get("main_seg_slice")),
        mat=mat,
        direction="F2B",
    )
    placeholder["main_seg_slice"] = mapped_slice
    mapped_uid = _sop_uid_by_slice_index(current_sorted, mapped_slice)
    if mapped_uid:
        placeholder["dicom_sop_instance_uid"] = mapped_uid
    else:
        LOGGER.info(
            "Regress fallback: no current slice uid (mapped_slice=%s), use projection mapping.",
            mapped_slice,
        )
        print(
            f"[FollowUp] Regress fallback: no current slice uid (mapped_slice={mapped_slice}), use projection mapping."
        )
        placeholder["dicom_sop_instance_uid"] = _map_sop_uid_by_projection(
            source_uid=_safe_str(prior_inst.get("dicom_sop_instance_uid")),
            source_sorted=prior_sorted,
            target_sorted=current_sorted,
            mat=mat,
            direction="F2B",
        )

    placeholder["followup"] = [
        {
            "mask_index": _safe_int(prior_inst.get("mask_index")),
            "old_diameter": _safe_str(prior_inst.get("diameter")),
            "old_volume": _safe_str(prior_inst.get("volume")),
            "old_prob_max": _safe_str(prior_inst.get("prob_max")),
            "status": "regress",
            "study_date": prior_item.base.study_date_raw,
            "main_seg_slice": _safe_int(prior_inst.get("main_seg_slice")),
            "jump_study_instance_uid": prior_item.base.study_instance_uid,
            "jump_series_instance_uid": _safe_str(prior_series_uid),
            "jump_sop_instance_uid": _safe_str(prior_inst.get("dicom_sop_instance_uid")),
            "seg_series_instance_uid": _safe_str(prior_inst.get("seg_series_instance_uid")),
            "seg_sop_instance_uid": _safe_str(prior_inst.get("seg_sop_instance_uid")),
            "is_ai": _safe_str(prior_inst.get("is_ai", "1")),
        }
    ]
    return placeholder


def _build_sorted_slice_entry(
    *,
    model_type: int,
    current_payload: Dict[str, Any],
    prior_payload: Dict[str, Any],
    sorted_mapping: List[Dict[str, Any]],
    reverse_sorted_mapping: List[Dict[str, Any]],
) -> Dict[str, Any]:
    current_series_uid, _ = _get_series_info(current_payload, model_type)
    prior_series_uid, _ = _get_series_info(prior_payload, model_type)
    return {
        "model_type": model_type,
        "current_study_instance_uid": _safe_str(_get_study_uid(current_payload)),
        "current_series_instance_uid": _safe_str(current_series_uid),
        "followup_study_instance_uid": _safe_str(_get_study_uid(prior_payload)),
        "followup_series_instance_uid": _safe_str(prior_series_uid),
        "sorted": sorted_mapping,
        "reverse-sorted": reverse_sorted_mapping,
    }


def _append_study_followup(
    *,
    current_model: Dict[str, Any],
    current_series_uid: str,
    prior_item: ModelStudyItem,
    prior_series_uid: str,
    prior_series_type: int,
) -> None:
    followup_list = current_model.get("followup")
    if not isinstance(followup_list, list):
        followup_list = []
        current_model["followup"] = followup_list

    entry = {
        "current_series_instance_uid": current_series_uid,
        "followup_study_date": prior_item.base.study_date_raw,
        "followup_study_instance_uid": prior_item.base.study_instance_uid,
        "followup_series_instance_uid": prior_series_uid,
        "followup_series_type": prior_series_type,
    }
    _append_unique_study_followup(followup_list, entry)


def _reset_followup(current_model: Dict[str, Any], current_mask_model: Dict[str, Any]) -> None:
    if not isinstance(current_model.get("followup"), list):
        current_model["followup"] = []
    series_list = current_mask_model.get("series") or []
    if not isinstance(series_list, list):
        return
    for series in series_list:
        if not isinstance(series, dict):
            continue
        instances = series.get("instances")
        if not isinstance(instances, list):
            continue
        for inst in instances:
            if isinstance(inst, dict):
                if not isinstance(inst.get("followup"), list):
                    inst["followup"] = []


def _ensure_sorted_slice_list(payload: Dict[str, Any], model_type: int) -> None:
    sorted_slice = payload.get("sorted_slice")
    if not isinstance(sorted_slice, list):
        payload["sorted_slice"] = []


def _replace_sorted_slice(payload: Dict[str, Any], model_type: int, entries: List[Dict[str, Any]]) -> None:
    existing = payload.get("sorted_slice")
    if not isinstance(existing, list):
        payload["sorted_slice"] = entries
        return
    payload["sorted_slice"] = _merge_sorted_slice(existing, entries)


def _append_unique_study_followup(followup_list: List[Dict[str, Any]], entry: Dict[str, Any]) -> None:
    if _has_study_followup(followup_list, entry):
        return
    followup_list.append(entry)


def _has_study_followup(followup_list: List[Dict[str, Any]], entry: Dict[str, Any]) -> bool:
    key = (
        _safe_str(entry.get("current_series_instance_uid")),
        _safe_str(entry.get("followup_study_instance_uid")),
        _safe_str(entry.get("followup_series_instance_uid")),
    )
    for item in followup_list:
        if not isinstance(item, dict):
            continue
        item_key = (
            _safe_str(item.get("current_series_instance_uid")),
            _safe_str(item.get("followup_study_instance_uid")),
            _safe_str(item.get("followup_series_instance_uid")),
        )
        if item_key == key:
            return True
    return False


def _append_unique_instance_followup(followup_list: List[Dict[str, Any]], entry: Dict[str, Any]) -> None:
    if _has_instance_followup(followup_list, entry):
        return
    followup_list.append(entry)


def _has_instance_followup(followup_list: List[Dict[str, Any]], entry: Dict[str, Any]) -> bool:
    key = (
        _safe_str(entry.get("status")),
        _safe_str(entry.get("study_date")),
        _safe_str(entry.get("jump_study_instance_uid")),
        _safe_str(entry.get("jump_series_instance_uid")),
        _safe_int(entry.get("mask_index")),
    )
    for item in followup_list:
        if not isinstance(item, dict):
            continue
        item_key = (
            _safe_str(item.get("status")),
            _safe_str(item.get("study_date")),
            _safe_str(item.get("jump_study_instance_uid")),
            _safe_str(item.get("jump_series_instance_uid")),
            _safe_int(item.get("mask_index")),
        )
        if item_key == key:
            return True
    return False


def _merge_sorted_slice(existing: List[Dict[str, Any]], entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = list(existing)
    existing_keys = {_sorted_slice_key(item) for item in existing if isinstance(item, dict)}
    for item in entries:
        if not isinstance(item, dict):
            continue
        key = _sorted_slice_key(item)
        if key in existing_keys:
            continue
        merged.append(item)
        existing_keys.add(key)
    return merged


def _sorted_slice_key(item: Dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        _safe_str(item.get("current_study_instance_uid")),
        _safe_str(item.get("current_series_instance_uid")),
        _safe_str(item.get("followup_study_instance_uid")),
        _safe_str(item.get("followup_series_instance_uid")),
        _safe_int(item.get("model_type")),
    )


def _build_studies(
    payload: Dict[str, Any],
    path_process: Path,
    *,
    latest_date_key: Optional[str],
    current_case_id: Optional[str],
    current_case_date_key: Optional[str],
) -> List[StudyItem]:
    need_followup = payload.get("needFollowup") or []
    if not isinstance(need_followup, list):
        return []

    items: List[StudyItem] = []
    for study in need_followup:
        if not isinstance(study, dict):
            continue
        patient_id = str(study.get("patient_id") or "").strip()
        study_date_raw = str(study.get("study_date") or "").strip()
        study_uid = str(study.get("studyInstanceUid") or "").strip()
        if not patient_id or not study_date_raw:
            continue
        study_date_key = _normalize_date_key(study_date_raw)
        case_dir = _resolve_case_dir(
            path_process=path_process,
            study=study,
            patient_id=patient_id,
            study_date_key=study_date_key,
            latest_date_key=latest_date_key,
            current_case_id=current_case_id,
            current_case_date_key=current_case_date_key,
        )
        case_id = case_dir.name
        model_types = _extract_model_types(study)
        items.append(
            StudyItem(
                patient_id=patient_id,
                study_date_raw=study_date_raw,
                study_date_key=study_date_key,
                study_instance_uid=study_uid,
                model_types=model_types,
                case_dir=case_dir,
                case_id=case_id,
            )
        )
    return items


def _get_latest_date_key(payload: Dict[str, Any]) -> Optional[str]:
    need_followup = payload.get("needFollowup") or []
    if not isinstance(need_followup, list):
        return None
    keys: List[str] = []
    for study in need_followup:
        if not isinstance(study, dict):
            continue
        raw = str(study.get("study_date") or "").strip()
        if not raw:
            continue
        try:
            keys.append(_normalize_date_key(raw))
        except ValueError:
            continue
    if not keys:
        return None
    return max(keys)


def _resolve_case_dir(
    *,
    path_process: Path,
    study: Dict[str, Any],
    patient_id: str,
    study_date_key: str,
    latest_date_key: Optional[str],
    current_case_id: Optional[str],
    current_case_date_key: Optional[str],
) -> Path:
    case_id = str(study.get("case_id") or "").strip()
    if case_id:
        return _ensure_case_dir(path_process, case_id)
    return _find_case_dir(path_process, patient_id, study_date_key)


def _case_id_to_date_key(case_id: Optional[str]) -> Optional[str]:
    if not case_id:
        return None
    parts = str(case_id).split("_")
    if len(parts) < 2:
        return None
    date_part = parts[1]
    if len(date_part) == 8 and date_part.isdigit():
        return date_part
    return None


def _build_case_id_study(
    path_process: Path,
    case_id: str,
    model: Optional[str],
    model_type: Optional[int],
) -> StudyItem:
    patient_id, study_date_key = _parse_case_id(case_id)
    case_dir = _ensure_case_dir(path_process, case_id)
    model_types = _infer_model_types(case_dir, model=model, model_type=model_type)
    if not model_types:
        raise ValueError(f"無法判定 model_type（case_id={case_id}）")
    return StudyItem(
        patient_id=patient_id,
        study_date_raw=_format_date_key(study_date_key),
        study_date_key=study_date_key,
        study_instance_uid="",
        model_types=model_types,
        case_dir=case_dir,
        case_id=case_id,
    )


def _parse_case_id(case_id: str) -> Tuple[str, str]:
    parts = case_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"case_id 格式錯誤：{case_id}")
    patient_id = parts[0]
    study_date_key = parts[1]
    if len(study_date_key) != 8 or not study_date_key.isdigit():
        raise ValueError(f"case_id study_date 錯誤：{case_id}")
    return patient_id, study_date_key


def _format_date_key(date_key: str) -> str:
    return f"{date_key[0:4]}-{date_key[4:6]}-{date_key[6:8]}"


def _infer_model_types(
    case_dir: Path,
    *,
    model: Optional[str],
    model_type: Optional[int],
) -> List[int]:
    if model_type:
        return [model_type]
    if model:
        mt = MODEL_TYPE_BY_NAME.get(model.strip().lower())
        return [mt] if mt else []
    types: List[int] = []
    for path in case_dir.glob("Pred_*_platform_json.json"):
        name = path.name.replace("Pred_", "").replace("_platform_json.json", "")
        mt = MODEL_TYPE_BY_NAME.get(name.strip().lower())
        if mt and mt not in types:
            types.append(mt)
    return types


def _build_case_id_study(
    path_process: Path,
    case_id: str,
    model: Optional[str],
    model_type: Optional[int],
) -> StudyItem:
    patient_id, study_date_key = _parse_case_id(case_id)
    case_dir = _ensure_case_dir(path_process, case_id)
    model_types = _infer_model_types(case_dir, model=model, model_type=model_type)
    if not model_types:
        raise ValueError(f"無法判定 model_type（case_id={case_id}）")
    return StudyItem(
        patient_id=patient_id,
        study_date_raw=_format_date_key(study_date_key),
        study_date_key=study_date_key,
        study_instance_uid="",
        model_types=model_types,
        case_dir=case_dir,
        case_id=case_id,
    )


def _parse_case_id(case_id: str) -> Tuple[str, str]:
    parts = case_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"case_id 格式錯誤：{case_id}")
    patient_id = parts[0]
    study_date_key = parts[1]
    if len(study_date_key) != 8 or not study_date_key.isdigit():
        raise ValueError(f"case_id study_date 錯誤：{case_id}")
    return patient_id, study_date_key


def _format_date_key(date_key: str) -> str:
    return f"{date_key[0:4]}-{date_key[4:6]}-{date_key[6:8]}"


def _infer_model_types(
    case_dir: Path,
    *,
    model: Optional[str],
    model_type: Optional[int],
) -> List[int]:
    if model_type:
        return [model_type]
    if model:
        mt = MODEL_TYPE_BY_NAME.get(model.strip().lower())
        return [mt] if mt else []
    types: List[int] = []
    for path in case_dir.glob("Pred_*_platform_json.json"):
        name = path.name.replace("Pred_", "").replace("_platform_json.json", "")
        mt = MODEL_TYPE_BY_NAME.get(name.strip().lower())
        if mt and mt not in types:
            types.append(mt)
    return types


def _ensure_case_dir(path_process: Path, case_id: str) -> Path:
    case_dir = path_process / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"找不到 case 資料夾：{case_dir}")
    if not case_dir.is_dir():
        raise NotADirectoryError(f"不是資料夾：{case_dir}")
    return case_dir


def _extract_model_types(study: Dict[str, Any]) -> List[int]:
    models = study.get("models") or []
    if not isinstance(models, list):
        return []
    out: List[int] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        mt = _safe_int(model.get("model_type"))
        if mt and mt not in out:
            out.append(mt)
    return out


def _select_current_study(
    items: List[StudyItem],
    *,
    case_id: Optional[str],
    case_date_key: Optional[str],
) -> StudyItem:
    case_item = None
    if case_id:
        for item in items:
            if item.case_id == case_id:
                case_item = item
                break

    candidates = [item for item in items if item.case_id != case_id] if case_id else items
    if not candidates and case_item:
        return case_item

    latest = max(candidates, key=lambda item: item.study_date_key)
    duplicates = [item for item in candidates if item.study_date_key == latest.study_date_key]
    if len(duplicates) > 1:
        raise ValueError("同日找到多個 current study，無法判定主體")

    if case_item and case_date_key and case_date_key >= latest.study_date_key:
        return case_item
    return latest


def _resolve_model_types(
    studies: List[StudyItem],
    *,
    model: Optional[str],
    model_type: Optional[int],
) -> List[int]:
    available: List[int] = []
    for item in studies:
        for mt in item.model_types:
            if mt not in available:
                available.append(mt)
    if model_type:
        if model_type not in available:
            return []
        return [model_type]
    if model:
        mt = MODEL_TYPE_BY_NAME.get(model.strip().lower())
        if not mt:
            raise ValueError(f"不支援的 model：{model}")
        if mt not in available:
            return []
        return [mt]
    return available


def _build_model_item(
    base: StudyItem,
    model_type: int,
    model_name: str,
    path_process: Path,
    platform_json_name: Optional[str],
) -> ModelStudyItem:
    pred_file = base.case_dir / f"Pred_{model_name}.nii.gz"
    synthseg_file = base.case_dir / _default_synthseg_filename(model_name)
    if platform_json_name:
        json_file = base.case_dir / platform_json_name
    else:
        json_file = base.case_dir / f"Pred_{model_name}_platform_json.json"
    _ensure_file(pred_file, "Pred NIfTI")
    _ensure_file(synthseg_file, "SynthSEG NIfTI")
    _ensure_file(json_file, "platform json")
    return ModelStudyItem(
        base=base,
        model_type=model_type,
        model_name=model_name,
        pred_file=pred_file,
        synthseg_file=synthseg_file,
        platform_json_file=json_file,
    )


def _default_synthseg_filename(model_name: str) -> str:
    model_key = model_name.strip().lower()
    if model_key == "cmb":
        return "synthseg_SWAN_original_CMB_from_T1BRAVO_AXI_original_CMB.nii.gz"
    return f"SynthSEG_{model_name}.nii.gz"


def _build_prior_items(
    studies: List[StudyItem],
    current_base: StudyItem,
    model_type: int,
    model_name: str,
    path_process: Path,
    platform_json_name: Optional[str],
) -> List[ModelStudyItem]:
    priors: List[ModelStudyItem] = []
    for item in studies:
        if item.case_id == current_base.case_id:
            continue
        if model_type not in item.model_types:
            continue
        priors.append(_build_model_item(item, model_type, model_name, path_process, platform_json_name))
    return priors


def _build_process_root(case_id: str, path_followup_root: Optional[Path]) -> Path:
    if path_followup_root:
        root = Path(path_followup_root)
    else:
        root = Path(".")
    base = root / "Deep_FollowUp" / case_id
    return base


def _find_case_dir(path_process: Path, patient_id: str, study_date_key: str) -> Path:
    prefix = f"{patient_id}_{study_date_key}"
    candidates = [p for p in path_process.iterdir() if p.is_dir()]
    matched: List[Path] = []
    for p in candidates:
        name_clean = p.name.replace("\r", "").replace("\n", "")
        if name_clean.startswith(prefix):
            matched.append(p)
    if not matched:
        raise FileNotFoundError(f"找不到 case 資料夾：{prefix}_MR_* in {path_process}")
    if len(matched) > 1:
        # 優先選擇名稱不含控制字元的資料夾
        clean_first = [p for p in matched if "\r" not in p.name and "\n" not in p.name]
        if clean_first:
            return clean_first[0]
        raise ValueError(f"同日找到多個 case 資料夾：{[p.name for p in matched]}")
    return matched[0]


def _find_models_by_type(payload: Dict[str, Any], model_type: int) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    study_model = None
    mask_model = _find_mask_model(payload, model_type)
    study = payload.get("study") or {}
    for item in study.get("model") or []:
        if _safe_int(item.get("model_type")) == model_type:
            study_model = item
            break
    return study_model, mask_model


def _find_mask_model(payload: Dict[str, Any], model_type: int) -> Optional[Dict[str, Any]]:
    mask = payload.get("mask") or {}
    for item in mask.get("model") or []:
        if _safe_int(item.get("model_type")) == model_type:
            return item
    return None


def _get_first_series(mask_model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    series_list = mask_model.get("series")
    if not isinstance(series_list, list) or not series_list:
        return None
    first = series_list[0]
    return first if isinstance(first, dict) else None


def _collect_instances_by_index(mask_model: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    series_list = mask_model.get("series") or []
    for series in series_list:
        if not isinstance(series, dict):
            continue
        instances = series.get("instances") or []
        if not isinstance(instances, list):
            continue
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            idx = _safe_int(inst.get("mask_index"))
            if idx:
                mapping[idx] = inst
    return mapping


def _collect_instances_by_index_for_series(
    mask_model: Dict[str, Any], series_uid: str
) -> Dict[int, Dict[str, Any]]:
    if not series_uid:
        return _collect_instances_by_index(mask_model)
    mapping: Dict[int, Dict[str, Any]] = {}
    series_list = mask_model.get("series") or []
    for series in series_list:
        if not isinstance(series, dict):
            continue
        if _safe_str(series.get("series_instance_uid")) != series_uid:
            continue
        instances = series.get("instances") or []
        if not isinstance(instances, list):
            continue
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            idx = _safe_int(inst.get("mask_index"))
            if idx:
                mapping[idx] = inst
    return mapping


def _get_series_info(payload: Dict[str, Any], model_type: int) -> Tuple[str, int]:
    series_type = _pick_series_type(payload, model_type)
    series_uid = _get_series_uid_by_type(payload, series_type)
    if not series_uid:
        series_uid = _get_first_series_uid(payload)
    return series_uid, series_type


def _get_model_series_types(payload: Dict[str, Any], model_type: int) -> List[int]:
    study = payload.get("study") or {}
    model_list = study.get("model") or []
    out: List[int] = []
    for model in model_list:
        if not isinstance(model, dict):
            continue
        mt = _safe_int(model.get("model_type"))
        if mt != model_type:
            continue
        series_type = model.get("series_type")
        if isinstance(series_type, list):
            for item in series_type:
                val = _safe_int(item)
                if val and val not in out:
                    out.append(val)
        else:
            val = _safe_int(series_type)
            if val and val not in out:
                out.append(val)
    return out


def _pick_series_type(payload: Dict[str, Any], model_type: int) -> int:
    types = _get_model_series_types(payload, model_type)
    return types[0] if types else 0


def _get_series_uid_by_type(payload: Dict[str, Any], series_type: int) -> str:
    if series_type <= 0:
        return ""
    study = payload.get("study") or {}
    series_list = study.get("series") or []
    for series in series_list:
        if not isinstance(series, dict):
            continue
        if _safe_int(series.get("series_type")) == series_type:
            return str(series.get("series_instance_uid", "") or "")
    return ""


def _get_first_series_uid(payload: Dict[str, Any]) -> str:
    study = payload.get("study") or {}
    series_list = study.get("series") or []
    if isinstance(series_list, list) and series_list:
        first = series_list[0]
        if isinstance(first, dict):
            return str(first.get("series_instance_uid", "") or "")
    return ""


def _select_series_by_type(mask_model: Dict[str, Any], series_type: int) -> Optional[Dict[str, Any]]:
    if series_type <= 0:
        return None
    series_list = mask_model.get("series") or []
    if not isinstance(series_list, list):
        return None
    for series in series_list:
        if not isinstance(series, dict):
            continue
        if _safe_int(series.get("series_type")) == series_type:
            return series
    return None


def _get_study_uid(payload: Dict[str, Any]) -> str:
    study = payload.get("study") or {}
    return str(study.get("study_instance_uid", "") or "")


def _model_name_from_type(model_type: int) -> str:
    name = MODEL_NAME_BY_TYPE.get(model_type)
    if not name:
        raise ValueError(f"不支援的 model_type：{model_type}")
    return name


def _get_sorted_slices(payload: Dict[str, Any], series_uid: str) -> List[SortedSliceItem]:
    sorted_block = payload.get("sorted")
    if not isinstance(sorted_block, dict):
        return []
    series = sorted_block.get("series")
    if not isinstance(series, list) or not series:
        return []
    chosen = None
    if series_uid:
        for item in series:
            if not isinstance(item, dict):
                continue
            if _safe_str(item.get("series_instance_uid")) == series_uid:
                chosen = item
                break
    if chosen is None:
        first = series[0]
        if not isinstance(first, dict):
            return []
        chosen = first

    instance_list = chosen.get("instance")
    if not isinstance(instance_list, list):
        return []
    out: List[SortedSliceItem] = []
    for item in instance_list:
        if not isinstance(item, dict):
            continue
        uid = str(item.get("sop_instance_uid", "") or "")
        if not uid:
            continue
        try:
            proj = float(item.get("projection"))
        except (TypeError, ValueError):
            continue
        out.append(SortedSliceItem(sop_instance_uid=uid, projection=proj))
    return out


def _sop_uid_by_slice_index(sorted_slices: List[SortedSliceItem], slice_index: int) -> str:
    if slice_index <= 0:
        return ""
    idx = slice_index - 1
    if idx >= len(sorted_slices):
        return ""
    return _safe_str(sorted_slices[idx].sop_instance_uid)


def _map_sop_uid_by_projection(
    *,
    source_uid: str,
    source_sorted: List[SortedSliceItem],
    target_sorted: List[SortedSliceItem],
    mat: np.ndarray,
    direction: str,
) -> str:
    if not source_uid or not source_sorted or not target_sorted:
        return ""
    z_src = None
    for item in source_sorted:
        if item.sop_instance_uid == source_uid:
            z_src = item.projection
            break
    if z_src is None:
        return ""
    m = mat if direction == "F2B" else _safe_inv(mat)
    z_tgt = float(m[2, 2] * z_src + m[2, 3])
    projections = np.asarray([t.projection for t in target_sorted], dtype=np.float64)
    idx = int(np.searchsorted(projections, z_tgt))
    if idx <= 0:
        return target_sorted[0].sop_instance_uid
    if idx >= len(target_sorted):
        return target_sorted[-1].sop_instance_uid
    prev_i = idx - 1
    chosen = idx if abs(projections[idx] - z_tgt) < abs(projections[prev_i] - z_tgt) else prev_i
    return target_sorted[chosen].sop_instance_uid


def _map_slice_index(value: int, *, mat: np.ndarray, direction: str) -> int:
    if value < 0:
        return 0
    if direction not in {"F2B", "B2F"}:
        return value
    m = mat if direction == "F2B" else _safe_inv(mat)
    z_tgt = float(m[2, 2] * value + m[2, 3])
    return int(np.rint(z_tgt))


def _build_sorted_mapping_one_based(
    *,
    current_slices: int,
    prior_slices: int,
    mat: np.ndarray,
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

    current_choice: List[Optional[int]] = [
        _select_candidate(cands) for cands in current_candidates
    ]
    current_choice_filled = _fill_empty_choices(current_choice)
    sorted_list = [
        {"current_slice": idx + 1, "followup_slice": prior_idx + 1}
        for idx, prior_idx in enumerate(current_choice_filled)
    ]

    prior_candidates: List[List[int]] = [[] for _ in range(prior_slices)]
    for prior_idx in range(prior_slices):
        z_tgt = float(mat[2, 2] * prior_idx + mat[2, 3])
        current_idx = _clamp_index(int(np.rint(z_tgt)), current_slices)
        prior_candidates[prior_idx].append(current_idx)
    empty_prior = sum(1 for cands in prior_candidates if not cands)

    prior_choice: List[Optional[int]] = [
        _select_candidate(cands) for cands in prior_candidates
    ]
    prior_choice_filled = _fill_empty_choices(prior_choice)
    reverse_sorted_list = [
        {"followup_slice": idx + 1, "current_slice": current_idx + 1}
        for idx, current_idx in enumerate(prior_choice_filled)
    ]

    LOGGER.info(
        "sorted_slice fill stats: empty_current=%d empty_prior=%d",
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
            "sorted_slice median pick: candidates=%s median=%.2f chosen=%d",
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

    mid_fill_count = 0
    for i, val in enumerate(filled):
        if val is not None:
            continue
        left = prev_idx[i]
        right = next_idx[i]
        if left == -1:
            filled[i] = filled[right]
            continue
        if right == -1:
            filled[i] = filled[left]
            continue
        filled[i] = filled[left] if (i - left) <= (right - i) else filled[right]
        mid_fill_count += 1

    if mid_fill_count:
        LOGGER.info("sorted_slice mid-gap fill count: %d", mid_fill_count)
    return [int(v) for v in filled]


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


def _normalize_date_key(date_str: str) -> str:
    clean = date_str.strip()
    if "-" in clean:
        parts = clean.split("-")
        if len(parts) == 3:
            return f"{parts[0]}{parts[1].rjust(2,'0')}{parts[2].rjust(2,'0')}"
    if len(clean) == 8 and clean.isdigit():
        return clean
    raise ValueError(f"無法解析 study_date：{date_str}")


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


def _safe_inv(mat: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.eye(4, dtype=np.float64)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_str(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 JSON：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Follow-up v3 pipeline (Platform JSON + NIfTI)")
    p.add_argument("--input_json", default="/data/4TB1/pipeline/chuan/process/Deep_FollowUp/00130846_followup_input.json")
    p.add_argument(
        "--path_process",
        default="/data/4TB1/pipeline/sean/rename_nifti/",
        help="case_id 根目錄（每個 case 資料夾內放 Pred/SynthSEG/JSON）",
    )
    p.add_argument(
        "--path_followup_root",
        default="/data/4TB1/pipeline/chuan/process",
        help="Follow-up 輸出根目錄（會建立 Deep_FollowUp/<case_id>/...）",
    )
    p.add_argument(
        "--case_id",
        default="00130846_20220105_MR_E211126000320",
        help="指定 current case_id（patientid_studydate_MR_accessnumber），用於無法由 ids 解出時",
    )
    p.add_argument("--model", default="CMB")
    p.add_argument("--model_type", type=int, default=2, help="優先度最高的")
    p.add_argument(
        "--platform_json_name",
        default="",
        help="Pred_<Model>_platform_json.json 檔名覆蓋",
    )
    p.add_argument("--fsl_flirt_path", default="/usr/local/fsl/bin/flirt")
    p.add_argument("--path_log", default="/data/4TB1/pipeline/chuan/log")
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
        LOGGER.info("Log file: %s", log_file)
        print(f"[FollowUp] Log file: {log_file}")
    else:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    try:
        outputs, has_model = pipeline_followup_v3_platform(
            input_json=Path(args.input_json),
            path_process=Path(args.path_process),
            model=args.model or None,
            model_type=args.model_type or None,
            case_id=args.case_id or None,
            platform_json_name=args.platform_json_name or None,
            path_followup_root=Path(args.path_followup_root) if args.path_followup_root else None,
            fsl_flirt_path=args.fsl_flirt_path,
        )
        if not has_model:
            LOGGER.warning("Skip follow-up: model/model_type not in needFollowup.models")
            print("[FollowUp] Skip: model/model_type not in needFollowup.models")
        for output in outputs:
            print(f"[Done] output={output}")
            LOGGER.info("[Done] output=%s", output)
    except Exception:
        LOGGER.error("Follow-up v3 platform failed.", exc_info=True)
        print("[Error] Follow-up v3 platform failed. Please check log for details.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
