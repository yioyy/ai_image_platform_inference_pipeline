"""Infarct 平台 JSON 與 DICOM-SEG 產生器."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pydicom

from code_ai.pipeline.dicomseg import utils
from code_ai.pipeline.dicomseg.schema.enum import ModelTypeEnum, SeriesTypeEnum
from code_ai.pipeline.dicomseg.schema.base import (
    InstanceRequest,
    SortedRequest,
    SortedSeriesRequest,
    StudyRequest,
    StudySeriesRequest,
    StudyModelRequest,
)
from code_ai.pipeline.dicomseg.schema.infarct import (
    InfarctAITeamRequest,
    InfarctMaskInstanceRequest,
    InfarctMaskModelRequest,
    InfarctMaskRequest,
    InfarctMaskSeriesRequest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "dataset.json"


@dataclass
class LesionDefinition:
    index: int
    mask: np.ndarray
    location: str
    size: float
    prob_max: float
    prob_mean: float
    mean_adc: float
    color: str


@dataclass
class SeriesBundle:
    sorted_paths: List[str]
    image: object
    first_dcm: pydicom.FileDataset
    source_images: List[pydicom.FileDataset]


def main(_id: str, path_root: Path, group_id: int = 57) -> Path:
    return make_pred_json(_id, path_root, group_id)


def make_pred_json(_id: str, path_root: Path, group_id: int = 57) -> Path:
    return execute_dicomseg_platform_json(_id, path_root, group_id)


def execute_dicomseg_platform_json(_id: str, root_path: Path, group_id: int = 57) -> Path:
    root_path = Path(root_path)
    path_dicom = root_path / "Dicom"
    path_result = root_path / "Result"
    excel_path = root_path / "excel" / "Infarct_Pred_list.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"找不到 Excel：{excel_path}")

    dataset_cfg = _load_dataset_config()
    excel_row = pd.read_excel(excel_path).fillna("").iloc[0]

    pred_mask = utils.get_array_to_dcm_axcodes(path_result / "Pred.nii.gz") > 0
    synth_array = utils.get_array_to_dcm_axcodes(path_result / "Pred_synthseg.nii.gz")

    lesions = _build_lesion_definitions(excel_row, pred_mask, synth_array, dataset_cfg)
    if not lesions:
        logging.info("未偵測到梗塞病灶，將產出不含 mask 的平台 JSON")

    series_data = _load_required_series(path_dicom)
    if SeriesTypeEnum.DWI1000.name not in series_data or SeriesTypeEnum.ADC.name not in series_data:
        raise FileNotFoundError("需要 DWI1000 與 ADC 序列才能建立 Infarct 平台資料")

    dcmseg_dir = path_dicom / "Dicom-Seg"
    dcmseg_dir.mkdir(exist_ok=True)

    lesion_map = {lesion.index: lesion for lesion in lesions}

    dwi1000_results = _export_segmentation_for_series(
        SeriesTypeEnum.DWI1000,
        lesions,
        series_data[SeriesTypeEnum.DWI1000.name],
        dcmseg_dir,
    )
    adc_results = _export_segmentation_for_series(
        SeriesTypeEnum.ADC,
        lesions,
        series_data[SeriesTypeEnum.ADC.name],
        dcmseg_dir,
    )

    mask_series: List[InfarctMaskSeriesRequest] = []
    for series_enum, results in (
        (SeriesTypeEnum.DWI1000, dwi1000_results),
        (SeriesTypeEnum.ADC, adc_results),
    ):
        if not results:
            continue
        bundle = series_data[series_enum.name]
        series_obj = _build_mask_series(series_enum, results, lesion_map, bundle.source_images)
        if series_obj:
            mask_series.append(series_obj)

    if not mask_series:
        # 無病灶：仍生成 mask 結構，但 instances 為空（對齊 aneurysm 行為）
        mask_series = _build_empty_mask_series(series_data)

    study_request = _build_study_request(series_data, group_id, len(lesions))
    sorted_request = _build_sorted_request(series_data)
    mask_request: Optional[InfarctMaskRequest] = InfarctMaskRequest(
        study_instance_uid=study_request.study_instance_uid,
        group_id=group_id,
        model=[
            InfarctMaskModelRequest(
                model_type=ModelTypeEnum.Infarct.value,
                series=mask_series,
            )
        ],
    )

    ai_request = InfarctAITeamRequest(
        study=study_request,
        sorted=sorted_request,
        mask=mask_request,
    )

    json_dir = root_path / "JSON"
    json_dir.mkdir(exist_ok=True)
    output_path = json_dir / f"{_id}_platform_json.json"
    payload = ai_request.model_dump(mode="json")
    _merge_study_models(payload)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    return output_path


def _load_dataset_config() -> Dict[str, Dict[str, List[int]]]:
    with open(DATASET_PATH, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {
        "labels": data.get("Infarct_labels_OHIF", {}),
        "colors": data.get("Infarct_colors_OHIF", {}),
    }


def _build_lesion_definitions(
    row: pd.Series,
    pred_mask: np.ndarray,
    synth_array: np.ndarray,
    dataset_cfg: Dict[str, Dict[str, List[int]]],
) -> List[LesionDefinition]:
    lesion_count = int(row.get("Lesion_Number", 0))
    lesions: List[LesionDefinition] = []
    label_map = dataset_cfg["labels"]
    color_map = dataset_cfg["colors"]

    for idx in range(1, lesion_count + 1):
        prefix = f"{idx}_"
        size = _safe_float(row.get(f"{prefix}size"))
        prob_max = _safe_float(row.get(f"{prefix}Prob_max"))
        prob_mean = _safe_float(row.get(f"{prefix}Prob_mean"))
        mean_adc = _safe_float(row.get(f"{prefix}MeanADC"))
        location = str(row.get(f"{prefix}Location", "")).strip()

        if idx == 1:
            mask = pred_mask.copy()
            display_name = "Total"
            color = "hotpink"
        else:
            if not location or location not in label_map:
                logging.warning("找不到腦區 %s 的標籤，跳過 index=%s", location, idx)
                continue
            mask = np.isin(synth_array, np.array(label_map[location], dtype=np.int32))
            mask &= pred_mask
            if not mask.any():
                logging.warning("%s 無有效遮罩，跳過 index=%s", location, idx)
                continue
            display_name = location
            color = color_map.get(location, "lawngreen")

        lesions.append(
            LesionDefinition(
                index=idx,
                mask=mask.astype(bool),
                location=display_name,
                size=size,
                prob_max=prob_max,
                prob_mean=prob_mean,
                mean_adc=mean_adc,
                color=color,
            )
        )
    return lesions


def _load_required_series(path_dicom: Path) -> Dict[str, SeriesBundle]:
    series_data: Dict[str, SeriesBundle] = {}
    for series_enum in (SeriesTypeEnum.DWI0, SeriesTypeEnum.DWI1000, SeriesTypeEnum.ADC):
        series_dir = path_dicom / series_enum.name
        if not series_dir.is_dir():
            logging.warning("序列資料夾不存在：%s", series_dir)
            continue
        sorted_dcms, image, first_dcm, source_images = utils.load_and_sort_dicom_files(str(series_dir))
        series_data[series_enum.name] = SeriesBundle(
            sorted_paths=sorted_dcms,
            image=image,
            first_dcm=first_dcm,
            source_images=source_images,
        )
    return series_data


def _export_segmentation_for_series(
    series_enum: SeriesTypeEnum,
    lesions: List[LesionDefinition],
    bundle: SeriesBundle,
    output_dir: Path,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for lesion in lesions:
        if not lesion.mask.any():
            continue
        label_dict = {
            1: {
                "SegmentLabel": lesion.location or f"Lesion {lesion.index}",
                "color": lesion.color,
            }
        }
        template = utils.get_dicom_seg_template(f"Infarct_{series_enum.name}", label_dict)
        dcm_seg = utils.make_dicomseg_file(
            lesion.mask.astype("uint8"),
            bundle.image,
            bundle.first_dcm,
            bundle.source_images,
            template,
        )
        seg_path = output_dir / f"{series_enum.name}_A{lesion.index}.dcm"
        dcm_seg.save_as(seg_path)
        main_slice = _calc_main_slice(lesion.mask)
        results.append(
            {
                "mask_index": lesion.index,
                "mask_path": seg_path,
                "main_seg_slice": main_slice,
            }
        )
    return results


def _calc_main_slice(mask: np.ndarray) -> int:
    indices = np.where(mask)
    if indices[0].size == 0:
        return 0
    return int(np.median(indices[0]))


def _build_mask_series(
    series_enum: SeriesTypeEnum,
    segmentation_results: List[Dict[str, object]],
    lesion_map: Dict[int, LesionDefinition],
    source_images: List[pydicom.FileDataset],
) -> Optional[InfarctMaskSeriesRequest]:
    instances: List[InfarctMaskInstanceRequest] = []
    num_images = len(source_images)
    for result in segmentation_results:
        lesion = lesion_map.get(result["mask_index"])
        if lesion is None:
            continue
        dcm_seg = pydicom.dcmread(str(result["mask_path"]), stop_before_pixels=True)
        slice_idx = max(0, min(int(result["main_seg_slice"]), num_images - 1))
        dicom_uid = source_images[slice_idx].get((0x0008, 0x0018)).value
        instances.append(
            InfarctMaskInstanceRequest(
                mask_index=lesion.index,
                mask_name=f"A{lesion.index}",
                volume=f"{lesion.size:.1f}",
                type="",
                location=lesion.location,
                mean_adc=str(int(round(lesion.mean_adc))),
                prob_max=f"{lesion.prob_max:.2f}",
                prob_mean=f"{lesion.prob_mean:.2f}",
                checked="1",
                is_ai="1",
                seg_series_instance_uid=dcm_seg[0x20, 0x000E].value,
                seg_sop_instance_uid=dcm_seg[0x08, 0x0018].value,
                dicom_sop_instance_uid=dicom_uid,
                main_seg_slice=slice_idx,
                is_main_seg=1,
            )
        )

    if not instances:
        return None

    series_instance_uid = source_images[0].get((0x0020, 0x000E)).value
    return InfarctMaskSeriesRequest(
        series_instance_uid=series_instance_uid,
        series_type=series_enum.name,
        instances=instances,
        model_type=ModelTypeEnum.Infarct.value,
    )


def _build_empty_mask_series(series_data: Dict[str, SeriesBundle]) -> List[InfarctMaskSeriesRequest]:
    """無病灶時依系列建立空的 mask series（instances 為空）。"""
    empty_series: List[InfarctMaskSeriesRequest] = []
    for series_enum in (SeriesTypeEnum.DWI1000, SeriesTypeEnum.ADC):
        bundle = series_data.get(series_enum.name)
        if not bundle or not bundle.source_images:
            continue
        series_instance_uid = bundle.source_images[0].get((0x0020, 0x000E)).value
        empty_series.append(
            InfarctMaskSeriesRequest(
                series_instance_uid=series_instance_uid,
                series_type=series_enum.name,
                instances=[],
                model_type=ModelTypeEnum.Infarct.value,
            )
        )
    return empty_series


def _build_study_request(
    series_data: Dict[str, SeriesBundle], group_id: int, lesion_count: int
) -> StudyRequest:
    if not series_data:
        raise ValueError("缺少 DICOM series 資料，無法建立 StudyRequest")

    primary_bundle = next(iter(series_data.values()))
    first_dcm = primary_bundle.source_images[0]

    series_entries, model_entries = _build_study_components(series_data, lesion_count)

    def _parse_date(value: str) -> datetime.date:
        try:
            return datetime.strptime(value, "%Y%m%d").date()
        except Exception:
            return datetime.utcnow().date()

    def _parse_age(value: str) -> int:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else 0

    study_date = _parse_date(getattr(first_dcm, "StudyDate", "19700101"))
    return StudyRequest(
        group_id=group_id,
        study_date=study_date,
        gender=getattr(first_dcm, "PatientSex", "O"),
        age=_parse_age(getattr(first_dcm, "PatientAge", "0")),
        study_name=getattr(first_dcm, "StudyDescription", "MR Infarct"),
        patient_name=str(getattr(first_dcm, "PatientName", "")),
        study_instance_uid=getattr(first_dcm, "StudyInstanceUID", ""),
        patient_id=getattr(first_dcm, "PatientID", ""),
        series=series_entries,
        model=model_entries,
    )


def _build_study_components(
    series_data: Dict[str, SeriesBundle], lesion_count: int
) -> tuple[List[StudySeriesRequest], List[StudyModelRequest]]:
    series_entries: List[StudySeriesRequest] = []
    model_entries: List[StudyModelRequest] = []

    for series_name, bundle in series_data.items():
        if series_name == SeriesTypeEnum.DWI0.name:
            # 不在 study/model 中包含 DWI0
            continue
        if not bundle.source_images:
            continue
        first_dcm = bundle.source_images[0]
        series_entries.append(
            StudySeriesRequest(
                series_type=series_name,
                series_instance_uid=first_dcm.get((0x0020, 0x000E)).value,
                resolution_x=int(getattr(first_dcm, "Rows", 512)),
                resolution_y=int(getattr(first_dcm, "Columns", 512)),
            )
        )
        enum_member = getattr(SeriesTypeEnum, series_name, None)
        series_code = enum_member.value if isinstance(enum_member, SeriesTypeEnum) else series_name
        model_entries.append(
            StudyModelRequest(
                series_type=series_code,
                model_type=ModelTypeEnum.Infarct.name,
                lession=lesion_count,
                status="1",
                report="",
            )
        )

    if not series_entries:
        raise ValueError("Study series 資料為空，無法建立 StudyRequest")

    return series_entries, model_entries


def _build_sorted_request(series_data: Dict[str, SeriesBundle]) -> SortedRequest:
    sorted_series: List[SortedSeriesRequest] = []
    study_uid = ""
    for series_name, bundle in series_data.items():
        if series_name == SeriesTypeEnum.DWI0.name:
            # 排序列表不包含 DWI0
            continue
        if not bundle.source_images:
            continue
        instances = []
        for ds in bundle.source_images:
            sop_uid = ds.get((0x0008, 0x0018)).value
            ipp = ds.get((0x0020, 0x0032)).value
            projection = float(ipp[-1]) if ipp and len(ipp) >= 3 else 0.0
            instances.append(
                InstanceRequest(
                    sop_instance_uid=sop_uid,
                    projection=str(round(projection, 6)),
                )
            )
        sorted_series.append(
            SortedSeriesRequest(
                series_instance_uid=bundle.source_images[0].get((0x0020, 0x000E)).value,
                instance=instances,
            )
        )
        if not study_uid:
            study_uid = bundle.source_images[0].get((0x0020, 0x000D)).value

    return SortedRequest(study_instance_uid=study_uid, series=sorted_series)


def _merge_study_models(payload: Dict[str, object]) -> None:
    study = payload.get("study")
    if not isinstance(study, dict):
        return
    models = study.get("model")
    if not isinstance(models, list) or not models:
        return

    merged = {
        "model_type": models[0].get("model_type", ""),
        "lession": models[0].get("lession", "0"),
        "status": models[0].get("status", "1"),
        "report": models[0].get("report", ""),
        "series_type": [m.get("series_type") for m in models if "series_type" in m],
    }
    study["model"] = [merged]


def _safe_float(value) -> float:
    if value in ("", None):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
