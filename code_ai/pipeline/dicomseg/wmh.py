"""WMH 平台 JSON 與 DICOM-SEG 產生器。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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
from code_ai.pipeline.dicomseg.schema.wmh import (
    WMHAITeamRequest,
    WMHMaskInstanceRequest,
    WMHMaskModelRequest,
    WMHMaskRequest,
    WMHMaskSeriesRequest,
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
    color: str


@dataclass
class SeriesBundle:
    sorted_paths: List[str]
    image: object
    first_dcm: pydicom.FileDataset
    source_images: List[pydicom.FileDataset]


def main(_id: str, path_root: Path, group_id: int = 57) -> Path:
    return execute_dicomseg_platform_json(_id, path_root, group_id)


def execute_dicomseg_platform_json(_id: str, root_path: Path, group_id: int = 57) -> Path:
    root_path = Path(root_path)
    path_dicom = root_path / "Dicom"
    path_result = root_path / "Result"
    excel_path = root_path / "excel" / "WMH_Pred_list.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"找不到 Excel：{excel_path}")

    dataset_cfg = _load_dataset_config()
    excel_row = pd.read_excel(excel_path).fillna("").iloc[0]

    pred_mask = utils.get_array_to_dcm_axcodes(path_result / "Pred.nii.gz") > 0
    synth_array = utils.get_array_to_dcm_axcodes(path_result / "Pred_synthseg.nii.gz")

    lesions = _build_lesion_definitions(excel_row, pred_mask, synth_array, dataset_cfg)
    if not lesions:
        raise ValueError("WMH_Pred_list.xlsx 無任何有效病灶資料")

    series_data = _load_required_series(path_dicom)
    if SeriesTypeEnum.T2FLAIR_AXI.name not in series_data:
        raise FileNotFoundError("需要 T2FLAIR 序列才能建立 WMH 平台資料")

    dcmseg_dir = path_dicom / "Dicom-Seg"
    dcmseg_dir.mkdir(exist_ok=True)

    lesion_map = {lesion.index: lesion for lesion in lesions}

    t2_results = _export_segmentation_for_series(
        SeriesTypeEnum.T2FLAIR_AXI,
        lesions,
        series_data[SeriesTypeEnum.T2FLAIR_AXI.name],
        dcmseg_dir,
    )

    mask_series: List[WMHMaskSeriesRequest] = []
    fazekas_value = excel_row.get("Fazekas", "")
    if t2_results:
        bundle = series_data[SeriesTypeEnum.T2FLAIR_AXI.name]
        series_obj = _build_mask_series(
            SeriesTypeEnum.T2FLAIR_AXI,
            t2_results,
            lesion_map,
            bundle.source_images,
            str(fazekas_value),
        )
        if series_obj:
            mask_series.append(series_obj)

    if not mask_series:
        raise ValueError("無法產生任何 DICOM-SEG")

    study_request = _build_study_request(series_data, group_id, len(lesions))
    sorted_request = _build_sorted_request(series_data)
    mask_request = WMHMaskRequest(
        study_instance_uid=study_request.study_instance_uid,
        group_id=group_id,
        model=[
            WMHMaskModelRequest(
                model_type=ModelTypeEnum.WMH.value,
                series=mask_series,
            )
        ],
    )

    ai_request = WMHAITeamRequest(
        study=study_request,
        sorted=sorted_request,
        mask=mask_request,
    )

    json_dir = root_path / "JSON"
    json_dir.mkdir(exist_ok=True)
    output_path = json_dir / f"{_id}_platform_json.json"
    payload = ai_request.model_dump(mode="json")
    _merge_study_models(payload)
    _ensure_series_type_list(payload)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    return output_path


def _load_dataset_config() -> Dict[str, Dict[str, List[int]]]:
    with open(DATASET_PATH, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {
        "labels": data.get("WMH_labels", {}),
        "colors": data.get("WMH_colors", {}),
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
        location = str(row.get(f"{prefix}Location", "")).strip()

        if idx == 1 or not location or location == "Total":
            mask = pred_mask.copy()
            display_name = "Total"
            color = "hotpink"
        else:
            label_ids = label_map.get(location)
            if not label_ids:
                logging.warning("找不到腦區 %s 的標籤，跳過 index=%s", location, idx)
                continue
            mask = np.isin(synth_array, np.array(label_ids, dtype=np.int32))
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
                color=color,
            )
        )
    return lesions


def _load_required_series(path_dicom: Path) -> Dict[str, SeriesBundle]:
    series_data: Dict[str, SeriesBundle] = {}
    series_enum = SeriesTypeEnum.T2FLAIR_AXI
    series_dir = path_dicom / series_enum.name
    if not series_dir.is_dir():
        logging.warning("序列資料夾不存在：%s", series_dir)
        return series_data
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
        template = utils.get_dicom_seg_template(f"WMH_{series_enum.name}", label_dict)
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
    fazekas: str,
) -> Optional[WMHMaskSeriesRequest]:
    instances: List[WMHMaskInstanceRequest] = []
    num_images = len(source_images)
    for result in segmentation_results:
        lesion = lesion_map.get(result["mask_index"])
        if lesion is None:
            continue
        dcm_seg = pydicom.dcmread(str(result["mask_path"]), stop_before_pixels=True)
        slice_idx = max(0, min(int(result["main_seg_slice"]), num_images - 1))
        dicom_uid = source_images[slice_idx].get((0x0008, 0x0018)).value
        instances.append(
            WMHMaskInstanceRequest(
                mask_index=lesion.index,
                mask_name=f"A{lesion.index}",
                volume=f"{lesion.size:.1f}",
                type="",
                location=lesion.location,
                fazekas=str(fazekas),
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
    return WMHMaskSeriesRequest(
        series_instance_uid=series_instance_uid,
        series_type=series_enum.name,
        instances=instances,
        model_type=ModelTypeEnum.WMH.value,
    )


def _build_study_request(
    series_data: Dict[str, SeriesBundle], group_id: int, lesion_count: int
) -> StudyRequest:
    if not series_data:
        raise ValueError("缺少 DICOM series 資料，無法建立 StudyRequest")

    primary_bundle = next(iter(series_data.values()))
    first_dcm = primary_bundle.source_images[0]

    series_entries, model_entries = _build_study_components(series_data, lesion_count)

    def _parse_date(value: str):
        try:
            return pd.to_datetime(value, format="%Y%m%d").date()
        except Exception:
            return pd.Timestamp.utcnow().date()

    def _parse_age(value: str):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else 0

    study_date = _parse_date(getattr(first_dcm, "StudyDate", "19700101"))
    return StudyRequest(
        group_id=group_id,
        study_date=study_date,
        gender=getattr(first_dcm, "PatientSex", "O"),
        age=_parse_age(getattr(first_dcm, "PatientAge", "0")),
        study_name=getattr(first_dcm, "StudyDescription", "MR WMH"),
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
                model_type=ModelTypeEnum.WMH.name,
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
    for bundle in series_data.values():
        if not bundle.source_images:
            continue
        study_uid = bundle.source_images[0].get((0x0020, 0x000d)).value
        instances = []
        for ds in bundle.source_images:
            sop_uid = ds.get((0x0008, 0x0018)).value
            ipp = ds.get((0x0020, 0x0032)).value
            projection = float(ipp[-1]) if ipp and len(ipp) >= 3 else 0.0
            instances.append(
                InstanceRequest(
                    sop_instance_uid=sop_uid,
                    projection=str(projection),
                )
            )
        series_request = SortedSeriesRequest(
            series_instance_uid=bundle.source_images[0].get((0x0020, 0x000E)).value,
            instance=instances,
        )
        sorted_series.append(series_request)

    return SortedRequest(
        study_instance_uid=study_uid,
        series=sorted_series,
    )


def _merge_study_models(payload: Dict[str, any]) -> None:
    """Merge duplicated model entries for Orthanc consumption."""
    try:
        models = payload["study"]["model"]
    except KeyError:
        return
    unique = {}
    for model in models:
        key = (model["series_type"], model["model_type"])
        unique[key] = model
    payload["study"]["model"] = list(unique.values())


def _ensure_series_type_list(payload: Dict[str, any]) -> None:
    try:
        models = payload["study"]["model"]
    except KeyError:
        return
    for model in models:
        series_type = model.get("series_type")
        if isinstance(series_type, str):
            model["series_type"] = [series_type]


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


