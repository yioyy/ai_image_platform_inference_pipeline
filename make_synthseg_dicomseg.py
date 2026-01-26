# -*- coding: utf-8 -*-
"""
Make SynthSEG DICOM-SEG (single multi-label file).

你目前環境的 `pydantic` 是 v1，但 `code_ai.pipeline.dicomseg` 依賴 pydantic v2，
所以本腳本 **完全不 import code_ai**，避免被 pydantic 版本卡死。

輸入：
- `SynthSEG_CMB.nii.gz`（多 label）
- 一個參考 DICOM series 目錄（同一個 series）

輸出：
- 單一檔案 `SynthSEG_CMB.dcm`（multi-segment DICOM-SEG，包含所有 label > 0）
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib
import pydicom
import pydicom_seg
import SimpleITK as sitk


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert label NIfTI (Pred/SynthSEG) to a single multi-label DICOM-SEG")
    p.add_argument(
        "--mode",
        choices=["pred", "synthseg"],
        default="pred",
        help="輸入 NIfTI 類型：pred 或 synthseg。會影響預設 output_name/series_description。",
    )
    p.add_argument(
        "--path_dicom",
        default="C:/Users/user/Desktop/orthanc_combine_code/目前pipeline版本/process/Deep_FollowUp/00130846_20220105_MR_E211126000320/dicom/followup",
        help="參考的 DICOM series 資料夾（需為同一個 series）",
    )
    p.add_argument(
        "--path_nii",
        default="C:/Users/user/Desktop/orthanc_combine_code/目前pipeline版本/process/Deep_FollowUp/00130846_20220105_MR_E211126000320_20260109/followup/Pred_CMB.nii.gz",
        help="要轉成 DICOM-SEG 的 NIfTI 路徑（Pred 或 SynthSEG，label map）",
    )
    # backwards-compat alias
    p.add_argument(
        "--path_synthseg_nii",
        default="",
        help="相容舊參數：請改用 --path_nii",
    )
    p.add_argument(
        "--output_dir",
        default="C:/Users/user/Desktop/orthanc_combine_code/目前pipeline版本/process/Deep_FollowUp/00130846_20220105_MR_E211126000320/followup",
        help="輸出資料夾",
    )
    p.add_argument("--output_name", default="Pred_CMB.dcm", help="輸出 DICOM-SEG 檔名（預設依 mode: Pred_CMB.dcm 或 SynthSEG_CMB.dcm）")
    p.add_argument("--series_description", default="Pred_CMB", help="DICOM-SEG SeriesDescription（預設依 mode: Pred_CMB 或 SynthSEG_CMB）")
    return p


def _as_float_array(value: Any, *, length: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.size < length:
        return None
    return arr[:length]


def _calc_projection(ds: pydicom.Dataset) -> Optional[float]:
    iop = _as_float_array(getattr(ds, "ImageOrientationPatient", None), length=6)
    ipp = _as_float_array(getattr(ds, "ImagePositionPatient", None), length=3)
    if iop is None or ipp is None:
        return None
    normal = np.cross(iop[0:3], iop[3:6])
    try:
        return float(np.dot(ipp[0:3], normal))
    except Exception:
        return None


def load_and_sort_dicom_series(path_dicom: str | Path) -> Tuple[sitk.Image, pydicom.Dataset, List[pydicom.Dataset]]:
    """
    - 依 projection（與 `get_dicom_info.py` 相同）排序
    - 回傳 SITK image（含幾何）、first_dcm、source_images（stop_before_pixels=True）
    """
    path_dicom = Path(path_dicom)
    if not path_dicom.exists():
        raise FileNotFoundError(f"path_dicom 不存在：{path_dicom}")

    files = [p for p in path_dicom.rglob("*") if p.is_file()]
    if not files:
        raise RuntimeError(f"在 {path_dicom} 找不到 DICOM")

    items: List[Tuple[float, int, Path]] = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue
        proj = _calc_projection(ds)
        if proj is None:
            continue
        inst = int(getattr(ds, "InstanceNumber", 2**31 - 1) or 2**31 - 1)
        items.append((float(proj), inst, p))

    if not items:
        raise RuntimeError("DICOM 缺少 IOP/IPP，無法排序")

    items.sort(key=lambda x: (x[0], x[1], x[2].name))
    sorted_files = [p for _, _, p in items]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in sorted_files])
    image = reader.Execute()

    first_dcm = pydicom.dcmread(str(sorted_files[0]), stop_before_pixels=True, force=True)
    source_images = [pydicom.dcmread(str(p), stop_before_pixels=True, force=True) for p in sorted_files]
    return image, first_dcm, source_images


def compute_orientation(init_axcodes, final_axcodes):
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)
    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)
    return ornt_transf, ornt_init, ornt_fin


def do_reorientation(data_array, init_axcodes, final_axcodes):
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array
    return nib.orientations.apply_orientation(data_array, ornt_transf)


def get_array_to_dcm_axcodes(path_nii: str | Path) -> np.ndarray:
    pred_nii = nib.load(str(path_nii))
    pred_data = np.asarray(pred_nii.dataobj)
    pred_nii_obj_axcodes = tuple(nib.aff2axcodes(pred_nii.affine))
    # 跟原 utils/base.py 一致：轉到 ('S','P','L')
    return do_reorientation(pred_data, pred_nii_obj_axcodes, ("S", "P", "L"))


def get_dicom_seg_template(series_description: str, label_ids: Sequence[int]) -> Dict[str, Any]:
    """
    產生 dcmqi metainfo（pydicom_seg 需要的 template）。
    每個 label_id 一個 segment，labelID=label_id。
    """
    segment_attributes: List[Dict[str, Any]] = []
    for lid in label_ids:
        segment_attributes.append(
            {
                "labelID": int(lid),
                "SegmentLabel": f"A{int(lid)}",
                "SegmentAlgorithmType": "MANUAL",
                "SegmentAlgorithmName": "SynthSEG",
                "SegmentedPropertyCategoryCodeSequence": {
                    "CodeValue": "M-01000",
                    "CodingSchemeDesignator": "SRT",
                    "CodeMeaning": "Morphologically Altered Structure",
                },
                "SegmentedPropertyTypeCodeSequence": {
                    "CodeValue": "M-01000",
                    "CodingSchemeDesignator": "SRT",
                    "CodeMeaning": "Tissue",
                },
                "recommendedDisplayRGBValue": [255, 0, 0],
            }
        )

    return {
        "ContentCreatorName": "Reader1",
        "ClinicalTrialSeriesID": "Session1",
        "ClinicalTrialTimePointID": "1",
        "SeriesDescription": series_description,
        "SeriesNumber": "300",
        "InstanceNumber": "1",
        "segmentAttributes": [segment_attributes],
        "ContentLabel": "SEGMENTATION",
        "ContentDescription": "SynthSEG",
        "ClinicalTrialCoordinatingCenterName": "SHH",
        "BodyPartExamined": "",
    }


def make_single_multilabel_dicomseg(
    *,
    label_volume_zyx: np.ndarray,
    reference_image: sitk.Image,
    source_images: List[pydicom.Dataset],
    series_description: str,
) -> pydicom.Dataset:
    """
    用 pydicom_seg.MultiClassWriter 產生單一 multi-label DICOM-SEG。
    """
    labels = np.unique(label_volume_zyx)
    labels = labels[labels != 0]
    if labels.size == 0:
        raise ValueError("NIfTI 只有背景（label=0），無法產生 DICOM-SEG")

    template_json = get_dicom_seg_template(series_description, [int(x) for x in labels.tolist()])
    template = pydicom_seg.template.from_dcmqi_metainfo(template_json)

    writer = pydicom_seg.MultiClassWriter(
        template=template,
        inplane_cropping=False,
        skip_empty_slices=True,
        skip_missing_segment=True,
    )

    seg_img = sitk.GetImageFromArray(label_volume_zyx.astype(np.uint16, copy=False))
    seg_img.CopyInformation(reference_image)
    return writer.write(seg_img, source_images)


def main() -> None:
    args = _build_parser().parse_args()
    path_dicom = Path(args.path_dicom)
    path_nii_str = args.path_synthseg_nii if args.path_synthseg_nii else args.path_nii
    path_nii = Path(path_nii_str)
    output_dir = Path(args.output_dir)

    if not path_nii.exists():
        raise FileNotFoundError(f"path_nii 不存在：{path_nii}")

    default_name = "Pred_CMB.dcm" if args.mode == "pred" else "SynthSEG_CMB.dcm"
    default_desc = "Pred_CMB" if args.mode == "pred" else "SynthSEG_CMB"
    output_name = args.output_name or default_name
    series_description = args.series_description or default_desc

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name

    image, _first_dcm, source_images = load_and_sort_dicom_series(path_dicom)
    seg_array = get_array_to_dcm_axcodes(path_nii)

    dcm_seg = make_single_multilabel_dicomseg(
        label_volume_zyx=seg_array,
        reference_image=image,
        source_images=source_images,
        series_description=series_description,
    )

    if out_path.exists():
        out_path.unlink()
    dcm_seg.save_as(str(out_path))
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()

