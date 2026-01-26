# -*- coding: utf-8 -*-
"""
讀取指定資料夾下的 DICOM，依照 pipeline 內計算 projection 的方式重排切片順序，
再依序讀取每張 DICOM header，最後把 header list 輸出成 JSON。

排序邏輯（與 `code/util_aneurysm.py` 一致）：
- IOP = ImageOrientationPatient (0020,0037)
- IPP = ImagePositionPatient    (0020,0032)
- normal = cross(IOP[:3], IOP[3:])
- projection = dot(IPP, normal)
- 依 projection 由小到大排序

輸出格式（依你的需求）：
payload["headers"][i] = {
  "file": "...",
  "projection": <float|null>,
  "sop_instance_uid": "...",
  "header": { ...該張 dicom 的所有 header... }
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset


# ====== 你可以直接改這兩個路徑 ======
path_dicom = r"C:\Users\user\Desktop\orthanc_combine_code\目前pipeline版本\process\Deep_FollowUp\00130846_20220105_MR_E211126000320\dicom\followup"
output = r"C:\Users\user\Desktop\orthanc_combine_code\目前pipeline版本\process\Deep_FollowUp\00130846_20220105_MR_E211126000320\followup"


@dataclass(frozen=True)
class DicomSortKey:
    projection: Optional[float]
    instance_number: Optional[int]
    filename: str


def _as_float_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.array(value, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _calc_projection(ds: Dataset) -> Optional[float]:
    """
    與 `code/util_aneurysm.py` 相同的 projection 計算。
    若缺少必要 tag，回傳 None。
    """
    iop = _as_float_array(getattr(ds, "ImageOrientationPatient", None))
    ipp = _as_float_array(getattr(ds, "ImagePositionPatient", None))
    if iop is None or ipp is None:
        return None
    if iop.size < 6 or ipp.size < 3:
        return None

    normal = np.cross(iop[0:3], iop[3:6])
    try:
        return float(np.dot(ipp[0:3], normal))
    except Exception:
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def find_dicom_files(root: str | Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"path_dicom 不存在：{root}")
    return [p for p in root.rglob("*") if p.is_file()]


def load_and_sort_dicom_files(path_dicom_dir: str | Path) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """
    回傳：
    - sorted_files: 依 projection（小到大）排序後的檔案路徑
    - sort_debug:  每張切片的排序資訊（projection/InstanceNumber 等）
    """
    candidates = find_dicom_files(path_dicom_dir)
    if not candidates:
        return [], []

    items: List[Tuple[DicomSortKey, Path]] = []
    debug: List[Dict[str, Any]] = []

    for p in candidates:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue

        projection = _calc_projection(ds)
        instance_number = _safe_int(getattr(ds, "InstanceNumber", None))
        sop_uid = _safe_str(getattr(ds, "SOPInstanceUID", ""))

        key = DicomSortKey(
            projection=projection,
            instance_number=instance_number,
            filename=p.name,
        )
        items.append((key, p))
        debug.append(
            {
                "file": str(p),
                "projection": projection,
                "instance_number": instance_number,
                "sop_instance_uid": sop_uid,
                "series_instance_uid": _safe_str(getattr(ds, "SeriesInstanceUID", "")),
            }
        )

    if not items:
        return [], []

    def sort_key(x: Tuple[DicomSortKey, Path]) -> Tuple[int, float, int, str]:
        k, _ = x
        proj_missing = 1 if k.projection is None else 0
        proj_value = float("inf") if k.projection is None else float(k.projection)
        inst_value = 2**31 - 1 if k.instance_number is None else int(k.instance_number)
        return (proj_missing, proj_value, inst_value, k.filename)

    items_sorted = sorted(items, key=sort_key)
    return [p for _, p in items_sorted], debug


def dicom_header_to_json_dict(ds: Dataset) -> Dict[str, Any]:
    """
    使用 pydicom 的 DICOM JSON Model，確保可被 json.dump 序列化，且包含所有 header。
    """
    return ds.to_json_dict()


def export_dicom_headers(path_dicom_dir: str | Path, output_dir: str | Path) -> Path:
    sorted_files, sort_debug = load_and_sort_dicom_files(path_dicom_dir)
    if not sorted_files:
        raise RuntimeError(f"在 {path_dicom_dir} 找不到可讀取的 DICOM")

    headers: List[Dict[str, Any]] = []
    for p in sorted_files:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        projection = _calc_projection(ds)
        sop_uid = _safe_str(getattr(ds, "SOPInstanceUID", ""))

        # 欄位順序依你的需求：projection -> sop_instance_uid -> header
        headers.append(
            {
                "file": str(p),
                "projection": projection,
                "sop_instance_uid": sop_uid,
                "header": dicom_header_to_json_dict(ds),
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dicom_headers.json"

    payload = {
        "path_dicom": str(Path(path_dicom_dir)),
        "count": len(headers),
        "sorted_debug": sort_debug,
        "headers": headers,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


def main() -> None:
    out_path = export_dicom_headers(path_dicom, output)
    print(f"[OK] 已輸出：{out_path}")


if __name__ == "__main__":
    main()

