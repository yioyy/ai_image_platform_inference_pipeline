from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pydicom
import nibabel as nib


OverlapMode = Literal["max", "overwrite"]


class MissingDicomHeaderFieldsError(ValueError):
    def __init__(self, missing_fields: List[str]):
        super().__init__("series_headers 缺少必要欄位：" + ", ".join(missing_fields))
        self.missing_fields = missing_fields


@dataclass(frozen=True)
class SeriesSlice:
    sop_instance_uid: str
    ipp_lps: np.ndarray  # (3,)
    iop_lps: np.ndarray  # (6,)
    projection: float


def _as_float_array(value: Any, *, length: int, field_name: str) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) == length:
        return np.asarray([float(x) for x in value], dtype=np.float64)
    if isinstance(value, str):
        # allow "a\\b\\c" or "a,b,c"
        raw = value.replace("\\", ",")
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        if len(parts) == length:
            return np.asarray([float(x) for x in parts], dtype=np.float64)
    raise ValueError(f"{field_name} 格式不正確（需要長度 {length} 的數值序列）：{value!r}")


def _get_any(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _get_tag_like(d: Dict[str, Any], tag_keys: Sequence[str]) -> Any:
    """
    兼容 Orthanc tags 格式：可能是 '0020,0037' 或 '00200037' 或 nested。
    """
    for k in tag_keys:
        if k in d:
            return d[k]
        if k.replace(",", "") in d:
            return d[k.replace(",", "")]
    tags = d.get("Tags")
    if isinstance(tags, dict):
        for k in tag_keys:
            if k in tags:
                return tags[k]
            if k.replace(",", "") in tags:
                return tags[k.replace(",", "")]
    return None


def _extract_required_fields(header: Dict[str, Any]) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    回傳：(SOPInstanceUID, IPP, IOP, PixelSpacing, Rows, Columns)
    """
    sop = _get_any(header, ["SOPInstanceUID", "sop_instance_uid"]) or _get_tag_like(header, ["0008,0018"])
    ipp = _get_any(header, ["ImagePositionPatient", "image_position_patient"]) or _get_tag_like(header, ["0020,0032"])
    iop = _get_any(header, ["ImageOrientationPatient", "image_orientation_patient"]) or _get_tag_like(header, ["0020,0037"])
    ps = _get_any(header, ["PixelSpacing", "pixel_spacing"]) or _get_tag_like(header, ["0028,0030"])
    rows = _get_any(header, ["Rows", "rows"]) or _get_tag_like(header, ["0028,0010"])
    cols = _get_any(header, ["Columns", "columns"]) or _get_tag_like(header, ["0028,0011"])

    missing: List[str] = []
    if not sop:
        missing.append("SOPInstanceUID")
    if ipp is None:
        missing.append("ImagePositionPatient")
    if iop is None:
        missing.append("ImageOrientationPatient")
    if ps is None:
        missing.append("PixelSpacing")
    if rows is None:
        missing.append("Rows")
    if cols is None:
        missing.append("Columns")
    if missing:
        raise MissingDicomHeaderFieldsError(missing)

    sop_uid = str(sop)
    ipp_arr = _as_float_array(ipp, length=3, field_name="ImagePositionPatient")
    iop_arr = _as_float_array(iop, length=6, field_name="ImageOrientationPatient")
    ps_arr = _as_float_array(ps, length=2, field_name="PixelSpacing")
    r = int(rows)
    c = int(cols)
    return sop_uid, ipp_arr, iop_arr, ps_arr, r, c


def build_series_geometry_from_headers(series_headers: List[Dict[str, Any]]) -> Tuple[List[SeriesSlice], np.ndarray, int, int]:
    """
    只用 per-instance header 建立 slice 排序與 RAS affine。

    回傳：
    - slices（依 projection 排序）
    - affine_ras（4x4）
    - rows, cols
    """
    if not series_headers:
        raise ValueError("series_headers 為空")

    parsed: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for h in series_headers:
        if not isinstance(h, dict):
            raise ValueError("series_headers 內元素必須是 dict")
        parsed.append(_extract_required_fields(h))

    # 使用第一張的幾何（假設同一序列一致）
    _, _, iop0, ps0, rows, cols = parsed[0]
    row_dir = iop0[:3]
    col_dir = iop0[3:]
    normal = np.cross(row_dir, col_dir)
    n_norm = float(np.linalg.norm(normal))
    if n_norm == 0.0:
        raise ValueError("ImageOrientationPatient 無法形成有效法向量（cross=0）")
    normal = normal / n_norm

    slices: List[SeriesSlice] = []
    for sop_uid, ipp, iop, _, _, _ in parsed:
        proj = float(np.dot(normal, ipp))
        slices.append(SeriesSlice(sop_instance_uid=sop_uid, ipp_lps=ipp, iop_lps=iop, projection=proj))

    slices.sort(key=lambda s: s.projection)

    # slice spacing：用投影差的中位數（更穩定）
    if len(slices) >= 2:
        diffs = np.diff([s.projection for s in slices])
        diffs = np.asarray([abs(float(x)) for x in diffs if float(x) != 0.0], dtype=np.float64)
        spacing = float(np.median(diffs)) if diffs.size else 1.0
    else:
        spacing = 1.0

    # 以最小 projection 的 slice 作為 origin
    origin_lps = slices[0].ipp_lps.astype(np.float64, copy=False)

    # DICOM: PixelSpacing = [row_spacing, col_spacing]
    row_spacing = float(ps0[0])
    col_spacing = float(ps0[1])

    # nibabel array axis0=i(rows) 沿 column direction；axis1=j(cols) 沿 row direction
    axis_i_lps = col_dir * row_spacing
    axis_j_lps = row_dir * col_spacing
    axis_k_lps = normal * spacing

    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = axis_i_lps
    affine_lps[:3, 1] = axis_j_lps
    affine_lps[:3, 2] = axis_k_lps
    affine_lps[:3, 3] = origin_lps

    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)
    affine_ras = lps_to_ras @ affine_lps
    return slices, affine_ras, rows, cols


@dataclass(frozen=True)
class SegFrame:
    referenced_sop_instance_uid: str
    segment_number: int
    mask_2d: np.ndarray  # shape (rows, cols), binary or 0/1


def assemble_label_volume(
    *,
    sorted_slices: List[SeriesSlice],
    rows: int,
    cols: int,
    frames: Iterable[SegFrame],
    overlap_mode: OverlapMode = "max",
) -> np.ndarray:
    """
    以 SOPInstanceUID 對應 slice index，組出 label volume（shape: rows, cols, num_slices）。

    - 支援 skip_empty_slices：沒出現的 slice 會維持 0
    - 多 label：以 segment_number 作為 label 值
    """
    sop_to_idx = {s.sop_instance_uid: i for i, s in enumerate(sorted_slices)}
    vol = np.zeros((rows, cols, len(sorted_slices)), dtype=np.uint16)

    for fr in frames:
        idx = sop_to_idx.get(fr.referenced_sop_instance_uid)
        if idx is None:
            continue
        mask = np.asarray(fr.mask_2d)
        if mask.shape != (rows, cols):
            raise ValueError(f"SEG frame mask shape 不一致：{mask.shape} vs {(rows, cols)}")
        seg_val = int(fr.segment_number)
        if seg_val <= 0:
            continue

        if overlap_mode == "overwrite":
            vol[:, :, idx][mask > 0] = seg_val
            continue
        # overlap_mode == "max"
        cur = vol[:, :, idx]
        m = mask > 0
        cur[m] = np.maximum(cur[m], seg_val)

    return vol


def _extract_frame_sop_uid(frame_fg: Any) -> str:
    # 常見路徑：DerivationImageSequence -> SourceImageSequence -> ReferencedSOPInstanceUID
    try:
        di = frame_fg.DerivationImageSequence[0]
        src = di.SourceImageSequence[0]
        uid = getattr(src, "ReferencedSOPInstanceUID", "") or ""
        return str(uid)
    except Exception:
        return ""


def _extract_frame_segment_number(frame_fg: Any) -> int:
    try:
        sid = frame_fg.SegmentIdentificationSequence[0]
        return int(getattr(sid, "ReferencedSegmentNumber", 0) or 0)
    except Exception:
        return 0


def _iter_seg_frames_from_dicomseg(ds: pydicom.Dataset) -> Iterable[SegFrame]:
    pixel = ds.pixel_array  # (nframes, rows, cols) for binary SEG is typical
    if pixel.ndim != 3:
        raise ValueError(f"DICOM-SEG pixel_array 維度不支援：ndim={pixel.ndim}")

    # 保證是 0/1
    if getattr(ds, "SegmentationType", "").upper() == "FRACTIONAL":
        pixel = (pixel > 0).astype(np.uint8)
    else:
        pixel = (pixel > 0).astype(np.uint8)

    pffg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if not pffg:
        raise ValueError("DICOM-SEG 缺少 PerFrameFunctionalGroupsSequence")
    if len(pffg) != pixel.shape[0]:
        raise ValueError("DICOM-SEG PerFrameFunctionalGroupsSequence 與 pixel frames 數量不一致")

    for i in range(pixel.shape[0]):
        fg = pffg[i]
        sop_uid = _extract_frame_sop_uid(fg)
        seg_num = _extract_frame_segment_number(fg)
        yield SegFrame(referenced_sop_instance_uid=sop_uid, segment_number=seg_num, mask_2d=pixel[i])


def load_series_headers_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "series_headers" in data and isinstance(data["series_headers"], list):
        return data["series_headers"]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise ValueError("series_headers.json 格式不支援（需為 list[dict] 或 {'series_headers': list[dict]}）")


def synthseg_dicomseg_to_nifti(
    *,
    dicomseg_path: Union[str, Path],
    series_headers: List[Dict[str, Any]],
    out_nii_path: Union[str, Path],
    overlap_mode: OverlapMode = "max",
) -> Path:
    """
    入口：DICOM-SEG + per-instance headers → SynthSEG NIfTI。
    """
    ds = pydicom.dcmread(str(dicomseg_path))
    sorted_slices, affine_ras, rows, cols = build_series_geometry_from_headers(series_headers)
    frames = _iter_seg_frames_from_dicomseg(ds)
    vol = assemble_label_volume(sorted_slices=sorted_slices, rows=rows, cols=cols, frames=frames, overlap_mode=overlap_mode)

    img = nib.Nifti1Image(vol.astype(np.uint16, copy=False), affine_ras)
    out = Path(out_nii_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out))
    return out

