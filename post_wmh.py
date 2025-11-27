"""WMH 後處理模組：產生統計、PNG 與 JSON 報告。"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import nibabel as nib
import numpy as np
import pydicom  # type: ignore

from code_ai.pipeline.dicomseg.schema.enum import ModelTypeEnum, SeriesTypeEnum

from nii_transforms import nii_img_replace


def data_translate(img: np.ndarray, nii: nib.Nifti1Image) -> np.ndarray:
    """將 NIfTI array 調整為 CNN 使用的軸順序。"""
    del nii
    img = np.swapaxes(img, 0, 1)
    img = np.flip(img, 0)
    img = np.flip(img, -1)
    return img


def data_translate_back(img: np.ndarray, nii: nib.Nifti1Image) -> np.ndarray:
    """將 CNN 輸出轉回原始方向。"""
    header = nii.header.copy()
    pixdim = header["pixdim"]
    if pixdim[0] > 0:
        img = np.flip(img, 1)
    img = np.flip(img, -1)
    img = np.flip(img, 0)
    img = np.swapaxes(img, 1, 0)
    return img


def create_json_report(
    json_path: str,
    patient_id: str,
    volume_ml: float,
    fazekas: int,
    report: str,
) -> None:
    payload = OrderedDict(
        PatientID=patient_id,
        volume=volume_ml,
        Fazekas=fazekas,
        Report=report,
        SeriesType=SeriesTypeEnum.T2FLAIR_AXI.value,
        ModelType=ModelTypeEnum.WMH.value,
    )
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def postprocess_wmh_results(
    patient_id: str,
    path_nnunet: str,
    path_code: str,
    path_output: str,
    path_json: str,
) -> bool:
    """
    處理 WMH nnUNet 推理結果。

    Args:
        patient_id: 病患 ID。
        path_nnunet: nnUNet 根目錄。
        path_code: 專案根目錄（讀取 dataset.json）。
        path_output: 專案輸出資料夾。
        path_json: JSON 匯出資料夾。

    Returns:
        bool: 成功返回 True。
    """

    path_nii = os.path.join(path_nnunet, "Image_nii")
    path_result = os.path.join(path_nnunet, "Result")
    os.makedirs(path_result, exist_ok=True)
    os.makedirs(path_output, exist_ok=True)
    os.makedirs(path_json, exist_ok=True)

    required = {
        "t2": os.path.join(path_nii, "T2FLAIR.nii.gz"),
        "synthseg": os.path.join(path_nii, "SynthSEG.nii.gz"),
        "pred": os.path.join(path_nii, "Pred.nii.gz"),
    }
    for name, path in required.items():
        if not os.path.exists(path):
            print(f"缺少必要檔案 {name}: {path}")
            return False

    t2_nii = nib.load(required["t2"])
    synthseg_nii = nib.load(required["synthseg"])
    pred_nii = nib.load(required["pred"])

    t2_array = data_translate(np.asanyarray(t2_nii.dataobj), t2_nii)
    synthseg_array = data_translate(np.asanyarray(synthseg_nii.dataobj), synthseg_nii)
    pred_mask = data_translate(np.asanyarray(pred_nii.dataobj), pred_nii)
    pred_mask = (pred_mask > 0).astype(np.uint8)

    voxel_ml = float(np.prod(t2_nii.header.get_zooms()) / 1000.0)
    wm_mask = (synthseg_array > 0).astype(np.uint8)
    synthseg_overlap = pred_mask * synthseg_array

    total_voxel = int(pred_mask.sum())
    total_volume_ml = round(total_voxel * voxel_ml, 1)

    brain_voxel = max(int(wm_mask.sum()), 1)
    percentage = (total_voxel / brain_voxel) * 100.0
    if total_voxel == 0:
        fazekas = 0
    elif percentage < 0.2923:
        fazekas = 1
    elif percentage > 1.4825:
        fazekas = 3
    else:
        fazekas = 2

    pred_native = data_translate_back(pred_mask, t2_nii).astype(np.uint8)
    synthseg_native = data_translate_back(synthseg_overlap, t2_nii).astype(np.int16)

    pred_native_nii = nii_img_replace(t2_nii, pred_native)
    synthseg_native_nii = nii_img_replace(t2_nii, synthseg_native)
    nib.save(pred_native_nii, os.path.join(path_result, "Pred.nii.gz"))
    nib.save(pred_native_nii, os.path.join(path_output, "Pred_WMH.nii.gz"))
    nib.save(synthseg_native_nii, os.path.join(path_result, "Pred_synthseg.nii.gz"))
    nib.save(synthseg_native_nii, os.path.join(path_output, "Pred_WMH_synthseg.nii.gz"))

    label_stats, labels_to_show = _compute_region_stats(
        path_code, synthseg_overlap, voxel_ml
    )

    report = _build_report_text(total_volume_ml, fazekas, label_stats)
    path_dcm_t2 = os.path.join(path_nnunet, "Dicom", "T2FLAIR_AXI")
    _export_pngs(
        patient_id=patient_id,
        mask=pred_mask,
        synthseg_overlap=synthseg_overlap,
        label_colors=labels_to_show,
        total_volume=total_volume_ml,
        fazekas=fazekas,
        path_result=path_result,
        path_dcm_t2=path_dcm_t2,
        fallback_array=t2_array,
    )

    path_json_out = os.path.join(path_nnunet, "JSON")
    os.makedirs(path_json_out, exist_ok=True)
    json_targets = [
        os.path.join(path_json, "Pred_WMH.json"),
        os.path.join(path_json_out, f"{patient_id}_platform_json.json"),
        os.path.join(path_output, "Pred_WMH.json"),
    ]
    for json_path in json_targets:
        create_json_report(json_path, patient_id, total_volume_ml, fazekas, report)

    print(
        f"WMH 後處理完成：體積 = {total_volume_ml} ml, Fazekas = {fazekas}, 病灶數 = {len(label_stats)}"
    )
    return True


def _compute_region_stats(
    path_code: str, synthseg_overlap: np.ndarray, voxel_ml: float
) -> Tuple[Dict[str, float], List[Tuple[str, float, str, int]]]:
    dataset_path = os.path.join(path_code, "dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as fp:
        dataset = json.load(fp)
    label_cfg = dataset.get("WMH_labels", {})
    palette_cfg = dataset.get("colorbar_15", {})

    region_volume: Dict[str, float] = {}
    for name, value in label_cfg.items():
        if name == "Background":
            continue
        mask = synthseg_overlap == value
        if not np.any(mask):
            continue
        volume = round(float(mask.sum() * voxel_ml), 1)
        if volume >= 0.1:
            region_volume[name] = volume

    ordered = sorted(region_volume.items(), key=lambda item: item[1], reverse=True)
    limited = ordered[:15]

    if isinstance(palette_cfg, dict):
        palette = list(palette_cfg.values())
    else:
        palette = list(palette_cfg)
    if not palette:
        palette = ["magenta"]

    color_assignments: List[str] = []
    count = max(1, len(limited))
    if len(palette) >= count:
        step = max(1, len(palette) // count)
        color_assignments = [palette[(i * step) % len(palette)] for i in range(count)]
    else:
        color_assignments = [palette[i % len(palette)] for i in range(count)]

    label_colors: List[Tuple[str, float, str, int]] = []
    for (name, volume), color_value in zip(limited, color_assignments):
        if not isinstance(color_value, str):
            color_value = "magenta"
        label_colors.append((name, volume, color_value, label_cfg[name]))
    return region_volume, label_colors


def _build_report_text(
    total_volume: float,
    fazekas: int,
    region_stats: Dict[str, float],
) -> str:
    severity = {0: "no", 1: "mild", 2: "moderate", 3: "severe"}.get(fazekas, "mild")
    text = f"T2 hyperintense lesions, estimated about {total_volume} ml, "
    if region_stats:
        text += "involving "
        text += ", ".join(label.lower() for label in region_stats.keys()) + ", "
    text += f"suggestive of {severity} white matter hyperintensity, Fazekas {fazekas}.<br><br>"
    text += "===================================================================================================<br>"
    text += f"Total Volume: {total_volume} ml, Fazekas grade {fazekas}.<br>"
    for name, vol in region_stats.items():
        text += f"{name}: {vol} ml<br>"
    return text


def _select_slices(z_dim: int) -> List[int]:
    if z_dim >= 20:
        return list(np.linspace(0, z_dim - 1, 20, dtype=int))
    return list(range(z_dim))


def _export_pngs(
    patient_id: str,
    mask: np.ndarray,
    synthseg_overlap: np.ndarray,
    label_colors: List[Tuple[str, float, str, int]],
    total_volume: float,
    fazekas: int,
    path_result: str,
    path_dcm_t2: str,
    fallback_array: np.ndarray,
) -> None:
    os.makedirs(path_result, exist_ok=True)
    if os.path.isdir(path_dcm_t2):
        try:
            _export_pngs_from_dicoms(
                patient_id,
                mask,
                synthseg_overlap,
                label_colors,
                total_volume,
                fazekas,
                path_result,
                path_dcm_t2,
            )
            return
        except Exception as exc:  # noqa: BLE001
            logging.warning("WMH PNG DICOM export failed (%s). Falling back to array rendering.", exc)
    _export_pngs_simple(
        patient_id,
        fallback_array,
        mask,
        synthseg_overlap,
        label_colors,
        total_volume,
        fazekas,
        path_result,
    )


def _export_pngs_simple(
    patient_id: str,
    t2_array: np.ndarray,
    mask: np.ndarray,
    synthseg_overlap: np.ndarray,
    label_colors: List[Tuple[str, float, str, int]],
    total_volume: float,
    fazekas: int,
    path_result: str,
) -> None:
    slices = _select_slices(t2_array.shape[-1])

    def _base_image(slice_idx: int) -> np.ndarray:
        base = t2_array[:, :, slice_idx]
        base = base - np.min(base)
        if np.max(base) > 0:
            base = base / np.max(base)
        rgb = np.stack([base] * 3, axis=-1)
        return (rgb * 255).astype(np.uint8)

    fig, axes = _make_grid(len(slices), cols=5, title="(For Research Purpose Only)")
    for idx, slice_idx in enumerate(slices):
        ax = axes[idx]
        ax.imshow(_base_image(slice_idx))
        ax.axis("off")
    fig.savefig(os.path.join(path_result, f"{patient_id}_noPred.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    fig, axes = _make_grid(
        len(slices),
        cols=5,
        title=f"Volume: {total_volume} ml, Fazekas grade {fazekas} \n\n(For Research Purpose Only)",
    )
    for idx, slice_idx in enumerate(slices):
        ax = axes[idx]
        overlay = _base_image(slice_idx)
        mask_idx = mask[:, :, slice_idx] > 0
        overlay[mask_idx] = [255, 0, 255]
        ax.imshow(overlay)
        ax.axis("off")
    fig.savefig(os.path.join(path_result, f"{patient_id}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    if label_colors:
        fig, axes = _make_grid(
            len(slices),
            cols=4,
            title=f"Volume: {total_volume} ml, Fazekas grade {fazekas} \n\n(For Research Purpose Only)",
        )
        color_map = {}
        for name, _, color, _ in label_colors:
            color_value = color if isinstance(color, str) else "magenta"
            color_map[name] = mcolors.to_rgb(color_value)
        for idx, slice_idx in enumerate(slices):
            ax = axes[idx]
            overlay = _base_image(slice_idx)
            slice_labels = synthseg_overlap[:, :, slice_idx]
            for name, _, _, label_value in label_colors:
                rgb = color_map[name]
                mask_idx = slice_labels == label_value
                if mask_idx.any():
                    overlay[mask_idx] = np.array(rgb) * 255
            ax.imshow(overlay)
            ax.axis("off")
        fig.savefig(os.path.join(path_result, f"{patient_id}_SynthSEG.png"), bbox_inches="tight", dpi=200)
        plt.close(fig)


def _export_pngs_from_dicoms(
    patient_id: str,
    mask: np.ndarray,
    synthseg_overlap: np.ndarray,
    label_colors: List[Tuple[str, float, str, int]],
    total_volume: float,
    fazekas: int,
    path_result: str,
    path_dcm_t2: str,
) -> None:
    ROWS, COLS = 512, 512
    files = sorted(
        [
            f
            for f in os.listdir(path_dcm_t2)
            if f.lower().endswith(".dcm")
        ]
    )
    if not files:
        raise FileNotFoundError("T2FLAIR DICOM 檔案不存在")

    dicoms = [pydicom.dcmread(os.path.join(path_dcm_t2, f)) for f in files]
    positions = []
    for ds in dicoms:
        pos = getattr(ds, "SliceLocation", None)
        if pos is None:
            pos = getattr(ds, "InstanceNumber", dicoms.index(ds))
        positions.append(float(pos))
    sort_idx = np.argsort(positions)[::-1]

    combine_num, combine_s_num = _determine_slice_indices(len(files))

    def _render_slice(ds):
        arr = ds.pixel_array.astype(np.float32)
        F_show = np.zeros((ROWS, COLS), dtype=np.float32)
        height, width = arr.shape
        slice_y = int((ROWS - height) / 2)
        slice_x = int((COLS - width) / 2)
        F_show[slice_y:slice_y + height, slice_x:slice_x + width] = arr
        return F_show

    def _prep_rgb(F_show):
        F_predict = F_show.copy()
        if F_predict.max() > 0:
            F_predict = (F_predict / np.max(F_predict)) * 355
        F_predict = np.clip(F_predict, 0, 255).astype(np.uint8)
        F_predict = np.stack([F_predict] * 3, axis=-1)
        return F_predict

    # Figure 1: no overlay
    fig = plt.figure()
    plt.style.use("dark_background")
    plt.axis("off")
    plt.text(0.5, 0.1, "\n\n(For Research Purpose Only)", fontsize=15, ha="center", va="center")
    for k, idx in enumerate(combine_num):
        dcm = dicoms[sort_idx[min(idx, len(sort_idx) - 1)]]
        F_rgb = _prep_rgb(_render_slice(dcm))
        ax = fig.add_axes([0 + 0.2 * (k % 5), 0.9 - 0.2 * (k // 5), 0.2, 0.2])
        ax.imshow(F_rgb)
        ax.axis("off")
    fig.savefig(os.path.join(path_result, f"{patient_id}_noPred.png"), bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)

    # Figure 2: mask overlay
    fig = plt.figure()
    plt.style.use("dark_background")
    plt.axis("off")
    plt.text(
        0.5,
        0.1,
        f"Volume: {total_volume} ml, Fazekas grade {fazekas} \n\n(For Research Purpose Only)",
        fontsize=15,
        ha="center",
        va="center",
    )
    for k, idx in enumerate(combine_num):
        dcm = dicoms[sort_idx[min(idx, len(sort_idx) - 1)]]
        F_rgb = _prep_rgb(_render_slice(dcm))
        slice_idx = min(idx, mask.shape[2] - 1)
        overlay = mask[:, :, slice_idx] > 0
        F_rgb[overlay] = [255, 0, 255]
        ax = fig.add_axes([0 + 0.2 * (k % 5), 0.9 - 0.2 * (k // 5), 0.2, 0.2])
        ax.imshow(F_rgb)
        ax.axis("off")
    fig.savefig(os.path.join(path_result, f"{patient_id}.png"), bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)

    # Figure 3: SynthSEG overlay
    if label_colors:
        fig = plt.figure()
        plt.style.use("dark_background")
        plt.axis("off")
        plt.text(
            0.5,
            0.1,
            f"Volume: {total_volume} ml, Fazekas grade {fazekas} \n\n(For Research Purpose Only)",
            fontsize=15,
            ha="center",
            va="center",
        )
        color_map = {}
        for name, _, color, _ in label_colors:
            color_value = color if isinstance(color, str) else "magenta"
            color_map[name] = mcolors.to_rgb(color_value)

        for k, idx in enumerate(combine_s_num):
            dcm = dicoms[sort_idx[min(idx, len(sort_idx) - 1)]]
            F_rgb = _prep_rgb(_render_slice(dcm))
            slice_idx = min(idx, synthseg_overlap.shape[2] - 1)
            slice_labels = synthseg_overlap[:, :, slice_idx]
            for name, _, _, label_value in label_colors:
                rgb = color_map[name]
                mask_idx = slice_labels == label_value
                if mask_idx.any():
                    F_rgb[mask_idx] = np.array(rgb) * 255
            ax = fig.add_axes([0 + 0.16 * (k % 4), 0.9 - 0.2 * (k // 4), 0.21, 0.21])
            ax.imshow(F_rgb)
            ax.axis("off")

        max_labels = max(1, min(15, len(label_colors)))
        legend_height = 0.05 * max_labels
        legend_y = 0.315 + 0.4 * (1 - len(label_colors) / 15)
        legend_ax = fig.add_axes([0.66, legend_y, 0.015, legend_height])
        legend_ax.axis("off")
        for idx, (name, volume, color_value, _) in enumerate(label_colors):
            rgb = color_map[name]
            y = 1 - 0.03 * idx
            legend_ax.hlines(y, 0, 1, color=rgb, linewidth=6)
            legend_ax.text(1.05, y, f"{name}: {volume} ml", fontsize=6, ha="left", va="center")

        fig.savefig(os.path.join(path_result, f"{patient_id}_SynthSEG.png"), bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)


def _make_grid(total_slices: int, cols: int, title: str) -> Tuple[plt.Figure, List[plt.Axes]]:
    rows = int(np.ceil(total_slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(title, fontsize=12)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for ax in axes:
        ax.axis("off")
    return fig, axes


def _determine_slice_indices(total_slices: int) -> Tuple[List[int], List[int]]:
    default_primary = [1, 2, 3, 4, 5, 6, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    default_secondary = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    def clamp_list(values: List[int]) -> List[int]:
        if total_slices <= 0:
            return []
        return [min(v, total_slices - 1) for v in values if v < total_slices]

    if total_slices >= 21:
        primary = clamp_list(default_primary)
        secondary = clamp_list(default_secondary)
    elif total_slices >= 19:
        primary = list(range(total_slices))
        secondary = clamp_list(default_secondary)
    else:
        primary = list(range(total_slices))
        secondary = list(range(min(total_slices, 16)))

    if not secondary:
        secondary = primary[:16]
    return primary, secondary


