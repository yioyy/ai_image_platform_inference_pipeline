#!/usr/bin/env python3
"""Infarct visual report — three PNG montages, wrapped as a DICOM series.

Ported from the needFollowup implementation, which split this across
post_infarct.py (_generate_png_images) and after_run_infarct.py
(_export_png_reports and friends). Only the upload was platform-specific; the
report itself is a clinical deliverable and was dropped by mistake when the two
were separated.

Output stays under the working directory:
    <process_dir>/report/<patient_id>{,_noPred,_SynthSEG}.png
    <process_dir>/nnUNet/Dicom/Infarct AI Report/*.dcm

Nothing here is delivered to AI_INFERENCE_RESULT_PATH and nothing is declared in
reformatted_series. The RADAX worked examples carry no report series, so
publishing one would put a key in prediction.json that the platform has not
agreed to parse.

Two deviations from the original, both deliberate:

- Dback_v (the background value that gets flattened to the display minimum) is
  read from the same array the montage draws, not from the raw volume. The
  original took it from the raw array and compared it against the z-scored one,
  so the test never matched and the background was never suppressed.
- cv2 is not used for the PNG round-trip. matplotlib wrote the file; PIL reads
  it back in RGB directly, without the BGR detour.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("infarct.report")

ROWS = COLS = 256
REPORT_SERIES_DESC = "Infarct AI Report"


def _translate(arr, nii):
    """Reference display orientation. Same transform as post_infarct.py."""
    import numpy as np

    arr = np.swapaxes(arr, 0, 1)
    arr = np.flip(arr, 0)
    arr = np.flip(arr, -1)
    if nii.header["pixdim"][0] > 0:
        arr = np.flip(arr, 1)
    return arr


def _montage_slices(z_i: int):
    """Which slices go on each sheet, and how many columns they wrap at."""
    if z_i >= 21:
        return list(range(20, 0, -1)), list(range(18, 2, -1))
    if z_i >= 19:
        return list(range(z_i - 1, -1, -1)), list(range(18, 2, -1))
    order = list(range(z_i - 1, -1, -1))
    return order, order


def _canvas(dwi, k, slice_y, slice_x, y_i, x_i, back_v):
    import numpy as np

    show = np.zeros((ROWS, COLS))
    show[slice_y:slice_y + y_i, slice_x:slice_x + x_i] = dwi[:, :, k].copy()
    show[show == back_v] = np.min(show)
    rgb = ((show - np.min(show)) * 40).copy()
    rgb = np.stack([rgb] * 3, axis=-1)
    return np.clip(rgb, 0, 255).astype("uint8")


def _generate_pngs(out_dir: str, patient_id: str, pred, dwi_nor, territory,
                   labels_config: Dict[str, List[int]], colors: Dict[str, str],
                   shown: Dict[str, float], volume_ml: float, mean_adc: int) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    y_i, x_i, z_i = dwi_nor.shape
    slice_y, slice_x = int((ROWS - y_i) / 2), int((COLS - x_i) / 2)
    back_v = (dwi_nor[slice_y:slice_y + y_i, slice_x:slice_x + x_i, 0][2, 2]
              if y_i > 2 and x_i > 2 else 0)
    combine, combine_s = _montage_slices(z_i)

    if float(np.sum(pred)) > 0:
        caption = ("Infarction Volume: %s ml (mean ADC = %s)\n\n"
                   "(For Research Purpose Only)" % (volume_ml, mean_adc))
    else:
        caption = "Infarction Volume: 0 ml\n\n(For Research Purpose Only)"

    written = []
    plt.style.use("dark_background")

    def _sheet(name, slices, per_row, cell, paint):
        fig = plt.figure()
        plt.axis("off")
        plt.text(0.5, 0.1, caption, fontsize=15,
                 verticalalignment="center", horizontalalignment="center")
        for k, z in enumerate(slices):
            img = _canvas(dwi_nor, z, slice_y, slice_x, y_i, x_i, back_v)
            paint(img, z)
            ax = fig.add_axes([cell * (k % per_row), 0.9 - 0.2 * (k // per_row),
                               cell + 0.01, cell + 0.01])
            ax.imshow(img)
            ax.axis("off")
        return fig

    # 1. the images alone, for comparison against the overlay
    fig = _sheet("noPred", combine, 5, 0.2, lambda img, z: None)
    p = os.path.join(out_dir, "%s_noPred.png" % patient_id)
    plt.savefig(p, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close("all")
    written.append(p)

    # 2. the lesion in magenta
    def _paint_pred(img, z):
        ys, xs = np.where(pred[:, :, z] > 0)
        if len(ys):
            img[ys, xs] = (255, 0, 255)

    fig = _sheet("pred", combine, 5, 0.2, _paint_pred)
    p = os.path.join(out_dir, "%s.png" % patient_id)
    plt.savefig(p, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close("all")
    written.append(p)

    # 3. the involved territories, colour-coded, with a legend. Skipped when no
    # territory is involved -- an empty legend sheet tells a reader nothing.
    if shown:
        def _paint_terr(img, z):
            plane = territory[:, :, z]
            for region in shown:
                rgb = mcolors.to_rgb(colors.get(region, "#ffffff"))
                for lid in labels_config.get(region, []):
                    ys, xs = np.where(plane == lid)
                    if len(ys):
                        img[ys, xs] = tuple(int(c * 255) for c in rgb)

        fig = _sheet("synthseg", combine_s, 4, 0.16, _paint_terr)
        ax = fig.add_axes([0.165 * 4, 0.315 + 0.4 * (1 - len(shown) / 15),
                           0.015, 0.05 * len(shown)])
        for i, region in enumerate(shown):
            ax.hlines(1 - 0.03 * i, 0, 1, color=colors.get(region, "#ffffff"), linewidth=6)
            ax.text(1.3, 0.996 - 0.03 * i, "%s: %s ml" % (region, shown[region]),
                    fontsize=6, horizontalalignment="left")
        ax.axis("off")
        p = os.path.join(out_dir, "%s_SynthSEG.png" % patient_id)
        plt.savefig(p, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close("all")
        written.append(p)

    return written


def _sorted_dicoms(dwi_dir: str) -> List[str]:
    """DWI1000 instances in slice order, for use as report templates."""
    import pydicom

    out = []
    for f in os.listdir(dwi_dir):
        if not f.lower().endswith(".dcm"):
            continue
        path = os.path.join(dwi_dir, f)
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            idx = getattr(ds, "InstanceNumber", None)
            if idx is None:
                pos = getattr(ds, "ImagePositionPatient", [0, 0, 0])
                idx = float(pos[2]) if len(pos) >= 3 else 0
        except Exception:
            idx = 0
        out.append((idx, path))
    out.sort(key=lambda t: t[0])
    return [p for _, p in out]


def _as_dicom(png_path: str, template: str, export_dir: str, seq: int) -> Optional[str]:
    """Wrap one PNG as an RGB secondary-capture instance."""
    import numpy as np
    import pydicom
    from PIL import Image

    img = np.array(Image.open(png_path).convert("RGB"))
    ds = pydicom.dcmread(template)
    h, w, _ = img.shape

    ds.PixelData = img.tobytes()
    ds[0x08, 0x0008].value = ["DERIVED", "SECONDARY", "OTHER"]
    ds[0x08, 0x103E].value = REPORT_SERIES_DESC
    ds[0x28, 0x0002].value = 3            # SamplesPerPixel
    ds[0x28, 0x0004].value = "RGB"
    ds.add_new(0x00280006, "US", "Planar Configuration")
    ds[0x28, 0x0006].value = 0
    ds[0x28, 0x0010].value = h
    ds[0x28, 0x0011].value = w
    ds[0x28, 0x0100].value = 8            # BitsAllocated
    ds[0x28, 0x0101].value = 8            # BitsStored
    ds[0x28, 0x0102].value = 7            # HighBit
    ds[0x28, 0x0103].value = 0            # PixelRepresentation
    # Window settings describe the greyscale source and mean nothing for RGB;
    # left in place some viewers apply them and wash the report out.
    for tag in ((0x0028, 0x1050), (0x0028, 0x1051)):
        if tag in ds:
            del ds[tag]

    series_uid = ds[0x20, 0x000E].value
    ds[0x20, 0x000E].value = "%s.%s" % (series_uid, series_uid)
    sop_uid = ds[0x08, 0x0018].value
    ds[0x08, 0x0018].value = "%s.%s" % (sop_uid, seq)

    original = ds.get((0x0020, 0x0011), None)
    if original is not None and str(original.value).isdigit():
        ds[0x20, 0x0011].value = str(original.value) + "03"

    out = os.path.join(export_dir, "%s.dcm" % ds[0x08, 0x0018].value[-10:])
    ds.save_as(out)
    return out


def generate_report(process_dir: str, patient_id: str, lesions: List[Dict],
                    labels_config: Dict[str, List[int]],
                    colors: Dict[str, str]) -> Dict[str, List[str]]:
    """Write the report PNGs and their DICOM series under process_dir."""
    import nibabel as nib
    import numpy as np

    png_dir = os.path.join(process_dir, "report")
    dcm_dir = os.path.join(process_dir, "nnUNet", "Dicom", REPORT_SERIES_DESC)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(dcm_dir, exist_ok=True)

    dwi_nii = nib.load(os.path.join(process_dir, "DWI1000.nii.gz"))
    pred_nii = nib.load(os.path.join(process_dir, "nnUNet", "Pred.nii.gz"))
    terr_nii = nib.load(os.path.join(process_dir, "SynthSEG.nii.gz"))

    dwi = np.asanyarray(dwi_nii.dataobj).astype(np.float32)
    dwi_nor = _translate((dwi - dwi.mean()) / (dwi.std() or 1.0), dwi_nii)
    pred = _translate(np.asanyarray(pred_nii.dataobj).astype(np.int32), pred_nii)
    terr = _translate(np.asanyarray(terr_nii.dataobj).astype(np.int32), terr_nii)

    total = next((l for l in lesions if l["location"] == "Total"), None)
    shown = {l["location"]: l["volume_ml"] for l in lesions
             if l["location"] != "Total"}

    pngs = _generate_pngs(
        png_dir, patient_id, pred, dwi_nor, terr, labels_config, colors, shown,
        total["volume_ml"] if total else 0.0,
        total["mean_adc"] if total else 0)

    dwi_dcm_dir = os.path.join(process_dir, "nnUNet", "Dicom", "DWI1000")
    dcms = []
    if not os.path.isdir(dwi_dcm_dir):
        logger.warning("[report] no DWI1000 DICOM directory; PNGs written, "
                       "no report series produced")
        return {"png": pngs, "dcm": dcms}

    ordered = _sorted_dicoms(dwi_dcm_dir)
    if not ordered:
        logger.warning("[report] DWI1000 DICOM directory is empty")
        return {"png": pngs, "dcm": dcms}

    # Templates come from the tail of the stack, where the reference took them:
    # the last slices are the least likely to carry anatomy a reader needs.
    for seq, (png, offset) in enumerate(
            zip(pngs, (-1, -2, -3)), start=1):
        template = ordered[offset] if abs(offset) <= len(ordered) else ordered[-1]
        try:
            out = _as_dicom(png, template, dcm_dir, seq)
            if out:
                dcms.append(out)
        except Exception as exc:
            logger.warning("[report] %s -> DICOM failed: %s",
                           os.path.basename(png), exc)

    logger.info("[report] %d PNG, %d DICOM under %s",
                len(pngs), len(dcms), process_dir)
    return {"png": pngs, "dcm": dcms}
