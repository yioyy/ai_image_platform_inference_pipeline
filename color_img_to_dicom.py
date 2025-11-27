"""Utilities to convert RGB numpy arrays into DICOM slices."""

from __future__ import annotations

import numpy as np


def color_img_to_dicom(dcm, img: np.ndarray, tag: str):
    """Replace pixel data in `dcm` with RGB `img` and tweak metadata."""
    y_i, x_i, _ = img.shape
    dcm.PixelData = img.tobytes()
    dcm[0x08, 0x0008].value = ["DERIVED", "SECONDARY", "OTHER"]
    dcm[0x08, 0x103E].value = tag
    dcm[0x28, 0x0002].value = 3
    dcm[0x28, 0x0004].value = "RGB"
    dcm.add_new(0x00280006, "US", "Planar Configuration")
    dcm[0x28, 0x0006].value = 0
    dcm[0x28, 0x0010].value = y_i
    dcm[0x28, 0x0011].value = x_i
    dcm[0x28, 0x0100].value = 8
    dcm[0x28, 0x0101].value = 8
    dcm[0x28, 0x0102].value = 7
    dcm[0x28, 0x0103].value = 0
    for tag_id in (0x0028, 0x1050), (0x0028, 0x1051):
        if tag_id in dcm:
            del dcm[tag_id]

    seriesiu = dcm[0x20, 0x000E].value
    dcm[0x20, 0x000E].value = f"{seriesiu}.{seriesiu}"
    sopiu = dcm[0x08, 0x0018].value
    dcm[0x08, 0x0018].value = f"{sopiu}.{sopiu}"
    return dcm


def color_img_to_dicom_combine(dcm, img: np.ndarray, tag: str, num: int):
    """Variant that appends `num` to Series/SOP instance UID."""
    y_i, x_i, _ = img.shape
    dcm.PixelData = img.tobytes()
    dcm[0x08, 0x0008].value = ["DERIVED", "SECONDARY", "OTHER"]
    dcm[0x08, 0x103E].value = tag
    dcm[0x28, 0x0002].value = 3
    dcm[0x28, 0x0004].value = "RGB"
    dcm.add_new(0x00280006, "US", "Planar Configuration")
    dcm[0x28, 0x0006].value = 0
    dcm[0x28, 0x0010].value = y_i
    dcm[0x28, 0x0011].value = x_i
    dcm[0x28, 0x0100].value = 8
    dcm[0x28, 0x0101].value = 8
    dcm[0x28, 0x0102].value = 7
    dcm[0x28, 0x0103].value = 0
    for tag_id in (0x0028, 0x1050), (0x0028, 0x1051):
        if tag_id in dcm:
            del dcm[tag_id]

    seriesiu = dcm[0x20, 0x000E].value
    dcm[0x20, 0x000E].value = f"{seriesiu}.{seriesiu}"
    sopiu = dcm[0x08, 0x0018].value
    dcm[0x08, 0x0018].value = f"{sopiu}.{num}"
    return dcm


