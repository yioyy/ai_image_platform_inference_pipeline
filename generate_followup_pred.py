"""
Generate follow-up merged prediction NIfTI.

輸入：
- baseline Pred_<model>.nii.gz
- followup Pred_<model>_registration.nii.gz（followup 對位到 baseline 空間）

輸出（同 baseline 空間）：
- Pred_<model>_followup.nii.gz

標籤定義（固定）：
- 1 = new           (baseline=1, followup=0)
- 2 = stable        (baseline=1, followup=1)
- 3 = disappeared   (baseline=0, followup=1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def generate_followup_pred(
    baseline_pred_file: Path,
    followup_reg_pred_file: Path,
    output_file: Path,
) -> Path:
    baseline_img, baseline_mask = _load_as_binary_mask(baseline_pred_file)
    _, followup_mask = _load_as_binary_mask(followup_reg_pred_file)

    if baseline_mask.shape != followup_mask.shape:
        raise ValueError(
            "baseline 與 followup(reg) shape 不一致："
            f"{baseline_mask.shape} vs {followup_mask.shape}"
        )

    merged = np.zeros(baseline_mask.shape, dtype=np.uint8)
    merged[(baseline_mask == 1) & (followup_mask == 0)] = 1
    merged[(baseline_mask == 1) & (followup_mask == 1)] = 2
    merged[(baseline_mask == 0) & (followup_mask == 1)] = 3

    out_img = nib.Nifti1Image(merged, baseline_img.affine, baseline_img.header)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_file))
    return output_file


def _load_as_binary_mask(path: Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 NIfTI 檔案：{path}")
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    mask = (data > 0).astype(np.uint8)
    return img, mask


