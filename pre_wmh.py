"""WMH 影像前處理：依據 SynthSEG 產生腦遮罩並套用到 T2FLAIR。"""

from __future__ import annotations

import logging
import os
from typing import Final

import nibabel as nib
import numpy as np

from nii_transforms import nii_img_replace

INPUT_T2: Final[str] = "T2FLAIR.nii.gz"
INPUT_SYNTHSEG: Final[str] = "SynthSEG_Brain.nii.gz"
OUTPUT_T2: Final[str] = "BET_T2FLAIR.nii.gz"
OUTPUT_MASK: Final[str] = "BET_mask.nii.gz"


def preprocess_wmh_images(path_process: str) -> bool:
    """
    針對 WMH 模型產生前處理結果。

    流程：
        1. 讀取 `SynthSEG.nii.gz` 產生二值腦遮罩。
        2. 遮罩套用到 `T2FLAIR.nii.gz` 產出 `BET_T2FLAIR.nii.gz`。
        3. 將遮罩輸出至 `BET_mask.nii.gz`。

    Args:
        path_process: 個案處理資料夾。

    Returns:
        bool: 成功為 True，失敗為 False。
    """

    synthseg_path = os.path.join(path_process, INPUT_SYNTHSEG)
    t2_path = os.path.join(path_process, INPUT_T2)
    bet_t2_path = os.path.join(path_process, OUTPUT_T2)
    bet_mask_path = os.path.join(path_process, OUTPUT_MASK)

    if not os.path.isfile(synthseg_path):
        logging.error("SynthSEG file missing: %s", synthseg_path)
        return False
    if not os.path.isfile(t2_path):
        logging.error("T2FLAIR file missing: %s", t2_path)
        return False

    try:
        synthseg_nii = nib.load(synthseg_path)
        synthseg_arr = np.asanyarray(synthseg_nii.dataobj)
        mask = np.zeros_like(synthseg_arr, dtype=np.uint8)
        mask[synthseg_arr > 0] = 1

        mask_nii = nii_img_replace(synthseg_nii, mask)
        nib.save(mask_nii, bet_mask_path)
        logging.info("Saved WMH mask: %s", bet_mask_path)

        t2_nii = nib.load(t2_path)
        t2_arr = np.asanyarray(t2_nii.dataobj)
        bet_t2_arr = t2_arr * mask
        bet_t2_nii = nii_img_replace(t2_nii, bet_t2_arr)
        nib.save(bet_t2_nii, bet_t2_path)
        logging.info("Saved BET T2FLAIR: %s", bet_t2_path)

        return True
    except Exception as exc:  # noqa: BLE001
        logging.error("preprocess_wmh_images failed: %s", exc)
        logging.error("Catch an exception.", exc_info=True)
        return False


