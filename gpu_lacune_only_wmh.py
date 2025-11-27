#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMH 推理腳本：呼叫 nnU-Net DeepLacune 模型並輸出 WMH 單類結果。
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time

import nibabel as nib
import numpy as np
import pynvml  # type: ignore
import tensorflow as tf  # type: ignore
from nnResUNet_long_BigBatch_cosine_AneDilate_classifier_test.gpu_nnUNet2D import (  # type: ignore # noqa: E501
    predict_from_raw_data,
)

CASE_PREFIX = "DeepLacune_00001"


def data_translate(img: np.ndarray, nii: nib.Nifti1Image) -> np.ndarray:
    """模仿 TensorFlow 版預處理的軸順序，供後處理使用。"""
    del nii  # 僅為保持介面一致
    img = np.swapaxes(img, 0, 1)
    img = np.flip(img, 0)
    img = np.flip(img, -1)
    return img


def data_translate_back(img: np.ndarray, nii: nib.Nifti1Image) -> np.ndarray:
    """將預測結果轉回原始方向。"""
    header = nii.header.copy()
    pixdim = header["pixdim"]
    if pixdim[0] > 0:
        img = np.flip(img, 1)
    img = np.flip(img, -1)
    img = np.flip(img, 0)
    img = np.swapaxes(img, 1, 0)
    return img


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def model_predict_wmh(
    path_code: str,
    path_process: str,
    path_nnunet_model: str,
    case_name: str,
    path_log: str,
    gpu_n: int,
) -> None:
    del path_code  # reserved for compatibility
    # 建立 log
    localt = time.localtime()
    time_str_short = (
        str(localt.tm_year) + str(localt.tm_mon).rjust(2, "0") + str(localt.tm_mday).rjust(2, "0")
    )
    log_file = os.path.join(path_log, f"{time_str_short}.log")
    _ensure_dir(path_log)
    if not os.path.isfile(log_file):
        with open(log_file, "a+", encoding="utf-8"):
            pass

    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode="a",
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    try:
        logging.info("!!! %s gpu_wmh call.", case_name)
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_n)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpum_rate = memory_info.used / memory_info.total

        if gpum_rate >= 0.6:
            logging.error("!!! %s Insufficient GPU Memory.", case_name)
            return

        gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
        if not gpus:
            raise RuntimeError("No GPU devices available")
        tf.config.experimental.set_visible_devices(devices=gpus[gpu_n], device_type="GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_n], True)

        bet_t2 = os.path.join(path_process, "BET_T2FLAIR.nii.gz")
        bet_mask = os.path.join(path_process, "BET_mask.nii.gz")
        if not os.path.isfile(bet_t2):
            raise FileNotFoundError(f"Missing BET_T2FLAIR: {bet_t2}")
        if not os.path.isfile(bet_mask):
            raise FileNotFoundError(f"Missing BET_mask: {bet_mask}")

        path_normimg = os.path.join(path_process, "Normalized_Image")
        path_brain = os.path.join(path_process, "Brain")
        _ensure_dir(path_normimg)
        _ensure_dir(path_brain)

        shutil.copy(bet_t2, os.path.join(path_normimg, f"{CASE_PREFIX}_0000.nii.gz"))
        shutil.copy(bet_mask, os.path.join(path_brain, f"{CASE_PREFIX}_0000.nii.gz"))

        start = time.time()
        predict_from_raw_data(
            path_normimg,
            path_brain,
            path_process,
            path_nnunet_model,
            (1,),
            0.25,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_gpu=True,
            verbose=True,
            save_probabilities=False,
            overwrite=False,
            checkpoint_name="checkpoint_final.pth",
            plans_json_name="plans.json",
            has_classifier_output=False,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=3,
            desired_gpu_index=gpu_n,
            batch_size=64,
        )
        logging.info("[Done WMH Inference] %.0f sec", time.time() - start)

        pred_path = os.path.join(path_process, f"{CASE_PREFIX}.nii.gz")
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"Prediction not found: {pred_path}")

        path_nnunet = os.path.join(path_process, "nnUNet")
        _ensure_dir(path_nnunet)

        pred_nii = nib.load(pred_path)
        pred_arr = np.asanyarray(pred_nii.dataobj)
        pred_arr = data_translate(pred_arr, pred_nii)

        if pred_arr.ndim == 4 and pred_arr.shape[0] > 1:
            wmh_prob = pred_arr[1]
            class_map = np.argmax(pred_arr, axis=0).astype(np.uint8)
        else:
            wmh_prob = pred_arr.astype(np.float32)
            class_map = (wmh_prob > 0.5).astype(np.uint8)

        wmh_mask = (class_map == 1).astype(np.uint8)
        wmh_prob_native = data_translate_back(wmh_prob, pred_nii).astype(np.float32)
        wmh_mask_native = data_translate_back(wmh_mask, pred_nii).astype(np.uint8)

        prob_nii = nib.nifti1.Nifti1Image(wmh_prob_native, pred_nii.affine, header=pred_nii.header)
        mask_nii = nib.nifti1.Nifti1Image(wmh_mask_native, pred_nii.affine, header=pred_nii.header)
        nib.save(prob_nii, os.path.join(path_nnunet, "Prob.nii.gz"))
        nib.save(mask_nii, os.path.join(path_nnunet, "Pred.nii.gz"))

        logging.info("!!! %s gpu_wmh finish.", case_name)
    except Exception as exc:  # noqa: BLE001
        logging.error("!!! %s gpu have error code: %s", case_name, exc)
        logging.error("Catch an exception.", exc_info=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WMH nnUNet inference")
    parser.add_argument("--path_code", type=str, required=True)
    parser.add_argument("--path_process", type=str, required=True)
    parser.add_argument("--path_nnunet_model", type=str, required=True)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--path_log", type=str, required=True)
    parser.add_argument("--gpu_n", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model_predict_wmh(
        path_code=args.path_code,
        path_process=args.path_process,
        path_nnunet_model=args.path_nnunet_model,
        case_name=args.case,
        path_log=args.path_log,
        gpu_n=args.gpu_n,
    )


