"""WMH 後處理：統計、JSON 生成與上傳。"""

from __future__ import annotations

import importlib
import logging
import os
import pathlib
import sys
import time
from datetime import datetime
from typing import List, Optional

import cv2  # type: ignore
import pydicom

from color_img_to_dicom import color_img_to_dicom_combine
from util_aneurysm import orthanc_zip_upload, upload_json_aiteam
from wmh_pipeline import WMHPipeline


def after_run_wmh(path_nnunet: str, path_output: str, patient_id: str, path_code: str) -> bool:
    del path_output  # 尚未使用，預留介面
    try:
        _configure_logging(path_code)
        start_time = time.time()
        logging.info("after_run_wmh start: %s", patient_id)

        path_result = os.path.join(path_nnunet, "Result")
        path_dcm = os.path.join(path_nnunet, "Dicom")
        path_json_out = os.path.join(path_nnunet, "JSON")
        path_zip = os.path.join(path_nnunet, "Dicom_zip")
        path_excel = os.path.join(path_nnunet, "excel")
        path_image_nii = os.path.join(path_nnunet, "Image_nii")

        if not os.path.isdir(path_result):
            raise FileNotFoundError(f"缺少 Result 資料夾: {path_result}")
        if not os.path.isdir(path_dcm):
            raise FileNotFoundError(f"缺少 Dicom 資料夾: {path_dcm}")

        for folder in (path_json_out, path_zip, path_excel):
            os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(path_dcm, "Dicom-Seg"), exist_ok=True)

        pipeline = WMHPipeline(
            path_dcm=path_dcm,
            path_result=path_result,
            path_excel=path_excel,
            patient_id=patient_id,
            dataset_path=os.path.join(path_code, "dataset.json"),
            path_image_nii=path_image_nii,
        )
        pipeline.run_all()

        make_wmh_pred_json = _load_make_pred_json(path_code)
        json_path = make_wmh_pred_json(
            _id=patient_id,
            path_root=pathlib.Path(path_nnunet),
            group_id=56,
        )

        os.makedirs(os.path.join(path_dcm, "T2FLAIR AI Report"), exist_ok=True)
        _export_png_reports(path_result, path_dcm, patient_id)

        try:
            orthanc_zip_upload(path_dcm, path_zip, ["T2FLAIR_AXI", "Dicom-Seg", "T2FLAIR AI Report"])
        except Exception as exc:  # noqa: BLE001
            logging.warning("上傳 DICOM 失敗: %s", exc)

        try:
            upload_json_aiteam(str(json_path))
        except Exception as exc:  # noqa: BLE001
            logging.warning("上傳 JSON 失敗: %s", exc)

        logging.info("after_run_wmh done: %s (%.0f sec)", patient_id, time.time() - start_time)
        return True
    except Exception as exc:  # noqa: BLE001
        logging.error("after_run_wmh 發生錯誤: %s", exc)
        logging.error("Catch an exception.", exc_info=True)
        return False


def _configure_logging(path_code: str) -> str:
    """將 log 寫到專案既定位置：<project_root>/log/YYYYMMDD.log（與其他 pipeline 一致）。"""
    project_root = pathlib.Path(path_code).resolve().parent
    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    time_str_short = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{time_str_short}.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    log_file_abs = os.path.abspath(str(log_file))
    has_same_file = any(
        isinstance(h, logging.FileHandler)
        and os.path.abspath(getattr(h, "baseFilename", "")) == log_file_abs
        for h in root.handlers
    )
    if not has_same_file:
        fh = logging.FileHandler(log_file_abs, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    return str(log_file)


def _export_png_reports(path_result: str, path_dcm: str, patient_id: str) -> None:
    t2_dir = os.path.join(path_dcm, "T2FLAIR_AXI")
    if not os.path.isdir(t2_dir):
        logging.warning("T2FLAIR_AXI 目錄不存在，無法輸出報告 DICOM：%s", t2_dir)
        return

    png_specs = [
        (f"{patient_id}_noPred.png", -1),
        (f"{patient_id}.png", -2),
        (f"{patient_id}_SynthSEG.png", -3),
    ]

    ordered_dcms = _sorted_dicom_paths(t2_dir)
    if not ordered_dcms:
        logging.warning("T2FLAIR_AXI 目錄中沒有可用的 DICOM 檔案：%s", t2_dir)
        return

    export_dir = os.path.join(path_dcm, "T2FLAIR AI Report")
    os.makedirs(export_dir, exist_ok=True)

    seq = 1
    for name, offset in png_specs:
        png_path = os.path.join(path_result, name)
        if not os.path.exists(png_path):
            continue
        template_path = _select_dicom_by_offset(ordered_dcms, offset)
        if not template_path:
            logging.warning("找不到可用的 DICOM 模板來生成 %s", name)
            continue
        _png_to_dicom(png_path, template_path, export_dir, seq)
        seq += 1


def _sorted_dicom_paths(t2_dir: str) -> List[str]:
    files = [
        os.path.join(t2_dir, f)
        for f in os.listdir(t2_dir)
        if f.lower().endswith(".dcm")
    ]
    indexed = []
    for path in files:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            idx = getattr(ds, "InstanceNumber", None)
            if idx is None:
                position = getattr(ds, "ImagePositionPatient", [0, 0, 0])
                idx = float(position[2]) if len(position) >= 3 else 0
        except Exception:
            idx = 0
        indexed.append((idx, path))

    indexed.sort(key=lambda item: item[0])
    return [path for _, path in indexed]


def _select_dicom_by_offset(paths: List[str], offset: int) -> Optional[str]:
    if not paths:
        return None
    if abs(offset) <= len(paths):
        return paths[offset]
    return paths[-1]


def _png_to_dicom(png_path: str, template_path: str, export_dir: str, seq: int) -> None:
    try:
        image = cv2.imread(png_path)
        if image is None:
            logging.warning("無法讀取 PNG：%s", png_path)
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dcm = pydicom.dcmread(template_path)
        new_dcm = color_img_to_dicom_combine(dcm, image, "WMH AI Report", seq)
        original_series = dcm.get((0x0020, 0x0011), None)
        if original_series is not None:
            series_str = str(original_series.value).strip()
            digits = "".join(ch for ch in series_str if ch.isdigit())
            if digits and len(digits) <= 2:
                new_series = digits + "01"
            elif digits and len(digits) >= 3:
                new_series = digits
            else:
                new_series = series_str
            new_dcm[0x0020, 0x0011].value = new_series
        sop_uid = new_dcm[0x0008, 0x0018].value
        out_path = os.path.join(export_dir, f"{sop_uid[-10:]}.dcm")
        new_dcm.save_as(out_path)
        logging.info("已輸出 WMH 報告 DICOM：%s", out_path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("PNG 轉 DICOM 失敗 (%s): %s", png_path, exc)


def _load_make_pred_json(path_code: str):
    """動態載入 code_ai.pipeline.dicomseg.wmh.main，並自動校正 sys.path。"""
    abs_path = os.path.abspath(path_code)
    candidates = [abs_path, os.path.join(abs_path, "code")]
    chosen = None
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "code_ai")):
            chosen = cand
            break
    if chosen is None:
        chosen = abs_path

    if chosen not in sys.path:
        sys.path.append(chosen)
    module = importlib.import_module("code_ai.pipeline.dicomseg.wmh")
    return module.main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WMH After Run Processing")
    parser.add_argument("--patient_id", type=str, required=True)
    parser.add_argument("--path_nnunet", type=str, required=True)
    parser.add_argument("--path_output", type=str, required=True)
    parser.add_argument("--path_code", type=str, required=True)
    args = parser.parse_args()

    _configure_logging(args.path_code)

    ok = after_run_wmh(
        path_nnunet=args.path_nnunet,
        path_output=args.path_output,
        patient_id=args.patient_id,
        path_code=args.path_code,
    )
    if ok:
        print(f"[SUCCESS] after_run_wmh completed for {args.patient_id}")
    else:
        print(f"[FAILED] after_run_wmh failed for {args.patient_id}")


