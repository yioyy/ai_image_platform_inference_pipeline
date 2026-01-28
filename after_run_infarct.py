# -*- coding: utf-8 -*-
"""
Infarct 後處理：統計、JSON 生成與上傳。
"""

import importlib
import os
import sys
import time
import logging
import pathlib
from datetime import datetime
from typing import List, Optional

import cv2  # type: ignore
import pydicom
from color_img_to_dicom import color_img_to_dicom_combine
from util_aneurysm import orthanc_zip_upload, upload_json_aiteam
from infarct_pipeline import InfarctPipeline


def after_run(
    path_nnunet: str, path_output: str, patient_id: str, path_code: str, group_id: int | None = None
) -> bool:
    """執行梗塞後處理流程並上傳結果。"""
    del path_output  # 尚未使用，預留介面
    try:
        _configure_logging(path_code)
        start_time = time.time()
        logging.info("after_run start: %s", patient_id)

        path_result = os.path.join(path_nnunet, "Result")
        path_dcm = os.path.join(path_nnunet, "Dicom")
        path_json_out = os.path.join(path_nnunet, "JSON")
        path_zip = os.path.join(path_nnunet, "Dicom_zip")
        path_excel = os.path.join(path_nnunet, "excel")
        path_image_nii = os.path.join(path_nnunet, "Image_nii")

        if not os.path.isdir(path_result):
            logging.error("缺少 Result 資料夾: %s", path_result)
            return False
        if not os.path.isdir(path_dcm):
            logging.error("缺少 Dicom 資料夾: %s", path_dcm)
            return False

        for folder in (path_json_out, path_zip, path_excel):
            os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(path_dcm, "Dicom-Seg"), exist_ok=True)
        
        _export_png_reports(path_result, path_dcm, patient_id)
        _update_dwi_series(path_dcm)

        pipeline = InfarctPipeline(
            path_dcm=path_dcm,
            path_result=path_result,
            path_excel=path_excel,
            patient_id=patient_id,
            dataset_path=os.path.join(path_code, "dataset.json"),
            path_image_nii=path_image_nii,
        )
        pipeline.run_all()

        make_infarct_pred_json = _load_make_pred_json(path_code)
        if group_id is None:
            group_id = int(os.getenv("RADX_INFARCT_GROUP_ID", "56") or "56")
        json_path = make_infarct_pred_json(
            _id=patient_id,
            path_root=pathlib.Path(path_nnunet),
            group_id=group_id,
        )

        try:
            orthanc_zip_upload(path_dcm, path_zip, ["ADC", "DWI1000", "Dicom-Seg", "Infarct AI Report"])
        except Exception as exc:  # noqa: BLE001
            logging.warning("上傳 DICOM 失敗: %s", exc)

        try:
            upload_json_aiteam(str(json_path))
        except Exception as exc:  # noqa: BLE001
            logging.warning("上傳 JSON 失敗: %s", exc)

        logging.info("after_run done: %s (%.0f sec)", patient_id, time.time() - start_time)
        return True
    except Exception as exc:  # noqa: BLE001
        logging.error("after_run 發生錯誤: %s", exc)
        logging.error("Catch an exception.", exc_info=True)
        return False


def _configure_logging(path_code: str) -> str:
    """將 log 寫到專案既定位置：<project_root>/log/YYYYMMDD.log（與其他 pipeline 一致）。"""
    project_root = pathlib.Path(path_code).resolve().parent
    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 與 gpu_infarct / pipeline_infarct_torch 相同：每日一個檔名
    time_str_short = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{time_str_short}.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 確保寫入檔案
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

    # 同時保留 console 輸出（避免只寫檔不好 debug）
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


def _load_make_pred_json(path_code: str):
    """動態載入 code_ai.pipeline.dicomseg.infarct.main，確保路徑正確。"""
    abs_path = os.path.abspath(path_code)
    if abs_path not in sys.path:
        sys.path.append(abs_path)
    module = importlib.import_module("code_ai.pipeline.dicomseg.infarct")
    return module.main


def _update_dwi_series(path_dcm: str) -> None:
    """調整 DWI0 與 DWI1000 影像的關鍵 DICOM 標籤。"""

    # specs = (
    #     ("DWI1000", "DWI b=1000", "02", "1000")
    # )

    specs = (
        ("DWI0", "DWI b=0", "01", "0"),
        ("DWI1000", "DWI b=1000", "02", "1000"),
    )
    for folder, description, series_suffix, uid_suffix in specs:
        target_dir = os.path.join(path_dcm, folder)
        if not os.path.isdir(target_dir):
            continue
        for name in os.listdir(target_dir):
            if not name.lower().endswith(".dcm"):
                continue
            dicom_path = os.path.join(target_dir, name)
            try:
                ds = pydicom.dcmread(dicom_path)
            except Exception as exc:  # noqa: BLE001
                logging.warning("讀取 DICOM 失敗 (%s): %s", dicom_path, exc)
                continue

            ds.SeriesDescription = description

            series_number = getattr(ds, "SeriesNumber", None)
            formatted = _format_series_number(series_number, series_suffix)
            if formatted:
                ds.SeriesNumber = formatted

            ds.SeriesInstanceUID = _append_uid(getattr(ds, "SeriesInstanceUID", None), uid_suffix)
            ds.SOPInstanceUID = _append_uid(getattr(ds, "SOPInstanceUID", None), uid_suffix)

            try:
                ds.save_as(dicom_path)
            except Exception as exc:  # noqa: BLE001
                logging.warning("寫入 DICOM 失敗 (%s): %s", dicom_path, exc)


def _append_uid(uid_value: Optional[str], suffix: str) -> str:
    uid = str(uid_value).strip() if uid_value else ""
    if uid:
        return f"{uid}{suffix}"
    generated = pydicom.uid.generate_uid()
    return f"{generated}{suffix}"


def _format_series_number(series_value: Optional[object], suffix: str) -> Optional[str]:
    if series_value is None:
        digits = ""
    else:
        series_str = str(series_value).strip()
        digits = "".join(ch for ch in series_str if ch.isdigit())
    if digits:
        if len(digits) <= 2:
            return digits + suffix
        return digits
    if suffix:
        return suffix
    return None


def _export_png_reports(path_result: str, path_dcm: str, patient_id: str) -> None:
    """Convert generated PNG previews into DICOM files for PACS consumption."""
    dwi_dir = os.path.join(path_dcm, "DWI1000")
    if not os.path.isdir(dwi_dir):
        logging.warning("DWI1000 資料夾不存在，無法產生報告 DICOM：%s", dwi_dir)
        return

    png_specs = [
        (f"{patient_id}_noPred.png", -1),
        (f"{patient_id}.png", -2),
        (f"{patient_id}_SynthSEG.png", -3),
    ]

    ordered_dcms = _sorted_dicom_paths(dwi_dir)
    if not ordered_dcms:
        logging.warning("DWI1000 目錄中沒有可用的 DICOM 檔案：%s", dwi_dir)
        return

    export_dir = os.path.join(path_dcm, "Infarct AI Report")
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


def _sorted_dicom_paths(dwi_dir: str) -> List[str]:
    files = [
        os.path.join(dwi_dir, f)
        for f in os.listdir(dwi_dir)
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
        new_dcm = color_img_to_dicom_combine(dcm, image, "Infarct AI Report", seq)
        original_series = dcm.get((0x0020, 0x0011), None)
        formatted = _format_series_number(original_series.value if original_series else None, "03")
        if formatted:
            new_dcm[0x0020, 0x0011].value = formatted
        sop_uid = new_dcm[0x0008, 0x0018].value
        out_path = os.path.join(export_dir, f"{sop_uid[-10:]}.dcm")
        new_dcm.save_as(out_path)
        logging.info("已輸出報告 DICOM：%s", out_path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("PNG 轉 DICOM 失敗 (%s): %s", png_path, exc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Infarct After Run Processing")
    parser.add_argument("--patient_id", type=str, required=True, help="病人 ID")
    parser.add_argument("--path_nnunet", type=str, required=True, help="nnUNet 資料夾路徑")
    parser.add_argument("--path_output", type=str, required=True, help="輸出資料夾路徑")
    parser.add_argument("--path_code", type=str, required=True, help="code 根目錄")
    args = parser.parse_args()

    _configure_logging(args.path_code)

    ok = after_run(
        path_nnunet=args.path_nnunet,
        path_output=args.path_output,
        patient_id=args.patient_id,
        path_code=args.path_code,
    )
    if ok:
        print(f"[SUCCESS] after_run completed for {args.patient_id}")
    else:
        print(f"[FAILED] after_run failed for {args.patient_id}")

