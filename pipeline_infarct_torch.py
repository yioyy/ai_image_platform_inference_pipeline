# -*- coding: utf-8 -*-
"""
RADAX Infarct AI Pipeline (Torch/nnUNet flow)

目標：參考 `code/pipeline_infarct.sh` + `code/pipeline_infarct_torch.py` 的推論流程，
差異在最後「上傳」：
- 不再走 aiteam/orthanc 上傳
- 改為複製結果到 `upload_dir/<StudyInstanceUID>/infarct_model/<inference_id>/...`
- 並呼叫 `POST /v1/ai-inference/inference-complete`
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pynvml  # type: ignore
import requests
import tensorflow as tf  # type: ignore

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _clean_path(p: str) -> str:
    return os.path.normpath(str(p).strip().replace("\r", "").replace("\n", ""))


def _ensure_code_dir_on_syspath(path_code: str) -> None:
    # 讓 radax 腳本能 import 到 `code/` 內的 pre/post 與 pipeline 模組
    repo_root = Path(__file__).resolve().parents[1]
    code_dir = repo_root / "code"
    if code_dir.is_dir():
        code_dir_str = str(code_dir)
        if code_dir_str not in os.sys.path:
            os.sys.path.append(code_dir_str)

    # 同時也允許使用者傳入的 path_code（在 server 上常是 /home/.../code/）
    path_code_abs = os.path.abspath(path_code)
    if path_code_abs not in os.sys.path and os.path.isdir(path_code_abs):
        os.sys.path.append(path_code_abs)


def _try_get_study_instance_uid_from_dicom_dir(dicom_dir: str) -> str:
    if not dicom_dir or not os.path.isdir(dicom_dir):
        return ""
    try:
        import pydicom  # lazy import
    except Exception:
        return ""

    for root, _, files in os.walk(dicom_dir):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            if not os.path.isfile(fp):
                continue
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                uid = str(getattr(ds, "StudyInstanceUID", "") or "")
                if uid:
                    return uid
            except Exception:
                continue
    return ""


def _extract_study_uid_from_platform_json(payload: Dict[str, Any]) -> str:
    study = payload.get("study")
    if isinstance(study, dict):
        return str(study.get("study_instance_uid") or "")
    return ""


def _post_inference_complete(
    *,
    api_url: str,
    study_instance_uid: str,
    model_name: str,
    inference_id: Optional[str],
    result: str,
) -> None:
    payload: Dict[str, Any] = {"studyInstanceUid": study_instance_uid, "modelName": model_name}
    if result != "":
        payload["result"] = result
    if inference_id and (result == "" or result == "success"):
        payload["inferenceId"] = inference_id

    try:
        resp = requests.post(api_url, json=payload, timeout=10)
        print(f"[API POST] payload={payload} Status Code: {resp.status_code}, Response: {resp.text}")
        logging.info("[API POST] payload=%s Status=%s Response=%s", payload, resp.status_code, resp.text)
    except Exception as e:
        print(f"[API POST Error] payload={payload} err={str(e)}")
        logging.error("[API POST Error] payload=%s err=%s", payload, str(e))


def _copy_dicom_series_to_uid_folder(src_dir: str, dst_root: str) -> None:
    if not os.path.isdir(src_dir):
        return
    try:
        import pydicom
    except Exception:
        return

    first_uid = ""
    for fn in sorted(os.listdir(src_dir)):
        fp = os.path.join(src_dir, fn)
        if not os.path.isfile(fp):
            continue
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            first_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
            if first_uid:
                break
        except Exception:
            continue

    if not first_uid:
        return

    dst_dir = os.path.join(dst_root, first_uid)
    os.makedirs(dst_dir, exist_ok=True)

    for fn in sorted(os.listdir(src_dir)):
        src_fp = os.path.join(src_dir, fn)
        if not os.path.isfile(src_fp):
            continue
        shutil.copy(src_fp, os.path.join(dst_dir, fn))


def _copy_dicomseg_files(src_dicomseg_dir: str, dst_dir: str) -> None:
    if not os.path.isdir(src_dicomseg_dir):
        return
    try:
        import pydicom
    except Exception:
        return

    for fn in sorted(os.listdir(src_dicomseg_dir)):
        if not fn.lower().endswith(".dcm"):
            continue
        src_fp = os.path.join(src_dicomseg_dir, fn)
        if not os.path.isfile(src_fp):
            continue
        try:
            ds = pydicom.dcmread(src_fp, stop_before_pixels=True, force=True)
            seg_series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        except Exception:
            seg_series_uid = ""

        mask_name = ""
        base = os.path.splitext(fn)[0]
        if "_" in base:
            mask_name = base.split("_")[-1]
        if not mask_name:
            mask_name = base

        dst_name = fn
        if seg_series_uid:
            dst_name = f"{seg_series_uid}_{mask_name}.dcm"
        shutil.copy(src_fp, os.path.join(dst_dir, dst_name))


def _generate_platform_json_and_dicomseg(*, patient_id: str, path_nnunet: str, path_code: str) -> str:
    # 依 code/after_run_infarct.py：先跑 InfarctPipeline 產 excel，再跑 code_ai dicomseg builder
    path_result = os.path.join(path_nnunet, "Result")
    path_dcm = os.path.join(path_nnunet, "Dicom")
    path_excel = os.path.join(path_nnunet, "excel")
    path_image_nii = os.path.join(path_nnunet, "Image_nii")
    os.makedirs(os.path.join(path_dcm, "Dicom-Seg"), exist_ok=True)
    os.makedirs(os.path.join(path_nnunet, "JSON"), exist_ok=True)
    os.makedirs(path_excel, exist_ok=True)

    if not os.path.isdir(path_result):
        raise FileNotFoundError(f"缺少 Result 資料夾: {path_result}")
    if not os.path.isdir(path_dcm):
        raise FileNotFoundError(f"缺少 Dicom 資料夾: {path_dcm}")

    from infarct_pipeline import InfarctPipeline  # type: ignore

    pipeline = InfarctPipeline(
        path_dcm=path_dcm,
        path_result=path_result,
        path_excel=path_excel,
        patient_id=patient_id,
        dataset_path=os.path.join(path_code, "dataset.json"),
        path_image_nii=path_image_nii,
    )
    pipeline.run_all()

    from code_ai.pipeline.dicomseg.infarct import main as make_infarct_pred_json

    json_path = make_infarct_pred_json(_id=patient_id, path_root=Path(path_nnunet), group_id=56)
    return str(json_path)


def _generate_radax_prediction_json(*, patient_id: str, path_nnunet: str, model_id: str) -> str:
    from code_ai.pipeline.dicomseg.build_infarct import execute_rdx_platform_json

    out_path = execute_rdx_platform_json(
        patient_id=patient_id,
        path_root=Path(path_nnunet),
        model_id=model_id,
    )
    return str(out_path)


def pipeline_infarct(
    ID: str,
    Inputs,
    DicomDirs,
    path_output: str,
    path_code: str = "/home/david/pipeline/chuan/code/",
    path_nnunet_model: str = "/home/david/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset040_DeepInfarct/nnUNetTrainer__nnUNetPlans__2d",
    path_processModel: str = "/home/david/pipeline/chuan/process/Deep_Infarct/",
    path_json: str = "/home/david/pipeline/chuan/json/",
    path_log: str = "/home/david/pipeline/chuan/log/",
    gpu_n: int = 0,
    *,
    upload_dir: str = "/home/david/ai-inference-result-testing",
    api_url: str = "http://localhost:24000/v1/ai-inference/inference-complete",
    model_id: str = "924d1538-597c-41d6-bc27-4b0b359111cf",
) -> None:
    ID = str(ID).strip().replace("\r", "").replace("\n", "")
    path_output = _clean_path(path_output)
    path_code = _clean_path(path_code)
    path_nnunet_model = _clean_path(path_nnunet_model)
    path_processModel = _clean_path(path_processModel)
    path_json = _clean_path(path_json)
    path_log = _clean_path(path_log)
    upload_dir = _clean_path(upload_dir)
    model_id = str(model_id).strip()

    os.makedirs(path_output, exist_ok=True)
    os.makedirs(path_processModel, exist_ok=True)
    os.makedirs(path_json, exist_ok=True)
    os.makedirs(path_log, exist_ok=True)

    # logger 設定（避免 TF 打爆）
    tf.get_logger().setLevel(logging.ERROR)

    # log 檔：每日一個
    time_str_short = time.strftime("%Y%m%d", time.localtime())
    log_file = os.path.join(path_log, f"{time_str_short}.log")
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode="a", format="%(asctime)s %(levelname)s %(message)s")

    _ensure_code_dir_on_syspath(path_code)
    from pre_infarct import preprocess_stroke_images  # type: ignore
    from post_infarct import postprocess_infarct_results  # type: ignore
    from util_aneurysm import resampleSynthSEG2original  # type: ignore

    path_processID = os.path.join(path_processModel, ID)
    os.makedirs(path_processID, exist_ok=True)
    print(ID, " Start...")
    logging.info("%s Start...", ID)

    # 依照不同情境拆分 try
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_n)
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpumRate = memoryInfo.used / memoryInfo.total

        if gpumRate >= 0.6:
            logging.error("Insufficient GPU Memory. gpumRate=%.3f", gpumRate)
            study_uid = _try_get_study_instance_uid_from_dicom_dir(DicomDirs[2] if DicomDirs else "")
            _post_inference_complete(
                api_url=api_url,
                study_instance_uid=study_uid,
                model_name="infarct_model",
                inference_id=None,
                result="failed",
            )
            return

        gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
        tf.config.experimental.set_visible_devices(devices=gpus[gpu_n], device_type="GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_n], True)

        ADC_file = _clean_path(Inputs[0])
        DWI0_file = _clean_path(Inputs[1])
        DWI1000_file = _clean_path(Inputs[2])
        SynthSEG_file = _clean_path(Inputs[3]) if len(Inputs) > 3 else None

        path_DcmADC = _clean_path(DicomDirs[0])
        path_DcmDWI0 = _clean_path(DicomDirs[1])
        path_DcmDWI1000 = _clean_path(DicomDirs[2])

        shutil.copy(ADC_file, os.path.join(path_processID, "ADC.nii.gz"))
        shutil.copy(DWI0_file, os.path.join(path_processID, "DWI0.nii.gz"))
        shutil.copy(DWI1000_file, os.path.join(path_processID, "DWI1000.nii.gz"))

        if SynthSEG_file and os.path.exists(SynthSEG_file):
            shutil.copy(SynthSEG_file, os.path.join(path_processID, "SynthSEG.nii.gz"))
        else:
            # 沒有 SynthSEG，自己產
            path_nii = os.path.join(path_processID, "nii")
            os.makedirs(path_nii, exist_ok=True)
            shutil.copy(ADC_file, os.path.join(path_nii, "ADC.nii.gz"))
            shutil.copy(DWI0_file, os.path.join(path_nii, "DWI0.nii.gz"))
            shutil.copy(DWI1000_file, os.path.join(path_nii, "DWI1000.nii.gz"))

            path_synthseg = os.path.join(path_code, "model_weights", "SynthSeg_parcellation_tf28", "code")
            cmd = [
                "python",
                os.path.join(path_synthseg, "main_tf28.py"),
                "-i",
                path_nii,
                "--input_name",
                "DWI0.nii.gz",
                "--template",
                path_nii,
                "--template_name",
                "DWI0",
                "--all",
                "False",
                "--DWI",
                "True",
            ]
            subprocess.run(cmd)
            resampleSynthSEG2original(path_nii, "DWI0", "DWI")
            shutil.copy(os.path.join(path_nii, "NEW_DWI0_DWI.nii.gz"), os.path.join(path_processID, "SynthSEG.nii.gz"))

        if not preprocess_stroke_images(path_processID):
            logging.error("%s preprocessing failed.", ID)
            return

        path_nnunet = os.path.join(path_processID, "nnUNet")
        os.makedirs(path_nnunet, exist_ok=True)

        path_dcm_n = os.path.join(path_nnunet, "Dicom")
        path_nii_n = os.path.join(path_nnunet, "Image_nii")
        path_post_processed_n = os.path.join(path_nnunet, "Post_processed")
        path_result_n = os.path.join(path_nnunet, "Result")
        path_excel_n = os.path.join(path_nnunet, "excel")
        for folder in (path_dcm_n, path_nii_n, path_post_processed_n, path_result_n, path_excel_n):
            os.makedirs(folder, exist_ok=True)

        print("Running stage 1: Infarct inference!!!")
        logging.info("Running stage 1: Infarct inference!!!")
        cmd = [
            "python",
            os.path.join(path_code, "gpu_infarct.py"),
            "--path_code",
            path_code,
            "--path_process",
            path_processID,
            "--path_nnunet_model",
            path_nnunet_model,
            "--case",
            ID,
            "--path_log",
            path_log,
            "--gpu_n",
            str(gpu_n),
        ]
        start = time.time()
        subprocess.run(cmd)
        logging.info("[Done AI Inference... ] spend %.0f sec", time.time() - start)

        # DICOM copy into nnUNet/Dicom
        path_dcm_adc = os.path.join(path_dcm_n, "ADC")
        path_dcm_dwi0 = os.path.join(path_dcm_n, "DWI0")
        path_dcm_dwi1000 = os.path.join(path_dcm_n, "DWI1000")
        if not os.path.isdir(path_dcm_adc):
            shutil.copytree(path_DcmADC, path_dcm_adc)
        if not os.path.isdir(path_dcm_dwi0):
            shutil.copytree(path_DcmDWI0, path_dcm_dwi0)
        if not os.path.isdir(path_dcm_dwi1000):
            shutil.copytree(path_DcmDWI1000, path_dcm_dwi1000)

        # nifti copy into Image_nii
        for src, dst in (
            (os.path.join(path_processID, "ADC.nii.gz"), os.path.join(path_nii_n, "ADC.nii.gz")),
            (os.path.join(path_processID, "DWI0.nii.gz"), os.path.join(path_nii_n, "DWI0.nii.gz")),
            (os.path.join(path_processID, "DWI1000.nii.gz"), os.path.join(path_nii_n, "DWI1000.nii.gz")),
            (os.path.join(path_processID, "SynthSEG.nii.gz"), os.path.join(path_nii_n, "SynthSEG.nii.gz")),
            (os.path.join(path_nnunet, "Pred.nii.gz"), os.path.join(path_nii_n, "Pred.nii.gz")),
        ):
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy(src, dst)

        prob_src = os.path.join(path_nnunet, "Prob.nii.gz")
        prob_dst = os.path.join(path_result_n, "Prob.nii.gz")
        if os.path.isfile(prob_src) and not os.path.isfile(prob_dst):
            shutil.copy(prob_src, prob_dst)

        print("Running stage 2: Post-processing!!!")
        logging.info("Running stage 2: Post-processing!!!")
        if not postprocess_infarct_results(
            patient_id=ID,
            path_nnunet=path_nnunet,
            path_code=path_code,
            path_output=path_output,
            path_json=path_json,
        ):
            logging.error("%s post-processing failed.", ID)
            return

        # 先產出平台 JSON + DICOMSEG（來源資料）
        _generate_platform_json_and_dicomseg(patient_id=ID, path_nnunet=path_nnunet, path_code=path_code)

        # 再產出 RADAX prediction json（rdx_infarct_pred_json.json）
        rdx_json_path = _generate_radax_prediction_json(patient_id=ID, path_nnunet=path_nnunet, model_id=model_id)

        # ---- RADAX 上傳：copy 到指定資料夾 + 打 inference-complete ----
        with open(rdx_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        # radax json 的 input_study_instance_uid 是 list
        study_uid = str((payload.get("input_study_instance_uid") or [""])[0] or "") or _try_get_study_instance_uid_from_dicom_dir(path_DcmDWI1000)
        inference_id = str(payload.get("inference_id") or "")
        if not inference_id:
            inference_id = str(uuid.uuid4())

        study_dir = os.path.join(upload_dir, study_uid)
        infer_dir = os.path.join(study_dir, "infarct_model", inference_id)
        os.makedirs(infer_dir, exist_ok=True)

        shutil.copy(rdx_json_path, os.path.join(infer_dir, "prediction.json"))

        # copy input DICOM series
        for series_folder in ("ADC", "DWI0", "DWI1000"):
            _copy_dicom_series_to_uid_folder(os.path.join(path_dcm_n, series_folder), infer_dir)

        # copy DICOMSEG (rename to <SeriesInstanceUID>_A?.dcm)
        _copy_dicomseg_files(os.path.join(path_dcm_n, "Dicom-Seg"), infer_dir)

        _post_inference_complete(
            api_url=api_url,
            study_instance_uid=study_uid,
            model_name="infarct_model",
            inference_id=inference_id,
            result="",
        )

        logging.info("!!! %s post_infarct finish.", ID)
        print("end!!!")
        return

    except Exception as e:
        logging.error("Catch an exception.", exc_info=True)
        logging.error("pipeline_infarct failed: %s", str(e))
        study_uid = ""
        try:
            study_uid = _try_get_study_instance_uid_from_dicom_dir(DicomDirs[2] if DicomDirs else "")
        except Exception:
            study_uid = ""
        _post_inference_complete(
            api_url=api_url,
            study_instance_uid=study_uid,
            model_name="infarct_model",
            inference_id=None,
            result="failed",
        )
        print("end!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", type=str, required=True)
    parser.add_argument("--Inputs", type=str, nargs="+", required=True)
    parser.add_argument("--DicomDir", type=str, nargs="+", required=True)
    parser.add_argument("--Output_folder", type=str, required=True)
    parser.add_argument("--gpu_n", type=int, default=0)
    parser.add_argument("--upload_dir", type=str, default="/home/david/ai-inference-result-testing")
    parser.add_argument("--api_url", type=str, default="http://localhost:24000/v1/ai-inference/inference-complete")
    parser.add_argument("--model_id", type=str, default="924d1538-597c-41d6-bc27-4b0b359111cf")
    args = parser.parse_args()

    pipeline_infarct(
        args.ID,
        args.Inputs,
        args.DicomDir,
        args.Output_folder,
        gpu_n=int(args.gpu_n),
        upload_dir=args.upload_dir,
        api_url=args.api_url,
        model_id=args.model_id,
    )

