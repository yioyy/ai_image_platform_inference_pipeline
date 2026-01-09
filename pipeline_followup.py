#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Follow-up pipeline (file-oriented).

主程式只負責：
- 參數解析
- 建立資料夾結構、複製輸入輸出
- 呼叫子模組：registration / pred merge / followup report
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np

from generate_followup_pred import generate_followup_pred
from followup_report import make_followup_json
from registration import run_registration_followup_to_baseline


def pipeline_followup(
    *,
    baseline_ID: str,
    baseline_Inputs: List[str],
    baseline_DicomDir: str,
    baseline_DicomSegDir: str,
    baseline_json: str,
    followup_ID: str,
    followup_Inputs: List[str],
    followup_DicomSegDir: str,
    followup_json: str,
    path_output: str,
    model: str = "CMB",
    path_code: str = "",
    path_process: str = "",
    path_json: str = "",
    path_log: str = "",
    gpu_n: int = 0,
    fsl_flirt_path: str = "/usr/local/fsl/bin/flirt",
    upload_json: bool = False,
) -> Path:
    del path_code, path_json, gpu_n  # 預留介面（對齊既有 pipeline 風格）

    start = time.time()
    _validate_inputs(baseline_Inputs, followup_Inputs)

    process_root = _build_process_root(path_process, baseline_ID)
    baseline_dir = process_root / "baseline"
    followup_dir = process_root / "followup"
    registration_dir = process_root / "registration"
    result_dir = process_root / "result"
    dicom_dir = process_root / "dicom"

    _ensure_dirs([baseline_dir, followup_dir, registration_dir, result_dir, dicom_dir])

    baseline_pred_src, baseline_synthseg_src = map(Path, baseline_Inputs)
    followup_pred_src, followup_synthseg_src = map(Path, followup_Inputs)

    baseline_pred = baseline_dir / f"Pred_{model}.nii.gz"
    baseline_synthseg = baseline_dir / f"SynthSEG_{model}.nii.gz"
    baseline_json_dst = baseline_dir / f"Pred_{model}_platform_json.json"

    followup_pred = followup_dir / f"Pred_{model}.nii.gz"
    followup_synthseg = followup_dir / f"SynthSEG_{model}.nii.gz"
    followup_json_dst = followup_dir / f"Pred_{model}_platform_json.json"

    _copy_file(baseline_pred_src, baseline_pred)
    _copy_file(baseline_synthseg_src, baseline_synthseg)
    _copy_file(Path(baseline_json), baseline_json_dst)

    _copy_file(followup_pred_src, followup_pred)
    _copy_file(followup_synthseg_src, followup_synthseg)
    _copy_file(Path(followup_json), followup_json_dst)

    # 重要：先用 SynthSEG 的幾何 header 覆蓋 Pred，確保 voxel grid 一致（避免 overlap=0）
    _overwrite_pred_geometry_from_synthseg(pred_file=baseline_pred, synthseg_file=baseline_synthseg)
    _overwrite_pred_geometry_from_synthseg(pred_file=followup_pred, synthseg_file=followup_synthseg)

    _copy_dir(Path(baseline_DicomSegDir), baseline_dir / "dicom-seg")
    _copy_dir(Path(followup_DicomSegDir), followup_dir / "dicom-seg")
    if baseline_DicomDir:
        _copy_dir(Path(baseline_DicomDir), dicom_dir / "baseline")

    mat_file = registration_dir / f"{baseline_ID}_{model}.mat"
    synthseg_reg = registration_dir / f"SynthSEG_{model}_registration.nii.gz"
    pred_reg = registration_dir / f"Pred_{model}_registration.nii.gz"

    start = time.time()
    run_registration_followup_to_baseline(
        baseline_pred_file=baseline_pred,
        baseline_synthseg_file=baseline_synthseg,
        followup_synthseg_file=followup_synthseg,
        followup_pred_file=followup_pred,
        out_dir=registration_dir,
        mat_file=mat_file,
        out_synthseg_reg_file=synthseg_reg,
        out_pred_reg_file=pred_reg,
        fsl_flirt_path=fsl_flirt_path,
    )
    print(f"[Done Registration... ] spend {time.time() - start:.0f} sec")
    logging.info(f"[Done Registration... ] spend {time.time() - start:.0f} sec")

    merged_pred = result_dir / f"Pred_{model}_followup.nii.gz"
    start = time.time()
    generate_followup_pred(
        baseline_pred_file=baseline_pred,
        followup_reg_pred_file=pred_reg,
        output_file=merged_pred,
    )
    print(f"[Done Generate Followup Pred... ] spend {time.time() - start:.0f} sec")
    logging.info(f"[Done Generate Followup Pred... ] spend {time.time() - start:.0f} sec")

    followup_json_out = result_dir / f"Followup_{model}_platform_json.json"
    start = time.time()
    make_followup_json(
        baseline_pred_file=baseline_pred,
        baseline_json_file=baseline_json_dst,
        followup_json_file=followup_json_dst,
        followup_reg_pred_file=pred_reg,
        registration_mat_file=mat_file,
        output_json_file=followup_json_out,
        group_id=56,
        transform_direction="F2B",
    )
    print(f"[Done Make Followup JSON... ] spend {time.time() - start:.0f} sec")
    logging.info(f"[Done Make Followup JSON... ] spend {time.time() - start:.0f} sec")

    output_root = Path(path_output) / baseline_ID
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_file(followup_json_out, output_root / followup_json_out.name)
    _copy_file(merged_pred, output_root / merged_pred.name)

    out_dicom_seg = output_root / "dicom-seg"
    _ensure_dirs([out_dicom_seg])
    _copy_dir(baseline_dir / "dicom-seg", out_dicom_seg / "baseline")
    _copy_dir(followup_dir / "dicom-seg", out_dicom_seg / "followup")

    if upload_json:
        _upload_json_to_platform(output_root / followup_json_out.name)

    logging.info("pipeline_followup done: %s (%.0f sec)", baseline_ID, time.time() - start)
    return output_root


def _overwrite_pred_geometry_from_synthseg(*, pred_file: Path, synthseg_file: Path) -> None:
    """
    將 SynthSEG 的 NIfTI 幾何資訊（affine/sform/qform）覆蓋到 Pred 上。

    前提：Pred 與 SynthSEG 的 data shape 必須一致；否則覆蓋幾何會造成嚴重錯位。
    """
    if not pred_file.exists():
        raise FileNotFoundError(f"找不到 Pred 檔案：{pred_file}")
    if not synthseg_file.exists():
        raise FileNotFoundError(f"找不到 SynthSEG 檔案：{synthseg_file}")

    pred_img = nib.load(str(pred_file))
    syn_img = nib.load(str(synthseg_file))

    pred_data = np.asanyarray(pred_img.dataobj)
    if pred_data.shape != syn_img.shape:
        raise ValueError(
            "Pred 與 SynthSEG shape 不一致，無法複製 header："
            f"pred={pred_file.name} shape={pred_data.shape} vs "
            f"synthseg={synthseg_file.name} shape={syn_img.shape}"
        )

    # 用 SynthSEG 的 affine + header（幾何），但保留 Pred 的 dtype
    header = syn_img.header.copy()
    header.set_data_shape(pred_data.shape)
    header.set_data_dtype(pred_data.dtype)

    out = nib.Nifti1Image(pred_data, syn_img.affine, header=header)
    # 明確同步 qform/sform，避免部分 viewer 只看其中一個
    out.set_qform(syn_img.get_qform(), code=int(syn_img.header.get("qform_code", 1) or 1))
    out.set_sform(syn_img.get_sform(), code=int(syn_img.header.get("sform_code", 1) or 1))
    nib.save(out, str(pred_file))


def _validate_inputs(baseline_inputs: List[str], followup_inputs: List[str]) -> None:
    if len(baseline_inputs) != 2:
        raise ValueError("baseline_Inputs 必須包含 2 個檔案：[Pred, SynthSEG]")
    if len(followup_inputs) != 2:
        raise ValueError("followup_Inputs 必須包含 2 個檔案：[Pred, SynthSEG]")


def _build_process_root(path_process: str, baseline_id: str) -> Path:
    """
    統一處理資料夾結構，並與 CLI default 相容：
    - 若 path_process 已指向 .../Deep_FollowUp/（或大小寫不同），直接用 <path_process>/<baseline_id>/
    - 否則沿用舊版行為：<path_process>/Deep_FollowUp/<baseline_id>/
    """
    base = Path(path_process)
    if base.name.lower() == "deep_followup":
        return base / baseline_id
    return base / "Deep_FollowUp" / baseline_id


def _ensure_dirs(paths: List[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"找不到檔案：{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"找不到資料夾：{src}")
    if not src.is_dir():
        raise NotADirectoryError(f"不是資料夾：{src}")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Follow-up comparison pipeline")
    p.add_argument("--baseline_ID", default = '00130846_20220105_MR_E211126000320', help='目前執行的case的baseline ID')
    p.add_argument("--baseline_Inputs", nargs="+", default = ['/data/4TB1/pipeline/chuan/example_input/00130846_20220105_MR_E211126000320/Pred_CMB.nii.gz', '/data/4TB1/pipeline/chuan/example_input/00130846_20220105_MR_E211126000320/SynthSEG_CMB.nii.gz'])
    p.add_argument("--baseline_DicomDir", default="/data/4TB1/pipeline/chuan/example_inputDicom/00130846_20220105_MR_E211126000320/SWAN", help='baseline的dicom資料夾')
    p.add_argument("--baseline_DicomSegDir", default = '/data/4TB1/pipeline/chuan/example_inputDicom/00130846_20220105_MR_E211126000320/Dicom-Seg', help='baseline的dicom-seg資料夾')
    p.add_argument("--baseline_json", default = '/data/4TB1/pipeline/chuan/example_input/00130846_20220105_MR_E211126000320/Pred_CMB_platform_json.json', help='baseline的json資料夾')

    p.add_argument("--followup_ID", default = '00130846_20210409_MR_21003160054', help='目前執行的case的followup ID')
    p.add_argument("--followup_Inputs", nargs="+", default = ['/data/4TB1/pipeline/chuan/example_input/00130846_20210409_MR_21003160054/Pred_CMB.nii.gz', '/data/4TB1/pipeline/chuan/example_input/00130846_20210409_MR_21003160054/SynthSEG_CMB.nii.gz'])
    p.add_argument("--followup_DicomSegDir", default = '/data/4TB1/pipeline/chuan/example_inputDicom/00130846_20210409_MR_21003160054/Dicom-Seg', help='followup的dicom-seg資料夾')
    p.add_argument("--followup_json", default = '/data/4TB1/pipeline/chuan/example_input/00130846_20210409_MR_21003160054/Pred_CMB_platform_json.json', help='followup的json資料夾')

    p.add_argument("--path_output", default = '/data/4TB1/pipeline/chuan/example_output/00130846_20220105_MR_E211126000320/', help='輸出資料夾')
    p.add_argument("--model", default="CMB", help='模型')
    p.add_argument("--path_code", default="/data/4TB1/pipeline/chuan/code/")
    p.add_argument("--path_process", default="/data/4TB1/pipeline/chuan/process/Deep_FollowUp/")
    p.add_argument("--path_json", default="/data/4TB1/pipeline/chuan/json/")
    p.add_argument("--path_log", default="/data/4TB1/pipeline/chuan/log/")
    p.add_argument("--gpu_n", default=0, type=int)
    p.add_argument("--fsl_flirt_path", default="/usr/local/fsl/bin/flirt")
    p.add_argument("--upload_json", action="store_true", help="是否上傳 Followup_CMB_platform_json.json 到平台")
    return p


def main(argv: List[str]) -> int:
    args = _build_parser().parse_args(argv)

    if args.path_log:
        Path(args.path_log).mkdir(parents=True, exist_ok=True)

        localt = time.localtime(time.time())
        time_str_short = str(localt.tm_year) + str(localt.tm_mon).rjust(2, "0") + str(localt.tm_mday).rjust(2, "0")
        log_file = str(Path(args.path_log) / f"{time_str_short}.log")

        if not Path(log_file).is_file():
            with open(log_file, "a+", encoding="utf-8") as f:
                f.write("")

        FORMAT = "%(asctime)s %(levelname)s %(message)s"
        logging.basicConfig(level=logging.INFO, filename=log_file, filemode="a", format=FORMAT)
    else:
        FORMAT = "%(asctime)s %(levelname)s %(message)s"
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format=FORMAT)

    pipeline_followup(
        baseline_ID=args.baseline_ID,
        baseline_Inputs=args.baseline_Inputs,
        baseline_DicomDir=args.baseline_DicomDir,
        baseline_DicomSegDir=args.baseline_DicomSegDir,
        baseline_json=args.baseline_json,
        followup_ID=args.followup_ID,
        followup_Inputs=args.followup_Inputs,
        followup_DicomSegDir=args.followup_DicomSegDir,
        followup_json=args.followup_json,
        path_output=args.path_output,
        model=args.model,
        path_code=args.path_code,
        path_process=args.path_process,
        path_json=args.path_json,
        path_log=args.path_log,
        gpu_n=args.gpu_n,
        fsl_flirt_path=args.fsl_flirt_path,
        upload_json=args.upload_json,
    )
    return 0


def _upload_json_to_platform(json_path: Path) -> None:
    try:
        from util_aneurysm import upload_json_aiteam
    except Exception as exc:  # noqa: BLE001
        logging.warning("無法載入 upload_json_aiteam：%s", exc)
        return

    try:
        upload_json_aiteam(str(json_path))
    except Exception as exc:  # noqa: BLE001
        logging.warning("上傳 JSON 失敗：%s", exc)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


