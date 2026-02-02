#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Follow-up pipeline (DICOM-SEG entry, zero-invasive).

此腳本提供「新入口」：
- 不修改既有 `pipeline_followup.py` 的行為與 CLI
- 在這裡支援：Pred DICOM-SEG / SynthSEG DICOM-SEG + per-instance DICOM headers → NIfTI
- 生成暫存的 Pred/SynthSEG NIfTI 後，呼叫既有 `pipeline_followup.pipeline_followup`
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

import pipeline_followup
from code_ai.pipeline.dicomseg.seg_to_nifti import load_series_headers_json, synthseg_dicomseg_to_nifti


def _require_file(path: str, name: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到檔案（{name}）：{p}")
    return p


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Follow-up comparison pipeline (from DICOM-SEG + headers)")
    p.add_argument("--baseline_ID", required=True)
    p.add_argument("--baseline_pred_dicomseg", required=True, help="baseline Pred 的 DICOM-SEG (.dcm)")
    p.add_argument("--baseline_synthseg_dicomseg", required=True, help="baseline SynthSEG DICOM-SEG (.dcm)")
    p.add_argument("--baseline_series_headers", required=True, help="baseline series per-instance headers json（無 PixelData）")
    p.add_argument("--baseline_json", required=True)

    p.add_argument("--followup_ID", required=True)
    p.add_argument("--followup_pred_dicomseg", required=True, help="followup Pred 的 DICOM-SEG (.dcm)")
    p.add_argument("--followup_synthseg_dicomseg", required=True, help="followup SynthSEG DICOM-SEG (.dcm)")
    p.add_argument("--followup_series_headers", required=True, help="followup series per-instance headers json（無 PixelData）")
    p.add_argument("--followup_json", required=True)

    p.add_argument("--path_output", required=True)
    p.add_argument("--model", default="CMB")
    p.add_argument("--path_code", default="")
    p.add_argument("--path_process", default="")
    p.add_argument("--path_json", default="")
    p.add_argument("--path_log", default="")
    p.add_argument("--gpu_n", default=0, type=int)
    p.add_argument("--fsl_flirt_path", default="/usr/local/fsl/bin/flirt")
    p.add_argument("--upload_json", action="store_true")
    p.add_argument("--overlap_mode", default="max", choices=["max", "overwrite"], help="SEG 重疊時的 label 合成規則")
    return p


def _materialize_dicomseg_dir(*, out_dir: Path, dicomseg_files: List[Path]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in dicomseg_files:
        if not f.exists():
            raise FileNotFoundError(f"找不到 DICOM-SEG 檔案：{f}")
        shutil.copy2(f, out_dir / f.name)
    return out_dir


def main(argv: List[str]) -> int:
    args = _build_parser().parse_args(argv)

    baseline_pred_seg = _require_file(args.baseline_pred_dicomseg, "baseline_pred_dicomseg")
    followup_pred_seg = _require_file(args.followup_pred_dicomseg, "followup_pred_dicomseg")
    baseline_seg = _require_file(args.baseline_synthseg_dicomseg, "baseline_synthseg_dicomseg")
    followup_seg = _require_file(args.followup_synthseg_dicomseg, "followup_synthseg_dicomseg")
    baseline_headers_json = _require_file(args.baseline_series_headers, "baseline_series_headers")
    followup_headers_json = _require_file(args.followup_series_headers, "followup_series_headers")

    baseline_headers = load_series_headers_json(baseline_headers_json)
    followup_headers = load_series_headers_json(followup_headers_json)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        baseline_dicomseg_dir = _materialize_dicomseg_dir(
            out_dir=td / "baseline_dicomseg",
            dicomseg_files=[baseline_pred_seg, baseline_seg],
        )
        followup_dicomseg_dir = _materialize_dicomseg_dir(
            out_dir=td / "followup_dicomseg",
            dicomseg_files=[followup_pred_seg, followup_seg],
        )
        baseline_pred = td / f"Pred_{args.model}_baseline.nii.gz"
        followup_pred = td / f"Pred_{args.model}_followup.nii.gz"
        baseline_syn = td / f"SynthSEG_{args.model}_baseline.nii.gz"
        followup_syn = td / f"SynthSEG_{args.model}_followup.nii.gz"

        synthseg_dicomseg_to_nifti(
            dicomseg_path=baseline_pred_seg,
            series_headers=baseline_headers,
            out_nii_path=baseline_pred,
            overlap_mode=args.overlap_mode,
        )
        synthseg_dicomseg_to_nifti(
            dicomseg_path=followup_pred_seg,
            series_headers=followup_headers,
            out_nii_path=followup_pred,
            overlap_mode=args.overlap_mode,
        )
        synthseg_dicomseg_to_nifti(
            dicomseg_path=baseline_seg,
            series_headers=baseline_headers,
            out_nii_path=baseline_syn,
            overlap_mode=args.overlap_mode,
        )
        synthseg_dicomseg_to_nifti(
            dicomseg_path=followup_seg,
            series_headers=followup_headers,
            out_nii_path=followup_syn,
            overlap_mode=args.overlap_mode,
        )

        pipeline_followup.pipeline_followup(
            baseline_ID=args.baseline_ID,
            baseline_Inputs=[str(baseline_pred), str(baseline_syn)],
            baseline_DicomDir="",
            baseline_DicomSegDir=str(baseline_dicomseg_dir),
            baseline_json=args.baseline_json,
            followup_ID=args.followup_ID,
            followup_Inputs=[str(followup_pred), str(followup_syn)],
            followup_DicomSegDir=str(followup_dicomseg_dir),
            followup_json=args.followup_json,
            path_output=args.path_output,
            model=args.model,
            path_code=args.path_code,
            path_process=args.path_process,
            path_json=args.path_json,
            path_log=args.path_log,
            gpu_n=int(args.gpu_n),
            fsl_flirt_path=args.fsl_flirt_path,
            upload_json=bool(args.upload_json),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

