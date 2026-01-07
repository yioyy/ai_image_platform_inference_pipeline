# 更新 Aneurysm DICOMSEG mask.instances.im 欄位為 DICOM Instance Number

## Why
原本 `mask.series[].instances[].im` 用來存放 `result['main_seg_slice']`（slice index）。目前需求改為：`im` 必須對應到該 `dicom_sop_instance_uid` 影像的 DICOM Tag **(0020,0013) Instance Number**，讓下游可以用 DICOM 原生序號定位切片。

## What Changes
- `mask.series[].instances[].im` 的值改為：對應 `dicom_sop_instance_uid` 那張 DICOM 的 `(0020,0013) Instance Number`
- `im` 欄位階層不變（與 `main_seg_slice` 同層）

## 影響範圍
- `code_ai/pipeline/dicomseg/aneurysm.py`: 更新 `ReviewAneurysmPlatformJSONBuilder.get_mask_instance()` 的 `im` 產生邏輯


