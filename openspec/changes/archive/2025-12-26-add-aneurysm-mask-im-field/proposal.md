# 新增 Aneurysm DICOMSEG mask.instances.im 欄位

## Why
目前 Aneurysm pipeline 最後輸出的 platform JSON 中，`mask.series[].instances[]` 只有 `main_seg_slice`（且在 review 流程中會被反向計算）。下游需要保留「原始的 main_seg_slice（未反向）」以供其他用途，但目前 JSON 沒有欄位可承載。

## What Changes
- 在 Aneurysm 的 `mask.instances[]` 中新增欄位 `im`
- `im` 與 `main_seg_slice` 位於相同階層
- `im` 的值為 `result['main_seg_slice']`（不做 `len(source_images) - ...` 反向計算）

## 影響範圍
- `code_ai/pipeline/dicomseg/aneurysm.py`: 產出 mask instance 時新增 `im`
- `code_ai/pipeline/dicomseg/schema/aneurysm.py`: schema 新增 `im` 欄位以確保 JSON 輸出包含該欄位

## 臨床考量
不改變 segmentation 結果與 DICOM-SEG 內容，僅增加額外的索引資訊欄位，對臨床判讀不造成風險。

## 技術考量
- `im` 欄位為新增、向後相容（不影響既有 consumer）。
- 維持既有 `main_seg_slice` 反向邏輯不變，避免破壞現行前端/平台使用方式。


