# 允許無病灶時仍輸出 Infarct 平台 JSON

## 問題陳述
- 目前梗塞流程在沒有產出任何 DICOM-SEG（無病灶或遮罩全為 0）時，會在 `execute_dicomseg_platform_json` 拋出 `ValueError: 無法產生任何 DICOM-SEG`，導致 `after_run_infarct` 失敗且無法回報「無病灶」結果。

## 解決方案
- 調整 infarct DICOM-SEG / JSON 生成邏輯：當沒有病灶遮罩時，不產生 DICOM-SEG，但仍產生平台 JSON，`mask` 欄位為空（或 null），`lession` 數量為 0。
- 保留 study / sorted 資訊，讓前端 / 平台可接收「無病灶」的成功結果。

## 影響範圍
- `code_ai/pipeline/dicomseg/infarct.py`（DICOM-SEG 與 JSON 生成邏輯）
- `after_run_infarct.py` 呼叫路徑將不再因無病灶而失敗

## 臨床考量
- 明確回傳「無病灶」以避免臨床流程中斷。
- 不產出空的 DICOM-SEG 物件，避免 PACS 顯示異常。

## 技術考量
- `mask` 可為空或 null，需符合現有 schema 容忍度。
- 保持向後相容：有病灶時行為不變。

