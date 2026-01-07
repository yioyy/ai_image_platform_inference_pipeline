# 修正 platform JSON `study.study_date` 被覆寫成今天日期

## 背景 / 問題
在 Infarct platform JSON 生成流程中，pipeline 已正確解析 DICOM `StudyDate`（例如 `20251223` → `2025-12-23`），但輸出的 JSON 檔與上傳 payload 仍會出現「今天日期」。

根因是 `code_ai/pipeline/dicomseg/schema/base.py` 的 `StudyRequest.study_date` Pydantic validator：
- 無法處理上游傳入的 `datetime.date`/`datetime.datetime`
- 解析失敗後進入 `except`，回傳 `now().date()`，導致日期被覆寫

## 目標
- platform JSON 的 `study.study_date` MUST 代表 DICOM StudyDate（或等價日期）
- schema 驗證/序列化 MUST NOT 將有效日期覆寫為今天

## 技術方案（實作摘要）
- 更新 `StudyRequest.parse_date()`：支援 `datetime.date`、`datetime.datetime`、`YYYYMMDD`、`YYYY-MM-DD` 四種輸入
- Infarct 平台 JSON 建立時，Study 代表來源 SHOULD 優先使用 DWI1000（若不存在則 fallback）

## 影響範圍
- `code_ai/pipeline/dicomseg/schema/base.py`：study_date validator
- `code_ai/pipeline/dicomseg/infarct.py`：Study 代表 DICOM 來源選擇與日期來源 fallback（僅梗塞）

