## 1. Schema 修正
- [x] 1.1 更新 `code_ai/pipeline/dicomseg/schema/base.py`：`StudyRequest.parse_date()` 支援 `date/datetime` 與常見字串格式，避免 fallback 成今天

## 2. 梗塞輸出行為補強
- [x] 2.1 更新 `code_ai/pipeline/dicomseg/infarct.py`：Study 代表 DICOM 優先使用 DWI1000（fallback ADC/DWI0）

## 3. 規格更新
- [x] 3.1 更新 `code/openspec/specs/dicomseg/spec.md`：新增 `study.study_date` 不得被覆寫的規格與可接受格式

