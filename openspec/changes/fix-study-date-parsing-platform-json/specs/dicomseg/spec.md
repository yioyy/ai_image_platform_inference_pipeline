# DICOMSEG（platform JSON study_date）規格增量

## ADDED Requirements
### Requirement: platform JSON 的 study.study_date MUST 保留正確的檢查日期
系統 MUST 在 platform JSON 輸出 `study.study_date`，並且其值 MUST 代表 DICOM StudyDate（或等價日期），不得因 schema 驗證/序列化而被覆寫成「今天」。

#### Scenario: study_date 接受多種輸入型別且不覆寫
- WHEN pipeline 計算得到 `study_date` 且其值為 `datetime.date` 或 `datetime.datetime`
- THEN schema 驗證 MUST 保留該日期（datetime 則取 `.date()`）
- AND MUST NOT fallback 成 `now().date()`

#### Scenario: study_date 接受常見字串格式
- WHEN `study_date` 為字串 `YYYYMMDD`（DICOM 常見）
- THEN 系統 MUST 正確解析並輸出為 `YYYY-MM-DD`
- WHEN `study_date` 為字串 `YYYY-MM-DD`
- THEN 系統 MUST 正確解析並輸出為 `YYYY-MM-DD`

#### Scenario: Infarct 平台 JSON 以 DWI1000 優先作為 Study 代表來源
- WHEN Infarct 平台 JSON 需要從 DICOM 讀取 Study 相關欄位（例如 StudyDate/PatientSex/PatientAge）
- THEN 系統 SHOULD 優先使用 DWI1000 序列的第一張影像作為代表來源
- AND 若 DWI1000 不存在，SHOULD 依序 fallback 至 ADC、DWI0 或任一可用序列

