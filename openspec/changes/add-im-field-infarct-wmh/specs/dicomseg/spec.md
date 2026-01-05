# DICOMSEG（Infarct / WMH）規格增量

## ADDED Requirements
### Requirement: Infarct/WMH mask instance MUST allow im field
系統 MUST 在 Infarct 與 WMH 的 platform JSON 輸出中，於 `mask...instances[]` 支援欄位 `im`，以避免 Pydantic 轉型時丟棄上游提供的 `im`。

#### Scenario: keep im during pydantic model validation
- WHEN 上游在 `mask...instances[]` 加入 `im`
- THEN 平台輸出 JSON MUST 保留並輸出 `instances.im`


