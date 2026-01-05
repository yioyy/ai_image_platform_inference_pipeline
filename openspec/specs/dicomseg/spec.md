# dicomseg Specification

## Purpose
TBD - created by archiving change add-aneurysm-mask-im-field. Update Purpose after archive.
## Requirements
### Requirement: Aneurysm mask instance MUST include raw slice index field
系統 MUST 在 Aneurysm 的 platform JSON 輸出中，於 `mask.series[].instances[]` 提供欄位 `im`，其值 MUST 等於該 `dicom_sop_instance_uid` 影像的 DICOM Tag **(0020,0013) Instance Number**。

#### Scenario: im equals DICOM Instance Number for dicom_sop_instance_uid
- WHEN 系統建立 `mask.series[].instances[]` 並填入 `dicom_sop_instance_uid`
- THEN 系統 MUST 從該影像讀取 `(0020,0013) Instance Number`
- AND 將其值輸出至 `instances.im`

### Requirement: Infarct/WMH mask instance MUST allow im field
系統 MUST 在 Infarct 與 WMH 的 platform JSON 輸出中，於 `mask...instances[]` 支援欄位 `im`，以避免 Pydantic 轉型時丟棄上游提供的 `im`。

#### Scenario: keep im during pydantic model validation
- WHEN 上游在 `mask...instances[]` 加入 `im`
- THEN 平台輸出 JSON MUST 保留並輸出 `instances.im`

