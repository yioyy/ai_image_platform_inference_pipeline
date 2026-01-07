# DICOMSEG（Aneurysm）規格增量

## MODIFIED Requirements
### Requirement: Aneurysm mask instance MUST include raw slice index field
系統 MUST 在 Aneurysm 的 platform JSON 輸出中，於 `mask.series[].instances[]` 提供欄位 `im`，其值 MUST 等於該 `dicom_sop_instance_uid` 影像的 DICOM Tag **(0020,0013) Instance Number**。

#### Scenario: im equals DICOM Instance Number for dicom_sop_instance_uid
- WHEN 系統建立 `mask.series[].instances[]` 並填入 `dicom_sop_instance_uid`
- THEN 系統 MUST 從該影像讀取 `(0020,0013) Instance Number`
- AND 將其值輸出至 `instances.im`


