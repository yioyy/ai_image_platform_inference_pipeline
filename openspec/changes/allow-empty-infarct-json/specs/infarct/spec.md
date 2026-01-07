# Infarct DICOMSEG 規格增量

## MODIFIED Requirements
### Requirement: Infarct 平台 JSON 生成
系統 MUST 在無病灶或未產生任何 DICOM-SEG 時，仍成功產生平台 JSON；當存在病灶時，保持既有行為。

#### Scenario: 無病灶 / 無 DICOM-SEG
- WHEN 梗塞遮罩全為 0 或 Excel 病灶數為 0，且未產生任何 DICOM-SEG
- THEN 系統 SHALL 仍建立 study 與 sorted 資訊
- AND 系統 SHALL 將 lession 數量設為 0
- AND 系統 SHALL 產出平台 JSON 且 mask 欄位為空或缺省
- AND 系統 SHALL 不拋出錯誤以中斷流程

#### Scenario: 有病灶
- WHEN 梗塞遮罩包含有效病灶
- THEN 系統 MUST 保持現有行為，產生 DICOM-SEG 與包含 mask 的平台 JSON


# Infarct DICOMSEG 規格增量

## MODIFIED Requirements
### Requirement: Infarct 平台 JSON 生成
系統 MUST 在無病灶或未產生任何 DICOM-SEG 時，仍成功產生平台 JSON；當存在病灶時，保持既有行為。

#### Scenario: 無病灶 / 無 DICOM-SEG
- WHEN 梗塞遮罩全為 0 或 Excel 病灶數為 0，且未產生任何 DICOM-SEG
- THEN 系統 SHALL 仍建立 study 與 sorted 資訊
- AND 系統 SHALL 將 lession 數量設為 0
- AND 系統 SHALL 產出平台 JSON 且 mask 欄位為空或缺省
- AND 系統 SHALL 不拋出錯誤以中斷流程

#### Scenario: 有病灶
- WHEN 梗塞遮罩包含有效病灶
- THEN 系統 MUST 保持現有行為，產生 DICOM-SEG 與包含 mask 的平台 JSON
