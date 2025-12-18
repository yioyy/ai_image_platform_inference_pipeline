# followup 規格增量

## MODIFIED Requirements

### Requirement: 病灶配對追蹤
系統 MUST 正確配對前後檢查中的相同病灶，並在重複產出 follow-up JSON 時維持關鍵張 SOP UID 的穩定性。

#### Scenario: 重疊（配對成功）病灶的 followup DICOM SOP UID 穩定性
- WHEN baseline 病灶與 followup 病灶配對成功（重疊）
- AND baseline JSON（之前 JSON）已存在該病灶的 `followup_dicom_sop_instance_uid`
- THEN 系統 MUST 沿用該 `followup_dicom_sop_instance_uid`
- AND MUST NOT 以 followup JSON 內的 `dicom_sop_instance_uid` 覆寫之

#### Scenario: 只有 new/disappeared 需要用配準矩陣推估關鍵張
- WHEN 病灶狀態為 `new`
- THEN 系統 MUST 使用配準矩陣（必要時取反矩陣）推估 `followup_dicom_sop_instance_uid`

- WHEN 病灶狀態為 `disappeared`
- THEN 系統 MUST 使用配準矩陣（必要時取反矩陣）推估 `dicom_sop_instance_uid`


