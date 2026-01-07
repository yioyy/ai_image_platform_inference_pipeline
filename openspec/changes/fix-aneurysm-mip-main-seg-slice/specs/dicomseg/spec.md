# DICOMSEG（Aneurysm / RDX JSON）規格增量

## ADDED Requirements
### Requirement: Reformatted series main_seg_slice MUST be computed from its own prediction volume
系統 MUST 在 `rdx_aneurysm_pred_json.json` 的 `reformatted_series[].detections[].main_seg_slice` 以該 `reformatted_series` 對應的 `*_pred.nii.gz` 計算，而非沿用 `detections[].main_seg_slice`（主序列）。

#### Scenario: MIP_Pitch uses MIP_Pitch_pred.nii.gz
- WHEN 系統輸出 `reformatted_series` 且 `series_description = mip_pitch`
- THEN 系統 MUST 以 `MIP_Pitch_pred.nii.gz` 計算每個病灶的 `main_seg_slice`

#### Scenario: MIP_Yaw uses MIP_Yaw_pred.nii.gz
- WHEN 系統輸出 `reformatted_series` 且 `series_description = mip_yaw`
- THEN 系統 MUST 以 `MIP_Yaw_pred.nii.gz` 計算每個病灶的 `main_seg_slice`

#### Scenario: Missing pred file or missing lesion label
- WHEN 對應的 `*_pred.nii.gz` 不存在，或該病灶在該 pred 中無任何標註
- THEN 系統 MUST 將 `main_seg_slice` 輸出為空字串 `""`，且整體 JSON 生成流程 MUST NOT 失敗


