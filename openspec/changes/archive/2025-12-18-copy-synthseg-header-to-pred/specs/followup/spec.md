# Follow-up 模組規格增量

## ADDED Requirements
### Requirement: 配準前 Pred 必須與 SynthSEG 使用相同的幾何 header
系統 MUST 在執行 Follow-up 配準前，確保 baseline 與 followup 的 `Pred_<model>.nii.gz` 與對應的 `SynthSEG_<model>.nii.gz` 使用相同的幾何資訊（affine / sform / qform），以避免 voxel grid 錯位導致 overlap 配對失敗。

#### Scenario: 覆蓋 Pred 幾何資訊
- WHEN 系統已在 `baseline/` 與 `followup/` 目錄準備好 `Pred_<model>.nii.gz` 與 `SynthSEG_<model>.nii.gz`
- THEN 系統 MUST 以 `SynthSEG_<model>.nii.gz` 的幾何資訊覆蓋 `Pred_<model>.nii.gz`
- AND 系統 MUST 保存更新後的 `Pred_<model>.nii.gz`

#### Scenario: shape 不一致
- WHEN `Pred_<model>.nii.gz` 與 `SynthSEG_<model>.nii.gz` 的 3D shape 不一致
- THEN 系統 MUST 立即中止並回報錯誤


