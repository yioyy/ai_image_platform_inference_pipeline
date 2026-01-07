# Follow-up 模組規格增量

## ADDED Requirements
### Requirement: min-1mm resample MUST 使用 conform 並更新幾何 header
系統 MUST 在執行 min-1mm resample 時使用 `nibabel.processing.conform`，並確保輸出 NIfTI 的 affine/header 反映新的 voxel size 與 shape，以避免後續配準與 overlap 計算發生錯位。

#### Scenario: min-1mm resample 使用 conform
- WHEN 系統需要將 label 影像做 min-1mm resample
- THEN 系統 MUST 使用 `nibabel.processing.conform`
- AND label resample MUST 使用最近鄰插值（order=0）


