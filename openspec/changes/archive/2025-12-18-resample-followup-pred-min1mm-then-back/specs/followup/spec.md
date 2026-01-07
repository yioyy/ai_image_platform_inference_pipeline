# Follow-up 模組規格增量

## ADDED Requirements
### Requirement: min-1mm 模式下 Pred 必須在同一工作空間套用矩陣並回 baseline Pred 原空間
系統 MUST 在啟用 min-1mm 配準模式時，將 followup Pred 先轉入 min-1mm 工作空間，並在該工作空間直接套用配準矩陣，最後輸出 MUST 回到 baseline Pred 的原始 voxel grid，以確保 voxel index overlap 配對正確。

#### Scenario: min-1mm 啟用時的 Pred 處理
- WHEN 系統啟用 min-1mm 配準模式
- THEN 系統 MUST 先將 baseline/followup Pred 轉換到 min-1mm 工作空間
- AND 系統 MUST 確保 SynthSEG 與 Pred 在同一工作空間 voxel grid 一致（必要時 resample SynthSEG 到 Pred grid）
- AND 系統 MUST 在 min-1mm 工作空間直接套用矩陣到 followup Pred
- AND 系統 MUST 將 Pred(reg) resample 回 baseline Pred 的原始 voxel grid


