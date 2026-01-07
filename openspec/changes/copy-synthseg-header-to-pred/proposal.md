# Follow-up：在 baseline/followup 目錄先將 SynthSEG header 複製到 Pred

## Why
Follow-up 報告的 overlap 配對是以 voxel index 直接計算交集。若 `Pred_<model>.nii.gz` 與 `SynthSEG_<model>.nii.gz` 的 NIfTI 幾何資訊（affine / sform / qform）不一致，即使視覺上疊圖看起來接近，實際上 voxel grid 會錯位，造成 overlap 為 0、JSON 病灶配對失敗。

## What Changes
- 在 `pipeline_followup.py` 將輸入檔複製到 `baseline/` 與 `followup/` 後、進入 registration 之前：
  - 以 `SynthSEG_<model>.nii.gz` 的 header 幾何資訊覆蓋 `Pred_<model>.nii.gz`
  - 保存回原本的 `Pred_<model>.nii.gz`
- 若 Pred/SynthSEG 的 shape 不一致，直接報錯（避免寫入錯誤幾何導致更難追查的錯位）。


