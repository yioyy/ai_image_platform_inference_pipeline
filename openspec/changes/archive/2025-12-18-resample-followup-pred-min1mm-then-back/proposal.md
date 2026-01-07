# Follow-up：Pred 也進入 min-1mm 工作空間，配準後再回 baseline 原空間

## Why
目前配準矩陣是以 SynthSEG 估計，若 Pred 與 SynthSEG 在 baseline/followup 各自的 affine/grid 不完全一致（常見於 origin 不同），即使視覺上對位成功，仍可能在 voxel index 上完全錯開，導致 follow-up 病灶 overlap 配對為 0。

為降低「中心點/平移錯置」與跨 grid 套矩陣造成的不確定性，我們改為：
- followup 的 Pred 也先進入 min-1mm 工作空間
- 在同一工作空間內直接套用配準矩陣
- 最後再 resample 回 baseline 原空間（以 voxel index 正確對齊 baseline Pred）

## What Changes
- 若啟用 min-1mm：
  - baseline/followup 的 Pred 先用 `nibabel.processing.conform` 做 min-1mm（label 使用 order=0）
  - SynthSEG 會 resample 到各自 Pred(min1mm) 的 voxel grid，以確保 Pred/SynthSEG 在同一工作空間
  - 估矩陣仍使用 SynthSEG(min1mm)
  - 套用矩陣時，直接對 Pred(min1mm) 套用
  - 最終 Pred(reg) 會 resample 回 baseline Pred 原始 voxel grid，確保 overlap 以 voxel index 計算仍可正確配對


