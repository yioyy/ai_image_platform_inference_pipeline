# 保存配準（registered/resampled）中間影像到 registration/resample/

## Why
目前流程會產生多個對位相關產物（min-1mm 空間的 registered 影像、最終輸出），但部分只存在於暫存或分散在不同位置，不利於人工 QA（例如檢查是否整體平移、翻轉、重疊失敗）。

## What Changes
- 系統在執行 Follow-up 配準時，除了產出既有 `*_registration.nii.gz`，也會將「對位後的 resample/registered 影像」集中保存到 `registration/resample/`：
  - min-1mm 空間的 registered SynthSEG（與對應 `_min1mm.mat`）
  - 最終輸出的 SynthSEG/Pred registration 檔


