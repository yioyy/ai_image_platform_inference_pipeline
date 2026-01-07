# Follow-up：min-1mm resample 統一使用 nibabel.processing.conform

## Why
先前 min-1mm resample 使用自建 affine 與 `resample_from_to`，實務上容易因為 header/affine 不一致造成「看起來平移」與 overlap 配對失敗。
參考既有穩定流程（`util_aneurysm.py`），改為統一使用 `nibabel.processing.conform`，確保 resample 後 affine/header 同步更新。

## What Changes
- `registration.py`：min-1mm resample 改用 `nibabel.processing.conform`（label order=0）。
- 測試調整：不再假設 world center 必定完全不動，改以 zoom/shape 與 header 一致性為主。


