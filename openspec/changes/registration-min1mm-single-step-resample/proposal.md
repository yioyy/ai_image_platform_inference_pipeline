# Follow-up registration：min-1mm resample 改為一次性重採樣

## Why
目前 `registration.py` 的 label min-1mm 重採樣雖然行為正確，但採逐軸連續 downsample 的方式實作。
參考既有程式（`util_aneurysm.py`）的作法，我們希望改成：
- 先判斷哪些軸需要調整 spacing
- 再以「一次性 resample」完成所有需要變動的軸

這可降低實作複雜度、避免軸順序造成的細微差異，並更貼近既有團隊習慣。

## What Changes
- `registration.py`：min-1mm label resample 改用 nibabel 的 processing（order=0）一次完成 resample（target spacing/shape 仍由 min-1mm 規則計算）。
- 行為不變：仍是「僅將 sub-mm 軸升至 1mm，其餘軸不動」，輸出回 baseline 原空間。


