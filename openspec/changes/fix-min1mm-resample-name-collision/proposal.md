# 修正 min-1mm resample 檔名覆寫，並分別保存 baseline/followup resample 檔

## Why
Follow-up registration 在執行 min-1mm resample 時，baseline 與 followup 的輸入檔名常常相同（例如都叫 `SynthSEG_CMB.nii.gz`）。
若直接用原檔名加 suffix 產生暫存檔，會導致 **baseline 與 followup 的 resample 輸出互相覆寫**，進而造成配準矩陣與輸出結果錯誤（包含疑似 z 軸方向錯置、病灶重疊失效等現象）。

同時，為了便於人工 QA，我們需要將 baseline 與 followup 的 resample 檔案 **分別保存**，而非只留下單一暫存檔。

## What Changes
- min-1mm resample 暫存檔名加入 `baseline_` / `followup_` 前綴，避免覆寫。
- 於 `registration/resample/` 額外保存兩份 resample 後的 SynthSEG NIfTI（baseline 與 followup 各一份），方便對照。


