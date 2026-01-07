# Follow-up 模組規格增量

## MODIFIED Requirements
### Requirement: 配準前 sub-mm spacing 正規化為 1mm 並回原空間輸出
系統 MUST 在執行 Follow-up → Baseline 線性配準前，針對 sub-mm 影像進行解析度正規化，以縮短配準時間，且最終輸出 MUST 保持在 baseline 原空間。

#### Scenario: baseline 或 followup 任一軸 spacing < 1mm（逐軸 min-1mm）
- WHEN baseline 或 followup 的 NIfTI 在 x/y/z 任一軸 spacing < 1mm
- THEN 系統 MUST 先將 baseline 與 followup 的 SynthSEG label 影像做逐軸重採樣：僅將 spacing < 1mm 的軸重採樣至 1mm，其餘軸保持不變
- AND 重採樣 label MUST 使用最近鄰插值（nearest neighbour）
- AND 系統 MUST 在重採樣後的空間估計配準矩陣
- AND 系統 MUST 將該矩陣換算回「原始影像 grid」對應的 FLIRT `.mat`
- AND 系統 MUST 使用換算後的 `.mat` 將 followup SynthSEG/Pred 套用到 baseline 原空間
- AND 產出的 `SynthSEG_<model>_registration.nii.gz` 與 `Pred_<model>_registration.nii.gz` MUST 與 baseline 影像在 shape 上一致


