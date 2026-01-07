# Follow-up 模組規格增量

## ADDED Requirements
### Requirement: min-1mm resample 檔案必須可追溯且不可互相覆寫
系統 MUST 在啟用配準前的 min-1mm resample 時，分別保存 baseline 與 followup 的 resample 產物，且 MUST NOT 發生檔名覆寫。

#### Scenario: baseline 與 followup 的輸入檔名相同
- WHEN baseline 與 followup 的 SynthSEG 輸入檔名相同（例如皆為 `SynthSEG_<model>.nii.gz`）
- THEN 系統 MUST 仍能分別保存兩份 resample 後的 NIfTI（baseline 與 followup 各一份）
- AND 系統 MUST NOT 因檔名覆寫導致配準矩陣或輸出錯誤


