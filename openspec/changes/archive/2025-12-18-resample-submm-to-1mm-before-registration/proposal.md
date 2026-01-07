# 配準前將 sub-mm spacing 影像降採樣至 1mm，再回原空間輸出

## Why
目前 Follow-up 配準使用 FSL FLIRT，若 baseline / followup 的 NIfTI 任一軸向 spacing < 1mm（sub-mm 高解析），配準耗時會顯著增加，且在嚴重腦中風（大病灶/缺損/腦室擴大）個案上更容易出現不穩定或卡住的最佳化行為。

## What Changes
在執行 FLIRT 估計配準矩陣前：
- 若 baseline 或 followup 的任一軸向 spacing < 1mm，先將兩者的 SynthSEG label 影像重採樣為 1mm isotropic（使用 label 友善的最近鄰）。
- 在 1mm 空間估計 `.mat` 後，再將矩陣換算回「原始影像 grid」對應的 FLIRT `.mat`，並用該矩陣套用到原始 followup SynthSEG/Pred，輸出仍在 baseline 原空間（不改既有輸出 shape/spacing）。

## 影響範圍
- `registration.py`：新增 spacing 檢測、1mm 暫存檔產生、矩陣換算與套用流程。
- `tests/`：新增單元測試以驗證矩陣在不同 voxel grid 間的換算公式。
- Follow-up pipeline 行為：對外輸出檔名與目錄結構不變；僅在 sub-mm 影像時改用加速策略。

## 臨床考量
- 嚴重腦中風病人常伴隨顯著腦結構改變，配準本身更困難。本變更僅做解析度正規化以降低計算負擔，並維持輸出在 baseline 原空間，避免後續病灶配對與臨床報告流程受影響。
- 標籤影像（腦區分割 33 類以上）必須使用最近鄰插值，避免產生非整數標籤值。

## 技術考量
- 以 nibabel 讀取 NIfTI spacing（header zooms）與 affine；用線性代數將 1mm 空間的 FLIRT 矩陣換算回原始影像 grid 的 FLIRT 矩陣，確保既有 `.mat` 消費者不需修改。
- 暫存 1mm 檔案放在 registration output 目錄下的私有子資料夾，並遵循 overwrite 行為。


