# 更新 Follow-up 模組 README 說明

## 問題陳述
目前 `README.md` 的 Follow-up 小節雖已有基本使用方式，但未完整反映 `pipeline_followup.py` 的關鍵流程細節（例如：配準前 Pred 幾何覆蓋、FSL flirt 依賴、baseline DICOM 資料夾可選），容易造成使用者誤解與排錯成本上升。

## 解決方案
在不增加篇幅的前提下，更新 `README.md` 的 Follow-up 小節，補齊：
- 配準前的 Pred 幾何（affine/sform/qform）覆蓋行為與限制（shape 必須一致）
- 配準工具依賴（FSL `flirt`）與主要輸出檔名
- `baseline_DicomDir` 可留空與 dicom-seg 複製行為

## 影響範圍
- 文件：`README.md`
- 規格：新增一則 Follow-up 文件可用性需求（不影響現有演算法與輸出）

## 臨床考量
此變更僅更新文件，不改動任何推論或後處理邏輯；對診斷準確性與患者安全無直接影響，但可降低錯用參數造成的流程失敗風險。

## 技術考量
- 僅文件更新，無執行效能影響
- 維持向後相容：既有參數與輸出不變


