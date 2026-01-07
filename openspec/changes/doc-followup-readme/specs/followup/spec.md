# followup 規格增量

## ADDED Requirements
### Requirement: Follow-up 使用說明（README）
系統 MUST 在專案 README 中提供 Follow-up（檔案導向）流程的最小可用說明，以降低誤用造成的失敗與排錯成本。

#### Scenario: README 描述關鍵流程
- WHEN 使用者查閱 `README.md` 的 Follow-up 小節
- THEN 文件 MUST 說明配準前會以 SynthSEG 的幾何資訊（affine/sform/qform）覆蓋 Pred
- AND MUST 提醒 Pred 與 SynthSEG 的 shape 必須一致
- AND MUST 說明配準依賴 FSL `flirt`（可透過 `--fsl_flirt_path` 指定）

#### Scenario: README 列出主要輸出與標籤定義
- WHEN Follow-up pipeline 執行完成
- THEN 文件 MUST 列出主要輸出檔名：
  - `Pred_<model>_followup.nii.gz`
  - `Followup_<model>_platform_json.json`
- AND MUST 說明 `Pred_<model>_followup.nii.gz` 的標籤定義為（固定）：
  - 1 = new
  - 2 = stable
  - 3 = disappeared


