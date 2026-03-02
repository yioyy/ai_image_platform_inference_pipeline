## 1. 規格 / 介面
- [ ] 1.1 `pipeline_infarct_torch.py` 新增 `--input_json` 參數（檔案路徑或 JSON 內容）
- [ ] 1.2 `pipeline_wmh_torch.py` 新增 `--input_json` 參數（檔案路徑或 JSON 內容）

## 2. 實作
- [ ] 2.1 Infarct：串接 followup-v3 platform，並依結果上傳 followup/原始平台 JSON
- [ ] 2.2 WMH：串接 followup-v3 platform，並依結果上傳 followup/原始平台 JSON
- [ ] 2.3 在 followup case 目錄補齊必要檔名（Pred/SynthSEG/platform JSON）

## 3. 驗證
- [ ] 3.1 基本語法檢查（兩支 pipeline 可成功啟動並解析參數）
- [ ] 3.2 `openspec validate extend-followup-v3-to-infarct-wmh`

