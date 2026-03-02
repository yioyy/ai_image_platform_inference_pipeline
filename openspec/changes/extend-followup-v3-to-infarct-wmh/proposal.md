# 延伸 followup-v3 到 Infarct/WMH pipeline

## 目標
- 讓 `pipeline_infarct_torch.py` 與 `pipeline_wmh_torch.py` 在有提供 `--input_json`（檔案路徑或 JSON 內容）時，能執行 `pipeline_followup_v3_platform.py` 產生 followup 平台 JSON，並依結果上傳（有 followup 就上傳 followup；否則上傳原始平台 JSON）。

## 背景
- 目前 followup-v3 platform 流程已在動脈瘤 pipeline (`pipeline_aneurysm_tensorflow.py`) 串接並可上傳。
- Infarct/WMH 流程雖能產生與上傳平台 JSON，但尚未支援 followup-v3 平台 JSON 生成與上傳。

## 解法概述
- 在兩支 pipeline 增加 `--input_json` 參數與對應處理：
  - 支援 `input_json` 為「檔案路徑」或「直接 JSON 字串」。
  - 生成/確保 followup 所需檔名（`Pred_<Model>.nii.gz`、`SynthSEG_<Model>.nii.gz`、`Pred_<Model>_platform_json.json`）存在於 followup case 目錄中。
  - 呼叫 `pipeline_followup_v3_platform()` 取得輸出 JSON 路徑清單並上傳。

## 影響範圍
- `code/pipeline_infarct_torch.py`
- `code/pipeline_wmh_torch.py`

