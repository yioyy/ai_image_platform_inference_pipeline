# 修正 Follow-up 檔名規則（model 大小寫、JSON 命名、MAT 命名、process 目錄）

## 問題陳述
Follow-up 模組近期調整了預設參數與輸出命名規則：
- `--path_process` default 改為 `.../process/Deep_Followup/`
- Follow-up 輸出 JSON 檔名改為 `Followup_<model>_platform_json.json`（檔名包含 model，且保留大小寫，例如 `CMB`）
- 配準矩陣檔名改為 `baseline_<model>.mat`

目前文件與部分程式碼仍使用舊命名（例如 `Followup_platform_json.json`、`<baseline_ID>_<model>.mat`、`process/Deep_FollowUp/...`），造成使用者與平台整合混淆。

## 解決方案
以 `pipeline_followup.py` 的 CLI default（parser 設定）為唯一依據，統一修正：
- 主流程輸出檔名（JSON、MAT）
- 處理資料夾根目錄（process root）推導規則：支援新舊 `path_process` 傳入方式
- 同步更新 README 與 OpenSpec 文件描述

## 影響範圍
- **程式碼**
  - `pipeline_followup.py`：輸出 JSON 與 `.mat` 命名、process root 組合邏輯
- **文件**
  - `README.md`
  - `openspec/changes/archive/2025-12-12-add-followup-comparison-module/proposal.md`
  - `openspec/changes/archive/2025-12-12-add-followup-comparison-module/tasks.md`

## 臨床/技術考量
- **向後相容性**：保留舊 `path_process` 的行為（若使用者仍傳入 `.../Followup/`，系統仍會在其下建立 `Deep_FollowUp/<baseline_id>/`）
- **可追溯性**：`.mat` 以 `baseline_<model>.mat` 命名，便於同一個 baseline case 下跨 model 管線一致存取


