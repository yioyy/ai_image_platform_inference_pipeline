# 新增 RADAX Infarct/WMH AI Pipeline（結果複製 + inference-complete API）

## 問題陳述
目前 `radax/` 僅有 Aneurysm/Vessel 的 RADAX 版上傳流程（複製結果到指定資料夾 + 呼叫 inference-complete API）。  
Infarct 與 WMH 的現有流程主要仍沿用舊的 aiteam/orthanc 上傳方式，無法滿足 RADAX 後端的 ingest 規格。

## 解決方案
新增兩支 RADAX 版 pipeline：

- `radax/pipeline_infarct_torch.py` + `radax/pipeline_infarct.sh`
- `radax/pipeline_wmh_torch.py` + `radax/pipeline_wmh.sh`

流程前半段（推理、前後處理、產生 Excel/JSON/DICOMSEG）沿用既有行為；  
最後一步改為：

1. 產出平台 JSON（`JSON/<patient_id>_platform_json.json`）與 DICOMSEG（`Dicom/Dicom-Seg/*.dcm`）
2. 複製結果到：
   - `upload_dir/<StudyInstanceUID>/infarct_model/<inference_id>/...`
   - `upload_dir/<StudyInstanceUID>/wmh_model/<inference_id>/...`
3. 呼叫 `POST /v1/ai-inference/inference-complete` 通知後端完成

## 影響範圍
- 新增 `radax/` 兩支 pipeline 入口與 `.sh` wrapper
- `radax/code_ai/pipeline/dicomseg/` 補齊 Infarct/WMH 的平台 JSON 與 DICOMSEG 產生器（讓 radax 可直接 import）
- 不修改 aneurysm/vessel 行為

## 臨床考量
- 不改變推理模型輸出本身（僅改變結果交付方式）
- 強制以 `StudyInstanceUID` 作為最外層資料夾鍵值，避免跨個案混用

## 技術考量
- `upload_dir` 與 `api_url` 需可透過 CLI 參數調整，以適應不同部署環境
- 失敗時仍應嘗試發送 inference-complete（result=failed），避免後端卡住狀態機

