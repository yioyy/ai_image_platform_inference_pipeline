## 1. OpenSpec / 規格
- [ ] 1.1 新增規格增量：Infarct/WMH RADAX pipeline 的輸出結構與 API 通知行為

## 2. 產生器（JSON / DICOMSEG）
- [ ] 2.1 將 `code_ai.pipeline.dicomseg.infarct` 移植到 `radax/code_ai/pipeline/dicomseg/infarct.py`
- [ ] 2.2 將 `code_ai.pipeline.dicomseg.wmh` 移植到 `radax/code_ai/pipeline/dicomseg/wmh.py`
- [ ] 2.3 補齊 `radax/code_ai/pipeline/dicomseg/schema/infarct.py`、`schema/wmh.py`
- [ ] 2.4 確認 `radax/code_ai/pipeline/dicomseg/schema/base.py` 的 `study_date` 能接受 `datetime.date`

## 3. RADAX Infarct Pipeline
- [ ] 3.1 新增 `radax/pipeline_infarct_torch.py`
- [ ] 3.2 複製結果到 `upload_dir/<StudyInstanceUID>/infarct_model/<inference_id>/`
- [ ] 3.3 `prediction.json` 內容為平台 JSON（`<patient_id>_platform_json.json`）
- [ ] 3.4 DICOMSEG 重新命名為 `<seg_series_instance_uid>_<mask_name>.dcm`
- [ ] 3.5 呼叫 `POST /v1/ai-inference/inference-complete`（modelName=infarct_model）
- [ ] 3.6 新增 `radax/pipeline_infarct.sh`（參數與舊版 shell 對齊）

## 4. RADAX WMH Pipeline
- [ ] 4.1 新增 `radax/pipeline_wmh_torch.py`
- [ ] 4.2 複製結果到 `upload_dir/<StudyInstanceUID>/wmh_model/<inference_id>/`
- [ ] 4.3 `prediction.json` 內容為平台 JSON（`<patient_id>_platform_json.json`）
- [ ] 4.4 DICOMSEG 重新命名為 `<seg_series_instance_uid>_<mask_name>.dcm`
- [ ] 4.5 呼叫 `POST /v1/ai-inference/inference-complete`（modelName=wmh_model）
- [ ] 4.6 新增 `radax/pipeline_wmh.sh`（參數與舊版 shell 對齊）

