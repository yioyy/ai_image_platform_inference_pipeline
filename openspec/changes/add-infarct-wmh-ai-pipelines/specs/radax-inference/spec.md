# RADAX Inference 交付規格增量（Infarct / WMH）

## ADDED Requirements

### Requirement: Infarct 結果交付（檔案結構）
系統 MUST 將 Infarct 推論結果複製到 `upload_dir/<StudyInstanceUID>/infarct_model/<inference_id>/`。

#### Scenario: 產出 prediction.json
- WHEN Infarct 推論完成（成功）
- THEN 系統 MUST 在 `<inference_id>/prediction.json` 寫入平台 JSON 內容（來源：`JSON/<patient_id>_platform_json.json`）

#### Scenario: 產出 dicom-seg
- WHEN Infarct 推論完成（成功）
- THEN 系統 MUST 將 `Dicom/Dicom-Seg/*.dcm` 複製到 `<inference_id>/`
- AND 系統 MUST 將 DICOMSEG 檔名改為 `<seg_series_instance_uid>_<mask_name>.dcm`

### Requirement: WMH 結果交付（檔案結構）
系統 MUST 將 WMH 推論結果複製到 `upload_dir/<StudyInstanceUID>/wmh_model/<inference_id>/`。

#### Scenario: 產出 prediction.json
- WHEN WMH 推論完成（成功）
- THEN 系統 MUST 在 `<inference_id>/prediction.json` 寫入平台 JSON 內容（來源：`JSON/<patient_id>_platform_json.json`）

#### Scenario: 產出 dicom-seg
- WHEN WMH 推論完成（成功）
- THEN 系統 MUST 將 `Dicom/Dicom-Seg/*.dcm` 複製到 `<inference_id>/`
- AND 系統 MUST 將 DICOMSEG 檔名改為 `<seg_series_instance_uid>_<mask_name>.dcm`

### Requirement: Inference Complete API（Infarct / WMH）
系統 MUST 在推論完成後呼叫 `POST /v1/ai-inference/inference-complete` 通知後端。

#### Scenario: Infarct 推論成功
- WHEN Infarct 推論成功
- THEN 系統 MUST 送出 payload：
  - `studyInstanceUid = <StudyInstanceUID>`
  - `modelName = "infarct_model"`
  - `inferenceId = <inference_id>`

#### Scenario: WMH 推論成功
- WHEN WMH 推論成功
- THEN 系統 MUST 送出 payload：
  - `studyInstanceUid = <StudyInstanceUID>`
  - `modelName = "wmh_model"`
  - `inferenceId = <inference_id>`

#### Scenario: 推論失敗
- WHEN 推論流程發生錯誤或 GPU 記憶體不足導致中止
- THEN 系統 SHOULD 仍嘗試呼叫 inference-complete
- AND 系統 SHOULD 使用 `result="failed"`（並省略 inferenceId）

