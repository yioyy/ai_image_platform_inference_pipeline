# Follow-up v3 API 規格增量

## ADDED Requirements
### Requirement: 同步 API 呼叫
系統 MUST 提供同步 API 端點以觸發 Follow-up v3 比對（radax v3）。

#### Scenario: API 路徑與方法
- WHEN 平台呼叫 API
- THEN 系統 MUST 使用 `POST /api/radax/followup`
- AND Content-Type MUST 為 `application/json`

#### Scenario: 傳輸形式
- WHEN 平台呼叫 API
- THEN 系統 MUST 以 JSON body 接收輸入
- AND 系統 MUST 以 JSON body 回傳結果

#### Scenario: 成功呼叫並回傳結果
- WHEN 平台提交有效的 JSON body（對齊 `follow_up_input.json` 結構）
- THEN 系統回傳對應的 `followup_manifest` JSON body

#### Scenario: 多模型輸入
- WHEN 輸入內包含多個 `model_id`
- THEN 系統逐模型產生 follow-up 結果並完整回傳

### Requirement: 輸入 JSON 驗證
系統 MUST 驗證輸入 JSON body 之必要欄位，至少包含：
- `current_study`
- `prior_study_list`
- `model_id`
- `current_study.study_date`
- `prior_study_list[].study_date`

#### Scenario: 欄位缺漏
- WHEN 輸入缺少必要欄位
- THEN 系統回傳錯誤回應並指出缺漏欄位

#### Scenario: 請求範例
- WHEN 平台送出標準請求
- THEN JSON body 可為：
```
{
  "model_id": "924d1538-597c-41d6-bc27-4b0b359111cf",
  "current_study": {
    "patient_id": "05730207",
    "study_date": "20240101",
    "study_instance_uid": "1.2.840.113619.2.408.5554020.1",
    "series_instance_uid": "1.2.840.113619.2.408.5554020.2"
  },
  "prior_study_list": [
    {
      "patient_id": "05730207",
      "study_date": "20221227",
      "study_instance_uid": "1.2.840.113619.2.408.5554020.3",
      "series_instance_uid": "1.2.840.113619.2.408.5554020.4"
    }
  ]
}
```

### Requirement: 回應格式
系統 MUST 回傳 `followup_manifest` JSON body，且 `items[]` 內為完整 grouped prediction（包含 `follow_up[]` 與 `sorted` 欄位），格式需對齊 `grouped_schema.md` 與 `manifest_schema.md`。

#### Scenario: 無可比對資料
- WHEN `prior_study_list` 為空
- THEN 系統回傳可辨識的空結果或錯誤訊息

#### Scenario: 回應範例
- WHEN 回傳成功結果
- THEN JSON body 可為：
```
{
  "manifest": {
    "inference_id": "1236156b-b9c6-43e8-bbaa-e866e9b8599d",
    "inference_timestamp": "2026-01-21T14:30:00Z",
    "model_id": "924d1538-597c-41d6-bc27-4b0b359111cf",
    "patient_id": "05730207"
  },
  "items": [
    {
      "inference_id": "1236156b-b9c6-43e8-bbaa-e866e9b8599d",
      "model_id": "924d1538-597c-41d6-bc27-4b0b359111cf",
      "patient_id": "05730207",
      "detections": [],
      "follow_up": [],
      "sorted": []
    }
  ]
}
```

### Requirement: 錯誤回應
系統 MUST 提供結構化錯誤回應，至少包含：
- `code`
- `message`
- `details`（可選）

#### Scenario: 錯誤碼清單
- WHEN 發生錯誤
- THEN 系統 MUST 依情況回傳下列錯誤碼：
  - 400 Bad Request（欄位缺漏/格式錯誤）
  - 404 Not Found（必要檔案缺失）
  - 408 Request Timeout（處理逾時）
  - 409 Conflict（資料狀態衝突）
  - 500 Internal Server Error（未預期錯誤）
  - 503 Service Unavailable（服務忙碌）

#### Scenario: 錯誤回應範例
- WHEN 輸入欄位缺漏
- THEN 系統回傳：
```
{
  "code": "BAD_REQUEST",
  "message": "Missing required fields",
  "details": ["current_study", "prior_study_list"]
}
```

- WHEN 必要檔案缺失
- THEN 系統回傳：
```
{
  "code": "NOT_FOUND",
  "message": "Required files missing",
  "details": ["Pred_Aneurysm.nii.gz", "Pred_Aneurysm_rdx_aneurysm_pred_json.json"]
}
```

- WHEN 處理逾時
- THEN 系統回傳：
```
{
  "code": "TIMEOUT",
  "message": "Follow-up processing timeout",
  "details": {"timeout_seconds": 300}
}
```

- WHEN 發生資料狀態衝突
- THEN 系統回傳：
```
{
  "code": "CONFLICT",
  "message": "Study date conflict",
  "details": {"patient_id": "05730207", "study_date": "20240101"}
}
```

- WHEN 未預期錯誤
- THEN 系統回傳：
```
{
  "code": "INTERNAL_ERROR",
  "message": "Unexpected error",
  "details": {"trace_id": "trace-12345"}
}
```

- WHEN 服務忙碌
- THEN 系統回傳：
```
{
  "code": "SERVICE_UNAVAILABLE",
  "message": "Service is busy",
  "details": {"retry_after_seconds": 30}
}
```

#### Scenario: 檔案缺失
- WHEN 必要的 Pred/SynthSEG/prediction JSON 缺失
- THEN 系統回傳錯誤回應並列出缺失清單

### Requirement: Log 紀錄
系統 MUST 記錄 API 每件事的處理紀錄，並寫入固定路徑與日期檔名。

#### Scenario: Log 位置與命名
- WHEN 產生 log
- THEN 系統 MUST 寫入 `/home/david/pipeline/chuan/log/YYYYMMDD.log`

#### Scenario: Log 內容格式
- WHEN 記錄事件
- THEN 每行 log MUST 包含日期時間戳與事件名稱
- AND SHOULD 包含 request_id 與狀態碼

#### Scenario: Log 範例
- WHEN 成功處理
- THEN log 可為：
```
2026-01-21T14:30:00+08:00 INFO request_id=abc123 action=followup_process status=success
```

### Requirement: 認證相容性
系統 MUST 允許在未啟用認證的環境下存取此 API，並保留後續可擴充認證的空間。
