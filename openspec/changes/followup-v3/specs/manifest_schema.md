# Follow-up Manifest Schema（v3）

目的：一次觸發回傳多份 prediction / follow_up 結果，避免多次通知後端 API。

## 檔名
`followup_manifest.json`

## 建議格式（manifest + items；prediction 本體完全不動）
```
{
  "manifest": {
    "inference_id": "1236156b-b9c6-43e8-bbaa-e866e9b8599d",
    "inference_timestamp": "2026-01-19T14:30:00Z",
    "model_id": "924d1538-597c-41d6-bc27-4b0b359111cf",
    "patient_id": "05730207"
  },
  "items": [
    {
      "...": "完整 prediction-grouped.json 內容（含 detections 與 follow_up）"
    },
    {
      "...": "完整 prediction-grouped.json 內容（含 detections 與 follow_up）"
    }
  ]
}
```

## 單次通知建議 payload
```
{
  "manifestPath": "<followup_manifest.json 的路徑或 URL>",
  "result": "success"
}
```

## 說明
- 主體規則一律以日期新者為主體，manifest 只列出「新日期為主體」的結果
- `items` 每個 item 就是一份完整 `prediction-grouped.json`（不可刪減）
- `manifest.inference_id`：本次觸發 follow-up 的推論 ID（通常對應新進資料）
- `manifest.inference_timestamp`：本次觸發 follow-up 的時間（ISO 8601）
- `manifest.model_id`：模型識別碼（與 items 內 prediction 一致）
- `manifest.patient_id`：病人 ID（與 items 內 prediction 一致）

