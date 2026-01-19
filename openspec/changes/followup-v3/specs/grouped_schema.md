# Grouped Prediction + Follow-up Schema（v3）

本文件定義 prediction.json（既有欄位）加上 `follow_up` 的最小欄位清單。

## 必要欄位（prediction.json 原有）
- `inference_id` (string)
- `inference_timestamp` (string, ISO 8601)
- `input_study_instance_uid` (string[])
- `input_series_instance_uid` (string[])
- `model_id` (string)
- `patient_id` (string)
- `detections` (array)

## 新增欄位：follow_up
```
follow_up: [
  {
    "current_study_date": "20221227",
    "prior_study_date": "20201022",
    "follow_up_timestamp": "2026-01-08T05:59:57.477056+00:00",
    "prior_study_instance_uid": "1.2.840....",
    "current_series_instance_uid": "1.2.826....",
    "prior_series_instance_uid": "1.2.826....",
    "status": {
      "new": [
        {
          "current_mask_index": 12,
          "current_main_seg_slice": 35,
          "prior_main_seg_slice": 37
        }
      ],
      "stable": [
        {
          "current_mask_index": 7,
          "current_main_seg_slice": 33,
          "prior_mask_index": 2,
          "prior_main_seg_slice": 32
        }
      ],
      "regress": [
        {
          "prior_mask_index": 5,
          "current_main_seg_slice": 40,
          "prior_main_seg_slice": 42
        }
      ]
    },
    "sorted": [
      {"current_slice": 0, "prior_slice_list": [0]},
      {"current_slice": 1, "prior_slice_list": [1,2]}
    ]
  }
]
```

### 狀態欄位規則
- `new`：只需 `current_mask_index`，但需同時提供 `current_main_seg_slice` 與推回的 `prior_main_seg_slice`
- `stable`：需 `current_mask_index` + `prior_mask_index`，兩邊 slice 都填
- `regress`：只需 `prior_mask_index`，但需同時提供 `prior_main_seg_slice` 與推回的 `current_main_seg_slice`

### 主體規則
- follow_up 以日期新者為主體（current），`current_study_date` 必須大於 `prior_study_date`

### sorted 欄位
以 current 為主體，一對多對應：
建議使用 `{"current_slice": int, "prior_slice_list": int[]}`。
如需相容舊格式，可提供轉換器或輸出雙格式。

