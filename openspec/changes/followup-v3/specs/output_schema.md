# Follow-up v3 輸出格式（platform_json）

本文件定義 `Pred_CMB_platform_json.json` 經過 followup-v3 模組後，輸出成
`Followup_CMB_platform_json_new.json` 的欄位變化與新增規則。

## 1. 檔案對照
- 輸入：`process/Pred_CMB_platform_json.json`
- 輸出：`process/Deep_FollowUp/.../Followup_CMB_platform_json_new.json`

## 2. 變更摘要
- 原本結構 `study` / `sorted` / `mask` **保留不動**
- 新增欄位：
  - `study.model[].followup[]`
  - `mask.model[].series[].instances[].followup[]`
  - `sorted_slice[]`（根層級）

## 2.1 欄位來源對照（輸入 → 輸出）
- `study.model[].followup[]`：來自 `needFollowup[].studyInstanceUid` 與 `study_date` 的組合
- `mask...followup[]`：來自 `needFollowup[].models[].mask[]` 的病灶資訊
- `sorted_slice[]`：由 NIfTI 對位產生的 slice 對應表（參考 followup_manifest 的 sorted）

## 3. study.model[].followup[]
用來描述「current study」對應到的多筆 follow-up 研究。

### schema（單筆）
```
{
  "current_series_instance_uid": "string",
  "followup_study_date": "YYYY-MM-DD",
  "followup_study_instance_uid": "string",
  "followup_series_instance_uid": "string",
  "followup_series_type": number
}
```

### 規則
- `current_series_instance_uid`：等於當前 study 的 series uid
- `followup_*`：對應每一個 prior study
- `followup_study_date` 必填，格式 `YYYY-MM-DD`
- 多筆 followup 建議依 `followup_study_date` 由新到舊排序
- 若 `followup_*` 無對應資料，可填空字串，但 `followup_study_date` 仍必填

## 4. mask.model[].series[].instances[].followup[]
用來描述「單顆病灶」對應到每一個 prior study 的變化狀態。

### schema（單筆）
```
{
  "mask_index": "string",
  "old_diameter": "string",
  "old_volume": "string",
  "status": "new | stable | regress",
  "study_date": "YYYY-MM-DD",
  "main_seg_slice": number,
  "jump_study_instance_uid": "string",
  "jump_series_instance_uid": "string",
  "jump_sop_instance_uid": "string",
  "seg_series_instance_uid": "string",
  "seg_sop_instance_uid": "string",
  "is_ai": "0 | 1"
}
```

### 規則
- `status`：`new` / `stable` / `regress`
- `study_date`：對應的 prior study 日期
- `mask_index`：
  - `new`：可為空字串
  - `stable` / `regress`：填入 prior study 的 `mask_index`
- `old_diameter` / `old_volume`：對應 prior study 的病灶資料（無資料可為空）
- `main_seg_slice`：
  - `stable` / `regress`：直接沿用 followup 端原 `main_seg_slice`
  - `new`：使用矩陣轉換後的 z 值
- 若為 `regress` 新增的最新日期 instance（placeholder）：
  - 最新日期原本沒有該病灶的 `instances`，需新增一筆全部空值的 instance
  - 該 instance 的 `followup[]` 放入一筆 `regress` 記錄
  - 透過這筆 `followup` 反推最新日期的 `instances[].dicom_sop_instance_uid` 與 `instances[].main_seg_slice`
  - `instances[].mask_index` 與 `instances[].is_main_seg` 為空字串
- `jump_*`：前端點選跳轉用 DICOM UID
- `seg_*`：對應 prior study 的 DICOM-SEG UID（無資料可為空）
- `is_ai`：沿用原始值（`"1"` 或 `"0"`）
- `jump_*` 來源為 prior study 對應的 DICOM UID
- `seg_*` 來源為 prior study 的 DICOM-SEG UID

## 5. sorted 欄位
`sorted` 區塊維持輸入檔原格式，不新增欄位。

## 6. sorted_slice 欄位（新增，根層級）
`sorted_slice` 與 `sorted` 同階級，提供 follow-up 的 slice 對應表（雙向嚴格對照）。
同一筆 `sorted_slice` 需標示 `model_type` 與 UID，以定義是哪個 model 的對應。

### schema（單筆）
```
{
  "model_type": number,
  "current_study_instance_uid": "string",
  "current_series_instance_uid": "string",
  "followup_study_instance_uid": "string",
  "followup_series_instance_uid": "string",
  "sorted": [
    {"current_slice": 1, "prior_slice": 2},
    {"current_slice": 2, "prior_slice": 3}
  ],
  "reverse-sorted": [
    {"prior_slice": 1, "current_slice": 1},
    {"prior_slice": 2, "current_slice": 1},
    {"prior_slice": 3, "current_slice": 2}
  ]
}
```

### 規則
- `sorted_slice` 為 list，代表多個 model 底下各一筆（每筆為一組 current/prior 對應）。
- `model_type`：主體模型類別，對應 `study.model[].model_type`
- `current_*` 與 `followup_*` 以 UID 區分每一個 pair。
- `sorted`：以 current_date 滾輪為主時的對照表（`current_slice -> prior_slice`）
- `reverse-sorted`：以 prior_date 滾輪為主時的對照表（`prior_slice -> current_slice`）
- `current_slice` / `prior_slice` 從 1 開始，且為嚴格一對一對照
- 對照表不可出現空值，需依規則補齊缺值（首段/尾段/中段）：
  - 若最前段為空，補第一個可用對應值（預設補 1）
  - 若最後段為空，補最後一個可用對應值
  - 若中段為空，比較上方已補值的最後一筆與下方已補值的第一筆距離，補較近者
  - 若候選值多於 2 筆，先取候選值中位數，再選最接近中位數的已標註值
- 結論：若對位矩陣為 prior -> current，則 `sorted` 應使用 inverse matrix 形成 current -> prior；`reverse-sorted` 使用原始矩陣形成 prior -> current。若方向相反屬於錯誤輸出。
- 若有指定 `model`，只輸出該 `model_type` 對應的 `sorted_slice` 記錄

## 7. 範例（節錄）
```
{
  "study": {
    "model": [
      {
        "model_type": "2",
        "followup": [
          {
            "current_series_instance_uid": "1.2.840...",
            "followup_study_date": "2022-02-04",
            "followup_study_instance_uid": "",
            "followup_series_instance_uid": "",
            "followup_series_type": 0
          }
        ]
      }
    ]
  },
  "mask": {
    "model": [
      {
        "series": [
          {
            "instances": [
              {
                "mask_index": 1,
                "mask_name": "A1",
                "diameter": "2.0214000195",
                "type": "CMB",
                "location": "L. occipital",
                "sub_location": null,
                "prob_max": "0.831705",
                "checked": "1",
                "is_ai": "1",
                "seg_series_instance_uid": "1.2.826...",
                "seg_sop_instance_uid": "1.2.826...",
                "dicom_sop_instance_uid": "1.2.840...",
                "main_seg_slice": 83,
                "is_main_seg": 1,
                "followup": [
                  {
                    "mask_index": "",
                    "old_diameter": "",
                    "old_volume": "",
                    "status": "new",
                    "study_date": "2021-04-09",
                    "main_seg_slice": 78,
                    "jump_sop_instance_uid": "1.2.840...",
                    "seg_series_instance_uid": "",
                    "seg_sop_instance_uid": "",
                    "is_ai": "1"
                  },
                  {
                    "mask_index": "2",
                    "old_diameter": "2.2460000217",
                    "old_volume": "",
                    "status": "stable",
                    "study_date": "2021-04-09",
                    "main_seg_slice": 66,
                    "jump_sop_instance_uid": "1.2.840...",
                    "seg_series_instance_uid": "1.2.826...",
                    "seg_sop_instance_uid": "1.2.826...",
                    "is_ai": "1"
                  }
                ]
              },
              {
                "mask_index": "",
                "mask_name": "",
                "diameter": "",
                "type": "",
                "location": "",
                "sub_location": null,
                "prob_max": "",
                "checked": "",
                "is_ai": "",
                "seg_series_instance_uid": "",
                "seg_sop_instance_uid": "",
                "dicom_sop_instance_uid": "1.2.840...",
                "main_seg_slice": 41,
                "is_main_seg": "",
                "followup": [
                  {
                    "mask_index": "5",
                    "old_diameter": "1.5320000143",
                    "old_volume": "",
                    "status": "regress",
                    "study_date": "2021-04-09",
                    "main_seg_slice": 41,
                    "jump_sop_instance_uid": "1.2.840...",
                    "seg_series_instance_uid": "1.2.826...",
                    "seg_sop_instance_uid": "1.2.826...",
                    "is_ai": "1"
                  }
                ]
              }
            ]
          }
        ]
      }
    ]
  },
  "sorted_slice": [
    {
      "model_type": 2,
      "current_study_instance_uid": "1.2.840...",
      "current_series_instance_uid": "1.2.840...",
      "followup_study_instance_uid": "1.2.840...",
      "followup_series_instance_uid": "1.2.840...",
      "sorted": [
        {"current_slice": 1, "prior_slice": 1},
        {"current_slice": 2, "prior_slice": 2}
      ],
      "reverse-sorted": [
        {"prior_slice": 1, "current_slice": 1},
        {"prior_slice": 2, "current_slice": 2},
        {"prior_slice": 3, "current_slice": 2}
      ]
    },
    {
      "model_type": 4,
      "current_study_instance_uid": "1.2.840...",
      "current_series_instance_uid": "1.2.840...",
      "followup_study_instance_uid": "1.2.840...",
      "followup_series_instance_uid": "1.2.840...",
      "sorted": [
        {"current_slice": 1, "prior_slice": 1},
        {"current_slice": 2, "prior_slice": 2}
      ],
      "reverse-sorted": [
        {"prior_slice": 1, "current_slice": 1},
        {"prior_slice": 2, "current_slice": 2}
      ]
    }
  ]
}
```
