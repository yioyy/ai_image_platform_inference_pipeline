# Follow-up v3 sorted 規格增量（radax）

## ADDED Requirements
### Requirement: 雙向嚴格對照
系統 MUST 輸出雙向的嚴格對照表，以支援 current/prior 兩種滾輪主導方式。

#### Scenario: current_date 主導滾輪
- WHEN 使用者以 current_date 滾動切片
- THEN 系統以 `sorted` 對照 `current_slice -> prior_slice`

#### Scenario: prior_date 主導滾輪
- WHEN 使用者以 prior_date 滾動切片
- THEN 系統以 `reverse-sorted` 對照 `prior_slice -> current_slice`

### Requirement: 對照欄位定義
系統 MUST 使用以下欄位格式，並存放於 `follow_up[]` 內：
```
"sorted": [
  {"current_slice": 1, "prior_slice": 2},
  {"current_slice": 2, "prior_slice": 3}
],
"reverse-sorted": [
  {"prior_slice": 1, "current_slice": 1},
  {"prior_slice": 2, "current_slice": 1},
  {"prior_slice": 3, "current_slice": 2}
]
```

### Requirement: 對照補齊規則
系統 MUST 補齊空白對照，確保每個 slice 都有對應值。

#### Scenario: 開頭區段為空
- WHEN 最前段對照列表為空
- THEN 將空段補為第一個可用對應值（預設補 1）

#### Scenario: 結尾區段為空
- WHEN 最後段對照列表為空
- THEN 將空段補為最後一個可用對應值

#### Scenario: 中段出現空對照
- WHEN 中間位置出現空對照
- THEN 比較前一段可用值的末尾與下一段可用值的開頭距離
- AND 補為距離較近的值

#### Scenario: 中段候選值超過兩筆
- WHEN 候選值數量大於 2
- THEN 取候選值的中位數
- AND 選擇最接近中位數的有標註座標

### Requirement: 矩陣方向規則
系統 MUST 依以下規則使用矩陣方向：
- `sorted`：使用 inverse matrix（prior -> current）
- `reverse-sorted`：使用原始 matrix（current -> prior）

## MODIFIED Requirements
### Requirement: sorted 對照型態
原先 `current_slice -> prior_slice_list` 的多對多設計改為一對一嚴格對照。

## REMOVED Requirements
### Requirement: 多對多映射
移除 `prior_slice_list` 的多對多映射結構。
