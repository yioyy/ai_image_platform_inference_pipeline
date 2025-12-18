# followup 規格增量

## ADDED Requirements
### Requirement: Follow-up 輸出檔名規則（含 model 與大小寫）
系統 MUST 依照 `model` 參數（保留大小寫）產出固定命名的 Follow-up 產物檔案。

#### Scenario: 輸出 Follow-up JSON（檔名含 model）
- WHEN Follow-up pipeline 完成報告輸出
- THEN 系統 MUST 產出 `Followup_<model>_platform_json.json`

#### Scenario: 輸出配準矩陣（固定前綴 baseline）
- WHEN Follow-up pipeline 完成影像配準
- THEN 系統 MUST 於 `registration/` 產出 `baseline_<model>.mat`

#### Scenario: process 根目錄相容 CLI default
- WHEN `path_process` 已指向 `.../Deep_Followup/`
- THEN 系統 MUST 使用 `<path_process>/<baseline_id>/` 作為處理根目錄
- AND MUST NOT 再額外拼接第二層 `Deep_FollowUp/`


