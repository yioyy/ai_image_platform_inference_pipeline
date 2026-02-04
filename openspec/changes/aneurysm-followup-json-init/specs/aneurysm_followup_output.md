# Aneurysm platform_json followup 欄位初始化

## ADDED Requirements
### Requirement: Aneurysm platform_json 預建 followup 欄位
系統 MUST 在 `make_pred_json` 產出的 Aneurysm platform_json 內預先建立空的 followup 相關欄位，
以確保後續 followup-v3 platform 能直接填值。

#### Scenario: followup 欄位初始化
- WHEN `make_pred_json` 產出 Aneurysm platform_json
- THEN `study.model[].followup` MUST 為空 list
- AND `mask.model[].series[].instances[].followup` MUST 為空 list
- AND 根層 `sorted_slice` MUST 為空 list

### Requirement: followup 與 sorted_slice 去重追加
系統 MUST 在重跑相同的 current/followup 組合時，保留既有內容並避免重複追加相同項目。

#### Scenario: 重跑相同 current/followup
- WHEN followup-v3 platform 針對同一組 current/followup 重新執行
- THEN `study.model[].followup` MUST 保留既有內容並去重
- AND `mask.model[].series[].instances[].followup` MUST 保留既有內容並去重
- AND `sorted_slice` MUST 保留既有內容並去重

### Requirement: Aneurysm followup JSON 上傳
系統 MUST 將當前日期主體的 JSON 上傳，且若產生其他日期主體的 JSON 亦需一併上傳。

#### Scenario: 產出 followup 結果
- WHEN followup-v3 platform 產生輸出 JSON
- THEN 系統 MUST 至少上傳「輸入日期主體」的 JSON（即使 followup 欄位為空）
- AND 若產生其他日期主體的 JSON，系統 MUST 一併上傳

### Requirement: 舊日期輸入延續 followup 結果
系統 MUST 在舊日期輸入時，讀取最新日期的 JSON 作為主體並延續既有 followup 欄位內容。

#### Scenario: 舊日期輸入
- WHEN 輸入日期小於 `needFollowup` 內最新日期
- THEN 系統 MUST 讀取最新日期的 JSON 作為輸出主體
- AND 若該 JSON 內 `study.model[].followup` 或 `mask.model[].series[].instances[].followup` 已有值
  MUST 保留並延續其內容
- AND `sorted_slice` 若已存在內容 MUST 保留
- AND 系統 MUST 追加建立「傳入舊日期」對「最新日期主體」的對應 `sorted_slice`

### Requirement: 輸入日期為主體的輸出規則
系統 MUST 以「輸入日期」為主要輸出主體，並依 `needFollowup` 其他日期建立對應關係。

#### Scenario: 輸入日期為最新
- WHEN 輸入日期是 `needFollowup` 內最新日期
- THEN 系統 MUST 只產出一個輸入日期的 JSON
- AND followup 欄位內容 MUST 與所有較舊日期比對後追加

#### Scenario: 輸入日期為最舊或非最新
- WHEN 輸入日期不是 `needFollowup` 內最新日期
- THEN 系統 MUST 產出「輸入日期」的 JSON（其 followup 只與更舊日期比對結果追加）
- AND 對於每一個比輸入日期新的日期，系統 MUST 另外產出一個「新日期為主體、輸入日期為比較對象」的 JSON

### Requirement: 多日期 needFollowup 的 sorted_slice 建立
系統 MUST 在 `needFollowup` 有多筆日期時，建立多筆 `sorted_slice` 對照表。

#### Scenario: 多日期對照
- WHEN `needFollowup` 含有多筆日期
- THEN `sorted_slice` MUST 為 list
- AND 每一筆 `sorted_slice` MUST 對應「主體時間」與一個較舊時間
- AND `sorted_slice` 的筆數 MUST 等於「主體時間」對應的較舊時間數量
