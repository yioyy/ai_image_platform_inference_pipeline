# followup v2 規格增量

## ADDED Requirements
### Requirement: needFollowup 輸入解析
系統 MUST 支援 followup v2 的 `needFollowup` 陣列輸入。

#### Scenario: 解析 needFollowup 與 models/mask 結構
- WHEN 收到 `needFollowup` 陣列
- THEN 系統 MUST 讀取 `patient_id`、`study_date`、`models[].mask[]` 與 `checked` 欄位
- AND `study_date` MUST 可解析為日期字串 `YYYY-MM-DD`

#### Scenario: checked=0 病灶處理
- WHEN `models[].mask[].checked = 0`
- THEN 該病灶 MUST 保留在輸出
- AND 該病灶 MUST NOT 參與前後比較
- AND 該病灶的 `followup` list MUST 為空或不更新

### Requirement: case 資料夾對位規則
系統 MUST 依 `patient_id + study_date` 找出對應的 case 資料夾。

#### Scenario: 對位 case 資料夾名稱
- WHEN 有 `patient_id` 與 `study_date`
- THEN 系統 MUST 將 `study_date` 轉為 `YYYYMMDD`
- AND MUST 以 `"<patient_id>_<YYYYMMDD>_MR_<access_number>"` 的資料夾命名規則比對
- AND 只取唯一對應資料夾（假設每組只會對應一個）

### Requirement: 日期主體比較規則
系統 MUST 以新日期為主體、舊日期對上去，且只比較當前 AI 日期與 needFollowup 日期。

#### Scenario: 只比較當前日期與 needFollowup
- WHEN 當前 case 已完成 AI 模組
- THEN 系統 MUST 僅比較「當前 AI 日期」與 `needFollowup[].study_date`
- AND MUST NOT 讓 `needFollowup` 彼此互相比較

#### Scenario: 新日期為主體
- WHEN 比較兩筆日期
- THEN 系統 MUST 以較新日期為主體
- AND 以較舊日期作為對應參照

### Requirement: 多情境輸出與上傳次數
系統 MUST 依日期情境輸出對應的 followup.json。

#### Scenario: 當前日期為最新
- WHEN 當前 AI 日期為所有日期中最新
- THEN 系統 MUST 只輸出一次 followup.json
- AND 其餘日期結果 MUST 直接對入同一份表中
- AND 上傳 followup.json MUST 只呼叫一次

#### Scenario: 當前日期較舊
- WHEN 當前 AI 日期比資料庫日期更舊
- THEN 系統 MUST 輸出當前日期的 prediction.json
- AND MUST 輸出「較新日期」對應的 followup.json

### Requirement: followup.json 輸出格式與欄位順序
系統 MUST 依指定範例輸出 followup.json，且欄位順序不得變更。

#### Scenario: 欄位順序遵循平台範例
- WHEN 產生 followup.json
- THEN 欄位順序 MUST 依 `process/Followup_CMB_platform_json.json`
- AND `followup` 欄位 MUST 為 list
- AND list 內容順序 MUST 與範例一致
