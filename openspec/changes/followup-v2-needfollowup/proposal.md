# Followup v2：needFollowup 輸入與日期主體規則

## 問題陳述
既有 v1 followup 只接受兩個 ID 與日期，無法支援新的 `needFollowup` 輸入結構與多日期比較情境。這導致：
- 無法依 `patient_id + study_date` 自動對位 case 資料夾
- 無法在同一個 case 新被呼叫 AI 模組後，依日期主體規則產生正確的 followup.json
- 無法處理醫師已判定非病灶（`checked=0`）的比較排除需求

## 解決方案
建立 v2 規格增量，定義：
- 輸入格式：以 `needFollowup` 陣列提供多筆病患/日期資料（參考 `process/followup_tkk.json`）
- 對位規則：以 `patient_id + study_date(YYYYMMDD)` 對應 `"<patient_id>_<study_date>_MR_<access_number>"` 資料夾
- 日期主體：比較以新日期為主體、舊日期對上去；僅比較「當前跑 AI 日期」與 `needFollowup[].study_date`
- `checked=0` 排除：病灶保留輸出但不進行比較（followup list 為空）
- 輸出格式：followup.json 欄位與排序依 `process/Followup_CMB_platform_json.json`
- 上傳策略：當前日期為最新僅上傳一次；若當前日期較舊，輸出對應新日期的 followup.json

## 影響範圍
- **followup 模組**：輸入解析、對位邏輯、日期主體規則、輸出結構
- **pipeline 行為**：在新 case 進入時，先跑該 case AI 模組，再跑 followup 模組
- **排除範圍**：不修改 `radax/` 任何檔案

## 臨床考量
- 以新日期為主體避免舊資料覆寫最新檢查結果
- 醫師已判定非病灶（`checked=0`）不參與比較，降低誤判風險
- 同一個 case 新進時，保持「先推論後比較」的臨床流程一致性

## 技術考量
- 需穩定解析 `study_date` 並轉成 `YYYYMMDD`
- 必須以 `patient_id + study_date` 對位 case 資料夾
- followup.json 欄位順序為平台依賴項，需嚴格遵循範例
- 上傳 JSON 次數需依日期情境最小化，避免重複覆蓋
