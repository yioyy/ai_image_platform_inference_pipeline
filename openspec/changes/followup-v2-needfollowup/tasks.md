## 1. 資料與輸入
- [ ] 1.1 定義 needFollowup 欄位映射與資料驗證（patient_id, study_date, models, mask.checked）
- [ ] 1.2 將 study_date 轉換為 YYYYMMDD 並建立 case 對位規則
- [ ] 1.3 規範 checked=0 病灶的排除比較行為與輸出保留策略

## 2. 核心比較邏輯
- [ ] 2.1 實作新日期為主體、舊日期對上去的比較規則
- [ ] 2.2 僅比較當前 AI 日期與 needFollowup 內日期，不做內部互相比較
- [ ] 2.3 支援多情境輸出（當前日期最新 vs 當前日期較舊）

## 3. 管道整合
- [ ] 3.1 新 case 流程：先跑 AI 模組，再執行 followup 模組
- [ ] 3.2 依情境控制 upload_json 呼叫次數（最新只一次）
- [ ] 3.3 確認不修改 `radax/` 相關檔案

## 4. 輸出格式
- [ ] 4.1 依 `process/Followup_CMB_platform_json.json` 固定欄位順序輸出
- [ ] 4.2 確認 followup 欄位為 list，內容與順序符合平台需求

## 5. 測試與驗證
- [ ] 5.1 以 `process/followup_tkk.json` 驗證輸入解析
- [ ] 5.2 覆蓋 11/1 新日期與 9/1 舊日期情境
- [ ] 5.3 檢查 checked=0 病灶不參與比較
