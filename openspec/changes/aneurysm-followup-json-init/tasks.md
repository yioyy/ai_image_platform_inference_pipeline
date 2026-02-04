## 1. 規格更新
- [x] 1.1 定義 Aneurysm platform_json followup 欄位初始化規則
- [x] 1.2 定義舊日期輸入的 followup 延續規則
- [x] 1.3 定義多筆 needFollowup 日期的 sorted_slice 建立規則

## 2. 程式實作
- [x] 2.1 `make_pred_json` 產出時加入空 followup 欄位
- [x] 2.2 `sorted_slice` 初始化為空 list
- [x] 2.3 舊日期輸入時，讀取最新 JSON 並延續 followup 欄位
- [x] 2.4 多日期 needFollowup 時建立多筆 sorted_slice
- [x] 2.5 followup 與 sorted_slice 追加時避免重複項目

## 3. 驗證
- [ ] 3.1 單日期輸入的 JSON 結構檢查
- [ ] 3.2 多日期輸入時 sorted_slice 筆數檢查
