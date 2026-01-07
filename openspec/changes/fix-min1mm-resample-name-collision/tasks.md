## 1. 實作
- [x] 1.1 `registration.py`：min-1mm resample 暫存檔加入 baseline_/followup_ 前綴，避免檔名覆寫
- [x] 1.2 `registration.py`：將 baseline/followup resample 後的檔案分別保存到 `registration/resample/`

## 2. 測試
- [x] 2.1 跑既有 registration 測試（resample/min-1mm、mat grid convert、followup_report）

## 3. OpenSpec 完成流程
- [ ] 3.1 `openspec validate fix-min1mm-resample-name-collision`
- [ ] 3.2 `openspec archive fix-min1mm-resample-name-collision --yes`


