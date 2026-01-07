## 1. 規格
- [ ] 1.1 新增 followup 規格增量：重疊病灶的 `followup_dicom_sop_instance_uid` 不得被覆寫

## 2. 實作
- [ ] 2.1 修改 `followup_report.py`：重疊（配對成功）時保留舊的 `followup_dicom_sop_instance_uid`
- [ ] 2.2 加入 SOP 映射診斷 log：輸出 source/target projection 與選到的 SOP UID（new/disappeared）

## 3. 測試
- [ ] 3.1 新增單元測試：配對成功時不覆寫舊 UID
- [ ] 3.2 新增單元測試：`new`/`disappeared` 仍使用矩陣推估 UID


