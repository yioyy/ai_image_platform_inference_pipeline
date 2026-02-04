# Aneurysm platform_json followup 欄位初始化

## 問題
目前由 `make_pred_json` 產出的 Aneurysm platform_json 並未先建立 follow-up 所需的欄位，
導致後續流程難以判斷是否已具備 follow-up 結果，亦缺少針對多時點比對的 `sorted_slice` 容器。

## 解決方案
在 Aneurysm platform_json 輸出時就預先建立以下欄位：
- `study.model[].followup`：空 list
- `mask.model[].series[].instances[].followup`：空 list
- 根層 `sorted_slice`：空 list

當 followup-v3 platform 有成功執行時再填入內容；若只做 `make_pred_json` 則保留空 list。

同時定義「舊日期輸入」與「多筆 needFollowup 日期」的延續規則：
- 若輸入為舊日期，需讀取最新日期的 JSON 作為基底，並延續既有 followup 欄位內容。
- 若 `needFollowup` 為多筆日期，`sorted_slice` 需為多筆 list（每筆為主體時間對應一個較舊時間）。

## 影響範圍
- `pipeline_aneurysm_tensorflow.py` 產出的 Aneurysm platform_json
- followup-v3 platform 的前後比對輸入/輸出 JSON 一致性

## 臨床考量
- 確保前後比較資料結構一致，避免 UI 顯示時缺少必要欄位
- 多時點比對的連續性可回溯

## 技術考量
- 需明確定義「主體時間」與「較舊時間」的排序規則
- `sorted_slice` 建立規則需支援多筆對照
