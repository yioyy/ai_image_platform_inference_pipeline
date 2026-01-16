## 1. 規格與輸入輸出對齊
- [ ] 1.1 將 grouped/flattened 兩種輸出 schema 寫成可驗證的欄位清單（對齊範例 JSON）
- [ ] 1.2 明確定義日期來源與排序規則（`study_date` / `inference_timestamp`）

## 2. 幾何與對位（DICOM-SEG → NIfTI → flirt）
- [ ] 2.1 定義 prediction.json 內 `affine` / `sorted` 的最小需求（缺欄位 fail-fast）
- [ ] 2.2 從 prediction DICOM-SEG + SynthSEG DICOM-SEG + `sorted/affine` 重建 flirt 輸入 NIfTI
- [ ] 2.3 產生 forward/inverse matrix，供 main_seg_slice 推估使用

## 3. 病灶配對與狀態
- [ ] 3.1 實作 overlap>0 的一對一配對與 mask_index 對應
- [ ] 3.2 產生 3 狀態（`new`/`stable`/`regress`），其中 `stable` 含原 3 狀態合併
- [ ] 3.3 `new`/`regress` 在未標註側補 main_seg_slice（使用反矩陣推回）

## 4. 產出兩版 JSON
- [ ] 4.1 產出 grouped 版（`status` 三陣列）
- [ ] 4.2 產出 flattened 版（`detections` 每筆帶 `status`）
- [ ] 4.3 支援「新資料比資料庫舊」時的多份輸出策略

## 5. Manifest 與通知
- [ ] 5.1 產出 `followup_manifest.json`（batch 結果清單）
- [ ] 5.2 單次通知指向 manifest（batch complete）

