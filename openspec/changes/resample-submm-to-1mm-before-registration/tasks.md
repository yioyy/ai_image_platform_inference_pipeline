## 1. 規格與文件
- [x] 1.1 新增 Follow-up registration 的規格增量：sub-mm 先轉 1mm，再回原空間輸出
- [x] 1.2 紀錄臨床取捨：label 最近鄰、輸出保持 baseline 原空間

## 2. 核心實作（registration）
- [x] 2.1 讀取 baseline/followup spacing（NIfTI header zooms）
- [x] 2.2 若任一軸 < 1mm，建立 1mm isotropic 暫存 SynthSEG（nearest neighbour）
- [x] 2.3 在 1mm 空間估計 FLIRT `.mat`
- [x] 2.4 將 1mm `.mat` 換算回「原始影像 grid」對應的 `.mat`
- [x] 2.5 使用換算後的 `.mat` 對原始 followup SynthSEG/Pred 做 -applyxfm（nearest neighbour）

## 3. 測試
- [x] 3.1 新增單元測試：矩陣跨 voxel grid 換算（至少 identity / 簡單縮放案例）

## 4. OpenSpec 完成流程
- [ ] 4.1 `openspec validate resample-submm-to-1mm-before-registration`
- [ ] 4.2 `openspec archive resample-submm-to-1mm-before-registration --yes`


