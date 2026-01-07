## 1. 實作
- [x] 1.1 `registration.py`：min-1mm 模式下，baseline/followup Pred 先 conform 到 min-1mm 工作空間
- [x] 1.2 `registration.py`：將 SynthSEG resample 到 Pred(min1mm) grid，避免 affine 不一致造成平移/錯位
- [x] 1.3 `registration.py`：矩陣在 min-1mm 空間估計並直接套用到 Pred(min1mm)
- [x] 1.4 `registration.py`：Pred(reg) resample 回 baseline Pred 原空間（確保 voxel index overlap）

## 2. 測試
- [x] 2.1 跑既有單元測試（registration / followup_report）

## 3. OpenSpec 完成流程
- [ ] 3.1 `openspec validate resample-followup-pred-min1mm-then-back`
- [ ] 3.2 `openspec archive resample-followup-pred-min1mm-then-back --yes`


