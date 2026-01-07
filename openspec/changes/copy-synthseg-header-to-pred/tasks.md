## 1. 實作
- [x] 1.1 `pipeline_followup.py`：在 baseline/followup 複製檔案後、registration 前，將 SynthSEG header 幾何資訊覆蓋 Pred 並保存
- [x] 1.2 加入 shape 不一致檢查（不一致就 raise）

## 2. 測試
- [x] 2.1 新增單元測試：Pred 的 affine 被成功覆蓋成 SynthSEG 的 affine

## 3. OpenSpec 完成流程
- [ ] 3.1 `openspec validate copy-synthseg-header-to-pred`
- [ ] 3.2 `openspec archive copy-synthseg-header-to-pred --yes`


