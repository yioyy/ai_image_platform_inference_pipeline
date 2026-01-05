## 1. 規格
- [x] 1.1 新增/補強 dicomseg：Infarct/WMH `mask.instances.im` 欄位輸出需求

## 2. 實作
- [x] 2.1 `schema/infarct.py` 的 `InfarctMaskInstanceRequest` 新增 `im`
- [x] 2.2 `schema/wmh.py` 的 `WMHMaskInstanceRequest` 新增 `im`

## 3. 驗證與收尾
- [ ] 3.1 `openspec validate add-im-field-infarct-wmh`
- [ ] 3.2 `openspec archive add-im-field-infarct-wmh --yes`


