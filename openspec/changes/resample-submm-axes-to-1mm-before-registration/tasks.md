## 1. 規格
- [x] 1.1 修改 Follow-up 配準的 sub-mm 規格：由「1mm isotropic」改為「逐軸 min-1mm」

## 2. 實作
- [x] 2.1 `registration.py`：逐軸判斷 x/y/z 哪些軸需要升到 1mm
- [x] 2.2 `registration.py`：只對 sub-mm 軸做最近鄰 downsample，其他軸不動

## 3. 測試
- [x] 3.1 新增/更新測試：驗證只改 sub-mm 軸的 zoom 與 shape

## 4. OpenSpec 完成流程
- [ ] 4.1 `openspec validate resample-submm-axes-to-1mm-before-registration`
- [ ] 4.2 `openspec archive resample-submm-axes-to-1mm-before-registration --yes`


