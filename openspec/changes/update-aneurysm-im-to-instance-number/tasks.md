## 1. 規格
- [x] 1.1 更新 `mask.instances.im` 定義：改為 DICOM (0020,0013) Instance Number

## 2. 實作
- [x] 2.1 更新 `ReviewAneurysmPlatformJSONBuilder.get_mask_instance()`：`im` 取對應 DICOM 的 Instance Number

## 3. 驗證與收尾
- [ ] 3.1 `openspec validate update-aneurysm-im-to-instance-number`
- [ ] 3.2 `openspec archive update-aneurysm-im-to-instance-number --yes`


