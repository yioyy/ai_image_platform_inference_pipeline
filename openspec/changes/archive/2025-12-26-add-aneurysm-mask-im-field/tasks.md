## 1. 規格
- [x] 1.1 新增 `mask.instances.im` 欄位需求到 OpenSpec 規格增量

## 2. 實作
- [x] 2.1 更新 `AneurysmMaskInstanceRequest` schema 以支援/輸出 `im`
- [x] 2.2 更新 `ReviewAneurysmPlatformJSONBuilder.get_mask_instance()` 產出 `im=result['main_seg_slice']`

## 3. 驗證與收尾
- [x] 3.1 `openspec validate add-aneurysm-mask-im-field`
- [x] 3.2 `openspec archive add-aneurysm-mask-im-field --yes`


