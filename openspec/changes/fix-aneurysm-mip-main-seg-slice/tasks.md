## 1. 規格
- [x] 1.1 新增規格增量：reformatted_series 的 main_seg_slice 必須使用各自 pred 重算

## 2. 實作
- [x] 2.1 修正 `radax/code_ai/pipeline/dicomseg/build_aneurysm.py`：MIP_Pitch / MIP_Yaw 依各自 `*_pred.nii.gz` 重算 `main_seg_slice`

## 3. 驗證
- [ ] 3.1 以任一實際 case 執行 `make_aneurysm_pred_json(...)`，確認輸出 JSON：
  - `reformatted_series[mip_pitch].detections[].main_seg_slice` 會隨 `MIP_Pitch_pred.nii.gz` 改變
  - `reformatted_series[mip_yaw].detections[].main_seg_slice` 會隨 `MIP_Yaw_pred.nii.gz` 改變
- [ ] 3.2 若 `*_pred.nii.gz` 不存在或該病灶在該 pred 中無標註，確認系統可容錯（fallback 行為正確）


