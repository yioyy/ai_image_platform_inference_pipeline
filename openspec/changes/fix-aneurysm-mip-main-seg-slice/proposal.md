# 修正 Aneurysm：MIP_Pitch / MIP_Yaw 的 main_seg_slice 不應照抄 MRA_BRAIN

## 問題陳述
目前 `rdx_aneurysm_pred_json.json` 的 `reformatted_series`（`mip_pitch` / `mip_yaw`）在輸出 `detections[].main_seg_slice` 時，會直接沿用主序列（通常是 `MRA_BRAIN`）的 `main_seg_slice`。

但 `MIP_Pitch_pred.nii.gz` 與 `MIP_Yaw_pred.nii.gz` 為各自序列的推論結果，其 Z 軸切片索引分布可能不同；因此 **MIP 的 main_seg_slice 必須以各自 pred 檔案重算**，否則前端定位病灶切片會錯位。

## 解決方案
- 在產出 `rdx_aneurysm_pred_json.json` 時，針對 `reformatted_series`：
  - `mip_pitch` 使用 `MIP_Pitch_pred.nii.gz` 計算每個病灶的 `main_seg_slice`
  - `mip_yaw` 使用 `MIP_Yaw_pred.nii.gz` 計算每個病灶的 `main_seg_slice`
- 計算方式參考既有邏輯：`code_ai/pipeline/dicomseg/aneurysm.py#get_excel_to_pred_json`

## 影響範圍
- `radax/code_ai/pipeline/dicomseg/build_aneurysm.py`
- 任何呼叫 `code_ai.pipeline.dicomseg.build_aneurysm.main()` 的 pipeline（如 `radax/pipeline_aneurysm_tensorflow.py` / `radax/pipeline_aneurysm_track.py`）

## 臨床考量
- main_seg_slice 用於 UI/工作站快速跳轉到最主要病灶切片；錯位會增加閱片時間並可能影響臨床判讀效率。

## 技術考量
- 需對 `*_pred.nii.gz` 缺檔或病灶標註不存在的情況做容錯（輸出 `main_seg_slice = ""`）。


