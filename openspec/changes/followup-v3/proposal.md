# Follow-up v3：以 NIfTI 對位 + 多日期比對（radax prediction.json 加 follow_up）

## 問題陳述
現行 Follow-up（v1）是以 NIfTI 檔案為主進行對位比對，輸入為 Pred/SynthSEG NIfTI + prediction.json。
v2 曾嘗試走 DICOM-SEG 還原 NIfTI，但流程繁瑣且不穩定。

本次 v3 改回 v1 的 NIfTI 對位方式，並加入「多日期輸入」與「單次通知」機制：
- 輸入 `follow_up_input.json`（current + multiple priors）
- 依 `study_date` 進行排序與多對一比對
- 輸出基於既有 prediction.json，新增 `follow_up` 欄位（grouped 格式）
- 一次觸發僅回傳一次（batch / manifest）

## 目標
1. **資料來源簡化**：只讀 case 資料夾內 NIfTI 與 prediction.json，不使用 DICOM-SEG 還原 NIfTI。
2. **多日期比對**：current 與所有更早 priors 逐一比較。
3. **輸出一致性**：prediction.json 增加 `follow_up` 欄位，格式對齊 `process/Deep_Radax/prediction-grouped.json`。
4. **單次通知**：多份更新結果以 manifest 打包，只通知一次。

## 資料來源與檔案命名
### Case 資料夾規則
case 根目錄（輸入 NIfTI/JSON）：`/home/david/pipeline/sean/rename_nifti/`
資料夾命名：`<patient_id>_<study_date>_MR_<access_number>`
以 `patient_id + study_date` 進行定位（同日僅一個資料夾）。

### Follow-up 輸出根目錄
Follow-up 結果輸出至：`/home/david/pipeline/chuan/process/Deep_FollowUp/<current_case_id>/...`

### 需要的檔案（依模型命名）
以模型名稱決定檔名（Aneurysm / CMB）：
- `Pred_Aneurysm.nii.gz`
- `SynthSEG_Aneurysm.nii.gz`
- `Pred_Aneurysm_rdx_aneurysm_pred_json.json`（由 `rdx_aneurysm_pred_json.json` 改名，作為輸出基底）

> 若為 CMB，檔名對應為 `Pred_CMB.nii.gz` / `SynthSEG_CMB.nii.gz` / `Pred_CMB_rdx_cmb_pred_json.json`
> （如需不同 JSON 檔名，可透過參數覆蓋）。

### 模型判定規則（由 input_json 推導）
使用 `follow_up_input.json` 中的 `model_id` 判定模型：
- `924d1538-597c-41d6-bc27-4b0b359111cf` → `Aneurysm`
- `48c0cfa2-347b-4d32-aa74-a7b1e20dd2e6` → `CMB`

## 時序與比對規則
1. 解析 `follow_up_input.json`：
   - `current_study`
   - `prior_study_list[]`
2. `study_date` 轉成 `YYYYMMDD` 8 碼後排序。
3. 以 **最新日期為 current**：
   - current 與所有更早 priors 逐一比較。
4. 以日期為主體判斷（最新為主體）：
   - 若新進資料為最新：回傳主體為「新日期」的結果，其他舊日期對到它上面。
   - 若新進資料比資料庫舊：主體為資料庫內較新的日期，回傳這些主體結果；新進 case 對到舊 case 上，仍以日期新者為主體。

## 對位方式
沿用 v1 NIfTI 對位流程（FSL flirt）：
- 以 SynthSEG 作為幾何基準進行對位
- 取得 forward / inverse matrix
- 用於對應 mask 與估算 main_seg_slice

## 病灶配對與狀態（3 類）
狀態合併成 3 類：
- `new`
- `stable`（包含 Enlarging / Stable / Shrinking）
- `regress`（原 disappear）

只保留必要欄位：
- `new`：只記 `current_mask_index` + `current_main_seg_slice`
- `regress`：只記 `prior_mask_index` + `prior_main_seg_slice`
- `stable`：記 `current_mask_index` + `prior_mask_index` + 兩邊 main_seg_slice

即使「未標註側」缺 main_seg_slice，仍需透過矩陣推回估算。

## sorted 欄位（slice 對應）
以 current 為主體：同步滾動時不是單純 z 值相同對齊，prior 會依對位後的 z 位置變動；
因此對每個 current 的 z slice 記錄「對位後對應到的 prior z 列表」（可能一對多），由 inverse matrix 回推：
建議格式：
```
[
  {"current_slice": 0, "prior_slice_list": [0]},
  {"current_slice": 1, "prior_slice_list": [1,2]}
]
```
如需相容 `process/Deep_Radax/prediction-grouped.json` 的既有格式，可在輸出時保留舊格式或提供轉換。

## 輸出與回寫規則（單次通知）
輸出以「更新後的 prediction.json + follow_up 欄位」為主，並搭配 manifest：
- `prediction.json`：同一份資料新增 `follow_up[]`
- `followup_manifest.json`：列出本次所有需要回傳的 prediction / followup 結果

主體規則一律以日期新者為主體；manifest 只列出「新日期為主體」的結果。
僅通知一次（batch complete），由後端去拉 manifest 取得多份結果。

## 驗收標準
- 只依 case 資料夾內 NIfTI / prediction.json 可完成比對
- 3 狀態分類正確且欄位簡化
- sorted 欄位包含完整 z 對應資訊
- 多日期輸入可產生多份 output，且只通知一次

