# Follow-up v3：以 NIfTI 對位 + 多日期比對（platform_json 加 followup）

## 問題陳述
現行 Follow-up（v1）以 NIfTI 對位比對為核心，輸入為 Pred/SynthSEG NIfTI + prediction.json。
v2 導入 DICOM-SEG 還原 NIfTI，流程繁瑣且不穩定，導致追蹤比對成功率不一致。

本次 v3 回歸 v1 的 NIfTI 對位方式，並改成「多日期輸入 + 以輸入為主體」的策略：
- 輸入 `00130846_followup_input.json`，包含 `ids` 與 `needFollowup` 清單
- 依 `study_date` 排序、以最新日期為 current，其餘為 prior
- 針對每個 `model_type` 做 follow-up 比對（同一 study 可同時處理多模型）
- 輸出 `Pred_*_platform_json.json` 的 follow-up 版本，產生 `Followup_*_platform_json_new.json`
- 依主體規則，可能產生多筆 output 並多次通知 API

## 目標
1. **資料來源清楚**：輸入由 `00130846_followup_input.json` 統一描述，不再依賴 DICOM-SEG 回推流程。
2. **多模型一致**：同一 study 中有幾個 `model_type` 就執行幾種 follow-up。
3. **輸出一致性**：`Pred_*_platform_json.json` 的 `study` 與 `mask` 都新增 followup 欄位，並加入 `sorted_slice` 對照。
4. **主體規則明確**：以日期新者為主體產出結果，必要時多筆結果多次通知。
5. **欄位完整**：輸出欄位對應 `Pred_*_platform_json.json` 與 DICOM UID，支援跳轉與 SEG 顯示。

## 輸入 JSON 定義（00130846_followup_input.json）
### 必要欄位
- `ids`: array，觸發 follow-up 的推論或任務 ID 清單（主體來源）
- `needFollowup`: array，候選比對 study 清單

### needFollowup 結構重點
每筆 `needFollowup` 必須包含：
- `patient_id` / `study_date` / `studyInstanceUid`
- `models[]`：多模型輸入
  - `model_type`: int（1=Aneurysm，2=CMB，3=Infarct，4=WMH）
  - `type`: string（與 model_type 對應）

## 資料來源與檔案命名
### Case 資料夾規則
case 根目錄（輸入 NIfTI/JSON）：`/data/4TB1/pipeline/sean/rename_nifti/`
資料夾命名：`<patient_id>_<study_date>_MR_<access_number>`
以 `patient_id + study_date` 進行定位（同日僅一個資料夾）。

### Follow-up 輸出根目錄
Follow-up 結果輸出至：`/data/4TB1/pipeline/chuan/process/Deep_FollowUp/<current_case_id>/...`

### 需要的檔案（依模型命名）
以模型名稱決定檔名（Aneurysm / CMB / WMH / Infarct）：
- `Pred_<Model>.nii.gz`
- `SynthSEG_<Model>.nii.gz`
- `Pred_<Model>_A1.dcm`...（A2.dcm, A3.dcm... 與預測顆數相同數量的 DICOM-SEG）(因為有Pred_<Model>_platform_json.json，可能用不到)
- `Pred_<Model>_platform_json.json`（作為輸出基底）

> CMB 範例：`Pred_CMB.nii.gz` / `SynthSEG_CMB.nii.gz` / `Pred_CMB_A1.dcm` / `Pred_CMB_platform_json.json`
> CMB 模型的 SynthSEG 預設檔名為 `synthseg_SWAN_original_CMB_from_T1BRAVO_AXI_original_CMB.nii.gz`
> （如需不同 JSON 檔名，可透過參數覆蓋）。

## 模型判定規則
- 若有指定 `model` 或 `model_type`，僅執行該模型，且需驗證該模型存在於 `needFollowup[].models[]`。
- 若驗證失敗，記錄 log 並跳過執行（不拋錯），並回傳 `False` 供外部判斷「模型不存在」情境。
- 若未指定 `model`，使用 `needFollowup[].models[]` 內的 `model_type` 判定模型。
- 同一 study 若同時出現 model_type=2 與 4，需分別產生 CMB 與 WMH 的 follow-up output。

## 時序與比對規則
1. 解析 `00130846_followup_input.json`：取得 `ids` 與 `needFollowup`。`ids` 由外部提供比較的 ID。
2. 將輸入 ID 的日期與 `needFollowup[].study_date` 轉成 `YYYYMMDD` 比對鍵。
3. 比較輸入 ID 日期與 `needFollowup` 各日期的先後（不需對 needFollowup 排序）。
4. 主體規則（以輸入 ID 為比較基準）：
   - 比輸入 ID **舊**的日期：全部放在「以輸入 ID 為主體」的 JSON 內。
   - 比輸入 ID **新**的日期：各自生成「以該日期為主體」的 JSON，且內部只與輸入 ID 比對。
5. 產出數量：
   - 若輸入 ID 為**最新**，僅產生 1 個 output（包含所有舊日期）。
   - 若輸入 ID 為**最舊**，需產生 `n` 個 output，`n = needFollowup` 長度。
   - 若輸入 ID 居中，產生「以輸入 ID 為主體」的 1 個 output，外加每個新日期各 1 個 output。
6. 輸出檔名以日期區分：`Followup_<model>_platform_json_new_<YYYYMMDD>.json`。

## 對位方式
沿用 v1 NIfTI 對位流程（FSL flirt）：
- 以 SynthSEG 作為幾何基準進行對位
- 取得 forward / inverse matrix
- 用於對應 mask 與估算 main_seg_slice

## 病灶配對與狀態（3 類）
狀態合併成 3 類，需寫入 status 欄位：
- `new`
- `stable`（包含 Enlarging / Stable / Shrinking）
- `regress`（原 disappear）

即使「未標註側」缺 main_seg_slice，仍需透過矩陣推回估算。

## sorted_slice 欄位（slice 對應）
以 current 為主體：同步滾動時不是單純 z 值相同對齊，followup 會依對位後的 z 位置變動；
因此對每個 current 的 z slice 記錄「對位後對應到的 followup z 列表」（可能一對多），由 inverse matrix 回推。
序號從 **1** 開始，還會需要 model_type 與 current/followup 的 study_instance_uid、series_instance_uid 定義是哪個 model 的 sorted_slice：
```
"sorted": [
  {"current_slice": 1, "followup_slice_list": [1, 2]},
  {"current_slice": 2, "followup_slice_list": [3]}
],
"reverse-sorted": [
  {"followup_slice": 1, "current_slice_list": [1]},
  {"followup_slice": 2, "current_slice_list": [1]},
  {"followup_slice": 3, "current_slice_list": [2]}
]
```

## 輸出與回寫規則（多次通知）
輸出以「更新後的 platform_json + followup 欄位」為主：
- `Pred_*_platform_json.json` → `Followup_*_platform_json_new.json`
- `study.model[].followup[]` + `mask.model[].series[].instances[].followup[]`
- `sorted_slice` 置於根層級，與 `sorted` 同階級

主體規則一律以日期新者為主體；若有多個主體則通知多次。

## 錯誤處理與邊界情境
- `needFollowup` 為空：不執行 follow-up，回傳空結果或提示。
- 同一 patient 同日多筆資料：視為輸入錯誤並中止。
- 檔案缺失（Pred/SynthSEG/平台 JSON）：中止並回報缺失清單。
- `needFollowup` 為空：不執行 follow-up，回傳空結果或提示。

## 驗收標準
- 需要 case 資料夾內 NIfTI / platform_json 可完成比對
- 3 狀態分類正確且欄位對應完整
- sorted_slice 欄位包含完整 z 對應資訊（索引從 1 起算）
- 多日期輸入可產生多份 output，且依主體規則多次通知

