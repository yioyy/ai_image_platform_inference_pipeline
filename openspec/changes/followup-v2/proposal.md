# Follow-up v2：以 DICOM-SEG + radax prediction.json 做縱向追蹤（輸出 grouped/flattened 兩版）

## 問題陳述
現行 Follow-up（v1）以「NIfTI 檔案導向」流程為中心，輸入是 Pred/SynthSEG NIfTI + platform JSON，輸出為 `Pred_<model>_followup.nii.gz` 與 `Followup_<model>_platform_json.json`。

但在 radax 串接場景中，實際的核心資料來源是：
- **AI prediction DICOM-SEG**（病灶 mask）
- **SynthSEG mask DICOM-SEG**（對位用/幾何基準）
- **多次不同時間的 radax prediction.json**（同一 patient 的多次檢查偵測結果）

此外，必須支援「新資料可能比資料庫舊」的情境：當補進較舊日期，應回寫較新日期的 follow-up 結果（以新日期找舊日期），並可能一次產生多份 follow-up JSON。

## 目標（v2）
建立 Follow-up v2 模組，輸入為 DICOM-SEG + 多筆 prediction.json（每筆含必要幾何欄位），輸出為：
- **對位結果的 prediction.json（新增 `follow_up` 欄位）**
- 同一份結果提供兩種格式：
  - **grouped**：`follow_up[].status.{new|stable|regress}[]`
  - **flattened**：`follow_up[].detections[]`（每筆帶 `status`）

狀態由 5 類縮成 3 類（文件與 JSON 欄位 key 均固定小寫）：
- `new`
- `stable`（包含原本 Enlarging / Stable / Shrinking）
- `regress`（原本 disappear）

## 核心方法（摘要）
1. 先判定 current study（依 `study_date` / `inference_timestamp`）
2. current 與所有更早的 prior 逐一比較（compare-all-older）
3. 以 DICOM-SEG + `sorted` + `affine` 重建 FSL `flirt` 需要的 NIfTI，完成對位
4. 對位後以 overlap>0 進行 mask_index 的一對一配對，生成 `new`/`stable`/`regress`
5. 若為 `new` 或 `regress`，需在「沒標註的那組」補出 main_seg_slice（透過對位矩陣與反矩陣推回）

## 輸出/回寫規則（資料庫情境）
- 若新進資料是最新日期：只回傳「最新日期」那一份 follow-up 結果（其 follow_up 內會包含所有更早 priors）
- 若新進資料比資料庫舊：回傳所有「比它新的日期」之 follow-up 結果（因為是以新日期找舊日期）
- 若新進資料是最舊：回傳資料庫內所有日期的 follow-up 結果（每一個 current 都新增了這個 prior）

## 輸出策略（manifest + 單次通知）
為了避免多主體更新造成多次通知，v2 採用批次輸出策略：
- 每次觸發產出 **一份** `followup_manifest.json`（batch 結果清單）
- 對外通知 **只發一次**，並指向 manifest（batch complete）
- manifest 內列出本次所有被更新的 current 結果與狀態

此策略使串接方只需處理單次事件，同時保留 1..N 份 current 結果的完整資訊。

## 驗收標準
- 能產出兩版輸出 JSON（grouped/flattened），格式對齊範例：`process/prediction-grouped.json`、`process/prediction-flattened.json`
- 3 狀態分類正確（`new`/`stable`/`regress`），其中 `stable` 含原 3 狀態合併
- 任一 study 若缺 `affine` 或 `sorted` 必須 fail-fast 並回報缺少欄位
- 能產出 `followup_manifest.json` 並以單次通知完成回傳

