# followup 規格增量（v2：DICOM-SEG + prediction.json）

## MODIFIED Requirements

### Requirement: 輸入資料（DICOM-SEG + 多筆 prediction.json）
系統 MUST 支援以以下輸入進行 Follow-up v2：
- AI prediction 的 DICOM-SEG（病灶 mask）
- 對位用 SynthSEG mask 的 DICOM-SEG
- 多筆不同時間的 prediction.json（同一 patient/model）

#### Scenario: prediction.json 必要幾何欄位（fail-fast）
- WHEN 任一筆 prediction.json 用於對位
- THEN 該 JSON MUST 含 `affine`（4x4）與 `sorted`（含 SOPInstanceUID 空間排序/投影）
- AND 若缺少上述任一欄位，系統 MUST 終止並回報缺少欄位清單

### Requirement: 日期與比較方向（以新日期找舊日期）
系統 MUST 以「current date 對上所有更早 prior」的方向產生 `follow_up`。

#### Scenario: compare-all-older（每個 current 比所有更早 priors）
- WHEN 系統取得同一 patient 的多筆 study（含日期）
- THEN 對每一個 current study，系統 MUST 對所有更早日期的 prior studies 各產生一筆 follow_up 結果
- AND follow_up MUST 依 prior 日期由近到遠排序（最近 prior 在前）

#### Scenario: 新資料可能比資料庫舊（輸出哪些 current）
- WHEN 新進 study 日期是最新
- THEN 系統 MUST 只輸出該最新 study 的 follow_up 結果
- WHEN 新進 study 日期不是最新
- THEN 系統 MUST 輸出所有「日期較新且其 priors 集合發生變動」的 studies 之 follow_up 結果

### Requirement: 以 DICOM-SEG + sorted/affine 重建 NIfTI 並使用 FSL flirt 對位
系統 MUST 能從 DICOM-SEG 與 JSON 的幾何資訊重建出 FSL `flirt` 可用的 NIfTI，並完成 current↔prior 的對位。

#### Scenario: flirt 對位產物
- WHEN 對位完成
- THEN 系統 MUST 產生 forward matrix（需明確定義方向：prior→current 或 current→prior）
- AND MUST 能取得 inverse matrix（用於 `new`/`regress` 的 main_seg_slice 推回）

### Requirement: 病灶配對與 3 狀態輸出（status key 固定小寫）
系統 MUST 以對位後的 mask overlap 判定配對並輸出 3 狀態（key 固定小寫）：
- `new`
- `stable`（包含原本 Enlarging/Stable/Shrinking）
- `regress`（原本 disappear）

#### Scenario: overlap>0 配對
- WHEN current 與 prior 的某兩個病灶在同一空間下 overlap voxel 數 > 0
- THEN 系統 MUST 將其視為同一顆（stable）
- AND MUST 建立 mask_index 對應（current_mask_index ↔ prior_mask_index）
- AND 未被配對到的 current 病灶為 new、prior 病灶為 regress

#### Scenario: new/regress 的 main_seg_slice 補齊
- WHEN 某筆偵測屬於 new（prior 無對應）
- THEN 系統 MUST 仍能給出 prior_main_seg_slice（透過對位矩陣/反矩陣推估）
- WHEN 某筆偵測屬於 regress（current 無對應）
- THEN 系統 MUST 仍能給出 current_main_seg_slice（透過對位矩陣/反矩陣推估）

### Requirement: 輸出 JSON 兩種版本（grouped / flattened）
系統 MUST 產出兩種等價內容的輸出格式。

#### Scenario: grouped 版本
- THEN 輸出 MUST 包含 `follow_up[]`
- AND 每筆 follow_up MUST 具有 `prior_study_instance_uid`
- AND MUST 具有 `status.new[]` / `status.stable[]` / `status.regress[]`

#### Scenario: flattened 版本
- THEN 輸出 MUST 包含 `follow_up[]`
- AND 每筆 follow_up MUST 具有 `prior_study_instance_uid`
- AND MUST 具有 `detections[]`
- AND `detections[]` 每筆 MUST 具有 `status`（`new`/`stable`/`regress`）與必要的 mask_index/main_seg_slice 欄位

### Requirement: 批次輸出與單次通知（manifest）
系統 MUST 以批次輸出方式回傳多主體更新結果，並以單次通知指向 manifest。

#### Scenario: manifest 輸出
- WHEN 一次觸發產生 1..N 份 current 結果
- THEN 系統 MUST 產生 `followup_manifest.json`
- AND manifest MUST 至少包含：
  - `batch_id`
  - `trigger_inference_id` 或 `trigger_study_date`
  - `affected_currents[]`

#### Scenario: 單次通知
- WHEN manifest 產出完成
- THEN 系統 MUST 僅送出一次通知（batch complete）
- AND 通知內容 MUST 指向該 manifest
