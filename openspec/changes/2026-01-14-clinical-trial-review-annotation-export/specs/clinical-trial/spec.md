# 臨床試驗：醫師審核/標註與下載統計（平台端規格）

## MODIFIED Requirements

### Requirement: 識別一筆病例（Case）與其 AI 推論來源
平台 MUST 以「可追溯」方式識別病例與推論來源，至少包含：
- `inference_id`
- `input_study_instance_uid[]`
- `input_series_instance_uid[]`
- `model_id`
- `inference_timestamp`

平台 MUST 將「審核/標註」與特定推論結果綁定（同一病例可有多次推論，需可區分）。

#### Scenario: 建立 ReviewSession
- WHEN 平台接收一筆 AI 推論輸出（含上述欄位）
- THEN 平台 MUST 建立一筆 `ReviewSession`
- AND MUST 將 `ReviewSession.inference_id` 綁定至該推論輸出

---

### Requirement: 臨床試驗欄位（Trial Context）
平台 MUST 支援至少一個臨床試驗欄位：
- `reader_id`：醫師代碼（誰完成審核/標註）

平台 SHOULD 預留下列可選欄位以避免未來重做 schema：
- `trial_id`
- `site_id`
- `subject_id`（試驗用代碼，非病歷號）
- `visit_timepoint`

#### Scenario: 審核完成時寫入 reader_id
- WHEN 醫師將 `ReviewSession` 提交（Submit）或簽核（Sign-off）
- THEN 平台 MUST 將 `reader_id` 記錄於該次提交對應的 `AnnotationRevision` 與 `AuditTrail`

---

### Requirement: Group（組別/試驗條件）包含 AI 版本與遮蔽規則
平台 MUST 支援以 `group` 表示臨床試驗條件（trial arm），並至少包含：
- `group_id`
- `group_name`
- `ai_visible`：是否在 UI 顯示 AI 標註（本專案不考慮不同 UI 呈現方式，只有顯示/不顯示）
- `ai_version`：此 group 對應的 AI 版本（例如 `model_id` 或等價版本識別；允許不同 group 對應不同 AI 版本）

平台 MUST 支援「遮蔽/盲測」以**帳號或 group**為粒度：
- 當 `ai_visible = false` 時，該 group 的醫師在 UI MUST 不能看到任何 AI 標註內容

#### Scenario: group 決定是否顯示 AI 標註
- WHEN 讀者以某個 `group_id` 開啟任務
- THEN 平台 MUST 依 `group.ai_visible` 決定是否在 UI 顯示 AI 標註

---

### Requirement: 任務（Task）粒度為 Case × Reader × Group（不仲裁）
平台 MUST 以任務（task）管理臨床試驗閱讀，task MUST 可唯一識別：
- `case_id`（可由 `study_instance_uid + series_instance_uid` 或平台自訂 ID 組成）
- `reader_id`
- `group_id`

平台 MUST 支援多位醫師同時閱讀同一個 case（同或不同 group），且各自的標註/確認結果 MUST 獨立保存，不進行共裁（adjudication）。

#### Scenario: 多醫師同案但互不影響
- WHEN 醫師 A 與醫師 B 同時閱讀同一個 case
- THEN 平台 MUST 產生兩個獨立 task（至少 `reader_id` 不同）
- AND MUST 分別保存各自的 `ReviewSession / AnnotationRevision`

---

### Requirement: ReviewSession 狀態機（可稽核）
平台 MUST 將審核/標註流程以狀態機表達，至少包含：
- `draft`：可編修、尚未提交
- `submitted`：已提交、不可再編修（除非建立新的 revision 並重新提交）

平台 SHOULD 支援：
- `reopened`：已提交後因需求重新開啟（需記錄原因）

#### Scenario: 提交後鎖定
- WHEN `ReviewSession` 由 `draft` 轉為 `submitted`
- THEN 平台 MUST 鎖定當下 `AnnotationRevision` 作為「該次提交的最終版」
- AND MUST 禁止對該 revision 做覆寫式修改（只能建立新 revision）

---

### Requirement: 標註資料模型（Lesion 為核心）
平台 MUST 以 lesion（病灶物件）作為可編修的核心單位。每個 lesion MUST 至少包含：
- `lesion_id`：平台生成的穩定 ID（不可用 array index 取代）
- `source`：`ai` | `human` | `ai_modified`
- `label`：顯示用標籤（例如 `A1`）
- `type`：病灶類型（例如 aneurysm subtype / CMB 類型）
- `location`：解剖位置（字串或 taxonomy code）
- `probability`：可選（AI 來源時保留）
- `metrics`：大小/直徑/體積等（可選，依模組）
- `dicom_ref`：至少需能引用到被標註的影像 series（見下）
 - `geometry`：可選（當需要最小化標註工具時使用；例如 anomaly box 或中心點）

平台 MUST 支援將既有 AI JSON 中的 `detections[].mask_index` 作為對應來源欄位：
- `source_mask_index`（可為 null；人工新增者為 null）

#### Scenario: 從 AI 推論初始化 lesions
- WHEN 平台收到 AI 推論輸出，其中包含 `detections[]`
- THEN 平台 MUST 以 `detections[]` 初始化第一版 `AnnotationRevision`
- AND MUST 將每筆 `detection.mask_index` 存為 `lesion.source_mask_index`
- AND MUST 保留 AI 原始欄位（如 `probability`、`main_seg_slice`）以利追溯

---

### Requirement: AI 標註確認（Confirm/Check）
當 task 屬於 `ai_visible = true` 且有 AI 標註時，平台 MUST 提供醫師「確認 AI 標註正確」的操作（Confirm/Check）。

#### Scenario: Confirm 表示接受 AI 標註
- WHEN 醫師對某個 AI lesion 按下 Confirm/Check
- THEN 平台 MUST 在該 task 的最新 `AnnotationRevision` 中記錄此確認狀態
- AND MUST 在 `AuditTrail` 記錄此事件（包含 `reader_id`、`group_id`、`revision_id`、時間戳）

---

### Requirement: doctor-only 任務建立（不顯示 AI，但保留 AI 作為後台可追溯）
平台 SHOULD 提供建立 `doctor-only` 任務的選項，用於效益實驗或盲測：
- UI 不顯示 AI 標註（等同 `ai_visible = false`）
- 平台仍 SHOULD 保留原始 AI 輸出作為後台 trace（供合規/比對/匯出時使用，但不提供給該 task 的醫師 UI）

#### Scenario: 由同一筆推論建立 doctor-only task
- WHEN 管理者選擇以既有 AI 推論輸出建立 doctor-only task
- THEN 平台 MUST 建立新的 task 並設定 `ai_visible = false`
- AND SHOULD 將 AI 原始輸出與該 task 關聯（後台可查）

---

### Requirement: 最小化標註工具（anomaly box 與固定大小中心點）
平台 MUST 支援「無 AI 標註」或「醫師需自行標註」的 task 之最小化標註能力。

平台 SHOULD 先支援：
- **anomaly box**（radax 既有工具）

平台 MAY 支援（後續擴充）：
- **brush 中心點**：以固定大小（例如 1mm）表示病灶中心，用於與 GT 計算重疊/命中

平台 MUST 對 lesion 的 `geometry` 提供可擴充的資料結構，至少包含：
- `geometry_type`：`anomaly_box` | `center_point` | `segmentation`（預留）
- `geometry_payload`：對應 type 的資料（座標系與欄位定義見待決問題）

#### Scenario: 無 AI 的 task 仍可產生人工 lesion
- WHEN task 沒有任何 AI 標註
- THEN 平台 MUST 允許醫師建立 `source = human` 的 lesion
- AND MUST 允許 lesion 以 `geometry_type = anomaly_box`（以及後續的 `center_point`）保存

---

### Requirement: 影像引用（DICOM Reference）必須足以生成 DICOM SEG
平台 MUST 在每個 lesion 或每個 revision 層級保存足以生成 DICOM SEG 的必要引用資訊：
- `study_instance_uid`
- `annotated_series_instance_uid`（或等價欄位；對應 AI JSON 中的 `annotated_series_instance_uid`）

平台 SHOULD 保存：
- `sop_instance_uids[]`（該 series 的 instance 列表，用於 SEG 參照）

#### Scenario: 缺少 SOP 列表仍可運作
- WHEN 平台未保存 `sop_instance_uids[]`
- THEN 平台 MUST 仍可透過 DICOM 後端（例如 PACS/Orthanc）查詢該 series 的 SOP 列表以完成 SEG 匯出

---

### Requirement: AnnotationRevision（版本化、可回溯、可比對）
平台 MUST 對每次儲存建立 `AnnotationRevision`，並至少包含：
- `revision_id`
- `parent_revision_id`（第一版可為 null）
- `created_at`
- `created_by`（包含 `reader_id` 或使用者帳號）
- `schema_version`（此規格版本）
- `lesions[]`（該 revision 的完整狀態，建議以 snapshot 存）

#### Scenario: 編修形成新 revision
- WHEN 醫師在 UI 中新增/刪除/調整 lesion 並按下儲存
- THEN 平台 MUST 建立新的 `AnnotationRevision`
- AND MUST 將上一版設為 `parent_revision_id`

---

### Requirement: 匯出（ExportJob）與稽核（AuditTrail）
平台 MUST 將每次下載視為一次 `ExportJob`，並寫入 `AuditTrail`：
- `export_job_id`
- `export_type`：`case_json` | `case_table` | `dicom_seg`
- `requested_by`
- `requested_at`
- `review_session_id`
- `revision_id`（該次下載使用的 revision）
- `result`：success/failure + 錯誤訊息（若失敗）

#### Scenario: 下載必須可追溯 revision
- WHEN 使用者下載任一格式
- THEN 平台 MUST 記錄該下載對應的 `revision_id`
- AND MUST 確保下載內容與該 `revision_id` 一致

---

### Requirement: 個案層級 JSON 匯出（Case JSON Export）
平台 MUST 提供一份「可供統計與回溯」的個案 JSON，至少包含：
- `case`：病例識別（Study/Series UID）
- `inference`：AI 推論來源（inference_id/model_id/timestamp）
- `annotations`：
  - `ai_original`（原始 AI detections；可直接嵌入或以 artifact 引用）
  - `final`（人工最終 lesions；以當次 revision snapshot）
- `review`：
  - `review_session_id`
  - `status`
  - `reader_id`
  - `revision_id`
  - `submitted_at`（若已提交）
- `provenance`：
  - `schema_version`
  - `pipeline_version`（若可取得）
  - `exported_at`

#### Scenario: JSON 需包含 AI 與人工最終
- WHEN 匯出 `case_json`
- THEN 平台 MUST 同時輸出 AI 原始與人工最終
- AND MUST 明確標示人工最終採用的 `revision_id`

---

### Requirement: 個案層級 CSV/Excel 匯出（Case Table Export）
平台 MUST 提供可供臨床試驗統計的表格匯出（CSV 或 Excel）。

平台 MUST 定義固定欄位（至少包含，下列為建議最小集合）：
- 個案欄位：`inference_id`, `model_id`, `study_instance_uid`, `series_instance_uid`, `inference_timestamp`
- 審核欄位：`review_session_id`, `revision_id`, `review_status`, `reader_id`, `submitted_at`
- lesion 欄位：`lesion_id`, `source_mask_index`, `label`, `type`, `location`, `diameter`, `probability`, `main_seg_slice`

表格 MUST 使用「一列一個 lesion」的 normal form（同一個案多個 lesion 會有多列），以便統計。

#### Scenario: 沒有任何 lesion 的個案
- WHEN 一筆病例最終沒有任何 lesion（例如全被刪除或原本就空）
- THEN 平台 MUST 仍輸出一列 case-level row（lesion 欄位為空），避免統計漏案

---

### Requirement: DICOM SEG 匯出（Series-level）
平台 MUST 支援以 series 為粒度匯出 DICOM SEG：
- SEG MUST 引用目標 `annotated_series_instance_uid` 的 SOP Instances
- SEG MUST 使用 `SegmentNumber` 對 lesion 做穩定映射

平台 SHOULD 採用以下映射規則（避免重跑造成對應漂移）：
- 若 lesion 來源為 AI 且有 `source_mask_index`：
  - `SegmentNumber = source_mask_index`
- 若 lesion 為人工新增（`source_mask_index == null`）：
  - `SegmentNumber` 由平台配置，但 MUST 在 revision 內唯一且穩定

SEG SHOULD 在 `SegmentLabel` 中保留 `label`（例如 `A1`）與必要的 `type/location`。

#### Scenario: 匯出使用人工最終 revision
- WHEN 匯出 `dicom_seg`
- THEN 平台 MUST 使用指定的 `revision_id` 作為 segmentation 的來源
- AND MUST 在 SEG 的 private tag 或 metadata 中記錄 `revision_id` 與 `inference_id`（若平台規範允許）

---

### Requirement: 存取控制（Access Control）
平台 MUST 支援至少下列權限控制：
- Reader（醫師）：可檢視、編修、提交（受試驗/病例授權限制）
- TrialAdmin/CRA：可檢視、管理任務、匯出統計（依授權）
- Backend/Ops/AI team：可下載標註資料（AI 與人工標註檔）以供統計與模型迭代

平台 MUST 將匯出能力拆分為至少兩種權限：
- `export_excel`：下載驗證結果報表（Excel）
- `export_annotations`：下載標註資料（AI 標註與人工標註）

其中 `export_annotations` MUST 預設只授權給後端/Ops/AI team；`export_excel` SHOULD 以「特殊權限」授與（通常為撰寫計劃/分析的醫師或試驗管理者）。

#### Scenario: 未授權不得下載
- WHEN 使用者無下載權限
- THEN 平台 MUST 禁止匯出並記錄拒絕事件到 `AuditTrail`

