# 臨床試驗：醫師審核/標註與下載統計（平台端需求）

## 變更概述
本變更定義一套「臨床試驗優先」的平台端需求，讓多家醫院資料進入後，醫師可以：
- 在平台上檢視 AI 結果並**編修標註**（新增/刪除/調整）
- 支援「依組別（group）控制是否顯示 AI」以及「同一病例可在不同組別對應不同 AI 版本」的效益實驗設計
- 產生可下載的輸出，用於臨床試驗統計與對外提交
  - **個案層級 JSON**
  - **個案層級 CSV/Excel**
  - **DICOM SEG**
- 全流程具備**版本化、稽核紀錄**與最小合規的治理能力

本文件只定義平台端需求；不包含 AI 端推論/部署/訓練實作。

## 問題陳述（Why）
目前臨床試驗需求的瓶頸在於：
1. **醫師確認的結果難以被系統化下載**：需要可追溯、可統計、可交付給 CRA/研究團隊的結構化輸出。
2. **需要可編輯標註而非單純 Yes/No**：臨床試驗往往要求對病灶做修訂，並保留 AI 結果與人工修訂的差異（provenance）。
3. **多家醫院資料與多人操作需要稽核**：下載、簽核、版本、操作者資訊必須可追溯。
4. **多醫師同案但不共裁**：同一病例可由不同資歷醫師同時閱讀（資深/young/住院醫師組），平台必須保存各自獨立版本供效益分析。

## 目標與非目標（What）

### 目標（Goals）
- 平台 MUST 支援以 AI 推論輸出為起點建立「審核/標註會話」（review session）。
- 平台 MUST 支援醫師在 UI 內對病灶做編修，並形成 revision（可回溯）。
- 平台 MUST 以（Case × Reader × Group）作為任務粒度，支援多醫師同案且互不影響（不共裁）。
- 平台 MUST 支援 group 遮蔽（doctor-only）與 group 指定 AI 版本（例如不同 `model_id`）。
- 平台 MUST 支援三種下載：
  - 個案層級 JSON（含 AI 原始 + 人工最終 + metadata + 版本/稽核）
  - 個案層級 CSV/Excel（固定欄位字典，便於統計）
  - DICOM SEG（以 series 為粒度，輸出 segmentation 給臨床端系統/工具使用）
- 平台 MUST 記錄稽核（審核/標註行為、下載行為、版本資訊）。
- 平台 MUST 至少支援 `reader_id`（醫師代碼）作為臨床試驗欄位；並保留擴充欄位（trial/site/subject/timepoint）。
 - 平台 SHOULD 先提供最小化標註工具（anomaly box），並預留中心點（固定 1mm）等擴充。

### 非目標（Non-goals）
- 不包含 AI 推論管線實作、模型訓練、模型評估工具鏈。
- 不規範 UI 的視覺設計（但規範必要的操作能力與資料落盤結果）。
- 不在本變更內強制定義所有醫院的去識別策略（列為待決/可配置）。

## 解決方案（How）
以平台端為主，新增一套一致的資料模型與流程：
- **ReviewSession**：綁定一次 AI 推論結果（`inference_id`）與目標病例（Study/Series UID），承載審核狀態。
- **AnnotationRevision**：每次儲存即形成一版，包含病灶物件列表（lesions）與必要的 DICOM/幾何引用資訊。
- **ExportJob**：每次下載建立 job 與紀錄，支援 JSON / CSV/Excel / DICOM SEG。
- **AuditTrail**：審核、標註、簽核、下載等事件的不可變紀錄。
 - **Group/Task**：以 group 定義試驗條件（含 AI 版本與遮蔽規則），以 task 落實 Case 分派（Case × Reader × Group）。

完整規格請見：`specs/clinical-trial/spec.md`。

## 影響範圍（Impact）
- 平台：資料庫/儲存層需新增 review/annotation/export/audit 的 schema 與 API。
- 前端：新增「審核/標註」頁面功能與下載入口。
- 與既有 AI JSON 的整合：需以 `inference_id`、`study_instance_uid / series_instance_uid` 與 `detections[].mask_index` 作為對齊基礎。

## 驗收標準（Acceptance Criteria）
- 可用最小流程（Minimum Viable Flow）：
  1. 匯入/顯示一筆 AI 推論結果（含 `inference_id`、UID、detections）
  2. 建立 ReviewSession
  3. 醫師新增 1 個病灶、刪除 1 個病灶、修改 1 個病灶屬性，形成至少 2 個 revision
  4. 下載三種格式（JSON、CSV/Excel、DICOM SEG）
  5. 可在稽核紀錄中查到：審核動作、revision、下載動作、操作者（含 `reader_id`）、時間戳、版本
- 同一筆病例重複下載 MUST 仍可追溯到當下使用的 revision 與 schema version。

## 待決問題（Meeting 要決策）
請見：`open_questions.md`

## 會議邀請模板與議程
請見：`meeting.md`

