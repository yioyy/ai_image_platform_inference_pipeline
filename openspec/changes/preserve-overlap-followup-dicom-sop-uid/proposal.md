# 保留重疊病灶的 followup DICOM SOP UID（避免重跑覆寫）

## 問題陳述
目前在產生 follow-up 平台 JSON 時，若 baseline 病灶有配對（重疊）到 followup 病灶，程式會直接以 followup JSON 的 `dicom_sop_instance_uid` 覆寫 `followup_dicom_sop_instance_uid`。

在重複執行同一組資料（或 followup JSON 內 SOP 排序/內容有微小差異）時，這會造成「重疊病灶的關鍵張 UID」不穩定，進而影響下游 UI/比對流程。

## 解決方案
- **重疊/配對到的病灶**：優先沿用「輸入 baseline JSON（之前 JSON）」內既有的 `followup_dicom_sop_instance_uid`，避免每次重跑都覆寫。
- **只有 `status` 為 `new` 或 `disappeared`**：才使用配準矩陣（含反矩陣）去推估/重新計算關鍵張 SOP UID。

## 影響範圍
- `followup_report.py`：更新 `_update_baseline_instances_with_followup()` 在配對成功時對 `followup_dicom_sop_instance_uid` 的覆寫行為。
- `tests/`：新增單元測試避免回歸。

## 臨床考量
- 穩定且可重現的關鍵張對臨床追蹤比對很重要；重跑造成的 UID 漂移可能導致使用者看到不同切片，影響判讀一致性。

## 技術考量
- 保持向後相容：若 baseline JSON 尚無 `followup_dicom_sop_instance_uid`（第一次產出），仍會補上 followup JSON 的值。
- 不改動 `new`/`disappeared` 的矩陣推估行為，維持原本「必要時才推估」的設計。


