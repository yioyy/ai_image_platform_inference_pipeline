# sorted_slice 重新設計（雙向嚴格對照）

## 問題陳述
現行 `sorted_slice` 的對照設計可能產生空切片對應，導致滑動滾輪時無法同步顯示對應影像。臨床使用上需要確保「以 current 日期為主」或「以 prior 日期為主」的滾輪同步，都能穩定找到對應切片。

## 解決方案
將 `sorted_slice` 改為雙向、嚴格一對一的切片對照：
- `sorted`：以 `current_slice` 為主對應 `prior_slice`
- `reverse-sorted`：以 `prior_slice` 為主對應 `current_slice`

對照建立時補齊空白對應，確保每個 slice 都有對應值，避免空字串造成同步失敗。

## 影響範圍
- `sorted_slice` 格式與生成規則
- `pipeline_followup_v3_platform.py` 中 sorted_slice 的生成邏輯
- `openspec/changes/followup-v3/specs/output_schema.md` 的欄位定義與範例

## 臨床考量
滾輪同步需保證「任一日期主導」時都能顯示可用切片，避免因缺對應造成影像空白或跳轉錯誤。

## 技術考量
需在生成對照表時補齊空白區段，確保 `sorted` / `reverse-sorted` 為嚴格一對一對照；並注意矩陣方向在 reverse-sorted 計算時使用原始矩陣。

