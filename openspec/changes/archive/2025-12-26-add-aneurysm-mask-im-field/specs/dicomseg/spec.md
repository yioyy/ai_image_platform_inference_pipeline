# DICOMSEG（Aneurysm）規格增量

## ADDED Requirements
### Requirement: Aneurysm mask instance MUST include raw slice index field
系統 MUST 在 Aneurysm 的 platform JSON 輸出中，於 `mask.series[].instances[]` 新增欄位 `im`，其值 MUST 等於 pipeline 產出之 `result['main_seg_slice']`（未進行反向計算）。

#### Scenario: include im at same level as main_seg_slice
- WHEN 系統在 Review Aneurysm 流程中建立 `mask.series[].instances[]`
- THEN 每個 instance MUST 包含 `im`
- AND `im` MUST 等於 `result['main_seg_slice']`


