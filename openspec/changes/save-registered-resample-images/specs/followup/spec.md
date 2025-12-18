# Follow-up 模組規格增量

## ADDED Requirements
### Requirement: 配準中間產物 MUST 可供 QA 檢視
系統 MUST 在 Follow-up 配準流程中，將對位後的 resample/registered 影像集中保存，供人工 QA 檢視是否有整體平移、翻轉或重疊失敗。

#### Scenario: 保存配準中間產物
- WHEN 系統執行 Follow-up → Baseline 配準
- THEN 系統 MUST 將對位後的影像保存到 `registration/resample/`
- AND 若啟用 min-1mm，系統 MUST 另外保存 min-1mm 空間的 registered SynthSEG 影像與對應的 `_min1mm.mat`


