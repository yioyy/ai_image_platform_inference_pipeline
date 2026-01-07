# Follow-up 模組規格增量

## MODIFIED Requirements
### Requirement: 配準前 sub-mm spacing 正規化為 1mm 並回原空間輸出
系統 MUST 在執行 Follow-up → Baseline 線性配準前，針對 sub-mm 影像進行解析度正規化（逐軸 min-1mm），且最終輸出 MUST 保持在 baseline 原空間。

#### Scenario: 單次重採樣完成 min-1mm（實作要求）
- WHEN 系統需要執行 min-1mm 重採樣
- THEN 系統 SHOULD 以單次重採樣操作同時完成所有需變動的軸（single-step）
- AND label 重採樣 MUST 使用最近鄰插值（nearest neighbour）


