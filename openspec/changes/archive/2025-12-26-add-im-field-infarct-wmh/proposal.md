# Infarct / WMH platform JSON 增加 mask.instances.im 欄位

## Why
`code_ai/pipeline/dicomseg/schema/infarct.py` 與 `code_ai/pipeline/dicomseg/schema/wmh.py` 的 `MaskInstanceRequest` 使用 Pydantic 明確欄位定義。當上游在 instances dict 中加入額外欄位（例如 `im`）時，Pydantic 轉型會丟棄未宣告欄位，導致最後輸出的 platform JSON 看不到 `im`。

## What Changes
- 在 Infarct / WMH 的 `mask...instances[]` schema 中新增 `im` 欄位（與 `main_seg_slice` 同層）
- 讓上游即使加入 `im` 也能被保留並輸出

## 影響範圍
- `code_ai/pipeline/dicomseg/schema/infarct.py`
- `code_ai/pipeline/dicomseg/schema/wmh.py`


