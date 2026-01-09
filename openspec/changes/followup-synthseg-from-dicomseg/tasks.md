# 任務清單 - Follow-up：DICOM-SEG + DICOM header 反推 SynthSEG NIfTI

## 1. 規格與介面（P0）
- [ ] 1.1 定義新輸入模式與 CLI 參數（不破壞舊參數）
- [ ] 1.2 明確列出 `series_headers` 的必需欄位（缺欄位必須回報）
- [ ] 1.3 定義多 label 的 label 來源（預設 SegmentNumber；若不適用需明確 mapping）

## 2. 核心轉換工具（P0）
- [ ] 2.1 新增 `code_ai/pipeline/dicomseg/seg_to_nifti.py`
- [ ] 2.2 實作「header → slice 排序」：用 IOP/IPP 計算 slice 法向量並依投影排序
- [ ] 2.3 實作「SEG frame → SOPInstanceUID」對應（以 ReferencedSOPInstanceUID 或等價欄位）
- [ ] 2.4 支援 `skip_empty_slices`：SEG 缺 slice 時補 0
- [ ] 2.5 支援多 label：把各 segment 的 mask 寫回 label volume（不覆蓋衝突時給明確規則/警告）

## 3. Pipeline 接線（P0）
- [ ] 3.1 修改 `pipeline_followup.py`：新增 `--baseline_synthseg_dicomseg` / `--followup_synthseg_dicomseg`（或等價）與 `--baseline_series_headers` / `--followup_series_headers`
- [ ] 3.2 pipeline 先重建 `SynthSEG_<model>.nii.gz` 再走既有流程（僅在新模式啟用）

## 4. 測試（P0）
- [ ] 4.1 單元測試：幾何建構（由合成 headers 驗證 affine/shape）
- [ ] 4.2 單元測試：缺 slice 補 0
- [ ] 4.3 單元測試：多 label 合成
- [ ] 4.4 單元測試：缺 header 欄位的錯誤訊息（列出缺欄位）

## 5. 文件（P1）
- [ ] 5.1 `README.md` 補充新模式使用方式與限制

