# followup 規格增量

## ADDED Requirements

### Requirement: 從 DICOM-SEG + DICOM header 重建 SynthSEG NIfTI
系統 MUST 支援在 Follow-up 流程中，僅使用「per-instance DICOM header（不含 PixelData）」與「SynthSEG 的 DICOM-SEG（含 PixelData）」重建 `SynthSEG_<model>.nii.gz`，以減少本地端保存原始 NIfTI 與下載影像的需求。

#### Scenario: 新模式輸入（無 DICOM PixelData）
- WHEN 使用者提供 baseline/followup 的 `SynthSEG_<model>.dcm`（DICOM-SEG）與 `series_headers.json`（per-instance headers）
- AND 未提供 `SynthSEG_<model>.nii.gz`
- THEN 系統 MUST 產生對應該 series 座標系的 `SynthSEG_<model>.nii.gz`
- AND MUST 可被 `registration.py` 的 FSL FLIRT 正常使用（可讀、3D、幾何合理）

#### Scenario: 必需 header 欄位檢查（fail-fast）
- WHEN `series_headers.json` 缺少下列任一欄位：
  - `SOPInstanceUID`
  - `ImagePositionPatient`
  - `ImageOrientationPatient`
  - `PixelSpacing`
  - `Rows`
  - `Columns`
- THEN 系統 MUST 終止處理並回報清楚錯誤
- AND MUST 列出缺少的欄位名稱清單

#### Scenario: 支援 skip_empty_slices 的 DICOM-SEG
- WHEN 輸入的 DICOM-SEG 僅包含部分切片（例如以 `skip_empty_slices=True` 生成）
- THEN 系統 MUST 仍可重建完整 3D volume
- AND MUST 將缺少的切片以 0（背景）補齊

#### Scenario: 多 label 保留
- WHEN `SynthSEG_<model>.dcm` 為多 label segmentation
- THEN 系統 MUST 以 segment metadata 回填 label 值（預設以 `SegmentNumber` 作為 label 值來源）
- AND MUST 在不同 segment 重疊時採用明確規則（例如「後寫覆蓋」或「保留較大 label」），並在文件中說明

