# Follow-up：以 DICOM header + SynthSEG DICOM-SEG 反推 SynthSEG NIfTI（減少本地檔案）

## 問題陳述
目前 Follow-up 模組（`pipeline_followup.py` / `registration.py`）在配準階段需要 `SynthSEG_<model>.nii.gz`（baseline 與 followup 各一份），典型流程會：

- 先下載/保存原始 DICOM 影像（含 PixelData）到本地
- 再用 dcm2niix / 其它方式轉出 NIfTI，或使用已保存的 NIfTI

但在實務上，我們往往只需要：

- **該 series 的幾何資訊**（每張切片的 DICOM header，無需 PixelData）
- **SynthSEG 的分割結果**（DICOM-SEG，本身就包含 mask 的像素資料）

因此，若能只靠「per-instance DICOM header（不含 PixelData）+ SynthSEG DICOM-SEG」重建 `SynthSEG_<model>.nii.gz`，就能：

- 大幅降低本地端 I/O、磁碟占用與資料治理成本
- 避免下載影像 pixel data（更快、更輕量）
- 仍維持 Follow-up 配準與後續邏輯不變（向後相容）

## 解決方案
新增一條 Follow-up 的可選輸入模式：

- **舊模式（維持）**：直接提供 `SynthSEG_<model>.nii.gz`
- **新模式（新增）**：提供
  - `SynthSEG_<model>.dcm`（DICOM-SEG，含 PixelData）
  - `series_headers.json`（每個 SOP instance 一筆 header，無 PixelData）
  - 系統先重建 `SynthSEG_<model>.nii.gz`（幾何對齊該 series），再進入既有 `registration.py` / `pipeline_followup.py` 流程

### 關鍵技術點
- DICOM-SEG 可能使用 `skip_empty_slices=True` 生成（見 `code_ai.pipeline.dicomseg.utils.base.make_dicomseg_file`），因此 SEG 內可能只含「有標註的 slices」。重建 NIfTI 時必須能用 header 的 SOPInstanceUID 排序，並把缺的 slices 補 0。
- SynthSEG 為 **多 label**（保留 label 值）。重建 NIfTI 時需能依 segment metadata 保留 label（至少以 `SegmentNumber` 作為 label 值的來源；若有 mapping 需求需明確定義）。

## 影響範圍
- 新增工具：`code_ai/pipeline/dicomseg/seg_to_nifti.py`（DICOM-SEG + headers → NIfTI）
- 新增獨立入口：`pipeline_followup_from_dicomseg.py`（不修改既有 `pipeline_followup.py`，以零侵入方式啟用新模式）
- 新增測試：針對幾何、缺 slice 補 0、多 label 合成、缺 header 欄位錯誤訊息

## 臨床考量
此變更僅影響「如何取得/生成配準用的 SynthSEG NIfTI」，不改變後續配準/病灶配對的核心邏輯；在輸入資料一致的前提下，應維持同等輸出可重現性。若 header 缺失或幾何不一致，系統需給出清楚錯誤以避免誤用造成錯位判讀。

## 技術考量與限制
- 需取得 **每個 SOP instance 的 header**（至少包含：`SOPInstanceUID`, `ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`, `Rows`, `Columns`）。
- 若影像幾何不完整（缺 IOP/IPP 或 slice spacing 無法推斷），只能 fail-fast 並提示缺欄位。
- 若 DICOM-SEG 的 segment label 編碼與原 SynthSEG label 值不一致，則無法保證 bitwise 相同；需定義 mapping 來源（例如 SegmentNumber 或 SegmentLabel→label 映射表）。

