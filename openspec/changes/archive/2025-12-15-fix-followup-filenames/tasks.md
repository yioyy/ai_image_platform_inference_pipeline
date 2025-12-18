## 1. 規格與文件
- [ ] 1.1 更新 OpenSpec 規格增量（檔名規則）
- [ ] 1.2 更新 `README.md`：輸出 JSON 檔名改為 `Followup_<model>_platform_json.json`、`--path_process` 預設路徑與範例一致、`--followup_json` 範例為 `Pred_<model>_platform_json.json`
- [ ] 1.3 更新 Follow-up 既有提案文件（archive）中的檔名與路徑描述

## 2. 程式碼修正
- [ ] 2.1 `pipeline_followup.py`：
  - [ ] 2.1.1 `result/` 輸出 JSON 改為 `Followup_<model>_platform_json.json`
  - [ ] 2.1.2 `registration/` 的 `.mat` 檔名改為 `baseline_<model>.mat`
  - [ ] 2.1.3 process root 支援 `--path_process` 已包含 `Deep_Followup/` 的情境（不重複拼接子目錄）
  - [ ] 2.1.4 **不得修改** CLI parser 的 default 行為（以 `pipeline_followup.py` 內既有 parser 設定為準）

## 3. 驗收
- [ ] 3.1 執行 Follow-up 後，輸出資料夾包含：
  - [ ] `Pred_<model>_followup.nii.gz`
  - [ ] `Followup_<model>_platform_json.json`
  - [ ] `dicom-seg/baseline/`、`dicom-seg/followup/`
- [ ] 3.2 處理資料夾 `registration/` 內包含：
  - [ ] `Pred_<model>_registration.nii.gz`
  - [ ] `SynthSEG_<model>_registration.nii.gz`
  - [ ] `baseline_<model>.mat`


