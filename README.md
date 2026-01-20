# RADAX Aneurysm + Vessel Dilated Pipeline

本專案提供一套以 `pipeline_aneurysm_tensorflow.py` 為主的推論流程，產出：

- **Aneurysm**：DICOM-SEG（MRA_BRAIN + MIP_Pitch + MIP_Yaw）與 `rdx_aneurysm_pred_json.json`
- **Vessel Dilated**：DICOM-SEG 與 `rdx_vessel_dilated_json.json`
- **上傳資料夾結構**：依 `StudyInstanceUID` + `inference_id` 分層複製到指定目錄
- **Inference Complete API**：對 `aneurysm_model` 與 `vessel_model` 各 POST 一次

---

## 1. 主要入口

- 推論主流程：`pipeline_aneurysm_tensorflow.py`
- Aneurysm JSON + DICOM-SEG 產生：`code_ai/pipeline/dicomseg/build_aneurysm.py`
- Vessel Dilated JSON + DICOM-SEG 產生：`code_ai/pipeline/dicomseg/build_vessel_dilated.py`

---

## 1.1 Follow-up（前後比較）模組（檔案導向）

此模組是依 `openspec` 中的 `add-followup-comparison-module` 規格移植，主要用於 **baseline vs followup** 的病灶比較（配準 → 合併 pred → 產出 Followup JSON）。

- 主管道：`pipeline_followup.py`
- 啟動腳本：`pipeline_followup.sh`
- 配準：`registration.py`（使用 FSL `flirt`）
- 合併 pred：`generate_followup_pred.py`（輸出 `Pred_<model>_followup.nii.gz`，label=1/2/3）
- 報告：`followup_report.py`（輸出 `Followup_<model>_platform_json.json`）
- 依賴：`requirements_followup.txt`

### 示範命令（舊入口：提供 NIfTI）

```bash
python pipeline_followup.py \
  --baseline_ID "<Baseline_ID>" \
  --baseline_Inputs "<Baseline_Pred.nii.gz>" "<Baseline_SynthSEG.nii.gz>" \
  --baseline_DicomDir "<Baseline_DICOM_Dir_or_empty>" \
  --baseline_DicomSegDir "<Baseline_DicomSegDir>" \
  --baseline_json "<Baseline_platform_json.json>" \
  --followup_ID "<Followup_ID>" \
  --followup_Inputs "<Followup_Pred.nii.gz>" "<Followup_SynthSEG.nii.gz>" \
  --followup_DicomSegDir "<Followup_DicomSegDir>" \
  --followup_json "<Followup_platform_json.json>" \
  --path_output "<Output_Folder>" \
  --model "CMB" \
  --path_process "./process/Deep_FollowUp/" \
  --path_log "./log/" \
  --fsl_flirt_path "/usr/local/fsl/bin/flirt" \
  --upload_json
```

### 示範命令（新入口：不提供任何 NIfTI，用 Pred/SynthSEG 的 DICOM-SEG + headers 反推）

```bash
python pipeline_followup_from_dicomseg.py \
  --baseline_ID "<Baseline_ID>" \
  --baseline_pred_dicomseg "<Baseline_Pred_<model>.dcm>" \
  --baseline_synthseg_dicomseg "<Baseline_SynthSEG_<model>.dcm>" \
  --baseline_series_headers "<Baseline_series_headers.json>" \
  --baseline_DicomDir "<Baseline_DICOM_Dir_or_empty>" \
  --baseline_DicomSegDir "<Baseline_DicomSegDir>" \
  --baseline_json "<Baseline_platform_json.json>" \
  --followup_ID "<Followup_ID>" \
  --followup_pred_dicomseg "<Followup_Pred_<model>.dcm>" \
  --followup_synthseg_dicomseg "<Followup_SynthSEG_<model>.dcm>" \
  --followup_series_headers "<Followup_series_headers.json>" \
  --followup_DicomSegDir "<Followup_DicomSegDir>" \
  --followup_json "<Followup_platform_json.json>" \
  --path_output "<Output_Folder>" \
  --model "CMB" \
  --path_process "./process/Deep_FollowUp/" \
  --path_log "./log/" \
  --fsl_flirt_path "/usr/local/fsl/bin/flirt" \
  --upload_json
```

---

## 1.2 Follow-up v3（多日期比較，JSON 導向）

`pipeline_followup_v3.py` 以 `follow_up_input.json` 作為輸入，讀取 current + prior 多筆日期，
自動找到對應 case 資料夾，並輸出 `followup_manifest.json` 與每個比較的
`prediction-grouped.json`（保留既有 v1 過程檔）。

### 輸入（follow_up_input.json）

- `current_study`：當前日期（包含 `patient_id`、`study_date`、`inference_id`）
- `prior_study_list`：多筆歷史日期

### 主要輸出

- `process/Deep_FollowUp/<current_case_id>/<model>/outputs/followup_manifest.json`
- `process/Deep_FollowUp/<current_case_id>/<model>/outputs/<case_id>/prediction-grouped.json`

### 示範命令

```bash
python pipeline_followup_v3.py \
  --input_json "/path/to/follow_up_input.json" \
  --path_process "/path/to/process/Deep_Aneurysm" \
  --model "Aneurysm" \
  --prediction_json_name "rdx_aneurysm_pred_json.json" \
  --fsl_flirt_path "/usr/local/fsl/bin/flirt"
```

---

## 2. 執行方式

### 2.1 直接執行（命令列）

在 Linux/WSL 環境下：

```bash
python pipeline_aneurysm_tensorflow.py \
  --ID "17390820_20250604_MR_21406040004" \
  --Inputs "/data/xxx/MRA_BRAIN.nii.gz" \
  --DicomDir "/data/xxx/MRA_BRAIN/" \
  --Output_folder "/data/xxx/output/"
```

> `--Inputs` 目前僅使用第 1 個（MRA_BRAIN nifti）。

---

## 3. 依賴與環境

此 pipeline 需要 GPU + TensorFlow 環境（實際推論由 `gpu_aneurysm.py` 執行），並使用多個影像/醫療影像套件：

- `tensorflow`
- `numpy`
- `pandas`
- `nibabel`
- `pydicom`
- `SimpleITK`
- `pydicom-seg`
- `scikit-image`
- `opencv-python`
- `matplotlib`
- `requests`
- `pynvml`

> 若你只要產 JSON / DICOM-SEG（不跑 GPU 推論），仍需 `pydicom`、`nibabel`、`SimpleITK`、`pydicom-seg` 等套件。

---

## 4. Pipeline 流程摘要（高階）

`pipeline_aneurysm_tensorflow.py` 的主要流程：

1. 建立 `process/<ID>/nnUNet/` 工作目錄
2. 呼叫 `gpu_aneurysm.py` 產出（範例）：
   - `Pred.nii.gz`
   - `Vessel.nii.gz`
   - `Vessel_16.nii.gz`
   - `Prob.nii.gz`（若有）
3. 將必要 nifti 複製到 `nnUNet/Image_nii/`，並做 reslice → `nnUNet/Image_reslice/`
4. 產生 MIP 相關資料（包含 `MIP_Pitch_pred.nii.gz`、`MIP_Yaw_pred.nii.gz`，與 `nnUNet/Dicom/MIP_Pitch`、`nnUNet/Dicom/MIP_Yaw`）
5. 由 Excel 產出 aneurysm 對應資訊（`Aneurysm_Pred_list.xlsx`）
6. 產生輸出：
   - Aneurysm：`rdx_aneurysm_pred_json.json` + `Dicom/Dicom-Seg/*.dcm`
   - Vessel Dilated：`rdx_vessel_dilated_json.json` + `Dicom/Dicom-Seg/MRA_BRAIN_Vessel_A1.dcm`
7. 複製檔案到 `upload_dir`（見下一節）
8. 對 inference-complete API 發送兩次 POST（aneurysm/vessel）

---

## 5. 上傳資料夾結構（upload_dir）

`pipeline_aneurysm_tensorflow.py` 會把結果複製到：

- `upload_dir/<StudyInstanceUID>/aneurysm_model/<aneurysm_inference_id>/...`
- `upload_dir/<StudyInstanceUID>/vessel_model/<vessel_inference_id>/...`

其中：

- Aneurysm 的 DICOM-SEG 檔名規則：`<detections[i].series_instance_uid>_<Ai>.dcm`
- Vessel Dilated 的 DICOM-SEG 檔名規則：`<detections[0].series_instance_uid>.dcm`
- Pitch/Yaw 的 series 資料夾（資料夾名 = `reformatted_series[*].series_instance_uid`）內放該序列 DICOM 影像（來源：`nnUNet/Dicom/MIP_Pitch` / `nnUNet/Dicom/MIP_Yaw`）

---

## 6. Inference Complete API

程式會對同一個 `studyInstanceUid` 呼叫兩次：

```json
{
  "studyInstanceUid": "xxx",
  "modelName": "aneurysm_model",
  "inferenceId": "..."
}
```

```json
{
  "studyInstanceUid": "xxx",
  "modelName": "vessel_model",
  "inferenceId": "..."
}
```

API 端點在 `pipeline_aneurysm_tensorflow.py` 內：

- `api_url = 'http://localhost:24000/v1/ai-inference/inference-complete'`

---

## 7. Model IDs

- **Aneurysm**：`924d1538-597c-41d6-bc27-4b0b359111cf`
- **Vessel Dilated**：`289fc383-34ff-46b7-bf0a-fbb22a104a18`

---

## 8. 輸出 JSON 檔案

### 8.1 Aneurysm：`rdx_aneurysm_pred_json.json`

- `inference_id`：uuid4
- `input_study_instance_uid` / `input_series_instance_uid`：list（以 MRA_BRAIN 為主）
- `detections`：主要病灶列表（含 pitch/yaw angle）
- `reformatted_series`：pitch/yaw series（不包含 `pitch_angle/yaw_angle` key）

### 8.2 Vessel Dilated：`rdx_vessel_dilated_json.json`

- `model_id` 固定為：`289fc383-34ff-46b7-bf0a-fbb22a104a18`
- `detections[0]` 對應單一 vessel mask 的 DICOM-SEG

---

## 9. 注意事項

- 本 repo 中沒有 `code_ai.pipeline.rdx.build.*`，因此 `build_vessel_dilated.py` 已改為自包含實作，不再依賴舊 builder。
- 若後端 API 回傳 DB FK constraint（modelId 不存在），需先在後端資料庫註冊對應 `model_id`。


