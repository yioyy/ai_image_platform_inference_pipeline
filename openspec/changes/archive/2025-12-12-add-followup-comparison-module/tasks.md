# 實作任務清單 - Follow-up 比較模組（檔案導向版本）

> 本清單以 **OpenSpec** 思維編排：每個任務都應產出「可驗收的檔案 / 函數 / 行為」，並可逐項勾選完成。

## 0. 術語與產物定義（先決條件）
- [ ] 0.1 **名詞統一**
  - **baseline**：基準檢查（Fixed / Reference 空間）
  - **followup**：追蹤檢查（Moving，需配準到 baseline 空間）
- [ ] 0.2 **狀態字串固定**：`new`、`stable`、`enlarging`、`shrinking`、`disappeared`
- [ ] 0.3 **Follow-up Pred 合併輸出標籤固定**
  - **1 = new**
  - **2 = stable（重疊/延續）**
  - **3 = disappeared**

## 1. 專案結構建立（P0）
- [ ] 1.1 建立處理資料夾根目錄（若不存在）
  - `process/Deep_FollowUp/`
- [ ] 1.2 新增 `pipeline_followup.py`
- [ ] 1.3 新增 `pipeline_followup.sh`
- [ ] 1.4 新增 `requirements_followup.txt`
- [ ] 1.5 更新 `README.md`（新增 Follow-up 用法/參數/輸出）

## 2. 影像配準模組 `registration.py`（P0）

### 2.1 導入既有配準流程（直接沿用）
- [ ] 2.1.1 將既有 `run_registration(...)` 整理為可重用函數（放進 `registration.py`）
- [ ] 2.1.2 修正文檔範例碼瑕疵（僅示意；實作以 `registration.py` 為準）
  - 移除 `from turtle import mode`（無用途）
  - `model='CMB '` 改為 `model='CMB'`（避免路徑多一個空白）
  - Step 編號修正（apply label 為 Step 2）

```python
import os

def run_registration(patient_id, patient_info_list, path_data, path_mat, model="CMB", fsl_path="/usr/local/fsl/bin/flirt"):
    """
    執行縱向影像對位 (Registration)
    邏輯：將 Follow-up (Moving) 對位到 Baseline (Reference) 空間
    儲存：結果存放在 Baseline 資料夾

    Args:
        patient_id (str): 病患 ID
        patient_info_list (list): 該 ID 對應的資訊列表 (dict_IDs[ID])
        path_data (str): 資料根目錄路徑
        path_mat (str): 矩陣檔案儲存路徑
        model (str): 模型代碼（用於組檔名）
        fsl_path (str): FSL flirt 指令路徑（包含 flirt 執行檔）
    """
    info = patient_info_list[0]
    baseline_id = info["baseline_ID"]
    followup_id = info["followup_ID"]

    out_label_reg_file = os.path.join(path_data, baseline_id, f"Pred_{model}_registration.nii.gz")
    if os.path.isfile(out_label_reg_file):
        return

    ref_synthseg_file = os.path.join(path_data, baseline_id, f"SynthSEG_{model}.nii.gz")
    mov_synthseg_file = os.path.join(path_data, followup_id, f"SynthSEG_{model}.nii.gz")
    mov_label_file = os.path.join(path_data, followup_id, f"Pred_{model}.nii.gz")

    out_synthseg_reg_file = os.path.join(path_data, baseline_id, f"SynthSEG_{model}_registration.nii.gz")
    mat_file = os.path.join(path_mat, f"{baseline_id}_{model}.mat")

    # Step 1: 計算矩陣（預設 dof 12；若要剛性可改 dof 6）
    cmd_calc_mat = (
        f"{fsl_path} -in {mov_synthseg_file} -ref {ref_synthseg_file} -out {out_synthseg_reg_file} "
        f"-omat {mat_file} -bins 256 -cost corratio "
        f"-searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp nearestneighbour"
    )
    os.system(cmd_calc_mat)

    # Step 2: 套用矩陣到 followup Pred（label 用 nearestneighbour）
    cmd_apply_label = (
        f"{fsl_path} -in {mov_label_file} -ref {ref_synthseg_file} -out {out_label_reg_file} "
        f"-applyxfm -init {mat_file} -interp nearestneighbour"
    )
    os.system(cmd_apply_label)
```

### 2.2 錯誤處理（P0）
- [ ] 2.2.1 如果 `flirt` 失敗（return code != 0 或輸出檔不存在）要回報明確錯誤訊息
- [ ] 2.2.2 規劃重試策略（例如：調整 cost / dof / 搜尋範圍），至少留下 TODO 與 log 訊息

## 3. Pred 合併模組 `generate_followup_pred.py`（P0）
- [ ] 3.1 實作 `generate_followup_pred(baseline_pred_file, followup_reg_pred_file, output_file)`：
  - 讀入 baseline 的 `Pred_<model>.nii.gz`
  - 讀入 followup 的 `Pred_<model>_registration.nii.gz`
  - 將兩者的 **>0 全部視為 1**（二值化）
  - 依規則輸出合併標籤：
    - baseline=1、followup=0 → **1 (new)**
    - baseline=1、followup=1 → **2 (stable)**
    - baseline=0、followup=1 → **3 (disappeared)**
  - 另存為 `Pred_<model>_followup.nii.gz`
- [ ] 3.2 補上最小單元測試（至少 3 個 case：全空、全重疊、只有其中一邊有）

## 4. Follow-up JSON 報告模組 `followup_report.py`（P0）

### 4.1 定義主要入口函數
- [ ] 4.1.1 實作 `make_followup_json(...)`（命名可依專案現況調整，但需維持單一入口）

```python
def make_followup_json(
    baseline_pred_file: str,
    baseline_json_file: str,
    followup_json_file: str,
    followup_reg_pred_file: str,
    registration_mat_file: str,
    output_json_file: str,
    group_id: int = 56,
):
    """產出 Followup_<model>_platform_json.json（以 baseline JSON 為底，加入 followup 欄位與病灶配對/狀態）"""
    ...
```

### 4.2 JSON 結構（示意，請以平台現行欄位為準）
- [ ] 4.2.1 在 `study.series[0]` 新增：
  - `followup_study_instance_uid`
  - `followup_series_instance_uid`
- [ ] 4.2.2 `mask.model[].series[].instances[]` 的每顆 baseline 病灶新增：
  - `followup_mask_index`
  - `followup_diameter`
  - `followup_seg_series_instance_uid`
  - `followup_seg_sop_instance_uid`
  - `followup_dicom_sop_instance_uid`

> 注意：**JSON 不支援註解**；下方僅為欄位示意（值用空字串/數字占位）。

```json
{
  "study": {
    "group_id": 56,
    "study_instance_uid": "BASELINE_STUDY_UID",
    "series": [
      {
        "series_type": "7",
        "series_instance_uid": "BASELINE_SERIES_UID",
        "followup_study_instance_uid": "FOLLOWUP_STUDY_UID",
        "followup_series_instance_uid": "FOLLOWUP_SERIES_UID",
        "resolution_x": 512,
        "resolution_y": 512
      }
    ]
  },
  "mask": {
    "study_instance_uid": "BASELINE_STUDY_UID",
    "group_id": 56,
    "model": [
      {
        "model_type": "2",
        "series": [
          {
            "series_instance_uid": "BASELINE_SERIES_UID",
            "followup_series_instance_uid": "FOLLOWUP_SERIES_UID",
            "series_type": "7",
            "instances": [
              {
                "mask_index": 1,
                "mask_name": "A1",
                "diameter": "2.0",
                "dicom_sop_instance_uid": "BASELINE_SOP_UID",
                "status": "new",
                "followup_mask_index": "",
                "followup_diameter": "",
                "followup_dicom_sop_instance_uid": "",
                "followup_seg_series_instance_uid": "",
                "followup_seg_sop_instance_uid": ""
              }
            ]
          }
        ]
      }
    ]
  }
}
```

### 4.3 病灶配對與狀態分類（核心邏輯要可測）（P0）
- [ ] 4.3.1 讀取：
  - baseline：`Pred_<model>.nii.gz`、`Pred_<model>_platform_json.json`
  - followup：`Pred_<model>_platform_json.json`
  - registration：`Pred_<model>_registration.nii.gz`、`<baseline_id>_<model>.mat`
- [ ] 4.3.2 建立「baseline label index → 是否與 followup(reg) label 重疊」的映射
- [ ] 4.3.3 若重疊到 followup label，從 followup JSON 取回對應欄位並填入：
  - `followup_mask_index`、`followup_diameter`
  - `followup_seg_series_instance_uid`、`followup_seg_sop_instance_uid`
  - `followup_dicom_sop_instance_uid`
- [ ] 4.3.4 狀態分類規則（必須完全一致）：
  - baseline 有、followup(reg) 無重疊 → `new`
  - baseline 有、followup(reg) 有重疊：
    - `diameter > followup_diameter` → `enlarging`
    - `diameter == followup_diameter` → `stable`
    - `diameter < followup_diameter` → `shrinking`
  - followup(reg) 有、baseline 無重疊 → 產生「baseline 缺席」的一筆 → `disappeared`

### 4.4 `registration.mat` 對 slice 的 SOP UID 轉換（P0）
> 目標：在 `new` / `disappeared` 狀態下，即使其中一側「沒有被標註到」，仍能用配準矩陣推回對應的 SOP UID（依你原描述的「z 比例」做近似）。

#### 4.4.A 方向定義（必須先釐清）
- **既定方向（預設）**：本模組的配準是 **Follow-up 對到 Baseline**，因此矩陣 \(T_{F \rightarrow B}\) 滿足：
  - WHEN 給定 followup 空間的點 \(p_F\)
  - THEN \(p_B = T_{F \rightarrow B}(p_F)\)
- **反向需求**：若需要從 Baseline 反推到 Follow-up，系統 MUST 使用反矩陣：
  - \(T_{B \rightarrow F} = (T_{F \rightarrow B})^{-1}\)

#### 4.4.B 輸入方向相反時的額外處理（OpenSpec 行為規範）
- WHEN 偵測到使用者/呼叫端提供的矩陣方向與既定方向不一致（例如輸入其實是 \(T_{B \rightarrow F}\)）
- THEN 系統 MUST 明確標記矩陣方向並做下列其一（不可默默混用）：
  - **方案 1（推薦）**：將輸入矩陣轉成既定方向：\(T_{F \rightarrow B} = (T_{B \rightarrow F})^{-1}\)
  - **方案 2**：保留輸入矩陣，但後續所有查詢 MUST 以 `transform_direction` 明確選用 `T` 或 `T^{-1}`

#### 4.4.C SOP UID 映射（z 比例近似）（任務拆解）
- [ ] 4.4.1 定義 `transform_direction`（enum）：`F2B`（followup→baseline）、`B2F`（baseline→followup）
- [ ] 4.4.2 將 SOP UID 在 `sorted` 陣列中的 index 轉為比例（0~1）
- [ ] 4.4.3 根據 `transform_direction` 選用 \(T_{F \rightarrow B}\) 或 \(T_{B \rightarrow F}\)，推算新 z 比例（只針對 z 軸）
- [ ] 4.4.4 將新 z 比例映射回另一個檢查的 `sorted` 陣列 index，回填 SOP UID
- [ ] 4.4.5 將以上步驟封裝為**單一純函數**（便於單元測試），並在輸入方向不明確時 early-return 並輸出可讀錯誤

## 5. 上傳模組（先寫在主腳本內）（P1）
- [ ] 5.1 參考 `after_run_infarct.py` 的 `upload_json_aiteam`，提供同等行為（URL/headers/錯誤處理一致）
- [ ] 5.2（選用）封裝 dicom-seg 上傳（若專案已有既存 uploader，直接重用）

## 6. 主管道腳本 `pipeline_followup.py`（P0）

### 6.1 CLI 參數（P0）
- [ ] 6.1.1 實作命令列參數解析（修正原文件的重複參數）
  - `--baseline_ID`
  - `--baseline_Inputs`（2 個：Pred、SynthSEG）
  - `--baseline_DicomDir`
  - `--baseline_DicomSegDir`
  - `--baseline_json`
  - `--followup_ID`
  - `--followup_Inputs`（2 個：Pred、SynthSEG）
  - `--followup_DicomSegDir`
  - `--followup_json`
  - `--path_output`
  - `--model`（預設 `cmb`）
  - `--path_code` / `--path_process` / `--path_json` / `--path_log`
  - `--gpu_n`

### 6.2 Stage 編排（P0）
- [ ] 6.2.1 Stage 1：驗證輸入、建立資料夾、複製輸入、初始化 log
- [ ] 6.2.2 Stage 2：呼叫 `registration.py` 產出 registration 檔與 `.mat`
- [ ] 6.2.3 Stage 3：呼叫 `generate_followup_pred.py` 輸出 `Pred_<model>_followup.nii.gz`
- [ ] 6.2.4 Stage 4：呼叫 `followup_report.py` 輸出 `Followup_<model>_platform_json.json`
- [ ] 6.2.5 Stage 5（P1）：上傳 JSON（與 dicom-seg）
- [ ] 6.2.6 複製結果到 `output/{baseline_id}/`

### 6.3 錯誤處理與日誌（P0）
- [ ] 6.3.1 每個 stage 用 try/except 包裝，log 出：開始/結束時間、錯誤堆疊
- [ ] 6.3.2 任何外部指令（如 `flirt`）須記錄完整 command 與關鍵輸出路徑

## 7. Shell 啟動腳本 `pipeline_followup.sh`（P0）
- [ ] 7.1 撰寫 Shell 腳本（參考既有 pipeline 腳本）
- [ ] 7.2 參數位置核對（避免 `$9` 之後位移錯）
- [ ] 7.3 設定執行權限（Linux）

## 8. 測試（P0）
- [ ] 8.1 單元測試：`generate_followup_pred` 的 3 個 case
- [ ] 8.2 單元測試：病灶配對與狀態分類（至少 new/stable/disappeared 各 1）
- [ ] 8.3 整合測試：準備 1 組 baseline/followup 真實資料，跑完整 pipeline，驗證輸出檔存在

## 9. 文件（P1）
- [ ] 9.1 `README.md`：加入 Follow-up 模組使用範例（CLI）
- [ ] 9.2 `README.md`：清楚列出輸出檔與欄位新增（followup_* 欄位）

## 10. 程式碼品質（P0）
- [ ] 10.1 Linus 風格：縮排 ≤3 層、Early Return、避免 if/else 鏈
- [ ] 10.2 DICOM 標籤存取使用 `.get()`（若涉及 pydicom）

## 11. 部署準備（P1）
- [ ] 11.1 更新 `requirements_followup.txt`（與 proposal 的依賴一致）
- [ ] 11.2 準備至少 3 組測試資料（穩定/新增/消失）作為回歸測試基準
