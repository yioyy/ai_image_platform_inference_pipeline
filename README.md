# 腦部影像 AI 分析系統

## 專案簡介

本系統是一個完整的腦部影像 AI 分析平台，整合三大臨床應用：**腦動脈瘤檢測**、**急性腦梗塞分析**、**白質病變評估**。使用深度學習模型對腦部 MRI/MRA 影像進行自動化分析，提供臨床決策支援。

### 三大核心功能模組

#### 1. 腦動脈瘤檢測 (Aneurysm Detection)
- **適用影像**: MRA (Magnetic Resonance Angiography)
- **主要功能**: 動脈瘤自動檢測、血管分割、MIP 影像生成
- **模型架構**: nnU-Net 3D Full Resolution
- **輸出**: 動脈瘤位置、大小、體積、所在血管區域

#### 2. 急性腦梗塞分析 (Acute Infarct Detection)
- **適用影像**: DWI (Diffusion Weighted Imaging), ADC
- **主要功能**: 急性梗塞自動偵測、體積量測、腦區定位
- **模型架構**: nnU-Net 2D
- **輸出**: 梗塞範圍、體積、受影響腦區、ASPECTS 評分

#### 3. 白質病變評估 (White Matter Hyperintensity)
- **適用影像**: T2-FLAIR
- **主要功能**: 白質病變自動分割、Fazekas 分級、體積量測
- **模型架構**: nnU-Net 2D
- **輸出**: WMH 體積、Fazekas 評分、分佈位置

---

## 系統需求

### 硬體需求
- **GPU**: NVIDIA GPU (建議 16GB 以上顯存)
- **記憶體**: 建議 32GB 以上
- **儲存空間**: 每個案例約需 2-5GB 暫存空間

### 軟體需求
- **作業系統**: Linux (Ubuntu 20.04+)
- **Python**: 3.9-3.11 (推薦 3.10)
- **Conda/Miniconda**: 用於環境管理
- **CUDA**: 12.4
- **cuDNN**: 對應 CUDA 12.4 的版本
- **GDCM**: 用於 DICOM 影像處理

### Python 套件依賴

| 套件 | 版本 | 用途 |
|------|------|------|
| PyTorch | 2.6.0 | nnU-Net 深度學習框架 (CUDA 12.4) |
| TensorFlow | 2.14.0 | 血管分割與 SynthSeg 模型 |
| NumPy | 1.26.4 | 數值計算 |
| nibabel | 5.2.1 | NIfTI 影像處理 |
| SimpleITK | 2.3.1 | 醫學影像處理 |
| scikit-image | 0.22.0 | 影像處理與形態學操作 |
| pydicom | latest | DICOM 影像處理 |
| opencv-python | latest | 影像處理 |
| pandas | latest | 資料處理 |
| matplotlib | latest | 視覺化 |
| pynvml | latest | GPU 記憶體監控 |
| brainextractor | latest | 腦部組織提取 (BET) |

**注意**: NumPy 版本限制為 < 2.0，因為 DICOM 壓縮功能尚未相容 NumPy 2.x

---

## 安裝說明

### 1. 安裝 Conda 環境

```bash
# 初始化 conda
source /home/tmu/miniconda3/etc/profile.d/conda.sh

# 建立環境 (Python 3.10)
conda create -n tf_2_14 python=3.10

# 啟動環境
conda activate tf_2_14
```

### 2. 安裝 Python 套件

```bash
# 步驟 1: 安裝 PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 步驟 2: 安裝 TensorFlow
pip install tensorflow==2.14.0

# 步驟 3: 安裝醫學影像處理套件
pip install numpy==1.26.4
pip install nibabel==5.2.1
pip install SimpleITK==2.3.1
pip install scikit-image==0.22.0
pip install pydicom opencv-python pandas matplotlib pynvml
pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg

# 步驟 4: 安裝 BET 腦部提取工具
pip install brainextractor
# 或安裝 FSL: sudo apt-get install fsl
```

### 3. 安裝 GDCM

```bash
# Ubuntu/Debian
sudo apt-get install libgdcm-tools

# 或從源碼編譯
# 詳見: https://gdcm.sourceforge.net/
```

### 4. 準備模型權重

確保以下模型權重已下載至正確路徑：
```bash
# 動脈瘤模型
/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset080_DeepAneurysm/

# 急性梗塞模型
/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset040_DeepInfarct/

# 白質病變模型
/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset015_DeepLacune/

# SynthSeg 腦區分割模型
/data/4TB1/pipeline/chuan/code/model_weights/SynthSeg_parcellation_tf28/
```

---

## 使用方法

### 模組 1: 腦動脈瘤檢測

#### 使用 Shell 腳本（推薦）
```bash
bash pipeline_aneurysm.sh \
    "<Patient_ID>" \
    "<MRA_BRAIN_NII_Path>" \
    "<DICOM_Directory>" \
    "<Output_Folder>"
```

#### 範例
```bash
bash pipeline_aneurysm.sh \
    "17390820_20250604_MR_21406040004" \
    "/data/input/17390820_20250604_MR_21406040004/MRA_BRAIN.nii.gz" \
    "/data/inputDicom/17390820_20250604_MR_21406040004/MRA_BRAIN/" \
    "/data/output/"
```

#### 直接執行 Python
```bash
conda activate tf_2_14

python pipeline_aneurysm_tensorflow.py \
    --ID "17390820_20250604_MR_21406040004" \
    --Inputs "/data/input/MRA_BRAIN.nii.gz" \
    --DicomDir "/data/inputDicom/MRA_BRAIN/" \
    --Output_folder "/data/output/"

conda deactivate
```

---

### 模組 2: 急性腦梗塞分析

#### 使用 Shell 腳本（推薦）
```bash
bash pipeline_infarct.sh \
    "<Patient_ID>" \
    "<ADC_NII_Path>" \
    "<DWI0_NII_Path>" \
    "<DWI1000_NII_Path>" \
    "[SynthSEG_NII_Path]" \
    "<ADC_DICOM_Dir>" \
    "<DWI0_DICOM_Dir>" \
    "<DWI1000_DICOM_Dir>" \
    "<Output_Folder>"
```

#### 範例
```bash
bash pipeline_infarct.sh \
    "01550089_20251117_MR_21411150023" \
    "/data/input/01550089_20251117_MR_21411150023/ADC.nii.gz" \
    "/data/input/01550089_20251117_MR_21411150023/DWI0.nii.gz" \
    "/data/input/01550089_20251117_MR_21411150023/DWI1000.nii.gz" \
    "" \
    "/data/inputDicom/01550089_20251117_MR_21411150023/ADC/" \
    "/data/inputDicom/01550089_20251117_MR_21411150023/DWI0/" \
    "/data/inputDicom/01550089_20251117_MR_21411150023/DWI1000/" \
    "/data/output/"
```

**註**: 如果未提供 SynthSEG，系統會自動執行 SynthSEG 腦區分割（增加約 60-90 秒處理時間）

#### 直接執行 Python
```bash
conda activate tf_2_14

python pipeline_infarct_torch.py \
    --ID "01550089_20251117_MR_21411150023" \
    --Inputs "/data/input/ADC.nii.gz" "/data/input/DWI0.nii.gz" "/data/input/DWI1000.nii.gz" \
    --DicomDir "/data/inputDicom/ADC/" "/data/inputDicom/DWI0/" "/data/inputDicom/DWI1000/" \
    --Output_folder "/data/output/"

conda deactivate
```

---

### 模組 3: 白質病變評估

#### 使用 Shell 腳本（推薦）
```bash
bash pipeline_wmh.sh \
    "<Patient_ID>" \
    "<T2FLAIR_NII_Path>" \
    "[SynthSEG_NII_Path]" \
    "<T2FLAIR_DICOM_Dir>" \
    "<Output_Folder>"
```

#### 範例
```bash
bash pipeline_wmh.sh \
    "01550089_20251117_MR_21411150023" \
    "/data/input/01550089_20251117_MR_21411150023/T2FLAIR_AXI.nii.gz" \
    "" \
    "/data/inputDicom/01550089_20251117_MR_21411150023/T2FLAIR_AXI/" \
    "/data/output/"
```

#### 直接執行 Python
```bash
conda activate tf_2_14

python pipeline_wmh_torch.py \
    --ID "01550089_20251117_MR_21411150023" \
    --Inputs "/data/input/T2FLAIR_AXI.nii.gz" \
    --DicomDir "/data/inputDicom/T2FLAIR_AXI/" \
    --Output_folder "/data/output/"

conda deactivate
```

---

### 模組 4: Follow-up 前後比較（檔案導向）

此模組用於比較 **baseline** 與 **followup** 兩次檢查（檔案導向），流程重點如下：
- 複製 baseline/followup 的 `Pred_<model>.nii.gz`、`SynthSEG_<model>.nii.gz` 與平台 JSON 到處理目錄
- **配準前會以 SynthSEG 的幾何資訊（affine/sform/qform）覆蓋 Pred**（shape 必須一致，避免 voxel grid 不一致導致 overlap=0）
- 使用 FSL `flirt` 將 followup 對位到 baseline，輸出 `<baseline_id>_<model>.mat` 與 `Pred_<model>_registration.nii.gz`
- 產出：
  - `Pred_<model>_followup.nii.gz`：合併標註（1=new、2=stable、3=disappeared）
  - `Followup_<model>_platform_json.json`：平台格式 JSON（在 baseline JSON 上新增 followup 欄位與病灶配對狀態）
- `--baseline_DicomDir` 可留空；baseline/followup 的 dicom-seg 會被複製到輸出資料夾

#### 直接執行 Python（建議在 Linux / 已安裝 FSL 的環境）

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
  --followup_json "<Pred_<model>_platform_json.json>" \
  --path_output "<Output_Folder>" \
  --model "CMB" \
  --path_process "./process/Deep_FollowUp/" \
  --path_log "./log/" \
  --fsl_flirt_path "/usr/local/fsl/bin/flirt" \
  --upload_json
```

#### （新入口）不提供任何 NIfTI：用 Pred/SynthSEG 的 DICOM-SEG + headers 反推 NIfTI

> 只需要 **per-instance headers json（無 PixelData）** + **Pred/SynthSEG 的 DICOM-SEG（含 PixelData）**，腳本會在暫存目錄生成 `Pred_<model>.nii.gz` 與 `SynthSEG_<model>.nii.gz`，再呼叫既有 `pipeline_followup.py` 流程。\n

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

#### 輸出（Follow-up）

```
<Output_Folder>/<Baseline_ID>/
├── Followup_<model>_platform_json.json
├── Pred_<model>_followup.nii.gz
└── dicom-seg/
    ├── baseline/     # baseline dicom-seg 複製
    └── followup/     # followup dicom-seg 複製
```

---

## 參數說明

### 模組 1: 動脈瘤檢測參數

| 參數 | 類型 | 說明 | 範例 |
|------|------|------|------|
| `--ID` | str | 病患識別碼 | `17390820_20250604_MR_21406040004` |
| `--Inputs` | str | MRA 影像路徑 | `/data/input/MRA_BRAIN.nii.gz` |
| `--DicomDir` | str | DICOM 目錄 | `/data/inputDicom/MRA_BRAIN/` |
| `--Output_folder` | str | 輸出資料夾 | `/data/output/` |

### 模組 2: 梗塞分析參數

| 參數 | 類型 | 說明 | 範例 |
|------|------|------|------|
| `--ID` | str | 病患識別碼 | `01550089_20251117_MR_21411150023` |
| `--Inputs` | list[str] | ADC, DWI0, DWI1000 路徑列表 | 見範例 |
| `--DicomDir` | list[str] | DICOM 目錄列表 | 見範例 |
| `--Output_folder` | str | 輸出資料夾 | `/data/output/` |

### 模組 3: WMH 評估參數

| 參數 | 類型 | 說明 | 範例 |
|------|------|------|------|
| `--ID` | str | 病患識別碼 | `01550089_20251117_MR_21411150023` |
| `--Inputs` | list[str] | T2FLAIR 路徑 (可選 SynthSEG) | `/data/input/T2FLAIR.nii.gz` |
| `--DicomDir` | list[str] | DICOM 目錄 | `/data/inputDicom/T2FLAIR/` |
| `--Output_folder` | str | 輸出資料夾 | `/data/output/` |

---

## 輸出說明

### 模組 1: 動脈瘤輸出檔案

```
<Output_Folder>/
├── Pred_Aneurysm.nii.gz              # 動脈瘤預測遮罩
├── Prob_Aneurysm.nii.gz              # 動脈瘤機率圖
├── Pred_Aneurysm_Vessel.nii.gz       # 血管分割結果
├── Pred_Aneurysm_Vessel16.nii.gz     # 16 區域血管分割
├── Pred_Aneurysm.json                # 預測結果 JSON
└── Pred_Aneurysm_platform_json.json  # 平台格式 JSON
```

### 模組 2: 梗塞輸出檔案

```
<Output_Folder>/
├── Pred_Infarct.nii.gz               # 梗塞預測遮罩
├── Prob_Infarct.nii.gz               # 梗塞機率圖
├── Pred_Infarct_parcellation.nii.gz  # 腦區標籤
├── Pred_Infarct.json                 # 預測結果 JSON
├── Pred_Infarct_platform_json.json   # 平台格式 JSON
└── visualization/                     # 視覺化圖片
    ├── overlay_*.png                  # 疊加圖
    └── report_*.png                   # 統計報告
```

### 模組 3: WMH 輸出檔案

```
<Output_Folder>/
├── Pred_WMH.nii.gz                   # WMH 預測遮罩
├── Prob_WMH.nii.gz                   # WMH 機率圖
├── Pred_WMH_parcellation.nii.gz      # 腦區標籤
├── Pred_WMH.json                     # 預測結果 JSON
├── Pred_WMH_platform_json.json       # 平台格式 JSON
└── visualization/                     # 視覺化圖片
    ├── overlay_*.png                  # 疊加圖
    └── fazekas_report.png             # Fazekas 評分報告
```

---

## 處理流程比較

### 模組 1: 動脈瘤檢測流程

```
1. 環境初始化 → GPU 記憶體檢查
2. 影像前處理 → 複製輸入 MRA
3. AI 推論 → nnU-Net 3D 動脈瘤偵測 + 血管分割
4. 影像重建 → Reslice + MIP 生成 (Pitch/Yaw)
5. 數據分析 → 動脈瘤量測 + Excel 報告
6. 格式轉換 → DICOM-SEG + JSON
7. 結果上傳 → Orthanc PACS + AI 平台
```

**預估時間**: 2-4 分鐘

### 模組 2: 梗塞分析流程

```
1. 環境初始化 → GPU 記憶體檢查
2. SynthSEG 處理 → 腦區分割 (如未提供)
3. 影像前處理 → BET 腦部提取 + 正規化
4. AI 推論 → nnU-Net 2D 梗塞偵測
5. 後處理 → 腦區統計 + 視覺化
6. 格式轉換 → DICOM-SEG + JSON + Excel
7. 結果上傳 → AI 平台
```

**預估時間**: 
- 有提供 SynthSEG: 90-120 秒
- 無 SynthSEG: 150-210 秒

### 模組 3: WMH 評估流程

```
1. 環境初始化 → GPU 記憶體檢查
2. SynthSEG 處理 → 腦區分割 (如未提供)
3. 影像前處理 → BET 腦部提取 + 正規化
4. AI 推論 → nnU-Net 2D WMH 分割
5. 後處理 → Fazekas 評分 + 視覺化
6. 格式轉換 → DICOM-SEG + JSON + Excel
7. 結果上傳 → AI 平台
```

**預估時間**: 
- 有提供 SynthSEG: 60-90 秒
- 無 SynthSEG: 120-180 秒

---

## 模型說明

### 動脈瘤檢測模型

| 項目 | 規格 |
|------|------|
| 模型架構 | nnU-Net 3D Full Resolution |
| 訓練資料集 | Dataset080_DeepAneurysm |
| 輸入 | MRA T1 影像 |
| 輸出 | 二元分割遮罩 + 機率圖 + 16 區域血管 |
| Group ID | 56 |
| 框架 | TensorFlow 2.14 |

### 急性梗塞檢測模型

| 項目 | 規格 |
|------|------|
| 模型架構 | nnU-Net 2D |
| 訓練資料集 | Dataset040_DeepInfarct |
| 輸入 | ADC + DWI0 + DWI1000 (3 通道) |
| 輸出 | 二元分割遮罩 + 機率圖 + 腦區統計 |
| Group ID | 57 |
| 框架 | PyTorch |

### 白質病變檢測模型

| 項目 | 規格 |
|------|------|
| 模型架構 | nnU-Net 2D |
| 訓練資料集 | Dataset015_DeepLacune |
| 輸入 | T2-FLAIR |
| 輸出 | 二元分割遮罩 + 機率圖 + Fazekas 評分 |
| Group ID | 58 |
| 框架 | PyTorch |

---

## 注意事項

### ⚠️ 重要限制

1. **套件版本限制**: 
   - **Python**: 3.9-3.11 (推薦 3.10)
   - **PyTorch**: 2.6.0 (CUDA 12.4)
   - **TensorFlow**: 2.14.0
   - **NumPy**: 1.26.4 (< 2.0，DICOM 壓縮功能尚未相容新版本)
   - **CUDA**: 12.4 (與 PyTorch 2.6.0 對應)

2. **GPU 記憶體需求**:
   - 動脈瘤模組: 至少 12GB，建議 16GB (3D Full Resolution)
   - 梗塞模組: 至少 8GB (2D 模型)
   - WMH 模組: 至少 8GB (2D 模型)

3. **系統資源需求**:
   - **RAM**: 推薦 64GB 以上
   - **CPU**: 推薦 16 核心以上
   - **儲存空間**: 建議 500GB 以上

4. **檔案格式**:
   - 輸入必須是 NIfTI 格式 (.nii 或 .nii.gz)
   - DICOM 目錄必須包含完整序列

5. **SynthSEG 選項**:
   - 梗塞和 WMH 模組可選擇提供或自動生成
   - 自動生成會增加 60-90 秒處理時間
   - 需要 TensorFlow 2.14.0

### 🔧 故障排除

#### GPU Out of Memory
```bash
# 檢查 GPU 使用狀況
nvidia-smi

# 確認沒有其他程式占用 GPU
# 程式會自動檢查 GPU 使用率 > 60% 時跳過執行
```

#### 前處理失敗
```bash
# 檢查 BET 腦部提取工具
which bet

# 如未安裝 FSL
# Ubuntu: sudo apt-get install fsl
# 或參考: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki
```

#### DICOM 壓縮失敗
```bash
# 確認已安裝 GDCM
sudo apt-get install libgdcm-tools

# 檢查 DICOM 檔案完整性
gdcminfo <dicom_file>
```

#### SynthSEG 執行錯誤
```bash
# 確認 SynthSEG 模型路徑正確
ls /data/4TB1/pipeline/chuan/code/model_weights/SynthSeg_parcellation_tf28/

# 檢查 TensorFlow 版本
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### 套件版本衝突
```bash
# 檢查 PyTorch 和 CUDA 版本
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 檢查 NumPy 版本
python -c "import numpy as np; print('NumPy:', np.__version__)"

# 如果版本不符，重新安裝
pip uninstall torch torchvision numpy -y
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.26.4
```

### 📊 效能優化建議

1. **GPU 使用率管理**:
   - 程式會自動檢查 GPU 使用率
   - 當使用率 > 60% 時會跳過執行

2. **多案例處理**:
   - 建議使用佇列系統避免 GPU 衝突
   - 可同時處理不同模組（動脈瘤 + 梗塞/WMH）

3. **暫存清理**:
   - 定期清理 `/data/4TB1/pipeline/chuan/process/` 以節省空間
   - 每個案例約佔用 2-5GB

4. **批次處理**:
   - 可使用 Shell 腳本批次執行多個案例
   - 建議每批不超過 5 個案例避免 GPU 記憶體不足

---

## Log 記錄

### Log 檔案位置
```
/data/4TB1/pipeline/chuan/log/<YYYYMMDD>.log
```

### Log 內容包含
- 執行開始/結束時間
- 各階段處理時間
- 錯誤訊息與堆疊追蹤
- GPU 記憶體使用狀況
- 前處理/後處理狀態

### Log 格式範例

#### 動脈瘤模組
```
2025-06-04 14:30:15 INFO !!! Pre_Aneurysm call.
2025-06-04 14:30:15 INFO 17390820_20250604_MR_21406040004 Start...
2025-06-04 14:30:15 INFO Running stage 1: Aneurysm inference!!!
2025-06-04 14:32:30 INFO [Done AI Inference... ] spend 135 sec
2025-06-04 14:32:50 INFO [Done reslice... ] spend 20 sec
2025-06-04 14:33:45 INFO [Done create_MIP_pred... ] spend 55 sec
2025-06-04 14:34:00 INFO [Done All Pipeline!!! ] spend 225 sec
```

#### 梗塞模組
```
2025-11-17 10:15:20 INFO !!! Pre_Infarct call.
2025-11-17 10:15:20 INFO 01550089_20251117_MR_21411150023 Start...
2025-11-17 10:15:25 INFO Running stage 1: Infarct inference!!!
2025-11-17 10:16:50 INFO [Done AI Inference... ] spend 85 sec
2025-11-17 10:17:10 INFO Running stage 2: Post-processing!!!
2025-11-17 10:17:35 INFO [Done Post-processing... ] spend 25 sec
```

#### WMH 模組
```
2025-11-17 10:20:15 INFO !!! Pre_WMH call.
2025-11-17 10:20:15 INFO 01550089_20251117_MR_21411150023 Start...
2025-11-17 10:20:20 INFO Running stage 1: WMH inference!!!
2025-11-17 10:21:10 INFO [Done AI Inference... ] spend 50 sec
2025-11-17 10:21:25 INFO Running stage 2: Post-processing!!!
2025-11-17 10:21:45 INFO [Done Post-processing... ] spend 20 sec
```

---

## 系統配置

### 共用路徑配置

可在各 pipeline 程式中修改以下路徑：

```python
# 程式碼路徑
path_code = '/data/4TB1/pipeline/chuan/code/'

# 處理暫存路徑
path_process = '/data/4TB1/pipeline/chuan/process/'

# JSON 輸出路徑
path_json = '/data/4TB1/pipeline/chuan/json/'

# Log 路徑
path_log = '/data/4TB1/pipeline/chuan/log/'

# GPU 編號
gpu_n = 0
```

### 模組特定路徑

#### 動脈瘤模組 (`pipeline_aneurysm_tensorflow.py`)
```python
path_processModel = os.path.join(path_process, 'Deep_Aneurysm')
path_nnunet_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset080_DeepAneurysm/nnUNetTrainer__nnUNetPlans__3d_fullres'
```

#### 梗塞模組 (`pipeline_infarct_torch.py`)
```python
path_processModel = os.path.join(path_process, 'Deep_Infarct')
path_nnunet_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset040_DeepInfarct/nnUNetTrainer__nnUNetPlans__2d'
```

#### WMH 模組 (`pipeline_wmh_torch.py`)
```python
path_processModel = os.path.join(path_process, 'Deep_WMH')
path_nnunet_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset015_DeepLacune/nnUNetTrainer__nnUNetPlans__2d'
```

---

## 授權與引用

本系統使用以下開源專案：
- **nnU-Net**: https://github.com/MIC-DKFZ/nnUNet
- **TensorFlow**: https://www.tensorflow.org/
- **PyTorch**: https://pytorch.org/
- **SynthSeg**: https://github.com/BBillot/SynthSeg
- **FSL**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki

---

## 版本歷史

### v2.1 (2025-11-27)
- ✨ 新增急性腦梗塞分析模組
- ✨ 新增白質病變評估模組
- 🔧 重構程式碼結構，模組化前處理/後處理
- 📝 更新文檔，整合三大模組說明

### v2.0 (2025-11-17)
- 🎉 初始版本 - 動脈瘤檢測模組
- 🔧 整合 nnU-Net 和 TensorFlow 模型
- 📊 支援 DICOM-SEG 輸出

---

## 聯絡資訊

如有問題或建議，請聯絡開發團隊。

**維護團隊**: TMU AI Medical Team  
**作者**: chuan  
**最後更新**: 2025-11-27