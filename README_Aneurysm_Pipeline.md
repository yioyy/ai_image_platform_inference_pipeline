# 腦動脈瘤 AI 預測分析系統

## 專案簡介

本系統是一個完整的腦動脈瘤（Aneurysm）AI 預測分析流程，使用深度學習模型對腦部 MRA (Magnetic Resonance Angiography) 影像進行自動化分析，包含動脈瘤檢測、血管分割、影像重建、數據統計等功能。

### 主要功能

- **動脈瘤檢測**: 使用 nnU-Net 深度學習模型自動檢測腦動脈瘤
- **血管分割**: 對腦血管進行 16 區域分割標記
- **MIP 影像生成**: 自動生成 Maximum Intensity Projection (MIP) 影像
- **量化分析**: 計算動脈瘤大小、位置、體積等臨床參數
- **DICOM-SEG 生成**: 產生標準 DICOM 分割格式
- **結果上傳**: 自動上傳至 Orthanc PACS 和 AI 平台

## 系統架構

```
pipeline_aneurysm.sh (Shell 腳本)
    ↓ 啟動 Conda 環境
pipeline_aneurysm_tensorflow.py (主程式)
    ↓ 呼叫
gpu_aneurysm.py (GPU 推論模組)
    ↓ 使用
nnU-Net Model + TensorFlow Models
    ↓ 產生
預測結果 + DICOM-SEG + JSON + Excel 報告
```

## 系統需求

### 硬體需求
- **GPU**: NVIDIA GPU (建議 16GB 以上顯存)
- **記憶體**: 建議 32GB 以上
- **儲存空間**: 每個案例約需 2-5GB 暫存空間

### 軟體需求
- **作業系統**: Linux (Ubuntu 18.04+)
- **Python**: 3.9+
- **Conda/Miniconda**: 用於環境管理
- **CUDA**: 11.x 以上
- **GDCM**: 用於 DICOM 影像處理

### Python 套件依賴
```
tensorflow >= 2.14
numpy < 2.0  # 注意版本限制
nibabel
pydicom
scikit-image
opencv-python
pandas
matplotlib
pynvml
SimpleITK
```

## 安裝說明

### 1. 安裝 Conda 環境

```bash
# 初始化 conda
source /home/tmu/miniconda3/etc/profile.d/conda.sh

# 建立環境
conda create -n tf_2_14 python=3.9

# 啟動環境
conda activate tf_2_14
```

### 2. 安裝 Python 套件

```bash
pip install tensorflow==2.14.0
pip install nibabel pydicom scikit-image opencv-python
pip install pandas matplotlib pynvml SimpleITK
pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
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
- nnU-Net 模型: `/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset080_DeepAneurysm/`
- TensorFlow 模型: `/data/4TB1/pipeline/chuan/code/model_weights/`

## 使用方法

### 方式一：使用 Shell 腳本（推薦）

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

### 方式二：直接執行 Python 程式

```bash
# 啟動環境
conda activate tf_2_14

# 執行程式
python pipeline_aneurysm_tensorflow.py \
    --ID "17390820_20250604_MR_21406040004" \
    --Inputs "/data/input/MRA_BRAIN.nii.gz" \
    --DicomDir "/data/inputDicom/MRA_BRAIN/" \
    --Output_folder "/data/output/"

# 停用環境
conda deactivate
```

## 參數說明

### Shell 腳本參數 (pipeline_aneurysm.sh)

| 位置 | 參數名稱 | 說明 | 範例 |
|------|---------|------|------|
| $1 | Patient_ID | 病患 ID 或研究 ID | `17390820_20250604_MR_21406040004` |
| $2 | MRA_Input | MRA 腦部影像 NIfTI 路徑 | `/data/input/MRA_BRAIN.nii.gz` |
| $3 | DICOM_Dir | 原始 DICOM 影像目錄 | `/data/inputDicom/MRA_BRAIN/` |
| $4 | Output_Dir | 輸出結果資料夾 | `/data/output/` |

### Python 程式參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--ID` | str | 必填 | 病患識別碼 |
| `--Inputs` | list[str] | 必填 | 輸入 NIfTI 檔案路徑列表 |
| `--DicomDir` | list[str] | 必填 | DICOM 影像目錄列表 |
| `--Output_folder` | str | 必填 | 輸出資料夾路徑 |

## 輸出說明

### 輸出檔案結構

```
<Output_Folder>/
├── Pred_Aneurysm.nii.gz              # 動脈瘤預測遮罩
├── Prob_Aneurysm.nii.gz              # 動脈瘤機率圖
├── Pred_Aneurysm_Vessel.nii.gz       # 血管分割結果
├── Pred_Aneurysm_Vessel16.nii.gz     # 16 區域血管分割
├── Pred_Aneurysm.json                # 預測結果 JSON
└── Pred_Aneurysm_platform_json.json  # 平台格式 JSON
```

### 暫存處理目錄

在處理過程中，會在以下路徑產生暫存檔案：

```
/data/4TB1/pipeline/chuan/process/Deep_Aneurysm/<Patient_ID>/
├── MRA_BRAIN.nii.gz           # 輸入影像副本
├── Vessel.nii.gz              # 血管分割結果
├── Vessel_16.nii.gz           # 16 區域血管分割
├── nnUNet/                    # nnU-Net 模型結果
│   ├── Pred.nii.gz            # 預測遮罩
│   ├── Prob.nii.gz            # 機率圖
│   ├── Dicom/                 # DICOM 輸出
│   │   ├── MRA_BRAIN/         # MRA 影像
│   │   ├── MIP_Pitch/         # Pitch MIP
│   │   ├── MIP_Yaw/           # Yaw MIP
│   │   └── Dicom-Seg/         # DICOM-SEG 分割
│   ├── Image_nii/             # NIfTI 影像
│   ├── Image_reslice/         # 重切影像
│   ├── excel/                 # Excel 分析報告
│   │   └── <ID>_measurements.xlsx
│   ├── JSON/                  # JSON 結果
│   │   └── <ID>_platform_json.json
│   └── Dicom_zip/             # 壓縮上傳檔案
└── tensorflow/                # TensorFlow 模型結果 (已停用)
```

### JSON 格式說明

輸出的 JSON 檔案包含以下資訊：
- 動脈瘤數量與位置
- 每個動脈瘤的尺寸（長軸、短軸、體積）
- 所在血管區域
- DICOM Series UID
- 模型版本資訊

## 處理流程

### 完整流程圖

```
1. 環境初始化
   ├── 檢查 GPU 記憶體使用率
   └── 設定 TensorFlow GPU 配置

2. 影像前處理
   ├── 複製輸入 MRA 影像
   └── 建立工作目錄

3. AI 推論 (gpu_aneurysm.py)
   ├── nnU-Net 動脈瘤偵測
   ├── 血管分割 (16 區域)
   └── 產生機率圖

4. 影像重建
   ├── 影像 Reslice
   ├── MIP 影像生成 (Pitch/Yaw)
   └── DICOM 解壓縮

5. 數據分析
   ├── 動脈瘤量測
   ├── 計算體積與尺寸
   └── 產生 Excel 報告

6. 格式轉換
   ├── 產生 DICOM-SEG
   └── 生成平台 JSON

7. 結果上傳
   ├── 上傳至 Orthanc PACS
   ├── 上傳 JSON 至 AI 平台
   └── 複製結果到輸出目錄

8. 清理與記錄
   ├── 寫入執行 Log
   └── 計算總執行時間
```

### 各階段時間估算

| 階段 | 預估時間 | GPU 使用 |
|------|---------|---------|
| AI 推論 | 60-120 秒 | 高 |
| 影像 Reslice | 10-20 秒 | 低 |
| MIP 生成 | 30-60 秒 | 中 |
| 數據分析 | 5-10 秒 | 低 |
| DICOM-SEG | 10-20 秒 | 低 |
| 上傳處理 | 10-30 秒 | 無 |
| **總計** | **2-4 分鐘** | - |

## 模型說明

### nnU-Net 模型
- **用途**: 動脈瘤主要檢測模型
- **架構**: nnU-Net 3D Full Resolution
- **訓練資料**: Dataset080_DeepAneurysm
- **輸出**: 二元分割遮罩 + 機率圖
- **Group ID**: 56

### 血管分割模型
- **用途**: 腦血管 16 區域分割
- **架構**: ResUNet
- **輸出**: 16 類別血管標籤

## 注意事項

### ⚠️ 重要限制

1. **NumPy 版本限制**: 目前程式需要 NumPy < 2.0，因為 DICOM 壓縮功能尚未相容新版本
2. **GPU 記憶體**: 至少需要 12GB 顯存，建議 16GB 以上
3. **檔案格式**: 輸入必須是 NIfTI 格式 (.nii 或 .nii.gz)
4. **DICOM 必要性**: 需要提供原始 DICOM 目錄以產生 DICOM-SEG

### 🔧 故障排除

#### GPU Out of Memory
```bash
# 減少批次大小或使用較小的影像切片
# 調整 gpu_aneurysm.py 中的參數
```

#### DICOM 壓縮失敗
```bash
# 確認已安裝 GDCM
sudo apt-get install libgdcm-tools

# 檢查 DICOM 檔案完整性
gdcminfo <dicom_file>
```

#### 環境啟動失敗
```bash
# 重新初始化 conda
source /home/tmu/miniconda3/etc/profile.d/conda.sh

# 確認環境存在
conda env list
```

### 📊 效能優化建議

1. **GPU 使用率管理**: 程式會檢查 GPU 使用率，當使用率 > 60% 時會跳過執行
2. **多案例處理**: 建議使用佇列系統避免 GPU 衝突
3. **暫存清理**: 定期清理處理目錄以節省空間

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

### Log 格式
```
2025-06-04 14:30:15 INFO !!! Pre_Aneurysm call.
2025-06-04 14:30:15 INFO 17390820_20250604_MR_21406040004 Start...
2025-06-04 14:30:15 INFO Running stage 1: Aneurysm inference!!!
2025-06-04 14:32:30 INFO [Done AI Inference... ] spend 135 sec
2025-06-04 14:32:50 INFO [Done reslice... ] spend 20 sec
2025-06-04 14:33:45 INFO [Done create_MIP_pred... ] spend 55 sec
2025-06-04 14:34:00 INFO [Done All Pipeline!!! ] spend 225 sec
```

## 系統配置

### 路徑配置
在 `pipeline_aneurysm_tensorflow.py` 中可修改以下路徑：

```python
# 程式碼路徑
path_code = '/data/4TB1/pipeline/chuan/code/'

# 處理暫存路徑
path_process = '/data/4TB1/pipeline/chuan/process/'

# nnU-Net 模型路徑
path_nnunet_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/...'

# JSON 輸出路徑
path_json = '/data/4TB1/pipeline/chuan/json/'

# Log 路徑
path_log = '/data/4TB1/pipeline/chuan/log/'

# GPU 編號
gpu_n = 0
```

## API 整合（已註解，可啟用）

程式碼中預留了 Radax 系統整合的部分，若需啟用：

1. 取消註解 Line 410-486
2. 修改 API 端點 URL
3. 確認 DICOM-SEG 檔案命名規則

```python
# API 端點範例
api_url = 'http://localhost:4000/v1/ai-inference/inference-complete'

# POST 資料格式
payload = {
    "studyInstanceUid": study_instance_uid,
    "seriesInstanceUid": series_instance_uid
}
```

## 授權與引用

本系統使用以下開源專案：
- nnU-Net: https://github.com/MIC-DKFZ/nnUNet
- TensorFlow: https://www.tensorflow.org/
- MONAI: https://monai.io/

## 聯絡資訊

如有問題或建議，請聯絡開發團隊。

---

**版本**: 2.0  
**最後更新**: 2025-01-13  
**作者**: chuan  
**維護團隊**: TMU AI Medical Team

