# 專案概述

## 專案名稱
醫學影像 AI 分析平台

## 專案描述
一個用於處理醫學影像（CT、MRI）的 AI 分析管道系統，支援：
- 腦梗塞（Infarct）檢測與分割
- 白質病變（WMH）檢測與分割
- 動脈瘤（Aneurysm）檢測與追蹤
- DICOM 影像處理與轉換
- 與 Orthanc PACS 系統整合

## 技術堆疊

### 核心框架
- **Python**: 主要開發語言（3.8+）
- **PyTorch**: 深度學習推理（腦梗塞、白質病變）
- **TensorFlow**: 深度學習推理（動脈瘤）
- **nnU-Net**: 醫學影像分割框架

### 醫學影像處理
- **MONAI**: 醫學影像 AI 框架
- **SimpleITK / ITK**: 影像配準與轉換
- **pydicom**: DICOM 檔案處理
- **nibabel**: NIfTI 檔案處理
- **SynthSeg**: 腦部分割工具

### 資料庫與儲存
- **MySQL**: 關聯式資料庫
- **SQLAlchemy**: ORM 框架
- **Orthanc**: DICOM PACS 系統

### API 與服務
- **FastAPI**: Web API 框架（如有使用）
- **RESTful API**: 與 Orthanc 通訊

## 架構模式

### 管道架構
每個分析任務（infarct, wmh, aneurysm）都遵循以下管道：
1. **前處理（pre_*.py）**: DICOM 下載、轉換為 NIfTI
2. **GPU 推理（gpu_*.py / pipeline_*.py）**: AI 模型推理
3. **後處理（post_*.py / after_run_*.py）**: 結果處理、DICOM-SEG 生成、上傳

### 目錄結構
```
code/
├── code_ai/                 # AI 核心功能模組
│   └── pipeline/           # 管道相關工具
├── model_weights/          # 訓練好的模型權重
├── nnUNet/                 # nnU-Net 相關檔案
└── *.py                    # 各種管道腳本
```

## 編碼慣例

### 命名規範
- **檔案名稱**: snake_case（如 `pipeline_wmh_torch.py`）
- **類別名稱**: PascalCase（如 `InfarctPipeline`）
- **函數名稱**: snake_case（如 `process_dicom_series`）
- **常數**: UPPER_SNAKE_CASE（如 `MAX_SLICES`）

### 程式碼風格
- 遵循 **Linus Torvalds 風格**優先原則（見 `.cursor/rules/linus.mdc`）
- 縮排不超過 3 層
- 使用 Early Return 避免深度嵌套
- 用資料結構取代複雜的 if/else 鏈
- 對核心算法模組可補充 Donald Knuth 風格文檔

### 醫學影像特定慣例
- DICOM 標籤存取使用 `.get()` 確保安全
- 所有 T1/T2 序列處理必須檢測 REFORMATTED 狀態
- 維持向後相容性，不破壞現有 API
- 複雜度檢查：避免超過 3 層縮排

## 關鍵需求

### 效能需求
- GPU 加速推理
- 批次處理支援
- 記憶體管理（大型 3D 影像）

### 臨床需求
- 符合 DICOM 標準
- 患者隱私保護（HIPAA / GDPR）
- 可追溯性與稽核日誌
- 準確性與臨床驗證

### 整合需求
- 與 Orthanc PACS 無縫整合
- 支援多種影像模態（CT、MRI）
- 資料庫狀態管理
- 錯誤處理與重試機制

## 開發工作流程

1. **新功能開發**: 使用 OpenSpec 創建變更提案
2. **規格審查**: 驗證規格符合臨床與技術需求
3. **實作**: 遵循現有架構與編碼慣例
4. **測試**: 單元測試 + 臨床驗證資料集
5. **歸檔**: 完成後將變更歸檔至 OpenSpec

## 品質標準

### 程式碼品質
- [ ] 通過 Linus 風格檢查（縮排 ≤ 3 層）
- [ ] 無深度嵌套邏輯
- [ ] 使用資料結構驅動設計
- [ ] 適當的錯誤處理

### 醫學 AI 品質
- [ ] DICOM 標籤安全存取
- [ ] REFORMATTED 序列檢測
- [ ] 臨床指標驗證（Dice, Sensitivity, Specificity）
- [ ] 偏差檢測與公平性

## 相關文件
- [README.md](../README.md) - 完整專案說明
- [Git分支管理完整指南.md](../../Git分支管理完整指南.md) - Git 工作流程
- `.cursor/rules/linus.mdc` - Linus 風格規範

