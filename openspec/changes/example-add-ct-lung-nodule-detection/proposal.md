# 新增 CT 肺部結節檢測功能

## 變更概述
新增 AI 驅動的 CT 影像肺部結節檢測與分析功能。

## 問題陳述

### 臨床背景
肺部結節是肺癌早期篩檢的重要指標。目前系統僅支援腦部影像分析（腦梗塞、白質病變、動脈瘤），缺乏胸腔 CT 分析能力。臨床醫師需要：
- 自動檢測肺部結節位置
- 測量結節大小（直徑、體積）
- 評估結節特性（實質性、磨玻璃樣、混合型）
- 追蹤結節隨時間的變化

### 技術挑戰
1. **DICOM 處理**: 胸腔 CT 序列與腦部 MRI 有不同的標籤結構
2. **演算法選擇**: 需要適合小物體檢測的 AI 模型
3. **效能需求**: CT 切片數量多（200-500 張），需要高效推理
4. **整合複雜度**: 需與現有 Orthanc 管道整合

## 解決方案

### 技術架構
1. **模型選擇**: 使用 nnU-Net 變體進行結節分割
   - 預訓練權重: LUNA16 / LNDb 資料集
   - 3D U-Net 架構，適合小物體檢測

2. **管道設計**: 遵循現有三階段模式
   ```
   前處理 → GPU 推理 → 後處理
   (pre_lung.py) → (gpu_lung.py) → (post_lung.py)
   ```

3. **DICOM 整合**:
   - 檢測 CT Chest 序列
   - 支援標準 Hounsfield 單位（HU）範圍
   - 產生 DICOM-SEG 結果

### 資料庫設計
新增資料表：`lung_nodule_results`
```sql
CREATE TABLE lung_nodule_results (
    id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id VARCHAR(50),
    study_instance_uid VARCHAR(100),
    series_instance_uid VARCHAR(100),
    nodule_count INT,
    nodule_details JSON,  -- [{"位置": [x,y,z], "體積": 120, "類型": "實質性"}]
    processing_time FLOAT,
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 影響範圍

### 新增檔案
- `pre_lung.py` - 胸腔 CT DICOM 前處理
- `gpu_lung.py` - 結節檢測推理管道
- `post_lung.py` - 結果後處理與 DICOM-SEG 產生
- `pipeline_lung.py` - 整合管道腳本
- `model_weights/lung_nodule/` - 模型權重目錄

### 修改檔案
- `README.md` - 新增肺部結節檢測說明
- `database schema` - 新增資料表
- `code_ai/pipeline/` - 可能新增共用工具函數

### 不影響的部分
- 現有腦部影像管道（infarct, wmh, aneurysm）
- Orthanc 整合邏輯（可重用）
- 資料庫連線與基礎設施

## 驗收標準

### 功能需求
- [ ] 成功檢測 ≥5mm 的肺部結節
- [ ] 準確測量結節體積（誤差 <10%）
- [ ] 正確分類結節類型（實質性、磨玻璃樣、混合型）
- [ ] 產生符合 DICOM 標準的 SEG 檔案
- [ ] 結果上傳至 Orthanc PACS

### 效能需求
- [ ] 單一 CT 序列處理時間 <5 分鐘（GPU: RTX 3090）
- [ ] GPU 記憶體使用 <8GB
- [ ] 支援批次處理（多個 series）

### 臨床驗證
- [ ] Dice 係數 >0.80（與專家標註比較）
- [ ] 敏感度（Sensitivity）>0.85
- [ ] 偽陽性率（FPR）<0.3 per scan

### 程式碼品質
- [ ] 通過 Linus 風格檢查（縮排 ≤3 層）
- [ ] 包含單元測試
- [ ] DICOM 標籤安全存取（使用 `.get()`）
- [ ] 完整錯誤處理與日誌

## 時程估計
- **前處理開發**: 2 天
- **AI 模型整合**: 3 天
- **後處理與 DICOM**: 2 天
- **測試與驗證**: 3 天
- **文檔撰寫**: 1 天
- **總計**: 約 11 個工作天

## 風險與緩解

### 技術風險
1. **模型效能不足**: 
   - 緩解: 準備多個預訓練模型備選
   
2. **記憶體限制**: 
   - 緩解: 實作切片批次處理，降低記憶體峰值

3. **DICOM 相容性**: 
   - 緩解: 參考現有管道，重用驗證邏輯

### 臨床風險
1. **偽陰性（漏檢）**: 
   - 緩解: 設定保守閾值，偏向多檢測
   
2. **偽陽性過多**: 
   - 緩解: 加入後處理過濾小體積偽陽性

## 後續擴展
- 結節追蹤（比對前後檢查）
- 惡性風險評分（Lung-RADS）
- 支援低劑量 CT
- 整合臨床報告生成

## 參考資料
- LUNA16 Challenge: https://luna16.grand-challenge.org/
- nnU-Net Paper: https://arxiv.org/abs/1904.08128
- DICOM Segmentation Standard: DICOM PS3.3

