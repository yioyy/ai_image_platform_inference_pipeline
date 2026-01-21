# Follow-up 比較模組變更提案

## 變更概述
開發檔案導向的前後檢查比較管道，類似 `pipeline_infarct_torch.py` 的架構，用於追蹤患者病情變化。

## 📁 變更結構

```
add-followup-comparison-module/
├── README.md           # 本檔案
├── proposal.md         # 詳細變更提案
├── tasks.md            # 75 項實作任務（簡化版）
└── specs/
    └── followup/
        └── spec.md     # 完整規格增量
```

## 🎯 核心功能

### 1. 輸入資料處理
- 接受兩組檢查資料（baseline 與 followup）
- 檔案完整性驗證
- 創建標準處理資料夾

### 2. 精確影像配準
- 多解析度剛性配準
- 配準品質自動評估
- 快速預覽模式

### 3. 病灶變化追蹤
- 自動病灶配對
- 新增/消失病灶檢測
- 體積變化精確計算
- 五種狀態分類（穩定、增大、縮小、新增、消失）

### 4. 豐富視覺化
- 並排影像對比
- 熱力圖差異顯示
- 3D 病灶渲染
- 統計圖表

### 5. 專業報告生成
- 文字報告（Markdown/HTML）
- PDF 報告
- DICOM-SR 結構化報告
- 自動結論與建議

## 📁 檔案結構設計

### 處理資料夾結構
```
process/Followup/{comparison_ID}/
├── baseline/            # 基準檢查資料
│   ├── Pred.nii.gz
│   ├── ADC.nii.gz
│   ├── DWI0.nii.gz
│   └── DWI1000.nii.gz
├── followup/            # 追蹤檢查資料
│   ├── Pred.nii.gz
│   ├── Pred_registered.nii.gz
│   ├── ADC_registered.nii.gz
│   └── ...
├── registration/        # 配準結果
│   ├── transform.tfm
│   └── registration_quality.txt
├── analysis/            # 分析結果
│   ├── lesion_matching.json
│   └── statistics.json
├── visualization/       # 視覺化圖片
│   ├── side_by_side.png
│   ├── difference_map.png
│   └── volume_chart.png
└── dicom/               # DICOM 檔案
    ├── baseline/
    ├── followup/
    └── comparison_seg.dcm
```

## 🏗️ 系統架構（檔案導向）

```
pipeline_followup.py (主腳本)
    ↓
Stage 1: 載入與驗證
    ├── 驗證輸入檔案
    ├── 創建處理資料夾
    └── 複製輸入檔案
    ↓
Stage 2: 影像配準 (registration.py)
    ├── SimpleITK 配準
    ├── 配準品質評估
    └── 應用配準變換
    ↓
Stage 3: 病灶變化分析 (lesion_analysis.py)
    ├── 病灶提取與配對
    ├── 變化計算
    └── 狀態分類
    ↓
Stage 4: 結果輸出
    ├── 視覺化 (followup_visualization.py)
    ├── JSON 報告 (followup_report.py)
    ├── DICOM-SEG (dicom_followup.py)
    └── 上傳 Orthanc
```

## 📈 實作進度

### 階段 1: 核心模組開發 (Week 1-2)
- [ ] 影像配準模組
- [ ] 病灶分析模組
- [ ] 檔案處理邏輯

### 階段 2: 輸出功能 (Week 3)
- [ ] 視覺化工具
- [ ] JSON 報告生成
- [ ] DICOM-SEG 生成

### 階段 3: 整合與測試 (Week 4)
- [ ] 主管道腳本整合
- [ ] 單元與整合測試
- [ ] 文檔撰寫

### 階段 4: 臨床驗證 (Week 5，選用)
- [ ] 真實案例測試
- [ ] 效能優化
- [ ] 最終調整

## 🎓 技術亮點

### 先進的配準技術
- 使用 **SimpleITK** 醫學影像配準庫
- 多解析度金字塔策略
- 自動配準品質評估（Mutual Information）

### 智能病灶追蹤
- 多特徵配對（空間距離 + 體積 + 形狀）
- 信心度量化
- 邊界案例特殊處理

### 臨床導向設計
- 符合臨床工作流程
- 清晰的視覺化展示
- 專業的報告輸出
- DICOM 標準合規

## 📝 關鍵檔案說明

### proposal.md
**內容**:
- 臨床背景與需求分析
- 技術架構設計
- 資料庫 Schema 設計
- 風險評估與緩解
- 時程估計（8 週）
- 驗收標準

**適合閱讀對象**: 所有團隊成員、專案經理、臨床顧問

### tasks.md
**內容**:
- 96 項詳細任務
- 16 個主要分類
- 4 個里程碑
- 優先級標記（P0-P3）

**適合閱讀對象**: 開發人員、技術主管

### specs/followup/spec.md
**內容**:
- 45+ 個功能需求
- 100+ 個場景定義
- 使用 MUST/SHALL/SHOULD 語法
- WHEN/THEN 條件描述

**適合閱讀對象**: 開發人員、測試人員、品質保證團隊

## 🚀 如何開始實作

### Step 1: 審查提案
```bash
# 查看完整提案
openspec show add-followup-comparison-module

# 或直接閱讀
cat openspec/changes/add-followup-comparison-module/proposal.md
```

### Step 2: 驗證規格
```bash
# 驗證規格格式
openspec validate add-followup-comparison-module
```

### Step 3: 準備環境
```bash
# 安裝依賴
pip install SimpleITK>=2.2.0 scipy>=1.10.0 scikit-image>=0.20.0

# 執行資料庫遷移
mysql -u user -p database < database_migration_followup.sql
```

### Step 4: 開始實作
```bash
# 使用 Cursor slash command
/openspec:apply add-followup-comparison-module

# 或自然語言
# "請開始實作 follow-up 比較模組"
```

## 🎯 驗收標準

### 功能驗收
- [ ] 接受兩組檢查資料並成功執行
- [ ] 配準 MI 分數 >0.7
- [ ] 病灶配對準確率 >85%
- [ ] 體積測量誤差 <10%
- [ ] 產生完整 JSON 報告
- [ ] 產生 DICOM-SEG 檔案
- [ ] 成功上傳至 Orthanc

### 效能驗收
- [ ] 配準時間 <2 分鐘
- [ ] 完整分析 <5 分鐘
- [ ] 記憶體使用 <4GB

### 臨床驗收（選用）
- [ ] 10-20 個真實案例測試
- [ ] 病灶配對準確性驗證
- [ ] 臨床實用性評估

### 程式碼品質
- [ ] Linus 風格檢查通過
- [ ] 核心函數包含單元測試
- [ ] 完整使用文檔

## 💡 設計決策

### 為什麼選擇 SimpleITK？
- 專為醫學影像設計
- 成熟穩定的配準演算法
- 支援多種配準方法
- 與 ITK 生態系統整合

### 為什麼使用多解析度配準？
- 提升配準速度
- 避免局部最優
- 提高配準成功率

### 為什麼設計兩個資料表？
- 分離主記錄與詳細追蹤
- 方便查詢與分析
- 支援複雜的病灶關聯查詢

### 為什麼支援多種報告格式？
- 文字報告：快速審閱
- PDF：正式報告、存檔
- DICOM-SR：PACS 整合、標準化

## 🔗 相關資源

### 內部文檔
- [專案規範](../../project.md)
- [Linus 風格指南](../../../.cursor/rules/linus.mdc)
- [AGENTS 指南](../../../AGENTS.md)

### 學術參考
- **ANTs**: Advanced Normalization Tools
- **SimpleITK Examples**: https://simpleitk.readthedocs.io/
- **DICOM SR**: DICOM PS3.3 Structured Reporting

### 類似系統
- Siemens syngo.via
- FreeSurfer Longitudinal Stream
- BrainVoyager

## 📞 聯絡資訊

### 專案負責人
[待指派]

### 技術顧問
[待指派]

### 臨床顧問
[待指派]

## 📅 時間軸

| 週次 | 階段 | 里程碑 |
|------|------|--------|
| Week 1-2 | 核心模組開發 | M1: 配準、分析模組完成 |
| Week 3 | 輸出功能開發 | M2: 視覺化、報告、DICOM |
| Week 4 | 整合與測試 | M3: 端到端可用 |
| Week 5 | 臨床驗證（選用）| M4: 準備上線 |

## ✅ 下一步行動

1. **團隊會議**: 討論提案，確認需求與時程
2. **資源分配**: 指派開發人員與時間
3. **環境準備**: 設定開發與測試環境
4. **開始實作**: 執行 `/openspec:apply add-followup-comparison-module`

---

**狀態**: 🟢 已重新規劃（檔案導向版本）  
**創建日期**: 2025-11-28  
**更新日期**: 2025-11-28  
**預計開始**: [待定]  
**預計完成**: [待定] (+4-5 週)

## 🔄 與 pipeline_infarct_torch.py 的對比

| 特性 | pipeline_infarct_torch.py | pipeline_followup.py |
|------|--------------------------|----------------------|
| 輸入 | 單次檢查（ADC, DWI0, DWI1000） | 兩次檢查（各含 Pred + 原始影像） |
| 處理階段 | 前處理 → AI推理 → 後處理 → after_run | 載入驗證 → 配準 → 分析 → 輸出 |
| 核心運算 | nnU-Net 推理 | SimpleITK 配準 + 病灶配對 |
| 輸出 | Pred.nii.gz, JSON, DICOM-SEG | 比較 JSON, 視覺化圖片, DICOM-SEG |
| 資料庫 | 無（檔案導向）| 無（檔案導向）|
| 執行方式 | Shell 腳本 + Python | Shell 腳本 + Python |

