# 新增前後比較 Follow-up 模組

## 變更概述
開發檔案導向的前後檢查比較管道，類似 `pipeline_infarct_torch.py` 的結構，支援影像配準、病灶變化追蹤與定量分析。

## 問題陳述

### 臨床背景
在臨床實務中，醫師需要比較患者在不同時間點的影像檢查，以評估：
- **疾病進展**: 腦梗塞、白質病變等的擴大或縮小
- **治療效果**: 介入治療後的改善程度
- **新發病灶**: 是否出現新的病灶
- **長期追蹤**: 慢性疾病的演變趨勢

目前系統的限制：
1. **無配準功能**: 前後影像可能有位置、角度差異，無法直接比較
2. **無定量比較**: 需要手動計算體積變化、新增病灶數量
3. **視覺化不足**: 缺乏並排比較、差異圖等視覺工具
4. **報告生成困難**: 需要手動整理比較結果
5. **無整合管道**: 需要獨立的前後比較工具

### 技術挑戰
1. **影像配準**: 處理不同掃描參數、患者姿勢差異
2. **病灶追蹤**: 正確配對前後檢查中的相同病灶
3. **變化檢測**: 區分真實變化與掃描差異
4. **效能考量**: 配準計算密集，需要優化
5. **檔案管理**: 妥善組織輸入輸出檔案結構

### 使用者需求
- 提供兩次檢查的資料即可執行比較
- 一鍵執行完整的前後比較分析
- 清晰的視覺化比較結果
- 定量報告（體積變化百分比、新增病灶等）
- 輸出 JSON 和 DICOM 格式結果

## 解決方案

### 技術架構（類似 pipeline_infarct_torch.py）

#### 輸入參數設計
```python
def pipeline_followup(
    baseline_ID,           # 基準檢查 ID
    baseline_Inputs,       # [Pred.nii.gz, ADC.nii.gz, DWI0.nii.gz, DWI1000.nii.gz]
    baseline_DicomDir,     # [ADC_dicom_dir, DWI0_dicom_dir, DWI1000_dicom_dir]
    followup_ID,           # 追蹤檢查 ID
    followup_Inputs,       # [Pred.nii.gz, ADC.nii.gz, DWI0.nii.gz, DWI1000.nii.gz]
    followup_DicomDir,     # [ADC_dicom_dir, DWI0_dicom_dir, DWI1000_dicom_dir]
    path_output,           # 輸出資料夾
    comparison_type='infarct',  # 'infarct', 'wmh', 'aneurysm'
    path_code='/data/4TB1/pipeline/chuan/code/',
    path_process='/data/4TB1/pipeline/chuan/process/Followup/',
    path_json='/data/4TB1/pipeline/chuan/json/',
    path_log='/data/4TB1/pipeline/chuan/log/',
    gpu_n=0
)
```

#### 處理流程（四階段）

##### Stage 1: 載入與驗證
- 載入兩次檢查的 Pred.nii.gz（分割結果）
- 載入對應的原始影像（ADC, DWI0, DWI1000）
- 驗證檔案完整性
- 創建處理資料夾結構

##### Stage 2: 影像配準
使用 **SimpleITK** 進行配準：
- **剛性配準**: 對齊兩次檢查的影像空間
- **配準品質評估**: Mutual Information 分數
- **配準結果應用**: 將 followup 影像配準到 baseline 空間
- **配準視覺化**: 產生檢查圖示配準品質

##### Stage 3: 病灶變化分析
- **病灶配對**: 基於空間距離與體積相似度配對前後病灶
- **新增病灶檢測**: 在 baseline 中不存在的病灶
- **消失病灶檢測**: 在 followup 中不再出現的病灶
- **體積變化計算**: 精確計算每個病灶的增減
- **狀態分類**: stable/enlarged/reduced/new/disappeared

##### Stage 4: 結果輸出
- **視覺化影像**: 並排比較圖、差異熱力圖
- **JSON 報告**: 包含定量統計與病灶追蹤資訊
- **DICOM-SEG**: 產生變化標記的 DICOM 分割檔
- **上傳 Orthanc**: 將結果上傳至 PACS

### 檔案結構設計

```
process/Followup/{comparison_ID}/
├── baseline/
│   ├── Pred.nii.gz                    # 基準分割結果
│   ├── ADC.nii.gz                     # 基準 ADC 影像
│   ├── DWI0.nii.gz                    # 基準 DWI0 影像
│   └── DWI1000.nii.gz                 # 基準 DWI1000 影像
├── followup/
│   ├── Pred.nii.gz                    # 追蹤分割結果
│   ├── Pred_registered.nii.gz         # 配準後分割結果
│   ├── ADC.nii.gz                     # 追蹤 ADC 影像
│   ├── ADC_registered.nii.gz          # 配準後 ADC 影像
│   ├── DWI0.nii.gz
│   ├── DWI0_registered.nii.gz
│   ├── DWI1000.nii.gz
│   └── DWI1000_registered.nii.gz
├── registration/
│   ├── transform.tfm                  # 配準變換參數
│   ├── registration_quality.txt       # 配準品質報告
│   └── checkerboard.png               # 棋盤格檢查圖
├── analysis/
│   ├── lesion_matching.json           # 病灶配對結果
│   ├── change_analysis.json           # 變化分析結果
│   └── statistics.json                # 統計摘要
├── visualization/
│   ├── side_by_side.png               # 並排比較圖
│   ├── difference_map.png             # 差異熱力圖
│   ├── overlay_baseline.png           # 基準疊加圖
│   ├── overlay_followup.png           # 追蹤疊加圖
│   └── volume_chart.png               # 體積變化圖表
└── dicom/
    ├── baseline/                      # 基準 DICOM 檔案
    ├── followup/                      # 追蹤 DICOM 檔案
    └── comparison_seg.dcm             # 比較結果 DICOM-SEG

output/{comparison_ID}/
├── {comparison_ID}_comparison.json    # 主要輸出 JSON
├── {comparison_ID}_report.pdf         # 報告（選用）
└── visualization/                     # 複製視覺化圖片
```

### 核心模組設計

#### 1. 影像配準模組 (registration.py)
```python
def register_followup_to_baseline(
    fixed_image_path,    # baseline 影像
    moving_image_path,   # followup 影像
    output_path,         # 配準結果輸出路徑
    transform_path       # 變換參數輸出路徑
) -> dict:
    """使用 SimpleITK 執行配準"""
    - 剛性配準（6 自由度）
    - 返回配準品質分數
    - 保存配準變換參數
```

#### 2. 病灶分析模組 (lesion_analysis.py)
```python
def analyze_lesion_changes(
    baseline_pred_path,   # baseline Pred.nii.gz
    followup_pred_path,   # followup Pred.nii.gz（已配準）
    voxel_spacing         # 體素間距
) -> dict:
    """分析病灶變化"""
    - 提取連通元件
    - 計算病灶配對
    - 計算體積變化
    - 分類病灶狀態
```

#### 3. 視覺化模組 (followup_visualization.py)
```python
def create_comparison_visualizations(
    baseline_image,
    followup_image,
    baseline_pred,
    followup_pred,
    output_dir
) -> list:
    """產生比較視覺化"""
    - 並排比較圖
    - 差異熱力圖
    - 疊加標記圖
    - 統計圖表
```

#### 4. 報告生成模組 (followup_report.py)
```python
def generate_followup_json(
    comparison_data,
    baseline_ID,
    followup_ID,
    output_path
) -> str:
    """產生 JSON 報告"""
    - 基本資訊（IDs、日期、間隔）
    - 配準品質
    - 病灶統計
    - 詳細變化列表
```

## 影響範圍

### 新增檔案
```
pipeline_followup.py            # 主管道腳本（類似 pipeline_infarct_torch.py）
pipeline_followup.sh            # Shell 啟動腳本（類似 pipeline_aneurysm.sh）

code_ai/followup/
├── __init__.py
├── registration.py             # 影像配準（SimpleITK）
├── lesion_analysis.py          # 病灶變化分析
├── followup_visualization.py   # 視覺化工具
├── followup_report.py          # JSON 報告生成
├── dicom_followup.py           # DICOM-SEG 生成
└── utils_followup.py           # 工具函數

requirements_followup.txt       # 新增依賴
```

### 修改檔案
- `README.md` - 新增 follow-up 模組使用說明
- 無需修改現有管道程式碼（完全獨立）

### 不影響的部分
- 現有推理管道（infarct, wmh, aneurysm）完全獨立運作
- 單次檢查分析流程不變
- 可重用部分工具函數（如 DICOM 處理、視覺化基礎）

## 驗收標準

### 功能需求
- [ ] 接受兩組檢查資料作為輸入
- [ ] 配準準確度：Mutual Information >0.7
- [ ] 病灶配對準確率 >85%
- [ ] 體積測量誤差 <10%
- [ ] 正確檢測新增/消失病灶
- [ ] 產生完整的 JSON 報告
- [ ] 產生 DICOM-SEG 檔案
- [ ] 成功上傳至 Orthanc
- [ ] 支援三種分析類型（infarct, wmh, aneurysm）

### 效能需求
- [ ] 配準時間 <2 分鐘（標準腦部 MRI）
- [ ] 完整比較分析 <5 分鐘
- [ ] 記憶體使用 <4GB

### 視覺化品質
- [ ] 並排影像清晰對齊
- [ ] 差異圖清楚標示變化區域
- [ ] 圖表美觀、資訊豐富
- [ ] 產生至少 4 種視覺化圖片

### 臨床驗證（選用）
- [ ] 測試 10-20 個真實案例
- [ ] 病灶配對準確性驗證
- [ ] 體積測量與手動測量比較
- [ ] 臨床實用性評估

### 程式碼品質
- [ ] 遵循 Linus 風格（縮排 ≤3 層）
- [ ] 核心函數包含單元測試
- [ ] 完整使用說明文檔
- [ ] 錯誤處理健全

## 時程估計

### Phase 1: 核心功能開發 (2 週)
- Week 1: 
  - 影像配準模組（SimpleITK 整合）
  - 檔案結構設計與處理邏輯
- Week 2:
  - 病灶變化分析演算法
  - 病灶配對與狀態分類

### Phase 2: 輸出功能 (1 週)
- Week 3:
  - 視覺化工具開發
  - JSON 報告生成
  - DICOM-SEG 產生

### Phase 3: 整合與測試 (1 週)
- Week 4:
  - 主管道腳本整合
  - 單元與整合測試
  - 文檔撰寫

### Phase 4: 臨床驗證 (選用，1 週)
- Week 5:
  - 臨床案例測試
  - 效能優化
  - 最終調整

**總計**: 約 4-5 週（1 個月）

## 風險與緩解

### 技術風險

#### 1. 配準失敗
**風險**: 前後影像差異過大導致配準失敗
**緩解**: 
- 實作多階段配準（粗配準 → 精配準）
- 提供手動調整選項
- 設定配準品質閾值，低於閾值時警告

#### 2. 病灶配對錯誤
**風險**: 將不同病灶誤判為同一病灶
**緩解**:
- 使用多重特徵匹配（位置 + 大小 + 形狀）
- 設定嚴格的配對閾值
- 提供人工審核界面

#### 3. 效能瓶頸
**風險**: 配準計算耗時過長
**緩解**:
- 實作多解析度配準
- GPU 加速（如可用）
- 快速預覽模式
- 背景批次處理

### 臨床風險

#### 1. 假陽性變化
**風險**: 掃描參數差異被誤判為病灶變化
**緩解**:
- 正規化影像強度
- 考慮掃描協議差異
- 設定變化顯著性閾值

#### 2. 遺漏真實變化
**風險**: 細微但重要的變化未被檢測
**緩解**:
- 調整敏感度參數
- 多層次分析（粗略 + 精細）
- 提供完整原始資料供醫師審查

#### 3. 報告誤導
**風險**: 自動報告可能被過度信任
**緩解**:
- 明確標示為「輔助工具」
- 包含信心度指標
- 要求醫師最終審核

## 擴展計畫

### 短期擴展（3-6 個月）
- 支援更多影像模態（CT 腦出血、WMH）
- 多時間點比較（>2 個時間點）
- PDF 報告生成
- 批次處理模式

### 中期擴展（6-12 個月）
- 自動顯著變化警示
- 與資料庫整合（記錄比較歷史）
- Web 界面查看結果
- 治療效果統計分析

### 長期願景（1-2 年）
- AI 預測疾病進展趨勢
- 跨時間點智能追蹤
- 個人化追蹤建議

## 依賴項目

### 新增 Python 套件
```txt
SimpleITK>=2.2.0          # 影像配準
scipy>=1.10.0             # 科學計算（連通元件、統計）
scikit-image>=0.20.0      # 影像處理
matplotlib>=3.7.0         # 圖表繪製
```

### 現有依賴（可重用）
- nibabel（已有）- NIfTI 檔案讀寫
- numpy, pandas（已有）- 數值計算與資料處理
- pydicom（已有）- DICOM 處理
- 現有的視覺化工具（post_infarct.py, post_wmh.py）

## 成功指標

### 定量指標
- 每日使用次數 >20
- 配準成功率 >95%
- 病灶配對準確率 >90%
- 報告生成成功率 >98%
- 醫師滿意度 >4.0/5.0

### 定性指標
- 減少醫師比較影像時間 >50%
- 提升變化檢測敏感度
- 改善臨床決策信心
- 獲得臨床團隊正面反饋

## 參考資料

### 學術文獻
- ANTs (Advanced Normalization Tools) - 配準標準工具
- SimpleITK Registration Examples
- Longitudinal Brain MRI Analysis Methods

### 類似產品
- Siemens syngo.via - 商業影像後處理平台
- BrainVoyager - 神經影像分析軟體
- FreeSurfer Longitudinal Stream

### DICOM 標準
- DICOM SR (Structured Report) - PS3.3
- DICOM Registration Object - PS3.3
- Spatial Registration Storage SOP Class

## 備註

此模組為臨床關鍵功能，需要：
1. 與臨床醫師密切合作
2. 充分的臨床驗證
3. 完整的錯誤處理
4. 清晰的使用文檔
5. 持續的效能監控

