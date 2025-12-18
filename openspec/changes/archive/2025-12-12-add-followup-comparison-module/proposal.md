# 新增前後比較 Follow-up 模組（檔案導向）

## 變更概述
新增一個**獨立**的「前後檢查比較」管道（Follow-up），以檔案輸入/輸出為主，流程與 `pipeline_infarct_torch.py` 類似，提供：
- **影像配準（baseline 空間為主）**
- **病灶配對與變化分類**
- **輸出 Follow-up JSON 報告**（延用既有平台 JSON 結構並新增 Follow-up 欄位）

## 問題陳述（Why）

### 臨床背景
臨床需要比較患者不同時間點影像，以評估：
- **疾病進展**：動脈瘤、CMB、腦梗塞、WMH 等病灶擴大/縮小
- **治療效果**：介入或藥物治療前後差異
- **新發病灶**：是否出現新病灶
- **長期追蹤**：慢性疾病演變趨勢

### 目前限制
1. **缺少配準**：前後影像有姿勢/角度差異，難以直接對照
2. **缺少定量比較**：新增/消失/變化需要人工整理
3. **缺少可機器讀取的報告**：平台需要結構化 JSON
4. **缺少一鍵整合管道**：需靠外部工具拼湊流程

### 技術挑戰
1. **配準穩定性**：不同掃描參數、病人姿勢差異
2. **病灶配對**：同一病灶前後對應、避免誤配
3. **差異判讀**：區分真實變化與掃描/配準誤差
4. **效能**：配準耗時、I/O 多
5. **檔案治理**：輸入輸出路徑一致、可追蹤

## 目標與非目標（What）

### 目標（Goals）
- **輸入兩次檢查**（baseline / followup）的 NIfTI 與 JSON，即可完成比較
- 產出：
  - **`Pred_<model>_followup.nii.gz`**（合併標註：new/stable/disappeared）
- **`Followup_<model>_platform_json.json`**（含 followup 的 study/series UID 與病灶配對結果）
- 支援 `model`：`Infarct`、`WMH`、`Aneurysm`、`CMB`（先以 CMB 的 JSON/流程作為基準）

### 非目標（Non-goals）
- 不在此變更中新增 UI（並排視覺化、差異圖等）
- 不在此變更中改動既有單次推論管道（infarct/wmh/aneurysm/cmb）
- DICOM-SEG 的**重新生成**（WMH/Infarct）先列為後續擴充；本變更先做到**複製既有 dicom-seg**

## 解決方案（How）

### 介面設計（主函數）
```python
def pipeline_followup(
    baseline_ID: str,
    baseline_Inputs: list[str],        # [Pred.nii.gz, SynthSEG.nii.gz]
    baseline_DicomDir: str,            # 未來 WMH/Infarct 產生新 dicom-seg 可能需要；本版可選
    baseline_DicomSegDir: str,         # baseline dicom-seg 來源資料夾
    baseline_json: str,                # baseline 平台 JSON
    followup_ID: str,
    followup_Inputs: list[str],        # [Pred.nii.gz, SynthSEG.nii.gz]
    followup_DicomSegDir: str,         # followup dicom-seg 來源資料夾
    followup_json: str,                # followup 平台 JSON
    path_output: str,                  # 最終輸出資料夾（回傳給平台/使用者）
    model: str = "CMB",                # 'infarct' | 'wmh' | 'aneurysm' | 'cmb'
    path_code: str = "/data/4TB1/pipeline/chuan/code/",
    path_process: str = "/data/4TB1/pipeline/chuan/process/Followup/",
    path_json: str = "/data/4TB1/pipeline/chuan/json/",
    path_log: str = "/data/4TB1/pipeline/chuan/log/",
    gpu_n: int = 0,
):
    ...
```

### 處理流程（Stage-based）

#### Stage 1：載入與驗證
- 載入 baseline / followup：
  - `Pred_<model>.nii.gz`（病灶標註）
  - `SynthSEG_<model>.nii.gz`（用於配準的 mask/參考）
  - `Pred_<model>_platform_json.json`（平台 JSON）
- 驗證檔案存在、格式可讀、shape 相容
- 建立處理資料夾（以 baseline 為主資料夾）

#### Stage 2：影像配準（Follow-up → Baseline）
- 使用 **FSL FLIRT**：以 SynthSEG 作為配準影像
- 產出：
  - `SynthSEG_<model>_registration.nii.gz`
  - `Pred_<model>_registration.nii.gz`
  - `<baseline_id>_<model>.mat`

#### Stage 3：病灶配對與變化分類
- 以 baseline 的 label index 為主鍵
- 使用 **重疊配對** 將 followup 的 label（registration 後）對應回 baseline
- 產出 `Pred_<model>_followup.nii.gz`（合併標註）
  - **1 = new**
  - **2 = stable（或重疊）**
  - **3 = disappeared**

#### Stage 4：報告輸出（JSON）
- 以 baseline JSON 結構為底，加入 followup 的 UID 與配對欄位
- 狀態分類（固定字串）：
  - `new` / `stable` / `enlarging` / `shrinking` / `disappeared`
- 產出 `Followup_<model>_platform_json.json`


### 檔案結構（Process 與 Output）

```
process/Deep_FollowUp/{baseline_id}/
├── baseline/
│   ├── Pred_<model>.nii.gz              # 病灶分割結果（注意：<model> 會保留大小寫，例如 CMB）
│   ├── SynthSEG_<model>.nii.gz          # 對位用腦區分割圖
│   ├── Pred_<model>_platform_json.json  # AI結果的json，用於提供followup模組的study與病灶資訊
│   └── Dicom-Seg/                     # baseline的dicom-seg資料夾
├── followup/
│   ├── Pred_<model>.nii.gz              # 病灶分割結果
│   ├── SynthSEG_<model>.nii.gz          # 對位用腦區分割圖
│   ├── Pred_<model>_platform_json.json  # AI結果的json，用於提供followup模組的study與病灶資訊
│   └── Dicom-Seg/                     # followup的dicom-seg資料夾
├── registration/
│   ├── Pred_<model>_registration.nii.gz          # 病灶分割結果，followup pred對位到baseline上
│   ├── SynthSEG_<model>_registration.nii.gz      # 腦區分割圖，followup pred對位到baseline上
│   └── <baseline_id>_<model>.mat                # 對位的矩陣檔
├── result/
│   ├── Pred_<model>_followup.nii.gz            # 把病灶對位的結果合併，定義新標籤 1.new 2.stable 3.disappeared 
│   └── Followup_<model>_platform_json.json     # 病灶對位後重新產生的json（以 baseline JSON 為底，不重新計算）
└── dicom/
    ├── baseline/model/                      # baseline的DICOM 檔案，DWI跟WMH產生新dicom-seg時需要，Aneurysm跟CMB不用   
    └── Dicom-Seg/                     # 全部的dicom-seg檔

#傳回松諭資料夾
output/{baseline_id}/
├── Followup_<model>_platform_json.json    # 主要輸出 JSON
├── Pred_<model>_followup.nii.gz   # 把病灶對位的結果合併，定義新標籤 1.new 2.stable 3.disappeared 
└── Dicom-Seg/                     # 全部的dicom-seg檔
```

### 核心模組（預計新增檔案）

#### 1) `registration.py`
- **責任**：執行 followup → baseline 的配準，並輸出 `.mat` 與 registration 後的影像
- **策略**：以 SynthSEG 作為配準影像；Pred 用 nearest-neighbour 套用矩陣

#### 2) `generate_followup_pred.py`
- **責任**：合併 baseline 與 followup(registration) 的 Pred，輸出 `Pred_<model>_followup.nii.gz`
- **輸出標籤**：1=new、2=stable、3=disappeared

#### 3) `followup_report.py`
- **責任**：產生 `Followup_<model>_platform_json.json`（以 baseline JSON 為底，追加 followup 欄位與病灶配對欄位）
- **注意**：本版本不重新計算既有統計，僅做「配對、欄位補齊、狀態分類」

#### 4) 上傳（先實作於主腳本中）
- **責任**：沿用 `after_run_infarct.py` 的上傳邏輯，完成 JSON 上傳與 dicom-seg 上傳（本版以複製/轉送為主）

## 影響範圍

### 新增檔案
```
pipeline_followup.py            # 主管道腳本（類似 pipeline_infarct_torch.py）
pipeline_followup.sh            # Shell 啟動腳本（類似 pipeline_aneurysm.sh）
generate_followup_pred.py       # 把病灶對位的結果合併，定義新標籤 1.new 2.stable 3.disappeared 
followup_report.py              # 產生 JSON 報告，可能產生新的dicom-seg檔

code_ai/ #不確定需要額外產生哪些檔案

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
- [ ] 接受兩組檢查資料作為輸入（baseline / followup）
- [ ] 配準成功並輸出 `.mat` 與 `*_registration.nii.gz`
- [ ] 產生 `Pred_<model>_followup.nii.gz`
- [ ] 產生 `Followup_<model>_platform_json.json`
- [ ] JSON 中包含 followup 的 `study_instance_uid` 與 `series_instance_uid`（欄位新增）
- [ ] 病灶狀態至少涵蓋：`new`、`stable`、`enlarging`、`shrinking`、`disappeared`
- [ ] 成功完成 JSON 上傳流程（對齊 `after_run_infarct.py` 行為）

### 效能需求
- [ ] 配準時間 <2 分鐘（標準腦部 MRI）
- [ ] 完整比較分析 <5 分鐘
- [ ] 記憶體使用 < 32GB

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

### 建議時程（可依資源調整）
- **Phase 1（1 天）**：配準 + 檔案結構 + pred 合併 + 基礎狀態分類
- **Phase 2（1 天）**：JSON 報告（UID/欄位補齊 + 對位後 slice 對應）+ 上傳整合
- **Phase 3（1 天）**：測試案例整理 + 整合測試 + 文件補齊
- **Phase 4（選用，1 週）**：臨床驗證與參數調校

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

## 擴展計畫

### 短期擴展（3-6 個月）
- 支援更多影像模態（CT 腦出血、WMH）
- 多時間點比較（>2 個時間點）
- 批次處理模式

### 中期擴展（6-12 個月）
- 自動顯著變化警示
- 與資料庫整合（記錄比較歷史）
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

