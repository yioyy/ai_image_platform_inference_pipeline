# 肺部結節檢測模組規格增量

## ADDED Requirements

### Requirement: CT 序列檢測
系統 MUST 自動識別胸腔 CT 影像序列。

#### Scenario: 標準胸腔 CT
- WHEN DICOM 標籤 Modality = "CT" AND BodyPartExamined = "CHEST"
- THEN 系統應識別為胸腔 CT 序列
- AND 標記為可進行結節檢測

#### Scenario: 非目標序列
- WHEN DICOM 序列不符合胸腔 CT 條件
- THEN 系統應跳過該序列
- AND 記錄跳過原因至日誌

### Requirement: 肺部結節檢測
系統 MUST 在 CT 影像中檢測並定位肺部結節。

#### Scenario: 單一結節檢測
- WHEN 輸入為標準胸腔 CT 序列
- AND 影像包含一個 ≥5mm 的結節
- THEN 系統應檢測到該結節
- AND 提供結節的 3D 中心座標
- AND 提供結節的邊界框（bounding box）

#### Scenario: 多發性結節
- WHEN 影像包含多個結節
- THEN 系統應分別檢測每個結節
- AND 為每個結節分配唯一 ID
- AND 依照體積大小排序結果

#### Scenario: 無結節影像
- WHEN 影像不包含可檢測結節
- THEN 系統應返回空結果列表
- AND 記錄 "未檢測到結節" 狀態

### Requirement: 結節測量
系統 SHALL 準確測量檢測到的結節特徵。

#### Scenario: 體積測量
- WHEN 系統檢測到結節
- THEN 系統應計算結節體積（cm³）
- AND 體積測量誤差應 <10%（相較於手動測量）

#### Scenario: 直徑測量
- WHEN 系統檢測到結節
- THEN 系統應測量結節最大直徑（mm）
- AND 提供長軸、短軸測量值
- AND 直徑測量誤差應 <2mm

#### Scenario: 密度分析
- WHEN 系統檢測到結節
- THEN 系統應計算結節平均 HU 值
- AND 計算 HU 值標準差
- AND 記錄最小/最大 HU 值

### Requirement: 結節分類
系統 SHOULD 根據密度特徵分類結節類型。

#### Scenario: 實質性結節
- WHEN 結節平均 HU 值 >-300
- THEN 系統應分類為 "實質性結節"
- AND 標記為高風險類型

#### Scenario: 磨玻璃樣結節
- WHEN 結節平均 HU 值在 -700 到 -300 之間
- THEN 系統應分類為 "磨玻璃樣結節"
- AND 標記為中等風險類型

#### Scenario: 混合型結節
- WHEN 結節同時包含實質性與磨玻璃樣成分
- AND HU 值標準差 >200
- THEN 系統應分類為 "混合型結節"
- AND 計算實質性成分比例

### Requirement: DICOM-SEG 產生
系統 MUST 產生符合 DICOM 標準的分割結果。

#### Scenario: 單一結節 SEG
- WHEN 檢測到一個結節
- THEN 系統應產生 DICOM-SEG 檔案
- AND SEG 檔案包含結節分割遮罩
- AND 設定正確的 Segment 屬性（名稱、類別、顏色）
- AND 參照原始 CT 序列 UID

#### Scenario: 多結節 SEG
- WHEN 檢測到多個結節
- THEN 系統應在同一 SEG 檔案中包含所有結節
- AND 每個結節對應一個 Segment
- AND Segment 編號按結節大小排序

#### Scenario: SEG 元資料
- WHEN 產生 DICOM-SEG
- THEN 系統應包含以下元資料：
  - Algorithm Name: "Lung Nodule Detection AI"
  - Model Version: 當前模型版本號
  - Processing Time: 實際處理時間
  - Nodule Count: 檢測到的結節數量

### Requirement: 結果儲存
系統 MUST 將檢測結果儲存至資料庫。

#### Scenario: 成功檢測儲存
- WHEN 結節檢測完成
- THEN 系統應將結果寫入 `lung_nodule_results` 資料表
- AND 包含以下欄位：
  - patient_id
  - study_instance_uid
  - series_instance_uid
  - nodule_count
  - nodule_details（JSON 格式）
  - processing_time
  - model_version

#### Scenario: JSON 結節詳情格式
- WHEN 儲存結節詳情
- THEN JSON 格式應為：
  ```json
  [
    {
      "nodule_id": 1,
      "location": [x, y, z],
      "volume_cm3": 1.2,
      "diameter_mm": 8.5,
      "type": "實質性結節",
      "mean_hu": 25.3,
      "confidence": 0.92
    }
  ]
  ```

### Requirement: Orthanc 整合
系統 MUST 將結果上傳至 Orthanc PACS 系統。

#### Scenario: DICOM-SEG 上傳
- WHEN DICOM-SEG 產生完成
- THEN 系統應透過 Orthanc REST API 上傳
- AND 驗證上傳成功（HTTP 200）
- AND 記錄 Orthanc Instance UID

#### Scenario: 上傳失敗重試
- WHEN 上傳失敗
- THEN 系統應重試最多 3 次
- AND 每次重試間隔 5 秒
- AND 記錄失敗原因
- AND 最終失敗時通知管理員

### Requirement: 效能標準
系統 MUST 滿足以下效能要求。

#### Scenario: 處理時間限制
- WHEN 處理標準 CT 序列（512x512x300 切片）
- THEN 總處理時間應 <5 分鐘
- AND 前處理時間 <1 分鐘
- AND AI 推理時間 <3 分鐘
- AND 後處理時間 <1 分鐘

#### Scenario: 記憶體使用限制
- WHEN 執行結節檢測
- THEN GPU 記憶體使用應 <8GB
- AND CPU 記憶體使用應 <16GB
- AND 能同時處理 2 個序列（批次模式）

### Requirement: 品質控制
系統 SHOULD 執行自動品質檢查。

#### Scenario: 影像品質驗證
- WHEN 載入 CT 影像
- THEN 系統應檢查：
  - 切片數量 ≥50
  - HU 值範圍合理（-1024 到 3071）
  - 無明顯偽影
- AND 品質不合格時發出警告

#### Scenario: 模型信心度過濾
- WHEN 檢測到結節
- THEN 系統應計算檢測信心度
- AND 僅報告信心度 >0.70 的結節
- AND 記錄低信心度檢測（0.50-0.70）供審查

### Requirement: 臨床驗證標準
系統 MUST 達到以下臨床驗證指標。

#### Scenario: Dice 係數要求
- WHEN 與專家標註比較
- THEN 平均 Dice 係數應 >0.80
- AND 針對 >10mm 結節 Dice 應 >0.85

#### Scenario: 敏感度要求
- WHEN 評估檢測敏感度
- THEN 對 ≥5mm 結節敏感度應 >0.85
- AND 對 ≥10mm 結節敏感度應 >0.95

#### Scenario: 偽陽性率控制
- WHEN 評估偽陽性率
- THEN 每個掃描平均偽陽性應 <0.3 個
- AND 大部分偽陽性應為 <5mm 的小結構

### Requirement: 錯誤處理
系統 MUST 妥善處理各種錯誤情況。

#### Scenario: DICOM 讀取失敗
- WHEN DICOM 檔案損壞或無法讀取
- THEN 系統應記錄詳細錯誤訊息
- AND 繼續處理其他序列
- AND 標記該序列為失敗狀態

#### Scenario: GPU 記憶體不足
- WHEN GPU 記憶體不足
- THEN 系統應自動降低批次大小
- AND 重試推理
- AND 記錄警告訊息

#### Scenario: 模型載入失敗
- WHEN 模型檔案不存在或損壞
- THEN 系統應立即終止並報告錯誤
- AND 提供模型檔案路徑資訊
- AND 建議解決方案

## 非功能性需求

### Requirement: 程式碼品質
程式碼 MUST 遵循專案編碼規範。

#### Scenario: Linus 風格檢查
- 縮排層數 ≤3
- 使用 Early Return 避免深度嵌套
- 用資料結構取代複雜 if/else 鏈
- DICOM 標籤使用 `.get()` 安全存取

#### Scenario: 文檔要求
- 所有公開函數包含 Docstring
- 複雜演算法包含註解說明
- README 包含使用範例

### Requirement: 可維護性
程式碼 SHOULD 易於維護與擴展。

#### Scenario: 模組化設計
- 前處理、推理、後處理分離
- 工具函數提取至獨立模組
- 配置參數外部化

#### Scenario: 測試覆蓋率
- 核心函數包含單元測試
- 整合測試涵蓋主要流程
- 測試資料集包含邊界案例

