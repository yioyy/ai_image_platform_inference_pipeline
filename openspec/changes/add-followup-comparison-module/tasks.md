# 實作任務清單 - Follow-up 比較模組（檔案導向版本）

## 1. 專案結構建立
- [ ] 1.1 創建專案資料夾結構
  ```
  code_ai/followup/
  process/Followup/
  ```
- [ ] 1.2 創建 `pipeline_followup.py` 主腳本
- [ ] 1.3 創建 `pipeline_followup.sh` Shell 啟動腳本
- [ ] 1.4 創建 `requirements_followup.txt`
- [ ] 1.5 更新 README.md

## 2. 影像配準模組 (registration.py)
- [ ] 2.1 實作 SimpleITK 剛性配準函數
  ```python
  def register_followup_to_baseline(fixed_path, moving_path, output_path)
  ```
  - 使用 CenteredTransformInitializer
  - RegularStepGradientDescent 優化器
  - MattesMutualInformation 相似度指標
- [ ] 2.2 實作配準品質評估
  ```python
  def assess_registration_quality(fixed, moving, transform)
  ```
  - 計算 Mutual Information 分數
  - 計算 Correlation Coefficient
  - 返回品質報告字典
- [ ] 2.3 實作配準結果應用
  ```python
  def apply_transform_to_image(image_path, transform_path, output_path)
  ```
  - 將配準變換應用到影像
  - 重採樣到基準空間
- [ ] 2.4 實作棋盤格視覺化
  ```python
  def create_checkerboard_visualization(fixed, moving, output_path)
  ```
  - 產生配準品質檢查圖
- [ ] 2.5 錯誤處理
  - 配準失敗時的重試邏輯
  - 品質不佳時的警告

## 3. 病灶分析模組 (lesion_analysis.py)
- [ ] 3.1 實作連通元件提取
  ```python
  def extract_lesions_from_pred(pred_array, min_volume_mm3=3.0)
  ```
  - 使用 scipy.ndimage.label
  - 過濾小體積偽陽性
  - 計算每個病灶的屬性（位置、體積）
- [ ] 3.2 實作病灶配對演算法
  ```python
  def match_lesions(baseline_lesions, followup_lesions, distance_threshold=10.0)
  ```
  - 基於歐式距離配對
  - 基於體積相似度驗證
  - 計算配對信心度
- [ ] 3.3 實作新增病灶檢測
  ```python
  def detect_new_lesions(baseline_lesions, followup_lesions, matched_pairs)
  ```
  - 找出在 followup 中無匹配的病灶
  - 排除邊緣與小體積病灶
- [ ] 3.4 實作消失病灶檢測
  ```python
  def detect_disappeared_lesions(baseline_lesions, matched_pairs)
  ```
  - 找出在 baseline 中無匹配的病灶
- [ ] 3.5 實作體積變化計算
  ```python
  def calculate_volume_changes(matched_pairs)
  ```
  - 絕對變化（cm³）
  - 相對變化（%）
  - 顯著性判斷
- [ ] 3.6 實作病灶狀態分類
  ```python
  def classify_lesion_status(matched_pairs, new_lesions, disappeared_lesions)
  ```
  - stable / enlarged / reduced / new / disappeared
- [ ] 3.7 實作統計摘要
  ```python
  def compute_statistics(all_lesions)
  ```
  - 總體積變化
  - 各狀態病灶數量
  - 平均變化率

## 4. 視覺化模組 (followup_visualization.py)
- [ ] 4.1 實作並排影像顯示
  ```python
  def create_side_by_side_comparison(baseline_img, followup_img, output_path)
  ```
  - 選擇關鍵切片
  - 並排排列
  - 標註日期與 ID
- [ ] 4.2 實作差異熱力圖
  ```python
  def create_difference_heatmap(baseline_img, followup_img, output_path)
  ```
  - 計算減法影像
  - 熱力圖著色
- [ ] 4.3 實作病灶疊加圖
  ```python
  def create_lesion_overlay(image, pred, lesion_data, output_path)
  ```
  - 疊加病灶輪廓
  - 不同顏色表示不同狀態
  - 標註病灶 ID 與體積
- [ ] 4.4 實作體積變化圖表
  ```python
  def create_volume_chart(statistics, output_path)
  ```
  - 柱狀圖比較前後體積
  - 圓餅圖顯示狀態分布
- [ ] 4.5 批次產生所有視覺化
  ```python
  def generate_all_visualizations(data_dict, output_dir)
  ```

## 5. 報告生成模組 (followup_report.py)
- [ ] 5.1 設計 JSON 報告格式
  ```json
  {
    "comparison_id": "...",
    "baseline_id": "...",
    "followup_id": "...",
    "date_interval_days": 120,
    "registration_quality": 0.85,
    "statistics": {...},
    "lesions": [...]
  }
  ```
- [ ] 5.2 實作 JSON 報告生成
  ```python
  def generate_comparison_json(comparison_data, output_path)
  ```
- [ ] 5.3 實作自動結論生成
  ```python
  def generate_conclusion(statistics)
  ```
  - 基於變化程度的模板化結論
  - 臨床建議
- [ ] 5.4 實作報告驗證
  - 檢查必要欄位
  - 數值合理性驗證

## 6. DICOM 處理模組 (dicom_followup.py)
- [ ] 6.1 實作 DICOM-SEG 產生
  ```python
  def create_comparison_dicom_seg(
      baseline_dicom_dir,
      followup_dicom_dir,
      comparison_pred_path,
      lesion_data,
      output_path
  )
  ```
  - 參考 post_infarct.py 的 DICOM-SEG 生成邏輯
  - 使用不同 Segment 表示不同狀態病灶
- [ ] 6.2 實作 Orthanc 上傳
  ```python
  def upload_to_orthanc(dicom_path, orthanc_url)
  ```
  - POST DICOM 檔案
  - 驗證上傳成功
  - 重試機制

## 7. 工具函數 (utils_followup.py)
- [ ] 7.1 檔案驗證函數
  ```python
  def validate_input_files(baseline_inputs, followup_inputs)
  ```
  - 檢查檔案存在
  - 檢查檔案格式
  - 檢查影像尺寸相容性
- [ ] 7.2 資料夾創建函數
  ```python
  def create_comparison_folders(comparison_id, base_path)
  ```
  - 創建標準資料夾結構
- [ ] 7.3 檔案複製函數
  ```python
  def copy_input_files_to_workspace(inputs, dicom_dirs, workspace)
  ```
- [ ] 7.4 日期計算函數
  ```python
  def calculate_date_interval(baseline_dicom, followup_dicom)
  ```
  - 從 DICOM 標籤提取日期
  - 計算間隔天數

## 8. 主管道腳本 (pipeline_followup.py)
- [ ] 8.1 實作命令列參數解析
  ```python
  parser.add_argument('--baseline_ID', ...)
  parser.add_argument('--baseline_Inputs', nargs='+', ...)
  parser.add_argument('--baseline_DicomDir', nargs='+', ...)
  parser.add_argument('--followup_ID', ...)
  parser.add_argument('--followup_Inputs', nargs='+', ...)
  parser.add_argument('--followup_DicomDir', nargs='+', ...)
  parser.add_argument('--Output_folder', ...)
  parser.add_argument('--comparison_type', default='infarct', ...)
  ```
- [ ] 8.2 實作主管道函數
  ```python
  def pipeline_followup(
      baseline_ID, baseline_Inputs, baseline_DicomDir,
      followup_ID, followup_Inputs, followup_DicomDir,
      path_output, comparison_type, ...
  )
  ```
- [ ] 8.3 實作 Stage 1: 載入與驗證
  - 驗證輸入檔案
  - 創建處理資料夾
  - 複製輸入檔案
  - 日誌初始化
- [ ] 8.4 實作 Stage 2: 影像配準
  - 配準 followup 到 baseline
  - 配準品質評估
  - 配準視覺化
  - 應用配準到所有 followup 影像
- [ ] 8.5 實作 Stage 3: 病灶變化分析
  - 提取病灶
  - 病灶配對
  - 變化計算
  - 狀態分類
  - 統計摘要
- [ ] 8.6 實作 Stage 4: 結果輸出
  - 視覺化生成
  - JSON 報告
  - DICOM-SEG 生成
  - 上傳 Orthanc
  - 複製結果到輸出資料夾
- [ ] 8.7 錯誤處理
  - try-except 包裹各階段
  - 記錄詳細錯誤日誌
  - 清理臨時檔案
- [ ] 8.8 日誌記錄
  - 各階段開始/完成時間
  - 關鍵指標記錄
  - 錯誤與警告記錄

## 9. Shell 啟動腳本 (pipeline_followup.sh)
- [ ] 9.1 撰寫 Shell 腳本
  ```bash
  #!/bin/bash
  source /home/tmu/miniconda3/etc/profile.d/conda.sh
  conda activate followup_env
  python /data/4TB1/pipeline/chuan/code/pipeline_followup.py \
    --baseline_ID "$1" \
    --baseline_Inputs "$2" "$3" "$4" \
    --baseline_DicomDir "$5" "$6" "$7" \
    --followup_ID "$8" \
    --followup_Inputs "$9" "${10}" "${11}" \
    --followup_DicomDir "${12}" "${13}" "${14}" \
    --Output_folder "${15}"
  conda deactivate
  ```
- [ ] 9.2 設定執行權限
  ```bash
  chmod +x pipeline_followup.sh
  ```

## 10. 測試
- [ ] 10.1 單元測試
  - 測試配準函數（使用測試影像）
  - 測試病灶配對邏輯
  - 測試體積計算精度
  - 測試視覺化生成
  - 測試 JSON 格式正確性
- [ ] 10.2 整合測試
  - 準備測試資料（2 次真實檢查）
  - 執行完整管道
  - 驗證所有輸出檔案
  - 檢查 Orthanc 上傳
- [ ] 10.3 效能測試
  - 測量各階段處理時間
  - 監控記憶體使用
  - 測試不同影像大小
- [ ] 10.4 邊界案例測試
  - 配準失敗情況
  - 無病灶案例
  - 大量病灶案例（>20 個）
  - 完全不同的影像（測試錯誤處理）

## 11. 文檔撰寫
- [ ] 11.1 更新 README.md
  - Follow-up 模組介紹
  - 輸入參數說明
  - 使用範例
  - 輸出說明
- [ ] 11.2 撰寫函數文檔
  - 所有公開函數包含 Docstring
  - 參數與返回值說明
  - 使用範例
- [ ] 11.3 撰寫使用指南
  - 如何準備輸入資料
  - 如何執行管道
  - 如何解讀結果
  - 常見問題排解
- [ ] 11.4 撰寫開發文檔（選用）
  - 架構設計說明
  - 配準演算法說明
  - 病灶配對邏輯說明

## 12. 程式碼審查與優化
- [ ] 12.1 Linus 風格檢查
  - 確保縮排 ≤3 層
  - 使用 Early Return
  - 資料結構驅動設計
  - DICOM 標籤使用 `.get()` 安全存取
- [ ] 12.2 效能優化
  - 配準參數調優
  - 記憶體使用優化
  - I/O 操作優化
- [ ] 12.3 程式碼重構
  - 提取重複邏輯
  - 改進命名
  - 新增註解
- [ ] 12.4 錯誤處理完善
  - 所有外部調用包含錯誤處理
  - 有意義的錯誤訊息
  - 適當的日誌記錄

## 13. 部署準備
- [ ] 13.1 更新依賴列表
  ```txt
  SimpleITK>=2.2.0
  scipy>=1.10.0
  scikit-image>=0.20.0
  matplotlib>=3.7.0
  nibabel>=5.0.0
  numpy>=1.21.0
  pydicom>=2.3.0
  ```
- [ ] 13.2 創建安裝腳本
  ```bash
  pip install -r requirements_followup.txt
  ```
- [ ] 13.3 準備測試資料集
  - 至少 5 對測試案例
  - 涵蓋不同情況（穩定、進展、改善）
- [ ] 13.4 創建部署檢查清單
  - 環境需求
  - 資料夾權限
  - 配置設定

## 14. 試運行與調整
- [ ] 14.1 內部測試
  - 使用真實案例測試
  - 驗證結果正確性
- [ ] 14.2 收集反饋
  - 臨床醫師使用反饋
  - 技術團隊反饋
- [ ] 14.3 最終調整
  - 根據反饋調整參數
  - 改進視覺化
  - 完善文檔

## 進度追蹤
- **預計開始日期**: [待定]
- **預計完成日期**: [待定]（約 4-5 週）
- **當前狀態**: 規劃階段
- **完成比例**: 0/75 (0%)

## 里程碑
- [ ] **M1 - 核心模組完成** (Week 2): 配準、病灶分析
- [ ] **M2 - 輸出功能完成** (Week 3): 視覺化、報告、DICOM
- [ ] **M3 - 整合測試完成** (Week 4): 端到端可用
- [ ] **M4 - 臨床驗證完成** (Week 5): 準備上線

## 優先級標記
- 🔴 **P0 - 核心必須**: 配準、病灶分析、基本輸出
- 🟡 **P1 - 重要功能**: 視覺化、JSON 報告、DICOM-SEG
- 🟢 **P2 - 增強功能**: 進階視覺化、批次處理
- ⚪ **P3 - 未來功能**: PDF 報告、Web 界面、資料庫整合
