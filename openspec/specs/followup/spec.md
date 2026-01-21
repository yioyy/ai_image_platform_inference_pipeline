# followup Specification

## Purpose
TBD - created by archiving change add-followup-comparison-module. Update Purpose after archive.
## Requirements
### Requirement: 輸入資料驗證
系統 MUST 驗證輸入資料的完整性與格式正確性。

#### Scenario: 檔案存在性檢查
- WHEN 提供輸入檔案路徑
- THEN 系統應檢查所有檔案是否存在
- AND 檔案不存在時返回清晰錯誤訊息
- AND 列出缺少的檔案

#### Scenario: 檔案格式驗證
- WHEN 載入 NIfTI 檔案
- THEN 系統應驗證檔案為有效的 .nii.gz 格式
- AND 檔案可被 nibabel 正確讀取
- AND 包含必要的 header 資訊

#### Scenario: 影像尺寸檢查
- WHEN 載入 baseline 與 followup 影像
- THEN 系統應檢查影像維度是否相容
- AND 體素間距（voxel spacing）是否合理
- AND 影像方向（orientation）是否一致

#### Scenario: DICOM 資料夾驗證
- WHEN 提供 DICOM 資料夾路徑
- THEN 系統應檢查資料夾存在且包含 DICOM 檔案
- AND 能夠讀取至少一個 DICOM 檔案
- AND 提取 StudyDate 等必要標籤

### Requirement: 影像配準
系統 MUST 執行精確的影像配準以對齊前後檢查。

#### Scenario: 剛性配準
- WHEN 執行標準配準
- THEN 系統應使用剛性配準（Rigid Registration）
- AND 配準應處理平移、旋轉
- AND 不改變影像形狀或大小

#### Scenario: 多解析度配準
- WHEN 執行配準
- THEN 系統應使用多解析度策略
- AND 先在低解析度粗配準
- AND 再在原始解析度精配準
- AND 至少使用 3 個金字塔層級

#### Scenario: 配準品質評估
- WHEN 配準完成
- THEN 系統應計算 Mutual Information (MI) 分數
- AND MI 分數應 >0.8 才視為成功
- AND MI 分數 <0.6 應標記為配準失敗
- AND 分數在 0.6-0.8 間應發出警告

#### Scenario: 配準失敗處理
- WHEN 配準 MI 分數 <0.6
- THEN 系統應嘗試不同初始化參數重新配準
- AND 最多重試 2 次
- AND 仍失敗時提供手動調整選項
- AND 記錄失敗原因至日誌

#### Scenario: 快速配準模式
- WHEN 使用者選擇快速預覽模式
- THEN 系統應使用降低解析度配準
- AND 降低迭代次數至 50（標準為 200）
- AND 配準時間應 <30 秒
- AND 標記結果為「預覽品質」

### Requirement: 病灶配對追蹤
系統 MUST 正確配對前後檢查中的相同病灶。

#### Scenario: 基於距離的配對
- WHEN 在配準後的影像中比對病灶
- THEN 系統應計算病灶中心點的歐式距離
- AND 距離 <10mm 的病灶視為候選配對
- AND 若多個候選，選擇距離最近的

#### Scenario: 基於體積的配對驗證
- WHEN 初步配對完成
- THEN 系統應驗證配對病灶的體積相似度
- AND 體積比 (smaller/larger) 應 >0.5
- AND 體積比 <0.5 應視為不同病灶

#### Scenario: 配對信心度
- WHEN 完成病灶配對
- THEN 系統應計算配對信心度分數
- AND 信心度基於：距離（50%）+ 體積相似度（30%）+ 形狀相似度（20%）
- AND 信心度 >0.8 為「高信心」
- AND 信心度 0.6-0.8 為「中等信心」
- AND 信心度 <0.6 應標記為「不確定配對」

#### Scenario: 無法配對的病灶
- WHEN baseline 中的病灶在 followup 中找不到匹配
- THEN 系統應標記為「可能消失」
- AND 檢查是否在配準範圍外
- AND 檢查是否因體積過小被過濾

### Requirement: 新增病灶檢測
系統 MUST 檢測在追蹤檢查中新出現的病灶。

#### Scenario: 標準新增病灶
- WHEN followup 檢查中有病灶
- AND 該病灶在 baseline 中不存在（無匹配）
- AND 病灶體積 ≥3mm³
- THEN 系統應標記為「新增病灶」
- AND 記錄病灶位置與體積

#### Scenario: 小體積病灶過濾
- WHEN 檢測到體積 <3mm³ 的未匹配病灶
- THEN 系統應標記為「小體積，可能為偽陽性」
- AND 不計入新增病灶統計
- AND 但保留於詳細報告中供審查

#### Scenario: 邊緣病灶處理
- WHEN 病灶位於影像邊緣（距離邊界 <5mm）
- AND 無法確定是否在 baseline 範圍內
- THEN 系統應標記為「邊緣病灶，不確定」
- AND 建議人工審查

#### Scenario: 掃描範圍差異
- WHEN baseline 與 followup 的掃描範圍不同
- THEN 系統應標示掃描範圍差異區域
- AND 該區域的新增病灶標記為「範圍外」
- AND 不計入真正的新增病灶

### Requirement: 消失病灶檢測
系統 SHALL 檢測在追蹤檢查中消失的病灶。

#### Scenario: 標準消失病灶
- WHEN baseline 檢查中有病灶
- AND 該病灶在 followup 中不存在（無匹配）
- AND 配準品質良好（MI >0.8）
- THEN 系統應標記為「消失病灶」
- AND 記錄原病灶位置與體積

#### Scenario: 配準品質不佳時
- WHEN 配準品質不佳（MI <0.7）
- AND 檢測到消失病灶
- THEN 系統應標記為「疑似消失，配準品質不佳」
- AND 建議重新配準或人工審查

#### Scenario: 體積縮小至閾值以下
- WHEN baseline 病灶體積 >3mm³
- AND followup 對應位置體積 <3mm³
- THEN 系統應標記為「顯著縮小」而非「消失」
- AND 記錄殘餘體積

### Requirement: 體積變化計算
系統 MUST 準確計算病灶的體積變化。

#### Scenario: 絕對體積變化
- WHEN 計算配對病灶的體積變化
- THEN 系統應計算：Volume_change = Followup_vol - Baseline_vol
- AND 以 cm³ 為單位
- AND 精度至小數點後 2 位

#### Scenario: 相對體積變化
- WHEN 計算相對變化
- THEN 系統應計算：Percent_change = (Volume_change / Baseline_vol) × 100%
- AND 精度至小數點後 1 位（如 +15.3%）
- AND 正值表示增大，負值表示縮小

#### Scenario: 顯著性判斷
- WHEN 評估體積變化的顯著性
- THEN 系統應使用以下閾值：
  - 變化 <10%：標記為「穩定」
  - 變化 10-30%：標記為「輕度變化」
  - 變化 30-50%：標記為「中度變化」
  - 變化 >50%：標記為「顯著變化」

#### Scenario: 測量誤差考量
- WHEN 體積變化 <5%
- THEN 系統應註明「接近測量誤差範圍」
- AND 建議謹慎解讀

#### Scenario: 總體積統計
- WHEN 計算所有病灶的總體積變化
- THEN 系統應分別計算：
  - 穩定病灶總體積變化
  - 新增病灶總體積
  - 消失病灶總體積
  - 整體總變化

### Requirement: 病灶狀態分類
系統 SHALL 為每個追蹤的病灶分配狀態標籤。

#### Scenario: 穩定病灶
- WHEN 病灶體積變化 <10%
- AND 配對信心度 >0.7
- THEN 系統應標記為 "stable"
- AND 使用綠色標示

#### Scenario: 增大病灶
- WHEN 病灶體積增加 ≥10%
- THEN 系統應標記為 "enlarged"
- AND 使用黃色標示
- AND 記錄增大百分比

#### Scenario: 縮小病灶
- WHEN 病灶體積減少 ≥10%
- THEN 系統應標記為 "reduced"
- AND 使用藍色標示
- AND 記錄縮小百分比

#### Scenario: 新增病灶
- WHEN 病灶在 baseline 中不存在
- THEN 系統應標記為 "new"
- AND 使用紅色標示

#### Scenario: 消失病灶
- WHEN 病灶在 followup 中不存在
- THEN 系統應標記為 "disappeared"
- AND 使用灰色標示

### Requirement: 視覺化輸出
系統 MUST 提供多種視覺化方式展示比較結果。

#### Scenario: 並排影像顯示
- WHEN 產生視覺化輸出
- THEN 系統應提供並排影像檢視
- AND 左側為 baseline，右側為 followup
- AND 影像應已配準對齊
- AND 支援切片同步滾動

#### Scenario: 差異圖
- WHEN 使用者請求差異圖
- THEN 系統應計算：Diff = Followup - Baseline
- AND 使用熱力圖著色（紅色增加、藍色減少）
- AND 提供強度調整滑桿

#### Scenario: 病灶標記覆蓋
- WHEN 顯示影像
- THEN 系統應在影像上疊加病灶輪廓
- AND 使用狀態對應的顏色
- AND 標註病灶 ID
- AND 點擊病灶顯示詳細資訊

#### Scenario: 3D 渲染
- WHEN 使用者請求 3D 視圖
- THEN 系統應產生病灶的 3D 模型
- AND 支援旋轉、縮放
- AND 可選擇性顯示/隱藏不同狀態的病灶
- AND 提供前後對比動畫

#### Scenario: 統計圖表
- WHEN 產生報告
- THEN 系統應包含以下圖表：
  - 圓餅圖：病灶狀態分布
  - 柱狀圖：體積比較（baseline vs followup）
  - 散點圖：個別病灶體積變化
  - 折線圖：時間序列（若有多時間點）

### Requirement: 報告生成
系統 MUST 產生完整的比較分析報告。

#### Scenario: 文字報告內容
- WHEN 生成報告
- THEN 報告應包含以下章節：
  1. 患者資訊與檢查資訊
  2. 檢查間隔時間
  3. 主要發現摘要
  4. 總體積變化統計
  5. 病灶狀態分布
  6. 詳細病灶追蹤表格
  7. 視覺化圖表
  8. 結論與建議

#### Scenario: PDF 報告格式
- WHEN 產生 PDF 報告
- THEN 報告應包含：
  - 專業排版
  - 嵌入影像與圖表
  - 頁碼與目錄
  - 機構標誌（如設定）
  - 產生日期與版本號

#### Scenario: DICOM-SR 產生
- WHEN 產生 DICOM Structured Report
- THEN SR 應符合 DICOM 標準
- AND 包含測量值（TID 1500 Measurement Report）
- AND 參照原始影像 Series
- AND 可上傳至 Orthanc PACS

#### Scenario: 自動結論生成
- WHEN 總體積變化 <10%
- THEN 結論應包含「病灶整體穩定」
- WHEN 總體積變化 10-30%
- THEN 結論應包含「病灶有輕度變化，建議持續追蹤」
- WHEN 總體積變化 >30%
- THEN 結論應包含「病灶有顯著變化，建議臨床評估」
- WHEN 有新增病灶
- THEN 結論應包含「出現 {N} 個新增病灶，建議進一步檢查」

### Requirement: 檔案結果儲存
系統 MUST 將比較結果儲存為檔案格式。

#### Scenario: 資料夾結構創建
- WHEN 開始比較分析
- THEN 系統應創建標準資料夾結構
- AND 包含 baseline/, followup/, registration/, analysis/, visualization/, dicom/ 子資料夾
- AND 資料夾命名使用 comparison_ID

#### Scenario: JSON 結果儲存
- WHEN 比較分析完成
- THEN 系統應產生 JSON 檔案
- AND 包含完整的比較資訊
- AND 檔案命名為 {comparison_ID}_comparison.json
- AND JSON 格式如下：
```json
{
  "comparison_id": "baseline_followup",
  "baseline_id": "...",
  "followup_id": "...",
  "comparison_type": "infarct",
  "date_interval_days": 120,
  "registration_quality": 0.85,
  "statistics": {
    "total_baseline_volume_cm3": 5.6,
    "total_followup_volume_cm3": 7.2,
    "volume_change_cm3": 1.6,
    "volume_change_percent": 28.6,
    "stable_count": 3,
    "enlarged_count": 2,
    "reduced_count": 1,
    "new_count": 2,
    "disappeared_count": 0
  },
  "lesions": [...]
}
```

#### Scenario: 視覺化檔案儲存
- WHEN 產生視覺化影像
- THEN 系統應儲存至 visualization/ 資料夾
- AND 檔案命名清晰描述內容（如 side_by_side.png）
- AND 至少產生 4 種視覺化圖片

#### Scenario: 配準結果儲存
- WHEN 配準完成
- THEN 系統應儲存配準變換參數至 registration/transform.tfm
- AND 儲存配準品質報告至 registration/registration_quality.txt
- AND 儲存配準後的影像至 followup/ 資料夾

### Requirement: 效能標準
系統 MUST 滿足以下效能要求。

#### Scenario: 配準時間限制
- WHEN 執行標準腦部 MRI 配準（256×256×180）
- THEN 完整配準時間應 <2 分鐘
- AND 快速模式配準應 <30 秒

#### Scenario: 完整分析時間
- WHEN 執行完整比較分析（配準 + 分析 + 視覺化 + 報告）
- THEN 總時間應 <5 分鐘
- AND 90% 案例應 <3 分鐘完成

#### Scenario: 記憶體使用限制
- WHEN 處理標準大小影像
- THEN 記憶體峰值使用應 <4GB
- AND 支援同時處理 2 個比較任務

#### Scenario: 檔案 I/O 效能
- WHEN 載入與儲存 NIfTI 檔案
- THEN 載入時間應 <10 秒（標準大小影像）
- AND 儲存時間應 <5 秒

### Requirement: 錯誤處理
系統 MUST 妥善處理各種錯誤情況。

#### Scenario: 配準失敗
- WHEN 配準多次嘗試仍失敗
- THEN 系統應記錄詳細錯誤訊息
- AND 通知使用者配準失敗原因
- AND 提供手動配準選項
- AND 不中斷其他批次任務

#### Scenario: 資料不完整
- WHEN 檢查記錄缺少必要資訊
- THEN 系統應列出缺少的欄位
- AND 標記為「無法比較」
- AND 建議補充資料

#### Scenario: 檔案寫入失敗
- WHEN 儲存結果時磁碟空間不足或權限問題
- THEN 系統應返回清晰錯誤訊息
- AND 記錄錯誤日誌
- AND 清理部分寫入的檔案

#### Scenario: Orthanc 上傳失敗
- WHEN DICOM-SR 上傳失敗
- THEN 系統應重試 3 次
- AND 將 SR 檔案保存至本地
- AND 記錄失敗的 Study UID
- AND 提供手動上傳選項

### Requirement: Follow-up 輸出檔名規則（含 model 與大小寫）
系統 MUST 依照 `model` 參數（保留大小寫）產出固定命名的 Follow-up 產物檔案。

#### Scenario: 輸出 Follow-up JSON（檔名含 model）
- WHEN Follow-up pipeline 完成報告輸出
- THEN 系統 MUST 產出 `Followup_<model>_platform_json.json`

#### Scenario: 輸出配準矩陣（固定前綴 baseline）
- WHEN Follow-up pipeline 完成影像配準
- THEN 系統 MUST 於 `registration/` 產出 `<baseline_id>_<model>.mat`

#### Scenario: process 根目錄相容 CLI default
- WHEN `path_process` 已指向 `.../Deep_FollowUp/`
- THEN 系統 MUST 使用 `<path_process>/<baseline_id>/` 作為處理根目錄
- AND MUST NOT 再額外拼接第二層 `Deep_FollowUp/`

### Requirement: 配準前 sub-mm spacing 正規化為 1mm 並回原空間輸出
系統 MUST 在執行 Follow-up → Baseline 線性配準前，針對 sub-mm 影像進行解析度正規化（逐軸 min-1mm），且最終輸出 MUST 保持在 baseline 原空間。

#### Scenario: 單次重採樣完成 min-1mm（實作要求）
- WHEN 系統需要執行 min-1mm 重採樣
- THEN 系統 SHOULD 以單次重採樣操作同時完成所有需變動的軸（single-step）
- AND label 重採樣 MUST 使用最近鄰插值（nearest neighbour）

### Requirement: min-1mm resample 檔案必須可追溯且不可互相覆寫
系統 MUST 在啟用配準前的 min-1mm resample 時，分別保存 baseline 與 followup 的 resample 產物，且 MUST NOT 發生檔名覆寫。

#### Scenario: baseline 與 followup 的輸入檔名相同
- WHEN baseline 與 followup 的 SynthSEG 輸入檔名相同（例如皆為 `SynthSEG_<model>.nii.gz`）
- THEN 系統 MUST 仍能分別保存兩份 resample 後的 NIfTI（baseline 與 followup 各一份）
- AND 系統 MUST NOT 因檔名覆寫導致配準矩陣或輸出錯誤

### Requirement: min-1mm 模式下 Pred 必須在同一工作空間套用矩陣並回 baseline Pred 原空間
系統 MUST 在啟用 min-1mm 配準模式時，將 followup Pred 先轉入 min-1mm 工作空間，並在該工作空間直接套用配準矩陣，最後輸出 MUST 回到 baseline Pred 的原始 voxel grid，以確保 voxel index overlap 配對正確。

#### Scenario: min-1mm 啟用時的 Pred 處理
- WHEN 系統啟用 min-1mm 配準模式
- THEN 系統 MUST 先將 baseline/followup Pred 轉換到 min-1mm 工作空間
- AND 系統 MUST 確保 SynthSEG 與 Pred 在同一工作空間 voxel grid 一致（必要時 resample SynthSEG 到 Pred grid）
- AND 系統 MUST 在 min-1mm 工作空間直接套用矩陣到 followup Pred
- AND 系統 MUST 將 Pred(reg) resample 回 baseline Pred 的原始 voxel grid

### Requirement: 配準前 Pred 必須與 SynthSEG 使用相同的幾何 header
系統 MUST 在執行 Follow-up 配準前，確保 baseline 與 followup 的 `Pred_<model>.nii.gz` 與對應的 `SynthSEG_<model>.nii.gz` 使用相同的幾何資訊（affine / sform / qform），以避免 voxel grid 錯位導致 overlap 配對失敗。

#### Scenario: 覆蓋 Pred 幾何資訊
- WHEN 系統已在 `baseline/` 與 `followup/` 目錄準備好 `Pred_<model>.nii.gz` 與 `SynthSEG_<model>.nii.gz`
- THEN 系統 MUST 以 `SynthSEG_<model>.nii.gz` 的幾何資訊覆蓋 `Pred_<model>.nii.gz`
- AND 系統 MUST 保存更新後的 `Pred_<model>.nii.gz`

#### Scenario: shape 不一致
- WHEN `Pred_<model>.nii.gz` 與 `SynthSEG_<model>.nii.gz` 的 3D shape 不一致
- THEN 系統 MUST 立即中止並回報錯誤

### Requirement: min-1mm resample MUST 使用 conform 並更新幾何 header
系統 MUST 在執行 min-1mm resample 時使用 `nibabel.processing.conform`，並確保輸出 NIfTI 的 affine/header 反映新的 voxel size 與 shape，以避免後續配準與 overlap 計算發生錯位。

#### Scenario: min-1mm resample 使用 conform
- WHEN 系統需要將 label 影像做 min-1mm resample
- THEN 系統 MUST 使用 `nibabel.processing.conform`
- AND label resample MUST 使用最近鄰插值（order=0）

### Requirement: 配準中間產物 MUST 可供 QA 檢視
系統 MUST 在 Follow-up 配準流程中，將對位後的 resample/registered 影像集中保存，供人工 QA 檢視是否有整體平移、翻轉或重疊失敗。

#### Scenario: 保存配準中間產物
- WHEN 系統執行 Follow-up → Baseline 配準
- THEN 系統 MUST 將對位後的影像保存到 `registration/resample/`
- AND 若啟用 min-1mm，系統 MUST 另外保存 min-1mm 空間的 registered SynthSEG 影像與對應的 `_min1mm.mat`

### Requirement: Follow-up 使用說明（README）
系統 MUST 在專案 README 中提供 Follow-up（檔案導向）流程的最小可用說明，以降低誤用造成的失敗與排錯成本。

#### Scenario: README 描述關鍵流程
- WHEN 使用者查閱 `README.md` 的 Follow-up 小節
- THEN 文件 MUST 說明配準前會以 SynthSEG 的幾何資訊（affine/sform/qform）覆蓋 Pred
- AND MUST 提醒 Pred 與 SynthSEG 的 shape 必須一致
- AND MUST 說明配準依賴 FSL `flirt`（可透過 `--fsl_flirt_path` 指定）

#### Scenario: README 列出主要輸出與標籤定義
- WHEN Follow-up pipeline 執行完成
- THEN 文件 MUST 列出主要輸出檔名：
  - `Pred_<model>_followup.nii.gz`
  - `Followup_<model>_platform_json.json`
- AND MUST 說明 `Pred_<model>_followup.nii.gz` 的標籤定義為（固定）：
  - 1 = new
  - 2 = stable
  - 3 = disappeared

