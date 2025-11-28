# OpenSpec 使用指南

本專案已整合 **OpenSpec** 進行規格驅動開發。OpenSpec 幫助我們在實作前先定義清晰的需求與規格。

## 📁 目錄結構

```
openspec/
├── project.md          # 專案概述、技術堆疊、編碼慣例
├── specs/              # 當前系統規格（真實來源）
├── changes/            # 進行中的變更提案
└── archive/            # 已完成的變更歷史
```

## 🚀 快速開始

### 1. 創建新功能提案

使用 Cursor 的 slash command：
```
/openspec:proposal 新增 CT 肺部結節檢測功能
```

或自然語言：
```
請創建一個 OpenSpec 變更提案，用於新增 CT 肺部結節檢測功能
```

這將創建：
```
openspec/changes/add-ct-lung-nodule-detection/
├── proposal.md         # 為什麼要做這個變更
├── tasks.md            # 實作任務清單
└── specs/              # 規格增量
    └── lung-detection/
        └── spec.md
```

### 2. 查看與驗證提案

```bash
# 列出所有活動變更
openspec list

# 查看特定變更
openspec show add-ct-lung-nodule-detection

# 驗證規格格式
openspec validate add-ct-lung-nodule-detection
```

或使用 slash command：
```
/openspec:show add-ct-lung-nodule-detection
/openspec:validate add-ct-lung-nodule-detection
```

### 3. 實作變更

一旦規格確定，開始實作：
```
/openspec:apply add-ct-lung-nodule-detection
```

AI 助手會：
- 讀取 `tasks.md` 中的任務
- 根據規格增量實作
- 標記完成的任務 `[x]`

### 4. 歸檔完成的變更

實作完成後：
```
/openspec:archive add-ct-lung-nodule-detection
```

或命令列：
```bash
openspec archive add-ct-lung-nodule-detection --yes
```

這將：
- 將變更移至 `openspec/archive/`
- 合併規格增量到 `openspec/specs/`
- 更新系統真實規格

## 📝 規格增量格式

規格增量使用以下格式：

```markdown
# <模組名稱> 規格增量

## ADDED Requirements
### Requirement: 新需求名稱
系統 SHALL/MUST 執行某行為。

#### Scenario: 場景描述
- WHEN 某條件成立
- THEN 系統應該做某事

## MODIFIED Requirements
### Requirement: 修改的需求
<完整更新後的需求描述>

## REMOVED Requirements
### Requirement: 移除的需求
<已棄用的需求描述>
```

### 範例：腦梗塞檢測規格增量

```markdown
# 腦梗塞檢測模組規格增量

## ADDED Requirements
### Requirement: 急性腦梗塞檢測
系統 MUST 在 T1/T2/DWI 序列中檢測急性腦梗塞區域。

#### Scenario: 檢測到急性梗塞
- WHEN 輸入為標準 DWI 序列
- THEN 系統應輸出梗塞區域分割遮罩
- AND 計算梗塞體積（cm³）

#### Scenario: 多發性梗塞
- WHEN 影像包含多個梗塞區域
- THEN 系統應分別標記每個區域
- AND 提供各區域的體積與位置資訊
```

## 🏥 醫學影像專案特定慣例

### 提案內容要求
每個 `proposal.md` 應包含：
- **臨床背景**: 解決什麼臨床問題
- **技術方案**: AI 模型、演算法選擇
- **DICOM 考量**: 需要哪些 DICOM 標籤
- **效能要求**: 處理時間、記憶體限制
- **驗證標準**: Dice 係數、敏感度、特異度等

### 任務分解範本
```markdown
## 1. 資料庫變更
- [ ] 1.1 新增結果資料表
- [ ] 1.2 更新 Schema 版本

## 2. DICOM 前處理
- [ ] 2.1 實作序列檢測邏輯
- [ ] 2.2 新增影像正規化

## 3. AI 模型整合
- [ ] 3.1 載入預訓練模型
- [ ] 3.2 實作推理管道
- [ ] 3.3 GPU 記憶體優化

## 4. 後處理與上傳
- [ ] 4.1 產生 DICOM-SEG
- [ ] 4.2 上傳至 Orthanc
- [ ] 4.3 更新資料庫狀態

## 5. 測試與驗證
- [ ] 5.1 單元測試
- [ ] 5.2 臨床驗證資料集測試
- [ ] 5.3 效能基準測試
```

## 🔧 命令參考

### 基本命令
```bash
# 列出所有活動變更
openspec list

# 互動式儀表板
openspec view

# 顯示變更詳情
openspec show <change-name>

# 驗證規格格式
openspec validate <change-name>

# 歸檔完成的變更（非互動式）
openspec archive <change-name> --yes
```

### Cursor Slash Commands
```
/openspec:proposal <description>    # 創建提案
/openspec:apply <change-name>       # 實作變更
/openspec:archive <change-name>     # 歸檔變更
/openspec:show <change-name>        # 顯示變更
/openspec:list                      # 列出變更
/openspec:validate <change-name>    # 驗證規格
```

## 💡 最佳實踐

### 1. 先規格，後實作
- 永遠先寫提案與規格增量
- 與團隊成員審查規格
- 確認臨床需求與技術可行性
- 再開始寫程式碼

### 2. 保持任務細粒度
- 每個任務應該是可獨立完成的
- 任務描述清晰、可驗證
- 包含驗收標準

### 3. 遵循編碼規範
- 優先遵循 Linus 風格（見 `.cursor/rules/linus.mdc`）
- 縮排不超過 3 層
- 使用 Early Return
- 資料結構驅動設計

### 4. 醫學影像特定
- 使用 `.get()` 安全存取 DICOM 標籤
- 檢測 REFORMATTED 序列
- 包含臨床驗證指標
- 記錄效能指標

### 5. 及時歸檔
- 完成實作後立即歸檔
- 保持活動變更數量最小
- 定期審查歸檔內容

## 📚 範例工作流程

### 完整範例：新增白質病變追蹤功能

#### Step 1: 創建提案
```
You: /openspec:proposal Add longitudinal WMH volume tracking

AI:  我將創建 WMH 追蹤功能的 OpenSpec 變更提案。
     *創建 openspec/changes/add-wmh-tracking/*
```

#### Step 2: 審查與完善
```bash
$ openspec show add-wmh-tracking
$ openspec validate add-wmh-tracking
```

手動編輯 `proposal.md` 與 `tasks.md`，補充：
- 臨床使用場景
- 資料庫 Schema 設計
- 時間序列比對演算法

#### Step 3: 實作
```
You: /openspec:apply add-wmh-tracking

AI:  開始實作 add-wmh-tracking 變更。
     *實作 tasks.md 中的任務*
     ✓ Task 1.1: 新增 wmh_tracking 資料表
     ✓ Task 1.2: 實作體積計算函數
     ✓ Task 2.1: 建立時間序列查詢 API
     ...
```

#### Step 4: 歸檔
```
You: /openspec:archive add-wmh-tracking

AI:  歸檔 add-wmh-tracking 變更。
     *執行: openspec archive add-wmh-tracking --yes*
     ✓ 變更已歸檔。規格已更新。準備下一個功能！
```

## 🆘 疑難排解

### Q: OpenSpec 命令找不到？
確認已全局安裝：
```bash
npm install -g @fission-ai/openspec@latest
openspec --version
```

### Q: Slash commands 沒有出現？
重啟 Cursor。Slash commands 在啟動時載入。

### Q: 如何更新 OpenSpec？
```bash
npm install -g @fission-ai/openspec@latest
openspec update  # 在專案目錄中執行
```

### Q: 可以同時有多個活動變更嗎？
可以，但建議保持最小數量（1-3 個）以維持專注。

### Q: 變更名稱格式？
使用 kebab-case：`add-feature-name`、`fix-bug-description`

## 🔗 相關資源

- **OpenSpec 官網**: https://openspec.dev/
- **GitHub**: https://github.com/Fission-AI/OpenSpec
- **專案規範**: [openspec/project.md](./project.md)
- **AI 代理指南**: [AGENTS.md](../AGENTS.md)
- **編碼規範**: [.cursor/rules/linus.mdc](../.cursor/rules/linus.mdc)

## 📞 支援

遇到問題？
1. 查看 [AGENTS.md](../AGENTS.md) 瞭解完整工作流程
2. 執行 `openspec validate` 檢查規格格式
3. 參考 OpenSpec 官方文檔

---

**記住**: OpenSpec 幫助我們在寫程式碼之前先達成行為共識。

