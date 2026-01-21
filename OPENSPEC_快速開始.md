# OpenSpec 快速開始指南

## ✅ 安裝完成

OpenSpec 已成功安裝並配置於本專案！

### 安裝版本
- **OpenSpec**: v0.16.0
- **安裝方式**: npm 全局安裝
- **配置工具**: Cursor (原生 slash commands 支援)

## 📁 已創建的目錄結構

```
C:\Users\user\Desktop\orthanc_combine_code\目前pipeline版本\code\
├── AGENTS.md                           # AI 代理工作流程指南
├── .cursor/
│   └── openspec-commands.md            # Cursor slash commands 定義
└── openspec/
    ├── README.md                       # OpenSpec 使用指南（中文）
    ├── project.md                      # 專案概述與編碼規範
    ├── specs/                          # 系統規格（當前真實來源）
    ├── changes/                        # 進行中的變更提案
    │   └── example-add-ct-lung-nodule-detection/  # 完整範例變更
    │       ├── README.md               # 範例說明
    │       ├── proposal.md             # 變更提案
    │       ├── tasks.md                # 任務清單（47 項）
    │       └── specs/
    │           └── lung-detection/
    │               └── spec.md         # 規格增量
    └── archive/                        # 已完成變更的歷史記錄
```

## 🚀 立即開始使用

### 方法 1: 使用 Cursor Slash Commands（推薦）

Cursor 已原生支援以下 OpenSpec 命令：

```
/openspec:proposal <功能描述>    # 創建新變更提案
/openspec:list                    # 列出所有活動變更
/openspec:show <變更名稱>         # 顯示變更詳情
/openspec:validate <變更名稱>     # 驗證規格格式
/openspec:apply <變更名稱>        # 實作變更
/openspec:archive <變更名稱>      # 歸檔完成的變更
```

**注意**: 如果 slash commands 沒有立即出現，請重啟 Cursor。

### 方法 2: 使用自然語言

您也可以直接用自然語言與 AI 助手互動：

```
You: 請創建一個 OpenSpec 變更提案，用於新增腦出血檢測功能

AI:  我將為您創建腦出血檢測功能的 OpenSpec 變更提案...
     *創建 openspec/changes/add-brain-hemorrhage-detection/*
```

### 方法 3: 使用命令列

在終端機中直接使用 OpenSpec CLI：

```bash
# 列出所有活動變更
openspec list

# 查看範例變更
openspec show example-add-ct-lung-nodule-detection

# 驗證規格格式
openspec validate example-add-ct-lung-nodule-detection
```

## 📖 學習資源

### 必讀文件（依優先順序）
1. **[openspec/README.md](openspec/README.md)** 
   - OpenSpec 完整使用指南（中文）
   - 工作流程說明
   - 命令參考

2. **[範例變更](openspec/changes/example-add-ct-lung-nodule-detection/README.md)**
   - 查看完整的變更提案範例
   - 學習如何撰寫提案、任務、規格

3. **[AGENTS.md](AGENTS.md)**
   - AI 代理工作流程指南
   - OpenSpec 操作細節

4. **[openspec/project.md](openspec/project.md)**
   - 專案技術堆疊
   - 編碼規範與慣例

## 🎯 實際應用範例

### 範例 1: 新增腦出血檢測功能

```
You: /openspec:proposal 新增 CT 腦出血自動檢測與分類功能

AI:  我將創建腦出血檢測功能的變更提案。
     *創建 openspec/changes/add-brain-hemorrhage-detection/*
     *產生 proposal.md, tasks.md, specs/*
```

### 範例 2: 改進現有白質病變管道

```
You: /openspec:proposal 優化白質病變檢測的處理速度，目標減少 30% 處理時間

AI:  我將創建 WMH 處理優化的變更提案。
     *創建 openspec/changes/optimize-wmh-processing/*
     *分析當前瓶頸，提出優化方案*
```

### 範例 3: 修復 DICOM 轉換 Bug

```
You: /openspec:proposal 修復 REFORMATTED 序列檢測失敗的問題

AI:  我將創建 DICOM 轉換修復的變更提案。
     *創建 openspec/changes/fix-reformatted-detection/*
     *記錄問題、提出修復方案*
```

## 🔄 完整工作流程示範

### Step 1: 創建提案
```
You: /openspec:proposal 新增 MRI FLAIR 序列支援

AI:  ✓ 已創建 openspec/changes/add-mri-flair-support/
```

### Step 2: 查看與完善
```bash
# 終端機
$ openspec show add-mri-flair-support
$ openspec validate add-mri-flair-support

# 或使用 Cursor
/openspec:show add-mri-flair-support
```

手動編輯檔案以完善細節：
- `proposal.md` - 補充臨床需求
- `tasks.md` - 調整任務清單
- `specs/` - 細化規格需求

### Step 3: 實作
```
You: 規格看起來不錯，開始實作吧
     /openspec:apply add-mri-flair-support

AI:  開始實作 add-mri-flair-support 變更...
     ✓ Task 1.1: 新增 FLAIR 序列檢測邏輯
     ✓ Task 1.2: 實作影像正規化
     ... (逐步完成任務)
```

### Step 4: 歸檔
```
You: /openspec:archive add-mri-flair-support

AI:  執行: openspec archive add-mri-flair-support --yes
     ✓ 變更已歸檔至 openspec/archive/
     ✓ 規格已合併至 openspec/specs/
     ✓ 準備下一個功能！
```

## 🛠️ 常用命令

```bash
# 查看當前狀態
openspec list

# 互動式儀表板（推薦）
openspec view

# 驗證所有變更
for change in openspec/changes/*; do
    openspec validate $(basename $change)
done

# 查看幫助
openspec --help
openspec <command> --help
```

## 📝 關鍵原則

### 1. 先規格，後實作
永遠先定義行為需求，再寫程式碼。OpenSpec 幫助您在實作前達成共識。

### 2. 保持活動變更最小
建議同時只有 1-3 個活動變更，完成後立即歸檔。

### 3. 遵循專案規範
所有變更都應遵循：
- **Linus 風格**優先（縮排 ≤3 層，Early Return）
- **醫學影像特定**慣例（DICOM 安全存取、REFORMATTED 檢測）
- **臨床驗證**標準（Dice、Sensitivity、Specificity）

### 4. 利用範例學習
參考 `openspec/changes/example-add-ct-lung-nodule-detection/` 瞭解：
- 如何撰寫完整提案
- 如何分解任務
- 如何編寫規格增量

## 🎓 進階使用

### 自訂 AI 助手行為
編輯 `AGENTS.md` 調整 AI 助手的工作流程與指引。

### 新增專案規範
編輯 `openspec/project.md` 更新技術堆疊、編碼慣例等。

### 查看已完成變更
```bash
ls openspec/archive/
# 已歸檔的變更都在這裡
```

## 🆘 疑難排解

### Q: Slash commands 沒有出現？
**A**: 重啟 Cursor。Slash commands 在啟動時載入。

### Q: `openspec` 命令找不到？
**A**: 確認已全局安裝：
```bash
npm install -g @fission-ai/openspec@latest
```

### Q: 如何更新 OpenSpec？
**A**: 
```bash
npm install -g @fission-ai/openspec@latest
openspec update  # 在專案目錄中執行以刷新配置
```

### Q: PowerShell 路徑編碼問題？
**A**: 使用絕對路徑或在正確目錄中執行命令。本指南已為您配置好所有必要檔案。

## 🔗 相關連結

- **OpenSpec 官網**: https://openspec.dev/
- **GitHub**: https://github.com/Fission-AI/OpenSpec
- **專案 README**: [README.md](README.md)
- **Git 分支管理**: [Git分支管理完整指南.md](../Git分支管理完整指南.md)

## ✨ 下一步

1. **閱讀範例**: 查看 `openspec/changes/example-add-ct-lung-nodule-detection/`
2. **創建第一個變更**: 使用 `/openspec:proposal` 創建您的第一個變更提案
3. **熟悉工作流程**: 完整走一遍 創建 → 審查 → 實作 → 歸檔 流程

---

**祝您使用 OpenSpec 開發愉快！** 🎉

有任何問題，請查看 [openspec/README.md](openspec/README.md) 或詢問 AI 助手。

