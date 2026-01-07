# 範例變更：CT 肺部結節檢測

> **注意**: 這是一個**示範性範例**，展示如何使用 OpenSpec 管理專案變更。
> 此變更尚未實作，僅作為學習與參考用途。

## 關於此範例

此範例展示了完整的 OpenSpec 變更提案結構，包括：
- **proposal.md** - 詳細的變更提案，包含問題陳述、解決方案、影響範圍
- **tasks.md** - 實作任務清單，涵蓋開發、測試、文檔
- **specs/lung-detection/spec.md** - 規格增量，定義系統行為需求

## 檔案結構

```
example-add-ct-lung-nodule-detection/
├── README.md               # 本檔案（範例說明）
├── proposal.md             # 變更提案
├── tasks.md                # 任務清單（共 47 項）
└── specs/                  # 規格增量
    └── lung-detection/
        └── spec.md         # 肺部結節檢測規格
```

## 如何使用此範例

### 1. 學習提案結構
閱讀 `proposal.md` 瞭解：
- 如何陳述臨床背景與技術挑戰
- 如何設計解決方案架構
- 如何評估影響範圍與風險

### 2. 瞭解任務分解
查看 `tasks.md` 學習：
- 如何將大功能拆解為小任務
- 任務的合理顆粒度
- 各階段任務的組織方式

### 3. 學習規格編寫
研讀 `specs/lung-detection/spec.md` 瞭解：
- Requirement / Scenario 格式
- MUST / SHALL / SHOULD 的使用
- WHEN / THEN 條件表達
- 功能性與非功能性需求的組織

## 將此範例應用到實際開發

### 複製此結構創建新變更
```bash
# 複製範例目錄
cp -r openspec/changes/example-add-ct-lung-nodule-detection \
      openspec/changes/your-new-feature

# 修改內容
cd openspec/changes/your-new-feature
# 編輯 proposal.md, tasks.md, specs/...
```

### 使用 AI 助手創建類似變更
```
You: 請參考 openspec/changes/example-add-ct-lung-nodule-detection/
     創建一個新的 OpenSpec 變更提案，用於新增 [您的功能描述]

或使用 Cursor slash command:
/openspec:proposal [您的功能描述]
```

## 關鍵學習點

### 1. 完整的問題分析
範例展示了如何從臨床需求出發，分析技術挑戰，設計解決方案。

### 2. 任務分解最佳實踐
- **分階段**: 資料庫 → 前處理 → AI → 後處理 → 測試 → 文檔
- **可追蹤**: 每個任務有明確的完成標準
- **可並行**: 某些任務可以同時進行

### 3. 規格編寫技巧
- **層次結構**: Requirements → Scenarios → WHEN/THEN
- **明確語言**: MUST（必須）、SHALL（應該）、SHOULD（建議）
- **場景覆蓋**: 正常情況、邊界案例、錯誤處理

### 4. 醫學影像特定考量
- **臨床驗證**: Dice 係數、敏感度、特異度
- **DICOM 標準**: 正確處理 DICOM 標籤與 SEG 格式
- **效能需求**: 處理時間、記憶體使用限制
- **品質控制**: 影像品質檢查、信心度過濾

## 實際開發流程

若要將此範例轉為實際變更：

1. **審查提案**: 與團隊討論技術方案與臨床需求
2. **調整任務**: 根據實際情況修改任務清單
3. **完善規格**: 補充遺漏的需求與場景
4. **開始實作**: 使用 `/openspec:apply` 開始實作
5. **追蹤進度**: 逐項標記完成的任務
6. **歸檔變更**: 完成後使用 `/openspec:archive` 歸檔

## 參考資源

### 專案內部
- [OpenSpec README](../../README.md) - OpenSpec 完整使用指南
- [AGENTS.md](../../../AGENTS.md) - AI 代理工作流程
- [project.md](../../project.md) - 專案規範與慣例

### 外部資源
- [LUNA16 Challenge](https://luna16.grand-challenge.org/) - 肺部結節檢測挑戰
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - 醫學影像分割框架
- [DICOM Standard](https://www.dicomstandard.org/) - DICOM 標準文檔

## 常見問題

### Q: 為什麼任務如此詳細？
A: 詳細的任務清單幫助：
- AI 助手理解實作細節
- 開發者追蹤進度
- 團隊成員瞭解開發狀態

### Q: 規格是否過於詳盡？
A: 對於關鍵功能（如醫學 AI），詳盡的規格是必要的，因為：
- 確保符合臨床標準
- 減少溝通成本
- 方便後續維護與擴展

### Q: 如何適應敏捷開發？
A: OpenSpec 與敏捷相容：
- 可以創建小的、增量的變更
- 規格可以迭代調整
- 完成的變更立即歸檔

---

**記住**: 這是一個學習範例。實際使用時，根據您的專案需求調整細節程度。

