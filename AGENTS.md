# OpenSpec Agent 指南

本專案使用 OpenSpec 進行規格驅動開發。作為 AI 編碼助手，請遵循以下工作流程。

## OpenSpec 工作流程

### 1. 創建變更提案
當用戶請求新功能或修改時：
1. 在 `openspec/changes/` 下創建新的變更資料夾（使用 kebab-case）
2. 創建以下檔案：
   - `proposal.md` - 變更的原因與內容
   - `tasks.md` - 實作檢查清單
   - `specs/` - 規格增量（delta）

### 2. 規格增量格式
在 `openspec/changes/<change-name>/specs/` 中創建規格增量：

```markdown
# <模組名稱> 規格增量

## ADDED Requirements
### Requirement: <需求名稱>
系統 SHALL/MUST <行為描述>

#### Scenario: <場景名稱>
- WHEN <條件>
- THEN <預期結果>

## MODIFIED Requirements
### Requirement: <需求名稱>
<完整更新後的需求文字>

## REMOVED Requirements
### Requirement: <需求名稱>
<已棄用的需求>
```

### 3. 實作任務
依照 `tasks.md` 中的檢查清單實作：
- 逐項完成任務
- 標記完成的任務為 `[x]`
- 保持任務顆粒度適中

### 4. 歸檔變更
完成實作後：
1. 執行 `openspec archive <change-name> --yes`
2. 變更將移至 `openspec/archive/`
3. 規格增量將合併至 `openspec/specs/`

## 常用命令

```bash
# 列出所有活動變更
openspec list

# 顯示變更詳情
openspec show <change-name>

# 驗證規格格式
openspec validate <change-name>

# 歸檔完成的變更
openspec archive <change-name> --yes
```

## 醫學影像專案特定指南

### 變更提案範本
創建 `proposal.md` 時包含：
- **問題陳述**: 要解決什麼臨床或技術問題
- **解決方案**: 提議的實作方法
- **影響範圍**: 受影響的模組與功能
- **臨床考量**: 對診斷準確性、患者安全的影響
- **技術考量**: 效能、相容性、資料處理

### 任務分解建議
將任務分為：
1. **資料庫變更**: Schema 更新、遷移腳本
2. **核心算法**: 影像處理、AI 模型整合
3. **管道整合**: 前處理、推理、後處理
4. **DICOM 處理**: 標籤擷取、轉換、上傳
5. **測試與驗證**: 單元測試、臨床驗證

### 規格編寫重點
- 使用臨床術語（如 "腦梗塞區域"、"白質病變"）
- 明確 DICOM 標籤需求
- 定義效能指標（處理時間、記憶體使用）
- 包含錯誤處理場景

## 範例：新增雙因素認證

### 1. 創建提案 (openspec/changes/add-2fa/proposal.md)
```markdown
# 新增雙因素認證

## 問題
系統目前僅使用密碼認證，需要增強安全性。

## 解決方案
實作基於 TOTP 的雙因素認證。

## 影響範圍
- 使用者資料庫 Schema
- 登入流程
- API 端點
```

### 2. 定義任務 (openspec/changes/add-2fa/tasks.md)
```markdown
## 1. 資料庫設定
- [ ] 1.1 新增 OTP secret 欄位至 users 表
- [ ] 1.2 創建 OTP 驗證日誌表

## 2. 後端實作
- [ ] 2.1 新增 OTP 生成端點
- [ ] 2.2 修改登入流程以要求 OTP
- [ ] 2.3 新增 OTP 驗證端點

## 3. 前端更新
- [ ] 3.1 創建 OTP 輸入元件
- [ ] 3.2 更新登入流程 UI
```

### 3. 規格增量 (openspec/changes/add-2fa/specs/auth/spec.md)
```markdown
# 認證模組規格增量

## ADDED Requirements
### Requirement: 雙因素認證
系統 MUST 在登入期間要求第二因素。

#### Scenario: 需要 OTP
- WHEN 使用者提交有效憑證
- THEN 系統要求 OTP 驗證

#### Scenario: OTP 驗證成功
- WHEN 使用者提交正確的 OTP
- THEN 系統發放存取權杖
```

## 注意事項

### 遵循專案規範
- 始終遵循 Linus 風格優先原則
- 縮排不超過 3 層
- 使用 Early Return
- 資料結構驅動設計

### 醫學影像特定
- DICOM 標籤使用 `.get()` 安全存取
- 檢測 REFORMATTED 序列
- 維持向後相容性
- 包含臨床驗證指標

### 變更管理
- 一次只處理一個活動變更
- 完成後立即歸檔
- 保持規格與實作同步
- 文檔化所有重要決策

## 支援的 AI 工具

本專案已配置支援以下 AI 編碼助手：
- **Cursor**: 原生 slash commands 支援
- **Claude Code (Anthropic)**: 原生支援
- **其他工具**: 透過此 AGENTS.md 手動整合

## 更新說明

當 OpenSpec 或工具配置更新時：
```bash
# 更新 OpenSpec
npm install -g @fission-ai/openspec@latest

# 在專案中刷新代理指令
openspec update
```

---

**記住**: OpenSpec 幫助我們在寫程式碼之前先達成行為共識。始終從規格開始，而非從實作開始。

