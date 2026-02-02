# Follow-up v3 模組化為平台同步 API

## 問題陳述
目前 Follow-up v3 僅作為管線內部流程運行，平台端無法透過正式 API 呼叫觸發。
缺少明確的 API 介面定義與錯誤回應約定，影響平台整合與追蹤結果一致性。

## 解決方案
建立 Follow-up v3 的同步 API 規格：
- 以 HTTP JSON body 作為輸入（對齊 `follow_up_input.json` 結構）
- 回傳 JSON body（`followup_manifest`，包含完整 `items[]`）
- 呼叫端可先將回傳內容寫入實體檔案，再以 Python 上傳
- 依 radax follow-up v3 的單次通知機制回傳 manifest
- 明確規範輸入驗證、錯誤回應與回傳格式
- 不變更既有 follow-up 演算法與輸出內容

## 影響範圍
- 平台端 API 呼叫與回應解析（manifest 單次通知）
- Follow-up v3 輸入 JSON body 驗證規則
- 結果回傳格式（manifest JSON body，含 `items[]`）
- 錯誤回應與缺檔回報格式

## 臨床考量
- 追蹤比對主體規則維持不變，避免回歸性偏差
- 病灶狀態（new/stable/regress）定義不變
- follow_up 欄位與 grouped schema 對齊，支援臨床追溯

## 技術考量
- API 端點路徑：`/api/radax/followup`
- 以 Flask 作為 API 實作框架
- 程式部署位置：`/home/david/pipeline/chuan/radax`
- 同步 API 需有處理時間上限與逾時回應規範
- JSON payload 可能較大，需明確欄位與缺漏處理
- Log 位置：`/home/david/pipeline/chuan/log/YYYYMMDD.log`（每件事需寫入日期時間）
- 認證機制暫不納入，保留未來擴充空間
