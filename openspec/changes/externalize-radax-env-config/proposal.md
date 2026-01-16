# Externalize Radax environment config (paths + API port)

## Problem
`deploy/radax-test` 與 `Deploy/radax official` 兩個分支目前主要差異只在：
- 檔案系統路徑（`path_code / path_process / model paths / json / log`）
- 呼叫 inference-complete API 的 port（`localhost:<port>`）

但目前這些差異散落在：
- `radax/pipeline_aneurysm_tensorflow.py`（hardcode api_url 與各種路徑）
- `radax/pipeline_aneurysm.sh`（hardcode conda env 與 python 腳本路徑）

導致測試機分支要合併到正式機分支時，需要大量手動調整與比對，容易出錯且費時。

## Solution
新增一份可被 bash 與 python 共用的 **環境設定檔**（`KEY=VALUE` 格式），將「路徑 + API port」集中管理：

- bash（`.sh`）啟動前 `source` env 檔，把 key export 到環境變數
- python 在 `__main__` 入口自動讀取：
  - `RADX_ENV_FILE` 指定的 env 檔（若有）
  - 或預設 `radax/config/radax.env`

`deploy/radax-test` 與 `Deploy/radax official` 之後只要維護不同的 env 檔即可，程式碼維持一致。

## Scope / Impact
### In scope
- `radax/pipeline_aneurysm_tensorflow.py`
  - API URL 由 env 決定（`RADX_API_URL` 或 `RADX_API_HOST/RADX_API_PORT`）
  - `__main__` 內的各路徑由 env 決定（如 `RADX_CODE_ROOT`、`RADX_*_MODEL` 等）
- `radax/pipeline_aneurysm.sh`
  - conda.sh、conda env、python binary、pipeline path 由 env 決定
- 新增 env 檔：
  - `radax/config/radax-test.env`
  - `radax/config/radax-official.env`
  - `radax/config/radax.env`（目前 active 的設定）

### Out of scope (for this change)
- 其它 pipeline（infarct/wmh/followup）也做同樣抽取（可另開 change）
- 將設定檔改為 JSON/YAML（本次先用最輕量的 KEY=VALUE）

## Acceptance Criteria
- 測試機/正式機僅需切換 env 檔（或 `RADX_ENV_FILE`）即可改變：
  - inference-complete API port
  - path_code/path_process/json/log/model paths
  - conda env 與 python 腳本路徑
- `radax/pipeline_aneurysm_tensorflow.py` 不再 hardcode `localhost:24000`
- `radax/pipeline_aneurysm.sh` 不再 hardcode conda env 與 python 腳本路徑

