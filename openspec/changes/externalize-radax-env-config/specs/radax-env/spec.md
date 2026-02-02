# Radax environment config 規格增量

## ADDED Requirements

### Requirement: Environment config file (KEY=VALUE)
系統 MUST 支援用單一環境設定檔集中管理「路徑」與「API 連線設定」，以便 `deploy/radax-test` 與 `Deploy/radax official` 只需切換 env 檔即可。

#### Scenario: Bash runner loads env file
- WHEN 使用者執行 `radax/pipeline_aneurysm.sh`
- THEN 系統 MUST 先讀取 env 檔並 export 變數
- AND THEN 以 env 變數決定 conda env / python binary / pipeline 腳本路徑

#### Scenario: Python entry loads env file
- WHEN 使用者直接執行 `radax/pipeline_aneurysm_tensorflow.py`
- THEN 系統 MUST 先嘗試載入 env 檔：
  - `RADX_ENV_FILE`（若存在）
  - 否則使用 `radax/config/radax.env`（若存在）

### Requirement: API endpoint configured by env
系統 MUST 以 env 決定 inference-complete API endpoint。

#### Scenario: RADX_API_URL override
- WHEN `RADX_API_URL` 有設定
- THEN 系統 MUST 使用 `RADX_API_URL` 作為 `POST /v1/ai-inference/inference-complete` 的 endpoint

#### Scenario: Host/Port compose
- WHEN `RADX_API_URL` 未設定
- THEN 系統 MUST 使用 `RADX_API_HOST` 與 `RADX_API_PORT` 組合 endpoint：
  - `http://{RADX_API_HOST}:{RADX_API_PORT}/v1/ai-inference/inference-complete`

### Requirement: Paths configured by env
系統 MUST 以 env 決定 aneurysm pipeline 需要的路徑（code root / process / json / log / model paths）。

#### Scenario: Env path overrides
- WHEN env 有設定以下任一 key
- THEN 系統 MUST 使用該值覆蓋內建預設值

Required keys (aneurysm):
- `RADX_CODE_ROOT`
- `RADX_PROCESS_ROOT`
- `RADX_JSON_ROOT`
- `RADX_LOG_ROOT`
- `RADX_BRAIN_MODEL`
- `RADX_VESSEL_MODEL`
- `RADX_ANEURYSM_MODEL`

