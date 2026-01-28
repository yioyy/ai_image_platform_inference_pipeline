#!/usr/bin/env bash
# Program:
#測試用sh呼叫新的環境執行model aneurysm predict的運算 0:檔名，1開始為參數
#測試用sh呼叫conda env
# History:
# 2021/09/23 3rd
# 初始化 conda 指令
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 載入環境設定（test/official 只要換這份檔案即可）
# - 預設讀：<repo>/config/radax.env
# - 或設定 RADX_ENV_FILE 指向 radax-test.env / radax-official.env
ENV_FILE="${RADX_ENV_FILE:-$SCRIPT_DIR/config/radax.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

CONDA_SH="${RADX_CONDA_SH:-/home/tmu/miniconda3/etc/profile.d/conda.sh}"
# shellcheck source=/dev/null
source "$CONDA_SH"

# 啟動環境
conda activate "${RADX_CONDA_ENV:-tf_2_14}"

# 執行你的 python 程式，並傳入參數
PY_BIN="${RADX_PYTHON_BIN:-python}"
PIPELINE_PY="${RADX_PIPELINE_INFARCT_PY:-$SCRIPT_DIR/pipeline_infarct_torch.py}"
$PY_BIN "$PIPELINE_PY" --ID "$1" --Inputs "$2" "$3" "$4" --DicomDir "$5" "$6" "$7" --Output_folder "$8"

# 停用環境
conda deactivate
