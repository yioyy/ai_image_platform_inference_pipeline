#!/bin/bash
# set -euo pipefail

# Follow-up pipeline launcher
# Usage (example):
#   ./pipeline_followup.sh <args...>
#
# 建議以 conda 啟動環境後再執行，或自行調整 conda 路徑：
#   source /home/david/miniconda3/etc/profile.d/conda.sh
#   conda activate followup_env
# 初始化 conda 指令
CONDA_SH="${RADX_CONDA_SH:-/home/david/miniconda3/etc/profile.d/conda.sh}"
# shellcheck source=/dev/null
source "$CONDA_SH"

# 啟動環境
conda activate "${RADX_CONDA_ENV:-tf_2_14}"

python pipeline_followup.py "$@"

# 停用環境
conda deactivate


