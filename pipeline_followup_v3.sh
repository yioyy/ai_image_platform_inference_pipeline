#!/bin/bash
# set -euo pipefail

# Follow-up v3 pipeline launcher
# Usage (example):
#   ./pipeline_followup_v3.sh <args...>
#
# 建議以 conda 啟動環境後再執行，或自行調整 conda 路徑：
#   source /home/tmu/miniconda3/etc/profile.d/conda.sh
#   conda activate followup_env
# 初始化 conda 指令
source /home/tmu/miniconda3/etc/profile.d/conda.sh

# 啟動環境
conda activate tf_2_14

python pipeline_followup_v3.py "$@"

# 停用環境
conda deactivate

