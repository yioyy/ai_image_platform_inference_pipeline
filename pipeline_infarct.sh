#!/bin/bash
# Program:
# 測試用 sh：呼叫 conda env 後執行 RADAX infarct pipeline（torch/nnUNet flow）
# 參數（與舊版一致）：
# 1: ID
# 2-4: Inputs (ADC DWI0 DWI1000) [+ 可選 SynthSEG]
# 5-7: DicomDir (ADC DWI0 DWI1000)
# 8: Output_folder

set -e

# 初始化 conda 指令（依部署環境調整）
source /home/tmu/miniconda3/etc/profile.d/conda.sh
conda activate tf_2_14

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/pipeline_infarct_torch.py" \
  --ID "$1" \
  --Inputs "$2" "$3" "$4" \
  --DicomDir "$5" "$6" "$7" \
  --Output_folder "$8"

conda deactivate

