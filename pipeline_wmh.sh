#!/bin/bash
# Program:
# 測試用 sh：呼叫 conda env 後執行 RADAX WMH pipeline（torch/nnUNet flow）
# 參數（與舊版一致）：
# 1: ID
# 2: Inputs (T2FLAIR) [+ 可選 SynthSEG]
# 3: DicomDir (T2FLAIR)
# 4: Output_folder

set -e

source /home/tmu/miniconda3/etc/profile.d/conda.sh
conda activate tf_2_14

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/pipeline_wmh_torch.py" \
  --ID "$1" \
  --Inputs "$2" \
  --DicomDir "$3" \
  --Output_folder "$4"

conda deactivate

