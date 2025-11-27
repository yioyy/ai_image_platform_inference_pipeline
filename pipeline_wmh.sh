#!/bin/bash
# Program:
#測試用sh呼叫新的環境執行model aneurysm predict的運算 0:檔名，1開始為參數
#測試用sh呼叫conda env
# History:
# 2021/09/23 3rd
# 初始化 conda 指令
source /home/tmu/miniconda3/etc/profile.d/conda.sh

# 啟動環境
conda activate tf_2_14

# 執行你的 python 程式，並傳入參數
python /data/4TB1/pipeline/chuan/code/pipeline_wmh_torch.py --ID "$1" --Inputs "$2" --DicomDir "$3" --Output_folder "$4"

# 停用環境
conda deactivate 