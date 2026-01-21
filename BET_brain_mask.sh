#!/bin/bash
# Program:
#測試用sh呼叫新的環境執行model stroke predict的運算 0:檔名，1開始為參數
#測試用sh呼叫conda env
# History:
# 2021/09/23 3rd
#source /home/tmu/anaconda3/bin/activate
#conda activate synthseg_38
#python /home/tmu/orthanc_test/orthanc_code/SynthSEG_WMH.py --case $1
source $1 $2
#conda activate synthseg_38
$3 $4 --file $5 --PixelSpacing $6 --SpacingBetweenSlices $7 --BET_iter $8
#conda deactivate
