#!/bin/bash
# Program:
#測試用sh呼叫新的環境執行model stroke predict的運算 0:檔名，1開始為參數
#測試用sh呼叫conda env
# History:
# 2021/09/23 3rd
/home/tmu/anaconda3/envs/tensorflow-infarct/bin/python3.9 /mnt/e/pipeline/chuan/code/gpu_stroke.py --case $1
