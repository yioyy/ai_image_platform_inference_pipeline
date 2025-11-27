#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

predict model，非常重要:使用這個程式，安裝的版本都需要一致，才能運行，目前wmh可以用跟stroke相同環境，不知道為什麼
針對wmh環境的安裝直接用以下指令，如果是一開始不用以下指令安裝然後再改tensorflow，就都裝不起來了 !!!
conda create -n orthanc-wmh tensorflow-gpu=2.1.0 anaconda python=3.7
在這狀態下keras要裝 pip install keras==2.3.1
@author: chuan
"""
import warnings
warnings.filterwarnings("ignore") # 忽略警告输出，對tensorflow-gpu沒有用啊


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #有用，但還是會剩Using TensorFlow backend.
import shutil
import glob
import time
import numpy as np
# from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
import argparse


#其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, help='目前執行的case的study_uid')
    args = parser.parse_args()

    case_name = str(args.case)

    #下面設定各個路徑
    path_code = '/home/tmu/orthanc_test/orthanc_code/'
    path_synthseg = '/home/tmu/SynthSeg_parcellation/code/'
    path_process = os.path.join('/home/tmu/orthanc_test/orthanc_process/stroke/', case_name)  #前處理dicom路徑(test case)
    path_nii = os.path.join(path_process, 'dicom', 'nii')    
    
    gpu_n = 0  #使用哪一顆gpu

    #以下做predict，為了避免gpu out of memory，還是以.sh來執行好惹
    gpu_line = 'python ' + os.path.join(path_synthseg, 'main.py -i ') + path_nii + ' --input_name DWI0.nii.gz --template ' + path_nii + ' --template_name DWI0 --all False --DWI TRUE'
    os.system(gpu_line)
