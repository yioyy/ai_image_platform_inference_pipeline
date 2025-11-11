
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

這一版會在1個group預測君彥的結果，另一個group預測nnU-Net的結果

@author: chuan
"""

import warnings
warnings.filterwarnings("ignore") # 忽略警告输出

# 所有import移到最上方，刪除重複與未使用的import
import os
import time
import numpy as np
import logging
import shutil
import json
import pandas as pd
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import matplotlib
import argparse
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import skimage
import skimage.feature
import skimage.measure
from skimage import measure, color, morphology
from mpl_toolkits.axes_grid1 import make_axes_locatable
from random import uniform
from IPython.display import clear_output
import math
from collections import OrderedDict
import matplotlib.colors as mcolors
from gpu_aneurysm import model_predict_aneurysm
import pynvml  # GPU memory info
from util_aneurysm import reslice_nifti_pred_nobrain, create_MIP_pred, AneurysmPipeline, \
    create_dicomseg_multi_file, compress_dicom_into_jpeglossless, orthanc_zip_upload, upload_json_aiteam, \
    decompress_dicom_with_gdcm
import subprocess
from code_ai.pipeline.dicomseg.build_aneurysm import main as make_aneurysm_pred_json
from code_ai.pipeline.dicomseg.build_vessel_dilated import main as make_vessel_pred_json

import pathlib
import requests


#會使用到的一些predict技巧
def data_translate(img, nii):
    img = np.swapaxes(img,0,1)
    img = np.flip(img,0)
    img = np.flip(img, -1)
    header = nii.header.copy() #抓出nii header 去算體積 
    pixdim = header['pixdim']  #可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)  
    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img

def data_translate_back(img, nii):
    header = nii.header.copy() #抓出nii header 去算體積 
    pixdim = header['pixdim']  #可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)      
    img = np.flip(img, -1)
    img = np.flip(img,0)
    img = np.swapaxes(img,1,0)
    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img

#case_json(json_path_name)     
def case_json(json_file_path, ID):
    json_dict = OrderedDict()
    json_dict["PatientID"] = ID #使用的程式是哪一支python api  

    with open(json_file_path, 'w', encoding='utf8') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False) #讓json能中文顯示

def pipeline_aneurysm(ID, 
                      MRA_BRAIN_file,  
                      path_output,
                      path_code = '/mnt/e/pipeline/chuan/code/', 
                      path_nnunet_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset080_DeepAneurysm/nnUNetTrainer__nnUNetPlans__3d_fullres',
                      path_processModel = '/mnt/e/pipeline/chuan/process/Deep_Aneurysm/', 
                      path_outdcm = '',
                      path_json = '/mnt/e/pipeline/chuan/json/',
                      path_log = '/mnt/e/pipeline/chuan/log/', 
                      gpu_n = 0
                      ):

    #當使用gpu有錯時才確認
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    #以log紀錄資訊，先建置log
    localt = time.localtime(time.time()) # 取得 struct_time 格式的時間
    #以下加上時間註記，建立唯一性
    time_str_short = str(localt.tm_year) + str(localt.tm_mon).rjust(2,'0') + str(localt.tm_mday).rjust(2,'0')
    log_file = os.path.join(path_log, time_str_short + '.log')
    if not os.path.isfile(log_file):  #如果log檔不存在
        f = open(log_file, "a+") #a+	可讀可寫	建立，不覆蓋
        f.write("")        #寫入檔案，設定為空
        f.close()      #執行完結束

    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT)

    logging.info('!!! Pre_Aneurysm call.')

    path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)
    if not os.path.isdir(path_processID):  #如果資料夾不存在就建立
        os.mkdir(path_processID) #製作nii資料夾

    print(ID, ' Start...')
    logging.info(ID + ' Start...')

    #依照不同情境拆分try需要小心的事項 <= 重要
    try:
        # %% Deep learning相關
        pynvml.nvmlInit()  # 初始化
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_n)  # 获取GPU i的handle，后续通过handle来处理
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 通过handle获取GPU i的信息
        gpumRate = memoryInfo.used / memoryInfo.total
        # print('gpumRate:', gpumRate) #先設定gpu使用率小於0.2才跑predict code

        if gpumRate < 0.6:
            # plt.ion()    # 開啟互動模式，畫圖都是一閃就過
            # 一些記憶體的配置
            autotune = tf.data.experimental.AUTOTUNE
            # print(keras.__version__)
            # print(tf.__version__)
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            tf.config.experimental.set_visible_devices(devices=gpus[gpu_n], device_type='GPU')
            # print(gpus, cpus)
            tf.config.experimental.set_memory_growth(gpus[gpu_n], True)

            #先判斷有無影像，複製過去
            #print('ADC_file:', ADC_file, ' copy:', os.path.join(path_nii, ID + '_ADC.nii.gz'))
            shutil.copy(MRA_BRAIN_file, os.path.join(path_processID, 'MRA_BRAIN.nii.gz'))
            
            #接下來，以下運行gpu領域
            #以下做predict，為了避免gpu out of memory，還是以.sh來執行好惹
            #gpu_line = 'bash ' + path_code + 'gpu_stroke.sh ' + ID + ' ' + path_processID
            #os.system(gpu_line)

            #所以這裡建立2個資料夾，一個是君彥的模型結果，一個是nnU-Net的模型結果
            path_tensorflow = os.path.join(path_processID, 'tensorflow')
            if not os.path.isdir(path_tensorflow):  #如果資料夾不存在就建立
                os.mkdir(path_tensorflow) #製作tensorflow資料夾
            path_nnunet = os.path.join(path_processID, 'nnUNet')
            if not os.path.isdir(path_nnunet):  #如果資料夾不存在就建立
                os.mkdir(path_nnunet) #製作nnUNet資料夾
            path_nnunetlow = os.path.join(path_processID, 'nnUNetlowth')
            #nnUNetlowth是用來做低閾值的nnU-Net，因為nnU-Net的預測結果有時候會有漏掉的情況，所以要再做一次低閾值的預測
            if not os.path.isdir(path_nnunetlow):  #如果資料夾不存在就建立
                os.mkdir(path_nnunetlow) #製作nnUNet資料夾

            #因為松諭會用排程，但因為ai跟mip都要用到gpu，所以還是要管gpu ram，#multiprocessing沒辦法釋放gpu，要改用subprocess.run()
            #model_predict_aneurysm(path_code, path_processID, ID, path_log, gpu_n)
            print("Running stage 1: Aneurysm inference!!!")
            logging.info("Running stage 1: Aneurysm inference!!!")

            # 定義要傳入的參數，建立指令
            cmd = [
                   "python", "/data/4TB1/pipeline/chuan/code/gpu_aneurysm.py",
                   "--path_code", path_code,
                   "--path_process", path_processID,
                   "--path_nnunet_model", path_nnunet_model,
                   "--case", ID,
                   "--path_log", path_log,
                   "--gpu_n", str(gpu_n)  # 注意要轉成字串
                  ]

            #result = subprocess.run(cmd, capture_output=True, text=True) 這會讓 subprocess.run() 自動幫你捕捉 stdout 和 stderr 的輸出，不然預設是印在 terminal 上，不會儲存。
            # 執行 subprocess
            start = time.time()
            subprocess.run(cmd)
            print(f"[Done AI Inference... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done AI Inference... ] spend {time.time() - start:.0f} sec")

            #接下來做mip影像，這邊要改分別在nnU-Net跟tensorflow的資料夾做mip影像
            # path_dcm_t = os.path.join(path_tensorflow, 'Dicom')
            # path_nii_t = os.path.join(path_tensorflow, 'Image_nii')
            # path_reslice_t = os.path.join(path_tensorflow, 'Image_reslice')
            # path_excel_t = os.path.join(path_tensorflow, 'excel')
            # if not os.path.isdir(path_dcm_t):  #如果資料夾不存在就建立
            #     os.mkdir(path_dcm_t) #製作nii資料夾
            # if not os.path.isdir(path_nii_t):  #如果資料夾不存在就建立
            #     os.mkdir(path_nii_t) #製作nii資料夾
            # if not os.path.isdir(path_reslice_t):  #如果資料夾不存在就建立
            #     os.mkdir(path_reslice_t) #製作nii資料夾
            # if not os.path.isdir(path_excel_t):  #如果資料夾不存在就建立
            #     os.mkdir(path_excel_t) #製作nii資料夾

            path_dcm_n = os.path.join(path_nnunet, 'Dicom')
            path_nii_n = os.path.join(path_nnunet, 'Image_nii')
            path_reslice_n = os.path.join(path_nnunet, 'Image_reslice')
            path_excel_n = os.path.join(path_nnunet, 'excel')
            if not os.path.isdir(path_dcm_n):  #如果資料夾不存在就建立
                os.mkdir(path_dcm_n) #製作nii資料夾
            if not os.path.isdir(path_nii_n):  #如果資料夾不存在就建立
                os.mkdir(path_nii_n) #製作nii資料夾
            if not os.path.isdir(path_reslice_n):  #如果資料夾不存在就建立
                os.mkdir(path_reslice_n) #製作nii資料夾
            if not os.path.isdir(path_excel_n):  #如果資料夾不存在就建立
                os.mkdir(path_excel_n) #製作nii資料夾

            # path_dcm_nl = os.path.join(path_nnunetlow, 'Dicom')
            # path_nii_nl = os.path.join(path_nnunetlow, 'Image_nii')
            # path_reslice_nl = os.path.join(path_nnunetlow, 'Image_reslice')
            # path_excel_nl = os.path.join(path_nnunetlow, 'excel')
            # if not os.path.isdir(path_dcm_nl):  #如果資料夾不存在就建立
            #     os.mkdir(path_dcm_nl) #製作nii資料夾
            # if not os.path.isdir(path_nii_nl):  #如果資料夾不存在就建立
            #     os.mkdir(path_nii_nl) #製作nii資料夾
            # if not os.path.isdir(path_reslice_nl):  #如果資料夾不存在就建立
            #     os.mkdir(path_reslice_nl) #製作nii資料夾
            # if not os.path.isdir(path_excel_nl):  #如果資料夾不存在就建立
            #     os.mkdir(path_excel_nl) #製作nii資料夾

            #複製nii到nii資料夾
            # shutil.copy(os.path.join(path_processID, 'MRA_BRAIN.nii.gz'), os.path.join(path_tensorflow, 'MRA_BRAIN.nii.gz'))
            # shutil.copy(os.path.join(path_processID, 'MRA_BRAIN.nii.gz'), os.path.join(path_nii_t, 'MRA_BRAIN.nii.gz'))
            # shutil.copy(os.path.join(path_tensorflow, 'Pred.nii.gz'), os.path.join(path_nii_t, 'Pred.nii.gz'))
            # shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'), os.path.join(path_nii_t, 'Vessel.nii.gz'))

            shutil.copy(os.path.join(path_processID, 'MRA_BRAIN.nii.gz'), os.path.join(path_nnunet, 'MRA_BRAIN.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'MRA_BRAIN.nii.gz'), os.path.join(path_nii_n, 'MRA_BRAIN.nii.gz'))
            shutil.copy(os.path.join(path_nnunet, 'Pred.nii.gz'), os.path.join(path_nii_n, 'Pred.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'), os.path.join(path_nii_n, 'Vessel.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'), os.path.join(path_nnunet, 'Vessel.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel_16.nii.gz'), os.path.join(path_nnunet, 'Vessel_16.nii.gz'))

            # shutil.copy(os.path.join(path_processID, 'MRA_BRAIN.nii.gz'), os.path.join(path_nnunetlow, 'MRA_BRAIN.nii.gz'))
            # shutil.copy(os.path.join(path_processID, 'MRA_BRAIN.nii.gz'), os.path.join(path_nii_nl, 'MRA_BRAIN.nii.gz'))
            # shutil.copy(os.path.join(path_nnunetlow, 'Pred.nii.gz'), os.path.join(path_nii_nl, 'Pred.nii.gz'))
            # shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'), os.path.join(path_nii_nl, 'Vessel.nii.gz'))
            # shutil.copy(os.path.join(path_processID, 'Vessel_16.nii.gz'), os.path.join(path_nnunetlow, 'Vessel_16.nii.gz'))

            #reslice，所有都要做2次
            start_reslice = time.time()
            # reslice_nifti_pred_nobrain(path_nii_t, path_reslice_t)
            reslice_nifti_pred_nobrain(path_nii_n, path_reslice_n)
            # reslice_nifti_pred_nobrain(path_nii_nl, path_reslice_nl)
            print(f"[Done reslice... ] spend {time.time() - start_reslice:.0f} sec")
            logging.info(f"[Done reslice... ] spend {time.time() - start_reslice:.0f} sec")

            #做mip影像
            #複製dicom影像
            # if not os.path.isdir(os.path.join(path_dcm_t, 'MRA_BRAIN')):  #如果資料夾不存在就建立
            #     shutil.copytree(path_outdcm, os.path.join(path_dcm_t, 'MRA_BRAIN'))
            if not os.path.isdir(os.path.join(path_dcm_n, 'MRA_BRAIN')):  #如果資料夾不存在就建立
                shutil.copytree(path_outdcm, os.path.join(path_dcm_n, 'MRA_BRAIN'))
            # if not os.path.isdir(os.path.join(path_dcm_nl, 'MRA_BRAIN')):  #如果資料夾不存在就建立
            #     shutil.copytree(path_outdcm, os.path.join(path_dcm_nl, 'MRA_BRAIN'))   

            #這邊對dicom進行解壓縮動作
            #執行gdcmconv 去還原影像後覆蓋
            start_decompress_JEPG = time.time()
            decompress_dicom_with_gdcm(path_dcm_n)
            print(f"[Done decompress_JEPG... ] spend {time.time() - start_decompress_JEPG:.0f} sec")
            logging.info(f"[Done decompress_JEPG... ] spend {time.time() - start_decompress_JEPG:.0f} sec")


            path_png = os.path.join(path_code, 'png')
            start_mip = time.time()
            # create_MIP_pred(path_dcm_t, path_reslice_t, path_png, gpu_n)
            create_MIP_pred(path_dcm_n, path_reslice_n, path_png, gpu_n)
            # create_MIP_pred(path_dcm_nl, path_reslice_nl, path_png, gpu_n)
            print(f"[Done create_MIP_pred... ] spend {time.time() - start_mip:.0f} sec")
            logging.info(f"[Done create_MIP_pred... ] spend {time.time() - start_mip:.0f} sec")

            shutil.copy(os.path.join(path_reslice_n, 'MIP_Pitch_pred.nii.gz'), os.path.join(path_nnunet, 'MIP_Pitch_pred.nii.gz'))
            shutil.copy(os.path.join(path_reslice_n, 'MIP_Yaw_pred.nii.gz'), os.path.join(path_nnunet, 'MIP_Yaw_pred.nii.gz'))

            #接下來是計算動脈瘤的各項數據，各做一次
            start_calculate = time.time()
            # aneurysm_analysis_pipeline = AneurysmPipeline(path_dcm_t, path_tensorflow, path_excel_t, ID)
            # aneurysm_analysis_pipeline.run_all()
            aneurysm_analysis_pipeline = AneurysmPipeline(path_dcm_n, path_nnunet, path_excel_n, ID)
            aneurysm_analysis_pipeline.run_all()
            #aneurysm_analysis_pipeline = AneurysmPipeline(path_dcm_nl, path_nnunetlow, path_excel_nl, ID)
            #aneurysm_analysis_pipeline.run_all()
            print(f"[Done calculate_aneurysm_long_axis... ] spend {time.time() - start_calculate:.0f} sec")
            logging.info(f"[Done calculate_aneurysm_long_axis... ] spend {time.time() - start_calculate:.0f} sec")

            #接下來製作dicom-seg
            # path_dicomseg_t = os.path.join(path_dcm_t, 'Dicom-Seg')
            # if not os.path.isdir(path_dicomseg_t):  #如果資料夾不存在就建立
            #     os.mkdir(path_dicomseg_t) #製作nii資料夾
            path_dicomseg_n = os.path.join(path_dcm_n, 'Dicom-Seg')
            if not os.path.isdir(path_dicomseg_n):  #如果資料夾不存在就建立
                os.mkdir(path_dicomseg_n) #製作nii資料夾
            # path_dicomseg_nl = os.path.join(path_dcm_nl, 'Dicom-Seg')
            # if not os.path.isdir(path_dicomseg_nl):  #如果資料夾不存在就建立
            #     os.mkdir(path_dicomseg_nl) #製作nii資料夾    

            #上線版不用做vessel
            start_dicomseg = time.time()
            # create_dicomseg_multi_file(path_code, path_dcm_t, path_nii_t, path_reslice_t, path_dicomseg_t, ID)
            #create_dicomseg_multi_file(path_code, path_dcm_n, path_nii_n, path_reslice_n, path_dicomseg_n, ID)
            #create_dicomseg_multi_file(path_code, path_dcm_nl, path_nii_nl, path_reslice_nl, path_dicomseg_nl, ID)
            print(f"[Done create_dicomseg... ] spend {time.time() - start_dicomseg:.0f} sec")
            logging.info(f"[Done create_dicomseg... ] spend {time.time() - start_dicomseg:.0f} sec")

            #將dicom壓縮不包含dicom-seg  Dicom_JPEGlossless => 由於要用numpy > 2.0，之後補強
            # path_dcmjpeglossless_t = os.path.join(path_tensorflow, 'Dicom_JPEGlossless')
            # if not os.path.isdir(path_dcmjpeglossless_t):  #如果資料夾不存在就建立
            #     os.mkdir(path_dcmjpeglossless_t) #製作nii資料夾
            path_dcmjpeglossless_n = os.path.join(path_nnunet, 'Dicom_JPEGlossless')
            if not os.path.isdir(path_dcmjpeglossless_n):  #如果資料夾不存在就建立
                os.mkdir(path_dcmjpeglossless_n) #製作nii資料夾
            # path_dcmjpeglossless_nl = os.path.join(path_nnunetlow, 'Dicom_JPEGlossless')
            # if not os.path.isdir(path_dcmjpeglossless_nl):  #如果資料夾不存在就建立
            #     os.mkdir(path_dcmjpeglossless_nl) #製作nii資料夾            
            #compress_dicom_into_jpeglossless(path_dcm_t, path_dcmjpeglossless_t)
            #compress_dicom_into_jpeglossless(path_dcm_n, path_dcmjpeglossless_n)
            #compress_dicom_into_jpeglossless(path_dcm_nl, path_dcmjpeglossless_nl)

            #建立json檔
            # path_json_out_t = os.path.join(path_tensorflow, 'JSON')
            # if not os.path.isdir(path_json_out_t):  #如果資料夾不存在就建立
            #     os.mkdir(path_json_out_t) #製作nii資料夾
            path_json_out_n = os.path.join(path_nnunet, 'JSON')
            if not os.path.isdir(path_json_out_n):  #如果資料夾不存在就建立
                os.mkdir(path_json_out_n) #製作nii資料夾
            # path_json_out_nl = os.path.join(path_nnunetlow, 'JSON')
            # if not os.path.isdir(path_json_out_nl):  #如果資料夾不存在就建立
            #     os.mkdir(path_json_out_nl) #製作nii資料夾

            #複製tensorflow的資料夾成multimodel資料夾，給後面新生成的group_id使用
            # path_multimodel = os.path.join(path_processID, 'multimodel')
            # if not os.path.isdir(path_multimodel):  
            #     shutil.copytree(path_tensorflow, path_multimodel) #資料夾不存在就複製
            # path_json_out_mu = os.path.join(path_multimodel, 'JSON')
            # if not os.path.isdir(path_json_out_mu):  #如果資料夾不存在就建立
            #     os.mkdir(path_json_out_mu) #製作nii資料夾
            
            # path_dcm_mu = os.path.join(path_multimodel, 'Dicom')
            # if not os.path.isdir(path_dcm_mu):  #如果資料夾不存在就建立
            #     os.mkdir(path_dcm_mu) #製作nii資料夾

            Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw']
            group_id1 = 53 # 君彥的模型
            group_id2 = 54 # nnU-Net的模型
            group_id3 = 55 # nnU-Net的模型，低閾值
            group_id4 = 56 # 君彥的模型，但會跟CMB在後端合併，所以額外拉一個group_id
            model_id = '924d1538-597c-41d6-bc27-4b0b359111cf'
            start_json = time.time()
            # make_pred_json(ID, pathlib.Path(path_tensorflow), group_id1)  #(_id, path_root, group_id)
            make_aneurysm_pred_json(ID, pathlib.Path(path_nnunet), model_id)  #(_id, path_root, group_id)
            # make_pred_json(ID, pathlib.Path(path_nnunetlow), group_id3)  #(_id, path_root, group_id)
            # make_pred_json(ID, pathlib.Path(path_multimodel), group_id4)  #(_id, path_root, group_id)
            #make_pred_json(excel_file, path_dcm, path_nii, path_reslice, path_dicomseg, path_json_out, [ID], Series, group_id)
            print(f"[Done make_pred_json... ] spend {time.time() - start_json:.0f} sec")
            logging.info(f"[Done make_pred_json... ] spend {time.time() - start_json:.0f} sec")

            #下面製作vessel dilate的dicom-seg跟json
            make_vessel_pred_json(ID, pathlib.Path(path_nnunet))

            #接下來上傳dicom到orthanc
            # path_zip_t = os.path.join(path_tensorflow, 'Dicom_zip')
            # if not os.path.isdir(path_zip_t):
            #     os.mkdir(path_zip_t)
            path_zip_n = os.path.join(path_nnunet, 'Dicom_zip')
            if not os.path.isdir(path_zip_n):
                os.mkdir(path_zip_n)
            # path_zip_nl = os.path.join(path_nnunetlow, 'Dicom_zip')
            # if not os.path.isdir(path_zip_nl): #如果資料夾不存在就建立
            #     os.mkdir(path_zip_nl)  
            # path_zip_mu = os.path.join(path_multimodel, 'Dicom_zip')
            # if not os.path.isdir(path_zip_mu):  #如果資料夾不存在就建立
            #     os.mkdir(path_zip_mu) #製作nii資料夾    

            Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw', 'Dicom-Seg']
            start_upload_json = time.time()
            # orthanc_zip_upload(path_dcm_t, path_zip_t, Series)
            #orthanc_zip_upload(path_dcm_n, path_zip_n, Series)
            # orthanc_zip_upload(path_dcm_nl, path_zip_nl, Series)
            # orthanc_zip_upload(path_dcm_mu, path_zip_mu, Series)
            
            print(f"[Done start_upload_json... ] spend {time.time() - start_upload_json:.0f} sec")
            logging.info(f"[Done start_upload_json... ] spend {time.time() - start_upload_json:.0f} sec")

            #接下來，上傳json
            # json_file_t = os.path.join(path_json_out_t, ID + '_platform_json.json')
            # json_file_n = os.path.join(path_json_out_n, ID + '_platform_json.json')
            # json_file_nl = os.path.join(path_json_out_nl, ID + '_platform_json.json')
            # json_file_mu = os.path.join(path_json_out_mu, ID + '_platform_json.json')
            # upload_json_aiteam(json_file_t)
            # upload_json_aiteam(json_file_n)
            # upload_json_aiteam(json_file_nl)
            # upload_json_aiteam(json_file_mu)

            #把json跟nii輸出到out資料夾，這裡只傳nnunet的結果，因為nnU-Net的結果比較好
            # shutil.copy(os.path.join(path_nnunet, 'Pred.nii.gz'), os.path.join(path_output, 'Pred_Aneurysm.nii.gz'))
            # shutil.copy(os.path.join(path_nnunet, 'Prob.nii.gz'), os.path.join(path_output, 'Prob_Aneurysm.nii.gz'))
            # shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'), os.path.join(path_output, 'Pred_Aneurysm_Vessel.nii.gz'))
            # shutil.copy(os.path.join(path_processID, 'Vessel_16.nii.gz'), os.path.join(path_output, 'Pred_Aneurysm_Vessel16.nii.gz'))
            # shutil.copy(os.path.join(path_json_out_n, ID + '_platform_json.json'), os.path.join(path_output, 'Pred_Aneurysm.json'))
            # shutil.copy(os.path.join(path_json_out_n, ID + '_platform_json.json'), os.path.join(path_output, 'Pred_Aneurysm_platform_json.json'))

            #radax步驟，接下來完成複製檔案到指定資料夾跟打api通知
            upload_dir = '/home/david/ai-inference-result' #目的資料夾
            aneurysm_json_file = os.path.join(path_nnunet, 'rdx_aneurysm_pred_json.json')
            vessel_json_file = os.path.join(path_nnunet, 'rdx_vessel_dilated_json.json')
            
            # 讀取aneurysm_json_file獲取study_instance_uid和series_instance_uid
            with open(aneurysm_json_file, 'r', encoding='utf-8') as f:
                aneurysm_data = json.load(f)
            
            study_instance_uid = aneurysm_data['study_instance_uid']
            series_instance_uid = aneurysm_data['series_instance_uid']
            
            # 在upload_dir建立study_instance_uid資料夾
            study_dir = os.path.join(upload_dir, study_instance_uid)
            if not os.path.isdir(study_dir):
                os.makedirs(study_dir)
            
            # 建立aneurysm_model資料夾
            aneurysm_model_dir = os.path.join(study_dir, 'aneurysm_model')
            if not os.path.isdir(aneurysm_model_dir):
                os.makedirs(aneurysm_model_dir)
            
            # 複製MRA_BRAIN_A*.dcm檔案到aneurysm_model資料夾
            aneurysm_files = sorted([f for f in os.listdir(path_dicomseg_n) if f.startswith('MRA_BRAIN_A') and f.endswith('.dcm')])
            for aneurysm_file in aneurysm_files:
                # 提取編號 (例如: MRA_BRAIN_A1.dcm -> A1)
                label = aneurysm_file.replace('MRA_BRAIN_', '').replace('.dcm', '')
                # 建立新檔名
                new_filename = f'{series_instance_uid}_{label}.dcm'
                src_path = os.path.join(path_dicomseg_n, aneurysm_file)
                dst_path = os.path.join(aneurysm_model_dir, new_filename)
                shutil.copy(src_path, dst_path)
                print(f"[Copy] {aneurysm_file} -> {new_filename}")
                logging.info(f"[Copy] {aneurysm_file} -> {new_filename}")
            
            # 複製aneurysm_json_file到aneurysm_model資料夾並改名
            aneurysm_json_dst = os.path.join(aneurysm_model_dir, f'{series_instance_uid}.json')
            shutil.copy(aneurysm_json_file, aneurysm_json_dst)
            print(f"[Copy] rdx_aneurysm_pred_json.json -> {series_instance_uid}.json")
            logging.info(f"[Copy] rdx_aneurysm_pred_json.json -> {series_instance_uid}.json")
            
            # 建立vessel_model資料夾
            vessel_model_dir = os.path.join(study_dir, 'vessel_model')
            if not os.path.isdir(vessel_model_dir):
                os.makedirs(vessel_model_dir)
            
            # 複製MRA_BRAIN_Vessel_A1.dcm到vessel_model資料夾並改名
            vessel_file = 'MRA_BRAIN_Vessel_A1.dcm'
            vessel_src_path = os.path.join(path_dicomseg_n, vessel_file)
            vessel_dst_path = os.path.join(vessel_model_dir, f'{series_instance_uid}.dcm')
            if os.path.exists(vessel_src_path):
                shutil.copy(vessel_src_path, vessel_dst_path)
                print(f"[Copy] {vessel_file} -> {series_instance_uid}.dcm")
                logging.info(f"[Copy] {vessel_file} -> {series_instance_uid}.dcm")
            
            # 複製vessel_json_file到vessel_model資料夾並改名
            vessel_json_dst = os.path.join(vessel_model_dir, f'{series_instance_uid}.json')
            shutil.copy(vessel_json_file, vessel_json_dst)
            print(f"[Copy] rdx_vessel_pred_json.json -> {series_instance_uid}.json")
            logging.info(f"[Copy] rdx_vessel_pred_json.json -> {series_instance_uid}.json")
            
            # 發送POST請求到 /v1/ai-inference/inference-complete
            # 請修改為正確的 API 端點，例如：
            # api_url = 'http://localhost:8080/v1/ai-inference/inference-complete'
            # api_url = 'http://10.103.1.193:3000/v1/ai-inference/inference-complete'
            api_url = 'http://localhost:4000/v1/ai-inference/inference-complete'  # TODO: 請修改為正確的 API 端點
            payload = {
                "studyInstanceUid": study_instance_uid,
                "seriesInstanceUid": series_instance_uid
            }
            try:
                response = requests.post(api_url, json=payload)
                print(f"[API POST] Status Code: {response.status_code}, Response: {response.text}")
                logging.info(f"[API POST] Status Code: {response.status_code}, Response: {response.text}")
            except Exception as e:
                print(f"[API POST Error] {str(e)}")
                logging.error(f"[API POST Error] {str(e)}")

            #刪除資料夾
            # if os.path.isdir(path_process):  #如果資料夾存在
            #     shutil.rmtree(path_process) #清掉整個資料夾 

            print(f"[Done All Pipeline!!! ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done All Pipeline!!! ] spend {time.time() - start:.0f} sec")
            logging.info('!!! ' + ID +  ' post_aneurysm finish.')
        
        else:
            logging.error('!!! ' + str(ID) + ' Insufficient GPU Memory.')
            # 以json做輸出
            code_pass = 1
            msg = "Insufficient GPU Memory"

            # #刪除資料夾
            # if os.path.isdir(path_process):  #如果資料夾存在
            #     shutil.rmtree(path_process) #清掉整個資料夾

    except Exception:
        logging.error('!!! ' + str(ID) + ' gpu have error code.')
        logging.error("Catch an exception.", exc_info=True)
        # 刪除資料夾
        # if os.path.isdir(path_process):  #如果資料夾存在
        #     shutil.rmtree(path_process) #清掉整個資料夾
   
    print('end!!!')
    return

#其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=str, default = '17390820_20250604_MR_21406040004', help='目前執行的case的patient_id or study id')
    parser.add_argument('--Inputs', type=str, nargs='+', default = ['/data/4TB1/pipeline/chuan/example_input/17390820_20250604_MR_21406040004/MRA_BRAIN.nii.gz'], help='用於輸入的檔案')
    parser.add_argument('--DicomDir', type=str, nargs='+', default = ['/data/4TB1/pipeline/chuan/example_inputDicom/17390820_20250604_MR_21406040004/MRA_BRAIN/'], help='用於輸入的檔案')
    parser.add_argument('--Output_folder', type=str, default = '/data/4TB1/pipeline/chuan/example_output/',help='用於輸出結果的資料夾')    
    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs  # 將列表合併為字符串，保留順序
    DicomDirs = args.DicomDir #對應的dicom資料夾，用來做dicom-seg
    path_output = str(args.Output_folder)

    #讀出DWI, DWI0, ADC, SynthSEG的檔案
    MRA_BRAIN_file = Inputs[0]
    path_DcmDir = DicomDirs[0] #這邊在做dicom-seg時才會用到

    #需要安裝 pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg => 先不壓縮，因為壓縮需要numpy > 2

    #下面設定各個路徑
    path_code = '/home/david/pipeline/chuan/code/'
    path_process = '/home/david/pipeline/chuan/process/'  #前處理dicom路徑(test case)
    path_nnunet_model = '/home/david/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset080_DeepAneurysm/nnUNetTrainer__nnUNetPlans__3d_fullres'
    path_processModel = os.path.join(path_process, 'Deep_Aneurysm')  #前處理dicom路徑(test case)
    #path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)

    #這裡先沒有dicom
    path_json = '/home/david/pipeline/chuan/json/'  #存放json的路徑，回傳執行結果
    #json_path_name = os.path.join(path_json, 'Pred_Infarct.json')
    path_log = '/home/david/pipeline/chuan/log/'  #log資料夾

    #自訂模型
    gpu_n = 0  #使用哪一顆gpu

    # 建置資料夾
    os.makedirs(path_processModel, exist_ok=True) # 如果資料夾不存在就建立，製作nii資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_output,exist_ok=True)

    #直接當作function的輸入，因為可能會切換成nnUNet的版本，所以自訂化模型移到跟model一起，synthseg自己做，不用統一
    pipeline_aneurysm(ID, MRA_BRAIN_file, path_output, path_code, path_nnunet_model, path_processModel, path_DcmDir, path_json, path_log, gpu_n)
    

    # #最後再讀取json檔結果
    # with open(json_path_name) as f:
    #     data = json.load(f)

    # logging.info('Json!!! ' + str(data))
