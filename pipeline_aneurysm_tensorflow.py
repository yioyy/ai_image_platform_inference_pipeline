
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
import sys
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

def _maybe_set_cuda_visible_from_argv() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    gpu_n = None
    if "--gpu_n" in sys.argv:
        idx = sys.argv.index("--gpu_n")
        if idx + 1 < len(sys.argv):
            gpu_n = sys.argv[idx + 1]
    else:
        for arg in sys.argv:
            if arg.startswith("--gpu_n="):
                gpu_n = arg.split("=", 1)[1]
                break
    if gpu_n is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_n).strip()

_maybe_set_cuda_visible_from_argv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
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
    decompress_dicom_with_gdcm, resampleSynthSEG2original
import subprocess
import tempfile

from code_ai.pipeline.dicomseg.aneurysm import main as make_pred_json
from pipeline_followup_v3_platform import pipeline_followup_v3_platform

# from code_ai.pipeline.dicomseg.build_aneurysm import main as make_aneurysm_pred_json
# from code_ai.pipeline.dicomseg.build_vessel_dilated import main as make_vessel_pred_json

import pathlib
import requests


# Optional env-file loader (for test vs official)
try:
    from config.load_env import load_env_from_default_locations
except Exception:
    load_env_from_default_locations = None


# 讀取環境變數時的 path 清理
def _clean_env_path(p: str) -> str:
    return os.path.normpath(str(p).strip().replace("\r", "").replace("\n", ""))


def _env_path(key: str, default: str) -> str:
    v = os.getenv(key, "").strip()
    return _clean_env_path(v) if v else _clean_env_path(default)


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
                      path_brain_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset134_DeepMRABrain/nnUNetTrainer__nnUNetPlans__3d_fullres',
                      path_vessel_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset135_DeepMRAVessel/nnUNetTrainer__nnUNetPlans__3d_fullres',
                      path_aneurysm_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset080_DeepAneurysm/nnUNetTrainer__nnUNetPlans__3d_fullres',
                      path_processModel = '/mnt/e/pipeline/chuan/process/Deep_Aneurysm/', 
                      path_outdcm = '',
                      input_json = '',
                      path_json = '/mnt/e/pipeline/chuan/json/',
                      path_log = '/mnt/e/pipeline/chuan/log/', 
                      gpu_n = 0
                      ):
    # ---- path hygiene ----
    # 常見問題：有些上游會把 Windows 換行符 \r\n 帶進參數，導致路徑變成 ".../\r/xxx"
    def _clean_path(p: str) -> str:
        return os.path.normpath(str(p).strip().replace("\r", "").replace("\n", ""))

    ID = str(ID).strip().replace("\r", "").replace("\n", "")
    MRA_BRAIN_file = _clean_path(MRA_BRAIN_file)
    path_output = _clean_path(path_output)
    path_code = _clean_path(path_code)
    path_brain_model = _clean_path(path_brain_model)
    path_vessel_model = _clean_path(path_vessel_model)
    path_aneurysm_model = _clean_path(path_aneurysm_model)
    path_processModel = _clean_path(path_processModel)
    path_outdcm = _clean_path(path_outdcm)
    input_json = str(input_json).strip().replace("\r", "").replace("\n", "")
    path_json = _clean_path(path_json)
    path_log = _clean_path(path_log)

    # 確保輸出資料夾存在，避免後面 shutil.copy 直接 FileNotFoundError
    os.makedirs(path_output, exist_ok=True)

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
            if not gpus:
                raise RuntimeError("No GPU devices found.")
            # 如果已設定 CUDA_VISIBLE_DEVICES，gpus 只會看到那一顆，需用 index 0
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                visible_idx = 0
            else:
                visible_idx = gpu_n
            if visible_idx >= len(gpus):
                visible_idx = 0
            tf.config.experimental.set_visible_devices(devices=gpus[visible_idx], device_type='GPU')
            # print(gpus, cpus)
            tf.config.experimental.set_memory_growth(gpus[visible_idx], True)

            #先判斷有無影像，複製過去
            #print('ADC_file:', ADC_file, ' copy:', os.path.join(path_nii, ID + '_ADC.nii.gz'))
            shutil.copy(MRA_BRAIN_file, os.path.join(path_processID, 'MRA_BRAIN.nii.gz'))
            
            #接下來，以下運行gpu領域
            #以下做predict，為了避免gpu out of memory，還是以.sh來執行好惹
            #gpu_line = 'bash ' + path_code + 'gpu_stroke.sh ' + ID + ' ' + path_processID
            #os.system(gpu_line)

            #所以這裡建立2個資料夾，一個是君彥的模型結果，一個是nnU-Net的模型結果
            path_nnunet = os.path.join(path_processID, 'nnUNet')
            if not os.path.isdir(path_nnunet):  #如果資料夾不存在就建立
                os.mkdir(path_nnunet) #製作nnUNet資料夾

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

            #因為要跑SynthSeg，所以先複製MRA_BRAIN.nii.gz過去
            shutil.copy(os.path.join(path_processID, 'MRA_BRAIN.nii.gz'), os.path.join(path_nnunet, 'MRA_BRAIN.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'MRA_BRAIN.nii.gz'), os.path.join(path_nii_n, 'MRA_BRAIN.nii.gz'))

            #因為松諭會用排程，但因為ai跟mip都要用到gpu，所以還是要管gpu ram，#multiprocessing沒辦法釋放gpu，要改用subprocess.run()
            #model_predict_aneurysm(path_code, path_processID, ID, path_log, gpu_n)
            print("Running stage 1: Aneurysm inference!!!")
            logging.info("Running stage 1: Aneurysm inference!!!")

            # 定義要傳入的參數，建立指令
            cmd = [
                   "python", os.path.join(path_code, "gpu_aneurysm.py"),
                   "--path_code", path_code,
                   "--path_process", path_processID,
                   "--path_brain_model", path_brain_model,
                   "--path_vessel_model", path_vessel_model,
                   "--path_aneurysm_model", path_aneurysm_model,
                   "--case", ID,
                   "--path_log", path_log,
                   "--gpu_n", str(gpu_n)  # 注意要轉成字串
                  ]

            #result = subprocess.run(cmd, capture_output=True, text=True) 這會讓 subprocess.run() 自動幫你捕捉 stdout 和 stderr 的輸出，不然預設是印在 terminal 上，不會儲存。
            # 執行 subprocess
            start = time.time()
            gpu_env = os.environ.copy()
            gpu_env["CUDA_VISIBLE_DEVICES"] = str(gpu_n)
            subprocess.run(cmd, env=gpu_env)
            print(f"[Done stage 1: Aneurysm AI Inference... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done stage 1: Aneurysm AI Inference... ] spend {time.time() - start:.0f} sec")

            #這裡再針對MRA_Brain跑一次SynthSeg，為了後續做前後比較的模組
            path_synthseg = os.path.join(path_code, 'model_weights','SynthSeg_parcellation_tf28','code')

            cmd = [
                "python", os.path.join(path_synthseg, 'main_tf28.py'),
                "-i", path_nii_n,
                "--input_name", 'MRA_BRAIN.nii.gz',
                "--all", "False",
                ]

            #result = subprocess.run(cmd, capture_output=True, text=True) 這會讓 subprocess.run() 自動幫你捕捉 stdout 和 stderr 的輸出，不然預設是印在 terminal 上，不會儲存。
            # 執行 subprocess
            start = time.time()
            synthseg_env = os.environ.copy()
            synthseg_env["CUDA_VISIBLE_DEVICES"] = str(gpu_n)
            subprocess.run(cmd, env=synthseg_env)
            print(f"[Done SynthSEG AI Inference... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done SynthSEG AI Inference... ] spend {time.time() - start:.0f} sec")

            #這邊還要做resample到original space
            SynthSEG_array = resampleSynthSEG2original(path_nii_n, 'MRA_BRAIN', 'synthseg33') 
            shutil.copy(
                os.path.join(path_nii_n, "NEW_MRA_BRAIN_synthseg33.nii.gz"),
                os.path.join(path_nnunet, "SynthSEG.nii.gz"),
            )

            shutil.copy(os.path.join(path_nnunet, 'Pred.nii.gz'), os.path.join(path_nii_n, 'Pred.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'), os.path.join(path_nii_n, 'Vessel.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'), os.path.join(path_nnunet, 'Vessel.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel_16.nii.gz'), os.path.join(path_nnunet, 'Vessel_16.nii.gz'))

            #reslice，所有都要做2次
            start_reslice = time.time()
            reslice_nifti_pred_nobrain(path_nii_n, path_reslice_n)
            print(f"[Done reslice... ] spend {time.time() - start_reslice:.0f} sec")
            logging.info(f"[Done reslice... ] spend {time.time() - start_reslice:.0f} sec")

            #做mip影像
            #複製dicom影像
            if not os.path.isdir(os.path.join(path_dcm_n, 'MRA_BRAIN')):  #如果資料夾不存在就建立
                shutil.copytree(path_outdcm, os.path.join(path_dcm_n, 'MRA_BRAIN'))

            #這邊對dicom進行解壓縮動作
            #執行gdcmconv 去還原影像後覆蓋
            start_decompress_JEPG = time.time()
            decompress_dicom_with_gdcm(path_dcm_n)
            print(f"[Done decompress_JEPG... ] spend {time.time() - start_decompress_JEPG:.0f} sec")
            logging.info(f"[Done decompress_JEPG... ] spend {time.time() - start_decompress_JEPG:.0f} sec")

            path_png = os.path.join(path_code, 'png')
            start_mip = time.time()
            create_MIP_pred(path_dcm_n, path_reslice_n, path_png, gpu_n)
            print(f"[Done create_MIP_pred... ] spend {time.time() - start_mip:.0f} sec")
            logging.info(f"[Done create_MIP_pred... ] spend {time.time() - start_mip:.0f} sec")

            shutil.copy(os.path.join(path_reslice_n, 'MIP_Pitch_pred.nii.gz'), os.path.join(path_nnunet, 'MIP_Pitch_pred.nii.gz'))
            shutil.copy(os.path.join(path_reslice_n, 'MIP_Yaw_pred.nii.gz'), os.path.join(path_nnunet, 'MIP_Yaw_pred.nii.gz'))

            #接下來是計算動脈瘤的各項數據，各做一次
            start_calculate = time.time()
            aneurysm_analysis_pipeline = AneurysmPipeline(path_dcm_n, path_nnunet, path_excel_n, ID)
            aneurysm_analysis_pipeline.run_all()
            print(f"[Done calculate_aneurysm_long_axis... ] spend {time.time() - start_calculate:.0f} sec")
            logging.info(f"[Done calculate_aneurysm_long_axis... ] spend {time.time() - start_calculate:.0f} sec")

            #接下來製作dicom-seg
            path_dicomseg_n = os.path.join(path_dcm_n, 'Dicom-Seg')
            if not os.path.isdir(path_dicomseg_n):  #如果資料夾不存在就建立
                os.mkdir(path_dicomseg_n) #製作nii資料夾   

            #上線版不用做vessel
            # start_dicomseg = time.time()
            # print(f"[Done create_dicomseg... ] spend {time.time() - start_dicomseg:.0f} sec")
            # logging.info(f"[Done create_dicomseg... ] spend {time.time() - start_dicomseg:.0f} sec")

            #將dicom壓縮不包含dicom-seg  Dicom_JPEGlossless => 由於要用numpy > 2.0，之後補強
            path_dcmjpeglossless_n = os.path.join(path_nnunet, 'Dicom_JPEGlossless')
            if not os.path.isdir(path_dcmjpeglossless_n):  #如果資料夾不存在就建立
                os.mkdir(path_dcmjpeglossless_n) #製作nii資料夾          
            #compress_dicom_into_jpeglossless(path_dcm_n, path_dcmjpeglossless_n)

            #建立json檔
            path_json_out_n = os.path.join(path_nnunet, 'JSON')
            if not os.path.isdir(path_json_out_n):  #如果資料夾不存在就建立
                os.mkdir(path_json_out_n) #製作nii資料夾

            Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw']
            #group_id1 = 53 # 君彥的模型
            group_id = int(os.getenv("RADX_ANEURYSM_GROUP_ID", "56") or "56")  # nnU-Net的模型
            #group_id2 = 57 # nnU-Net 低threshold的模型
            # model_id = '924d1538-597c-41d6-bc27-4b0b359111cf'
            start_json = time.time()
            #make_aneurysm_pred_json(ID, pathlib.Path(path_nnunet), model_id)  #(_id, path_root, group_id)
            make_pred_json(ID, pathlib.Path(path_nnunet), group_id)  #(_id, path_root, group_id)
            #make_pred_json(excel_file, path_dcm, path_nii, path_reslice, path_dicomseg, path_json_out, [ID], Series, group_id)
            print(f"[Done make_pred_json... ] spend {time.time() - start_json:.0f} sec")
            logging.info(f"[Done make_pred_json... ] spend {time.time() - start_json:.0f} sec")

            #下面製作vessel dilate的dicom-seg跟json
            # make_vessel_pred_json(ID, pathlib.Path(path_nnunet))

            #接下來上傳dicom到orthanc
            path_zip_n = os.path.join(path_nnunet, 'Dicom_zip')
            if not os.path.isdir(path_zip_n):
                os.mkdir(path_zip_n)

            Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw', 'Dicom-Seg']
            start_upload_dicom = time.time()
            orthanc_zip_upload(path_dcm_n, path_zip_n, Series)
            print(f"[Done start_upload_dicom... ] spend {time.time() - start_upload_dicom:.0f} sec")
            logging.info(f"[Done start_upload_dicom... ] spend {time.time() - start_upload_dicom:.0f} sec")

            #把json跟nii輸出到out資料夾，這裡只傳nnunet的結果，因為nnU-Net的結果比較好
            shutil.copy(os.path.join(path_nnunet, 'Pred.nii.gz'), os.path.join(path_output, 'Pred_Aneurysm.nii.gz'))
            shutil.copy(os.path.join(path_nnunet, 'Prob.nii.gz'), os.path.join(path_output, 'Prob_Aneurysm.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'), os.path.join(path_output, 'Pred_Aneurysm_Vessel.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel_16.nii.gz'), os.path.join(path_output, 'Pred_Aneurysm_Vessel16.nii.gz'))
            shutil.copy(os.path.join(path_json_out_n, ID + '_platform_json.json'), os.path.join(path_output, 'Pred_Aneurysm.json'))
            shutil.copy(os.path.join(path_json_out_n, ID + '_platform_json.json'), os.path.join(path_output, 'Pred_Aneurysm_platform_json.json'))

            #SynthSEG要複製到output資料夾
            shutil.copy(os.path.join(path_nii_n, 'NEW_MRA_BRAIN_synthseg33.nii.gz'), os.path.join(path_output, 'SynthSEG_Aneurysm.nii.gz'))

            #radax步驟，接下來完成複製檔案到指定資料夾跟打api通知
            # upload_dir = '/home/david/ai-inference-result' #目的資料夾
            # aneurysm_json_file = os.path.join(path_nnunet, 'rdx_aneurysm_pred_json.json')
            # vessel_json_file = os.path.join(path_nnunet, 'rdx_vessel_dilated_json.json')
            
            # # 讀取aneurysm_json_file獲取study_instance_uid和series_instance_uid
            # with open(aneurysm_json_file, 'r', encoding='utf-8') as f:
            #     aneurysm_data = json.load(f)
            
            # study_instance_uid = aneurysm_data['study_instance_uid']
            # series_instance_uid = aneurysm_data['series_instance_uid']
            
            # # 在upload_dir建立study_instance_uid資料夾
            # study_dir = os.path.join(upload_dir, study_instance_uid)
            # if not os.path.isdir(study_dir):
            #     os.makedirs(study_dir)
            
            # # 建立aneurysm_model資料夾
            # aneurysm_model_dir = os.path.join(study_dir, 'aneurysm_model')
            # if not os.path.isdir(aneurysm_model_dir):
            #     os.makedirs(aneurysm_model_dir)
            
            # # 複製MRA_BRAIN_A*.dcm檔案到aneurysm_model資料夾
            # aneurysm_files = sorted([f for f in os.listdir(path_dicomseg_n) if f.startswith('MRA_BRAIN_A') and f.endswith('.dcm')])
            # for aneurysm_file in aneurysm_files:
            #     # 提取編號 (例如: MRA_BRAIN_A1.dcm -> A1)
            #     label = aneurysm_file.replace('MRA_BRAIN_', '').replace('.dcm', '')
            #     # 建立新檔名
            #     new_filename = f'{series_instance_uid}_{label}.dcm'
            #     src_path = os.path.join(path_dicomseg_n, aneurysm_file)
            #     dst_path = os.path.join(aneurysm_model_dir, new_filename)
            #     shutil.copy(src_path, dst_path)
            #     print(f"[Copy] {aneurysm_file} -> {new_filename}")
            #     logging.info(f"[Copy] {aneurysm_file} -> {new_filename}")
            
            # # 複製aneurysm_json_file到aneurysm_model資料夾並改名
            # aneurysm_json_dst = os.path.join(aneurysm_model_dir, f'{series_instance_uid}.json')
            # shutil.copy(aneurysm_json_file, aneurysm_json_dst)
            # print(f"[Copy] rdx_aneurysm_pred_json.json -> {series_instance_uid}.json")
            # logging.info(f"[Copy] rdx_aneurysm_pred_json.json -> {series_instance_uid}.json")
            
            # # 建立vessel_model資料夾
            # vessel_model_dir = os.path.join(study_dir, 'vessel_model')
            # if not os.path.isdir(vessel_model_dir):
            #     os.makedirs(vessel_model_dir)
            
            # # 複製MRA_BRAIN_Vessel_A1.dcm到vessel_model資料夾並改名
            # vessel_file = 'MRA_BRAIN_Vessel_A1.dcm'
            # vessel_src_path = os.path.join(path_dicomseg_n, vessel_file)
            # vessel_dst_path = os.path.join(vessel_model_dir, f'{series_instance_uid}.dcm')
            # if os.path.exists(vessel_src_path):
            #     shutil.copy(vessel_src_path, vessel_dst_path)
            #     print(f"[Copy] {vessel_file} -> {series_instance_uid}.dcm")
            #     logging.info(f"[Copy] {vessel_file} -> {series_instance_uid}.dcm")
            
            # # 複製vessel_json_file到vessel_model資料夾並改名
            # vessel_json_dst = os.path.join(vessel_model_dir, f'{series_instance_uid}.json')
            # shutil.copy(vessel_json_file, vessel_json_dst)
            # print(f"[Copy] rdx_vessel_pred_json.json -> {series_instance_uid}.json")
            # logging.info(f"[Copy] rdx_vessel_pred_json.json -> {series_instance_uid}.json")
            
            # # 發送POST請求到 /v1/ai-inference/inference-complete
            # # 請修改為正確的 API 端點，例如：
            # # api_url = 'http://localhost:8080/v1/ai-inference/inference-complete'
            # # api_url = 'http://10.103.1.193:3000/v1/ai-inference/inference-complete'
            # api_url = 'http://localhost:4000/v1/ai-inference/inference-complete'  # TODO: 請修改為正確的 API 端點
            # payload = {
            #     "studyInstanceUid": study_instance_uid,
            #     "seriesInstanceUid": series_instance_uid
            # }
            # try:
            #     response = requests.post(api_url, json=payload)
            #     print(f"[API POST] Status Code: {response.status_code}, Response: {response.text}")
            #     logging.info(f"[API POST] Status Code: {response.status_code}, Response: {response.text}")
            # except Exception as e:
            #     print(f"[API POST Error] {str(e)}")
            #     logging.error(f"[API POST Error] {str(e)}")

            #刪除資料夾
            # if os.path.isdir(path_process):  #如果資料夾存在
            #     shutil.rmtree(path_process) #清掉整個資料夾 

            print(f"[Done All Pipeline!!! ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done All Pipeline!!! ] spend {time.time() - start:.0f} sec")
            logging.info('!!! ' + ID +  ' post_aneurysm finish.')

            followup_process_root = pathlib.Path(
                os.getenv("RADX_FOLLOWUP_CASE_ROOT", os.getenv("RADX_FOLLOWUP_ROOT", path_process))
            )
            followup_output_root = pathlib.Path(
                os.getenv("RADX_FOLLOWUP_ROOT", path_process)
            )
            temp_json_paths = []
            # 若有提供 input_json（可能是檔案路徑或 JSON 內容）
            if input_json:
                # 若是檔案路徑，直接使用
                if os.path.isfile(input_json):
                    input_json_path = input_json
                else:
                    # 不是檔案路徑時，直接處理 JSON 內容
                    try:
                        json.loads(input_json)
                    except json.JSONDecodeError:
                        logging.warning("Skip followup: input_json is not valid JSON content.")
                        input_json_path = ""
                    else:
                        os.makedirs(path_processModel, exist_ok=True)
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            suffix=".json",
                            delete=False,
                            dir=path_processModel,
                            encoding="utf-8",
                        ) as temp_fp:
                            temp_fp.write(input_json)
                            input_json_path = temp_fp.name
                            temp_json_paths.append(temp_fp.name)
            else:
                # 未提供 input_json 時，不執行 followup
                input_json_path = ""

            outputs = []
            if input_json_path:
                start_followup = time.time()
                try:
                    outputs, has_model = pipeline_followup_v3_platform(
                        input_json=pathlib.Path(input_json_path),
                        path_process=followup_process_root,
                        model="Aneurysm",
                        model_type=1,
                        case_id=ID,
                        platform_json_name="Pred_Aneurysm_platform_json.json",
                        path_followup_root=followup_output_root,
                    )
                    logging.info(
                        "Followup-v3 outputs=%d has_model=%s", len(outputs), has_model
                    )
                    print(
                        f"[FollowUp] outputs={len(outputs)} has_model={has_model}"
                    )
                except Exception as exc:
                    has_model = False
                    logging.error("Followup-v3 platform failed.", exc_info=True)
                    print(f"[FollowUp] Error: followup-v3 platform failed: {exc!r}")

                print(f"[Done followup-v3 platform!!! ] spend {time.time() - start_followup:.0f} sec")
                logging.info(f"[Done followup-v3 platform!!! ] spend {time.time() - start_followup:.0f} sec")
            else:
                has_model = False
                logging.info("Skip followup-v3 platform: input_json is empty.")
                print("[FollowUp] Skip: input_json is empty.")

            for temp_path in temp_json_paths:
                try:
                    os.remove(temp_path)
                    logging.info("Remove temp json: %s", temp_path)
                except OSError:
                    logging.warning("Failed to remove temp json: %s", temp_path)

            #接下來，上傳json
            json_file_n = os.path.join(path_json_out_n, ID + '_platform_json.json')
            # followup 有執行且存在該模型時，上傳 followup 結果（含當前日期主體）
            if has_model and outputs:
                for json_path in outputs:
                    try:
                        resp = upload_json_aiteam(str(json_path))
                        logging.info("Upload followup json: %s resp=%s", json_path, resp)
                        print(f"[Upload] followup json={json_path} resp={resp}")
                    except Exception as exc:
                        logging.error("Upload followup json failed: %s", json_path, exc_info=True)
                        print(f"[Upload] followup json failed: {json_path} err={exc}")
            else:
                # followup 未執行或無模型時，上傳原版 JSON
                try:
                    resp = upload_json_aiteam(json_file_n)
                    logging.info("Upload original json: %s resp=%s", json_file_n, resp)
                    print(f"[Upload] original json={json_file_n} resp={resp}")
                except Exception as exc:
                    logging.error("Upload original json failed: %s", json_file_n, exc_info=True)
                    print(f"[Upload] original json failed: {json_file_n} err={exc}")
        
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
    # Load env file (RADX_ENV_FILE or code/config/radax.env) if present
    if load_env_from_default_locations:
        try:
            load_env_from_default_locations()
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=str, default = '17390820_20250604_MR_21406040004', help='目前執行的case的patient_id or study id')
    parser.add_argument('--Inputs', type=str, nargs='+', default = ['/data/4TB1/pipeline/chuan/example_input/17390820_20250604_MR_21406040004/MRA_BRAIN.nii.gz'], help='用於輸入的檔案')
    parser.add_argument('--DicomDir', type=str, nargs='+', default = ['/data/4TB1/pipeline/chuan/example_inputDicom/17390820_20250604_MR_21406040004/MRA_BRAIN/'], help='用於輸入的檔案')
    parser.add_argument('--Output_folder', type=str, default = '/data/4TB1/pipeline/chuan/example_output/',help='用於輸出結果的資料夾')    
    parser.add_argument('--input_json', type=str, default = '', help='followup-v3 平台輸入 json 內容或檔案路徑')
    parser.add_argument('--gpu_n', type=int, default = 0, help='使用哪一顆gpu')
    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs  # 將列表合併為字符串，保留順序
    DicomDirs = args.DicomDir #對應的dicom資料夾，用來做dicom-seg
    path_output = str(args.Output_folder)
    if "--Output_folder" not in sys.argv:
        path_output = _env_path("RADX_OUTPUT_ROOT", path_output)
    input_json = str(args.input_json)
    #path_output = args.Output_folder
    # 避免命令列/文字檔帶入 \r\n 造成路徑錯誤
    path_output = os.path.normpath(path_output.strip().replace("\r", "").replace("\n", ""))
    input_json = os.path.normpath(input_json.strip().replace("\r", "").replace("\n", ""))

    #讀出DWI, DWI0, ADC, SynthSEG的檔案
    MRA_BRAIN_file = Inputs[0]
    path_DcmDir = DicomDirs[0] #這邊在做dicom-seg時才會用到

    #需要安裝 pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg => 先不壓縮，因為壓縮需要numpy > 2

    #下面設定各個路徑
    path_code = _env_path("RADX_CODE_ROOT", "/data/4TB1/pipeline/chuan/code/")
    path_process = _env_path("RADX_PROCESS_ROOT", "/data/4TB1/pipeline/chuan/process/")  #前處理dicom路徑(test case)
    path_brain_model = _env_path(
        "RADX_BRAIN_MODEL",
        "/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset134_DeepMRABrain/nnUNetTrainer__nnUNetPlans__3d_fullres",
    )
    path_vessel_model = _env_path(
        "RADX_VESSEL_MODEL",
        "/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset135_DeepMRAVessel/nnUNetTrainer__nnUNetPlans__3d_fullres",
    )
    path_aneurysm_model = _env_path(
        "RADX_ANEURYSM_MODEL",
        "/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset080_DeepAneurysm/nnUNetTrainer__nnUNetPlans__3d_fullres",
    )
    path_processModel = os.path.join(path_process, 'Deep_Aneurysm')  #前處理dicom路徑(test case)
    #path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)

    #這裡先沒有dicom
    path_json = _env_path("RADX_JSON_ROOT", "/data/4TB1/pipeline/chuan/json/")  #存放json的路徑，回傳執行結果
    #json_path_name = os.path.join(path_json, 'Pred_Infarct.json')
    path_log = _env_path("RADX_LOG_ROOT", "/data/4TB1/pipeline/chuan/log/")  #log資料夾

    #自訂模型
    gpu_n = int(args.gpu_n)

    # 建置資料夾
    os.makedirs(path_processModel, exist_ok=True) # 如果資料夾不存在就建立，製作nii資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    #os.makedirs(path_output,exist_ok=True)

    #直接當作function的輸入，因為可能會切換成nnUNet的版本，所以自訂化模型移到跟model一起，synthseg自己做，不用統一
    pipeline_aneurysm(
        ID,
        MRA_BRAIN_file,
        path_output,
        path_code,
        path_brain_model,
        path_vessel_model,
        path_aneurysm_model,
        path_processModel,
        path_DcmDir,
        input_json,
        path_json,
        path_log,
        gpu_n,
    )
    

    # #最後再讀取json檔結果
    # with open(json_path_name) as f:
    #     data = json.load(f)

    # logging.info('Json!!! ' + str(data))
