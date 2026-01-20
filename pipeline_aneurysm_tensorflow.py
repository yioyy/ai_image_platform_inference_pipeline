
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
    decompress_dicom_with_gdcm, resampleSynthSEG2original
import subprocess
from code_ai.pipeline.dicomseg.build_aneurysm import main as make_aneurysm_pred_json
from code_ai.pipeline.dicomseg.build_vessel_dilated import main as make_vessel_pred_json

import pathlib
import requests
from typing import Any, Dict, List, Optional

# Optional env-file loader (for deploy/radax-test vs Deploy/radax official)
try:
    from config.load_env import load_env_from_default_locations
except Exception:
    load_env_from_default_locations = None


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


def _clean_env_path(p: str) -> str:
    return os.path.normpath(str(p).strip().replace("\r", "").replace("\n", ""))


def _env_path(key: str, default: str) -> str:
    v = os.getenv(key, "").strip()
    return _clean_env_path(v) if v else _clean_env_path(default)


def _get_api_url_from_env() -> str:
    url = os.getenv("RADX_API_URL", "").strip()
    if url:
        return url
    host = os.getenv("RADX_API_HOST", "localhost").strip() or "localhost"
    port = os.getenv("RADX_API_PORT", "24000").strip() or "24000"
    return f"http://{host}:{port}/v1/ai-inference/inference-complete"

def pipeline_aneurysm(ID, 
                      MRA_BRAIN_file,  
                      path_output,
                      path_code = '/mnt/e/pipeline/chuan/code/', 
                      path_brain_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset134_DeepMRABrain/nnUNetTrainer__nnUNetPlans__3d_fullres',
                      path_vessel_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset135_DeepMRAVessel/nnUNetTrainer__nnUNetPlans__3d_fullres',
                      path_aneurysm_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset080_DeepAneurysm/nnUNetTrainer__nnUNetPlans__3d_fullres',
                      path_processModel = '/mnt/e/pipeline/chuan/process/Deep_Aneurysm/', 
                      path_outdcm = '',
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
    path_json = _clean_path(path_json)
    path_log = _clean_path(path_log)

    # 確保輸出資料夾存在，避免後面 shutil.copy 直接 FileNotFoundError
    os.makedirs(path_output, exist_ok=True)

    #當使用gpu有錯時才確認
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    def _try_get_study_instance_uid_from_dicom_dir(dicom_dir: str) -> str:
        """
        失敗時用：盡力從輸入 DICOM 資料夾抓 StudyInstanceUID。
        取不到就回傳空字串，避免整個 pipeline 因為通知 API 而再炸一次。
        """
        if not dicom_dir or not os.path.isdir(dicom_dir):
            return ""
        try:
            import pydicom  # lazy import：避免環境沒有 pydicom 時直接掛掉
        except Exception:
            return ""

        for root, _, files in os.walk(dicom_dir):
            for fn in sorted(files):
                fp = os.path.join(root, fn)
                if not os.path.isfile(fp):
                    continue
                try:
                    ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                    uid = str(getattr(ds, "StudyInstanceUID", "") or "")
                    if uid:
                        return uid
                except Exception:
                    continue
        return ""

    def _post_inference_complete(
        *,
        api_url: str,
        study_instance_uid: str,
        aneurysm_inference_id: Optional[str],
        vessel_inference_id: Optional[str],
        result: str,
    ) -> None:
        """呼叫 /v1/ai-inference/inference-complete；aneurysm_model / vessel_model 各送一次。"""
        def _with_optional_result(payload: Dict[str, Any]) -> Dict[str, Any]:
            if result != "":
                payload["result"] = result
            return payload

        payloads: List[Dict[str, Any]]
        if result == "success" or result == "":
            payloads = [
                _with_optional_result(
                    {
                        "studyInstanceUid": study_instance_uid,
                        "modelName": "aneurysm_model",
                        "inferenceId": aneurysm_inference_id,
                    }
                ),
                _with_optional_result(
                    {
                        "studyInstanceUid": study_instance_uid,
                        "modelName": "vessel_model",
                        "inferenceId": vessel_inference_id,
                    }
                ),
            ]
        else:
            payloads = [
                _with_optional_result(
                    {
                        "studyInstanceUid": study_instance_uid,
                        "modelName": "aneurysm_model",
                    }
                ),
                _with_optional_result(
                    {
                        "studyInstanceUid": study_instance_uid,
                        "modelName": "vessel_model",
                    }
                ),
            ]

        for payload in payloads:
            try:
                response = requests.post(api_url, json=payload)
                print(f"[API POST] payload={payload} Status Code: {response.status_code}, Response: {response.text}")
                logging.info(f"[API POST] payload={payload} Status Code: {response.status_code}, Response: {response.text}")
            except Exception as e:
                print(f"[API POST Error] payload={payload} err={str(e)}")
                logging.error(f"[API POST Error] payload={payload} err={str(e)}")

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
                   "python", "/home/david/pipeline/chuan/radax/gpu_aneurysm.py",
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
            subprocess.run(cmd)
            print(f"[Done AI Inference... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done AI Inference... ] spend {time.time() - start:.0f} sec")

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
            subprocess.run(cmd)
            print(f"[Done SynthSEG AI Inference... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done SynthSEG AI Inference... ] spend {time.time() - start:.0f} sec")

            #這邊還要做resample到original space
            SynthSEG_array = resampleSynthSEG2original(path_nii_n, 'MRA_BRAIN', 'synthseg33') 
            shutil.copy(os.path.join(path_nii_n, 'NEW_MRA_BRAIN_synthseg33.nii.gz'), os.path.join(path_process, 'SynthSEG.nii.gz'))

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
            start_dicomseg = time.time()
            #create_dicomseg_multi_file(path_code, path_dcm_n, path_nii_n, path_reslice_n, path_dicomseg_n, ID)
            print(f"[Done create_dicomseg... ] spend {time.time() - start_dicomseg:.0f} sec")
            logging.info(f"[Done create_dicomseg... ] spend {time.time() - start_dicomseg:.0f} sec")

            #將dicom壓縮不包含dicom-seg  Dicom_JPEGlossless => 由於要用numpy > 2.0，之後補強
            path_dcmjpeglossless_n = os.path.join(path_nnunet, 'Dicom_JPEGlossless')
            if not os.path.isdir(path_dcmjpeglossless_n):  #如果資料夾不存在就建立
                os.mkdir(path_dcmjpeglossless_n) #製作nii資料夾         
            #compress_dicom_into_jpeglossless(path_dcm_t, path_dcmjpeglossless_t)
            #compress_dicom_into_jpeglossless(path_dcm_n, path_dcmjpeglossless_n)
            #compress_dicom_into_jpeglossless(path_dcm_nl, path_dcmjpeglossless_nl)

            #建立json檔
            path_json_out_n = os.path.join(path_nnunet, 'JSON')
            if not os.path.isdir(path_json_out_n):  #如果資料夾不存在就建立
                os.mkdir(path_json_out_n) #製作nii資料夾

            Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw']
            group_id3 = 56 # nnU-Net的模型
            model_id = '924d1538-597c-41d6-bc27-4b0b359111cf'
            start_json = time.time()
            make_aneurysm_pred_json(ID, pathlib.Path(path_nnunet), model_id)  #(_id, path_root, group_id)
            print(f"[Done make_pred_json... ] spend {time.time() - start_json:.0f} sec")
            logging.info(f"[Done make_pred_json... ] spend {time.time() - start_json:.0f} sec")

            #下面製作vessel dilate的dicom-seg跟json
            make_vessel_pred_json(ID, pathlib.Path(path_nnunet))

            #接下來上傳dicom到orthanc
            path_zip_n = os.path.join(path_nnunet, 'Dicom_zip')
            if not os.path.isdir(path_zip_n):
                os.mkdir(path_zip_n)

            Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw', 'Dicom-Seg']
            start_upload_json = time.time()
            #orthanc_zip_upload(path_dcm_n, path_zip_n, Series)
            
            print(f"[Done start_upload_json... ] spend {time.time() - start_upload_json:.0f} sec")
            logging.info(f"[Done start_upload_json... ] spend {time.time() - start_upload_json:.0f} sec")

            #接下來，上傳json
            # json_file_n = os.path.join(path_json_out_n, ID + '_platform_json.json')

            # upload_json_aiteam(json_file_n)

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
            
            # 讀取 aneurysm_json_file（新版 schema）
            with open(aneurysm_json_file, 'r', encoding='utf-8') as f:
                aneurysm_data = json.load(f)
            
            # 新版 schema：study/series 都是 list
            study_instance_uid = (aneurysm_data.get('input_study_instance_uid') or [''])[0]
            input_series_instance_uid = (aneurysm_data.get('input_series_instance_uid') or [''])[0]
            aneurysm_inference_id = str(aneurysm_data.get('inference_id') or 'unknown')
            
            # 在upload_dir建立study_instance_uid資料夾
            study_dir = os.path.join(upload_dir, study_instance_uid)
            if not os.path.isdir(study_dir):
                os.makedirs(study_dir)
            
            # 建立 aneurysm_model/<aneurysm_inference_id> 資料夾
            aneurysm_model_root = os.path.join(study_dir, 'aneurysm_model')
            os.makedirs(aneurysm_model_root, exist_ok=True)
            aneurysm_infer_dir = os.path.join(aneurysm_model_root, aneurysm_inference_id)
            os.makedirs(aneurysm_infer_dir, exist_ok=True)

            # 複製 aneurysm prediction.json
            aneurysm_pred_dst = os.path.join(aneurysm_infer_dir, 'prediction.json')
            shutil.copy(aneurysm_json_file, aneurysm_pred_dst)
            print(f"[Copy] rdx_aneurysm_pred_json.json -> {aneurysm_pred_dst}")
            logging.info(f"[Copy] rdx_aneurysm_pred_json.json -> {aneurysm_pred_dst}")

            # 1) MRA_BRAIN dicom-seg：放在 aneurysm_infer_dir 根目錄
            for det in aneurysm_data.get('detections', []) or []:
                label = str(det.get('label') or '')
                seg_series_uid = str(det.get('series_instance_uid') or '')
                if not label or not seg_series_uid:
                    continue

                src_file = f"MRA_BRAIN_{label}.dcm"
                src_path = os.path.join(path_dicomseg_n, src_file)
                if not os.path.exists(src_path):
                    logging.warning(f"[Skip] missing MRA seg: {src_path}")
                    continue

                # 需求：dicom-seg 檔名用 detections.series_instance_uid + _A?.dcm
                dst_filename = f"{seg_series_uid}_{label}.dcm"
                dst_path = os.path.join(aneurysm_infer_dir, dst_filename)
                shutil.copy(src_path, dst_path)
                print(f"[Copy] {src_file} -> {dst_filename}")
                logging.info(f"[Copy] {src_file} -> {dst_filename}")

            # 2) pitch/yaw：
            # - 建立 series folder：放該 series 的 DICOM 影像（來源：nnUNet/Dicom/MIP_Pitch、nnUNet/Dicom/MIP_Yaw）
            # - dicom-seg 檔案：與 MRA_BRAIN dicom-seg 同層，放在 aneurysm_infer_dir（不放進 series folder）
            reformatted_series = aneurysm_data.get('reformatted_series', []) or []
            for series_item in reformatted_series:
                series_folder_uid = str(series_item.get('series_instance_uid') or '')
                if not series_folder_uid:
                    continue

                series_description = str(series_item.get('series_description') or '').lower()
                if series_description == 'mip_pitch':
                    src_prefix = "MIP_Pitch"
                elif series_description == 'mip_yaw':
                    src_prefix = "MIP_Yaw"
                else:
                    continue

                series_dir = os.path.join(aneurysm_infer_dir, series_folder_uid)
                os.makedirs(series_dir, exist_ok=True)

                # 複製該 series 的 DICOM 影像進資料夾
                src_dicom_dir = os.path.join(path_nnunet, "Dicom", src_prefix)
                if os.path.isdir(src_dicom_dir):
                    for fn in sorted(os.listdir(src_dicom_dir)):
                        src_fp = os.path.join(src_dicom_dir, fn)
                        if not os.path.isfile(src_fp):
                            continue
                        # 通常都是 .dcm；非 dcm 也一併複製避免漏檔
                        dst_fp = os.path.join(series_dir, fn)
                        shutil.copy(src_fp, dst_fp)
                else:
                    logging.warning(f"[Skip] missing source dicom folder: {src_dicom_dir}")

                for det in series_item.get('detections', []) or []:
                    label = str(det.get('label') or '')
                    seg_series_uid = str(det.get('series_instance_uid') or '')
                    if not label or not seg_series_uid:
                        continue

                    src_file = f"{src_prefix}_{label}.dcm"
                    src_path = os.path.join(path_dicomseg_n, src_file)
                    if not os.path.exists(src_path):
                        logging.warning(f"[Skip] missing {src_prefix} seg: {src_path}")
                        continue

                    # 需求：dicom-seg 檔名用 detections.series_instance_uid + _A?.dcm
                    dst_filename = f"{seg_series_uid}_{label}.dcm"
                    # dicom-seg 與 MRA_BRAIN 同層
                    dst_path = os.path.join(aneurysm_infer_dir, dst_filename)
                    shutil.copy(src_path, dst_path)
                    print(f"[Copy] {src_file} -> {dst_filename}")
                    logging.info(f"[Copy] {src_file} -> {dst_filename}")
            
            # 建立 vessel_model/<vessel_inference_id> 資料夾
            with open(vessel_json_file, 'r', encoding='utf-8') as f:
                vessel_data = json.load(f)
            vessel_inference_id = str(vessel_data.get('inference_id') or 'unknown')

            vessel_model_root = os.path.join(study_dir, 'vessel_model')
            os.makedirs(vessel_model_root, exist_ok=True)
            vessel_infer_dir = os.path.join(vessel_model_root, vessel_inference_id)
            os.makedirs(vessel_infer_dir, exist_ok=True)

            # 複製 vessel prediction.json
            vessel_pred_dst = os.path.join(vessel_infer_dir, 'prediction.json')
            shutil.copy(vessel_json_file, vessel_pred_dst)
            print(f"[Copy] rdx_vessel_dilated_json.json -> {vessel_pred_dst}")
            logging.info(f"[Copy] rdx_vessel_dilated_json.json -> {vessel_pred_dst}")

            # 複製 vessel dicom-seg（檔名：<detections.series_instance_uid>.dcm）
            vessel_file = 'MRA_BRAIN_Vessel_A1.dcm'
            vessel_src_path = os.path.join(path_dicomseg_n, vessel_file)
            vessel_seg_uid = ''
            try:
                vessel_seg_uid = str(((vessel_data.get('detections') or [{}])[0]).get('series_instance_uid') or '')
            except Exception:
                vessel_seg_uid = ''
            if os.path.exists(vessel_src_path) and vessel_seg_uid:
                vessel_dst_path = os.path.join(vessel_infer_dir, f'{vessel_seg_uid}.dcm')
                shutil.copy(vessel_src_path, vessel_dst_path)
                print(f"[Copy] {vessel_file} -> {vessel_seg_uid}.dcm")
                logging.info(f"[Copy] {vessel_file} -> {vessel_seg_uid}.dcm")
            else:
                logging.warning(f"[Skip] vessel seg missing or uid empty: {vessel_src_path}, uid={vessel_seg_uid}")
            
            # 發送POST請求到 /v1/ai-inference/inference-complete
            api_url = _get_api_url_from_env()

            _post_inference_complete(
                api_url=api_url,
                study_instance_uid=study_instance_uid,
                aneurysm_inference_id=aneurysm_inference_id,
                vessel_inference_id=vessel_inference_id,
                result="success",
            )

            # #刪除資料夾
            # # if os.path.isdir(path_process):  #如果資料夾存在
            # #     shutil.rmtree(path_process) #清掉整個資料夾 

            print(f"[Done All Pipeline!!! ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done All Pipeline!!! ] spend {time.time() - start:.0f} sec")
            logging.info('!!! ' + ID +  ' post_aneurysm finish.')
        
        else:
            logging.error('!!! ' + str(ID) + ' Insufficient GPU Memory.')
            # 以json做輸出
            code_pass = 1
            msg = "Insufficient GPU Memory"

            # 失敗也要建立 json 並上傳（依需求：result = failed，不帶 inferenceId）
            api_url = _get_api_url_from_env()
            study_instance_uid = _try_get_study_instance_uid_from_dicom_dir(path_outdcm)
            if not study_instance_uid:
                logging.warning(f"[Failed notify] cannot resolve StudyInstanceUID from dicom dir: {path_outdcm}")
            _post_inference_complete(
                api_url=api_url,
                study_instance_uid=study_instance_uid,
                aneurysm_inference_id=None,
                vessel_inference_id=None,
                result="failed",
            )

            # #刪除資料夾
            # if os.path.isdir(path_process):  #如果資料夾存在
            #     shutil.rmtree(path_process) #清掉整個資料夾

    except Exception:
        logging.error('!!! ' + str(ID) + ' gpu have error code.')
        logging.error("Catch an exception.", exc_info=True)
        # 發生例外也視為 inference 失敗，補送 failed（不帶 inferenceId）
        try:
            api_url = _get_api_url_from_env()
            study_instance_uid = _try_get_study_instance_uid_from_dicom_dir(path_outdcm)
            if not study_instance_uid:
                logging.warning(f"[Failed notify] cannot resolve StudyInstanceUID from dicom dir: {path_outdcm}")
            _post_inference_complete(
                api_url=api_url,
                study_instance_uid=study_instance_uid,
                aneurysm_inference_id=None,
                vessel_inference_id=None,
                result="failed",
            )
        except Exception:
            # 避免通知失敗影響主流程的錯誤處理
            logging.error("[Failed notify] error while posting inference-complete failed.", exc_info=True)
        # 刪除資料夾
        # if os.path.isdir(path_process):  #如果資料夾存在
        #     shutil.rmtree(path_process) #清掉整個資料夾
   
    print('end!!!')
    return

#其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    # Load env file (RADX_ENV_FILE or radax/config/radax.env) if present
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
    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs  # 將列表合併為字符串，保留順序
    DicomDirs = args.DicomDir #對應的dicom資料夾，用來做dicom-seg
    path_output = str(args.Output_folder)

    #讀出DWI, DWI0, ADC, SynthSEG的檔案
    MRA_BRAIN_file = Inputs[0]
    path_DcmDir = DicomDirs[0] #這邊在做dicom-seg時才會用到

    #需要安裝 pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg => 先不壓縮，因為壓縮需要numpy > 2

    # 下面設定各個路徑（以環境設定檔/環境變數覆蓋）
    path_code = _env_path("RADX_CODE_ROOT", "/home/david/pipeline/chuan/radax/")
    path_process = _env_path("RADX_PROCESS_ROOT", "/home/david/pipeline/chuan/process/")  # 前處理 dicom 路徑
    path_brain_model = _env_path(
        "RADX_BRAIN_MODEL",
        "/home/david/pipeline/chuan/radax/nnUNet/nnUNet_results/Dataset134_DeepMRABrain/nnUNetTrainer__nnUNetPlans__3d_fullres",
    )
    path_vessel_model = _env_path(
        "RADX_VESSEL_MODEL",
        "/home/david/pipeline/chuan/radax/nnUNet/nnUNet_results/Dataset135_DeepMRAVessel/nnUNetTrainer__nnUNetPlans__3d_fullres",
    )
    path_aneurysm_model = _env_path(
        "RADX_ANEURYSM_MODEL",
        "/home/david/pipeline/chuan/radax/nnUNet/nnUNet_results/Dataset080_DeepAneurysm/nnUNetTrainer__nnUNetPlans__3d_fullres",
    )
    path_processModel = os.path.join(path_process, "Deep_Aneurysm")
    #path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)

    #這裡先沒有dicom
    path_json = _env_path("RADX_JSON_ROOT", "/home/david/pipeline/chuan/json/")  # 存放 json 的路徑
    #json_path_name = os.path.join(path_json, 'Pred_Infarct.json')
    path_log = _env_path("RADX_LOG_ROOT", "/home/david/pipeline/chuan/log/")  # log 資料夾

    #自訂模型
    gpu_n = 0  #使用哪一顆gpu

    # 建置資料夾
    os.makedirs(path_processModel, exist_ok=True) # 如果資料夾不存在就建立，製作nii資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_output,exist_ok=True)

    #直接當作function的輸入，因為可能會切換成nnUNet的版本，所以自訂化模型移到跟model一起，synthseg自己做，不用統一
    pipeline_aneurysm(ID, MRA_BRAIN_file, path_output, path_code, path_brain_model, path_vessel_model, path_aneurysm_model, path_processModel, path_DcmDir, path_json, path_log, gpu_n)

    # #最後再讀取json檔結果
    # with open(json_path_name) as f:
    #     data = json.load(f)

    # logging.info('Json!!! ' + str(data))
