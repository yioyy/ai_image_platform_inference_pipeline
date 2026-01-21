# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

這邊分別去執行3個

@author: chuan
"""
import warnings
warnings.filterwarnings("ignore") # 忽略警告输出


from asyncio.log import logger
import os
import time
import numpy as np
import pydicom
import nibabel
import logging
import glob
import shutil
import time
from nii_transforms import nii_img_replace
import json
import pandas as pd
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import matplotlib
import argparse
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import pandas as pd
# from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
import skimage 
import skimage.feature
import skimage.measure
from skimage import measure,color,morphology
from mpl_toolkits.axes_grid1 import make_axes_locatable
from random import uniform
from IPython.display import clear_output
import math
from collections import OrderedDict
import matplotlib.colors as mcolors
#from revise import revise2d, revise3d
from gpu_aneurysm import model_predict_aneurysm
import pynvml  # 导包
from util_aneurysm import reslice_nifti_pred_nobrain, create_MIP_pred, make_aneurysm_vessel_location_16labels_pred, calculate_aneurysm_long_axis_make_pred, make_table_row_patient_pred, make_table_add_location, \
                          create_dicomseg_multi_file, compress_dicom_into_jpeglossless, make_pred_json, orthanc_zip_upload, upload_json_aiteam
import create_dicomseg_multi_file_json_claude
#from multiprocessing import Process
import subprocess


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
            #shutil.copy(MRA_BRAIN_file, os.path.join(path_processID, 'MRA_BRAIN.nii.gz'))
            
            #接下來，以下運行gpu領域
            #以下做predict，為了避免gpu out of memory，還是以.sh來執行好惹
            #gpu_line = 'bash ' + path_code + 'gpu_stroke.sh ' + ID + ' ' + path_processID
            #os.system(gpu_line)

            #因為松諭會用排程，但因為ai跟mip都要用到gpu，所以還是要管gpu ram，#multiprocessing沒辦法釋放gpu，要改用subprocess.run()
            #model_predict_aneurysm(path_code, path_processID, ID, path_log, gpu_n)
            print("Running stage 1: Aneurysm inference!!!")

            # 定義要傳入的參數，建立指令
            cmd = [
                   "python", "gpu_aneurysm.py",
                   "--path_code", path_code,
                   "--path_process", path_processID,
                   "--case", ID,
                   "--path_log", path_log,
                   "--gpu_n", str(gpu_n)  # 注意要轉成字串
                  ]

            #result = subprocess.run(cmd, capture_output=True, text=True) 這會讓 subprocess.run() 自動幫你捕捉 stdout 和 stderr 的輸出，不然預設是印在 terminal 上，不會儲存。
            # 執行 subprocess
            #subprocess.run(cmd)

            #接下來做mip影像
            path_dcm = os.path.join(path_processID, 'Dicom')
            path_nii = os.path.join(path_processID, 'Image_nii')
            path_reslice = os.path.join(path_processID, 'Image_reslice')
            path_excel = os.path.join(path_processID, 'excel')
            if not os.path.isdir(path_dcm):  #如果資料夾不存在就建立
                os.mkdir(path_dcm) #製作nii資料夾
            if not os.path.isdir(path_nii):  #如果資料夾不存在就建立
                os.mkdir(path_nii) #製作nii資料夾
            if not os.path.isdir(path_reslice):  #如果資料夾不存在就建立
                os.mkdir(path_reslice) #製作nii資料夾
            if not os.path.isdir(path_excel):  #如果資料夾不存在就建立
                os.mkdir(path_excel) #製作nii資料夾

            #複製nii到nii資料夾
            shutil.copy(os.path.join(path_processID, 'MRA_BRAIN.nii.gz'), os.path.join(path_nii, 'MRA_BRAIN.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Pred.nii.gz'), os.path.join(path_nii, 'Pred.nii.gz'))
            shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'), os.path.join(path_nii, 'Vessel.nii.gz'))

            #reslice
            reslice_nifti_pred_nobrain(path_nii, path_reslice)

            #做mip影像
            #複製dicom影像
            if not os.path.isdir(os.path.join(path_dcm, 'MRA_BRAIN')):  #如果資料夾不存在就建立
                shutil.copytree(path_outdcm, os.path.join(path_dcm, 'MRA_BRAIN'))

            path_png = os.path.join(path_code, 'png')
            create_MIP_pred(path_dcm, path_reslice, path_png, gpu_n)

            #接下來是計算動脈瘤的各項數據
            #make_aneurysm_vessel_location_16labels_pred(path_processID)
            #calculate_aneurysm_long_axis_make_pred(path_dcm, path_processID, path_excel, ID)
            #make_table_row_patient_pred(path_excel, ID)
            #make_table_add_location(path_processID, path_excel)

            #接下來製作dicom-seg
            path_dicomseg = os.path.join(path_dcm, 'Dicom-Seg')
            if not os.path.isdir(path_dicomseg):  #如果資料夾不存在就建立
                os.mkdir(path_dicomseg) #製作nii資料夾

            #上線版不用做vessel
            create_dicomseg_multi_file(path_code, path_dcm, path_nii, path_reslice, path_dicomseg, ID)

            #將dicom壓縮不包含dicom-seg  Dicom_JPEGlossless => 由於要用numpy > 2.0，之後補強
            path_dcmjpeglossless = os.path.join(path_processID, 'Dicom_JPEGlossless')
            if not os.path.isdir(path_dcmjpeglossless):  #如果資料夾不存在就建立
                os.mkdir(path_dcmjpeglossless) #製作nii資料夾

            #compress_dicom_into_jpeglossless(path_dcm, path_dcmjpeglossless)

            #建立json檔
            path_json_out = os.path.join(path_processID, 'JSON')
            if not os.path.isdir(path_json_out):  #如果資料夾不存在就建立
                os.mkdir(path_json_out) #製作nii資料夾

            Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw']
            group_id = 49
            excel_file = os.path.join(path_excel, 'Aneurysm_Pred_list.xlsx')
            #make_pred_json(excel_file, path_dcm, path_nii, path_reslice, path_dicomseg, path_json_out, [ID], Series, group_id)

            #接下來上傳dicom到orthanc
            path_zip = os.path.join(path_processID, 'Dicom_zip')
            if not os.path.isdir(path_zip):
                os.mkdir(path_zip)

            Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw', 'Dicom-Seg']
            #orthanc_zip_upload(path_dcm, path_zip, Series)

            #接下來，上傳json
            json_file = os.path.join(path_json_out, ID + '_platform_json.json')
            #upload_json_aiteam(json_file)

            #把json跟nii輸出到out資料夾
            # shutil.copy(os.path.join(path_processID, 'Pred.nii.gz'), os.path.join(path_output, 'Pred_Aneurysm.nii.gz'))
            # shutil.copy(os.path.join(path_processID, 'Prob.nii.gz'), os.path.join(path_output, 'Prob_Aneurysm.nii.gz'))
            # shutil.copy(os.path.join(path_processID, 'Vessel.nii.gz'), os.path.join(path_output, 'Pred_Aneurysm_Vessel.nii.gz'))
            # shutil.copy(os.path.join(path_processID, 'Vessel_16.nii.gz'), os.path.join(path_output, 'Pred_Aneurysm_Vessel16.nii.gz'))
            # shutil.copy(os.path.join(path_json_out, ID + '_platform_json.json'), os.path.join(path_output, 'Pred_Aneurysm.json'))
            # shutil.copy(os.path.join(path_json_out, ID + '_platform_json.json'), os.path.join(path_output, 'Pred_Aneurysm_platform_json.json'))
            
            #刪除資料夾
            # if os.path.isdir(path_process):  #如果資料夾存在
            #     shutil.rmtree(path_process) #清掉整個資料夾 

            logging.info('!!! ' + ID +  ' post_aneurysm finish.')
        
        else:
            logging.error('!!! ' + str(ID) + ' Insufficient GPU Memory.')
            # 以json做輸出
            code_pass = 1
            msg = "Insufficient GPU Memory"

            # #刪除資料夾
            # if os.path.isdir(path_process):  #如果資料夾存在
            #     shutil.rmtree(path_process) #清掉整個資料夾

    except:
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
    parser.add_argument('--ID', type=str, default = '1C95C88E_20171122_MR_1C95C88E', help='目前執行的case的patient_id or study id')
    parser.add_argument('--Inputs', type=str, nargs='+', default = ['/data/4TB1/pipeline/chuan/example_input/1C95C88E_20171122_MR_1C95C88E/MRA_BRAIN.nii.gz'], help='用於輸入的檔案')
    parser.add_argument('--DicomDir', type=str, nargs='+', default = ['/data/4TB1/pipeline/chuan/example_inputDicom/1C95C88E_20171122_MR_1C95C88E/MRA_BRAIN/'], help='用於輸入的檔案')
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
    path_code = '/data/4TB1/pipeline/chuan/code/'
    path_process = '/data/4TB1/pipeline/chuan/process/'  #前處理dicom路徑(test case)
    path_processModel = os.path.join(path_process, 'Deep_Aneurysm_grimmer')  #前處理dicom路徑(test case)
    #path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)

    #這裡先沒有dicom
    path_json = '/data/4TB1/pipeline/chuan/json/'  #存放json的路徑，回傳執行結果
    #json_path_name = os.path.join(path_json, 'Pred_Infarct.json')
    path_log = '/data/4TB1/pipeline/chuan/log/'  #log資料夾

    #自訂模型
    gpu_n = 0  #使用哪一顆gpu

    # 建置資料夾
    os.makedirs(path_processModel, exist_ok=True) # 如果資料夾不存在就建立，製作nii資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_output,exist_ok=True)

    #直接當作function的輸入，因為可能會切換成nnUNet的版本，所以自訂化模型移到跟model一起，synthseg自己做，不用統一
    pipeline_aneurysm(ID, MRA_BRAIN_file, path_output, path_code, path_processModel, path_DcmDir, path_json, path_log, gpu_n)
    

    # #最後再讀取json檔結果
    # with open(json_path_name) as f:
    #     data = json.load(f)

    # logging.info('Json!!! ' + str(data))