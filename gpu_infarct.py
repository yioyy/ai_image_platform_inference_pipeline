#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

predict model，非常重要:使用這個程式，安裝的版本都需要一致，才能運行
conda create -n orthanc-stroke tensorflow-gpu=2.3.0 anaconda python=3.7
這邊弄2個版本，一個是nnU-Net的版本，另一個是君彥的版本

@author: chuan
"""
import warnings
warnings.filterwarnings("ignore") # 忽略警告输出


import glob, re, os, sys, time, itertools
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import glob
import cv2 
import time
import pydicom
import nibabel as nib
#import gdcm
import numpy as np
import keras
import pandas as pd
# from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
import skimage 
import skimage.feature
import skimage.measure
from skimage import measure,color,morphology
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from scipy import signal
from scipy import ndimage as ndi
from skimage.morphology import disk, ball, binary_dilation, binary_closing, remove_small_objects, binary_erosion, binary_opening, skeletonize
from skimage.filters import threshold_multiotsu, gaussian, threshold_otsu, frangi
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import watershed, expand_labels
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt as dist_field
from IPython.display import display, HTML
from brainextractor import BrainExtractor
import random
from random import uniform
import argparse
import logging
import json
import psutil
import pynvml #导包
from collections import OrderedDict
from nii_transforms import nii_img_replace
autotune = tf.data.experimental.AUTOTUNE
from nnResUNet_long_BigBatch_cosine_AneDilate_classifier_test.gpu_nnUNet2D import predict_from_raw_data


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

#nii改變影像後儲存，data => nib.load的輸出，new_img => 更改的影像
def nii_img_replace(data, new_img):
    affine = data.affine
    header = data.header.copy()
    new_nii = nib.nifti1.Nifti1Image(new_img, affine, header=header)
    return new_nii

#篩選掉threshold不到的動脈瘤（原始版本，基於外接矩形）
def filter_aneurysm(pred_prob_map, spacing, conf_th=0.1, min_diameter=2, top_k=4, obj_th=0.67):
    # object_analysis
    pred_label = label(pred_prob_map > conf_th)
    # pred_label measurement，取得pred的各種屬性數值，例如intensity_max, intensity_max
    props = regionprops_table(pred_label, pred_prob_map, properties=('label', 'bbox', 'intensity_max', 'intensity_mean'))
    df_pred = pd.DataFrame({'ori_Pred_label':props['label'],
                            'Pred_diameter':((props['bbox-3']-props['bbox-0'])*spacing[0] + (props['bbox-4']-props['bbox-1'])*spacing[1]) / 2,
                            'Pred_max':props['intensity_max'],
                            'Pred_mean':props['intensity_mean'],})
    
    df_pred = df_pred[(df_pred['Pred_diameter'] >= min_diameter)]  # filter too small object
    df_pred = df_pred.sort_values(by='Pred_max', ascending=False) #用最大強度來排序
    df_pred['Pred_label'] = np.arange(1, df_pred.shape[0]+1)
    if top_k > 0:
        df_pred = df_pred[:top_k]  # top_k filter
    df_pred = df_pred[df_pred['Pred_max'] >= obj_th]  # object filter
    # remap pred_label array
    new_pred_label = np.zeros_like(pred_label)
    for k, v in zip(df_pred['ori_Pred_label'].values, df_pred['Pred_label'].values):
        new_pred_label[pred_label==k] = v
        
    return pred_prob_map, df_pred, new_pred_label.astype(int)

#篩選腦梗塞區域（基於連通區域實際體積）
def filter_infarct_by_volume(pred_prob_map, spacing, conf_th=0.1, min_volume=0.3, top_k=30, obj_th=0.5):
    """
    過濾腦梗塞預測結果，基於連通區域的實際體積
    
    參數:
        pred_prob_map: 預測機率圖
        spacing: 體素間距 [spacing_x, spacing_y, spacing_z] (單位: mm)
        conf_th: 機率閾值，用於二值化
        min_volume: 最小體積閾值 (單位: cm³)
        top_k: 保留前k個最大機率的區域 (0表示不限制)
        obj_th: 區域最大機率閾值
    
    返回:
        pred_prob_map: 原始機率圖
        df_pred: 過濾後的區域屬性資料框
        new_pred_label: 重新編號的標籤圖
    """
    # 二值化並進行連通區域標記
    pred_label = label(pred_prob_map > conf_th)
    
    # 計算每個連通區域的屬性
    props = regionprops_table(
        pred_label, 
        pred_prob_map, 
        properties=('label', 'area', 'intensity_max', 'intensity_mean')
    )
    
    # 計算每個區域的實際體積 (cm³)
    # area 是像素數量，乘以體素體積並轉換為 cm³
    # spacing[0] = x 方向, spacing[1] = y 方向, spacing[2] = z 方向
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_volume_cm3 = voxel_volume_mm3 / 1000.0  # mm³ 轉 cm³
    
    df_pred = pd.DataFrame({
        'ori_Pred_label': props['label'],
        'Pred_volume': props['area'] * voxel_volume_cm3,  # 體積 (cm³)
        'Pred_max': props['intensity_max'],
        'Pred_mean': props['intensity_mean'],
    })
    
    # 過濾體積過小的區域
    df_pred = df_pred[df_pred['Pred_volume'] >= min_volume]
    
    # 按最大機率排序
    df_pred = df_pred.sort_values(by='Pred_max', ascending=False)
    
    # 重新編號
    df_pred['Pred_label'] = np.arange(1, df_pred.shape[0] + 1)
    
    # 保留前 top_k 個區域
    if top_k > 0:
        df_pred = df_pred[:top_k]
    
    # 過濾最大機率過低的區域
    df_pred = df_pred[df_pred['Pred_max'] >= obj_th]
    
    # 重新映射標籤圖
    new_pred_label = np.zeros_like(pred_label)
    for k, v in zip(df_pred['ori_Pred_label'].values, df_pred['Pred_label'].values):
        new_pred_label[pred_label == k] = 1 #因為是體積，所以全部等於1就好
    
    return pred_prob_map, df_pred, new_pred_label.astype(int)

#"主程式"
#model_predict_aneurysm(path_code, path_process, case_name, path_log, gpu_n)
def model_predict_infarct(path_code, path_process, path_nnunet_model, case_name, path_log, gpu_n):

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

    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT)

    def case_json(json_file_path, ret, code, msg):
        json_dict = OrderedDict()
        json_dict["ret"] = ret #使用的程式是哪一支python api
        json_dict["code"] = code #目前程式執行的結果狀態 0: 成功 1: 失敗
        json_dict["msg"] = msg #描述狀態訊息
       
        with open(json_file_path, 'w', encoding='utf8') as json_file:
            json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False) #讓json能中文顯示

    try:
        logging.info('!!! ' + case_name + ' gpu_infarct call.')

        #把一些json要記錄的資訊弄出空集合或欲填入的值，這樣才能留空資訊
        ret = "gpu_infarct.py " #使用的程式是哪一支python api
        code_pass = 0 #確定是否成功
        msg = "ok" #描述狀態訊息
        
        #%% Deep learning相關
        pynvml.nvmlInit() #初始化
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_n)#获取GPU i的handle，后续通过handle来处理
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)#通过handle获取GPU i的信息
        gpumRate = memoryInfo.used/memoryInfo.total
        #print('gpumRate:', gpumRate) #先設定gpu使用率小於0.2才跑predict code
    
        if gpumRate < 0.6 :
            #plt.ion()    # 開啟互動模式，畫圖都是一閃就過
            #一些記憶體的配置
            autotune = tf.data.experimental.AUTOTUNE
            #print(keras.__version__)
            #print(tf.__version__)
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
            tf.config.experimental.set_visible_devices(devices=gpus[gpu_n], device_type='GPU')
            #print(gpus, cpus)
            tf.config.experimental.set_memory_growth(gpus[0],True)

            #%%
            path_normimg = os.path.join(path_process, 'Normalized_Image')
            path_mask = os.path.join(path_process, 'Brain')

            if not os.path.isdir(path_normimg):
                os.mkdir(path_normimg)
            if not os.path.isdir(path_mask):
                os.mkdir(path_mask)

            #複製檔案到指定的資料夾底下就可以跑nnU-Net
            shutil.copy(os.path.join(path_process, 'BET_ADC.nii.gz'), os.path.join(path_normimg, 'DeepInfarct_00001_0000.nii.gz'))
            shutil.copy(os.path.join(path_process, 'BET_DWI1000.nii.gz'), os.path.join(path_normimg, 'DeepInfarct_00001_0001.nii.gz'))
            shutil.copy(os.path.join(path_process, 'BET_mask.nii.gz'), os.path.join(path_mask, 'DeepInfarct_00001_0000.nii.gz'))
            
            print("[save] --->", os.path.join(path_normimg, 'DeepInfarct_00001_0000.nii.gz'))  
            print("[save] --->", os.path.join(path_normimg, 'DeepInfarct_00001_0000.nii.gz'))   
            print("[save] --->", os.path.join(path_mask, 'DeepInfarct_00001_0000.nii.gz'))

            predict_from_raw_data(path_normimg,
                                  path_mask,
                                  path_process,
                                  path_nnunet_model,
                                  (0,),
                                  0.25,
                                  use_gaussian=True,
                                  use_mirroring=False,
                                  perform_everything_on_gpu=True,
                                  verbose=True,
                                  save_probabilities=False,
                                  overwrite=False,
                                  checkpoint_name='checkpoint_best.pth',
                                  plans_json_name='plans.json',  # 可以根據需要修改json檔名
                                  has_classifier_output=False,  # 如果模型有classifier輸出則設為True
                                  num_processes_preprocessing=2,
                                  num_processes_segmentation_export=3,
                                  desired_gpu_index = 0,
                                  batch_size=64
                                 )
            
            #複製inference result
            path_nnunet = os.path.join(path_process, 'nnUNet')

            shutil.copy(os.path.join(path_process, 'DeepInfarct_00001.nii.gz'), os.path.join(path_nnunet, 'Prob.nii.gz')) #取threshold跟cluster放到後面做

            #用threshold修改輸出
            prob_nii = nib.load(os.path.join(path_nnunet, 'Prob.nii.gz'))
            prob = np.array(prob_nii.dataobj) #讀出label的array矩陣      #256*256*22   
            prob = data_translate(prob, prob_nii)
        
            #改讀取mifti的pixel_size資訊
            header_true = prob_nii.header.copy() #抓出nii header 去算體積 
            pixdim = header_true['pixdim']  #可以借此從nii的header抓出voxel size
        
            spacing_nn = [pixdim[1], pixdim[2], pixdim[3]]  # [x, y, z] 方向的 spacing
        
            pred_prob_map, df_pred, new_pred_label = filter_infarct_by_volume(prob, spacing_nn, conf_th=0.1, min_volume=0.3, top_k=30, obj_th=0.5)
        
            #最後存出新mask，存出nifti
            new_pred_label = data_translate_back(new_pred_label, prob_nii).astype(int)
            new_pred_label_nii = nii_img_replace(prob_nii, new_pred_label)
            nib.save(new_pred_label_nii, os.path.join(path_nnunet, 'Pred.nii.gz'))  

            #以json做輸出
            time.sleep(1)
            logging.info('!!! ' + str(case_name) +  ' gpu_infarct finish.')

        else:
            logging.error('!!! ' + str(case_name) + ' Insufficient GPU Memory.')
            #以json做輸出
            code_pass = 1
            msg = "Insufficient GPU Memory"
            
            # #刪除資料夾
            # if os.path.isdir(path_process):  #如果資料夾存在
            #     shutil.rmtree(path_process) #清掉整個資料夾 

    except:
        logging.error('!!! ' + str(case_name) + ' gpu have error code.')
        logging.error("Catch an exception.", exc_info=True)
        #以json做輸出
        code_pass = 1
        msg = "have error code"
        #刪除資料夾
        # if os.path.isdir(path_process):  #如果資料夾存在
        #     shutil.rmtree(path_process) #清掉整個資料夾
 
    return

#其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    #model_predict_aneurysm(path_code, path_processID, ID, path_log, gpu_n)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_code', type=str, help='目前執行的code')
    parser.add_argument('--path_process', type=str, help='目前執行的資料夾')
    parser.add_argument('--path_nnunet_model', type=str, help='nnU-Net model路徑')
    parser.add_argument('--case', type=str, help='目前執行的case的ID')
    parser.add_argument('--path_log', type=str, help='log資料夾')
    parser.add_argument('--gpu_n', type=int, help='第幾顆gpu')
    args = parser.parse_args()

    path_code = str(args.path_code)
    path_process = str(args.path_process)
    path_nnunet_model = str(args.path_nnunet_model)
    case_name = str(args.case)
    path_log = str(args.path_log)
    gpu_n = args.gpu_n

    model_predict_infarct(path_code, path_process, path_nnunet_model, case_name, path_log, gpu_n)
