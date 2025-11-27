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
import keras
# from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, Activation, Add, Conv2D, Multiply, UpSampling2D, MaxPooling2D, Layer, LeakyReLU,Conv2DTranspose
from tensorflow.keras.layers import Activation, add, multiply, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers.experimental import SyncBatchNormalization
#from keras import backend as K
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import History
import os
import pickle
import tensorflow.keras.utils as conv_utils
from tensorflow.python.ops import nn_ops
from random import uniform
import math
import argparse
import logging
import json
import pydicom
import nibabel as nib
import psutil 
#pip install psutil
import pynvml #导包
#pip install pynvml
from sklearn.utils import shuffle
from scipy.ndimage.morphology import binary_fill_holes
from collections import OrderedDict
autotune = tf.data.experimental.AUTOTUNE
from nii_transforms import nii_img_replace


def model_predict_wmh(path_code, path_process, model_wmh_file, case_name, path_log, gpu_n):

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

    def case_json(json_file_path, ret, code, msg, chr_no, request_date, pat_name, AccessionNumber, StudyID, PatientBirth, PatientAge, PatientSex, PatientWeight, MagneticFieldStrength,
                  StudyDescription, ImagesAcquisition, PixelSpacing, SpacingBetweenSlices, WMHSliceNum, WMHVoxel, volume, Fazekas, Report):
        json_dict = OrderedDict()
        json_dict["ret"] = ret #使用的程式是哪一支python api
        json_dict["code"] = code #目前程式執行的結果狀態 0: 成功 1: 失敗
        json_dict["msg"] = msg #描述狀態訊息
        json_dict["chr_no"] = chr_no #patient id
        json_dict["request_date"] = request_date #studydate
        json_dict["pat_name"] = pat_name #病人姓名
        json_dict["AccessionNumber"] = AccessionNumber 
        json_dict["StudyID"] = StudyID 
        json_dict["PatientBirth"] = PatientBirth 
        json_dict["PatientAge"] = PatientAge         
        json_dict["PatientSex"] = PatientSex 
        json_dict["PatientWeight"] = PatientWeight 
        json_dict["MagneticFieldStrength"] = MagneticFieldStrength 
        json_dict["StudyDescription"] = StudyDescription 
        json_dict["ImagesAcquisition"] = ImagesAcquisition 
        json_dict["PixelSpacing"] = PixelSpacing 
        json_dict["SpacingBetweenSlices"] = SpacingBetweenSlices 
        json_dict["WMHSliceNum"] = WMHSliceNum 
        json_dict["WMHVoxel"] = WMHVoxel #infarct core 體積(voxel數)
        json_dict["volume"] = volume #infarct core 體積(ml) 
        json_dict["Fazekas"] = Fazekas 
        json_dict["Report"] = Report 
       
        with open(json_file_path, 'w', encoding='utf8') as json_file:
            json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False) #讓json能中文顯示

    try:
        logging.info('!!! ' + case_name + ' gpu_wmh call.')

        #把一些json要記錄的資訊弄出空集合或欲填入的值，這樣才能留空資訊
        ret = "gpu_wmh.py " #使用的程式是哪一支python api
        code_pass = 0 #確定是否成功
        msg = "ok" #描述狀態訊息

        #%% Deep learning相關
        pynvml.nvmlInit() #初始化
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_n)#获取GPU i的handle，后续通过handle来处理
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)#通过handle获取GPU i的信息
        gpumRate = memoryInfo.used/memoryInfo.total
        #print('gpumRate:', gpumRate) #先設定gpu使用率小於0.2才跑predict code

        if gpumRate < 0.999 :
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
            #底下正式開始predict任務
            ROWS=512 #y
            COLS=512 #x

            #先跑SynthSEG就可以不需要用到2張gpu   +
            path_nii = os.path.join(path_process, 'nii')         
            path_test_img = os.path.join(path_process, 'test_case')
            path_predict = os.path.join(path_process, 'predict_map')
            path_npy = os.path.join(path_test_img, 'npy')

            if not os.path.isdir(path_test_img):
                os.mkdir(path_test_img) 
            if not os.path.isdir(path_predict):
                os.mkdir(path_predict) 
            if not os.path.isdir(path_npy):
                os.mkdir(path_npy) 

            # #但以下跑SynthSEG，為了避免gpu out of memory，還是以.sh來執行好惹
            # gpu_line = 'bash ' + os.path.join(path_code, 'SynthSEG_wmh.sh ') + case_name
            # #以下做predict，為了避免gpu out of memory，還是以.sh來執行好惹
            # #gpu_line = 'python ' + os.path.join(path_synthseg, 'main.py -i ') + path_nii + ' --all False --WMH TRUE'
            # os.system(gpu_line)
            # time.sleep(2)

            
            #一些與model有關的設定     
            #%%
            # model, optimizer, and checkpoint must be created under `strategy.scope`.
            inputs = Input(shape=(512, 512, 1))
            enc = Conv2D(filters=64, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(inputs)
            enc = SyncBatchNormalization()(enc)
            enc = Activation('selu')(enc)
            #enc = LeakyReLU(alpha=0.3)(enc)
            enc = Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(enc)
            enc = SyncBatchNormalization()(enc)
            enc = Activation("selu")(enc)
            #enc = LeakyReLU(alpha=0.3)(enc)

            enc2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(enc)

            enc2 = Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(enc2)
            enc2 = SyncBatchNormalization()(enc2)
            enc2 = Activation("selu")(enc2)
            #enc2 = LeakyReLU(alpha=0.3)(enc2)
            enc2 = Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(enc2)
            enc2 = SyncBatchNormalization()(enc2)
            enc2 = Activation("selu")(enc2)
            #enc2 = LeakyReLU(alpha=0.3)(enc2)

            enc3 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(enc2)

            enc3 = Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(enc3)
            enc3 = SyncBatchNormalization()(enc3)
            enc3 = Activation("selu")(enc3)
            #enc3 = LeakyReLU(alpha=0.3)(enc3)
            enc3 = Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(enc3)
            enc3 = SyncBatchNormalization()(enc3)
            enc3 = Activation("selu")(enc3)
            #enc3 = LeakyReLU(alpha=0.3)(enc3)

            enc4 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(enc3)

            enc4 = Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(enc4)
            enc4 = SyncBatchNormalization()(enc4)
            enc4 = Activation("selu")(enc4)
            #enc4 = LeakyReLU(alpha=0.3)(enc4)
            enc4 = Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(enc4)
            enc4 = SyncBatchNormalization()(enc4)
            enc4 = Activation("selu")(enc4)
            #enc4 = LeakyReLU(alpha=0.3)(enc4)

            dec = Conv2DTranspose(512, kernel_size=(2,2), strides=(2, 2),padding='same', kernel_initializer='glorot_normal')(enc4)
            dec = Concatenate(axis=-1)([dec, enc3])
            dec = Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(dec)
            dec = SyncBatchNormalization()(dec)
            dec = Activation("selu")(dec)
            #dec = LeakyReLU(alpha=0.3)(dec)
            dec = Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(dec)
            dec = SyncBatchNormalization()(dec)
            dec = Activation("selu")(dec)
            #dec = LeakyReLU(alpha=0.3)(dec)

            dec2 = Conv2DTranspose(256, kernel_size=(2,2), strides=(2, 2),padding='same', kernel_initializer='glorot_normal')(dec)
            dec2 = Concatenate(axis=-1)([dec2, enc2])
            dec2 = Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(dec2)
            dec2 = SyncBatchNormalization()(dec2)
            dec2 = Activation("selu")(dec2)
            #dec2 = LeakyReLU(alpha=0.3)(dec2)
            dec2 = Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(dec2)
            dec2 = SyncBatchNormalization()(dec2)
            dec2 = Activation("selu")(dec2)
            #dec2 = LeakyReLU(alpha=0.3)(dec2)

            dec3 = Conv2DTranspose(128, kernel_size=(2,2), strides=(2, 2),padding='same', kernel_initializer='glorot_normal')(dec2)
            dec3 = Concatenate(axis=-1)([dec3, enc])
            dec3 = Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(dec3)
            dec3 = SyncBatchNormalization()(dec3)
            dec3 = Activation("selu")(dec3)
            #dec3 = LeakyReLU(alpha=0.3)(dec3)
            dec3 = Conv2D(filters=64, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(dec3)
            dec3 = SyncBatchNormalization()(dec3)
            dec3 = Activation("selu")(dec3)
            #dec3 = LeakyReLU(alpha=0.3)(dec3)

            outputs = Conv2D(filters=1, kernel_size=(1,1), kernel_initializer='glorot_normal')(dec3)
            outputs = Activation('sigmoid', name='sigmoid')(outputs)
            model = Model(inputs, outputs) 
            optimizer = tf.keras.optimizers.Adam(0.0001)

            #會使用到的一些predict技巧
            def data_translate(img):
                img = np.swapaxes(img,0,1)
                img = np.flip(img,0)
                img = np.flip(img, -1)
                # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
                return img
            
            #會使用到的一些predict技巧
            def data_translate_back(img):
                img = np.flip(img, -1)
                img = np.flip(img,0)
                img = np.swapaxes(img,1,0)
                # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
                return img

            #%%
            #load model
            #讀取模型
            #使用clone的方式不共享兩模型權重
            model_wmh = tf.keras.models.clone_model(model)
            model_wmh.load_weights(model_wmh_file) #最下面為val dice最高

            X = np.load(os.path.join(path_test_img, 'npy', case_name + '_T2FLAIR.npy')) #讀出image的array矩陣，用nii讀影像的話會有背景問題            

            #只要做WMH AI，不用跑去頭皮
            Y_pred = model_wmh.predict(X, batch_size=1) #Y_pred.shape => 23*512*512*1  512 512 23 1
            #Y_pred = model_wmh.predict(X)                               #  0  1   2  3   1   2   0  3
            np.save(os.path.join(path_predict, case_name + '_prediction.npy'), Y_pred)

            #在gpu程式中就要把predction用nifti保存才能接resample，先讀取nifti影像
            img_nii = nib.load(os.path.join(path_nii, case_name + '_T2FLAIR.nii.gz')) #讀取影像

            #預測軸方向交換 z, y, x => y, x, z
            Y_pred = np.transpose(Y_pred, (1, 2, 0, 3))
            Y_pred = Y_pred.swapaxes(0,1)
            Y_pred = np.flip(Y_pred, 2)
            Y_pred = np.flip(Y_pred, 1)
            Y_pred  = Y_pred[:,:,:,0]

            Y_pred_nii = nii_img_replace(img_nii, Y_pred)
            nib.save(Y_pred_nii, os.path.join(path_predict, case_name + '_prediction.nii.gz'))       
            
            #以json做輸出
            time.sleep(1)
            logging.info('!!! ' + str(case_name) +  ' gpu_wmh finish.')     

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, help='目前執行的case的study_uid')
    args = parser.parse_args()

    case_name = str(args.case)

    #下面設定各個路徑
    path_code = '/data/pipeline/chuan/code/'
    path_process = os.path.join('/data/pipeline/chuan/process/Deep_WMH', case_name)  #前處理dicom路徑(test case)
    #path_dcm = os.path.join(path_process, 'dicom')
    path_json = '/data/pipeline/chuan/json/'  #存放json的路徑，回傳執行結果
    #json_path_name = os.path.join(path_json, 'gpu_wmh.json')
    path_log = '/data/pipeline/chuan/log/'  #log資料夾
    model_wmh = '/data/pipeline/chuan/code/model_weights/Unet-0084-0.16622-0.19441-0.87395.h5'  #WMH model路徑+檔案
    
    gpu_n = 0  #使用哪一顆gpu

    model_predict_wmh(path_code, path_process, model_wmh, case_name, path_log, gpu_n)