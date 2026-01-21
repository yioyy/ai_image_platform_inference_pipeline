#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

predict model，非常重要:使用這個程式，安裝的版本都需要一致，才能運行
conda create -n orthanc-stroke tensorflow-gpu=2.3.0 anaconda python=3.7

@author: chuan
"""
import warnings
warnings.filterwarnings("ignore") # 忽略警告输出


import os
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
#from keras.models import Model
#from keras.layers import Input, Concatenate, BatchNormalization, Activation, Add, Conv2D, Multiply, UpSampling2D, MaxPooling2D, Layer,LeakyReLU,Conv2DTranspose
#from keras.layers import Activation, add, multiply, Lambda
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, Activation, Add, Conv2D, Multiply, UpSampling2D, MaxPooling2D, Layer, LeakyReLU,Conv2DTranspose
from tensorflow.keras.layers import Activation, add, multiply, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers.experimental import SyncBatchNormalization
#from keras import backend as K
import tensorflow.keras.backend as K
#print("Tensorflow version:", tf.__version__)
from tensorflow.keras.callbacks import History
import pickle
import tensorflow.keras.utils as conv_utils
from tensorflow.python.ops import nn_ops
from tensorflow.keras.utils import plot_model
import random
from random import uniform
#import tensorflow_addons as tfa
#import tensorflow_datasets as tfds
import math
import scipy
import argparse
import logging
import json
import psutil
import pynvml #导包
from collections import OrderedDict
from nii_transforms import nii_img_replace
from util import resampleSynthSEG2original, make_SynthSEG_Brain_mask, make_BET_ADC_DWI1000, ADC_DWI1000_nor
from tensorflow.keras.layers import *
autotune = tf.data.experimental.AUTOTUNE


def model_predict_stroke(path_code, path_process, cuatom_model, case_name, path_log, gpu_n):

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
                  StudyDescription, ImagesAcquisition, PixelSpacing, SpacingBetweenSlices, InfarctSliceNum, InfarctVoxel, volume, MeanADC, Report):
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
        json_dict["InfarctSliceNum"] = InfarctSliceNum 
        json_dict["InfarctVoxel"] = InfarctVoxel #infarct core 體積(voxel數)
        json_dict["volume"] = volume #infarct core 體積(ml) 
        json_dict["MeanADC"] = MeanADC 
        json_dict["Report"] = Report
       
        with open(json_file_path, 'w', encoding='utf8') as json_file:
            json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False) #讓json能中文顯示

    try:
        logging.info('!!! ' + case_name + ' gpu_stroke call.')

        #把一些json要記錄的資訊弄出空集合或欲填入的值，這樣才能留空資訊
        ret = "gpu_stroke.py " #使用的程式是哪一支python api
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
            #建立Infarct model之前先運行SynthSEG，應該可以使其用於同一張顯卡
            #底下正式開始predict任務
            ROWS=256 #y
            COLS=256 #x

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

            # #以下要先做SynthSEG加上去頭皮，再跑ai model，但以下跑SynthSEG，為了避免gpu out of memory，還是以.sh來執行好惹，由於後面有計算且暫時不用gpu，所以不加Sleep
            # gpu_line = 'bash ' + os.path.join(path_code, 'SynthSEG_stroke.sh ') + case_name
            # #以下做predict，為了避免gpu out of memory，還是以.sh來執行好惹
            # #gpu_line = 'python ' + os.path.join(path_synthseg, 'main.py -i ') + path_nii + ' --all False --WMH TRUE'
            # os.system(gpu_line)

            # #以下使用去頭皮矩陣
            # SynthSEG_array = resampleSynthSEG2original(path_process, case_name, 'DWI0', 'DWI') 


            #%%
            #一些與model有關的設定，因為model可能含有自定義的層，所以需要把model架構再放入程式碼
            #WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
            #model
            class ResBlock(Layer):
                """
                Represents the Residual Block in the ResUNet architecture.
                """
                def __init__(self, filters, strides, **kwargs):
                    super(ResBlock, self).__init__(**kwargs)
                    self.filters = filters
                    self.strides = strides

                    self.bn1 = BatchNormalization()
                    self.lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.01)
                    self.conv1 = Conv2D(filters=filters, kernel_size=3, strides=strides, padding="same", use_bias=False)

                    self.bn2 = BatchNormalization()
                    self.lrelu2 = tf.keras.layers.LeakyReLU(alpha=0.01)
                    self.conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False)

                    self.conv_skip = Conv2D(filters=filters, kernel_size=1, strides=strides, padding="same", use_bias=False)
                    self.bn_skip = BatchNormalization()

                    self.add = Add()

                def call(self, inputs, training=False, **kwargs):
                    x = inputs
                    x = self.bn1(x, training=training)
                    x = self.lrelu1(x)
                    x = self.conv1(x)

                    x = self.bn2(x, training=training)
                    x = self.lrelu2(x)
                    x = self.conv2(x)

                    skip = self.conv_skip(inputs)
                    skip = self.bn_skip(skip, training=training)

                    res = self.add([x, skip])
                    return res

                def get_config(self):
                    return dict(filters=self.filters, strides=self.strides, **super(ResBlock, self).get_config())


            def ResUNet(input_shape, classes: int, filters_root: int = 64, depth: int = 3):
                """
                Builds ResUNet model.
                :param input_shape: Shape of the input images (h, w, c). Note that h and w must be powers of 2.
                :param classes: Number of classes that will be predicted for each pixel. Number of classes must be higher than 1.
                :param filters_root: Number of filters in the root block.
                :param depth: Depth of the architecture. Depth must be <= min(log_2(h), log_2(w)).
                :return: Tensorflow model instance.
                """
                if classes < 1:
                    raise ValueError("The number of classes must be larger than 1.")
                if not math.log(input_shape[0], 2).is_integer() or not math.log(input_shape[1], 2):
                    raise ValueError(f"Input height ({input_shape[0]}) and width ({input_shape[1]}) must be power of two.")
                if 2 ** depth > min(input_shape[0], input_shape[1]):
                    raise ValueError(f"Model has insufficient height ({input_shape[0]}) and width ({input_shape[1]}) compared to its desired depth ({depth}).")

                input = Input(shape=input_shape)

                layer = input

                # ENCODER
                encoder_blocks = []

                filters = filters_root
                layer = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(layer)

                branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False)(layer)
                branch = BatchNormalization()(branch)
                branch = tf.keras.layers.LeakyReLU(alpha=0.01)(branch)
                branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=True)(branch)
                layer = Add()([branch, layer])

                encoder_blocks.append(layer)

                for _ in range(depth - 1):
                    filters *= 2
                    layer = ResBlock(filters, strides=2)(layer)

                    encoder_blocks.append(layer)

                # BRIDGE
                filters *= 2
                layer = ResBlock(filters, strides=2)(layer)

                # DECODER
                for i in range(1, depth + 1):
                    filters //= 2
                    skip_block_connection = encoder_blocks[-i]

                    layer = UpSampling2D()(layer)
                    layer = Concatenate()([layer, skip_block_connection])
                    layer = ResBlock(filters, strides=1)(layer)

                layer = Conv2D(filters=classes, kernel_size=1, strides=1, padding="same")(layer)

                if classes == 1:
                    layer = Activation(activation="sigmoid")(layer)
                else:
                    layer = Softmax()(layer)

                output = layer

                return Model(input, output)


            model = ResUNet(input_shape=(256, 256, 2), classes=1, filters_root=32, depth=5)
            
            #%%
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
            # load model
            model_unet = tf.keras.models.clone_model(model)
            model_unet.load_weights(cuatom_model) #最下面為val dice最高
            #model_unet = load_model(cuatom_model, custom_objects = {'model': model, 'loss_func': loss_func,'dice_coef':dice_coef,'dice_loss':dice_loss,'recall':recall,'precision':precision,'f1':f1,'y_pred_num':y_pred_num,'label_num':label_num})

            #在gpu程式中就要把predction用nifti保存才能接resample，先讀取nifti影像
            img_nii = nib.load(os.path.join(path_nii, case_name + '_DWI1000.nii.gz')) #讀取影像

            #以上前處理完成，開始predict => .png
            ADC_array = np.load(os.path.join(path_npy, case_name + '_ADC.npy')) #讀出image的array矩陣，用nii讀影像的話會有背景問題   #256*256*22
            DWI1000_array = np.load(os.path.join(path_npy, case_name + '_DWI1000.npy')) #讀出image的array矩陣，用nii讀影像的話會有背景問題    
            ADC_array = data_translate(ADC_array)    #change image to use cnn
            DWI1000_array = data_translate(DWI1000_array)    #change image to use cnn
            y_i, x_i , z_i = ADC_array.shape #得出尺寸

            slice_y = int((ROWS - y_i)/2)
            slice_x = int((COLS - x_i)/2)

            img_test = np.zeros((2,ROWS,COLS,z_i)) #2 because ADC and DWI1000.
            img_test[0,slice_y:slice_y+y_i,slice_x:slice_x+x_i,:] = ADC_array
            img_test[1,slice_y:slice_y+y_i,slice_x:slice_x+x_i,:] = DWI1000_array
            img_test = img_test.swapaxes(0,3) #交换轴0和3，因為model_predict_batch 是 張數,y,x,c
                
            #y_pred = model_unet.predict_on_batch(img_test) #y_pred為4維矩陣，結果2080ti竟然在跑24張時out of memory...，所以小改寫法，之後修正...
            #y_pred = model_unet.predict(img_test, batch_size = 10) #y_pred為4維矩陣，結果2080ti竟然在跑24張時out of memory...，所以小改寫法，之後修正...
            y_pred = model_unet.predict(img_test) #
            #y_pred_unet = categorise_prediction(np.argmax(y_pred, axis=-1), 0)
            y_pred_unet = y_pred[:,:,:,0]
            np.save(os.path.join(path_predict, case_name + '.npy'), y_pred_unet) 

            #以下轉向
            Y_pred = np.expand_dims(y_pred_unet, axis=-1)
            Y_pred = Y_pred.swapaxes(0,3) #交换轴0和3，因為model_predict_batch 是 張數,y,x,c
            Y_pred = Y_pred[0,:,:,:]
            Y_pred = data_translate_back(Y_pred)
            Y_pred_nii = nii_img_replace(img_nii, Y_pred)
            nib.save(Y_pred_nii, os.path.join(path_predict, case_name + '_prediction.nii.gz'))

            #以json做輸出
            time.sleep(1)
            logging.info('!!! ' + str(case_name) +  ' gpu_stroke finish.')

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
    path_process = os.path.join('/data/pipeline/chuan/process/Deep_Infarct', case_name)  #前處理dicom路徑(test case)
    #path_dcm = os.path.join(path_process, 'dicom')
    path_json = '/data/pipeline/chuan/json/'  #存放json的路徑，回傳執行結果
    #json_path_name = os.path.join(path_json, 'gpu_stroke.json')
    path_log = '/data/pipeline/chuan/log/'  #log資料夾
    cuatom_model = '/data/pipeline/chuan/code/model_weights/Unet-0194-0.17124-0.14731-0.86344.h5' #model路徑+檔案
    gpu_n = 0  #使用哪一顆gpu

    model_predict_stroke(path_code, path_process, cuatom_model, case_name, path_log, gpu_n)
