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
from util import class_studydescriptoion, dcm2nii, class_studydescriptoion, move_T2FLAIR, dcm2nii, T2FLAIR_nor, check_tag_num_wmh
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
#import tensorflow_addons as tfa
#import tensorflow_datasets as tfds
from IPython.display import clear_output
import math
import scipy
#import base64
from collections import OrderedDict
import matplotlib.colors as mcolors
import numba as nb 
#from revise import revise2d, revise3d
from gpu_wmh import model_predict_wmh


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


@nb.jit(nopython=True, nogil=True)
def parcellation(nii_array, x1, x2, c1, new_array):
    #先整理出包含真實值的x2(coordinate_class1)與y2(coordinate_class2)
    
    #for k in range(len(x1[0])):
    y_coord = x1[0]
    x_coord = x1[1]
    for y_i, x_i in zip(y_coord, x_coord):
        max_v = nii_array.shape[0]*nii_array.shape[0] + nii_array.shape[1]*nii_array.shape[1]
        for l in range(c1.shape[0]):
            #一個點k做4次因為不支援帶參數的sort
            #mm110[l,0] = np.sort((x2[l,0,:] - y_i)*(x2[l,0,:] - y_i) + (x2[l,1,:] - x_i)*(x2[l,1,:] - x_i) + (x2[l,2,:] - z_i)*(x2[l,2,:] - z_i))[0]
            new_x2 = (x2[l,0,:] - y_i)*(x2[l,0,:] - y_i) + (x2[l,1,:] - x_i)*(x2[l,1,:] - x_i)
            #new_x2 = np.square(x2[l,0,:] - y_i) + np.square(x2[l,1,:] - x_i) + np.square(x2[l,2,:] - z_i)
            #new_x2 = (x2[l,0,:] - y_i) + (x2[l,1,:] - x_i) + (x2[l,2,:] - z_i)
            for k in new_x2:
                if max_v > k:
                    max_v = k
                    new_array[y_i, x_i] = c1[l]
        #if k % 10000 == 0:  # print the progress once every 10000 iterations
        #    print('First: ', k , ' too slow, Total:', len(x1[0]))
    return new_array

#case_json(json_path_name, InfarctVolume, MeanADC, Report)     
def case_json(json_file_path, ID, volume, Fazekas, Report):
    json_dict = OrderedDict()
    json_dict["PatientID"] = ID #使用的程式是哪一支python api
    json_dict["volume"] = volume #infarct core 體積(ml)
    json_dict["Fazekas"] = Fazekas 
    json_dict["Report"] = Report       

    with open(json_file_path, 'w', encoding='utf8') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False) #讓json能中文顯示


def pipeline_wmh(ID, 
                T2FAIR_file, 
                SynthSEG_WM_file, 
                SynthSEG_file, 
                path_output,
                path_code = '/data/pipeline/chuan/code/', 
                path_processModel = '/data/pipeline/chuan/process/Deep_WMH/', 
                path_json = '/data/pipeline/chuan/json/',
                path_log = '/data/pipeline/chuan/log/', 
                cuatom_model = '/data/pipeline/chuan/code/model_weights/Unet-0084-0.16622-0.19441-0.87395.h5', 
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

    logging.info('!!! Pre_Infarct call.')

    path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)
    if not os.path.isdir(path_processID):  #如果資料夾不存在就建立
        os.mkdir(path_processID) #製作nii資料夾

    print(ID, ' Start...')

    #依照不同情境拆分try需要小心的事項 <= 重要
    try:
        path_nii = os.path.join(path_processID, 'nii')
        if not os.path.isdir(path_nii):  #如果資料夾不存在就建立
            os.mkdir(path_nii) #製作nii資料夾

        #將影像複製過去
        #print('ADC_file:', ADC_file, ' copy:', os.path.join(path_nii, ID + '_ADC.nii.gz'))
        shutil.copy(T2FAIR_file, os.path.join(path_nii, ID + '_T2FLAIR.nii.gz'))
        shutil.copy(SynthSEG_WM_file, os.path.join(path_nii, ID + '_SynthSEG_WM.nii.gz'))
        shutil.copy(SynthSEG_file, os.path.join(path_nii, ID + '_SynthSEG.nii.gz'))

        #把T2FLAIR_nor進行正規化
        path_test_img = os.path.join(path_processID, 'test_case')
        if not os.path.isdir(path_test_img):  #如果資料夾不存在就建立
            os.mkdir(path_test_img) #製作nii資料夾        
        T2FLAIR_nor(path_processID, path_test_img, ID)
    
        time.sleep(1) #做前暫停1秒

        path_tsnpy = os.path.join(path_test_img, 'npy')

        #最後回去確認test_img的數據是否存在，存在則結束
        test_npy_t2flair = os.path.join(path_tsnpy, ID + '_T2FLAIR.npy')

        if os.path.isfile(test_npy_t2flair): 
            time.sleep(0.0001) #做前暫停0.0001秒
            logging.info('!!! ' + ID + ' pre_wmh finish.')
        else:
            #以json做輸出
            logging.error('!!! ' + ID + ' processing is error.') 

            # #刪除資料夾
            # if os.path.isdir(path_processID):  #如果資料夾存在
            #     shutil.rmtree(path_processID) #清掉整個資料夾
        
        #接下來，以下運行gpu領域
        #以下做predict，為了避免gpu out of memory，還是以.sh來執行好惹
        #gpu_line = 'bash ' + path_code + 'gpu_stroke.sh ' + ID + ' ' + path_processID
        #os.system(gpu_line)

        #因為松諭會用排程，所以這邊改成call function不管gpu了
        model_predict_wmh(path_code, path_processID, cuatom_model, ID, path_log, gpu_n)

        #底下正式開始畫圖任務
        ROWS=512 #y
        COLS=512 #x

        #predict => .png。這邊讀取的影像都是有去過頭皮的了
        T2FLAIR_nii = nib.load(os.path.join(path_nii, ID + '_T2FLAIR.nii.gz')) #讀取影像        
        header_true = T2FLAIR_nii.header.copy() #抓出nii header 去算體積 
        pixdim = header_true['pixdim']  #可以借此從nii的header抓出voxel size
        #print('pixdim:',pixdim)
        ml_size = (pixdim[1]*pixdim[2]*pixdim[3]) / 1000 #轉換成ml
        #ml_size = (dcm[0x28, 0x0030].value[0]*dcm[0x28, 0x0030].value[1]*dcm[0x18, 0x0088].value) / 1000 #轉換成ml
        T2FLAIR_array = np.load(os.path.join(path_tsnpy, ID + '_T2FLAIR.npy')) #讀出image的array矩陣，用nii讀影像的話會有背景問題
        SynthSEG_WM_nii = nib.load(os.path.join(path_nii, ID + '_SynthSEG_WM.nii.gz')) #讀取SynthSEG，把頭皮去除(為了rapid)
        SynthSEG_WM_array = np.array(SynthSEG_WM_nii.dataobj) #讀出image的array矩陣
        SynthSEG_nii = nib.load(os.path.join(path_nii, ID + '_SynthSEG.nii.gz')) #讀取SynthSEG，把頭皮去除(為了rapid)
        SynthSEG_array = np.array(SynthSEG_nii.dataobj) #讀出image的array矩陣
        #T2FLAIR_array = data_translate(T2FLAIR_array, T2FLAIR_nii)    #change image to use cnn
        SynthSEG_WM_array = data_translate(SynthSEG_WM_array, SynthSEG_WM_nii)
        SynthSEG_array = data_translate(SynthSEG_array, SynthSEG_nii)
        #print('T2FLAIR_array.shape:', T2FLAIR_array.shape)
        z_i, y_i, x_i , c_i = T2FLAIR_array.shape #得出尺寸

        #將結果計算，統計出原版跟CT2MRI的比較，predict mask都是屬於轉正的狀態，跟nifti不一樣
        #original_mask = np.load(os.path.join(path_processID, 'predict_map', ID + '.npy'))
        #z_o, y_o, x_o = original_mask.shape

        #因為cronjob切點跟不同影像z軸大小的狀況，切成2種張碼去展示
        if z_i >= 21:
            combine_num = [1,2,3,4,5,6,6,8,9,10,11,12,13,14,15,16,17,18,19,20]
            combine_s_num = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] #16張
        elif z_i >= 19 and z_i < 21:
            combine_num = [i for i in range(z_i)] #舊的12張，新的就20張
            combine_s_num = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] #16張
        else:
            combine_num = [i for i in range(z_i)] #舊的12張，新的就20張
            combine_s_num = combine_num

        slice_y = int((ROWS - y_i)/2)
        slice_x = int((COLS - x_i)/2)
            
        path_predict = os.path.join(path_processID, 'predict_map')
        #讀取predict的結果，已經修改成nifti
        Y_pred_nii = nib.load(os.path.join(path_predict, ID + '_prediction.nii.gz')) 
        Y_pred = np.array(Y_pred_nii.dataobj) 
        Y_pred = data_translate(Y_pred, T2FLAIR_nii)
        y_pred_nor1 = np.zeros(Y_pred.shape)
        y_pred_nor1[Y_pred >= 0.5] = 1
        y_pred_nor1[Y_pred < 0.5] = 0
        #y_pred_nor1是3d矩陣，可以直接用來計算cluster，再把小於1ml的cluster去除
        y_pred_nor1 = y_pred_nor1.astype('int')
        #print('np.max(y_pred_nor1):',np.max(y_pred_nor1),'np.min(y_pred_nor1):',np.min(y_pred_nor1))

        #去頭皮矩陣
        mask = np.zeros(Y_pred.shape)
        mask[SynthSEG_WM_array > 0] = 1
        mask_BET = np.zeros(Y_pred.shape)
        mask_BET[SynthSEG_array > 0] = 1

        y_m, x_m, z_m = mask.shape
        #找頭皮最上面是第幾張切片以往下切兩張
        brain_slice = 0 #做全腦的上半部就好
        for i_m in range(int(z_m/2)):
            if not np.sum(mask_BET[:,:,i_m]) >= 10:
                brain_slice += 1 #表示還沒到頭頂，所以繼續記數

        Y_pred_WMH = y_pred_nor1 * mask

        #因為資料集加上去頭皮程式(去不完整)，所以去除前2張跟後1/3的mask
        Y_pred_WMH[:,:,:int(brain_slice+2)] = 0
        back_site = int(0.67*Y_pred_WMH.shape[-1]) #後1/3張數的位置 -1 張
        Y_pred_WMH[:,:,back_site:] = 0
        Y_SynthSEG_WMH = Y_pred_WMH.copy() * SynthSEG_WM_array
        y_pred_show = Y_pred_WMH.copy()
        y_pred_show = (y_pred_show*255).astype('uint8')
        
        WM_array = SynthSEG_array.copy().astype(int)
        WM_array[SynthSEG_array==3] = 0

        #以下算出Fazekas，用wmh / 腦內
        if np.sum(WM_array) > 0:
            percentage = np.sum(Y_pred_WMH) / np.sum(WM_array) * 100
        else:
            percentage = 0.0
        if percentage == 0.0: level = 0
        elif percentage < 0.2923: level = 1
        elif percentage > 1.4825: level = 3
        else: level = 2
        Fazekas = level

        #存入各個計算數據y_pred
        #統計有標註的張數y_pred_cluster
        WMHSlice_list = [1 for y in range(z_m) if np.sum(Y_pred_WMH[:,:,y]) > 0] #如果有infarct的切片，記數為1
        WMHSliceNum = len(WMHSlice_list)

        #把WMH的mask大小與換算體積(ml)紀錄
        WMHVoxel = np.sum(Y_pred_WMH)
        total_volume = round(np.sum(Y_pred_WMH)*ml_size, 1)
        WMHVolume = total_volume

        #這邊先存出nifti結果
        new_y_pred_nor1 = data_translate_back(Y_pred_WMH, T2FLAIR_nii).astype(int)
        new_y_pred_nor1_nii = nii_img_replace(T2FLAIR_nii, new_y_pred_nor1)
        new_y_pred_synthseg = data_translate_back(Y_SynthSEG_WMH, T2FLAIR_nii).astype(int)
        new_y_pred_synthseg_nii = nii_img_replace(T2FLAIR_nii, new_y_pred_synthseg)
        nib.save(new_y_pred_nor1_nii, os.path.join(path_output, 'Pred_WMH.nii.gz'))
        nib.save(new_y_pred_synthseg_nii, os.path.join(path_output, 'Pred_WMH_synthseg.nii.gz'))


        #報告敘述是連圖上沒有的小點也顯示出來
        text = 'T2 hyperintense lesions, estimated about ' + str(WMHVolume) + ' ml, ' 
        text2 = ''

        text += 'suggestive of '
        if Fazekas <= 1:
            text += 'mild'
        elif Fazekas == 2:
            text += 'moderate'
        elif Fazekas == 3:
            text += 'severe'

        text += ' white matter hyperintensity, Fazekas ' + str(Fazekas) + '.<br><br>'
        text += '===================================================================================================<br>' #間隔線
        text += 'Total Volume: ' + str(WMHVolume) + ' ml, ' + 'Fazekas grade ' + str(Fazekas) + '.<br>'
        text += text2


        # #建置資料夾
        # if not os.path.isdir(os.path.join(path_json, ID)):  #如果資料夾不存在就建立
        #     os.mkdir(os.path.join(path_json, ID)) #製作nii資料夾

        #以json做輸出
        Report = text
        json_path_name = os.path.join(path_json, 'Pred_WMH.json')
        json_path_name2 = os.path.join(path_output, 'Pred_WMH.json')
        case_json(json_path_name, ID, WMHVolume, Fazekas, Report)  
        case_json(json_path_name2, ID, WMHVolume, Fazekas, Report)    

        logging.info('!!! ' + ID +  ' post_wmh finish.')


    except:
        logging.warning('Retry!!! have error code or no any study.')
        logging.error("Catch an exception.", exc_info=True)
        print('error!!!')
        #必須要清空opy資料夾跟class資料夾
        # copy_list = os.listdir(path_copy)
        # if len(copy_list) > 0:
        #     for i in range(len(copy_list)):
        #         os.remove(os.path.join(path_copy, copy_list[i])) #清除檔案
        # class_list = os.listdir(path_class)
        # if len(class_list) > 0:
        #     for i in range(len(class_list)):
        #         shutil.rmtree(os.path.join(path_class, class_list[i])) #清除檔案     
    
    print('end!!!')

#其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=str, default = '00003092_20201007_MR_20910070157', help='目前執行的case的patient_id or study id')
    parser.add_argument('--Inputs', type=str, nargs='+', default = ['/mnt/e/pipeline/chuan/example_input/00003092_20201007_MR_20910070157/T2FLAIR_AXI.nii.gz', '/mnt/e/pipeline/chuan/example_input/00003092_20201007_MR_20910070157/synthseg_T2FLAIR_AXI_original_WMH.nii.gz', '/mnt/e/pipeline/chuan/example_input/00003092_20201007_MR_20910070157/synthseg_T2FLAIR_AXI_original_synthseg5.nii.gz'], help='用於輸入的檔案')
    parser.add_argument('--Output_folder', type=str, default = '/mnt/e/pipeline/chuan/example_output/',help='用於輸出結果的資料夾')    
    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs  # 將列表合併為字符串，保留順序
    path_output = str(args.Output_folder)

    #讀出DWI, DWI0, ADC, SynthSEG的檔案
    T2FLAIR_file = Inputs[0]
    SynthSEG_WM_file = Inputs[1]
    SynthSEG_file = Inputs[2]

    #下面設定各個路徑
    path_code = '/mnt/e/pipeline/chuan/code/'
    path_process = '/mnt/e/pipeline/chuan/process/'  #前處理dicom路徑(test case)
    path_processModel = os.path.join(path_process, 'Deep_WMH')  #前處理dicom路徑(test case)
    #path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)

    #這裡先沒有dicom
    path_json = '/mnt/e/pipeline/chuan/json/'  #存放json的路徑，回傳執行結果
    #json_path_name = os.path.join(path_json, 'Pred_Infarct.json')
    path_log = '/mnt/e/pipeline/chuan/log/'  #log資料夾
    cuatom_model = '/mnt/e/pipeline/chuan/code/model_weights/Unet-0084-0.16622-0.19441-0.87395.h5'  #model路徑+檔案
    gpu_n = 0  #使用哪一顆gpu

    #建置資料夾
    if not os.path.isdir(path_processModel):  #如果資料夾不存在就建立
        os.mkdir(path_processModel) #製作nii資料夾

    #直接當作function的輸入
    pipeline_wmh(ID, T2FLAIR_file, SynthSEG_WM_file, SynthSEG_file, path_output, path_code, path_processModel, path_json, path_log, cuatom_model, gpu_n)

    # #最後再讀取json檔結果
    # with open(json_path_name) as f:
    #     data = json.load(f)

    # logging.info('Json!!! ' + str(data))