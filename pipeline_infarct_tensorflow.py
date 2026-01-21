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
from util import class_studydescriptoion, move_ADC_DWI_T2FLAIR, check_tag_num_stroke, check_tag_num_stroke_currently, dcm2nii, nii_info_DWI1000_for_ADC, use_BET_DWI0, make_BET_mask, make_BET_ADC_DWI1000, ADC_DWI1000_nor
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
from gpu_stroke import model_predict_stroke


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
def case_json(json_file_path, ID, volume, MeanADC, Report):
    json_dict = OrderedDict()
    json_dict["PatientID"] = ID #使用的程式是哪一支python api
    json_dict["volume"] = volume #infarct core 體積(ml)
    json_dict["MeanADC"] = MeanADC 
    json_dict["Report"] = Report       

    with open(json_file_path, 'w', encoding='utf8') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False) #讓json能中文顯示


def pipeline_infarct(ID, 
                     ADC_file, 
                     DWI0_file, 
                     DWI1000_file, 
                     SynthSEG_file, 
                     path_output,
                     path_code = '/mnt/e/pipeline/chuan/code/', 
                     path_processModel = '/mnt/e/pipeline/chuan/process/Deep_Infarct/', 
                     path_json = '/mnt/e/pipeline/chuan/json/',
                     path_log = '/mnt/e/pipeline/chuan/log/', 
                     cuatom_model = '/mnt/e/pipeline/chuan/code/model_weights/ResUnet-025-0.14466-0.29462-0.81407.h5', 
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
        shutil.copy(ADC_file, os.path.join(path_nii, ID + '_ADC.nii.gz'))
        shutil.copy(DWI0_file, os.path.join(path_nii, ID + '_DWI0.nii.gz'))
        shutil.copy(DWI1000_file, os.path.join(path_nii, ID + '_DWI1000.nii.gz'))
        shutil.copy(SynthSEG_file, os.path.join(path_nii, ID + '_SynthSEG.nii.gz'))

        #把DWI1000的info複製到ADC
        nii_info_DWI1000_for_ADC(path_nii, ID)
        #將DWI0使用BET處理
        path_BET = os.path.join(path_processID, 'BET_DWI0')
        if not os.path.isdir(path_BET):  #如果資料夾不存在就建立
            os.mkdir(path_BET)
        use_BET_DWI0(path_nii, path_BET, ID)
        #將BET_DWI0做成mask
        path_mask = os.path.join(path_processID, 'BET_mask')
        if not os.path.isdir(path_mask):  #如果資料夾不存在就建立
            os.mkdir(path_mask)
        make_BET_mask(path_BET, path_mask, ID)
        #將BET過的DWI0影像當作標準，切割DWI1000跟ADC影像成BET
        make_BET_ADC_DWI1000(path_processID, ID)
        #把BET_ADC跟BET_DWI1000進行正規化
        path_test_img = os.path.join(path_processID, 'test_case')
        ADC_DWI1000_nor(path_processID, ID, path_test_img)
    
        time.sleep(1) #做前暫停1秒

        path_tsnii = os.path.join(path_test_img, 'nii')
        path_tsnpy = os.path.join(path_test_img, 'npy')

        #最後回去確認test_img的數據是否存在，存在則結束
        test_nii_adc = os.path.join(path_tsnii, ID + '_ADC.nii.gz') 
        test_nii_dwi = os.path.join(path_tsnii, ID + '_DWI1000.nii.gz')
        test_npy_adc = os.path.join(path_tsnpy, ID + '_ADC.npy')
        test_npy_dwi = os.path.join(path_tsnpy, ID + '_DWI1000.npy')

        if os.path.isfile(test_nii_adc) and os.path.isfile(test_nii_dwi) and os.path.isfile(test_npy_adc) and os.path.isfile(test_npy_dwi): 
            time.sleep(0.0001) #做前暫停0.0001秒
            logging.info('!!! ' + ID + ' pre_stroke finish.')
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
        model_predict_stroke(path_code, path_processID, cuatom_model, ID, path_log, gpu_n)

        #底下正式開始畫圖任務
        ROWS=256 #y
        COLS=256 #x

        #predict => .png。這邊讀取的影像都是有去過頭皮的了
        ADC_nii = nib.load(os.path.join(path_nii, ID + '_ADC.nii.gz')) #讀取影像
        ADC_array = np.load(os.path.join(path_tsnpy, ID + '_ADC.npy')) #讀出image的array矩陣，用nii讀影像的話會有背景問題   #256*256*22
        DWI1000_nii = nib.load(os.path.join(path_nii, ID + '_DWI1000.nii.gz')) #讀取影像        
        header_true = DWI1000_nii.header.copy() #抓出nii header 去算體積 
        pixdim = header_true['pixdim']  #可以借此從nii的header抓出voxel size
        #print('pixdim:',pixdim)
        ml_size = (pixdim[1]*pixdim[2]*pixdim[3]) / 1000 #轉換成ml
        #ml_size = (dcm[0x28, 0x0030].value[0]*dcm[0x28, 0x0030].value[1]*dcm[0x18, 0x0088].value) / 1000 #轉換成ml
        DWI1000_array = np.load(os.path.join(path_tsnpy, ID + '_DWI1000.npy')) #讀出image的array矩陣，用nii讀影像的話會有背景問題
        SynthSEG_nii = nib.load(os.path.join(path_nii, ID + '_SynthSEG.nii.gz')) #讀取SynthSEG，把頭皮去除(為了rapid)
        SynthSEG_array = np.array(SynthSEG_nii.dataobj) #讀出image的array矩陣
        ADC_array = data_translate(ADC_array, ADC_nii)    #change image to use cnn
        DWI1000_array = data_translate(DWI1000_array, DWI1000_nii)    #change image to use cnn
        SynthSEG_array = data_translate(SynthSEG_array, SynthSEG_nii)
        y_i, x_i , z_i = ADC_array.shape #得出尺寸

        #將結果計算，統計出原版跟CT2MRI的比較，predict mask都是屬於轉正的狀態，跟nifti不一樣
        #original_mask = np.load(os.path.join(path_processID, 'predict_map', ID + '.npy'))
        #z_o, y_o, x_o = original_mask.shape

        #因為cronjob切點跟不同影像z軸大小的狀況，切成2種張碼去展示
        if z_i >= 21:
            combine_num = [20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1] #舊的12張，新的就20張
            combine_s_num = [18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3] #16張
        elif z_i >= 19 and z_i < 21:
            combine_num = [i for i in range(z_i)] #舊的12張，新的就20張
            combine_num = combine_num[::-1]
            combine_s_num = [18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3] #16張
        else:
            #combine_num = [2,3,5,6,8,9,11,12,14,16,17,18]
            #combine_num = [18,17,16,14,12,11,9,8,6,5,3,2] #舊的12張，新的就20張
            combine_num = [i for i in range(z_i)] #舊的12張，新的就20張
            combine_num = combine_num[::-1]
            combine_s_num = combine_num

        slice_y = int((ROWS - y_i)/2)
        slice_x = int((COLS - x_i)/2)
            
        path_predict = os.path.join(path_processID, 'predict_map')
        #讀取predict的結果，已經修改成nifti
        Y_pred_nii = nib.load(os.path.join(path_predict, ID + '_prediction.nii.gz')) 
        Y_pred = np.array(Y_pred_nii.dataobj) 
        Y_pred = data_translate(Y_pred, DWI1000_nii)
        y_pred_nor1 = np.zeros(Y_pred.shape)
        y_pred_nor1[Y_pred >= 0.5] = 1
        y_pred_nor1[Y_pred < 0.5] = 0
        #y_pred_nor1是3d矩陣，可以直接用來計算cluster，再把小於1ml的cluster去除
        y_pred_nor1 = y_pred_nor1.astype('int')
        #print('np.max(y_pred_nor1):',np.max(y_pred_nor1),'np.min(y_pred_nor1):',np.min(y_pred_nor1))

        #以下計算出ADCth的Pred
        Pred_ADCth = (y_pred_nor1 * ADC_array * 600).copy()
        #下面乘以SynthSEG的mask
        Pred_SynthSEG = (y_pred_nor1 * SynthSEG_array).copy()

        #這邊先存出nifti結果
        new_y_pred_nor1 = data_translate_back(y_pred_nor1, DWI1000_nii).astype(int)
        new_y_pred_nor1_nii = nii_img_replace(DWI1000_nii, new_y_pred_nor1)
        nib.save(new_y_pred_nor1_nii, os.path.join(path_output, 'Pred_Infarct.nii.gz'))

        Pred_ADCth = data_translate_back(Pred_ADCth, DWI1000_nii).astype(int)
        Pred_ADCth_nii = nii_img_replace(DWI1000_nii, Pred_ADCth)
        nib.save(Pred_ADCth_nii, os.path.join(path_output, 'Pred_Infarct_ADCth.nii.gz'))

        Pred_SynthSEG = data_translate_back(Pred_SynthSEG, DWI1000_nii).astype(int)
        Pred_SynthSEG_nii = nii_img_replace(DWI1000_nii, Pred_SynthSEG)
        nib.save(Pred_SynthSEG_nii, os.path.join(path_output, 'Pred_Infarct_synthseg.nii.gz'))

        y_pred_cluster = measure.label(y_pred_nor1,connectivity=3)  #Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor => 3d => 26
        cluster_num = np.max(y_pred_cluster) 
        #分類unet的cluster，然後把大小小於1ml的去除
        ml_voxel = 1/ml_size #算出1 ml需要多少voxel
        #print('ml_voxel:',ml_voxel)
        for cluster_i in range(cluster_num):
            y_cl, x_cl, z_cl = np.where(y_pred_cluster==(cluster_i +1))
            if len(y_cl) < 0.3*ml_voxel or np.mean(ADC_array[y_cl,x_cl,z_cl]) > (800/600) :
                y_pred_cluster[y_cl,x_cl, z_cl] = 0 #把過小的cluster去除(小於0.2ml), #把ADC過亮的cluster去除(mean小於800)

        y_pred_cluster[y_pred_cluster>0] = 1 #最後把其他的cluster==1 #去算出剩下的大顆的統計值，因為沒要分顆，所以可以這樣做

        #統計有標註的張數y_pred_cluster
        InfarctSlice_list = [1 for y in range(z_i) if np.sum(y_pred_cluster[:,:,y]) > 0] #如果有infarct的切片，記數為1
        InfarctSliceNum = len(InfarctSlice_list)

        #計算mask下的平均ADC是多少，寫入影像中，因為正規化ADC的方式是/600，所以最後乘600就好
        total_mean_ADC = np.sum(ADC_array*y_pred_cluster)
        if np.sum(y_pred_cluster) > 0:
            MeanADC = int((total_mean_ADC/np.sum(y_pred_cluster))*600) #轉成整數
        else:
            MeanADC = 0

        #把Infarct的mask大小與換算體積(ml)紀錄
        InfarctVoxel = int(np.sum(y_pred_cluster))

        #因為體積2張圖要顯示相同，所以重分區再加總要先做，但total有分能畫分區圖或是不能畫，所以分區前的體積還是要計算保留
        InfarctVolume = round(InfarctVoxel*ml_size, 1) #計算總體積，無條件進位

        #%%以下做有關SynthSEG所有計算跟圖，要重分區CSF跟Background
        json_file = os.path.join(path_code, 'dataset.json')
        #讀取json以獲得資料集參數
        with open(json_file) as f:
            json_data = json.load(f)

        Infarct_labels = json_data['Infarct_labels_OHIF'] 
        colorbar = json_data['Infarct_colors_OHIF'] #標籤都固定顏色

        outside_infarct_labels = {}
        outside_labels = []
        for idx_labels, label in enumerate(Infarct_labels):
            if label != 'Background' and label != 'CSF':
                outside_infarct_labels[label] = Infarct_labels[label]
                for jdx_labels, seg_id in enumerate(Infarct_labels[label]):
                    outside_labels.append(seg_id)
                
        outside_labels = np.array(outside_labels)

        #實際上是mask在CSF、背景要重分區，所以使用BET去頭皮，在BET跟CSF、Backgound的地方重分區
        BET_SynthSeg_array = SynthSEG_array.copy() 
        BET_SynthSeg_array[BET_SynthSeg_array == Infarct_labels['Background']] = 0 #去CSF、Backgound，因為CSF要重分區
        BET_SynthSeg_array[BET_SynthSeg_array == Infarct_labels['CSF']] = 0 #去CSF、Backgound，因為CSF要重分區
        label_overlap = y_pred_cluster.copy() * BET_SynthSeg_array.copy() #因為1*值=值，0*值=0，label超出去的地方會變0
        
        #取出在BET上沒分到SynthSEG的label => 這裡只取出重分配的CSF
        SynthSEG_mirror = (SynthSEG_array == Infarct_labels['CSF']) + (SynthSEG_array == Infarct_labels['Background'])
        label_outside = SynthSEG_mirror*y_pred_cluster.copy()

        outside_array = np.zeros(SynthSEG_array.shape)

        #3d整個矩陣吃不下，2D一層一層做，有label才做
        for i in range(z_i):
            if np.sum(label_outside[:,:,i]) > 0 and np.sum(BET_SynthSeg_array[:,:,i]) > 0:
                label_slice = label_outside[:,:,i]
                SynthSeg_slice = BET_SynthSeg_array[:,:,i]
                new_slice = np.zeros((SynthSeg_slice.shape))
                intersection = np.intersect1d(outside_labels, np.unique(SynthSeg_slice)) #取交集
                coordinate = np.array(np.where(label_slice==1)) #[[1,2],[3,4],[5,6]]
                long_class = max([len(np.where(SynthSeg_slice==y)[0]) for y in intersection])
                coordinate_class = np.zeros((len(intersection), 2, long_class)) #class*xy*maxlong
            
                #np.where不適合用於加速，丟到外面去
                for i_c in range(intersection.shape[0]):
                    c_coord = np.array(np.where(SynthSeg_slice==intersection[i_c]))
                    coordinate_class[i_c, :, :c_coord.shape[1]] = c_coord
                    coordinate_class[i_c, 0, c_coord.shape[1]+1:] = c_coord[0,0]
                    coordinate_class[i_c, 1, c_coord.shape[1]+1:] = c_coord[1,0]
                
                new_slice = parcellation(SynthSeg_slice, coordinate, coordinate_class, intersection, new_slice) 
                outside_array[:,:,i] = new_slice

        """
        # 計算距離的各群，1mm的因為標註會跑離腦很多，所以要用3d重分區
        SynthSeg_revise = SynthSEG_array.copy()
        SynthSeg_revise[label_outside==1] = 10000
        #先找出有SynthSEG的第一層
        top_seg = np.min(np.where(np.sum(SynthSEG_array, axis = (0,1)) > 0)[0])
        if np.max(SynthSEG_array[:,:,top_seg]) > 1:
            outside_array = revise3d(10000, SynthSeg_revise, miss_target=[Infarct_labels['Background'], Infarct_labels['CSF']])
        else:
            outside_array = revise3d(10000, SynthSeg_revise, miss_target=[Infarct_labels['Background'], Infarct_labels['CSF']])
        outside_array = outside_array * label_outside #只留下label那些
        """

        #將補做的SynthSeg_array加入原本的SynthSEG
        new_SynthSeg_array = label_overlap + outside_array

        #開始計算每個區塊pred的大小，每個標籤都有固定的顏色，每個區塊pred的大小沒要由大到小，照順序
        new_Infarct_labels = {}
        total_seg_volume = 0
        for idx_labels, label in enumerate(Infarct_labels):
            if label == 'Background' or label == 'CSF':
                pass
            else:
                synseg_ml = 0
                for j_labels, label_index in enumerate(Infarct_labels[label]):
                    synseg_ml += np.sum(new_SynthSeg_array == label_index)*ml_size #變true or false矩陣
                if round(synseg_ml, 1) >= 0.3:
                    new_Infarct_labels[label] = round(synseg_ml, 1) #變true or false矩陣
                total_seg_volume += synseg_ml

        #根據new_Infarct_labels長度表現不同情形，show的長度最長定義為15
        Infarct_labels_show = {}
        color_num = 0
        if len(new_Infarct_labels) >= 10:
            #只列1mm以上
            for idx_labels, label in enumerate(new_Infarct_labels):
                if new_Infarct_labels[label] >= 1 and color_num < 15:
                    Infarct_labels_show[label] = new_Infarct_labels[label]
                    color_num +=1
        else:
            #全show
            for idx_labels, label in enumerate(new_Infarct_labels):
                Infarct_labels_show[label] = new_Infarct_labels[label] 

        if total_seg_volume > InfarctVolume:
            InfarctVolume = round(total_seg_volume, 1)

        #報告敘述是連圖上沒有的小點也顯示出來
        if InfarctVolume > 0:
            text = 'DWI bright, ADC dark (mean ADC value: ' + str(MeanADC) + ') lesions, estimated about ' + str(InfarctVolume) + ' ml'
            text2 = ''
            
            if len(new_Infarct_labels) > 0:
                text += ', involving ' 
            
                #用for迴圈取出涉及有Infarct的區域
                for idx_label, new_label in enumerate(new_Infarct_labels):
                    text += new_label.lower() + ', '
                    text2 += new_label + ': ' + str(new_Infarct_labels[new_label]) + ' ml<br>'

                text += 'suggestive of recent infarction.<br><br>'
                text += '===================================================================================================<br>' #間隔線
            else:
                text += '.'

            text += text2
        else:
            text = 'No definite diffusion restricted lesion in the study.'

        #show predict png
        y_pred_show = y_pred_cluster.copy()
        y_pred_show = (y_pred_show*255).astype('uint8')

        for j in range(z_i):
            thresh = cv2.Canny(y_pred_show[:,:,j], 128, 256).copy()
            thresh = thresh.astype('int')
            A_show = np.zeros((ROWS, COLS))
            D_show = np.zeros((ROWS, COLS))        
            A_show[slice_y:slice_y+y_i,slice_x:slice_x+x_i] = ADC_array[:,:,j].copy()
            D_show[slice_y:slice_y+y_i,slice_x:slice_x+x_i] = DWI1000_array[:,:,j].copy()
            min_value = np.min(D_show)
            Dback_v = D_show[2,2]
            #print('Dback_v:', Dback_v)
            D_show[D_show==Dback_v] = min_value  

        # 正常显示中文字体
        #plt.rcParams['font.sans-serif']=['Microsoft YaHei']
        plt.style.use('dark_background') #使用黑色當背景色，繪圖風格
        fig = plt.figure() #開figure
        plt.axis('off')
        #left, bottom, width, height = 0.05, 0.05, 0.95, 0.95
        #ax1 = fig.add_axes([left, bottom, width, height])  # main axes
        #因為有沒有stroke病灶的狀況，當然就沒有平均，所以用一個判別式去區分
        if np.sum(y_pred_cluster) > 0:
            volume_result = '\n\n(For Research Purpose Only)'  #只顯示兩位數
        else:
            volume_result = '\n\n(For Research Purpose Only)'
        plt.text(0.5,0.1, volume_result,fontsize=15, verticalalignment='center', horizontalalignment='center') #字的放置位置跟字體大小，前兩個x,y是放置在影像中的比率，設定垂直水平對齊哪
        for k in range(len(combine_num)):
            y_color = y_pred_cluster[:,:,combine_num[k]].copy()
            D_show = np.zeros((ROWS, COLS))        
            D_show[slice_y:slice_y+y_i,slice_x:slice_x+x_i] = DWI1000_array[:,:,combine_num[k]].copy() 
            min_value = np.min(D_show)
            D_show[D_show==Dback_v] = min_value
            
            predict_show = ((D_show - np.min(D_show))*180).copy()
            predict_show = np.expand_dims(predict_show, axis=-1)
            predict_show_one = predict_show.copy()
            predict_show = np.concatenate([predict_show,predict_show_one],2)
            predict_show = np.concatenate([predict_show,predict_show_one],2)
            predict_show[predict_show>255] = 255
            predict_show = predict_show.astype('uint8') 

            #ax2.subplot(int(len(combine_num)//4),4,k+1)
            if k <5:
                ax2 = fig.add_axes([0 + 0.2*k, 0.9, 0.2, 0.2])  # inside axes    axes([左邊位置, 下邊位置, 寬, 高])- 設定圖表本身的位置(左下角座標為(0,0), 右上角座標為(1,1))
                ax2.imshow(predict_show)
                ax2.axis('off')
            elif 5 <= k <10:
                ax2 = fig.add_axes([0 + 0.2*(k-5), 0.7, 0.2, 0.2])  # inside axes
                ax2.imshow(predict_show)
                ax2.axis('off')
            elif 10 <= k < 15:
                ax2 = fig.add_axes([0 + 0.2*(k-10), 0.5, 0.2, 0.2])  # inside axes
                ax2.imshow(predict_show)
                ax2.axis('off')                    
            else:
                ax2 = fig.add_axes([0 + 0.2*(k-15), 0.3, 0.2, 0.2])  # inside axes
                ax2.imshow(predict_show)
                ax2.axis('off')
        
        #table = plt.table(cellText = volume_result,cellLoc='center',loc='bottom' , edges='open')
        plt.savefig(os.path.join(path_predict, ID + '_noPred.png'), bbox_inches='tight', pad_inches=0, dpi=300) #dpi太大，orthanc web viewer處理不來
        #plt.show()   #最後把combine img顯示到jupyter notebook畫面上，以下展示影像，可進行隱藏
        plt.close('all')


        # 正常显示中文字体
        #plt.rcParams['font.sans-serif']=['Microsoft YaHei']
        plt.style.use('dark_background') #使用黑色當背景色，繪圖風格
        fig = plt.figure() #開figure
        plt.axis('off')
        #left, bottom, width, height = 0.05, 0.05, 0.95, 0.95
        #ax1 = fig.add_axes([left, bottom, width, height])  # main axes
        #因為有沒有stroke病灶的狀況，當然就沒有平均，所以用一個判別式去區分
        if np.sum(y_pred_cluster) > 0:
            volume_result = 'Infarction Volume: ' + str(InfarctVolume) + ' ml (mean ADC = ' + str(MeanADC) + ')\n\n(For Research Purpose Only)'  #只顯示兩位數
        else:
            volume_result = 'Infarction Volume: 0 ml\n\n(For Research Purpose Only)'
        plt.text(0.5,0.1, volume_result,fontsize=15, verticalalignment='center', horizontalalignment='center') #字的放置位置跟字體大小，前兩個x,y是放置在影像中的比率，設定垂直水平對齊哪
        for k in range(len(combine_num)):
            y_color = y_pred_cluster[:,:,combine_num[k]].copy()
            D_show = np.zeros((ROWS, COLS))        
            D_show[slice_y:slice_y+y_i,slice_x:slice_x+x_i] = DWI1000_array[:,:,combine_num[k]].copy() 
            min_value = np.min(D_show)
            D_show[D_show==Dback_v] = min_value
            
            predict_show = ((D_show - np.min(D_show))*180).copy()
            predict_show = np.expand_dims(predict_show, axis=-1)
            predict_show_one = predict_show.copy()
            predict_show = np.concatenate([predict_show,predict_show_one],2)
            predict_show = np.concatenate([predict_show,predict_show_one],2)
            predict_show[predict_show>255] = 255
            predict_show = predict_show.astype('uint8') 
            y_c, x_c = np.where(y_color> 0) #y_color是0或1
            if len(y_c) > 0:    
                predict_show[y_c,x_c,0] = 255
                predict_show[y_c,x_c,1] = 0
                predict_show[y_c,x_c,2] = 255

            #ax2.subplot(int(len(combine_num)//4),4,k+1)
            if k <5:
                ax2 = fig.add_axes([0 + 0.2*k, 0.9, 0.2, 0.2])  # inside axes    axes([左邊位置, 下邊位置, 寬, 高])- 設定圖表本身的位置(左下角座標為(0,0), 右上角座標為(1,1))
                ax2.imshow(predict_show)
                ax2.axis('off')
            elif 5 <= k <10:
                ax2 = fig.add_axes([0 + 0.2*(k-5), 0.7, 0.2, 0.2])  # inside axes
                ax2.imshow(predict_show)
                ax2.axis('off')
            elif 10 <= k < 15:
                ax2 = fig.add_axes([0 + 0.2*(k-10), 0.5, 0.2, 0.2])  # inside axes
                ax2.imshow(predict_show)
                ax2.axis('off')                    
            else:
                ax2 = fig.add_axes([0 + 0.2*(k-15), 0.3, 0.2, 0.2])  # inside axes
                ax2.imshow(predict_show)
                ax2.axis('off')
        
        #table = plt.table(cellText = volume_result,cellLoc='center',loc='bottom' , edges='open')
        plt.savefig(os.path.join(path_predict, ID + '.png'), bbox_inches='tight', pad_inches=0, dpi=300) #dpi太大，orthanc web viewer處理不來
        #plt.show()   #最後把combine img顯示到jupyter notebook畫面上，以下展示影像，可進行隱藏
        plt.close('all')


        if len(Infarct_labels_show) > 0:
            # 正常显示中文字体
            #plt.rcParams['font.sans-serif']=['Microsoft YaHei']
            plt.style.use('dark_background') #使用黑色當背景色，繪圖風格
            fig = plt.figure() #開figure
            plt.axis('off')
            #left, bottom, width, height = 0.05, 0.05, 0.95, 0.95
            #ax1 = fig.add_axes([left, bottom, width, height])  # main axes
            #因為有沒有stroke病灶的狀況，當然就沒有平均，所以用一個判別式去區分
            if np.sum(y_pred_cluster) > 0:
                volume_result = 'Infarction Volume: ' + str(InfarctVolume) + ' ml (mean ADC = ' + str(MeanADC) + ')\n\n(For Research Purpose Only)'  #只顯示兩位數
            else:
                volume_result = 'Infarction Volume: 0 ml\n\n(For Research Purpose Only)'
            plt.text(0.5,0.1, volume_result,fontsize=15, verticalalignment='center', horizontalalignment='center') #字的放置位置跟字體大小，前兩個x,y是放置在影像中的比率，設定垂直水平對齊哪

            for k in range(len(combine_s_num)):
                y_color = new_SynthSeg_array[:,:,combine_s_num[k]].copy()
                D_show = np.zeros((ROWS, COLS))        
                D_show[slice_y:slice_y+y_i,slice_x:slice_x+x_i] = DWI1000_array[:,:,combine_s_num[k]].copy() 
                min_value = np.min(D_show)
                
                Dback_v = D_show[2,2]
                #print('Dback_v:', Dback_v)
                D_show[D_show==Dback_v] = min_value
                    
                predict_show = ((D_show - np.min(D_show))*180).copy()
                predict_show = np.expand_dims(predict_show, axis=-1)
                predict_show_one = predict_show.copy()
                predict_show = np.concatenate([predict_show,predict_show_one],2)
                predict_show = np.concatenate([predict_show,predict_show_one],2)
                predict_show[predict_show>255] = 255
                predict_show = predict_show.astype('uint8') 
                
                #按照大小上彩虹色，而不做一個區域壓一種顏色(可能會顏色相近)
                for idx_labels, label_one in enumerate(Infarct_labels_show):
                    for jdx_labels, label_num in enumerate(Infarct_labels[label_one]):
                        y_c, x_c = np.where(y_color == label_num) #變true or false矩陣
                        if len(y_c) > 0: 
                            rgb = mcolors.to_rgb(colorbar[label_one])
                            predict_show[y_c,x_c,0] = rgb[0]*255
                            predict_show[y_c,x_c,1] = rgb[1]*255
                            predict_show[y_c,x_c,2] = rgb[2]*255

                #ax2.subplot(int(len(combine_num)//4),4,k+1)
                if k <4:
                    ax2 = fig.add_axes([0 + 0.16*k, 0.9, 0.21, 0.21])  # inside axes    axes([左邊位置, 下邊位置, 寬, 高])- 設定圖表本身的位置(左下角座標為(0,0), 右上角座標為(1,1))
                    ax2.imshow(predict_show)
                    ax2.axis('off')
                elif 4 <= k <8:
                    ax2 = fig.add_axes([0 + 0.16*(k-4), 0.7, 0.21, 0.21])  # inside axes
                    ax2.imshow(predict_show)
                    ax2.axis('off')
                elif 8 <= k < 12:
                    ax2 = fig.add_axes([0 + 0.16*(k-8), 0.5, 0.21, 0.21])  # inside axes
                    ax2.imshow(predict_show)
                    ax2.axis('off')                    
                else:
                    ax2 = fig.add_axes([0 + 0.16*(k-12), 0.3, 0.21, 0.21])  # inside axes
                    ax2.imshow(predict_show)
                    ax2.axis('off')
                
            ax2 = fig.add_axes([0 + 0.165*4, 0.315+0.4*(1-len(Infarct_labels_show)/15), 0.015, 0.05*len(Infarct_labels_show)])  # inside axes
            for idx_labels, label in enumerate(Infarct_labels_show):
                ax2.hlines(1-0.03*idx_labels, 0, 1, color = colorbar[label], linewidth = 6)
                #text右上是(1,1) 
                ax2.text(1.3, 0.996-0.03*idx_labels, label + ': ' + str(Infarct_labels_show[label]) + ' ml', fontsize = 6, horizontalalignment ='left')
                ax2.axis('off')
            
            #table = plt.table(cellText = volume_result,cellLoc='center',loc='bottom' , edges='open')
            plt.savefig(os.path.join(path_predict, ID + '_SynthSEG.png'), bbox_inches='tight', pad_inches=0, dpi=300) #dpi太大，orthanc web viewer處理不來
            #plt.show()   #最後把combine img顯示到jupyter notebook畫面上，以下展示影像，可進行隱藏
            plt.close('all')

        
        #以json做輸出
        Report = text
        json_path_name = os.path.join(path_json, 'Pred_Infarct.json')
        json_path_name2 = os.path.join(path_output, 'Pred_Infarct.json')
        case_json(json_path_name, ID, InfarctVolume, MeanADC, Report)  
        case_json(json_path_name2, ID, InfarctVolume, MeanADC, Report)      
        #刪除資料夾
        # if os.path.isdir(path_process):  #如果資料夾存在
        #     shutil.rmtree(path_process) #清掉整個資料夾 

        logging.info('!!! ' + ID +  ' post_stroke finish.')


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
    parser.add_argument('--ID', type=str, default = '00052669_20191210_MR_20812100074', help='目前執行的case的patient_id or study id')
    parser.add_argument('--Inputs', type=str, nargs='+', default = ['/mnt/e/pipeline/chuan/example_input/00052669_20191210_MR_20812100074/ADC.nii.gz', '/mnt/e/pipeline/chuan/example_input/00052669_20191210_MR_20812100074/DWI0.nii.gz', '/mnt/e/pipeline/chuan/example_input/00052669_20191210_MR_20812100074/DWI1000.nii.gz', '/mnt/e/pipeline/chuan/example_input/00052669_20191210_MR_20812100074/synthseg_DWI0_original_DWI.nii.gz'], help='用於輸入的檔案')
    parser.add_argument('--Output_folder', type=str, default = '/mnt/e/pipeline/chuan/example_output/',help='用於輸出結果的資料夾')    
    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs  # 將列表合併為字符串，保留順序
    path_output = str(args.Output_folder)

    #讀出DWI, DWI0, ADC, SynthSEG的檔案
    ADC_file = Inputs[0]
    DWI0_file = Inputs[1]
    DWI1000_file = Inputs[2]
    SynthSEG_file = Inputs[3]

    #下面設定各個路徑
    path_code = '/mnt/e/pipeline/chuan/code/'
    path_process = '/mnt/e/pipeline/chuan/process/'  #前處理dicom路徑(test case)
    path_processModel = os.path.join(path_process, 'Deep_Infarct')  #前處理dicom路徑(test case)
    #path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)

    #這裡先沒有dicom
    path_json = '/mnt/e/pipeline/chuan/json/'  #存放json的路徑，回傳執行結果
    #json_path_name = os.path.join(path_json, 'Pred_Infarct.json')
    path_log = '/mnt/e/pipeline/chuan/log/'  #log資料夾
    cuatom_model = '/mnt/e/pipeline/chuan/code/model_weights/Unet-0194-0.17124-0.14731-0.86344.h5'  #model路徑+檔案
    gpu_n = 0  #使用哪一顆gpu

    #建置資料夾
    if not os.path.isdir(path_processModel):  #如果資料夾不存在就建立
        os.mkdir(path_processModel) #製作nii資料夾

    #直接當作function的輸入
    pipeline_infarct(ID, ADC_file, DWI0_file, DWI1000_file, SynthSEG_file, path_output, path_code, path_processModel, path_json, path_log, cuatom_model, gpu_n)

    # #最後再讀取json檔結果
    # with open(json_path_name) as f:
    #     data = json.load(f)

    # logging.info('Json!!! ' + str(data))