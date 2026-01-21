#! /home/tmu/anaconda3/envs/platform-stroke/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:59:21 2021

取名data util去記錄所有資料前處理的過程

@author: chuan
"""


#from logging.config import _RootLoggerConfiguration
import os
import time
import numpy as np
import pydicom
import glob
import shutil
#import gdcm
import nibabel as nib
import nibabel.processing
from nii_transforms import nii_img_replace
import matplotlib
import matplotlib.pyplot as plt
import sys
import logging
import pandas as pd

#要安裝pillow 去開啟 JPEG2000格式的DICOM 影像
#conda install Pillow


#去抓影像的dicom以便於去切資料夾放置不同種類的影像，做法是偵測到不同的dicom tag就新開一個該tag名的資料夾
#抓的dicom tag => (0008,1030) Study Description [Chest VCT(-C) (Health check)]

def class_studydescriptoion(path_in, path_out, log_file):
    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT)
    file_list = os.listdir(path_in)
    file_list = sorted(file_list)
    for i in range(len(file_list)):
        #因為同一個資料夾影像幾乎是相同序列，所以只檢查第一張就好
        dicom_slice = pydicom.dcmread(os.path.join(path_in, file_list[i])) #第一張影像
        dcm_patientid = dicom_slice[0x0010,0x0020].value #Patient ID
        dcm_studydate = dicom_slice[0x08,0x0020].value #Study Date
        dcm_studyid = dicom_slice[0x0020,0x000d].value #StudyInstanceUID <- 以此為資料夾命名
        dcm_series = dicom_slice[0x08,0x103E].value #Series Description <- 以此為資料夾命名
        dcm_series = dcm_series.replace(":", " ") #把冒號替換成空格，因為冒號不能當作資料夾名稱
        dcm_series = dcm_series.replace("^", " ") #把冒號替換成空格，因為冒號不能當作資料夾名稱     
        dcm_series = dcm_series.replace("/", " ") #把冒號替換成空格，因為冒號不能當作資料夾名稱     
        dcm_series = dcm_series.encode(encoding='UTF-8') #因為有上標，所以編碼格式要改
        dcm_series = dcm_series.decode(encoding='cp950')
        dcm_sopuid = dicom_slice[0x0008, 0x0018].value #SOPInstanceUID <- 以此為檔案命名
        if not os.path.isdir(os.path.join(path_out, dcm_studyid)):
            os.mkdir(os.path.join(path_out, dcm_studyid))
        if not os.path.isdir(os.path.join(path_out, dcm_studyid, 'img_classification')):
            os.mkdir(os.path.join(path_out, dcm_studyid, 'img_classification'))    
        if not os.path.isdir(os.path.join(path_out, dcm_studyid, 'img_classification', dcm_series)):
            os.mkdir(os.path.join(path_out, dcm_studyid, 'img_classification', dcm_series))
        #複製檔案過去資料夾，不管之前有沒有存在該tag反正一律複製到new_tag資料夾就好，有就有，沒有就沒有
        try:
            shutil.copy(os.path.join(path_in, file_list[i]), os.path.join(path_out, dcm_studyid, 'img_classification', dcm_series, dcm_sopuid + '.dcm'))
        except:
            logging.warning('Retry!!! Image:' + file_list[i] + ' can not be copied. ', exc_info=True)
    #最後直接由資料夾去紀錄有幾個人
    case_name = os.listdir(path_out)
    patientid = str(dcm_patientid).rjust(8,'0') + '_' + str(dcm_studydate).rjust(8,'0')

    #print('All cases are ok.')
    return case_name, patientid


#去抓影像資料夾的名稱，找出ADC跟DWI1000跟T2FLAIR，把影像複製過去跟分類，確定只有只有一筆才執行
def move_ADC_DWI_T2FLAIR(path_class, path_dcm, case_name, patientid, log_file):
    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT)
    
    path_dcmADC = os.path.join(path_dcm, 'ADC')
    path_dcmDWI0 = os.path.join(path_dcm, 'DWI0')
    path_dcmDWI1000 = os.path.join(path_dcm, 'DWI1000')
    path_dcmT2FLAIR = os.path.join(path_dcm, 'T2FLAIR')

    series_list = os.listdir(path_class) #檢查單名稱
    series_nopred = [s for s in series_list if s.lower().find('pred') < 0] #先把有Pred的去除
    series_nopred = [s for s in series_nopred if s.lower().find('report') < 0] #先把有report的去除
    series_nopred = sorted(series_nopred)
    #以下針對series list去分類出adc, dwi0, dwi100, t2flair的資料夾
    #偵測下方ADC字串
    matching = [s for s in series_nopred if s.lower().find('adc') >=0] #找出名稱有ADC的資料夾，ADC因為沒有其他確認方法，當第一個是因為第一個字串最短
    matching1 = [s for s in series_nopred if s.lower().find('apparent diffusion coefficient') >=0] #找出名稱有ADC的資料夾，ADC因為沒有其他確認方法，當第一個是因為第一個字串最短
    matching = sorted(matching)
    if len(matching) > 1:
        #複數以上只能無腦以第一個複製
        path_ADC = os.path.join(path_class, matching[0]) #沒辦法確定有幾個符合ADC，只能無腦當作第一個
        shutil.copytree(path_ADC, path_dcmADC)
        logging.warning('!!!　' + patientid + ' ' + str(case_name) + ' ADC too much.')
    elif len(matching) == 1:
        path_ADC = os.path.join(path_class, matching[0]) #確定有1個符合ADC
        shutil.copytree(path_ADC, path_dcmADC)
    elif len(matching) == 0 and len(matching1) > 1:
        #複數以上只能無腦以第一個複製
        path_ADC = os.path.join(path_class, matching1[0]) #沒辦法確定有幾個符合ADC，只能無腦當作第一個
        shutil.copytree(path_ADC, path_dcmADC)
        logging.warning('!!!　' + patientid + ' ' + str(case_name) + ' apparent diffusion coefficient too much.')   
    elif len(matching) == 0 and len(matching1) == 1: 
        path_ADC = os.path.join(path_class, matching1[0]) #沒辦法確定有幾個符合ADC，只能無腦當作第一個
        shutil.copytree(path_ADC, path_dcmADC)

    #以下做dwi，比較複雜，要分DWI0跟DWI1000，以下執行DWI，分0跟1000   folder_list還是要先得知，DWI不會有純這個名稱
    matching = [s for s in series_nopred if s.lower().find('dwi') >=0] #找出名稱有DWI的資料夾，DWI一定有Ax? 不確定
    matching1 = [s for s in series_nopred if s.lower().find('autodiff') >=0] #找出名稱有DWI的資料夾，DWI一定有Ax? 不確定
    matching = sorted(matching)
    matching1 = sorted(matching1)
    path_DWI = '' #先以空str
    if len(matching) > 1:
        #複數以上只能無腦以第一個複製
        path_DWI = os.path.join(path_class, matching[0]) #沒辦法確定有幾個符合DWI，只能無腦當作第一個
        logging.warning('!!! ' + patientid + ' ' + str(case_name) + ' DWI too much.')
    elif len(matching) == 1:
        path_DWI = os.path.join(path_class, matching[0]) #剛好1
    elif len(matching) == 0 and len(matching1) > 1:
        path_DWI = os.path.join(path_class, matching1[0]) #剛好名稱為autodiff
        logging.warning('!!! ' + patientid + ' ' + str(case_name) + ' autodiff too much.')
    elif len(matching) == 0 and len(matching1) == 1:
        path_DWI = os.path.join(path_class, matching1[0]) #剛好名稱為autodiff
    
    #如果有path_DWI路徑，做下操作，複製dwi到特定資料夾
    if os.path.isdir(path_DWI):
        #把DWI的影像資料分類
        file_list = os.listdir(path_DWI) #DWI大數量
        file_list = sorted(file_list)
        #接下來要先逐筆影像讀取instance number，加快速度，先用list去紀錄，然後藉由[0x20,0x0013]去區分是b=0還是b=1000
        for j in range(len(file_list)):
            dcm = pydicom.dcmread(os.path.join(path_DWI, file_list[j]))
            dwi_tag = dcm[0x0043, 0x1039].value #得知是b=0或是b=1000
            dwi_tag = dwi_tag[0] #取第一個值
            #複製dwi影像到相應資料夾
            if dwi_tag==0:
                if not os.path.isdir(path_dcmDWI0):  #如果資料夾不存在就建立
                    os.mkdir(path_dcmDWI0)
                shutil.copy(os.path.join(path_DWI, file_list[j]), os.path.join(path_dcmDWI0, file_list[j]))
            elif dwi_tag==1000:
                if not os.path.isdir(path_dcmDWI1000):  #如果資料夾不存在就建立
                    os.mkdir(path_dcmDWI1000)
                shutil.copy(os.path.join(path_DWI, file_list[j]), os.path.join(path_dcmDWI1000, file_list[j]))
            else:
                logging.error('!!! ' + patientid + ' ' + str(case_name) +  ' DWI image has tag neither b=0 nor b=1000.')
                
    #以下做T2FLAIR，還不確定要考慮什麼
    #1. 因為T1也有FLAIR，所以直接把T1砍掉才是最省行數
    #2. 找名稱裡有flair，雷的點是放射師有單打flair的series，所以只能硬當t2 flair
    matching = [s for s in series_nopred if not s.lower().find('t1') >=0] #找出名稱沒有T1的資料夾
    if len(matching) > 0:
        matching1 = [s for s in matching if s.lower().find('flair') >=0] #找出名稱有FLAIR的資料夾
        if len(matching1) > 1:
            matching2 = [s for s in matching1 if s.lower().find('t2') >=0] #找出名稱有T2的資料夾
            matching2 = sorted(matching2)
            if len(matching2) > 1:
                path_T2FLAIR = os.path.join(path_class, matching2[0]) #沒辦法確定有幾個符合T2FLAIR，只能無腦當作第一個
                shutil.copytree(path_T2FLAIR, path_dcmT2FLAIR)
                logging.warning('!!! ' + patientid + ' ' + str(case_name) + ' T2 FLAIR too much.')
            elif len(matching2) == 1:
                path_T2FLAIR = os.path.join(path_class, matching2[0]) #確定有1個符合T2FLAIR
                shutil.copytree(path_T2FLAIR, path_dcmT2FLAIR)
        elif len(matching1) == 1:
            path_T2FLAIR = os.path.join(path_class, matching1[0]) #只有這筆有FLAIR字串，所以當作T2 FLAIR(還是有機會T1...)
            shutil.copytree(path_T2FLAIR, path_dcmT2FLAIR)
    return


#去抓影像資料夾的名稱，找出T2FLAIR，把影像複製過去跟分類
def move_T2FLAIR(path_class, path_dcm, case_name, patientid, log_file):
    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT)

    path_dcmT2FLAIR = os.path.join(path_dcm, 'T2FLAIR')

    series_list = os.listdir(path_class) #檢查單名稱
    series_nopred = [s for s in series_list if s.lower().find('pred') < 0] #先把有Pred的去除
    series_nopred = [s for s in series_nopred if s.lower().find('report') < 0] #先把有report的去除
    series_nopred = sorted(series_nopred)
    #以下針對series list去分類出t2flair的資料夾
    #以下做T2FLAIR，還不確定要考慮什麼
    #1. 因為T1也有FLAIR，所以直接把T1砍掉才是最省行數
    #2. 找名稱裡有flair，雷的點是放射師有單打flair的series，所以只能硬當t2 flair
    matching = [s for s in series_nopred if not s.lower().find('t1') >=0] #找出名稱沒有T1的資料夾
    if len(matching) > 0:
        matching1 = [s for s in matching if s.lower().find('flair') >=0] #找出名稱有FLAIR的資料夾
        if len(matching1) > 1:
            matching2 = [s for s in matching1 if s.lower().find('t2') >=0] #找出名稱有T2的資料夾
            matching2 = sorted(matching2)
            if len(matching2) > 1:
                path_T2FLAIR = os.path.join(path_class, matching2[0]) #沒辦法確定有幾個符合T2FLAIR，只能無腦當作第一個
                shutil.copytree(path_T2FLAIR, path_dcmT2FLAIR)
                logging.warning('!!! ' + patientid + ' ' + str(case_name) + ' T2 FLAIR too much.')
            elif len(matching2) == 1:
                path_T2FLAIR = os.path.join(path_class, matching2[0]) #確定有1個符合T2FLAIR
                shutil.copytree(path_T2FLAIR, path_dcmT2FLAIR)
        elif len(matching1) == 1:
            path_T2FLAIR = os.path.join(path_class, matching1[0]) #只有這筆有FLAIR字串，所以當作T2 FLAIR(還是有機會T1...)
            shutil.copytree(path_T2FLAIR, path_dcmT2FLAIR)
    return


#去抓影像資料夾的名稱，找出C-，把影像複製過去跟分類，Study Description判斷是不是brain ct只能靠php
def move_CTich(path_class, path_dcm, case_name, patientid, log_file):
    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT)

    path_dcmCT = os.path.join(path_dcm, 'C-') #目的資料夾

    series_list = os.listdir(path_class) #檢查單名稱
    series_nopred = [s for s in series_list if s.lower().find('pred') < 0] #先把有Pred的去除
    series_nopred = sorted([s for s in series_nopred if s.lower().find('report') < 0]) #先把有report的去除
    #以下針對series list去分類出c-的資料夾
    #以下做c-，還不確定要考慮什麼
    #1. 如果沒有c-，就找axi
    #2. 如果沒有axi，就找processed images
    matching = sorted([s for s in series_nopred if s.lower().find('c-') >=0]) #找出名稱沒有T1的資料夾
    if len(matching) > 1:
        #只能當作名稱最短是c-了
        path_CT = os.path.join(path_class, matching[0]) #沒辦法確定有幾個符合T2FLAIR，只能無腦當作第一個
        shutil.copytree(path_CT, path_dcmCT)
        logging.warning('!!! ' + patientid + ' ' + str(case_name) + ' C- too much.')
    elif len(matching) == 1:
        #當好符合，讚
        path_CT = os.path.join(path_class, matching[0]) #沒辦法確定有幾個符合T2FLAIR，只能無腦當作第一個
        shutil.copytree(path_CT, path_dcmCT)
    else:
        #先找有沒有axi
        matching1 = sorted([s for s in matching if s.lower().find('axi') >=0]) #找出名稱有FLAIR的資料夾
        if len(matching1) > 1:
            path_CT = os.path.join(path_class, matching1[0]) #沒辦法確定有幾個符合T2FLAIR，只能無腦當作第一個
            shutil.copytree(path_CT, path_dcmCT)
            logging.warning('!!! ' + patientid + ' ' + str(case_name) + ' C- too much.')
        elif len(matching1) == 1:
            #當好符合，讚
            path_CT = os.path.join(path_class, matching1[0]) #沒辦法確定有幾個符合T2FLAIR，只能無腦當作第一個
            shutil.copytree(path_CT, path_dcmCT)
    return


#接下來確認目標資料夾(Infarct)下的影像是否符合數量以決定是否重做或是往下做
def check_tag_num_stroke(path_dcm, case_name, patientid, log_file):
    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT) 

    path_dcmADC = os.path.join(path_dcm, 'ADC')
    path_dcmDWI0 = os.path.join(path_dcm, 'DWI0')
    path_dcmDWI1000 = os.path.join(path_dcm, 'DWI1000')
    path_dcmT2FLAIR = os.path.join(path_dcm, 'T2FLAIR')

    #要for 迴圈的所有case都ok才算影像傳送ok 
    series_list = os.listdir(path_dcm) #檢查單名稱
    ans = [] # 0 => ok , 1 =>　retry , 2 => error
    DWI0_n = 0
    DWI1000_n = 0
    T2FLAIR_n = 0
    ADC_n = 0
    DWI0_list = [s for s in series_list if s.find('DWI0') >=0]
    if len(DWI0_list) == 1:
        file_list = os.listdir(path_dcmDWI0)
        dcm = pydicom.dcmread(os.path.join(path_dcmDWI0, file_list[0])) #讀取第一張確認影像總張數
        ImagesAcquisition = dcm[0x0020, 0x1002].value #得知Images in Acquisition
        if len(file_list) > ImagesAcquisition/2 :
            ans.append(2)
            logging.error('!!! ' + patientid + ' ' + str(case_name) + ' DWI0 slice too much.')
        elif len(file_list) == ImagesAcquisition/2 :
            DWI0_n = len(file_list)
        else:
            ans.append(1)
            logging.warning('Retry!!! ' + patientid + ' ' + str(case_name) + ' DWI0 slice insufficient.')
    
    DWI1000_list = [s for s in series_list if s.find('DWI1000') >=0]
    if len(DWI1000_list) == 1:
        file_list = os.listdir(path_dcmDWI1000)
        dcm = pydicom.dcmread(os.path.join(path_dcmDWI1000, file_list[0])) #讀取第一張確認影像總張數
        ImagesAcquisition = dcm[0x0020, 0x1002].value #得知Images in Acquisition
        if len(file_list) > ImagesAcquisition/2 :
            ans.append(2)
            logging.error('!!! ' + patientid + ' ' + str(case_name) + ' DWI1000 slice too much.')
        elif len(file_list) == ImagesAcquisition/2 :
            DWI1000_n = len(file_list)
        else:
            ans.append(1)
            logging.warning('Retry!!! ' + patientid + ' ' + str(case_name) + ' DWI1000 slice insufficient.')

    ADC_list = [s for s in series_list if s.find('ADC') >=0]
    if len(ADC_list) == 1 and len(DWI0_list) == 1 and len(DWI1000_list) == 1:
        file_list = os.listdir(path_dcmADC)
        #因為ADC沒有dcm[0x0020, 0x1002].value 得知Images in Acquisition，所以借DWI來用
        if len(file_list) > ImagesAcquisition/2 :
            ans.append(2)
            logging.error('!!! ' + patientid + ' ' + str(case_name) + ' ADC slice too much.')
        elif len(file_list) == ImagesAcquisition/2 :
            ADC_n = len(file_list)
        else:
            ans.append(1)
            logging.warning('Retry!!! ' + patientid + ' ' + str(case_name) + ' ADC slice insufficient.')
    
    T2FLAIR_list = [s for s in series_list if s.find('T2FLAIR') >=0]
    if len(T2FLAIR_list) == 1:
        file_list = os.listdir(path_dcmT2FLAIR)
        dcm = pydicom.dcmread(os.path.join(path_dcmT2FLAIR, file_list[0])) #讀取第一張確認影像總張數
        ImagesAcquisition = dcm[0x0020, 0x1002].value #得知Images in Acquisition
        if len(file_list) > ImagesAcquisition :
            ans.append(2)
            logging.error('!!! ' + patientid + ' ' + str(case_name) + ' T2FLAIR slice too much.')
        elif len(file_list) == ImagesAcquisition :
            T2FLAIR_n = len(file_list)
        else:
            ans.append(1)
            logging.warning('Retry!!! ' + patientid + ' ' + str(case_name) + ' T2FLAIR slice insufficient.')

    if DWI0_n ==  DWI1000_n and  DWI0_n == ADC_n and  DWI0_n == T2FLAIR_n and DWI0_n != 0 :
        ans.append(0)
    
    if len(ans) > 0:
        ans = np.array(ans)
        ans = np.max(ans)
    else:
        ans = 2
        logging.error('!!! ' + patientid + ' ' + str(case_name) + ' No any match tag.')

    return ans


#接下來確認目標資料夾(Infarct)下的影像是否符合數量以決定是否重做或是往下做，目前規劃先確認ADC跟DWI
def check_tag_num_stroke_currently(path_dcm, case_name, patientid, log_file):
    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT) 

    path_dcmADC = os.path.join(path_dcm, 'ADC')
    path_dcmDWI0 = os.path.join(path_dcm, 'DWI0')
    path_dcmDWI1000 = os.path.join(path_dcm, 'DWI1000')

    #要for 迴圈的所有case都ok才算影像傳送ok 
    series_list = os.listdir(path_dcm) #檢查單名稱
    ans = [] # 0 => ok , 1 =>　retry , 2 => error
    DWI0_n = 0
    DWI1000_n = 0
    ADC_n = 0
    DWI0_list = [s for s in series_list if s.find('DWI0') >=0]
    if len(DWI0_list) == 1:
        file_list = os.listdir(path_dcmDWI0)
        dcm = pydicom.dcmread(os.path.join(path_dcmDWI0, file_list[0])) #讀取第一張確認影像總張數
        ImagesAcquisition = dcm[0x0020, 0x1002].value #得知Images in Acquisition
        if len(file_list) > ImagesAcquisition/2 :
            ans.append(2)
            logging.error('!!! ' + patientid + ' ' + str(case_name) + ' DWI0 slice too much.')
        elif len(file_list) == ImagesAcquisition/2 :
            DWI0_n = len(file_list)
        else:
            ans.append(1)
            logging.warning('Retry!!! ' + patientid + ' ' + str(case_name) + ' DWI0 slice insufficient.')
    
    DWI1000_list = [s for s in series_list if s.find('DWI1000') >=0]
    if len(DWI1000_list) == 1:
        file_list = os.listdir(path_dcmDWI1000)
        dcm = pydicom.dcmread(os.path.join(path_dcmDWI1000, file_list[0])) #讀取第一張確認影像總張數
        ImagesAcquisition = dcm[0x0020, 0x1002].value #得知Images in Acquisition
        if len(file_list) > ImagesAcquisition/2 :
            ans.append(2)
            logging.error('!!! ' + patientid + ' ' + str(case_name) + ' DWI1000 slice too much.')
        elif len(file_list) == ImagesAcquisition/2 :
            DWI1000_n = len(file_list)
        else:
            ans.append(1)
            logging.warning('Retry!!! ' + patientid + ' ' + str(case_name) + ' DWI1000 slice insufficient.')

    ADC_list = [s for s in series_list if s.find('ADC') >=0]
    if len(ADC_list) == 1 and len(DWI0_list) == 1 and len(DWI1000_list) == 1:
        file_list = os.listdir(path_dcmADC)
        #因為ADC沒有dcm[0x0020, 0x1002].value 得知Images in Acquisition，所以借DWI來用
        if len(file_list) > ImagesAcquisition/2 :
            ans.append(2)
            logging.error('!!! ' + patientid + ' ' + str(case_name) + ' ADC slice too much.')
        elif len(file_list) == ImagesAcquisition/2 :
            ADC_n = len(file_list)
        else:
            ans.append(1)
            logging.warning('Retry!!! ' + patientid + ' ' + str(case_name) + ' ADC slice insufficient.')

    if DWI0_n ==  DWI1000_n and  DWI0_n == ADC_n and DWI0_n != 0 :
        ans.append(0)
    
    if len(ans) > 0:
        ans = np.array(ans)
        ans = np.max(ans)
    else:
        ans = 2
        logging.error('!!! ' + patientid + ' ' + str(case_name) + ' No any match tag.')

    return ans


#接下來確認目標資料夾(WMH)下的影像是否符合數量以決定是否重做或是往下做
def check_tag_num_wmh(path_dcm, case_name, patientid, log_file):
    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT) 
 
    series_list = os.listdir(path_dcm) #檢查單名稱
    ans = [] # 0 => ok , 1 =>　retry , 2 => error

    path_dcmT2FLAIR = os.path.join(path_dcm, 'T2FLAIR')

    T2FLAIR_list = [s for s in series_list if s.find('T2FLAIR') >=0]
    if len(T2FLAIR_list) == 1:
        file_list = os.listdir(path_dcmT2FLAIR)
        dcm = pydicom.dcmread(os.path.join(path_dcmT2FLAIR, file_list[0])) #讀取第一張確認影像總張數
        ImagesAcquisition = dcm[0x0020, 0x1002].value #得知Images in Acquisition
        if len(file_list) > ImagesAcquisition :
            ans.append(2)
            logging.error('!!! ' + patientid + ' ' + str(case_name) +  ' T2FLAIR slice too much.')
        elif len(file_list) == ImagesAcquisition :
            ans.append(0)
        else:
            ans.append(1)
            logging.warning('Retry!!! ' + patientid + ' ' + str(case_name) +  ' T2FLAIR slice insufficient.')
    
    if len(ans) > 0:
        ans = np.array(ans)
        ans = np.max(ans)
    else:
        ans = 2
        logging.error('!!! ' + patientid + ' ' + str(case_name) +  ' No any match tag.')

    return ans


#接下來確認目標資料夾(ICH)下的影像是否符合數量以決定是否重做或是往下做，因為ICH沒有[0x0020, 0x1002] Images in Acquisition，只能預估張數，我猜7mm*18 or 5mm*25以上好惹
def check_tag_num_ct(path_dcm, case_name, patientid, log_file):
    FORMAT = '%(asctime)s %(levelname)s %(message)s'  #日期時間, 格式為 YYYY-MM-DD HH:mm:SS,ms，日誌的等級名稱，訊息
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', format=FORMAT) 
 
    series_list = os.listdir(path_dcm) #檢查單名稱
    ans = [] # 0 => ok , 1 =>　retry , 2 => error

    path_dcmCT = os.path.join(path_dcm, 'C-')

    CT_list = [s for s in series_list if s.find('C-') >=0]
    if len(CT_list) == 1:
        file_list = os.listdir(path_dcmCT)
        #dcm = pydicom.dcmread(os.path.join(path_dcmCT, file_list[0])) #讀取第一張確認影像總張數
        #ImagesAcquisition = dcm[0x0020, 0x1002].value #得知Images in Acquisition，CT竟然沒有 
        DCM_list = [[y, pydicom.dcmread(os.path.join(path_dcmCT, y))[0x20,0x0032].value[-1]] for y in file_list] #(0020,0032) Image Position (Patient) [-119.600,-124.458,89.460]
        Position = np.array([y[-1] for y in DCM_list])
        sortP = np.sort(Position)
        gap = [sortP[y+1] - sortP[y] for y in range(len(sortP)-1)]
        estimated = 126/min(gap)
        if len(file_list) >= estimated :
            ans.append(0)
        else:
            ans.append(1)
            logging.warning('Retry!!! ' + patientid + ' ' + str(case_name) +  ' C- slice insufficient.')
    
    if len(ans) > 0:
        ans = np.max(np.array(ans))
    else:
        ans = 2
        logging.error('!!! ' + patientid + ' ' + str(case_name) +  ' No any match tag.')

    return ans


#把.dcm的檔案轉換成nii，有一些步驟:1. .dcm的檔案要解壓縮，因為可能是用jepj2000 2. dcm轉nii，要使用特定的安裝包(查)
#該電腦需要先apt-get install libgdcm-tools，還要安裝 conda install -c conda-forge dcm2niix
def dcm2nii(path, case_name, tag):
    #case_name = ['1.2.840.113619.2.25.4.114366292.1618293007.177']
    #tag = ['ADC', 'DWI0', 'DWI1000']
    path_nii = os.path.join(path, 'nii')
    if not os.path.isdir(path_nii):  #如果資料夾不存在就建立
        os.mkdir(path_nii) #製作nii資料夾
    for j in range(len(tag)):
        bash_line = 'dcm2niix -v 0 -z y -o ' +  path_nii + ' ' + os.path.join(path, tag[j])  #把tag檔案複製，-v : verbose (n/y or 0/1/2, default 0) [no, yes, logorrheic]
        #print('bash_line:', bash_line)
        os.system(bash_line) 
        #改檔案名稱，因為dcm2nii輸出的名稱怪怪
        nii_list = [y for y in os.listdir(path_nii) if '.nii.gz' in y] #找出.nii的list
        new_name = case_name + '_' + tag[j] + '.nii.gz'
        os.rename(os.path.join(path_nii, [y for y in nii_list if tag[j] in y][0]), os.path.join(path_nii, new_name)) #os.rename("old.txt", "new.txt")
    return #print('All case is converted to nii')


#nii影像，因為ADC影像的info缺損，然後affine matrix遺失，所以立體資訊消失，目前使用DWI1000補回失去的資訊。
def nii_info_DWI1000_for_ADC(path, case_name):
    nii_header_fake = 'ADC'
    nii_header_true = 'DWI1000'
    img_true = nib.load(os.path.join(path, case_name + '_' + nii_header_true + '.nii.gz')) #DWI1000 nii
    img_fake = nib.load(os.path.join(path, case_name + '_' + nii_header_fake + '.nii.gz')) #ADC nii
    img_only = np.array(img_fake.dataobj)
    true_affine = img_true.affine
    header_true = img_true.header.copy()
    new_img = nib.nifti1.Nifti1Image(img_only, true_affine, header=header_true)
    nib.save(new_img, os.path.join(path, case_name + '_' + nii_header_fake + '.nii.gz'))
    return


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


#將SynthSEG的結果轉到original上
def resampleSynthSEG2original(path, case_name, series, SynthSEGmodel):
    path_nii = os.path.join(path, 'dicom','nii')
    #先讀取原始影像並抓出SpacingBetweenSlices跟PixelSpacing
    img_nii = nib.load(os.path.join(path_nii, case_name +  '_' + series + '.nii.gz'))
    img_array = np.array(img_nii.dataobj)
    img_array = data_translate(img_array)
    img_1mm_nii = nib.load(os.path.join(path_nii, case_name +  '_' + series + '_resample.nii.gz'))
    img_1mm_array = np.array(img_1mm_nii.dataobj)    
    SynthSEG_1mm_nii = nib.load(os.path.join(path_nii, case_name + '_' + series + '_' + SynthSEGmodel + '.nii.gz')) #230*230*140

    y_i, x_i, z_i = img_array.shape
    y_i1, x_i1, z_i1 = img_1mm_array.shape
    
    header_img = img_nii.header.copy() #抓出nii header 去算體積 
    pixdim_img = header_img['pixdim']  #可以借此從nii的header抓出voxel size
    header_img_1mm = img_1mm_nii.header.copy() #抓出nii header 去算體積 
    pixdim_img_1mm = header_img_1mm['pixdim']  #可以借此從nii的header抓出voxel size    
    
    #先把影像從230*230*140轉成256*256*140
    img_1mm_256_nii = nibabel.processing.conform(img_1mm_nii, ((y_i, x_i, z_i1)),(pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3]),order = 3) #影像用
    img_1mm_256 = np.array(img_1mm_256_nii.dataobj)
    img_1mm_256 = data_translate(img_1mm_256)
    img_1mm_256 = np.flip(img_1mm_256, 1)
    img_1mm_256_back = data_translate_back(img_1mm_256)
    img_1mm_256_nii2 = nii_img_replace(img_1mm_256_nii, img_1mm_256_back)
    nib.save(img_1mm_256_nii2, os.path.join(path_nii, case_name +  '_' + series + '_resample_256.nii.gz'))
    #再將SynthSEG從230*230*140轉成256*256*140
    SynthSEG_1mm_256_nii = nibabel.processing.conform(SynthSEG_1mm_nii, ((y_i, x_i, z_i1)),(pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3]),order = 0) #影像用
    SynthSEG_1mm_256 = np.array(SynthSEG_1mm_256_nii.dataobj)
    SynthSEG_1mm_256 = data_translate(SynthSEG_1mm_256)
    SynthSEG_1mm_256 = np.flip(SynthSEG_1mm_256, 1)
    SynthSEG_1mm_256_back = data_translate_back(SynthSEG_1mm_256)
    SynthSEG_1mm_256_nii2 = nii_img_replace(SynthSEG_1mm_256_nii, SynthSEG_1mm_256_back)
    nib.save(SynthSEG_1mm_256_nii2, os.path.join(path_nii, case_name +  '_' + series + '_' + SynthSEGmodel + '_256.nii.gz'))   

    #以下將1mm重新組回otiginal    
    new_array = np.zeros(img_array.shape)
    img_repeat = np.expand_dims(img_array, -1).repeat(z_i1, axis=-1)
    img_1mm_repeat = np.expand_dims(img_1mm_256, 2).repeat(z_i, axis=2)
    diff = np.sum(np.abs(img_1mm_repeat - img_repeat), axis = (0,1))
    argmin = np.argmin(diff, axis=1)
    new_array = SynthSEG_1mm_256[:,:,argmin]
    #最後重新組回nifti
    new_array_save = data_translate_back(new_array)
    new_SynthSeg_nii = nii_img_replace(img_nii, new_array_save)
    nib.save(new_SynthSeg_nii, os.path.join(path_nii, case_name + '_NEW_' + series + '_' + SynthSEGmodel + '.nii.gz'))   
    return new_array


# #nii使用BET的程式碼，BET是用FSL的功能。
# def use_BET_DWI0(path, path_BET, case_name):
#     if not os.path.isdir(path_BET):  #如果資料夾不存在就建立
#         os.mkdir(path_BET)
#     code = 'bet ' + os.path.join(path, case_name + '_DWI0.nii.gz') + ' ' + os.path.join(path_BET, case_name + '_DWI0.nii.gz') + ' -f 0.4'
#     #print(code)
#     os.system(code)
#     return


#nii使用BET的程式碼，BET是用FSL的功能。
def use_BET_DWI0(path, path_BET, case_name):
    if not os.path.isdir(path_BET):  #如果資料夾不存在就建立
        os.mkdir(path_BET)
    img_true_header = nib.load(os.path.join(path, case_name + '_SynthSEG.nii.gz'))
    img_array = np.array(img_true_header.dataobj) #讀出image的array矩陣
    img_array[img_array<0] = 0
    img_array[img_array>0] = 1
    true_affine = img_true_header.affine
    header_true = img_true_header.header.copy()
    new_img = nib.nifti1.Nifti1Image(img_array, true_affine, header=header_true)
    nib.save(new_img, os.path.join(path_BET, case_name + '_DWI0.nii.gz'))     
    return


#將BET出來的DWI0影像作成背景0，label:1 的mask
def make_BET_mask(path, path_mask, case_name):
    img_true_header = nib.load(os.path.join(path, case_name + '_DWI0.nii.gz'))
    img_array = np.array(img_true_header.dataobj) #讀出image的array矩陣
    img_array[img_array<0] = 0
    img_array[img_array>0] = 1
    true_affine = img_true_header.affine
    header_true = img_true_header.header.copy()
    new_img = nib.nifti1.Nifti1Image(img_array, true_affine, header=header_true)
    nib.save(new_img, os.path.join(path_mask, case_name + '.nii.gz'))     
    return


#將SynthSEG出來的DWI0影像作成背景0，label:1 的mask
def make_SynthSEG_Brain_mask(path, case_name):
    path_nii = os.path.join(path, 'dicom', 'nii')
    path_mask = os.path.join(path, 'dicom','BET_mask')
    if not os.path.isdir(path_mask):  #如果資料夾不存在就建立
        os.mkdir(path_mask)
    img_true_header = nib.load(os.path.join(path_nii, case_name + '_NEW_DWI0_DWI.nii.gz'))
    img_array = np.array(img_true_header.dataobj) #讀出image的array矩陣
    img_array[img_array<0] = 0
    img_array[img_array>0] = 1
    true_affine = img_true_header.affine
    header_true = img_true_header.header.copy()
    new_img = nib.nifti1.Nifti1Image(img_array, true_affine, header=header_true)
    nib.save(new_img, os.path.join(path_mask, case_name + '.nii.gz'))     
    return


#將BET過的DWI0影像當作標準，切割DWI1000跟ADC影像成BET影像，需要更正，因為正規化時可能又除到全影像而不是腦的部分，所以背景變負值。
#此沒做正規化
def make_BET_ADC_DWI1000(path, case_name):
    key = ['ADC', 'DWI1000']       # 源文件夹中的文件包含字符key
    ROWS=256 #y
    COLS=256 #x
    path_BET_img = os.path.join(path, 'BET_img')
    if not os.path.isdir(path_BET_img):  #如果資料夾不存在就建立
        os.mkdir(path_BET_img)
    for j in range(len(key)):
        img_nii = nib.load(os.path.join(path, 'nii', case_name + '_' + key[j] + '.nii.gz')) #讀取影像 -7 => DWI0.nii.gz
        img_array = np.array(img_nii.dataobj) #讀出image的array矩陣
        label_nii = nib.load(os.path.join(path, 'BET_mask', case_name + '.nii.gz')) #DWI0 used BET image
        label_array = np.array(label_nii.dataobj) #讀出label的array矩陣
        img_BET = img_array[:ROWS,:COLS,:]*label_array #相乘等於留下bet範圍影像
        img_BET[img_BET<0] = 0 #把背景的部分去除
        new_img = nii_img_replace(img_nii, img_BET)
        nib.save(new_img, os.path.join(path_BET_img, case_name + '_' +key[j] + '.nii.gz'))       
    return 


def custom_normalize_1(volume, mask=None, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5, use_positive_only=True):
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
    where 0 = np.min
    :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
    where 100 = np.max
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :return: rescaled volume
    """
    # select intensities
    new_volume = volume.copy()
    new_volume = new_volume.astype("float32")
    if (mask is not None)and(use_positive_only):
        intensities = new_volume[mask].ravel()
        intensities = intensities[intensities > 0]
    elif mask is not None:
        intensities = new_volume[mask].ravel()
    elif use_positive_only:
        intensities = new_volume[new_volume > 0].ravel()
    else:
        intensities = new_volume.ravel()

    # define min and max intensities in original image for normalisation
    robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
    robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)

    # trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:  # avoid dividing by zero
        return np.zeros_like(new_volume)


#最新，DWI1000跟ADC經過BET的影像做正規化然後轉nii儲存，統計分其他程式碼
def ADC_DWI1000_nor(path_dcm, case_name, path_test_img):
    path_nii = os.path.join(path_test_img, 'nii')
    path_npy = os.path.join(path_test_img, 'npy')
    if not os.path.isdir(path_test_img):  #如果資料夾不存在就建立
        os.mkdir(path_test_img)
    if not os.path.isdir(path_nii):  #如果資料夾不存在就建立
        os.mkdir(path_nii)
    if not os.path.isdir(path_npy):  #如果資料夾不存在就建立
        os.mkdir(path_npy)

    ADC_nii = nib.load(os.path.join(path_dcm, 'BET_img', case_name + '_ADC.nii.gz')) #讀取影像
    ADC_array = np.array(ADC_nii.dataobj).astype(float) #讀出image的array矩陣
    DWI1000_nii = nib.load(os.path.join(path_dcm, 'BET_img', case_name + '_DWI1000.nii.gz')) #讀取影像
    DWI1000_array = np.array(DWI1000_nii.dataobj).astype(float) #讀出image的array矩陣 
    BET_nii = nib.load(os.path.join(path_dcm, 'BET_mask', case_name + '.nii.gz')) #讀取影像
    BET_array = np.array(BET_nii.dataobj).astype(float) #讀出BET_mask的array矩陣 
    
    y_i, x_i , z_i = ADC_array.shape #因為影像尺寸有差異
    #正規化要只針對mask空間的部分，之前連背景也一起正規化做錯
    sum_BET = np.sum(BET_array) #得出BET_mask的大小，用來計算mean,std
    #接下來算adc,dwi1000的標準差，可能在Bet的區域會有0，所以必須1個點一個點對，麻煩
    y_B, x_B, z_B = np.where(BET_array>0)
    #ADC正規化是除600，所以不用去計算前處理的mean, std....等數值，DWI1000要    
    DWI1000_site_list = DWI1000_array[y_B,x_B,z_B]
    mean_DWI1000 = DWI1000_site_list.mean()
    std_DWI1000 = DWI1000_site_list.std()
    ADC_array[y_B,x_B,z_B] = ADC_array[y_B,x_B,z_B] / 600
    #print('mean_DWI1000:',mean_DWI1000)
    #DWI1000_array[y_B,x_B,z_B] = (DWI1000_array[y_B,x_B,z_B] - mean_DWI1000)/std_DWI1000  #高斯 normalize
    DWI1000_array = custom_normalize_1(DWI1000_array)
    #存出nii檔
    new_ADC_nii = nii_img_replace(ADC_nii, ADC_array)
    new_DWI1000_nii = nii_img_replace(DWI1000_nii, DWI1000_array)
    nib.save(new_ADC_nii, os.path.join(path_nii, case_name + '_ADC.nii.gz'))
    nib.save(new_DWI1000_nii, os.path.join(path_nii, case_name + '_DWI1000.nii.gz'))
    np.save(os.path.join(path_npy, case_name + '_ADC.npy'), ADC_array)
    np.save(os.path.join(path_npy, case_name + '_DWI1000.npy'), DWI1000_array)
    return 

def T2FLAIR_nor(path_nii, path_test_img, case_name):
    path_npy = os.path.join(path_test_img, 'npy')

    if not os.path.isdir(path_npy):  #如果資料夾不存在就建立
        os.mkdir(path_npy)

    ROWS = 512 #y
    COLS = 512 #x

    #讀取nifti檔
    T2FLAIR_nii = nib.load(os.path.join(path_nii, 'nii', case_name + '_T2FLAIR.nii.gz')) #讀取影像
    T2FLAIR_array = np.array(T2FLAIR_nii.dataobj).astype(float) #讀出image的array矩陣
    T2FLAIR_array = data_translate(T2FLAIR_array)
    X = np.zeros((T2FLAIR_array.shape[-1], ROWS, COLS, 1), dtype = 'float32')

    #一個NIFTI檔完成
    for i in range(T2FLAIR_array.shape[-1]):
        sample = T2FLAIR_array[:,:,i]
        l = sample.shape[0]
        w = sample.shape[1]
        img = np.expand_dims(np.expand_dims(sample, axis = 0), axis = -1)
        img = (img - np.mean(img)) / np.std(img)
        slice_y = int((ROWS - l)/2)
        slice_x = int((COLS - w)/2)
        X[i,slice_y:slice_y+l,slice_x:slice_x+w,:] = img
    
    np.save(os.path.join(path_npy, case_name + '_T2FLAIR.npy'), X)
    return

#%%這邊紀錄ICH的function
#path_dicom是整理完的C-影像，path_csv是輸出的csv位置
#做CT ICH model需要用到的csv，因為鴻鵠的api是要放csv
def make_clinical_study_csv(path_in, path_out, path_csv):
    DCMs = sorted(os.listdir(os.path.join(path_in, 'C-')))
    #接下來開始做成csv當複製影像整理出來
    DCM_list = [[y, pydicom.dcmread(os.path.join(path_in, 'C-', y))[0x20,0x0032].value[-1]] for y in DCMs] #(0020,0032) Image Position (Patient) [-119.600,-124.458,89.460]
    Position = np.array([y[-1] for y in DCM_list])
    sortP = np.sort(Position)
    gap = [sortP[y+1] - sortP[y] for y in range(len(sortP)-1)]
    #assert min(gap) > 3, 'gap差距過小，有重複掃描的狀況' 

    idxP = np.argsort(Position) #小到大，符合數據需求
    #filename	any	epidural	intraparenchymal	intraventricular	subarachnoid	subdural	ID	PatientID	Modality	StudyInstance	SeriesInstance	PhotoInterpretation	Position0	
    #Position1	Position2	Orientation0	Orientation1	Orientation2	Orientation3	Orientation4	Orientation5	PixelSpacing0	PixelSpacing1
    text = [['filename', 'any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'ID',
            'PatientID', 'Modality', 'StudyInstance', 'SeriesInstance', 'PhotoInterpretation', 'Position0', 
            'Position1', 'Position2', 'Orientation0', 'Orientation1', 'Orientation2', 'Orientation3', 'Orientation4',
            'Orientation5', 'PixelSpacing0', 'PixelSpacing1']]
    
    stop = []
    for i in range(len(DCMs)):
        DCM = pydicom.dcmread(os.path.join(path_in, 'C-', DCM_list[idxP[i]][0])) #讀取
        PID = str(DCM[0x10, 0x0020].value).rjust(8,'0') #(0010, 0020) Patient ID

        StudyInstance = DCM[0x20, 0x000d].value #(0020,000D) Study Instance UID
        if StudyInstance not in stop:
            stop.append(StudyInstance)

        SeriesInstance = DCM[0x20, 0x000e].value #(0020,000E) Series Instance UID
        Position0 = DCM[0x20, 0x0032].value[0] #(0020,0032) Image Position (Patient) [-131.000,-122.237,51.914]
        Position1 = DCM[0x20, 0x0032].value[1] #(0020,0032) Image Position (Patient) [-131.000,-122.237,51.914]
        Position2 = DCM[0x20, 0x0032].value[2] #(0020,0032) Image Position (Patient) [-131.000,-122.237,51.914]
        Orientation0 = DCM[0x20, 0x0037].value[0] #(0020,0037) Image Orientation (Patient) [1.000000,0.000000,0.000000,0.000000,0.976296,-0.216440]
        Orientation1 = DCM[0x20, 0x0037].value[1] #(0020,0037) Image Orientation (Patient) [1.000000,0.000000,0.000000,0.000000,0.976296,-0.216440]
        Orientation2 = DCM[0x20, 0x0037].value[2] #(0020,0037) Image Orientation (Patient) [1.000000,0.000000,0.000000,0.000000,0.976296,-0.216440]
        Orientation3 = DCM[0x20, 0x0037].value[3] #(0020,0037) Image Orientation (Patient) [1.000000,0.000000,0.000000,0.000000,0.976296,-0.216440]
        Orientation4 = DCM[0x20, 0x0037].value[4] #(0020,0037) Image Orientation (Patient) [1.000000,0.000000,0.000000,0.000000,0.976296,-0.216440]
        Orientation5 = DCM[0x20, 0x0037].value[5] #(0020,0037) Image Orientation (Patient) [1.000000,0.000000,0.000000,0.000000,0.976296,-0.216440]
        PixelSpacing0 = DCM[0x28, 0x0030].value[0] #(0028,0030) Pixel Spacing [0.488281,0.488281]
        PixelSpacing1 = DCM[0x28, 0x0030].value[1] #(0028,0030) Pixel Spacing [0.488281,0.488281]
        
        text.append([DCM_list[idxP[i]][0], 0, 0, 0, 0, 0, 0, DCM_list[idxP[i]][0][:-4], PID, 'CT', StudyInstance, 
                    SeriesInstance, 'MONOCHROME2', Position0, Position1, Position2, Orientation0, Orientation1,
                    Orientation2, Orientation3, Orientation4, Orientation5, PixelSpacing0, PixelSpacing1])
        
    #存出csv檔
    text_df = pd.DataFrame(text[1:], columns=text[0]) #我们将 data[0] 作为 columns 参数传递给 pd.DataFrame，这样它会将该列表的值作为列名。
    text_df.to_csv(os.path.join(path_csv, StudyInstance + '.csv'), index = False)
        
    #存影像資料夾
    if not os.path.isdir(os.path.join(path_out, StudyInstance)):
        shutil.copytree(os.path.join(path_in, 'C-'), os.path.join(path_out, StudyInstance))
    
    return

def make_clinicaldata_csv(path, path_excel, case_name):
    text = [['test_case_name']]

    text.append([case_name])
        
    #存成csv
    text_df = pd.DataFrame(text[1:], columns=text[0]) #我们将 data[0] 作为 columns 参数传递给 pd.DataFrame，这样它会将该列表的值作为列名。
    text_df.to_csv(os.path.join(path_excel, case_name + '.csv'), index=False)
    return

def model_ensemble(result1, result2, result3, result_out):
    smooth = 1e-7
    cnn1 = pd.read_csv(result1)
    IDs = list(cnn1['slice_name'])
    GT_ICH = list(cnn1['GT_ICH'])
    PRED_ICH_cnn1 = np.array(list(cnn1['PRED_ICH']))
    GT_EDH = list(cnn1['GT_EDH'])
    PRED_EDH_cnn1 = np.array(list(cnn1['PRED_EDH']))
    GT_IPH = list(cnn1['GT_IPH'])
    PRED_IPH_cnn1 = np.array(list(cnn1['PRED_IPH']))
    GT_IVH = list(cnn1['GT_IVH'])
    PRED_IVH_cnn1 = np.array(list(cnn1['PRED_IVH']))
    GT_SAH = list(cnn1['GT_SAH'])
    PRED_SAH_cnn1 = np.array(list(cnn1['PRED_SAH']))
    GT_SDH = list(cnn1['GT_SDH'])
    PRED_SDH_cnn1 = np.array(list(cnn1['PRED_SDH']))

    cnn2 = pd.read_csv(result2)
    PRED_ICH_cnn2 = np.array(list(cnn2['PRED_ICH']))
    PRED_EDH_cnn2 = np.array(list(cnn2['PRED_EDH']))
    PRED_IPH_cnn2 = np.array(list(cnn2['PRED_IPH']))
    PRED_IVH_cnn2 = np.array(list(cnn2['PRED_IVH']))
    PRED_SAH_cnn2 = np.array(list(cnn2['PRED_SAH']))
    PRED_SDH_cnn2 = np.array(list(cnn2['PRED_SDH']))

    cnn3 = pd.read_csv(result3)
    PRED_ICH_cnn3 = np.array(list(cnn3['PRED_ICH']))
    PRED_EDH_cnn3 = np.array(list(cnn3['PRED_EDH']))
    PRED_IPH_cnn3 = np.array(list(cnn3['PRED_IPH']))
    PRED_IVH_cnn3 = np.array(list(cnn3['PRED_IVH']))
    PRED_SAH_cnn3 = np.array(list(cnn3['PRED_SAH']))
    PRED_SDH_cnn3 = np.array(list(cnn3['PRED_SDH']))

    RankAvg = [['slice_name', 'PRED_ICH', 'PRED_EDH', 'PRED_IPH', 'PRED_IVH', 'PRED_SAH', 'PRED_SDH']]

    rankavg_ICH = (PRED_ICH_cnn1/np.max(PRED_ICH_cnn1 + smooth) + PRED_ICH_cnn2/np.max(PRED_ICH_cnn2 + smooth) + PRED_ICH_cnn3/np.max(PRED_ICH_cnn3 + smooth))/3
    rankavg_EDH = (PRED_EDH_cnn1/np.max(PRED_EDH_cnn1 + smooth) + PRED_EDH_cnn2/np.max(PRED_EDH_cnn2 + smooth) + PRED_EDH_cnn3/np.max(PRED_EDH_cnn3 + smooth))/3
    rankavg_IPH = (PRED_IPH_cnn1/np.max(PRED_IPH_cnn1 + smooth) + PRED_IPH_cnn2/np.max(PRED_IPH_cnn2 + smooth) + PRED_IPH_cnn3/np.max(PRED_IPH_cnn3 + smooth))/3
    rankavg_IVH = (PRED_IVH_cnn1/np.max(PRED_IVH_cnn1 + smooth) + PRED_IVH_cnn2/np.max(PRED_IVH_cnn2 + smooth) + PRED_IVH_cnn3/np.max(PRED_IVH_cnn3 + smooth))/3
    rankavg_SAH = (PRED_SAH_cnn1/np.max(PRED_SAH_cnn1 + smooth) + PRED_SAH_cnn2/np.max(PRED_SAH_cnn2 + smooth) + PRED_SAH_cnn3/np.max(PRED_SAH_cnn3 + smooth))/3
    rankavg_SDH = (PRED_SDH_cnn1/np.max(PRED_SDH_cnn1 + smooth) + PRED_SDH_cnn2/np.max(PRED_SDH_cnn2 + smooth) + PRED_SDH_cnn3/np.max(PRED_SDH_cnn3 + smooth))/3

    for idx, ID in enumerate(IDs):
        RankAvg.append([ID, rankavg_ICH[idx], rankavg_EDH[idx], rankavg_IPH[idx], rankavg_IVH[idx], rankavg_SAH[idx], rankavg_SDH[idx]])    
    
    #存出csv檔        
    RankAvg_df = pd.DataFrame(RankAvg[1:], columns=RankAvg[0]) #我们将 data[0] 作为 columns 参数传递给 pd.DataFrame，这样它会将该列表的值作为列名。
    RankAvg_df.to_csv(result_out, index=False)

    return


def result_slice_convert2_case(result_file):
    dtypes = {'slice_name': str}
    df = pd.read_csv(result_file, dtype=dtypes).fillna('')
    Slices = list(df['slice_name'])
    PredICHs = list(df['PRED_ICH'])
    PredEDHs = list(df['PRED_EDH'])
    PresIPHs = list(df['PRED_IPH'])
    PresIVHs = list(df['PRED_IVH'])
    PresSAHs = list(df['PRED_SAH'])
    PresSDHs = list(df['PRED_SDH'])

    #只有一個人
    max_ICH = round(np.max(np.array(PredICHs)), 2)
    max_EDH = round(np.max(np.array(PredEDHs)), 2)
    max_IPH = round(np.max(np.array(PresIPHs)), 2)
    max_IVH = round(np.max(np.array(PresIVHs)), 2)
    max_SAH = round(np.max(np.array(PresSAHs)), 2)
    max_SDH = round(np.max(np.array(PresSDHs)), 2)  

    return max_ICH, max_EDH, max_IPH, max_IVH, max_SAH, max_SDH
