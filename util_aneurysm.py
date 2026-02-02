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
from pydicom.uid import JPEGLSLossless 

import glob
import shutil
import nibabel as nib
import nibabel.processing
import matplotlib
import matplotlib.pyplot as plt
import sys
import logging
import cv2
import pandas as pd
import pydicom_seg
import SimpleITK as sitk
import matplotlib.colors as mcolors
#要安裝pillow 去開啟
from skimage import measure,color,morphology
from scipy.ndimage import rotate
from skimage.transform import resize
from scipy import ndimage
from collections import Counter
import pydicom_seg
import json
from collections import OrderedDict
from scipy import ndimage
from upload_orthanc import upload_data
import requests

import tensorflow as tf
import tensorflow.keras.backend as K
#print("Tensorflow version:", tf.__version__)
autotune = tf.data.experimental.AUTOTUNE
#s.environ["CUDA_VISIBLE_DEVICES"] = "1"
from collections import defaultdict


import torch
from torchvision.transforms.functional import rotate as rotate_torch
from torchvision.transforms import InterpolationMode

from create_dicomseg_multi_file_json_claude import load_and_sort_dicom_files, make_study_json, MaskRequest, make_study_series_json


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

def reslice_nifti_pred_nobrain(path_nii, path_reslice):
    img_nii = nib.load(os.path.join(path_nii, 'MRA_BRAIN.nii.gz')) #
    img = np.array(img_nii.dataobj) #讀出label的array矩陣      #
    pred_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz')) #
    vessel_nii = nib.load(os.path.join(path_nii, 'Vessel.nii.gz')) #
    original_affine = img_nii.affine.copy()  # 原始 affine
        
    img = data_translate(img, img_nii)
    y_i, x_i, z_i = img.shape #y, x, z 
        
    header_img = img_nii.header.copy() #抓出nii header 去算體積 
    pixdim_img = header_img['pixdim']  #可以借此從nii的header抓出voxel size
    
    #算出新版z軸要多少
    new_y_i = int(z_i * (pixdim_img[3] / pixdim_img[1]))

    #先把影像從 original*original*103轉成 original*original*103 * (pixdim_img[3] / pixdim_img[1])
    new_img_nii = nibabel.processing.conform(img_nii, ((x_i, y_i, new_y_i)),(pixdim_img[1], pixdim_img[2], pixdim_img[1]),order = 1) #影像用
    new_pred_nii = nibabel.processing.conform(pred_nii, ((x_i, y_i, new_y_i)),(pixdim_img[1], pixdim_img[2], pixdim_img[1]),order = 0) #影像用
    new_vessel_nii = nibabel.processing.conform(vessel_nii, ((x_i, y_i, new_y_i)),(pixdim_img[1], pixdim_img[2], pixdim_img[1]),order = 0) #影像用
    
    #讀出矩陣，然後第一值是負就反存，讀取 conform 後的 affine
    conformed_affine = new_img_nii.affine.copy()

    # 修正 affine，使其第一個 voxel dimension 方向與原始影像一致
    if np.sign(original_affine[0, 0]) != np.sign(conformed_affine[0, 0]):
        conformed_affine[0, :] *= -1  # 翻轉 X 軸方向
    
    # 建立新的影像
    fixed_img_nii = nib.Nifti1Image(new_img_nii.get_fdata(), conformed_affine, new_img_nii.header)
    fixed_pred_nii = nib.Nifti1Image(new_pred_nii.get_fdata().astype(int), conformed_affine, new_pred_nii.header)
    fixed_vessel_nii = nib.Nifti1Image(new_vessel_nii.get_fdata().astype(int), conformed_affine, new_vessel_nii.header)

    #輸出結果
    nib.save(fixed_img_nii, os.path.join(path_reslice, 'MRA_BRAIN.nii.gz'))
    nib.save(fixed_pred_nii, os.path.join(path_reslice, 'Pred.nii.gz')) 
    nib.save(fixed_vessel_nii, os.path.join(path_reslice, 'Vessel.nii.gz')) 

    return print('reslice OK!!!')

def reslice_nifti_label_pred_nobrain(path_nii, path_reslice):
    img_nii = nib.load(os.path.join(path_nii, 'MRA_BRAIN.nii.gz')) #
    img = np.array(img_nii.dataobj) #讀出label的array矩陣      #
    label_nii = nib.load(os.path.join(path_nii, 'Label.nii.gz')) #
    pred_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz')) #
    vessel_nii = nib.load(os.path.join(path_nii, 'Vessel.nii.gz')) #
    original_affine = img_nii.affine.copy()  # 原始 affine
        
    img = data_translate(img, img_nii)
    y_i, x_i, z_i = img.shape #y, x, z 
        
    header_img = img_nii.header.copy() #抓出nii header 去算體積 
    pixdim_img = header_img['pixdim']  #可以借此從nii的header抓出voxel size
    
    #算出新版z軸要多少
    new_y_i = int(z_i * (pixdim_img[3] / pixdim_img[1]))

    #先把影像從 original*original*103轉成 original*original*103 * (pixdim_img[3] / pixdim_img[1])
    new_img_nii = nibabel.processing.conform(img_nii, ((x_i, y_i, new_y_i)),(pixdim_img[1], pixdim_img[2], pixdim_img[1]),order = 1) #影像用
    new_label_nii = nibabel.processing.conform(label_nii, ((x_i, y_i, new_y_i)),(pixdim_img[1], pixdim_img[2], pixdim_img[1]),order = 0) #影像用
    new_pred_nii = nibabel.processing.conform(pred_nii, ((x_i, y_i, new_y_i)),(pixdim_img[1], pixdim_img[2], pixdim_img[1]),order = 0) #影像用
    new_vessel_nii = nibabel.processing.conform(vessel_nii, ((x_i, y_i, new_y_i)),(pixdim_img[1], pixdim_img[2], pixdim_img[1]),order = 0) #影像用
    
    #讀出矩陣，然後第一值是負就反存，讀取 conform 後的 affine
    conformed_affine = new_img_nii.affine.copy()

    # 修正 affine，使其第一個 voxel dimension 方向與原始影像一致
    if np.sign(original_affine[0, 0]) != np.sign(conformed_affine[0, 0]):
        conformed_affine[0, :] *= -1  # 翻轉 X 軸方向
    
    # 建立新的影像
    fixed_img_nii = nib.Nifti1Image(new_img_nii.get_fdata(), conformed_affine, new_img_nii.header)
    fixed_label_nii = nib.Nifti1Image(new_label_nii.get_fdata(), conformed_affine, new_label_nii.header)
    fixed_pred_nii = nib.Nifti1Image(new_pred_nii.get_fdata().astype(int), conformed_affine, new_pred_nii.header)
    fixed_vessel_nii = nib.Nifti1Image(new_vessel_nii.get_fdata().astype(int), conformed_affine, new_vessel_nii.header)

    #輸出結果
    nib.save(fixed_img_nii, os.path.join(path_reslice, 'MRA_BRAIN.nii.gz'))
    nib.save(fixed_label_nii, os.path.join(path_reslice, 'Label.nii.gz'))
    nib.save(fixed_pred_nii, os.path.join(path_reslice, 'Pred.nii.gz')) 
    nib.save(fixed_vessel_nii, os.path.join(path_reslice, 'Vessel.nii.gz')) 

    return print('reslice OK!!!')

def create_MIP_pred(path_dcm, path_nii, path_png, gpu_num, create_label_mip=False):

    #gpu_available = tf.config.list_physical_devices('GPU')
    #print(gpu_available)
    # set GPU
    #gpu_num = 0
    #tf.config.experimental.set_visible_devices(devices=gpu_available[gpu_num], device_type='GPU')

    #for gpu in gpu_available:
    #    tf.config.experimental.set_memory_growth(gpu, True)
    
    def img_to_MIPdicom(dcm, img, tag, series, SE, angle, count, fixed_slice_thickness=None):
        y_i, x_i = img.shape
        dcm.PixelData = img.tobytes()  # 影像置換
        dcm[0x08,0x0008].value = ['DERIVED', 'SECONDARY', 'OTHER']
        dcm[0x08,0x103E].value = tag
        dcm[0x20,0x0011].value = SE

        dcm.add_new(0x00280006, 'US', 1)
        dcm[0x28,0x0010].value = y_i
        dcm[0x28,0x0011].value = x_i
        dcm[0x28,0x0100].value = 16
        dcm[0x28,0x0101].value = 16
        dcm[0x28,0x0102].value = 15

        try:
            del dcm[0x28,0x1050]
        except:
            pass
        try:
            del dcm[0x28,0x1051]
        except:
            pass

        seriesiu = dcm[0x20,0x000E].value
        new_seriesiu = seriesiu + '.' + str(series)
        dcm[0x20,0x000E].value = new_seriesiu
        sopiu = dcm[0x08,0x0018].value
        new_sopiu = sopiu + '.' + str(series) + str(angle)
        dcm[0x08,0x0018].value = new_sopiu

        PixelSpacing = dcm[0x28,0x0030].value
        ImagePosition = dcm[0x20,0x0032].value
        ImageOrientation = dcm[0x20,0x0037].value

        # 取得 Image Orientation 中的兩個向量
        row_vector = np.array(ImageOrientation[:3], dtype=np.float64)
        col_vector = np.array(ImageOrientation[3:], dtype=np.float64)

        # 計算法向量 (cross product)
        normal_vector = np.cross(row_vector, col_vector)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 標準化

        # ===== 關鍵改進：確保全系列使用一致的 slice_thickness =====
        # 1. 優先使用傳入的固定 slice_thickness（全系列一致）
        if fixed_slice_thickness is not None:
            slice_thickness = fixed_slice_thickness
        elif hasattr(dcm, 'SliceThickness') and dcm.SliceThickness:
            slice_thickness = float(dcm.SliceThickness)
        elif hasattr(dcm, 'SpacingBetweenSlices') and dcm.SpacingBetweenSlices:
            slice_thickness = float(dcm.SpacingBetweenSlices)
        else:
            # 使用 PixelSpacing 的平均值作為合理默認值（保持各向同性）
            slice_thickness = float(np.mean(PixelSpacing))
        
        # 2. 確保設置 SliceThickness tag（dcm2niix 會參考）
        dcm.SliceThickness = slice_thickness
        
        # 3. 設置 SpacingBetweenSlices（更精確描述切片間距）
        dcm.SpacingBetweenSlices = slice_thickness

        # 4. 計算新位置（使用 float64 精度）
        new_position = np.array(ImagePosition, dtype=np.float64) - normal_vector * slice_thickness * count

        dcm[0x20,0x0032].value = [float(new_position[0]), float(new_position[1]), float(new_position[2])]
        dcm[0x20,0x1041].value = float(new_position[2])
        dcm[0x20,0x0013].value = int(count)

        return dcm
    
    #跟gpu相關的function
    def dilation3d(x):
        kernel = np.ones((3, 3, 3), dtype=int) #3d dilation
        x_di = ndimage.binary_dilation(x, structure=kernel, iterations=15) #亂做n次，kernel越小算越快
        return x_di

    def rotate_tf(x, i, axes):
        c = np.zeros((int(x.shape[0]), int(x.shape[1]), 1))
        for j in range(0,i*5, i):
            r_x = rotate(x, j, axes, reshape=False)
            MIP_x = createMIP(r_x)
            c = np.concatenate([c, np.expand_dims(MIP_x, axis=-1)], axis = -1)
        #把第一個去除
        return c[:,:,1:]

    def rotation_3d(X, axis, theta, expand=True, fill=0.0, label=False, gpu=None):
        """
        The rotation is based on torchvision.transforms.functional.rotate, which is originally made for a 2d image rotation
        :param X: the data that should be rotated, a torch.tensor or an ndarray, with lenx * leny * lenz shape.
        :param axis: the rotation axis based on the keynote request. 0 for x axis, 1 for y axis, and 2 for z axis.
        :param expand:  (bool, optional) – Optional expansion flag. If true, expands the output image to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.
        :param fill:  (sequence or number, optional) –Pixel fill value for the area outside the transformed image. If given a number, the value is used for all bands respectively.
        :param theta: the rotation angle, Counter-clockwise rotation, [-180, 180] degrees.
        :return: rotated tensor.
        expand=True時要修正旋轉的大小
        原版是Z, H, W這樣的矩陣方向再去旋轉
        我這邊矩陣放 H, W, Z，所以在axis=0時變成左右轉
        """
        #先獲得x的長寬以用於更新尺寸
        #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        """
        初始化類並設置裝置。
        
        參數：
        - gpu：可選的整數，指定要使用的 GPU 設備編號（例如0、1、2、3等）。如果為 None，則根據 CUDA 是否可用選擇默認 GPU。
        """
        if gpu is not None:
            device = f'cuda:{gpu}'
        else:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if isinstance(X, np.ndarray):
            # Make a copy of the array to avoid negative strides issue
            X = np.copy(X)
            X = np.copy(X)
            X = torch.from_numpy(X).float()
            
        X = X.to(device)

        if axis == 0:
            interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
            X = rotate_torch(X, interpolation=interpolation_mode, angle=theta, expand=expand, fill=fill)
            
        elif axis == 1:
            X = X.permute((1, 0, 2))
            interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
            X = rotate_torch(X, interpolation=interpolation_mode, angle=theta, expand=expand, fill=fill)
            X = X.permute((1, 0, 2))
            X = torch.flip(X, [2])

        elif axis == 2:
            X = X.permute((2, 1, 0))
            interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
            X = rotate_torch(X, interpolation=interpolation_mode, angle=-theta, expand=expand, fill=fill)
            X = X.permute((2, 1, 0))
            X = torch.flip(X, [2]) #這個還要修正
            
        else:
            raise Exception('Not invalid axis')

        return X.squeeze(0)

    # def createMIP_single(np_img):
    #     Mip_z = np.max(np_img, axis=-1)
    #     return Mip_z
    def createMIP_single(tensor3d):
        return torch.amax(tensor3d, dim=-1)    

    class createMIP:
        # 使用示例：
        # createMIP
        #createMIP = createMIP()

        # 假設已定義了translated_vessel_img、angle、angle_one、y_i和x_i
        # 呼叫process_images方法來處理圖像
        #processed_images = rotator.process_images(translated_vessel_img, angle, angle_one, y_i, x_i)
        
        def __init__(self):
            #__init__：初始化類並設置裝置（如果GPU可用則為'cuda:0'，否則為'cpu'）。
            #self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            return

        def rotation_3d(self, X, axis, theta, expand=True, fill=0.0, label=False):
            """
            將一個3D tensor或numpy陣列沿指定軸進行旋轉。

            參數：
            - X：torch.tensor或np.ndarray，輸入的資料，形狀為(lenx, leny, lenz)。
            - axis：整數，旋轉軸（0表示x軸，1表示y軸，2表示z軸）。 0目前是水平轉, 1是垂直轉, 2是側手翻
            - theta：浮點數，旋轉角度（遞減角度為正）。
            - expand：布林值，可選，擴展輸出圖像大小（默認為True）。
            - fill：序列或數字，可選，填充值，用於轉換圖像外部區域。
            - label：布林值，可選，對標籤使用最近鄰插值（默認為False）。

            返回：
            - 旋轉後的tensor：torch.tensor，旋轉後的輸入tensor。
            """
            if isinstance(X, np.ndarray):
                X = np.copy(X)
                #X = torch.from_numpy(X).float()
                X = torch.from_numpy(X)
            
            X = X.to(self.device)

            if axis == 0:
                interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
                X = rotate_torch(X, interpolation=interpolation_mode, angle=theta, expand=expand, fill=fill)
            
            elif axis == 1:
                X = X.permute((1, 0, 2))
                interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
                X = rotate_torch(X, interpolation=interpolation_mode, angle=theta, expand=expand, fill=fill)
                X = X.permute((1, 0, 2))
                X = torch.flip(X, [2]) #這個還要修正

            elif axis == 2:
                X = X.permute((2, 1, 0))
                interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
                X = rotate_torch(X, interpolation=interpolation_mode, angle=-theta, expand=expand, fill=fill)
                X = X.permute((2, 1, 0))
                X = torch.flip(X, [2]) #這個還要修正
        
            else:
                raise ValueError('無效的軸值。預期為0、1或2。')

            return X

        def process_images(self, translated_img, angle, angle_step, y_i, x_i, axis, label=False, gpu=None):
            """
            初始化類並設置裝置。
            
            參數：
            - gpu：可選的整數，指定要使用的 GPU 設備編號（例如0、1、2、3等）。如果為 None，則根據 CUDA 是否可用選擇默認 GPU。
            """
            if gpu is not None:
                self.device = f'cuda:{gpu}'
            else:
                self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            img_list = []

            #這邊補0的大小要跟axial view一樣


            angle_list = np.arange(0, angle, angle_step)
            
            for i in angle_list:
                if axis == 0:
                    output1 = self.rotation_3d(translated_img, 0, -float(i), expand=True, label=label) #水平
                elif axis == 1:
                    output1 = self.rotation_3d(translated_img, 1, -float(i), expand=True, label=label)  # 垂直旋轉
                    
                MIP = torch.zeros(y_i, x_i).to(self.device)

                #到一個output1 => 4g
                MIP_r = output1.amax(-1)
                y_r, x_r = MIP_r.shape[0], MIP_r.shape[1]

                if axis == 0:
                    if x_r > x_i:
                        slice_y = int((y_i - y_r) / 2)
                        slice_x = int((x_r - x_i) / 2)
                        MIP[slice_y:slice_y+y_r,:] = MIP_r[:,slice_x:slice_x+x_i]
                    else:
                        slice_y = int((y_i - y_r) / 2)
                        slice_x = int((x_i - x_r) / 2)
                        MIP[slice_y:slice_y+y_r,slice_x:slice_x+x_r] = MIP_r            
                
                elif axis == 1:
                    if y_r > y_i:
                        slice_y = int((y_r - y_i) / 2)
                        slice_x = int((x_r - x_i) / 2)
                        MIP = MIP_r[slice_y:slice_y + y_i, slice_x:slice_x + x_i]
                    else:
                        slice_y = int((y_i - y_r) / 2)
                        slice_x = int((x_i - x_r) / 2)
                        MIP[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = MIP_r

                img_list.append(MIP)

            mip_img = torch.stack(img_list, dim=-1)
            #mip_img = torch.flip(mip_img, dims=[0])
            #mip_img = mip_img.permute(1, 2, 0)
            mip_img_np = mip_img.cpu().numpy()

            return mip_img_np
        
    
    #以下開始製作MIP跟label
    angle = 180
    angle_step = 3
    Series = ['MIP_Pitch', 'MIP_Yaw'] #用作2種方式
    #Series = ['MIP_Yaw', 'MIP_Pitch'] #用作2種方式
    
    #血管的mask也要跟著轉然後保存
    start = time.time()
    path_dcms = os.path.join(path_dcm, 'MRA_BRAIN')
    #先讀影像
    img_nii = nib.load(os.path.join(path_nii, 'MRA_BRAIN.nii.gz')) #ADC要讀因為會取ADC值判斷
    img = np.array(img_nii.dataobj) #讀出label的array矩陣      #256*256*22
    pred_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz')) #ADC要讀因為會取ADC值判斷
    pred = np.array(pred_nii.dataobj) #讀出label的array矩陣      #256*256*22 
    vessel_nii = nib.load(os.path.join(path_nii, 'Vessel.nii.gz')) #ADC要讀因為會取ADC值判斷
    vessel = np.array(vessel_nii.dataobj) #讀出label的array矩陣      #256*256*22     
    
    # 可選載入 label_nii，處理方式與 pred_nii 完全相同
    if create_label_mip:
        label_nii = nib.load(os.path.join(path_nii, 'Label.nii.gz'))
        label = np.array(label_nii.dataobj) #讀出label的array矩陣      #256*256*22
    
    img = data_translate(img, img_nii)
    pred = data_translate(pred, pred_nii)
    vessel = data_translate(vessel, vessel_nii)
    if create_label_mip:
        label = data_translate(label, label_nii)

    #先找出投影后最前面那點pred的位置，以下都改成pred
    print('np.max(pred):', np.max(pred), type(np.max(pred)))
    pred_num = int(np.round(np.max(pred)))

    #以下改用tf的gpu測試
    vessel_tf = tf.convert_to_tensor(vessel, dtype=tf.bool)
    dilated_tf_npfunc = tf.numpy_function(dilation3d, [vessel_tf],tf.int32)
    vessel_di = dilated_tf_npfunc.numpy()
    
    #使用mask，brain mask要補齊vessel mask的部分，不然血管像被切掉
    vessel_img = (img * vessel_di).copy()
    
    #把影像的y軸裁切，使得旋轉時置中
    vessel_z = np.sum(vessel_img,  axis=(0, 1)) #對z軸sum
    vessel_z_list = [y for y, z in enumerate(vessel_z) if z > 0] #做前面那點label是在哪裡
    vessel_img = vessel_img[:,:,vessel_z_list[0]:vessel_z_list[-1]]
    vessel = vessel[:,:,vessel_z_list[0]:vessel_z_list[-1]]
    pred = pred[:,:,vessel_z_list[0]:vessel_z_list[-1]]
    if create_label_mip:
        label = label[:,:,vessel_z_list[0]:vessel_z_list[-1]]
    y_i, x_i, z_i = img.shape

    #讀取第一張dicom，剩下的資訊用這張來改
    reader = sitk.ImageSeriesReader()
    dcms_tofmra = reader.GetGDCMSeriesFileNames(path_dcms)
    
    # ===== 計算統一的 slice_thickness（確保全系列幾何一致性）=====
    if len(dcms_tofmra) >= 2:
        # 方法1：從相鄰切片的 Image Position 計算真實間距
        dcm0 = pydicom.dcmread(dcms_tofmra[0])
        dcm1 = pydicom.dcmread(dcms_tofmra[1])
        pos0 = np.array(dcm0.ImagePositionPatient)
        pos1 = np.array(dcm1.ImagePositionPatient)
        calculated_spacing = float(np.linalg.norm(pos1 - pos0))
        print(f"從 Image Position 計算的 slice spacing: {calculated_spacing:.4f} mm")
    else:
        # 方法2：從單個 DICOM 的 tag 讀取
        dcm0 = pydicom.dcmread(dcms_tofmra[0])
        if hasattr(dcm0, 'SliceThickness') and dcm0.SliceThickness:
            calculated_spacing = float(dcm0.SliceThickness)
            print(f"從 SliceThickness tag 讀取: {calculated_spacing:.4f} mm")
        elif hasattr(dcm0, 'SpacingBetweenSlices') and dcm0.SpacingBetweenSlices:
            calculated_spacing = float(dcm0.SpacingBetweenSlices)
            print(f"從 SpacingBetweenSlices tag 讀取: {calculated_spacing:.4f} mm")
        else:
            # 方法3：使用 PixelSpacing 的平均值（保持各向同性）
            calculated_spacing = float(np.mean(dcm0.PixelSpacing))
            print(f"使用 PixelSpacing 平均值: {calculated_spacing:.4f} mm")
    
    print(f"MIP 序列將使用統一的 slice_thickness: {calculated_spacing:.4f} mm")
    
    #接下來做上下左右轉，需要先將矩陣轉置
    translated_vessel_img = np.swapaxes(vessel_img,0,-1).copy()
    #translated_vessel_img = np.flip(translated_vessel_img, 1)
    
    #label跟vessel還是要轉方向
    translated_vessel = np.swapaxes(vessel,0,-1).copy()
    #translated_vessel = np.flip(translated_vessel, 1)
    translated_pred = np.swapaxes(pred,0,-1).copy()
    translated_pred = torch.from_numpy(translated_pred).to(torch.int16).to("cuda")
    if create_label_mip:
        translated_label = np.swapaxes(label,0,-1).copy()
        # 先四捨五入，再轉成int16，避免浮點數截斷導致標籤值錯誤
        translated_label = np.round(translated_label).astype(np.int16)
        translated_label = torch.from_numpy(translated_label).to("cuda")

    print('img.shape:', img.shape,'translated_pred:', translated_pred.shape) 
    for series in Series:
        path_vesselMIP = os.path.join(path_dcm, series)    
        if not os.path.isdir(path_vesselMIP):
            os.mkdir(path_vesselMIP)
    
        #這邊讀取旋轉mark的圖片，準備做第一章的圖片疊圖
        png_landmark = cv2.imread(os.path.join(path_png, series + '.png'))
        png_landmark = png_landmark[:,:,0]    

        if series == 'MIP_Pitch':
            axis = 1
            SE = 850
            series_uid = 3
        elif series == 'MIP_Yaw':
            axis = 0
            SE = 851
            series_uid = 6
            
        #先做垂直轉Pitch，再做水平轉Yaw
        start_img = time.time()
        createrMIP = createMIP()
        # 假設已定義了translated_vessel_img、angle、angle_one、y_i和x_i。 y_i和x_i是轉出目標影像的大小，以背景補0為主
        # 呼叫process_images方法來處理圖像
        MIP_images = createrMIP.process_images(translated_vessel_img, angle, angle_step, y_i, x_i, axis=axis, label=False, gpu=gpu_num).astype('int16')
        #print('Time taken for {} sec\n'.format(time.time()-start_img), ' MIP_images is OK!!!, Next...')
    
        #血管確認是否有阻擋是改成前後一起看，所以可以當作MIP後確認重疊比例，所以可以label也做mip、vessel也做mip
        #MIP血管還是要做，因為要產生血管的mask
        MIP_vessels = createrMIP.process_images(translated_vessel, angle, angle_step, y_i, x_i, axis=axis, label=True, gpu=gpu_num).astype('int16')
        #print('Time taken for {} sec\n'.format(time.time()-start), 'MIP_images.shape:', MIP_images.shape)

        #label比較麻煩，需要重新根據旋轉的血管結果修正
        count_pred = 0 #紀錄現在是第幾張
        cover_ranges_pred = np.zeros((int(MIP_images.shape[-1]) , pred_num)) #記錄哪張血管遮擋最少

        angle_list = np.arange(0, angle, angle_step)
        #new_pred = np.zeros((y_i, x_i, angle_list.shape[0])) #會跟mip的影像一樣大，雖已知大小，但跟for迴圈有關，跟旋轉的角度一樣多
        # Rewritten version for CUDA computation
        new_pred = torch.zeros((y_i, x_i, angle_list.shape[0]), dtype=torch.int16, device='cuda')

        for i in angle_list:
            #結果還是要做出血管個別旋轉的圖
            if axis == 0:
                rotated_vessel = rotation_3d(translated_vessel,  axis, -float(i), expand=True, label=True, gpu=gpu_num) #rotated_vessel大小會一直變動
            elif axis ==1:
                rotated_vessel = rotation_3d(translated_vessel,  axis, -float(i),  expand=True, label=True, gpu=gpu_num) #rotated_vessel大小會一直變動
            
            new_pred_slice = torch.zeros((y_i, x_i), dtype=torch.int16, device='cuda')
            for j in range(pred_num):
                pred_one = (translated_pred == j + 1).to(torch.uint8)

                if axis == 0:
                    rotated_pred_one = rotation_3d(pred_one,  axis, -float(i), expand=True, label=True, gpu=gpu_num) #rotated_vessel大小會一直變動
                elif axis == 1:
                    rotated_pred_one = rotation_3d(pred_one,  axis, -float(i),  expand=True, label=True, gpu=gpu_num) #rotated_vessel大小會一直變動

                pred_z = rotated_pred_one.sum(dim=(0, 1)) #對z軸sum
                pred_z_list = torch.nonzero(pred_z > 0, as_tuple=False).squeeze() #做前面那點label是在哪裡

                if pred_z_list.numel() == 0:
                    pred_z_list = torch.tensor([1], device='cuda')
                
                index_front = max(int(pred_z_list[0]) - 3, 1)
                index_back = min(int(pred_z_list[-1]) + 3, rotated_vessel.shape[-1] - 1)

                #對label做projection去獲得投影後的x,y
                pred_mip = createMIP_single(rotated_pred_one)
                pred_mip_s = pred_mip.clone() #原版都沒阻擋

                vessel_mip_front = createMIP_single(rotated_vessel[:, :, :index_front])
                pred_mip[vessel_mip_front > 0] = 0 #前面有阻擋到的就等於0

                vessel_mip_back = createMIP_single(rotated_vessel[:, :, index_back:])
                pred_cover = pred_mip.clone()
                pred_cover[vessel_mip_back > 0] = 0 #後面有阻擋到的就等於0

                extra_pred = torch.zeros((y_i, x_i), dtype=torch.int16, device='cuda')
                y_r, x_r = pred_mip.shape
                
                if axis == 0:
                    if x_r > x_i:
                        slice_y = (y_i - y_r) // 2
                        slice_x = (x_r - x_i) // 2
                        extra_pred[slice_y:slice_y + y_r, :] = pred_mip[:, slice_x:slice_x + x_i]
                    else:
                        slice_y = (y_i - y_r) // 2
                        slice_x = (x_i - x_r) // 2
                        extra_pred[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = pred_mip
                else:
                    if y_r > y_i:
                        slice_y = (y_r - y_i) // 2
                        slice_x = (x_r - x_i) // 2
                        extra_pred = pred_mip[slice_y:slice_y + y_i, slice_x:slice_x + x_i]
                    else:
                        slice_y = (y_i - y_r) // 2
                        slice_x = (x_i - x_r) // 2
                        extra_pred[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = pred_mip
                
                new_pred_slice[extra_pred > 0] = j + 1
    
                #這邊先根據MIP計算出遮擋最大值，這邊統計一個數字是標註被血管擋掉的百分比(前後)，減少跑for迴圈
                #print('slice:', i, ' ane:', j, ' np.sum(label_cover):', np.sum(label_cover), ' np.sum(extra_label):', np.sum(extra_label))
                if torch.sum(extra_pred) > 0:
                    cover_range = torch.sum(pred_cover).float() / torch.sum(pred_mip_s).float()
                    cover_range = min(cover_range.item(), 0.98)
                else:
                    cover_range = 0.0
                cover_ranges_pred[count_pred, j] = cover_range

            new_pred[:, :, count_pred] = new_pred_slice
            count_pred += 1

        # 將 pred 移到 CPU（但不刪除 translated_pred，因為下一個 series 還需要使用）
        new_pred = new_pred.cpu().numpy().astype(np.int16)
        
        # 如果需要處理 label，在 pred 處理完成後才開始（節省 GPU RAM）
        if create_label_mip:
            print('開始處理 label MIP...')
            count_label = 0
            label_num = int(np.round(np.max(label)))
            cover_ranges_label = np.zeros((int(MIP_images.shape[-1]) , label_num))
            new_label = torch.zeros((y_i, x_i, angle_list.shape[0]), dtype=torch.int16, device='cuda')
            
            for i in angle_list:
                # 重新計算旋轉的血管（與 pred 循環相同）
                if axis == 0:
                    rotated_vessel = rotation_3d(translated_vessel,  axis, -float(i), expand=True, label=True, gpu=gpu_num)
                elif axis ==1:
                    rotated_vessel = rotation_3d(translated_vessel,  axis, -float(i),  expand=True, label=True, gpu=gpu_num)
                
                new_label_slice = torch.zeros((y_i, x_i), dtype=torch.int16, device='cuda')
                for j in range(label_num):
                    label_one = (translated_label == j + 1).to(torch.uint8)

                    if axis == 0:
                        rotated_label_one = rotation_3d(label_one,  axis, -float(i), expand=True, label=True, gpu=gpu_num)
                    elif axis == 1:
                        rotated_label_one = rotation_3d(label_one,  axis, -float(i),  expand=True, label=True, gpu=gpu_num)

                    label_z = rotated_label_one.sum(dim=(0, 1))
                    label_z_list = torch.nonzero(label_z > 0, as_tuple=False).squeeze()

                    if label_z_list.numel() == 0:
                        label_z_list = torch.tensor([1], device='cuda')
                    
                    index_front = max(int(label_z_list[0]) - 3, 1)
                    index_back = min(int(label_z_list[-1]) + 3, rotated_vessel.shape[-1] - 1)

                    label_mip = createMIP_single(rotated_label_one)
                    label_mip_s = label_mip.clone()

                    vessel_mip_front = createMIP_single(rotated_vessel[:, :, :index_front])
                    label_mip[vessel_mip_front > 0] = 0

                    vessel_mip_back = createMIP_single(rotated_vessel[:, :, index_back:])
                    label_cover = label_mip.clone()
                    label_cover[vessel_mip_back > 0] = 0

                    extra_label = torch.zeros((y_i, x_i), dtype=torch.int16, device='cuda')
                    y_r, x_r = label_mip.shape
                    
                    if axis == 0:
                        if x_r > x_i:
                            slice_y = (y_i - y_r) // 2
                            slice_x = (x_r - x_i) // 2
                            extra_label[slice_y:slice_y + y_r, :] = label_mip[:, slice_x:slice_x + x_i]
                        else:
                            slice_y = (y_i - y_r) // 2
                            slice_x = (x_i - x_r) // 2
                            extra_label[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = label_mip
                    else:
                        if y_r > y_i:
                            slice_y = (y_r - y_i) // 2
                            slice_x = (x_r - x_i) // 2
                            extra_label = label_mip[slice_y:slice_y + y_i, slice_x:slice_x + x_i]
                        else:
                            slice_y = (y_i - y_r) // 2
                            slice_x = (x_i - x_r) // 2
                            extra_label[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = label_mip
                    
                    new_label_slice[extra_label > 0] = j + 1
        
                    if torch.sum(extra_label) > 0:
                        cover_range = torch.sum(label_cover).float() / torch.sum(label_mip_s).float()
                        cover_range = min(cover_range.item(), 0.98)
                    else:
                        cover_range = 0.0
                    cover_ranges_label[count_label, j] = cover_range

                new_label[:, :, count_label] = new_label_slice
                count_label += 1
            
            # 將 label 移到 CPU（但不刪除 translated_label，因為下一個 series 還需要使用）
            new_label = new_label.cpu().numpy().astype(np.int16)
            print('label MIP 處理完成')

        #這邊在第一張時多做一個含有landmark的圖
        MIP_mark = MIP_images[:,:,0].astype('int16').copy()
        fig_pitch = np.zeros((y_i, x_i)).astype('int16')
        #將圖帶入然後設定下大小，png_pitch縮小
        #因為輸入影像有大小章關係，所以用出動態的縮放值
        percentage_x = y_i / 512
        y_displacement = int(250*percentage_x)
        x_displacement = int(110*percentage_x)
        
        pitch_resize = resize(png_landmark, (int((png_landmark.shape[0]*percentage_x)/3),int((png_landmark.shape[1]*percentage_x)/3)), order=0, mode='symmetric', cval=0, clip=False, preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None)
        fig_pitch[y_displacement:pitch_resize.shape[0]+y_displacement, x_displacement:pitch_resize.shape[1]+x_displacement] = pitch_resize
        fig_pitch[fig_pitch > 0] = np.max(MIP_images[:,:,0]) #因為疊上去讓他報數值，這樣背景就會變暗
        output_pitch = cv2.add(MIP_mark, fig_pitch)  # 疊加第一張
        dcm_slice = pydicom.dcmread(dcms_tofmra[-1]) #dicom每次都會連著修改一下，所以範本要在迴圈中讀取
        print("Transfer syntax:", dcm_slice.file_meta.TransferSyntaxUID)
        img_example = dcm_slice.pixel_array
        print("Image shape:", img_example.shape, "dtype:", img_example.dtype)
        new_dcm_seg = img_to_MIPdicom(dcm_slice, output_pitch, series, series_uid, SE, '00', 0, 
                                       fixed_slice_thickness=calculated_spacing) #從0開始，使用統一的 slice_thickness
        new_dcm_seg.save_as(os.path.join(path_vesselMIP, series + '_' + str(0).rjust(4,'0') + '_.dcm')) #存出新dicom，同存同一層，因為有壓時間，所以不會互相覆蓋        

        #因為做出新標註需要影像的affine matrix，所以這邊先做出dicom並轉出nifti
        for k in range(int(MIP_images.shape[-1])):    
            dcm_slice = pydicom.dcmread(dcms_tofmra[-1]) #dicom每次都會連著修改一下，所以範本要在迴圈中讀取
            new_dcm_seg = img_to_MIPdicom(dcm_slice, MIP_images[:,:,k], series, series_uid, SE, k, k+1,
                                          fixed_slice_thickness=calculated_spacing) #最後一項是image position所以多下降一次，使用統一的 slice_thickness
            new_dcm_seg.save_as(os.path.join(path_vesselMIP, series + '_' + str(k).rjust(4,'0') + '.dcm')) #存出新dicom，同存同一層，因為有壓時間，所以不會互相覆蓋        
    
        #先製作出影像的nifti，如後讀取後將相關資訊套用到新label中，以下跑dcm2niix
        bash_line = 'dcm2niix -z y -f ' + series + ' -o ' + path_nii + ' ' + path_vesselMIP #把tag檔案複製，-v : verbose (n/y or 0/1/2, default 0) [no, yes, logorrheic]
        #print('bash_line:', bash_line)
        os.system(bash_line) 
        
        #讀取nifti取得新affine matrix
        img_pitch_nii = nib.load(os.path.join(path_nii, series + '.nii.gz')) #ADC要讀因為會取ADC值判斷

        #這邊先做血管mask存出mask，取完前三張，最後加上1張最頂端landmark，再重組回去
        vessel_landmark = np.zeros((y_i, x_i, 1))
        vessel_pitch = np.concatenate([vessel_landmark, MIP_vessels], axis=-1)
        #存出nifti
        vessel_pitch = data_translate_back(vessel_pitch, img_pitch_nii).astype('int16')
        vessel_pitch_nii = nii_img_replace(img_pitch_nii, vessel_pitch)
        nib.save(vessel_pitch_nii, os.path.join(path_nii, series + '_vessel.nii.gz')) 

        #for迴圈完成的標註，這邊開始處理更新版標註，差異在只取重疊率最小加上+-1張
        #new_pred = new_pred[:,:,1:]    
        pred_top = np.zeros((new_pred.shape)).astype(int)
        #print('cover_ranges:', cover_ranges)
        #每顆標註只取top3張，規則是找沒有被1. 血管阻擋的 2. 之後最大面積
        for l in range(pred_num):
            cover_list = cover_ranges_pred[:,l] #遮擋率
            cover_z = np.where(cover_list == np.max(cover_list))[0] #第一個條件 => 遮擋率最低
            predone = (new_pred ==l+1).astype(int).copy()
            areas = list(np.sum(predone[:,:,cover_z], axis = (0, 1))) #第二條件，如果相同遮中，遮擋率都是1)，即可知道哪張切片面積最大，cover_z這幾張比大小
                    
            index_area = cover_z[areas.index(max(areas))]
            pred_area = np.zeros((new_pred.shape))
            #判斷位置只取3張的mask
            if index_area == 0:
                pred_area[:,:,0:3] = predone[:,:,0:3]
            elif index_area == new_pred.shape[-1]:
                pred_area[:,:,-3:] = predone[:,:,-3:]
            else:
                pred_area[:,:,index_area-1:index_area+2] = predone[:,:,index_area-1:index_area+2]
            pred_top[pred_area > 0] = int(l+1)
            # if series == 'MIP_Yaw' and l==1:
            #     print(series, ' cover_list:', cover_list)
            #     print('np.sum(label_area:):', np.sum(label_area))
            #     print('areas:', areas, ' index_area:', index_area)

        #取完前三張，最後加上1張最頂端landmark，再重組回去
        pred_landmark = np.zeros((y_i, x_i, 1))
        pred_top = np.concatenate([pred_landmark, pred_top], axis=-1) #加入第一張空的
    
        #存出nifti
        pred_top_save = data_translate_back(pred_top, img_pitch_nii).astype('int16')
        pred_top_save_nii = nii_img_replace(img_pitch_nii, pred_top_save)
        nib.save(pred_top_save_nii, os.path.join(path_nii, series + '_pred.nii.gz')) 
        
        # 如果需要處理 label，執行與 pred 完全相同的邏輯
        if create_label_mip:
            label_top = np.zeros((new_label.shape)).astype(int)
            for l in range(label_num):
                cover_list = cover_ranges_label[:,l]
                cover_z = np.where(cover_list == np.max(cover_list))[0]
                labelone = (new_label ==l+1).astype(int).copy()
                areas = list(np.sum(labelone[:,:,cover_z], axis = (0, 1)))
                        
                index_area = cover_z[areas.index(max(areas))]
                label_area = np.zeros((new_label.shape))
                if index_area == 0:
                    label_area[:,:,0:3] = labelone[:,:,0:3]
                elif index_area == new_label.shape[-1]:
                    label_area[:,:,-3:] = labelone[:,:,-3:]
                else:
                    label_area[:,:,index_area-1:index_area+2] = labelone[:,:,index_area-1:index_area+2]
                label_top[label_area > 0] = int(l+1)

            label_landmark = np.zeros((y_i, x_i, 1))
            label_top = np.concatenate([label_landmark, label_top], axis=-1)
        
            label_top_save = data_translate_back(label_top, img_pitch_nii).astype('int16')
            label_top_save_nii = nii_img_replace(img_pitch_nii, label_top_save)
            nib.save(label_top_save_nii, os.path.join(path_nii, series + '_label.nii.gz'))
        
        #print('Time taken for {} sec\n'.format(time.time()-start))

    # 所有 series 處理完成後，釋放 GPU 記憶體
    del translated_pred
    if create_label_mip:
        del translated_label
    torch.cuda.empty_cache()


class AneurysmPipeline:
    def __init__(self, path_dcm, path_nii, path_excel, patient_id):
        self.path_dcm = path_dcm
        self.path_nii = path_nii
        self.path_excel = path_excel
        self.patient_id = patient_id

    ## 使用 CPU 版本 dilation（替代 GPU PyTorch）
    @staticmethod
    def dilation3d(x):
        kernel = np.ones((3, 3, 3), dtype=int)
        x_di = ndimage.binary_dilation(x, structure=kernel, iterations=1)
        return x_di
    
    #會使用到的一些predict技巧
    @staticmethod
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

    @staticmethod
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
    @staticmethod
    def nii_img_replace(data, new_img):
        affine = data.affine
        header = data.header.copy()
        new_nii = nib.nifti1.Nifti1Image(new_img, affine, header=header)
        return new_nii

    def make_aneurysm_vessel_location_16labels_pred(self):
        label_nii = nib.load(os.path.join(self.path_nii, 'Pred.nii.gz'))
        label = np.array(label_nii.dataobj)
        label = self.data_translate(label, label_nii)

        vessel_nii = nib.load(os.path.join(self.path_nii, 'Vessel_16.nii.gz'))
        vessel = np.array(vessel_nii.dataobj)
        vessel = self.data_translate(vessel, vessel_nii)

        new_label = np.zeros(label.shape, dtype=int)

        for i in range(int(np.max(label))):
            label_one = label == i + 1
            if np.sum(label_one * vessel) > 0:
                ane_overlap = label_one * vessel
                non_zero_elements = ane_overlap[ane_overlap != 0]
                counter = Counter(non_zero_elements)
                most_common_value, _ = counter.most_common(1)[0]
                new_label[label_one] = int(most_common_value)
            else:
                dilation_one = label_one.copy()
                while np.sum(dilation_one * vessel) == 0:
                    dilation_one = self.dilation3d(dilation_one)
                ane_overlap = dilation_one * vessel
                non_zero_elements = ane_overlap[ane_overlap != 0]
                counter = Counter(non_zero_elements)
                most_common_value, _ = counter.most_common(1)[0]
                new_label[label_one] = int(most_common_value)
                print(f'失敗慘了 {i + 1} 是沒有在血管mask上, 但最後分配給: {int(most_common_value)}')

        new_label_back = self.data_translate_back(new_label, label_nii).astype(int)
        new_label_nii = self.nii_img_replace(label_nii, new_label_back)
        nib.save(new_label_nii, os.path.join(self.path_nii, 'Pred_Location16labels.nii.gz'))

    def calculate_aneurysm_long_axis_make_pred(self):
        excel_file = os.path.join(self.path_excel, 'Aneurysm_Pred_long_axis_list.xlsx')
        text = [['PatientID', 'StudyDate', 'AccessionNumber', 'Aneurysm_Number', 'Value', 'Size',
                 'Prob_max', 'Prob_mean', 'Location4labels', 'Location6labels']]
        df = pd.DataFrame(text[1:], columns=text[0])
        df.to_excel(excel_file, index=False)

        PID = self.patient_id[:8]
        Sdate = self.patient_id[9:17]
        AN = '_'.join(self.patient_id.split('/')[-1].split('_')[3:4])

        img_nii = nib.load(os.path.join(self.path_nii, 'MRA_BRAIN.nii.gz'))
        img_array = np.array(img_nii.dataobj)
        img_array = self.data_translate(img_array, img_nii)

        # 嘗試讀取DICOM資訊取得pixel_size及spacing
        try:
            if os.path.isdir(os.path.join(self.path_dcm, 'TOF_MRA')):
                img_list = sorted(os.listdir(os.path.join(self.path_dcm, 'TOF_MRA')))
                dcm = pydicom.dcmread(os.path.join(self.path_dcm, 'TOF_MRA', img_list[1]))
            elif os.path.isdir(os.path.join(self.path_dcm, 'MRA_BRAIN')):
                img_list = sorted(os.listdir(os.path.join(self.path_dcm, 'MRA_BRAIN')))
                dcm = pydicom.dcmread(os.path.join(self.path_dcm, 'MRA_BRAIN', img_list[1]))
            else:
                print('GG啦!!!! 影像資料夾不存在！')

            pixel_size = dcm[0x28, 0x0030].value
            spacing = float(dcm[0x18, 0x0088].value)
            ml_size = pixel_size[0] * pixel_size[1] * spacing / 1000
        except Exception:
            header_true = img_nii.header.copy()
            pixdim = header_true['pixdim']
            ml_size = (pixdim[1] * pixdim[2] * pixdim[3]) / 1000

        nii = nib.load(os.path.join(self.path_nii, 'Pred.nii.gz'))
        mask_array = np.array(nii.dataobj)
        mask_array = self.data_translate(mask_array, nii)

        prob_nii = nib.load(os.path.join(self.path_nii, 'Prob.nii.gz'))
        prob = np.array(prob_nii.dataobj)
        prob = self.data_translate(prob, prob_nii)

        ane_index = 0
        for i in range(int(np.max(mask_array))):
            y_cl, x_cl, z_cl = np.where(mask_array == (i + 1))
            label_one = mask_array == (i + 1)
            prob_one = prob[label_one]

            if len(y_cl) > 0:
                cluster_size = len(y_cl)
                cluster_ml = cluster_size * ml_size
                aneurysm_ml = float(f'{cluster_ml:.2f}')

                cluster_matrix = np.zeros(mask_array.shape, dtype=np.uint8)
                cluster_matrix[y_cl, x_cl, z_cl] = 254

                cluster_z_list = np.sum(cluster_matrix, axis=(0, 1))
                z_cluster_l = np.where(cluster_z_list > 0)[0]
                z_c_l_num = len(z_cluster_l)

                z_cluster_m = np.zeros(z_c_l_num)
                z_cluster_int = np.zeros(z_c_l_num)
                site = []

                for j in range(z_c_l_num):
                    mask_fig = cluster_matrix[:, :, z_cluster_l[j]]
                    mask_fig = np.expand_dims(mask_fig, axis=-1).astype(np.uint8)
                    ret, thresh = cv2.threshold(mask_fig, 127, 255, cv2.THRESH_BINARY)
                    contours, hierarchy = cv2.findContours(thresh, 1, 2)
                    cnt = contours[0]
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    site.append([x, y, radius])
                    diameter_mm = 2 * radius * pixel_size[0]
                    z_cluster_m[j] = diameter_mm
                    diameter = 2 * radius
                    z_cluster_int[j] = diameter

                max_maxis = np.max(z_cluster_m)
                long_axis = round(max_maxis, 1)

                # 將資料填入Excel
                df.at[ane_index, 'PatientID'] = PID
                df.at[ane_index, 'StudyDate'] = Sdate
                df.at[ane_index, 'AccessionNumber'] = AN
                df.at[ane_index, 'Aneurysm_Number'] = int(np.max(mask_array))
                df.at[ane_index, 'Value'] = int(i + 1)
                df.at[ane_index, 'Prob_max'] = round(np.max(prob_one), 2)
                df.at[ane_index, 'Prob_mean'] = round(np.mean(prob_one), 2)
                df.at[ane_index, 'Size'] = long_axis - 0.5
                df.at[ane_index, 'Location4labels'] = ''
                df.at[ane_index, 'Location6labels'] = ''

                ane_index += 1

        df.to_excel(excel_file, index=False)

    def make_table_row_patient_pred(self):
        excel_name2 = 'Aneurysm_Pred_long_axis_list.xlsx'
        excel_file = os.path.join(self.path_excel, excel_name2)

        PID1s = [self.patient_id[:8]]
        SDate1s = [self.patient_id[9:17]]
        AN1s = ['_'.join(self.patient_id.split('/')[-1].split('_')[3:4])]

        dtypes1 = {'ID': str, 'PatientID': str, 'StudyDate': str, 'AccessionNumber': str, 'SubjectLabel': str,
                   'ExperimentLabel': str, 'SeriesInstanceUID': str, 'url': str}

        df2 = pd.read_excel(excel_file, dtype=dtypes1).fillna('')

        PID2s = list(df2['PatientID'])
        AN2s = list(df2['AccessionNumber'])
        ID2s = [x + '_' + y for x, y in zip(PID2s, AN2s)]

        AneurysmNums = list(df2['Aneurysm_Number'])
        Values = list(df2['Value'])
        Sizes = list(df2['Size'])
        Prob_maxs = list(df2['Prob_max'])
        Prob_means = list(df2['Prob_mean'])
        Location4labels = list(df2['Location4labels'])

        max_num = max(AneurysmNums) if AneurysmNums else 0

        mm_list = ['PatientID', 'StudyDate', 'AccessionNumber', 'Aneurysm_Number']
        for i in range(max_num):
            mm_list.extend([
                f'{i + 1}_size',
                f'{i + 1}_Prob_max',
                f'{i + 1}_Prob_mean',
                f'{i + 1}_Confirm',
                f'{i + 1}_type',
                f'{i + 1}_Location',
                f'{i + 1}_SubLocation',
                f'{i + 1}_Location4labels',
                f'{i + 1}_Location6labels',
                f'{i + 1}_Comment',
            ])

        mm_dict = {key: [] for key in mm_list}

        for idx, (PID1, SDate1, AN1) in enumerate(zip(PID1s, SDate1s, AN1s)):
            print(f"[{idx}] {PID1}_{AN1} Start...")
            ane_list = [jdx for jdx, y in enumerate(ID2s) if y == f'{PID1}_{AN1}']
            mm_dict['PatientID'].append(PID1)
            mm_dict['StudyDate'].append(SDate1)
            mm_dict['AccessionNumber'].append(AN1)

            if ane_list:
                mm_dict['Aneurysm_Number'].append(int(max_num))
                for j, ane_inx in enumerate(ane_list):
                    mm_dict[f'{j + 1}_size'].append(round(Sizes[ane_inx], 3))
                    mm_dict[f'{j + 1}_Prob_max'].append(Prob_maxs[ane_inx])
                    mm_dict[f'{j + 1}_Prob_mean'].append(Prob_means[ane_inx])
                    mm_dict[f'{j + 1}_Confirm'].append('')
                    mm_dict[f'{j + 1}_type'].append('')
                    mm_dict[f'{j + 1}_Location'].append('')
                    mm_dict[f'{j + 1}_SubLocation'].append('')
                    mm_dict[f'{j + 1}_Location4labels'].append('')
                    mm_dict[f'{j + 1}_Location6labels'].append('')
                    mm_dict[f'{j + 1}_Comment'].append('')

                for k in range(AneurysmNums[ane_list[0]], max_num):
                    mm_dict[f'{k + 1}_size'].append('')
                    mm_dict[f'{k + 1}_Prob_max'].append('')
                    mm_dict[f'{k + 1}_Prob_mean'].append('')
                    mm_dict[f'{k + 1}_Confirm'].append('')
                    mm_dict[f'{k + 1}_type'].append('')
                    mm_dict[f'{k + 1}_Location'].append('')
                    mm_dict[f'{k + 1}_SubLocation'].append('')
                    mm_dict[f'{k + 1}_Location4labels'].append('')
                    mm_dict[f'{k + 1}_Location6labels'].append('')
                    mm_dict[f'{k + 1}_Comment'].append('')
            else:
                mm_dict['Aneurysm_Number'].append(int(max_num))
                for j in range(max_num):
                    mm_dict[f'{j + 1}_size'].append('')
                    mm_dict[f'{j + 1}_Prob_max'].append('')
                    mm_dict[f'{j + 1}_Prob_mean'].append('')
                    mm_dict[f'{j + 1}_Confirm'].append('')
                    mm_dict[f'{j + 1}_type'].append('')
                    mm_dict[f'{j + 1}_Location'].append('')
                    mm_dict[f'{j + 1}_SubLocation'].append('')
                    mm_dict[f'{j + 1}_Location4labels'].append('')
                    mm_dict[f'{j + 1}_Location6labels'].append('')
                    mm_dict[f'{j + 1}_Comment'].append('')

        mm_dict_pd = pd.DataFrame(mm_dict)
        out_excel = os.path.join(self.path_excel, 'Aneurysm_Pred_list.xlsx')
        mm_dict_pd.to_excel(out_excel, index=False)

    def make_table_add_location(self):
        dtypes = {'PatientID': str, 'StudyDate': str, 'AccessionNumber': str, 'PatientName': str,
                  'Doctor': str, 'birth_date': str, 'study_time': str}

        excel_file = os.path.join(self.path_excel, 'Aneurysm_Pred_list.xlsx')
        df = pd.read_excel(excel_file, dtype=dtypes).fillna('')

        PIDs = list(df['PatientID'])
        Sdates = list(df['StudyDate'])
        ANs = list(df['AccessionNumber'])
        PAs = [f"{x}_{y}_MR_{z}" for x, y, z in zip(PIDs, Sdates, ANs)]

        for idx, (PID, Sdate, AN, PA) in enumerate(zip(PIDs, Sdates, ANs, PAs)):
            print(f"[{idx}] {PA} Start...")

            label_nii = nib.load(os.path.join(self.path_nii, 'Pred.nii.gz'))
            label = np.array(label_nii.dataobj)
            label = self.data_translate(label, label_nii).astype('int16')

            labels16_nii = nib.load(os.path.join(self.path_nii, 'Pred_Location16labels.nii.gz'))
            labels16 = np.array(labels16_nii.dataobj)
            labels16 = self.data_translate(labels16, labels16_nii).astype('int16')

            Label_num = int(np.max(label))

            for k in range(Label_num):
                label_one = label == (k + 1)
                labels16_list = labels16[label_one]
                num_unique_values_16labels = len(np.unique(labels16_list))
                assert num_unique_values_16labels == 1, '挫屎!!! 同一顆怎麼那麼多location'

                label_val = int(np.unique(labels16_list)[0])
                loc, subloc = self.decode_location(label_val)
                df.at[idx, f'{k + 1}_Location'] = loc
                df.at[idx, f'{k + 1}_SubLocation'] = subloc

            df.to_excel(excel_file, index=False)

    @staticmethod
    def decode_location(val):
        if val in (1, 3):
            return 'ICA', ''
        elif val in (2, 4):
            return 'MCA', ''
        elif val in (5, 6):
            return 'ACA', '1'
        elif val == 7:
            return 'ACA', ''
        elif val in (8, 9):
            return 'ICA', 'p'
        elif val in (10, 11):
            return 'PCA', ''
        elif val in (12, 13):
            return 'BA', 's'
        elif val == 14:
            return 'BA', ''
        elif val in (15, 16):
            return 'VA', ''
        else:
            return '', ''

    def run_all(self):
        print("Start: make_aneurysm_vessel_location_16labels_pred()")
        self.make_aneurysm_vessel_location_16labels_pred()
        print("Done: make_aneurysm_vessel_location_16labels_pred()")

        print("Start: calculate_aneurysm_long_axis_make_pred()")
        self.calculate_aneurysm_long_axis_make_pred()
        print("Done: calculate_aneurysm_long_axis_make_pred()")

        print("Start: make_table_row_patient_pred()")
        self.make_table_row_patient_pred()
        print("Done: make_table_row_patient_pred()")

        print("Start: make_table_add_location()")
        self.make_table_add_location()
        print("Done: make_table_add_location()")

"""
if __name__ == "__main__":
    # 範例使用說明
    # 請設定好以下路徑與病人ID再執行
    path_dicom = '/path/to/dicom_folder'
    path_nifti = '/path/to/nifti_folder'
    path_excel = '/path/to/excel_folder'
    patient_id = '12345678_20230101_SOMEID'  # 請替換成實際ID格式

    pipeline = AneurysmPipeline(path_dicom, path_nifti, path_excel, patient_id)
    pipeline.run_all()
"""

def compute_orientation(init_axcodes, final_axcodes):
    """
    A thin wrapper around ``nib.orientations.ornt_transform``

    :param init_axcodes: Initial orientation codes
    :param final_axcodes: Target orientation codes
    :return: orientations array, start_ornt, end_ornt
    """
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)

    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)

    return ornt_transf, ornt_init, ornt_fin


def do_reorientation(data_array, init_axcodes, final_axcodes):
    """
    source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation
    Performs the reorientation (changing order of axes)

    :param data_array: 3D Array to reorient
    :param init_axcodes: Initial orientation
    :param final_axcodes: Target orientation
    :return data_reoriented: New data array in its reoriented form
    """
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array

    return nib.orientations.apply_orientation(data_array, ornt_transf)


def create_dicomseg_multi_file(path_code, path_img_dcm, path_nii, path_nii_resample, path_dcmseg, ID):
    example_file = os.path.join(path_code, 'example', 'SEG_20230210_160056_635_S3.dcm')
    #針對5類設計影像顏色
    model_name = 'Aneurysm_AI'

    labels_vessel = {
            1: {
                "SegmentLabel": "Vessel",
                "color":"red"
                }
        }

    #讀範例文件，其實就是抓出某些tag格式不想自己設定
    dcm_example = pydicom.dcmread(example_file)

    def get_dicom_seg_template(model, label_dict):
        # key is model output label
        # values is meta data
        unique_labels = list(label_dict.keys())
        
        segment_attributes = []
        for i, idx in enumerate(unique_labels):
            name = label_dict[idx]["SegmentLabel"]
            
            rgb_rate = mcolors.to_rgb(label_dict[idx]["color"])
            rgb = [int(y*255) for y in rgb_rate]
            

            segment_attribute = {
                    "labelID": int(idx),
                    "SegmentLabel": name,
                    "SegmentAlgorithmType": "MANUAL",
                    "SegmentAlgorithmName": "MONAILABEL",
                    "SegmentedPropertyCategoryCodeSequence": {
                        "CodeValue": "M-01000",
                        "CodingSchemeDesignator": "SRT",
                        "CodeMeaning": "Morphologically Altered Structure",
                    },
                    "SegmentedPropertyTypeCodeSequence": {
                        "CodeValue": "M-35300",
                        "CodingSchemeDesignator": "SRT",
                        "CodeMeaning": "Embolus",
                    },
                    "recommendedDisplayRGBValue": rgb,
                }
                
            segment_attributes.append(segment_attribute)

        template = {
            "ContentCreatorName": "Reader1",
            "ClinicalTrialSeriesID": "Session1",
            "ClinicalTrialTimePointID": "1",
            "SeriesDescription": model,
            "SeriesNumber": "300",
            "InstanceNumber": "1",
            "segmentAttributes": [segment_attributes],
            "ContentLabel": "SEGMENTATION",
            "ContentDescription": "MONAI Label - Image segmentation",
            "ClinicalTrialCoordinatingCenterName": "MONAI",
            "BodyPartExamined": "",
        }
        return template


    def make_dicomseg(mask, path_dcms, dcm_example, model_name, labels):
        #先做dicom-seg空格式
        template_json = get_dicom_seg_template(model_name, labels)
        template = pydicom_seg.template.from_dcmqi_metainfo(template_json)

        #可以使用 編寫多類分割MultiClassWriter，它有一些可用的自定義選項。如果分割數據非常稀疏，這些選項可以大大減少生成的文件大小。
        writer = pydicom_seg.MultiClassWriter(
            template=template,
            inplane_cropping=False,  # Crop image slices to the minimum bounding box on
                                    # x and y axes
            skip_empty_slices=True,  # Don't encode slices with only zeros
            skip_missing_segment=True,  # If a segment definition is missing in the
                                        # template, then raise an error instead of
                                        # skipping it.
        )
        
        #這邊把資料填入dicom-seg結構中
        y_l, x_l, z_l = mask.shape
        #先排序並生成一個原版的dicom-seg，沒任何AI標註True不適合用套件做dicom-seg
        if np.max(mask) > 0:
            #為了說明從分割數據創建 DICOM-SEG，典型的圖像分析工作流程如下所示。首先，圖像作為            
            
            reader = sitk.ImageSeriesReader()
            dcms = sorted(reader.GetGDCMSeriesFileNames(path_dcms))
                
            #Stack the 2D slices to form a 3D array representing the volume
            slices = [pydicom.dcmread(dcm) for dcm in dcms]
                
            slice_dcm = []
            #需要依照真實切片位置來更新順序
            for (slice, dcm_slice) in zip(slices, dcms):
                IOP = np.array(slice.ImageOrientationPatient)
                IPP = np.array(slice.ImagePositionPatient)
                normal = np.cross(IOP[0:3], IOP[3:])
                projection = np.dot(IPP, normal)
                slice_dcm += [{"d": projection, "dcm": dcm_slice}]
                
            #sort the slices，sort是由小到大，但頭頂是要由大到小，所以要反向
            #由於dicom-seg套件影像排序就是照reader.GetGDCMSeriesFileNames，所以不用再自己改頭到腳(原本腳到頭)
            slice_dcms = sorted(slice_dcm, key = lambda i: i['d'])
                
            new_dcms = [y['dcm'] for y in slice_dcms]
            new_dcms = new_dcms
            #print('new_dcms:', new_dcms)
                
            #接下來讀取第一張影像
            dcm_one = pydicom.dcmread(new_dcms[0], force=True) #dcm的切片
                
            #重建空影像給dicomSEG
            reader.SetFileNames(new_dcms)
            image = reader.Execute()
            #image_data = sitk.GetArrayFromImage(image)

            #將原本label的shape轉換成 z,y,x，把影像重新存入
            #mask = np.flip(mask, 0)
            #mask = np.flip(mask, -1) #製作dicom-seg時標註要腳到頭
            segmentation_data = mask.astype(np.uint8)
            #segmentation_data = mask.transpose(2, 0, 1).astype(np.uint8)
            #segmentation_data = np.swapaxes(segmentation_data,1,2)
            #segmentation_data = np.flip(segmentation_data, 1) #y軸
            #segmentation_data = np.flip(segmentation_data, 2)
            #segmentation_data = np.flip(segmentation_data, 0)
            #print('segmentation_data.shape:', segmentation_data.shape)

            #由於pydicom_seg期望 aSimpleITK.Image作為作者的輸入，因此需要將分割數據與有關圖像網格的所有相關信息一起封裝。
            #這對於編碼片段和引用相應的源 DICOM 文件非常重要。
            segmentation = sitk.GetImageFromArray(segmentation_data)
            segmentation.CopyInformation(image)
            

            #最後，可以執行 DICOM-SEG 的實際創建。因此，需要加載源 DICOM 文件pydicom，但出於優化目的，可以跳過像素數據
            source_images = [pydicom.dcmread(x, stop_before_pixels=True) for x in new_dcms]
            dcm_seg_ = writer.write(segmentation, source_images)

            #增加一些標註需要的資訊，作法，輸入範例資訊，再把實際case的值填入
            dcm_seg_[0x10,0x0010].value = dcm_one[0x10,0x0010].value #(0010,0010) Patient's Name <= (0010,0020) Patient ID
            dcm_seg_[0x20,0x0011].value = dcm_one[0x20,0x0011].value # (0020,0011) Series Number [3]
            #dcm_seg[0x62,0x0002].value = dcm_example[0x62,0x0002].value #Segment Sequence Attribute
            dcm_seg_[0x5200,0x9229].value = dcm_example[0x5200,0x9229].value
            dcm_seg_[0x5200,0x9229][0][0x20,0x9116][0][0x20,0x0037].value = dcm_one[0x20,0x0037].value #(0020,0037) Image Orientation (Patient)
            dcm_seg_[0x5200,0x9229][0][0x28,0x9110][0][0x18,0x0050].value = dcm_one[0x18,0x0050].value #(0018,0050) Slice Thickness 
            dcm_seg_[0x5200,0x9229][0][0x28,0x9110][0][0x18,0x0088].value = dcm_one[0x18,0x0088].value #(0018,0088) Spacing Between Slices
            dcm_seg_[0x5200,0x9229][0][0x28,0x9110][0][0x28,0x0030].value = dcm_one[0x28,0x0030].value #(0028,0030) Pixel Spacing    
        
        return dcm_seg_


    IDs = [ID]

    Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw'] #三軸都要做dicom-seg

    for idx, ID in enumerate(IDs):
        print([idx], ID, ' Start....')
        
        for idx_s, series in enumerate(Series): 
            path_dcms = os.path.join(path_img_dcm, series)
            
            #讀取pred跟vessel
            if series == 'MRA_BRAIN':
                Pred_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz'))
                Pred = np.array(Pred_nii.dataobj) #讀出label的array矩陣      #256*256*22
                Vessel_nii = nib.load(os.path.join(path_nii, 'Vessel.nii.gz'))
                #Vessel_nii = nib.as_closest_canonical(Vessel_nii) #轉成RAS軸
                Vessel = np.array(Vessel_nii.dataobj) #讀出label的array矩陣      #256*256*22
                #Pred = data_translate(Pred, Pred_nii)
                #Vessel = data_translate(Vessel, Vessel_nii)  
                pred_nii_obj_axcodes = tuple(nib.aff2axcodes(Pred_nii.affine))
                Pred = do_reorientation(Pred, pred_nii_obj_axcodes, ('S', 'P', 'L'))          
                
            else:
                Pred_nii = nib.load(os.path.join(path_nii_resample, series + '_pred.nii.gz'))
                Pred = np.array(Pred_nii.dataobj) #讀出label的array矩陣      #256*256*22
                Vessel_nii = nib.load(os.path.join(path_nii_resample, series + '_vessel.nii.gz'))
                Vessel = np.array(Vessel_nii.dataobj) #讀出label的array矩陣      #256*256*22
                #Pred = data_translate(Pred, Pred_nii)
                # Vessel = data_translate(Vessel, Vessel_nii)   
                pred_nii_obj_axcodes = tuple(nib.aff2axcodes(Pred_nii.affine))
                Pred = do_reorientation(Pred, pred_nii_obj_axcodes, ('S', 'P', 'L'))  
                
            #vessel的dicom seg可以直接做
            #dcm_seg = make_dicomseg(Vessel.astype('uint8'), path_dcms, dcm_example, model_name, labels_vessel)
            #dcm_seg.save_as(os.path.join(path_dcmseg, ID + '_' + series + '_' + labels_vessel[1]['SegmentLabel'] + '.dcm')) # 
        
            
            #以下做動脈瘤，要一顆一顆做，血管一定有標註所以有dicom-seg，動脈瘤沒標註就沒有dicom-seg
            for i in range(int(np.max(Pred))):
                new_Pred = np.zeros((Pred.shape))
                new_Pred[Pred==i+1] = 1
                
                labels_ane = {}
                labels_ane[1] = {'SegmentLabel': 'A' + str(i+1), 'color': 'red'}
                
                #底下做成dicom-seg
                dcm_seg = make_dicomseg(new_Pred.astype('uint8'), path_dcms, dcm_example, model_name, labels_ane)
                dcm_seg.save_as(os.path.join(path_dcmseg, ID + '_' + series + '_' + labels_ane[1]['SegmentLabel'] + '.dcm')) # 

    return print('Dicom-SEG ok!!!')


def decompress_dicom_with_gdcm(path_base):
    """
    使用系統的 gdcmconv (apt版本) 解壓縮 DICOM 檔案
    功能等同於 Decompress_JEPG.sh
    
    參數:
        path_base: 包含子資料夾的基礎路徑，會遍歷所有子資料夾並解壓縮其中的 .dcm 檔案
    """
    import subprocess
    import glob
    
    # 獲取所有子資料夾（排序後倒序）
    folders = sorted([f for f in os.listdir(path_base) 
                     if os.path.isdir(os.path.join(path_base, f))], reverse=True)
    
    for folder in folders:
        folder_path = os.path.join(path_base, folder)
        
        # 找到該資料夾中所有的 .dcm 檔案
        dcm_files = glob.glob(os.path.join(folder_path, "*.dcm"))
        
        # 對每個 .dcm 檔案執行 gdcmconv --raw
        for dcm_file in dcm_files:
            # 使用系統的 gdcmconv (apt版本)
            cmd = ['gdcmconv', '--raw', dcm_file, dcm_file]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {dcm_file}: {e}")
                if e.stderr:
                    print(f"stderr: {e.stderr}")
            except FileNotFoundError:
                print("Error: gdcmconv not found. Please install with: sudo apt-get install libgdcm-tools")
                raise

def compress_dicom_into_jpeglossless(path_dcm_in, path_dcm_out):

    Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw']
  
    for jdx, series in enumerate(Series):
        if not os.path.isdir(os.path.join(path_dcm_out, series)):
            os.mkdir(os.path.join(path_dcm_out, series))
        
        #讀取dicom影像，然後轉成jpeglossless
        dcms = sorted(os.listdir(os.path.join(path_dcm_in, series)))
        for dcm in dcms:
            dcm_slice = pydicom.dcmread(os.path.join(path_dcm_in, series, dcm)) #ADC的切片
            #(0002,0003) Media Stored SOP Instance UID  [1.2.840.113619.2.260.6945.3202356.28278.1273105263.493]
            SOPUID = dcm_slice[0x08, 0x0018].value #(0008,0018) SOP Instance UID [1.2.840.113619.2.260.6945.3202356.28278.1273105263.493]
            dcm_slice.compress(JPEGLSLossless)
            #dcm_slice[0x02, 0x0003].value = SOPUID
            dcm_slice[0x08, 0x0018].value = SOPUID

            # Need to set this flag to indicate the Pixel Data is compressed
            #dcm_slice['PixelData'].is_undefined_length = True

            # The Transfer Syntax UID needs to match the type of (JPEG) compression
            # For example, if you were to compress using JPEG Lossless, Hierarchical, First-Order Prediction (1.2.840.10008.1.2.4.70)
            #from pydicom.uid import JPEGLosslessSV1 
            #dcm_slice.file_meta.TransferSyntaxUID = JPEGLosslessSV1

            dcm_slice.save_as(os.path.join(path_dcm_out, series, dcm))
    
    #最後複製dicom-seg
    shutil.copytree(os.path.join(path_dcm_in, 'Dicom-Seg'), os.path.join(path_dcm_out, 'Dicom-Seg'))

#建立json，如果有讀取excel，就會有xnat_link
def case_json(json_file_path, study_dict, sorted_dict, mask_dict):
    json_dict = OrderedDict()
    json_dict["study"] = study_dict
    json_dict["sorted"] = sorted_dict
    json_dict["mask"] = mask_dict
   
    with open(json_file_path, 'w', encoding='utf8') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False, default=int) #讓json能中文顯示

#建立json，如果有讀取excel，就會有xnat_link
def sort_json(json_file_path, data):
    json_dict = OrderedDict()
    json_dict["data"] = data

    with open(json_file_path, 'w', encoding='utf8') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '), ensure_ascii=False, default=int) #讓json能中文顯示


#黃色被拿去當成人工標註，所以先不要用 => 目前顏色順序就固定，先設定7個
#因為醫師還沒看過，預設select全部都打True
def make_pred_json(excel_file, path_dcm, path_nii, path_reslice, path_dicomseg, path_json, IDs, Series, group_id=49):

    colorbar = {
        "A1": "red",
        "A2": "green",
        "A3": "cyan",
        "A4": "fuchsia",
        "A5": "orange",
        "A6": "lawngreen",
        "A7": "blue",
        "A8": "purple",
        "A9": "salmon",
        "A10": "medium spring green",
        "A11": "dodger blue",
        "A12": "chocolate"
    }

    #顏色這邊改成只有紅跟黃
    colorbar1 = ["red", "green", "cyan", "fuchsia", "orange", "lawngreen", "blue", "purple", "salmon", "medium spring green", "dodger blue", "chocolate"] #只記錄顏色，因為要重新編號

    #讀取excel以獲得臨床資訊
    dtypes = {'PatientID': str, 'StudyDate': str, 'AccessionNumber': str, 'PatientName': str, 'Doctor': str} #年齡在這裡就弄成數字吧
    #excel_name = 'Aneurysm_Pred_list_Val_merged_report.xlsx'
    df = pd.read_excel(excel_file, dtype=dtypes).fillna('') #將指定欄位改成特定格式
    PIDs = list(df['PatientID'])
    Sdates = list(df['StudyDate'])
    ANs = list(df['AccessionNumber'])
    AneurysmNums = list(df['Aneurysm_Number'])
    
    PAs = [x + '_' + y + '_MR_' + z for x,y, z in zip(PIDs, Sdates, ANs)]
    #Xlinks = list(df['XNAT_Link'])

    #tags = ['TOF_MRA', 'MIP_Pitch', 'MIP_Yaw']
    #file_endings = ['_TOF_MRA.nii.gz', '_MIP_Pitch.nii.gz', '_MIP_Yaw.nii.gz']

    for idx, ID in enumerate(IDs):
        print([idx], ID, ' mask json start...')

        study_index = PAs.index(ID)
        try:
            AneurysmNumber = int(df['Aneurysm_Number'][study_index])
        except:
            AneurysmNumber = 0

        if os.path.isfile(os.path.join(path_nii, 'Pred.nii.gz')): 
            #先一致讀出標註檔跟影像檔
            sort_data = []
            for jdx, series in enumerate(Series):
                if series == 'MRA_BRAIN':
                    #沒有label，只要讀入Pred就好
                    pred_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz')) #這邊預設不同影像也是同標註
                    pred = np.array(pred_nii.dataobj) #讀出image的array矩陣
                    pred = data_translate(pred, pred_nii).astype('int16')
                    resolution_y, resolution_x, resolution_z = pred.shape #這裡獲得影像大小
                            
                    #標註的顏色已經固定A1,A2...是啥了
                    #整理dicom的排序，tag目前只有1，不用迴圈顯示
                    reader = sitk.ImageSeriesReader()
                    files = sorted(reader.GetGDCMSeriesFileNames(os.path.join(path_dcm, series)))
                    #Stack the 2D slices to form a 3D array representing the volume
                    dcms = [pydicom.dcmread(os.path.join(path_dcm, series, y)) for y in files]
                        
                    sort_slices = []
                    sort_instances = []
                    #需要依照真實切片位置來更新順序
                    for (dcm, file) in zip(dcms, files):
                        SOPuid = dcm[0x0008, 0x0018].value #SOPInstanceUID <- 以此為檔案命名
                        IOP = np.array(dcm.ImageOrientationPatient)
                        IPP = np.array(dcm.ImagePositionPatient)
                        normal = np.cross(IOP[0:3], IOP[3:])
                        projection = np.dot(IPP, normal)
                        sort_slices += [{"d": projection, "file": file}]
                        sort_instances += [{"sop_instance_uid": SOPuid, "projection": projection}]
 
                    #sort the slices，sort是由小到大，但頭頂是要由大到小，所以要反向
                    #由於dicom-seg套件影像排序就是照reader.GetGDCMSeriesFileNames，所以不用再自己改頭到腳(原本腳到頭)
                    slice_dcms = sorted(sort_slices, key = lambda i: i['d'])
                
                    new_dcms = [y['file'] for y in slice_dcms]
                    new_dcms = new_dcms
                    #print('new_dcms:', new_dcms)
                
                    #接下來讀取第一張影像
                    dcm_one = pydicom.dcmread(new_dcms[0], force=True) #dcm的切片
                
                    #重建空影像給dicomSEG
                    reader.SetFileNames(new_dcms)
                    image = reader.Execute()
                    #image_data = sitk.GetArrayFromImage(image)

                    #將原本label的shape轉換成 z,y,x，把影像重新存入
                    #mask = np.flip(mask, 0)
                    #mask = np.flip(mask, -1) #製作dicom-seg時標註要腳到頭
                    segmentation_data = mask.astype(np.uint8)
                    #segmentation_data = mask.transpose(2, 0, 1).astype(np.uint8)
                    #segmentation_data = np.swapaxes(segmentation_data,1,2)
                    #segmentation_data = np.flip(segmentation_data, 1) #y軸
                    #segmentation_data = np.flip(segmentation_data, 2)
                    #segmentation_data = np.flip(segmentation_data, 0)
                    #print('segmentation_data.shape:', segmentation_data.shape)

                    #由於pydicom_seg期望 aSimpleITK.Image作為作者的輸入，因此需要將分割數據與有關圖像網格的所有相關信息一起封裝。
                    #這對於編碼片段和引用相應的源 DICOM 文件非常重要。
                    segmentation = sitk.GetImageFromArray(segmentation_data)
                    segmentation.CopyInformation(image)
            

                    #最後，可以執行 DICOM-SEG 的實際創建。因此，需要加載源 DICOM 文件pydicom，但出於優化目的，可以跳過像素數據
                    source_images = [pydicom.dcmread(x, stop_before_pixels=True) for x in new_dcms]
                    dcm_seg_ = writer.write(segmentation, source_images)

                    #增加一些標註需要的資訊，作法，輸入範例資訊，再把實際case的值填入
                    dcm_seg_[0x10,0x0010].value = dcm_one[0x10,0x0010].value #(0010,0010) Patient's Name <= (0010,0020) Patient ID
                    dcm_seg_[0x20,0x0011].value = dcm_one[0x20,0x0011].value # (0020,0011) Series Number [3]
                    #dcm_seg[0x62,0x0002].value = dcm_example[0x62,0x0002].value #Segment Sequence Attribute
                    dcm_seg_[0x5200,0x9229].value = dcm_example[0x5200,0x9229].value
                    dcm_seg_[0x5200,0x9229][0][0x20,0x9116][0][0x20,0x0037].value = dcm_one[0x20,0x0037].value #(0020,0037) Image Orientation (Patient)
                    dcm_seg_[0x5200,0x9229][0][0x28,0x9110][0][0x18,0x0050].value = dcm_one[0x18,0x0050].value #(0018,0050) Slice Thickness 
                    dcm_seg_[0x5200,0x9229][0][0x28,0x9110][0][0x18,0x0088].value = dcm_one[0x18,0x0088].value #(0018,0088) Spacing Between Slices
                    dcm_seg_[0x5200,0x9229][0][0x28,0x9110][0][0x28,0x0030].value = dcm_one[0x28,0x0030].value #(0028,0030) Pixel Spacing    
        
        return dcm_seg_


def orthanc_zip_upload(path_dicom, path_zip, Series):          
    upload_data(path_dicom, path_zip, Series)       

def build_upload_study(json_data):
    data = {}
    data['study[0][study_name]'] = json_data['StudyDescription']
    data['study[0][age]'] = str(json_data['Age'])
    data['study[0][patient_id]'] = json_data['PatientID']
    data['study[0][study_date]'] = json_data['StudyDate']
    data['study[0][gender]'] = json_data['Gender']
    data['study[0][patient_name]'] = json_data['PatientName']
    #data['study[0][xnat_link]'] = json_data['XNAT_Link']
    data['study[0][aneurysm_lession]'] = str(json_data['Aneurysm_Number'])
    data['study[0][resolution_x]'] = json_data['resolution_x']
    data['study[0][resolution_y]'] = json_data['resolution_y']
    data['study[0][study_instance_uid]'] = json_data['StudyInstanceUID']
    data['study[0][group_id]'] = json_data['group_id']
    return data

def upload_json(path_json):
    url = 'http://localhost:84/api/cornerstone/demoStudy'
    Series = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw']

    files = sorted(os.listdir(path_json))

    IDs = [y.split('_')[0:1][0] + '_' + y.split('_')[1:2][0] + '_' + y.split('_')[2:3][0] + '_' + y.split('_')[3:4][0] for y in files]
    IDs = sorted(list(set(IDs)))

    for idx, ID in enumerate(IDs):
        print([idx], ID, 'Upload JSON Start...')
        for jdx, series in enumerate(Series):
            #先讀取json，因為是以study為主，所以只要讀取MRA_BRAIN就好
            if series == 'MRA_BRAIN':
                json_file = os.path.join(path_json, ID + '_' + series + '.json')
                #讀取json以獲得資料集參數
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    
                study_data = build_upload_study(json_data)
                # 发出 POST 请求
                response = requests.post(url, data=study_data, headers={'Accept': 'application/json'})

                # 打印响应内容
                print('Status Code:', response.status_code)
                print('Response Text:', response.text)
                
                #上傳某 series 的 mask json 到 DB，有mask才上傳
                if len(str(json_data['Aneurysm_Number'])) > 0:
                    if json_data['Aneurysm_Number'] > 0:
                        bash_line = 'cd /var/www/shh-pacs && php artisan import:mask ' + os.path.join(path_json, ID + '_' + series + '.json')
                        uploadJSON_line_line  = os.popen(bash_line)
                        print('uploadJSON_line_line:', uploadJSON_line_line.read())
                        time.sleep(2)
            
            else:
                #這邊就不用上傳study list了
                json_file = os.path.join(path_json, ID + '_' + series + '.json')
                #讀取json以獲得資料集參數
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)            
                
                #上傳某 series 的 mask json 到 DB，有mask才上傳
                if len(str(json_data['Aneurysm_Number'])) > 0:
                    if json_data['Aneurysm_Number'] > 0:
                        bash_line = 'cd /var/www/shh-pacs && php artisan import:mask ' + os.path.join(path_json, ID + '_' + series + '.json')
                        uploadJSON_line_line  = os.popen(bash_line)
                        print('uploadJSON_line_line:', uploadJSON_line_line.read())
                        time.sleep(2)
        
        #sort可以放到最後上傳mask後做，上傳完study list，接下來 快取某 series 的 imageIds 順序到 DB
        bash_line = 'cd /var/www/shh-pacs && php artisan import:sorted-image-ids ' + os.path.join(path_json, ID + '_sort.json')
        sortDcm_line  = os.popen(bash_line)
        print('sortDcm_line:', sortDcm_line.read())
        time.sleep(2)

import warnings
import httpx
import orjson
import os
import argparse

warnings.filterwarnings("ignore")  # 忽略 SSL 警告

def upload_json_aiteam(json_file):                               
    UPLOAD_DATA_JSON_URL = 'http://localhost:84/api/ai_team/create_or_update'
    client = httpx.Client(timeout=10.0, verify=False)  # 加入 timeout/不驗證 SSL

    def post_json(path):
        with open(path, 'rb') as f:
            data = orjson.loads(f.read())
        try:
            study_date = ""
            if isinstance(data.get("study"), dict):
                study_date = data["study"].get("study_date", "")
            logging.info("Uploading platform JSON: file=%s study_date=%r", path, study_date)
        except Exception:
            pass
        response = client.post(UPLOAD_DATA_JSON_URL, json=data)
        logging.info("Upload JSON done: file=%s status=%s", os.path.basename(path), response.status_code)
        try:
            logging.info("Upload JSON response: %s", response.json())
        except Exception:
            logging.info("Upload JSON response (raw): %s", response.text)

    with client:
        if isinstance(json_file, str):
            post_json(json_file)
        elif isinstance(json_file, list):
            for path in json_file:
                post_json(path)
        else:
            print("❌ Invalid input: must be a str or list of str")


#建立字典找出日期的配對
def organize_data(list_a, list_b):
    data_dict = defaultdict(list)
    
    # 先將 list_a 轉換為方便查找的結構
    id_to_dates = defaultdict(list)
    id_to_full = defaultdict(dict)
    
    for item in list_a:
        parts = item.split("_")
        ID, Date = parts[0], parts[1]
        id_to_dates[ID].append(Date)
        id_to_full[(ID, Date)] = item
    
    # 確保日期是排序的
    for ID in id_to_dates:
        id_to_dates[ID].sort()
    
    # 遍歷 list_b，填充 dict
    for item in list_b:
        parts = item.split("_")
        ID, Date = parts[0], parts[1]
        
        # 找到 before_Date（list_a 中同 ID 且日期小於當前日期的最大值）
        before_Date = None
        for d in reversed(id_to_dates[ID]):  # 逆序查找
            if d < Date:
                before_Date = d
                break
        
        # 設定 before_IDSdate
        before_IDSdate = id_to_full.get((ID, before_Date), "") if before_Date else ""
        
        # 加入結果
        data_dict[ID].append({
            "ID": ID,
            "Date": Date,
            "IDSdate": item,
            "before_Date": before_Date if before_Date else "",
            "before_IDSdate": before_IDSdate
        })
    
    return dict(data_dict)

#將SynthSEG的結果轉到original上
def resampleSynthSEG2original(path_nii, series, SynthSEGmodel):
    #先讀取原始影像並抓出SpacingBetweenSlices跟PixelSpacing
    img_nii = nib.load(os.path.join(path_nii, series + '.nii.gz'))
    img_array = np.array(img_nii.dataobj)
    img_array = data_translate(img_array, img_nii)
    img_1mm_nii = nib.load(os.path.join(path_nii, series + '_resample.nii.gz'))
    img_1mm_array = np.array(img_1mm_nii.dataobj)    
    SynthSEG_1mm_nii = nib.load(os.path.join(path_nii, series + '_' + SynthSEGmodel + '.nii.gz')) #230*230*140

    y_i, x_i, z_i = img_array.shape
    y_i1, x_i1, z_i1 = img_1mm_array.shape
    
    header_img = img_nii.header.copy() #抓出nii header 去算體積 
    pixdim_img = header_img['pixdim']  #可以借此從nii的header抓出voxel size
    header_img_1mm = img_1mm_nii.header.copy() #抓出nii header 去算體積 
    pixdim_img_1mm = header_img_1mm['pixdim']  #可以借此從nii的header抓出voxel size    
    
    #先把影像從230*230*140轉成256*256*140
    img_1mm_256_nii = nibabel.processing.conform(img_1mm_nii, ((y_i, x_i, z_i1)),(pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3]),order = 3) #影像用
    img_1mm_256 = np.array(img_1mm_256_nii.dataobj)
    img_1mm_256 = data_translate(img_1mm_256, img_1mm_256_nii)
    img_1mm_256_back = data_translate_back(img_1mm_256, img_1mm_256_nii)
    img_1mm_256_nii2 = nii_img_replace(img_1mm_256_nii, img_1mm_256_back)
    nib.save(img_1mm_256_nii2, os.path.join(path_nii, series + '_resample_256.nii.gz'))
    #再將SynthSEG從230*230*140轉成256*256*140
    SynthSEG_1mm_256_nii = nibabel.processing.conform(SynthSEG_1mm_nii, ((y_i, x_i, z_i1)),(pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3]),order = 0) #影像用
    SynthSEG_1mm_256 = np.array(SynthSEG_1mm_256_nii.dataobj)
    SynthSEG_1mm_256 = data_translate(SynthSEG_1mm_256, SynthSEG_1mm_256_nii)
    SynthSEG_1mm_256_back = data_translate_back(SynthSEG_1mm_256, SynthSEG_1mm_256_nii)
    SynthSEG_1mm_256_nii2 = nii_img_replace(SynthSEG_1mm_256_nii, SynthSEG_1mm_256_back)
    nib.save(SynthSEG_1mm_256_nii2, os.path.join(path_nii, series + '_' + SynthSEGmodel + '_256.nii.gz'))   

    #以下將1mm重新組回otiginal    
    new_array = np.zeros(img_array.shape)
    img_repeat = np.expand_dims(img_array, -1).repeat(z_i1, axis=-1)
    img_1mm_repeat = np.expand_dims(img_1mm_256, 2).repeat(z_i, axis=2)
    diff = np.sum(np.abs(img_1mm_repeat - img_repeat), axis = (0,1))
    argmin = np.argmin(diff, axis=1)
    new_array = SynthSEG_1mm_256[:,:,argmin]
    #最後重新組回nifti
    new_array_save = data_translate_back(new_array, img_nii)
    new_SynthSeg_nii = nii_img_replace(img_nii, new_array_save)
    nib.save(new_SynthSeg_nii, os.path.join(path_nii, 'NEW_' + series + '_' + SynthSEGmodel + '.nii.gz'))   
    return new_array