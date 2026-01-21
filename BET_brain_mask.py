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
# from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
import argparse
import glob, re, os, sys, time, itertools
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal
from scipy import ndimage as ndi
from skimage.morphology import ball, binary_dilation, binary_closing, remove_small_objects, binary_erosion, binary_opening, skeletonize
from skimage.filters import threshold_multiotsu, gaussian, threshold_otsu, frangi
from skimage.measure import label, regionprops, regionprops_table
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt as dist_field
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from brainextractor import BrainExtractor

def resize_volume(arr, spacing=None, target_spacing=None, target_size=None, dtype='float32'):
    ori_size = arr.shape
    order = 1 if 'float' in dtype else 0
    if (spacing is not None)and(target_spacing is not None):
        # from spacing to target_spacing
        scale = np.array(spacing) / np.array(target_spacing)
        out_vol = ndi.zoom(arr, zoom=scale, order=order, prefilter=True, grid_mode=False)
    elif target_size is not None:
        # to target_size
        scale = np.array(target_size) / np.array(arr.shape)
        out_vol = ndi.zoom(arr, zoom=scale, output=np.zeros(target_size, dtype=dtype), order=order, prefilter=False, grid_mode=False)
    return out_vol.astype(dtype)

#@title BET
def get_BET_brain_mask(image_arr, spacing, bet_iter=1000):
    # resize to isotropic
    target_spacing = np.array([spacing[0], spacing[0], spacing[0]], dtype=np.float32)
    print("target_spacing =", target_spacing)
    image_iso = resize_volume(image_arr, spacing, target_spacing=target_spacing, dtype='float32')

    # nibabel obj
    affine = np.array([ [spacing[0], 0, 0, 0],
                        [0, spacing[0], 0, 0],
                        [0, 0, spacing[0], 0],
                        [0, 0, 0, 1]], dtype=np.float32)
    nib_image = nib.Nifti1Image(image_iso[::-1, ::-1, :], affine=affine)  # LPS to RAS orientation

    # BET process
    bet = BrainExtractor(img=nib_image)
    bet.run(iterations=bet_iter)
    brain_mask_iso = bet.compute_mask()[::-1, ::-1, :] > 0  # RAS to LPS orientation

    # back to original spacing
    brain_mask = resize_volume(brain_mask_iso, target_size=image_arr.shape, dtype='bool')
    return brain_mask



#其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='要做bet的影像')
    parser.add_argument('--PixelSpacing', type=float, help='PixelSpacing')
    parser.add_argument('--SpacingBetweenSlices', type=float, help='SpacingBetweenSlices')
    parser.add_argument('--BET_iter', type=int, help='BET_iter')
    args = parser.parse_args()

    #
    file_path = str(args.file)
    spacing = [float(args.PixelSpacing), float(args.PixelSpacing), float(args.SpacingBetweenSlices)]
    bet_iter = int(args.BET_iter)

    #以下做predict，為了避免gpu out of memory，還是以.sh來執行好惹
    image_arr = np.load(file_path)
    try:
        brain_bottom = get_BET_brain_mask(image_arr, spacing, bet_iter=bet_iter)
        np.save(file_path[:-4] + '_mask.npy', brain_bottom)
    except:
        print(f"[error] BET failed with bet_iter={bet_iter}")

