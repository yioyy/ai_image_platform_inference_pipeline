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

def _maybe_set_cuda_visible_from_argv() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    gpu_n = None
    if "--gpu_n" in sys.argv:
        idx = sys.argv.index("--gpu_n")
        if idx + 1 < len(sys.argv):
            gpu_n = sys.argv[idx + 1]
    else:
        for arg in sys.argv:
            if arg.startswith("--gpu_n="):
                gpu_n = arg.split("=", 1)[1]
                break
    if gpu_n is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_n).strip()

_maybe_set_cuda_visible_from_argv()
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
from nnResUNet_long_BigBatch_cosine_AneDilate_classifier_test.gpu_nnUNet import predict_from_raw_data
import subprocess
from util_aneurysm import resampleSynthSEG2original

def custom_normalize_1(volume, mask=None, new_min=0., new_max=0.5, min_percentile=0.5, max_percentile=99.5, use_positive_only=True):
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

#     # trim values outside range
#     new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        new_volume = new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:  # avoid dividing by zero
        new_volume = np.zeros_like(new_volume)

#     # clip normalized values [0:1]
#     new_volume = np.clip(new_volume, 0, 1)
    return new_volume

def get_struc(diameter, spacing):  # diameter in mm
    structure_size = np.round(diameter / np.array(spacing)).astype('int32')
    structure_size = np.maximum([1,1,1], structure_size)
    structure = resize_volume(ball(diameter*5), target_size=structure_size, dtype='bool')
    center_point = np.array(structure.shape) // 2
    structure[center_point] = True
    return structure

# resampling
def resize_volume(arr, spacing=None, target_spacing=None, target_size=None, order=3, dtype='float32'):
    if ("int8" in dtype)or(dtype == "bool"):
        order = 0

    if (spacing is not None)and(target_spacing is not None):
        # from spacing to target_spacing
        scale = np.array(spacing) / np.array(target_spacing)
        out_vol = ndi.zoom(arr, zoom=scale, order=order,
                           mode='grid-mirror', prefilter=True, grid_mode=False)
    elif target_size is not None:
        # to target_size
        scale = np.array(target_size) / np.array(arr.shape)
        out_vol = ndi.zoom(arr, zoom=scale, output=np.zeros(target_size, dtype=dtype), order=order,
                           mode='grid-mirror', prefilter=True, grid_mode=False)

    if 'int' in dtype:  # clip values
        dtype_info = np.iinfo(dtype)
        out_vol = np.clip(out_vol, dtype_info.min, dtype_info.max)
    return out_vol.astype(dtype)

#@title Load image
def load_volume(path_volume, im_only=False, squeeze=True, dtype=None, LPS_coor=True):
    """
    Load volume file.
    :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
    If npz format, 1) the variable name is assumed to be 'vol_data',
    2) the volume is associated with an identity affine matrix and blank header.
    :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
    :param squeeze: (optional) whether to squeeze the volume when loading.
    :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
    The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
    :return: the volume, with corresponding affine matrix and header if im_only is False.
    """
    path_volume = str(path_volume)
    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        x = nib.as_closest_canonical(x)  # to RAS space
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
        spacing = list(x.header.get_zooms())
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
        spacing = [1., 1., 1.]
    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)
    if LPS_coor:
        volume = volume[::-1,::-1,:]

    if im_only:
        return volume
    else:
        return volume, spacing, aff

def get_brain_radius_area(brain_mask, spacing, radius_mm=40):
    # 確保 brain_mask 是 bool，避免後續位元運算把遮罩變成 int(0/1) 造成 fancy indexing 爆記憶體
    brain_mask = brain_mask.astype(bool)
    # disk 是在 XY 平面做半徑限制；避免 r 過大造成負 index/越界
    spacing_xy = float(spacing[0]) if spacing is not None else 1.0
    r = int(max(1, round(radius_mm / max(spacing_xy, 1e-6))))
    area = np.zeros(brain_mask.shape[:2], dtype=bool)
    disk_mask = disk(r) > 0

    cx, cy = (np.array(brain_mask.shape[:2]) // 2).tolist()
    x0 = cx - disk_mask.shape[0] // 2
    y0 = cy - disk_mask.shape[1] // 2
    x1 = x0 + disk_mask.shape[0]
    y1 = y0 + disk_mask.shape[1]

    # clip 到影像範圍，並同步裁切 disk_mask
    xx0 = max(0, x0)
    yy0 = max(0, y0)
    xx1 = min(area.shape[0], x1)
    yy1 = min(area.shape[1], y1)
    if (xx1 > xx0) and (yy1 > yy0):
        dx0 = xx0 - x0
        dy0 = yy0 - y0
        dx1 = dx0 + (xx1 - xx0)
        dy1 = dy0 + (yy1 - yy0)
        area[xx0:xx1, yy0:yy1] = disk_mask[dx0:dx1, dy0:dy1]

    area = np.repeat(area[..., np.newaxis], brain_mask.shape[2], axis=-1)
    return brain_mask & area

def get_vessel_seed(candidates, mask=None, spacing=(1,1,1)):
    # Remove the region below brain
    candidates_mask = candidates > 0
    if mask is not None:  # keep brain region
        candidates_mask = candidates > 0
        z_idx = np.where(np.any(mask > 0, axis=(0,1)))[0]
        if z_idx.size > 0:
            brain_bottom_idx = int(z_idx[0])
            candidates_mask[:, :, :brain_bottom_idx] = False
        else:
            brain_bottom_idx = None

    # make major seeds
    major_seed_mask = binary_erosion(candidates_mask.copy(), get_struc(2.5, spacing))  # 2.5, 3.0
    area_mask = get_brain_radius_area(mask, spacing, radius_mm=45)
    major_seed_mask = (major_seed_mask & area_mask).astype(bool)

    # make center seeds
    center_seed_mask = candidates_mask.copy()
    if mask is not None:  # keep brain region
        if brain_bottom_idx is not None:
            upper1of3_brain_height = (candidates_mask.shape[2] - brain_bottom_idx) // 3
            center_seed_mask[:, :, :brain_bottom_idx + upper1of3_brain_height] = False
            center_seed_mask[:, :, -upper1of3_brain_height:] = False
    center_seed_mask = binary_erosion(center_seed_mask, get_struc(1.5, spacing))
    area_mask = get_brain_radius_area(mask, spacing, radius_mm=20)
    center_seed_mask = (center_seed_mask & area_mask).astype(bool)

    # make upper seeds
    upper_seed_mask = candidates_mask.copy()
    if mask is not None:  # keep brain region
        if brain_bottom_idx is not None:
            upper1of3_brain_height = (candidates_mask.shape[2] - brain_bottom_idx) // 3
            upper_seed_mask[:, :, :brain_bottom_idx + upper1of3_brain_height*2] = False
    upper_seed_mask = binary_erosion(upper_seed_mask, get_struc(1.5, spacing))
    area_mask = get_brain_radius_area(mask, spacing, radius_mm=40)
    upper_seed_mask = (upper_seed_mask & area_mask).astype(bool)
    return (major_seed_mask | center_seed_mask | upper_seed_mask).astype(bool)

#@title combine two vessel results
def combine_vessel_brain(vessel_threshold, seed_mask=None, brain_mask=None):
    vessel_threshold = (vessel_threshold > 0)
    if brain_mask is not None:  # keep brain region
        z_idx = np.where(np.any(brain_mask > 0, axis=(0,1)))[0]
        if z_idx.size > 0:
            brain_bottom_idx = int(z_idx[0])
            vessel_threshold[:, :, :brain_bottom_idx] = False  # cut off below brain area

    vessel_mask = vessel_threshold > 0
    label_map = label(vessel_mask)
    if seed_mask is None:
        vessel_mask = label_map > 0
    else:  # select vessels via seeds
        seed_mask = seed_mask.astype(bool)
        if seed_mask.shape != label_map.shape:
            raise ValueError(f"seed_mask shape {seed_mask.shape} != label_map shape {label_map.shape}. "
                             f"請確認 pred/brain 都在同一個 data_translate 座標系下再做種子挑選。")
        lbs = np.unique(label_map[seed_mask])
        lbs = [lb for lb in lbs if lb != 0]  # drop zero
        vessel_mask = np.isin(label_map, lbs)

    vessel_mask = remove_small_objects(vessel_mask, min_size=500)
    return vessel_mask

#@title predict vessel_16labels (w/ isotropic 0.8mm vol)
def predict_vessel_16labels(vessel_mask, model3, spacing, post_proc=True, min_size=3, verbose=False):

    def find_end_points(skeleton_mask):
        ends_map = np.zeros(skeleton_mask.shape, dtype='uint8')
        centers = np.stack(np.where(skeleton_mask), axis=-1)
        for center in centers:
            patch = skeleton_mask[center[0]-1:center[0]+2, center[1]-1:center[1]+2, center[2]-1:center[2]+2]
            if np.sum(patch) == 3:  # line
                ends_map[center[0], center[1], center[2]] = 1
            elif np.sum(patch) == 2:  # end point
                ends_map[center[0], center[1], center[2]] = 2
            elif np.sum(patch) > 3:  # branch point
                ends_map[center[0], center[1], center[2]] = 6
        return ends_map

    def crop_resample(arr, spacing, target_shape=(160, 160, 160)):
        arr = resize_volume(arr, spacing=spacing, target_spacing=(0.8, 0.8, 0.8), dtype='uint8')
        zd = np.ceil(max(0, target_shape[2] - arr.shape[2])).astype('int32')
        arr = np.pad(arr, ((0,0),(0,0),(zd,0)), 'constant')
        # xy center crop
        x0, y0 = np.floor(np.array(arr.shape[:2])/2).astype('int32') - np.array(target_shape[:2])//2
        arr = arr[x0:x0+target_shape[0], y0:y0+target_shape[1], -target_shape[2]:]
        return arr

    def back_to_origin(arr, spacing, target_shape=vessel_mask.shape):
        arr = resize_volume(arr, spacing=(0.8, 0.8, 0.8), target_spacing=spacing, dtype='uint8')
        p0 = int((target_shape[0] - arr.shape[0]) // 2)
        p1 = int((target_shape[1] - arr.shape[1]) // 2)
        p2 = max(0, int(target_shape[2] - arr.shape[2]))
        arr = np.pad(arr, ((p0,p0+1),(p1,p1+1),(p2,0)), 'constant')
        arr = arr[:target_shape[0], :target_shape[1], -target_shape[2]:]
        return arr

    # run
    if verbose: print(">>> predict_vessel_16labels ", end='')
    vessel_mask_crop = crop_resample(vessel_mask, spacing).astype("float32")
    vessel_16labels_crop = model3.serve(vessel_mask_crop[np.newaxis,:,:,:,np.newaxis]).numpy()[0]
    vessel_16labels_crop[:, :, 0] = -1  # ignore background 0
    vessel_16labels_crop = np.argmax(vessel_16labels_crop, axis=-1).astype('uint8')
    vessel_16labels = back_to_origin(vessel_16labels_crop, spacing)

    # post-process
    if post_proc:  # post-proc2
        # reduce noise of preds
        preds = np.zeros_like(vessel_16labels)
        for lb in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 18]:  # ignore 17(branch)
            mask = ndi.binary_erosion(vessel_16labels==lb)
            mask = ndi.binary_fill_holes(mask)
            preds[mask&(preds==0)] = lb
            if verbose: print(".", end='')

        # filter with skeleton fragments
        ends_map = find_end_points(skeletonize(vessel_mask) > 0)
        fragments_markers = label(ends_map==1, connectivity=3)
        # watershed fragments
        distance = ndi.distance_transform_edt(vessel_mask, sampling=spacing)
        fragments = watershed(-distance, fragments_markers, mask=vessel_mask)

        markers = np.zeros_like(fragments_markers, dtype="uint8")
        for idx in range(1, fragments_markers.max()):
            masked_pred = preds[fragments==idx]
            unique, counts = np.unique(masked_pred[masked_pred > 0], return_counts=True)
            filter_idx = np.where(counts > min_size)[0]
            unique, counts = unique[filter_idx], counts[filter_idx]
            if len(unique) > 0:
                markers[fragments_markers==idx] = unique[np.argmax(counts)]
            if verbose:
                if idx % 50 == 0: print(".", end='')
        # watershed vessels
        vessel_16labels = watershed(-distance, markers, mask=vessel_mask)
        vessel_16labels[vessel_16labels==18] = 0  # exclude
    else:
        markers = vessel_16labels
        # watershed vessels
        distance = ndi.distance_transform_edt(vessel_mask, sampling=spacing)
        vessel_16labels = watershed(-distance, markers, mask=vessel_mask)

    # add one pixel annotation
    for lb in range(1, 16+1):
        vessel_16labels[lb, lb, -lb] = lb
        vessel_16labels[-lb-1, -lb-1, -lb-1] = lb

    if verbose: print(">>>", vessel_16labels.shape)
    return vessel_16labels

#新增加修正vessel_16labels
def modify_vessel_16labels(vessel_16labels, spacing):
    new_vessel_mask = vessel_16labels > 0
    # regenrate vessel_seed
    new_vessel_seed = binary_erosion(new_vessel_mask, get_struc(2.5, spacing))  # 2.5, 3.0

    # select connected vessels via seeds
    label_map = label(new_vessel_mask)
    lbs = np.unique(label_map[new_vessel_seed])
    lbs = [lb for lb in lbs if lb != 0]  # drop zero
    new_vessel_mask = np.isin(label_map, lbs)
    new_vessel_mask = remove_small_objects(new_vessel_mask, min_size=500)

    return new_vessel_mask.astype('uint8') * vessel_16labels, new_vessel_mask
 
#@title save nifti，這邊是用我的
def save_nii_brain(path_process, image_arr, brain_arr, out_dir='test_label_out', verbose=False):
    ## load original image
    #建立分別放正規化影像與vessel的資料夾
    if not os.path.isdir(os.path.join(out_dir, 'Image')):
        os.mkdir(os.path.join(out_dir, 'Image'))
    if not os.path.isdir(os.path.join(out_dir, 'Ones')):
        os.mkdir(os.path.join(out_dir, 'Ones'))

    img = nib.load(os.path.join(path_process, 'MRA_BRAIN.nii.gz'))
    img = nib.as_closest_canonical(img)  # to RAS space
    aff = img.affine
    hdr = img.header
    spacing = tuple(img.header.get_zooms())
    shape = tuple(img.header.get_data_shape())
    original_size = tuple(img.header.get_data_shape())
    #nib.save(img, f"{out_dir}/image.nii.gz")
    #print("[save] --->", f"{out_dir}/image.nii.gz")

     #影像直接複製
    if not os.path.isfile(os.path.join(out_dir, 'Image', 'DeepAneurysm_00001_0000.nii.gz')):
        shutil.copy(os.path.join(path_process, 'MRA_BRAIN.nii.gz'), os.path.join(out_dir, 'Image', 'DeepAneurysm_00001_0000.nii.gz'))

    # build NifTi1 brain
    brain_arr = brain_arr[::-1, ::-1, :]  # LPS to RAS orientation
    x = nib.nifti1.Nifti1Image(brain_arr, affine=aff, header=hdr)
    nib.save(x, os.path.join(out_dir, 'Ones', 'DeepAneurysm_00001_0000.nii.gz'))
    print("[save] --->", os.path.join(out_dir, 'Ones', 'DeepAneurysm_00001_0000.nii.gz'))
    return 

#@title save nifti，這邊是用我的
def save_nii_vessel(path_process, image_arr, pred_vessel, vessel_16labels, out_dir='test_label_out', verbose=False):
    ## load original image
    #建立分別放正規化影像與vessel的資料夾
    if not os.path.isdir(os.path.join(out_dir, 'Normalized_Image')):
        os.mkdir(os.path.join(out_dir, 'Normalized_Image'))
    if not os.path.isdir(os.path.join(out_dir, 'Vessel')):
        os.mkdir(os.path.join(out_dir, 'Vessel'))

    img = nib.load(os.path.join(path_process, 'MRA_BRAIN.nii.gz'))
    aff = img.affine
    hdr = img.header
    spacing = tuple(img.header.get_zooms())
    shape = tuple(img.header.get_data_shape())
    original_size = tuple(img.header.get_data_shape())

    img = nib.as_closest_canonical(img)  # to RAS space
    #nib.save(img, f"{out_dir}/image.nii.gz")
    #print("[save] --->", f"{out_dir}/image.nii.gz")

    # Make sure the dimensions are the same as the original image
    if image_arr.shape != original_size:
        image_arr = resize_volume(image_arr, target_size=original_size, dtype='uint8')  
    if pred_vessel.shape != original_size:
        pred_vessel = resize_volume(pred_vessel, target_size=original_size, dtype='uint8')  
    if vessel_16labels.shape != original_size:
        vessel_16labels = resize_volume(vessel_16labels, target_size=original_size, dtype='uint8')

    #做正規化
    image_arr = custom_normalize_1(image_arr)  

    # build NifTi1 image
    image_arr = image_arr[::-1, ::-1, :]  # LPS to RAS orientation
    x = nib.nifti1.Nifti1Image(image_arr, affine=aff, header=hdr)
    nib.save(x, os.path.join(out_dir, 'Normalized_Image', 'DeepAneurysm_00001_0000.nii.gz'))
    print("[save] --->", os.path.join(out_dir, 'Normalized_Image', 'DeepAneurysm_00001_0000.nii.gz'))    

    # pred_vessel
    # build NifTi1 image
    pred_vessel = pred_vessel[::-1, ::-1, :].astype('uint8')  # LPS to RAS orientation
    x = nib.nifti1.Nifti1Image(pred_vessel, affine=aff, header=hdr)
    # nib.save(x, os.path.join(out_dir, 'Vessel', 'DeepAneurysm_00001_0000.nii.gz'))
    # print("[save] --->", os.path.join(out_dir, 'Vessel', 'DeepAneurysm_00001_0000.nii.gz'))
    nib.save(x, f"{out_dir}/Vessel.nii.gz")
    print("[save] --->", f"{out_dir}/Vessel.nii.gz")

    # pred_vessel_16labels
    vessel_16labels = vessel_16labels[::-1, ::-1, :].astype('uint8')  # LPS to RAS orientation
    x = nib.nifti1.Nifti1Image(vessel_16labels, affine=aff, header=hdr)
    nib.save(x, f"{out_dir}/Vessel_16.nii.gz")
    print("[save] --->", f"{out_dir}/Vessel_16.nii.gz")
    return 

#篩選掉threshold不到的動脈瘤
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


#"主程式"
#model_predict_aneurysm(path_code, path_process, case_name, path_log, gpu_n)
def model_predict_aneurysm(path_code, path_process, path_brain_model, path_vessel_model, path_aneurysm_model, case_name, path_log, gpu_n):
    path_model = os.path.join(path_code, 'model_weights')

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
        logging.info('!!! ' + case_name + ' gpu_aneurysm call.')

        #把一些json要記錄的資訊弄出空集合或欲填入的值，這樣才能留空資訊
        ret = "gpu_aneurysm.py " #使用的程式是哪一支python api
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
            if not gpus:
                raise RuntimeError("No GPU devices found.")
            # 如果已設定 CUDA_VISIBLE_DEVICES，gpus 只會看到那一顆，需用 index 0
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                visible_idx = 0
            else:
                visible_idx = gpu_n
            if visible_idx >= len(gpus):
                visible_idx = 0
            tf.config.experimental.set_visible_devices(devices=gpus[visible_idx], device_type='GPU')
            #print(gpus, cpus)
            tf.config.experimental.set_memory_growth(gpus[visible_idx], True)

            #%%
            #底下正式開始predict任務
            verbose = True
            MRA_BRAIN_file = os.path.join(path_process, 'MRA_BRAIN.nii.gz')

            # 1 load image
            image_arr, spacing, _ = load_volume(MRA_BRAIN_file, dtype='int16')

            #@title load vessel 16labels model
            #model_root = '/content/gdrive/Shareddrives/2024_雙和_Aneurysm/model/dataV2-Vessel_16label-ResUnet'  # google drive
            model_root = os.path.join(path_model, 'dataV2-Vessel_16label-ResUnet')
            # model_root = '../model/dataV2-Vessel_16label-ResUnet'  # server
            model_name = "MP-Vessel_16label+branch_PostProc2AIAA_Aug2of4_160_b4-ResUnet_8f_L5_BN_mish_19ch_ema_cw" #我有改短名稱
            model3 = tf.saved_model.load(f"{model_root}/{model_name}/best_saved_model")
            model3.trainable = False
            model3.jit_compile = True

            # 2 Get brain mask，改用nnU-Net的model
            #先複製檔案過去跟製作全是1的同尺寸mask
            ones_img = np.ones_like(image_arr, dtype=np.int16)
            save_nii_brain(path_process, image_arr, ones_img, out_dir=path_process)

            #先建立資料夾
            path_nnunet = os.path.join(path_process, 'nnUNet')
            path_img = os.path.join(path_process, 'Image')
            path_ones = os.path.join(path_process, 'Ones')
            path_brain = os.path.join(path_process, 'Brain')
            path_vessel = os.path.join(path_process, 'Vessel')
            if not os.path.isdir(path_brain):
                os.mkdir(path_brain)
            if not os.path.isdir(path_vessel):
                os.mkdir(path_vessel)

            start = time.time()
            predict_from_raw_data(path_img,
                                  path_ones,
                                  path_brain,
                                  path_brain_model,
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
                                  batch_size=16
                                 )            

            #對預測的brain mask進行後處理
            prob_nii = nib.load(os.path.join(path_brain, 'DeepAneurysm_00001.nii.gz'))
            prob = np.array(prob_nii.dataobj) #讀出label的array矩陣      #256*256*22   
            prob = data_translate(prob, prob_nii)
        
            #改讀取mifti的pixel_size資訊
            header_true = prob_nii.header.copy() #抓出nii header 去算體積 
            pixdim = header_true['pixdim']  #可以借此從nii的header抓出voxel size        
            new_pred_label = prob > 0.1
        
            #最後存出新mask，存出nifti
            new_pred_label = data_translate_back(new_pred_label, prob_nii).astype(int)
            new_pred_label_nii = nii_img_replace(prob_nii, new_pred_label)
            nib.save(new_pred_label_nii, os.path.join(path_brain, 'DeepAneurysm_00001_0000.nii.gz'))
            shutil.copy(os.path.join(path_brain, 'DeepAneurysm_00001_0000.nii.gz'), os.path.join(path_nnunet, 'brain_mask.nii.gz')) #取threshold跟cluster放到後面做
            os.remove(os.path.join(path_brain, 'DeepAneurysm_00001.nii.gz')) #刪除原本的nii
            print(f"[Done Get brain mask... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done Get brain mask... ] spend {time.time() - start:.0f} sec")
                
            # 3 Get vessel mask
            start = time.time()
            predict_from_raw_data(path_img,
                                  path_brain,
                                  path_vessel,
                                  path_vessel_model,
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
                                  batch_size=32
                                 )                
            print(f"[Done Vessel AI Inference... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done Vessel AI Inference... ] spend {time.time() - start:.0f} sec")

            start = time.time()
            #對預測的vessel mask進行後處理
            prob_nii = nib.load(os.path.join(path_vessel, 'DeepAneurysm_00001.nii.gz'))
            prob = np.array(prob_nii.dataobj) #讀出label的array矩陣      #256*256*22   
            prob = data_translate(prob, prob_nii)
        
            # NIfTI voxel spacing：建議用 header.get_zooms()，避免誤用 pixdim[0] (qfac)
            spacing = prob_nii.header.get_zooms()[:3]
            #print('spacing(x,y,z):', spacing)
            # 注意：這裡先保持在 data_translate 後的座標系做種子/連通塊處理
            new_pred_label_t = prob > 0.1
            #底下讀取brain mask然後這段是在用「腦遮罩 + 中心半徑限制 + z 方向高度限制 + 形態學腐蝕」，
            # 從血管候選區挑出更可靠的主血管核心種子，避免把腦外/腦底雜訊當成血管，並讓後續的「挑主要血管連通塊」更穩定。
            brain_nii = nib.load(os.path.join(path_nnunet, 'brain_mask.nii.gz'))
            brain = np.array(brain_nii.dataobj)
            brain = data_translate(brain, brain_nii)

            # 3.1 Post-process vessel mask
            vessel_seed = get_vessel_seed(new_pred_label_t, mask=brain, spacing=spacing)
            vessel_mask = combine_vessel_brain(new_pred_label_t, seed_mask=vessel_seed, brain_mask=brain)

            #最後存出新mask，存出nifti
            new_vessel_mask = data_translate_back(vessel_mask, prob_nii).astype(int)
            new_vessel_mask_nii = nii_img_replace(prob_nii, new_vessel_mask)

            nib.save(new_vessel_mask_nii, os.path.join(path_vessel, 'DeepAneurysm_00001_0000.nii.gz'))
            shutil.copy(os.path.join(path_vessel, 'DeepAneurysm_00001_0000.nii.gz'), os.path.join(path_nnunet, 'Vessel.nii.gz')) #取threshold跟cluster放到後面做
            os.remove(os.path.join(path_vessel, 'DeepAneurysm_00001.nii.gz')) #刪除原本的nii
            print(f"[Done Get vessel post-process... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done Get vessel post-process... ] spend {time.time() - start:.0f} sec")

            # 4 get vessel 16labels
            start = time.time()
            vessel_file = os.path.join(path_nnunet, 'Vessel.nii.gz')
            vessel_mask, spacing, _ = load_volume(vessel_file, dtype='int16')
            vessel_16labels = predict_vessel_16labels(vessel_mask, model3, spacing, verbose=verbose)
            vessel_16labels, vessel_mask = modify_vessel_16labels(vessel_16labels, spacing)  # 20250716 add
            save_nii_vessel(path_process, image_arr, vessel_mask, vessel_16labels, out_dir=path_process)
            print(f"[Done Get vessel 16labels... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done Get vessel 16labels... ] spend {time.time() - start:.0f} sec")

            # 5 Pred aneurysm，從這一步開始，底下置換成nnU-Net的model，先把正規化的image跟vessel mask存出，準備放入nnU-Net中
            # 5.1 先存出正規化的影像跟血管
            start = time.time()
            path_normimg = os.path.join(path_process, 'Normalized_Image')
            predict_from_raw_data(path_normimg,
                                  path_vessel,
                                  path_process,
                                  path_aneurysm_model,
                                  (13,),
                                  0.25,
                                  use_gaussian=True,
                                  use_mirroring=False,
                                  perform_everything_on_gpu=True,
                                  verbose=True,
                                  save_probabilities=False,
                                  overwrite=False,
                                  checkpoint_name='checkpoint_best.pth',
                                  plans_json_name='nnUNetPlans_5L-b900.json',  # 可以根據需要修改json檔名
                                  has_classifier_output=False,  # 如果模型有classifier輸出則設為True
                                  num_processes_preprocessing=2,
                                  num_processes_segmentation_export=3,
                                  desired_gpu_index = 0,
                                  batch_size=112
                                 )

            shutil.copy(os.path.join(path_process, 'DeepAneurysm_00001.nii.gz'), os.path.join(path_nnunet, 'Prob.nii.gz')) #取threshold跟cluster放到後面做
            # shutil.copy(os.path.join(path_process, 'DeepAneurysm_00001.nii.gz'), os.path.join(path_nnunetlow, 'Prob.nii.gz')) #取threshold跟cluster放到後面做

            #用threshold修改輸出
            prob_nii = nib.load(os.path.join(path_nnunet, 'Prob.nii.gz'))
            prob = np.array(prob_nii.dataobj) #讀出label的array矩陣      #256*256*22   
            prob = data_translate(prob, prob_nii)
        
            #改讀取mifti的pixel_size資訊
            header_true = prob_nii.header.copy() #抓出nii header 去算體積 
            pixdim = header_true['pixdim']  #可以借此從nii的header抓出voxel size
        
            spacing_nn = [pixdim[1], pixdim[2]]
        
            pred_prob_map, df_pred, new_pred_label = filter_aneurysm(prob, spacing_nn, conf_th=0.1, min_diameter=2, top_k=4, obj_th=0.73) #0.73是高threshold的模型
            #pred_prob_map, df_pred, new_pred_label = filter_aneurysm(prob, spacing_nn, conf_th=0.1, min_diameter=2, top_k=4, obj_th=0.46) #0.40是低threshold的模型

        
            #最後存出新mask，存出nifti
            new_pred_label = data_translate_back(new_pred_label, prob_nii).astype(int)
            new_pred_label_nii = nii_img_replace(prob_nii, new_pred_label)
            nib.save(new_pred_label_nii, os.path.join(path_nnunet, 'Pred.nii.gz'))  
            print(f"[Done Aneurysm AI Inference... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done Aneurysm AI Inference... ] spend {time.time() - start:.0f} sec")

            #以json做輸出
            time.sleep(1)
            logging.info('!!! ' + str(case_name) +  ' gpu_aneurysm finish.')

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
    parser.add_argument('--path_brain_model', type=str, help='brain model路徑')
    parser.add_argument('--path_vessel_model', type=str, help='vessel model路徑')
    parser.add_argument('--path_aneurysm_model', type=str, help='aneurysm model路徑')
    parser.add_argument('--case', type=str, help='目前執行的case的ID')
    parser.add_argument('--path_log', type=str, help='log資料夾')
    parser.add_argument('--gpu_n', type=int, help='第幾顆gpu')
    args = parser.parse_args()

    path_code = str(args.path_code)
    path_process = str(args.path_process)
    path_brain_model = str(args.path_brain_model)
    path_vessel_model = str(args.path_vessel_model)
    path_aneurysm_model = str(args.path_aneurysm_model)
    case_name = str(args.case)
    path_log = str(args.path_log)
    gpu_n = args.gpu_n

    model_predict_aneurysm(path_code, path_process, path_brain_model, path_vessel_model, path_aneurysm_model, case_name, path_log, gpu_n)
