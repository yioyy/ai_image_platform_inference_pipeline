#!/usr/bin/env python3
"""SynthSeg -> native-grid resampling, lifted out of util_aneurysm.

Verbatim copies of four functions from
/home/david/pipeline/chuan/code/util_aneurysm.py. They are duplicated rather
than imported because that module imports tensorflow at module scope, and this
container has no other use for TensorFlow -- carrying it would add hundreds of
megabytes to an image built deliberately lean (it also has no FSL).

resampleSynthSEG2original reads three files out of one directory:
    {path_nii}/{series}.nii.gz                     the native-grid image
    {path_nii}/{series}_resample.nii.gz            the 1mm image
    {path_nii}/{series}_{SynthSEGmodel}.nii.gz     SynthSeg output at 1mm
and writes {path_nii}/NEW_{series}_{SynthSEGmodel}.nii.gz.

The label volume goes through nibabel.processing.conform with order=0 while the
image uses order=3. That asymmetry is the point: the labels are categorical, and
interpolating them invents classes that do not exist. Do not "tidy" the two
calls into one.

Kept byte-identical to the source so the two stay comparable; aneurysm and WMH
still call the original.
"""
import os

import nibabel as nib
import nibabel.processing
import numpy as np


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


def nii_img_replace(data, new_img):
    affine = data.affine
    header = data.header.copy()
    new_nii = nib.nifti1.Nifti1Image(new_img, affine, header=header)
    return new_nii


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
