# -*- coding: utf-8 -*-
"""
Stroke Image Preprocessing Pipeline

使用 SynthSEG 分割結果產生 Brain Mask：
1. 讀取 SynthSEG.nii.gz（腦組織分割結果）
2. 產生 brain mask（將所有 > 0 的標籤設為 1）
3. 將 mask 應用到 ADC、DWI0 和 DWI1000 影像

輸入檔案（固定名稱）：
    - ADC.nii.gz
    - DWI0.nii.gz
    - DWI1000.nii.gz
    - SynthSEG.nii.gz

輸出檔案：
    - BET_ADC.nii.gz
    - BET_DWI0.nii.gz
    - BET_DWI1000.nii.gz
    - BET_mask.nii.gz

@author: chuan
"""
import os
import logging
import warnings

import numpy as np
import nibabel as nib

from nii_transforms import nii_img_replace

warnings.filterwarnings("ignore")


def preprocess_stroke_images(path_process):
    """
    中風影像前處理 Pipeline：使用 SynthSEG 產生 Brain Mask
    
    此函數使用 SynthSEG 分割結果作為 brain mask，直接在 path_process 資料夾內處理
    不創建額外的子資料夾，使用固定的檔案命名規則
    
    處理流程：
    1. 讀取 SynthSEG.nii.gz（腦組織分割結果）
    2. 產生 brain mask（將所有 > 0 的標籤設為 1）
    3. 將 brain mask 應用到 ADC、DWI0 和 DWI1000
    4. 驗證所有輸出檔案
    
    參數:
        path_process: 處理資料夾路徑（包含輸入檔案，輸出也在同一資料夾）
    
    返回:
        bool: 成功返回 True，失敗返回 False
    """
    try:
        # 定義固定的檔案路徑
        dwi0_path = os.path.join(path_process, 'DWI0.nii.gz')
        adc_path = os.path.join(path_process, 'ADC.nii.gz')
        dwi1000_path = os.path.join(path_process, 'DWI1000.nii.gz')
        synthseg_path = os.path.join(path_process, 'SynthSEG.nii.gz')
        
        bet_dwi0_path = os.path.join(path_process, 'BET_DWI0.nii.gz')
        bet_adc_path = os.path.join(path_process, 'BET_ADC.nii.gz')
        bet_dwi1000_path = os.path.join(path_process, 'BET_DWI1000.nii.gz')
        bet_mask_path = os.path.join(path_process, 'BET_mask.nii.gz')
        
        # 驗證輸入檔案存在
        required_inputs = [dwi0_path, adc_path, dwi1000_path, synthseg_path]
        for file_path in required_inputs:
            if not os.path.isfile(file_path):
                logging.error(f'Input file not found: {file_path}')
                return False
        
        # === Step 1: 從 SynthSEG 產生 brain mask ===
        logging.info('Generating brain mask from SynthSEG...')
        synthseg_nii = nib.load(synthseg_path)
        synthseg_array = np.array(synthseg_nii.dataobj)
        
        # 產生二值化 mask（SynthSEG 中 > 0 的區域都是腦組織）
        mask_array = np.zeros_like(synthseg_array, dtype=np.uint8)
        mask_array[synthseg_array > 0] = 1
        
        # 儲存 mask
        mask_nii = nii_img_replace(synthseg_nii, mask_array)
        nib.save(mask_nii, bet_mask_path)
        logging.info(f'Brain mask saved: {bet_mask_path}')
        
        # === Step 2: 將 mask 應用到 DWI0 ===
        logging.info('Applying brain mask to DWI0...')
        dwi0_nii = nib.load(dwi0_path)
        dwi0_array = np.array(dwi0_nii.dataobj)
        bet_dwi0_array = dwi0_array * mask_array
        bet_dwi0_nii = nii_img_replace(dwi0_nii, bet_dwi0_array)
        nib.save(bet_dwi0_nii, bet_dwi0_path)
        
        # === Step 3: 將 mask 應用到 ADC ===
        logging.info('Applying brain mask to ADC...')
        adc_nii = nib.load(adc_path)
        adc_array = np.array(adc_nii.dataobj)
        bet_adc_array = adc_array * mask_array
        bet_adc_nii = nii_img_replace(adc_nii, bet_adc_array)
        nib.save(bet_adc_nii, bet_adc_path)
        
        # === Step 4: 將 mask 應用到 DWI1000 ===
        logging.info('Applying brain mask to DWI1000...')
        dwi1000_nii = nib.load(dwi1000_path)
        dwi1000_array = np.array(dwi1000_nii.dataobj)
        bet_dwi1000_array = dwi1000_array * mask_array
        bet_dwi1000_nii = nii_img_replace(dwi1000_nii, bet_dwi1000_array)
        nib.save(bet_dwi1000_nii, bet_dwi1000_path)
        
        # === Step 5: 驗證所有輸出檔案 ===
        required_outputs = [
            bet_dwi0_path,
            bet_adc_path,
            bet_dwi1000_path,
            bet_mask_path
        ]
        
        if all(os.path.isfile(f) for f in required_outputs):
            logging.info('Stroke preprocessing pipeline completed successfully')
            return True
        else:
            missing_files = [f for f in required_outputs if not os.path.isfile(f)]
            logging.error(f'Output files missing: {missing_files}')
            return False
            
    except Exception as e:
        logging.error('Stroke preprocessing pipeline error')
        logging.error(f"Exception: {str(e)}", exc_info=True)
        return False