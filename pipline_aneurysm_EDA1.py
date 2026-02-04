"""
最終答案：29% 真實模型 AUC 推論
使用最可靠的線性近似方法
"""

import os
import time
import numpy as np
import pydicom
import glob
import shutil
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import sys
import logging
import cv2
import pandas as pd
import subprocess

# #load excel
# path_excel = '/data/4TB1/pipeline/chuan/excel/'

# excel_name = 'ClinicalTest.xlsx'
# #excel_name = 'AI_Normal.xlsx'

# #先讀取excel，取得patientid跟accession_number當成資料夾檔名條件
# dtypes1 = {'ID': str, 'PatientID': str, 'StudyDate': str, 'AccessionNumber': str, 'SubjectLabel': str, 'ExperimentLabel': str, 
#           'SeriesInstanceUID': str, 'url': str}

# df = pd.read_excel(os.path.join(path_excel, excel_name), dtype=dtypes1).fillna('')
# IDs = list(df['ID'])

IDs = sorted(os.listdir('/data/chuan/Deep_Aneurysm/data/EDA_TFDA_20251001/EDA_TFDA_dicom_split/EDA_TFDA_dicom1'))

#這邊是用舊的for迴圈方式
for idx, ID in enumerate(IDs):
    if idx >= 324:
        print([idx], ID, ' Start...')
        Inputs = os.path.join('/data/chuan/Deep_Aneurysm/data/EDA_TFDA_20251001/EDA_TFDA_20251001_nifti_split/EDA_TFDA_20251001_nifti1', ID + '_MRA_BRAIN.nii.gz')
        DicomDir = os.path.join('/data/chuan/Deep_Aneurysm/data/EDA_TFDA_20251001/EDA_TFDA_dicom_split/EDA_TFDA_dicom1', ID, 'MRA_BRAIN')
        Output_folder = os.path.join('/data/chuan/Deep_Aneurysm/Result/EDA_TFDA_20251001/example_output/EDA1', ID)
        cmd = [
               "python", "pipeline_aneurysm_tensorflow.py",
               "--ID", ID,
               "--Inputs", Inputs,
               "--DicomDir", DicomDir,
               "--Output_folder", Output_folder,
               "--gpu_n", '1',
              ]

        subprocess.run(cmd)