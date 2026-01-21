# -*- coding: utf-8 -*-
"""
在docker內還是可以用
"""

import io
import os
import shutil
import argparse
import bz2
import gzip
import json
import os
import requests
import sys
import tarfile
import zipfile
import glob
import pydicom
import zipfile

from requests.auth import HTTPBasicAuth

def write_slice_zip(dcm, img_data, slice_index, zip_file,series_description):
    """
    write a single DICOM slice

    dcm – nii2dcm DICOM object
    img_data - [nX, nY, nSlice] image pixel data, such as from NIfTI file
    slice_index – slice index in nibabel img_data array (important: counts from 0, whereas DICOM instances count from 1)
    output_dir – output DICOM file save location
    """
    dicom_buffer = io.BytesIO()
    output_filename = r'IM_%04d.dcm' % (slice_index + 1)  # begin filename from 1, e.g. IM_0001.dcm

    # img_slice = img_data[:,:,slice_index]
    img_slice = img_data[slice_index]

    # Instance UID – unique to current slice
    dcm.ds.SOPInstanceUID = pydicom.uid.generate_uid(None)

    # write pixel data
    dcm.ds.PixelData = img_slice.tobytes()

    # write DICOM file
    dcm.ds.save_as(dicom_buffer, write_like_original=False)
    zip_file.writestr(f'{series_description}/{output_filename}', dicom_buffer.getvalue())

def zip_folder(folder, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                # 計算檔案在 zip 檔案中的相對路徑
                arcname = os.path.relpath(file_path, start=folder)
                zipf.write(file_path, arcname=arcname)

def zip_folders_individually(folders):
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"資料夾 {folder} 不存在")
            continue
        zip_filename = f"{folder}.zip"
        zip_folder(folder, zip_filename)
        print(f"{folder} 已壓縮成 {zip_filename}")

def upload_data(path_dcm, path_zip, Series):
    shutil.copytree(path_dcm, path_zip, dirs_exist_ok=True)
    
    # 要壓縮的資料夾清單
    folders_to_zip = [os.path.join(path_zip, y) for y in Series]  # 替換成你的資料夾路徑

    #針對每個series生成壓縮檔# 將每個資料夾壓縮成單獨的 ZIP 檔案
    zip_folders_individually(folders_to_zip)

    #接下來用curl一筆一筆上傳到orthanc上       
    # 設定 URL 和檔案路徑
    #url = 'http://10.103.21.40:8042/instances'
    url = 'http://10.103.1.193:8042/instances'

    files_zip = [y + '.zip' for y in folders_to_zip]

    for file_zip in files_zip:
        # 使用 'rb' 模式開啟檔案並發送 POST 請求
        with open(file_zip, 'rb') as f:
            response = requests.post(url, files={'file': f})

        # 輸出伺服器回應
        print(response.status_code)
        print(response.text)
    
    return print('OK!!!')