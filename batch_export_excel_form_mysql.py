# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:56:16 2023

寫一個程式去執行唐國庭批次匯出excel檔並整理成一個excel
使用excel當作匯入工具，確認第一列標題欄位是否有甚麼甚麼來確定要匯入的條件
excel搜尋，要嘛要有Patient ID、要嘛要有Accession Number
export:predict '2023-04-18 00:00:00' '2200-12-31 00:00:00' 隨便設一個誇張的數字，給設定但預設誇張 
 ‘WMH’ / ‘STROKE’ / ‘CVR’


除了可以設定on變數之外，pd.merge()還有其他常用的變數可以設定，包括：

how: 設定合併的方式，預設為 'inner'。常用的方式還包括 'left', 'right', 'outer'，分別表示左邊DataFrame以key為基準做合併、
右邊DataFrame以key為基準做合併、以及左右DataFrame都以key為基準做合併。
left_on和right_on: 如果兩個要合併的DataFrame中的欄位名稱不同，可以使用這兩個變數分別指定兩個DataFrame中要合併的欄位。
left_index和right_index: 如果要以索引值作為合併的基準，可以將這兩個變數設為True。
suffixes: 指定相同欄位名稱的兩個DataFrame合併時，要如何標記欄位名稱的後綴字元，預設為 ('_x', '_y')。

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--list', nargs='+', type=str)

args = parser.parse_args(['--list', 'a', 'b', 'c'])
print(args.list)  # ['a', 'b', 'c']

@author: user
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
#要安裝pillow 去開啟
import argparse
import time

#path_in = '/var/www/tmu-pacs/laravel/storage/app/study.xlsx' #輸入的excel的路徑
#path_out = '/home/tmu/Desktop/study.xlsx' #輸出的位置跟檔名，預設桌面
#models = ['WMH', 'STROKE', 'CVR'] #選擇查詢哪個模型

def export_excel_from_mysql(path_in, path_out, models):
    # 指定欄位數據類型
    dtypes = {'id': str, '模型': str, '建立時間': str, '更新時間': str, 'PatientID': str, 'StudyDate': str,
            'PatientName': str, 'AccessionNumber': str, 'StudyID': str, 'PatientBirth': str, 'PatientAge': str,
            'PatientSex': str, 'MagneticFieldStrength': str, 'StudyDescription': str}

    print('Start...')
    start_time = '2023-04-18 00:00:00'
    end_time = '2200-12-31 00:00:00' #隨便設一個誇張的數字，給設定但預設誇張 

    #先移除前一次的batch檔，並且建立新的空白batch檔
    excel_export = '/var/www/tmu-pacs/laravel/storage/app/study.xlsx' #php程式輸出的excel
    excel_batch = '/var/www/tmu-pacs/laravel/storage/app/study_batch.xlsx'
    if os.path.isfile(excel_batch):
        os.remove(excel_batch)
    # 建立一個空的DataFrame物件
    df_batch = pd.DataFrame(columns=['id', '模型', 'volume', '建立時間', '更新時間', 'PatientID', 'StudyDate', 'PatientName', 
                            'AccessionNumber', 'StudyID', 'PatientBirth', 'PatientAge', 'PatientSex', 'PatientWeight',
                            'MagneticFieldStrength', 'StudyDescription', 'ImagesAcquisition', 'PixelSpacing', 
                            'SpacingBetweenSlices', 'InfarctSliceNum', 'InfarctVoxel', 'MeanADC', 'WMHSliceNum',
                            'WMHVoxel', 'Fazekas']) #所有內容
    # 儲存為Excel檔案
    df_batch.to_excel(excel_batch, index=False)
        
    #先讀取excel
    df = pd.read_excel(path_in)

    if 'PatientID' in df.columns:
        PIDs = list(df['PatientID'])
        if len(PIDs) > 0:
            PIDs = [str(y).rjust(8,'0') for y in PIDs] #改成字串
        else:
            PIDs = ''
    else:
        PIDs = '' #沒任何東西

    if 'StudyDate' in df.columns:
        SDATEs = list(df['StudyDate'])
        if len(SDATEs) > 0:
            #確定StudyDate是8位(年月日)還是12位(年月日十分)，8位則改成12位，12位也建立bg_request_date跟sm_request_date
            if len(str(SDATEs[0])) == 12:
                #12位，不用更改條件，直接建立bg_request_date跟sm_request_date
                bg_request_date = [str(y).rjust(12,'0') for y in SDATEs]
                sm_request_date = [str(y).rjust(12,'0') for y in SDATEs]
            elif len(str(SDATEs[0])) == 8:
                #8位bg_request_date跟sm_request_date，加上00:00跟12:31，因為指令是包含
                bg_request_date = [str(y).rjust(8,'0') + '0000' for y in SDATEs] #加上0點0分
                sm_request_date = [str(y).rjust(8,'0') + '2359' for y in SDATEs] #加上23點59分
            else:
                bg_request_date = ''
                sm_request_date = ''
        else:
            bg_request_date = ''
            sm_request_date = ''
    else:
        bg_request_date = ''
        sm_request_date = ''
        
    if 'AccessionNumber' in df.columns:
        ANs = list(df['AccessionNumber'])
        if len(ANs):
            ANs = [str(y) for y in ANs] #轉成字串
        else:
            ANs = '' #沒任何東西
    else:
        ANs = '' #沒任何東西    
        
    #使用判別式決定撈資料的欄位依據，因為PatientID跟StudyDate跟AccessionNumber都有的話長度會是一致
    #一行一行撈
    if len(PIDs) > 0 and len(bg_request_date) > 0 and len(ANs) > 0:
        for pid, brd, srd, an in zip(PIDs, bg_request_date, sm_request_date, ANs):
            #一行一行去確認能不能輸出正確的東西
            if len(pid + brd) == 20 or len(an) > 0:
                for model in models:
                    bash_line = 'cd /var/www/tmu-pacs/laravel && php artisan export:predict "' + start_time + '" "' + end_time \
                                + '" --type=' + model + ' --chr_no=' + pid + ' --bg_request_date=' + brd + ' --sm_request_date='\
                                + srd + ' --accession_number=' + an
                    os.system(bash_line)
                    #print(bash_line)
                    #讀取php輸出的excel，將內容合併到excel_batch中
                    df1 = pd.read_excel(excel_batch, dtype=dtypes) #excel_batch
                    df2 = pd.read_excel(excel_export, dtype=dtypes)
                    # 以某個欄位作為合併依據
                    concat_df = pd.concat([df1, df2]) #以左邊excel為主，合併df1沒有的
                    concat_df = concat_df.fillna(' ')
                    # 將合併結果寫入新的 Excel 檔案
                    concat_df.to_excel(excel_batch, index=False)
                    
    elif len(PIDs) > 0 and len(bg_request_date) > 0 and len(ANs) < 1:
        #沒有accession number但有patient id + study date
        for pid, brd, srd in zip(PIDs, bg_request_date, sm_request_date):
            #一行一行確認要同時符合有PIDs跟bg_request_date
            if len(pid) > 0 and len(brd) > 0:
                for model in models:
                    bash_line = 'cd /var/www/tmu-pacs/laravel && php artisan export:predict "' + start_time + '" "' + end_time \
                                + '" --type=' + model + ' --chr_no=' + pid + ' --bg_request_date=' + brd + ' --sm_request_date='\
                                + srd
                    os.system(bash_line)
                    #print(bash_line)
                    #讀取php輸出的excel，將內容合併到excel_batch中
                    df1 = pd.read_excel(excel_batch, dtype=dtypes) #excel_batch
                    df2 = pd.read_excel(excel_export, dtype=dtypes)
                    # 以某個欄位作為合併依據
                    concat_df = pd.concat([df1, df2]) #以左邊excel為主，合併df1沒有的
                    concat_df = concat_df.fillna(' ')
                    # 將合併結果寫入新的 Excel 檔案
                    concat_df.to_excel(excel_batch, index=False)   

    elif len(PIDs) > 0 and len(bg_request_date) < 1 and len(ANs) > 0:
        for pid, an in zip(PIDs, ANs):
            #一行一行去確認能不能輸出正確的東西
            if len(pid) == 8 or len(an) > 0:
                for model in models:
                    bash_line = 'cd /var/www/tmu-pacs/laravel && php artisan export:predict "' + start_time + '" "' + end_time \
                                + '" --type=' + model + ' --chr_no=' + pid + ' --accession_number=' + an
                    os.system(bash_line)
                    #print(bash_line)
                    #讀取php輸出的excel，將內容合併到excel_batch中
                    df1 = pd.read_excel(excel_batch, dtype=dtypes) #excel_batch
                    df2 = pd.read_excel(excel_export, dtype=dtypes)
                    # 以某個欄位作為合併依據
                    concat_df = pd.concat([df1, df2]) #以左邊excel為主，合併df1沒有的
                    concat_df = concat_df.fillna(' ')
                    # 將合併結果寫入新的 Excel 檔案
                    concat_df.to_excel(excel_batch, index=False)
                    
    elif len(PIDs) > 0 and len(bg_request_date) < 1 and len(ANs) < 1:
        #只有patient id，其他都沒有
        for pid in PIDs:
            #一行一行確認要同時符合有PIDs跟bg_request_date
            if len(pid):
                for model in models:
                    bash_line = 'cd /var/www/tmu-pacs/laravel && php artisan export:predict "' + start_time + '" "' + end_time \
                                + '" --type=' + model + ' --chr_no=' + pid 
                    os.system(bash_line)
                    #print(bash_line)
                    #讀取php輸出的excel，將內容合併到excel_batch中
                    df1 = pd.read_excel(excel_batch, dtype=dtypes) #excel_batch
                    df2 = pd.read_excel(excel_export, dtype=dtypes)
                    # 以某個欄位作為合併依據
                    concat_df = pd.concat([df1, df2]) #以左邊excel為主，合併df1沒有的
                    concat_df = concat_df.fillna(' ')
                    # 將合併結果寫入新的 Excel 檔案
                    concat_df.to_excel(excel_batch, index=False) 

    elif len(PIDs) < 1 and len(ANs) > 0:
        #沒有patient id，有accession number，不管bg_request_date
        for an in ANs:
            #一行一行確認
            if len(an) > 0:
                for model in models:
                    bash_line = 'cd /var/www/tmu-pacs/laravel && php artisan export:predict "' + start_time + '" "' + end_time \
                                + '" --type=' + model + ' --accession_number=' + an
                    os.system(bash_line)
                    #print(bash_line)
                    #讀取php輸出的excel，將內容合併到excel_batch中
                    df1 = pd.read_excel(excel_batch, dtype=dtypes) #excel_batch
                    df2 = pd.read_excel(excel_export, dtype=dtypes)
                    # 以某個欄位作為合併依據
                    concat_df = pd.concat([df1, df2]) #以左邊excel為主，合併df1沒有的
                    concat_df = concat_df.fillna(' ')
                    # 將合併結果寫入新的 Excel 檔案
                    concat_df.to_excel(excel_batch, index=False)   
                    
    else:
        print('No conditions can be used')
                    
    #最後把excel_batch輸出到path_out中
    shutil.copy(excel_batch, path_out)
    
    return print('End')
                
#其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default='/home/tmu/Desktop/export_excel_example/example1.xlsx', help='撈取MySQL資料要輸入的Excel檔案路徑，欄位必須有PatientID或是AccessionNumber，加上StudyDate(可選)') #在同一層輸出結果
    parser.add_argument('-o', type=str, default='/home/tmu/Desktop/study.xlsx', help='撈取MySQL資料要輸出的Excel檔案路徑') #如果沒有標註檔，常態維持None
    parser.add_argument('-model', nargs='+', type=str, default=['WMH', 'STROKE', 'CVR'], help='要查詢哪些模型的結果，要以list方式輸入，例如: WMH STROKE')
    args = parser.parse_args()
    
    export_excel_from_mysql(args.i, args.o, args.model)            