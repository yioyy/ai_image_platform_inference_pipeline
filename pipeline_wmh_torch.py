# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

主程只負責call子程式跟檔案的複製，不宜做任何細部運算。

@author: chuan
"""
import warnings
import os
import time
import logging
import shutil
import argparse
import subprocess
import tensorflow as tf  # type: ignore[import]

import pynvml  # type: ignore[import]  # GPU memory info

from pre_wmh import preprocess_wmh_images
from post_wmh import postprocess_wmh_results
from util_aneurysm import resampleSynthSEG2original
from after_run_wmh import after_run_wmh

warnings.filterwarnings("ignore")  # 忽略警告输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def pipeline_wmh(ID, 
                 Inputs,
                 DicomDirs,
                 path_output,
                 path_code = '/mnt/e/pipeline/chuan/code/', 
                 path_nnunet_model = '/mnt/e/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset015_DeepLacune/nnUNetTrainer__nnUNetPlans__2d',
                 path_processModel = '/mnt/e/pipeline/chuan/process/Deep_WMH/', 
                 path_json = '/mnt/e/pipeline/chuan/json/',
                 path_log = '/mnt/e/pipeline/chuan/log/', 
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

    logging.info('!!! Pre_WMH call.')

    path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)
    os.makedirs(path_processID, exist_ok=True)

    print(ID, ' Start...')
    logging.info(ID + ' Start...')


    #依照不同情境拆分try需要小心的事項 <= 重要
    try:
        # %% Deep learning相關
        pynvml.nvmlInit()  # 初始化
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_n)  # 获取GPU i的handle，后续通过handle来处理
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 通过handle获取GPU i的信息
        gpumRate = memoryInfo.used / memoryInfo.total
        # print('gpumRate:', gpumRate) #先設定gpu使用率小於0.2才跑predict code

        if gpumRate < 0.6:
            # plt.ion()    # 開啟互動模式，畫圖都是一閃就過
            # 一些記憶體的配置
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            tf.config.experimental.set_visible_devices(devices=gpus[gpu_n], device_type='GPU')
            # print(gpus, cpus)
            tf.config.experimental.set_memory_growth(gpus[gpu_n], True)

            if not Inputs:
                raise ValueError("Inputs 至少需要提供 T2FLAIR 影像")
            T2FLAIR_file = Inputs[0]
            SynthSEG_file = Inputs[1] if len(Inputs) > 1 else None

            if not DicomDirs:
                raise ValueError("DicomDirs 至少需要提供一個 T2FLAIR 目錄")
            path_DcmT2FLAIR = DicomDirs[0]

            shutil.copy(T2FLAIR_file, os.path.join(path_processID, 'T2FLAIR.nii.gz'))

            if SynthSEG_file is not None:
                shutil.copy(str(SynthSEG_file), os.path.join(path_processID, 'SynthSEG.nii.gz'))
            else:
                print('SynthSEG_file is None, 運行SynthSEG產生SynthSEG.nii.gz')
                path_nii = os.path.join(path_processID, 'nii')
                os.makedirs(path_nii, exist_ok=True)
                shutil.copy(T2FLAIR_file, os.path.join(path_nii, 'T2FLAIR.nii.gz'))

                path_synthseg = os.path.join(path_code, 'model_weights','SynthSeg_parcellation_tf28','code')

                cmd = [
                    "python", os.path.join(path_synthseg, 'main_tf28.py'),
                    "-i", path_nii,
                    "--input_name", 'T2FLAIR.nii.gz',
                    "--all", "False",
                    "--WMH", "True"
                    ]

                start = time.time()
                subprocess.run(cmd)
                print(f"[Done AI SynthSEG Inference... ] spend {time.time() - start:.0f} sec")
                logging.info(f"[Done AI SynthSEG Inference... ] spend {time.time() - start:.0f} sec")

                resampleSynthSEG2original(path_nii, 'T2FLAIR', 'WMH')
                resampleSynthSEG2original(path_nii, 'T2FLAIR', 'synthseg33')
                shutil.copy(os.path.join(path_nii, 'NEW_T2FLAIR_WMH.nii.gz'), os.path.join(path_processID, 'SynthSEG.nii.gz'))
                shutil.copy(os.path.join(path_nii, 'NEW_T2FLAIR_synthseg33.nii.gz'), os.path.join(path_processID, 'SynthSEG_Brain.nii.gz'))

            # 呼叫前處理函數：BET 處理、正規化、驗證輸出
            preprocess_success = preprocess_wmh_images(path_processID)

            if not preprocess_success:
                logging.error('!!! ' + ID + ' preprocessing failed.')
                return logging.error('!!! ' + ID + ' preprocessing failed.')

            # 建立 nnUNet 資料夾結構(類似 aneurysm 的 nnUNet 資料夾)
            path_nnunet = os.path.join(path_processID, 'nnUNet')
            if not os.path.isdir(path_nnunet):
                os.mkdir(path_nnunet)

            # 在 nnUNet 資料夾內建立子資料夾
            path_dcm_n = os.path.join(path_nnunet, 'Dicom')
            path_nii_n = os.path.join(path_nnunet, 'Image_nii')
            path_post_processed_n = os.path.join(path_nnunet, 'Post_processed')
            path_result_n = os.path.join(path_nnunet, 'Result')
            path_excel_n = os.path.join(path_nnunet, 'excel')
            
            for folder in [path_dcm_n, path_nii_n, path_post_processed_n, path_result_n, path_excel_n]:
                if not os.path.isdir(folder):
                    os.mkdir(folder)

            #因為松諭會用排程，但因為ai跟mip都要用到gpu，所以還是要管gpu ram，#multiprocessing沒辦法釋放gpu，要改用subprocess.run()
            print("Running stage 1: WMH inference!!!")
            logging.info("Running stage 1: WMH inference!!!")

            cmd = [
                   "python", os.path.join(path_code, "gpu_lacune_only_wmh.py"),
                   "--path_code", path_code,
                   "--path_process", path_processID,
                   "--path_nnunet_model", path_nnunet_model,
                   "--case", ID,
                   "--path_log", path_log,
                   "--gpu_n", str(gpu_n)
                  ]

            #result = subprocess.run(cmd, capture_output=True, text=True) 這會讓 subprocess.run() 自動幫你捕捉 stdout 和 stderr 的輸出，不然預設是印在 terminal 上，不會儲存。
            # 執行 subprocess
            start = time.time()
            subprocess.run(cmd)
            print(f"[Done AI Inference... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done AI Inference... ] spend {time.time() - start:.0f} sec")


            # 在 Dicom 資料夾內建立子資料夾並複製 T2FLAIR DICOM 檔案
            path_dcm_t2 = os.path.join(path_dcm_n, 'T2FLAIR_AXI')
            if not os.path.isdir(path_dcm_t2):
                shutil.copytree(path_DcmT2FLAIR, path_dcm_t2)

            #複製nifti檔到nnUNet資料夾
            if not os.path.isfile(os.path.join(path_nii_n, 'T2FLAIR.nii.gz')):
                shutil.copy(os.path.join(path_processID, 'T2FLAIR.nii.gz'), os.path.join(path_nii_n, 'T2FLAIR.nii.gz'))
            if not os.path.isfile(os.path.join(path_nii_n, 'SynthSEG.nii.gz')):
                shutil.copy(os.path.join(path_processID, 'SynthSEG.nii.gz'), os.path.join(path_nii_n, 'SynthSEG.nii.gz'))
            if not os.path.isfile(os.path.join(path_nii_n, 'Pred.nii.gz')):
                shutil.copy(os.path.join(path_nnunet, 'Pred.nii.gz'), os.path.join(path_nii_n, 'Pred.nii.gz'))
            
            #prob要複製到result資料夾中
            if not os.path.isfile(os.path.join(path_result_n, 'Prob.nii.gz')):
                shutil.copy(os.path.join(path_nnunet, 'Prob.nii.gz'), os.path.join(path_result_n, 'Prob.nii.gz'))

            # 後處理：生成視覺化圖片和報告
            print("Running stage 2: Post-processing!!!")
            logging.info("Running stage 2: Post-processing!!!")

            postprocess_success = postprocess_wmh_results(
                patient_id=ID,
                path_nnunet=path_nnunet,
                path_code=path_code,
                path_output=path_output,
                path_json=path_json
            )
            
            


            if not postprocess_success:
                logging.error('!!! ' + ID + ' post-processing failed.')
                return logging.error('!!! ' + ID + ' post-processing failed.')      

            after_run_success = after_run_wmh(path_nnunet, path_output, ID, path_code)
            if not after_run_success:
                logging.error('!!! ' + ID + ' after_run failed.')
                return logging.error('!!! ' + ID + ' after_run failed.')

            logging.info('!!! ' + ID +  ' post_wmh finish.')


    except Exception as e:
        logging.warning(f'Retry!!! have error code or no any study: {str(e)}')
        logging.error("Catch an exception.", exc_info=True)
        print(f'error: {str(e)}')     
    
    print('end!!!')

#其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=str, default = '01550089_20251117_MR_21411150023', help='目前執行的case的patient_id or study id')
    parser.add_argument('--Inputs', type=str, nargs='+', default = ['/data/4TB1/pipeline/chuan/example_input/01550089_20251117_MR_21411150023/T2FLAIR_AXI.nii.gz'], help='用於輸入的檔案')
    parser.add_argument('--DicomDir', type=str, nargs='+', default = ['/data/4TB1/pipeline/chuan/example_inputDicom/01550089_20251117_MR_21411150023/T2FLAIR_AXI/'], help='用於輸入的檔案')
    parser.add_argument('--Output_folder', type=str, default = '/data/4TB1/pipeline/chuan/example_output/01550089_20251117_MR_21411150023/',help='用於輸出結果的資料夾')    
    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs  # 將列表合併為字符串，保留順序
    DicomDirs = args.DicomDir #對應的dicom資料夾，用來做dicom-seg
    path_output = str(args.Output_folder)

    #下面設定各個路徑
    path_code = '/data/4TB1/pipeline/chuan/code/'
    path_process = '/data/4TB1/pipeline/chuan/process/'  #前處理dicom路徑(test case)
    path_nnunet_model = '/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset015_DeepLacune/nnUNetTrainer__nnUNetPlans__2d'

    path_processModel = os.path.join(path_process, 'Deep_WMH')  #前處理dicom路徑(test case)
    #path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)

    #這裡先沒有dicom
    path_json = '/data/4TB1/pipeline/chuan/json/'  #存放json的路徑，回傳執行結果
    #json_path_name = os.path.join(path_json, 'Pred_Infarct.json')
    path_log = '/data/4TB1/pipeline/chuan/log/'  #log資料夾
    gpu_n = 0  #使用哪一顆gpu

    # 建置資料夾
    os.makedirs(path_processModel, exist_ok=True) # 如果資料夾不存在就建立，製作nii資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_output,exist_ok=True)

    #直接當作function的輸入，因為可能會切換成nnUNet的版本，所以自訂化模型移到跟model一起，synthseg自己做，不用統一
    pipeline_wmh(ID, Inputs, DicomDirs, path_output, path_code, path_nnunet_model, path_processModel, path_json, path_log, gpu_n)

    # #最後再讀取json檔結果
    # with open(json_path_name) as f:
    #     data = json.load(f)

    # logging.info('Json!!! ' + str(data))