# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

主程只負責call子程式跟檔案的複製，不宜做任何細部運算。

@author: chuan
"""
import warnings
warnings.filterwarnings("ignore")  # 忽略警告输出

import os
import sys
import time
import logging
import shutil
import argparse
import subprocess
from pre_infarct import preprocess_stroke_images
from post_infarct import postprocess_infarct_results
from util_aneurysm import resampleSynthSEG2original
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pynvml  # GPU memory info
from after_run_infarct import after_run


# Optional env-file loader (for test vs official)
try:
    from config.load_env import load_env_from_default_locations
except Exception:
    load_env_from_default_locations = None


# 讀取環境變數時的 path 清理
def _clean_env_path(p: str) -> str:
    return os.path.normpath(str(p).strip().replace("\r", "").replace("\n", ""))


def _env_path(key: str, default: str) -> str:
    v = os.getenv(key, "").strip()
    return _clean_env_path(v) if v else _clean_env_path(default)


def prepare_synthseg_file(SynthSEG_file, path_processID, ADC_file, DWI0_file, DWI1000_file, path_code):
    """
    準備 SynthSEG 檔案：如果提供了檔案就直接複製，否則運行 SynthSEG 推理
    
    參數:
        SynthSEG_file: SynthSEG 檔案路徑（可為 None）
        path_processID: 處理資料夾路徑
        ADC_file: ADC 影像檔案路徑
        DWI0_file: DWI0 影像檔案路徑
        DWI1000_file: DWI1000 影像檔案路徑
        path_code: 程式碼根目錄路徑
    
    返回:
        SynthSEG.nii.gz 的完整路徑
    """
    synthseg_output = os.path.join(path_processID, 'SynthSEG.nii.gz')
    
    if SynthSEG_file is not None:
        print('SynthSEG_file is not None')
        shutil.copy(SynthSEG_file, synthseg_output)
    else:
        print('SynthSEG_file is None, 運行SynthSEG產生SynthSEG.nii.gz')
        
        # 建立 nii 資料夾
        path_nii = os.path.join(path_processID, 'nii')
        if not os.path.isdir(path_nii):
            os.mkdir(path_nii)
        
        # 複製影像檔案
        shutil.copy(ADC_file, os.path.join(path_nii, 'ADC.nii.gz'))
        shutil.copy(DWI0_file, os.path.join(path_nii, 'DWI0.nii.gz'))
        shutil.copy(DWI1000_file, os.path.join(path_nii, 'DWI1000.nii.gz'))
        
        # 設定 SynthSEG 路徑
        path_synthseg = os.path.join(path_code, 'model_weights', 'SynthSeg_parcellation_tf28', 'code')
        
        # 建立指令
        cmd = [
            "python", os.path.join(path_synthseg, 'main_tf28.py'),
            "-i", path_nii,
            "--input_name", 'DWI0.nii.gz',
            "--template", path_nii,
            "--template_name", 'DWI0',
            "--all", "False",
            "--DWI", "True"
        ]
        
        # 執行 SynthSEG 推理
        start = time.time()
        synthseg_env = os.environ.copy()
        synthseg_env["CUDA_VISIBLE_DEVICES"] = str(gpu_n)
        subprocess.run(cmd, env=synthseg_env)
        print(f"[Done AI SynthSEG Inference... ] spend {time.time() - start:.0f} sec")
        logging.info(f"[Done AI SynthSEG Inference... ] spend {time.time() - start:.0f} sec")
        
        # Resample 到原始空間
        SynthSEG_array = resampleSynthSEG2original(path_nii, 'DWI0', 'DWI')
        
        # 複製結果
        shutil.copy(os.path.join(path_nii, 'NEW_DWI0_DWI.nii.gz'), synthseg_output)
    
    return synthseg_output


# 這些函數已移到 post_infarct.py：data_translate, data_translate_back, parcellation, case_json


def pipeline_infarct(ID, 
                     Inputs,
                     DicomDirs,
                     path_output,
                     path_code = '/mnt/e/pipeline/chuan/code/', 
                     path_nnunet_model = '/mnt/e/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset040_DeepInfarct/nnUNetTrainer__nnUNetPlans__2d',
                     path_processModel = '/mnt/e/pipeline/chuan/process/Deep_Infarct/', 
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

    logging.info('!!! Pre_Infarct call.')

    path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)
    if not os.path.isdir(path_processID):  #如果資料夾不存在就建立
        os.mkdir(path_processID) #製作nii資料夾

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
            autotune = tf.data.experimental.AUTOTUNE
            # print(keras.__version__)
            # print(tf.__version__)
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            tf.config.experimental.set_visible_devices(devices=gpus[gpu_n], device_type='GPU')
            # print(gpus, cpus)
            tf.config.experimental.set_memory_growth(gpus[gpu_n], True)

            #讀出DWI, DWI0, ADC, SynthSEG的檔案
            ADC_file = Inputs[0]
            DWI0_file = Inputs[1]
            DWI1000_file = Inputs[2]
            if len(Inputs) > 3:
                SynthSEG_file = Inputs[3]
            else:
                #沒有SynthSEG的檔案，則要自己手動作synthseg
                SynthSEG_file = None

            path_DcmADC = DicomDirs[0] #這邊在做dicom-seg時才會用到
            path_DcmDWI0 = DicomDirs[1] #這邊在做dicom-seg時才會用到
            path_DcmDWI1000 = DicomDirs[2] #這邊在做dicom-seg時才會用到

            # 建立主要的輸入影像資料夾(複製到主資料夾下，類似 aneurysm 的 MRA_BRAIN.nii.gz)
            shutil.copy(ADC_file, os.path.join(path_processID, 'ADC.nii.gz'))
            shutil.copy(DWI0_file, os.path.join(path_processID, 'DWI0.nii.gz'))
            shutil.copy(DWI1000_file, os.path.join(path_processID, 'DWI1000.nii.gz'))

            if SynthSEG_file is not None:
                print('SynthSEG_file is not None')
                shutil.copy(str(SynthSEG_file), os.path.join(path_processID, 'SynthSEG.nii.gz'))
            else:
                print('SynthSEG_file is None, 運行SynthSEG產生SynthSEG.nii.gz')
                #沒有SynthSEG的檔案，則要自己手動作synthseg

                #原先指令為：
                #gpu_line = 'python ' + os.path.join(path_synthseg, 'main.py -i ') + path_nii + ' --input_name DWI0.nii.gz --template ' + path_nii + ' --template_name DWI0 --all False --DWI TRUE'
                #os.system(gpu_line)
                # 定義要傳入的參數，建立指令

                path_nii = os.path.join(path_processID, 'nii')
                if not os.path.isdir(path_nii):
                    os.mkdir(path_nii)
                shutil.copy(ADC_file, os.path.join(path_nii, 'ADC.nii.gz'))
                shutil.copy(DWI0_file, os.path.join(path_nii, 'DWI0.nii.gz'))
                shutil.copy(DWI1000_file, os.path.join(path_nii, 'DWI1000.nii.gz'))

                path_synthseg = os.path.join(path_code, 'model_weights','SynthSeg_parcellation_tf28','code')

                cmd = [
                    "python", os.path.join(path_synthseg, 'main_tf28.py'),
                    "-i", path_nii,
                    "--input_name", 'DWI0.nii.gz',
                    "--template", path_nii,
                    "--template_name", 'DWI0',
                    "--all", "False",
                    "--DWI", "True"
                    ]


                #result = subprocess.run(cmd, capture_output=True, text=True) 這會讓 subprocess.run() 自動幫你捕捉 stdout 和 stderr 的輸出，不然預設是印在 terminal 上，不會儲存。
                # 執行 subprocess
                start = time.time()
                synthseg_env = os.environ.copy()
                synthseg_env["CUDA_VISIBLE_DEVICES"] = str(gpu_n)
                subprocess.run(cmd, env=synthseg_env)
                print(f"[Done AI SynthSEG Inference... ] spend {time.time() - start:.0f} sec")
                logging.info(f"[Done AI SynthSEG Inference... ] spend {time.time() - start:.0f} sec")

                #這邊還要做resample到original space
                SynthSEG_array = resampleSynthSEG2original(path_nii, 'DWI0', 'DWI') 

                shutil.copy(os.path.join(path_nii, 'NEW_DWI0_DWI.nii.gz'), os.path.join(path_processID, 'SynthSEG.nii.gz'))

            # 呼叫前處理函數：BET 處理、正規化、驗證輸出
            preprocess_success = preprocess_stroke_images(path_processID)

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
            #model_predict_aneurysm(path_code, path_processID, ID, path_log, gpu_n)
            print("Running stage 1: Infarct inference!!!")
            logging.info("Running stage 1: Infarct inference!!!")

            # 定義要傳入的參數，建立指令
            cmd = [
                   "python", "/data/4TB1/pipeline/chuan/code/gpu_infarct.py",
                   "--path_code", path_code,
                   "--path_process", path_processID,
                   "--path_nnunet_model", path_nnunet_model,
                   "--case", ID,
                   "--path_log", path_log,
                   "--gpu_n", str(gpu_n)  # 注意要轉成字串
                  ]

            #result = subprocess.run(cmd, capture_output=True, text=True) 這會讓 subprocess.run() 自動幫你捕捉 stdout 和 stderr 的輸出，不然預設是印在 terminal 上，不會儲存。
            # 執行 subprocess
            start = time.time()
            subprocess.run(cmd)
            print(f"[Done AI Inference... ] spend {time.time() - start:.0f} sec")
            logging.info(f"[Done AI Inference... ] spend {time.time() - start:.0f} sec")


            # 在 Dicom 資料夾內建立子資料夾並複製 DICOM 檔案
            path_dcm_adc = os.path.join(path_dcm_n, 'ADC')
            path_dcm_dwi0 = os.path.join(path_dcm_n, 'DWI0')
            path_dcm_dwi1000 = os.path.join(path_dcm_n, 'DWI1000')
            
            if not os.path.isdir(path_dcm_adc):
                shutil.copytree(path_DcmADC, path_dcm_adc)
            if not os.path.isdir(path_dcm_dwi0):
                shutil.copytree(path_DcmDWI0, path_dcm_dwi0)
            if not os.path.isdir(path_dcm_dwi1000):
                shutil.copytree(path_DcmDWI1000, path_dcm_dwi1000)

            #複製nifit檔到nnUNet資料夾
            if not os.path.isfile(os.path.join(path_nii_n, 'ADC.nii.gz')):
                shutil.copy(os.path.join(path_processID, 'ADC.nii.gz'), os.path.join(path_nii_n, 'ADC.nii.gz'))
            if not os.path.isfile(os.path.join(path_nii_n, 'DWI0.nii.gz')):
                shutil.copy(os.path.join(path_processID, 'DWI0.nii.gz'), os.path.join(path_nii_n, 'DWI0.nii.gz'))
            if not os.path.isfile(os.path.join(path_nii_n, 'DWI1000.nii.gz')):
                shutil.copy(os.path.join(path_processID, 'DWI1000.nii.gz'), os.path.join(path_nii_n, 'DWI1000.nii.gz'))
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

            #複製nifit檔到nnUNet資料夾
            if not os.path.isfile(os.path.join(path_nii_n, 'SynthSEG.nii.gz')):
                shutil.copy(os.path.join(path_processID, 'SynthSEG.nii.gz'), os.path.join(path_nii_n, 'SynthSEG.nii.gz'))

            postprocess_success = postprocess_infarct_results(
                patient_id=ID,
                path_nnunet=path_nnunet,
                path_code=path_code,
                path_output=path_output,
                path_json=path_json
            )
            
            if not postprocess_success:
                logging.error('!!! ' + ID + ' post-processing failed.')
                return logging.error('!!! ' + ID + ' post-processing failed.')      

            #後面是after_run，主要是做統計、生成dicom、生成dicom-seg、上傳json
            group_id = int(os.getenv("RADX_INFARCT_GROUP_ID", "56") or "56")
            after_run_success = after_run(path_nnunet, path_output, ID, path_code, group_id=group_id)
            if not after_run_success:
                logging.error('!!! ' + ID + ' after_run failed.')
                return logging.error('!!! ' + ID + ' after_run failed.')

            logging.info('!!! ' + ID +  ' post_stroke finish.')


    except Exception as e:
        logging.warning(f'Retry!!! have error code or no any study: {str(e)}')
        logging.error("Catch an exception.", exc_info=True)
        print(f'error: {str(e)}')     
    
    print('end!!!')

#其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    # Load env file (RADX_ENV_FILE or code/config/radax.env) if present
    if load_env_from_default_locations:
        try:
            load_env_from_default_locations()
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=str, default = '01550089_20251117_MR_21411150023', help='目前執行的case的patient_id or study id')
    parser.add_argument('--Inputs', type=str, nargs='+', default = ['/data/4TB1/pipeline/chuan/example_input/01550089_20251117_MR_21411150023/ADC.nii.gz', '/data/4TB1/pipeline/chuan/example_input/01550089_20251117_MR_21411150023/DWI0.nii.gz', '/data/4TB1/pipeline/chuan/example_input/01550089_20251117_MR_21411150023/DWI1000.nii.gz'], help='用於輸入的檔案')
    parser.add_argument('--DicomDir', type=str, nargs='+', default = ['/data/4TB1/pipeline/chuan/example_inputDicom/01550089_20251117_MR_21411150023/ADC/', '/data/4TB1/pipeline/chuan/example_inputDicom/01550089_20251117_MR_21411150023/DWI0/', '/data/4TB1/pipeline/chuan/example_inputDicom/01550089_20251117_MR_21411150023/DWI1000/'], help='用於輸入的檔案')
    parser.add_argument('--Output_folder', type=str, default = '/data/4TB1/pipeline/chuan/example_output/01550089_20251117_MR_21411150023/',help='用於輸出結果的資料夾')    
    parser.add_argument('--gpu_n', type=int, default = 0, help='使用哪一顆gpu')
    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs  # 將列表合併為字符串，保留順序
    DicomDirs = args.DicomDir #對應的dicom資料夾，用來做dicom-seg
    path_output = str(args.Output_folder)
    if "--Output_folder" not in sys.argv:
        path_output = _env_path("RADX_OUTPUT_ROOT", path_output)

    #下面設定各個路徑
    path_code = _env_path("RADX_CODE_ROOT", "/data/4TB1/pipeline/chuan/code/")
    path_process = _env_path("RADX_PROCESS_ROOT", "/data/4TB1/pipeline/chuan/process/")  #前處理dicom路徑(test case)
    path_nnunet_model = _env_path(
        "RADX_INFARCT_MODEL",
        "/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset040_DeepInfarct/nnUNetTrainer__nnUNetPlans__2d",
    )

    path_processModel = os.path.join(path_process, 'Deep_Infarct')  #前處理dicom路徑(test case)
    #path_processID = os.path.join(path_processModel, ID)  #前處理dicom路徑(test case)

    #這裡先沒有dicom
    path_json = _env_path("RADX_JSON_ROOT", "/data/4TB1/pipeline/chuan/json/")  #存放json的路徑，回傳執行結果
    #json_path_name = os.path.join(path_json, 'Pred_Infarct.json')
    path_log = _env_path("RADX_LOG_ROOT", "/data/4TB1/pipeline/chuan/log/")  #log資料夾
    gpu_n = int(args.gpu_n)  #使用哪一顆gpu

    # 建置資料夾
    os.makedirs(path_processModel, exist_ok=True) # 如果資料夾不存在就建立，製作nii資料夾
    os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，
    os.makedirs(path_output,exist_ok=True)

    #直接當作function的輸入，因為可能會切換成nnUNet的版本，所以自訂化模型移到跟model一起，synthseg自己做，不用統一
    pipeline_infarct(ID, Inputs, DicomDirs, path_output, path_code, path_nnunet_model, path_processModel, path_json, path_log, gpu_n)

    # #最後再讀取json檔結果
    # with open(json_path_name) as f:
    #     data = json.load(f)

    # logging.info('Json!!! ' + str(data))