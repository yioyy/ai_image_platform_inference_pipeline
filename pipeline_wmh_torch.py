# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

主程只負責call子程式跟檔案的複製，不宜做任何細部運算。

@author: chuan
"""
import warnings
import os
import sys
import time
import logging
import shutil
import argparse
import subprocess
import json
import pathlib
import tempfile
import tensorflow as tf  # type: ignore[import]

import pynvml  # type: ignore[import]  # GPU memory info

from pre_wmh import preprocess_wmh_images
from post_wmh import postprocess_wmh_results
from util_aneurysm import resampleSynthSEG2original, upload_json_aiteam
from after_run_wmh import after_run_wmh
from pipeline_followup_v3_platform import pipeline_followup_v3_platform

warnings.filterwarnings("ignore")  # 忽略警告输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


def pipeline_wmh(ID, 
                 Inputs,
                 DicomDirs,
                 path_output,
                 path_code = '/mnt/e/pipeline/chuan/code/', 
                 path_nnunet_model = '/mnt/e/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset015_DeepLacune/nnUNetTrainer__nnUNetPlans__2d',
                 path_processModel = '/mnt/e/pipeline/chuan/process/Deep_WMH/', 
                 path_json = '/mnt/e/pipeline/chuan/json/',
                 path_log = '/mnt/e/pipeline/chuan/log/', 
                 gpu_n = 0,
                 input_json: str = "",
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
                synthseg_env = os.environ.copy()
                synthseg_env["CUDA_VISIBLE_DEVICES"] = str(gpu_n)
                subprocess.run(cmd, env=synthseg_env)
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
            if not os.path.isfile(os.path.join(path_nii_n, 'SynthSEG_Brain.nii.gz')):
                shutil.copy(os.path.join(path_processID, 'SynthSEG_Brain.nii.gz'), os.path.join(path_nii_n, 'SynthSEG_Brain.nii.gz'))
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

            group_id = int(os.getenv("RADX_WMH_GROUP_ID", "56") or "56")
            after_run_success = after_run_wmh(path_nnunet, path_output, ID, path_code, group_id=group_id)
            if not after_run_success:
                logging.error('!!! ' + ID + ' after_run failed.')
                return logging.error('!!! ' + ID + ' after_run failed.')

            # ============ followup-v3 platform（可選）============
            followup_process_root = pathlib.Path(
                os.getenv(
                    "RADX_FOLLOWUP_CASE_ROOT",
                    os.getenv("RADX_FOLLOWUP_ROOT", str(pathlib.Path(path_output).resolve().parent)),
                )
            )
            followup_output_root = pathlib.Path(
                os.getenv("RADX_FOLLOWUP_ROOT", str(followup_process_root))
            )

            # 先把本次 case 的必要檔名補到 followup case 目錄（避免 followup 找不到檔）
            try:
                case_dir = (
                    followup_process_root
                    if followup_process_root.name == ID
                    else followup_process_root / ID
                )
                os.makedirs(case_dir, exist_ok=True)

                src_pred = os.path.join(path_output, "Pred_WMH.nii.gz")
                if os.path.isfile(src_pred):
                    shutil.copy(src_pred, os.path.join(case_dir, "Pred_WMH.nii.gz"))

                src_synthseg = os.path.join(path_nnunet, "Image_nii", "SynthSEG.nii.gz")
                if os.path.isfile(src_synthseg):
                    shutil.copy(src_synthseg, os.path.join(case_dir, "SynthSEG_WMH.nii.gz"))

                json_file_n = os.path.join(path_nnunet, "JSON", ID + "_platform_json.json")
                if os.path.isfile(json_file_n):
                    shutil.copy(
                        json_file_n,
                        os.path.join(case_dir, "Pred_WMH_platform_json.json"),
                    )
            except Exception as exc:
                logging.warning("Prepare followup case files failed: %s", exc)

            temp_json_paths = []
            # 若有提供 input_json（可能是檔案路徑或 JSON 內容）
            input_json = str(input_json).strip().replace("\r", "").replace("\n", "")
            if input_json:
                raw_data = None
                if os.path.isfile(input_json):
                    try:
                        with open(input_json, "r", encoding="utf-8") as _f:
                            raw_data = json.load(_f)
                    except (json.JSONDecodeError, OSError):
                        logging.warning("Skip followup: input_json file is not valid JSON.")
                else:
                    try:
                        raw_data = json.loads(input_json)
                    except json.JSONDecodeError:
                        logging.warning("Skip followup: input_json is not valid JSON content.")

                if raw_data is not None:
                    if isinstance(raw_data, list):
                        raw_data = {"needFollowup": raw_data}
                    os.makedirs(path_processModel, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        suffix=".json",
                        delete=False,
                        dir=path_processModel,
                        encoding="utf-8",
                    ) as temp_fp:
                        json.dump(raw_data, temp_fp, ensure_ascii=False)
                        input_json_path = temp_fp.name
                        temp_json_paths.append(temp_fp.name)
                else:
                    input_json_path = ""
            else:
                # 未提供 input_json 時，不執行 followup
                input_json_path = ""

            outputs = []
            if input_json_path:
                start_followup = time.time()
                try:
                    outputs, has_model = pipeline_followup_v3_platform(
                        input_json=pathlib.Path(input_json_path),
                        path_process=followup_process_root,
                        model="WMH",
                        model_type=4,
                        case_id=ID,
                        platform_json_name="Pred_WMH_platform_json.json",
                        path_followup_root=followup_output_root,
                    )
                    logging.info(
                        "Followup-v3 outputs=%d has_model=%s", len(outputs), has_model
                    )
                    print(
                        f"[FollowUp] outputs={len(outputs)} has_model={has_model}"
                    )
                except Exception as exc:
                    has_model = False
                    logging.error("Followup-v3 platform failed.", exc_info=True)
                    print(f"[FollowUp] Error: followup-v3 platform failed: {exc!r}")

                print(
                    f"[Done followup-v3 platform!!! ] spend {time.time() - start_followup:.0f} sec"
                )
                logging.info(
                    f"[Done followup-v3 platform!!! ] spend {time.time() - start_followup:.0f} sec"
                )
            else:
                has_model = False
                logging.info("Skip followup-v3 platform: input_json is empty.")
                print("[FollowUp] Skip: input_json is empty.")

            for temp_path in temp_json_paths:
                try:
                    os.remove(temp_path)
                    logging.info("Remove temp json: %s", temp_path)
                except OSError:
                    logging.warning("Failed to remove temp json: %s", temp_path)

            # 接下來，上傳 json
            json_file_n = os.path.join(path_nnunet, "JSON", ID + "_platform_json.json")
            # followup 有執行且存在該模型時，上傳 followup 結果（含當前日期主體）
            if has_model and outputs:
                for json_path in outputs:
                    try:
                        resp = upload_json_aiteam(str(json_path))
                        logging.info("Upload followup json: %s resp=%s", json_path, resp)
                        print(f"[Upload] followup json={json_path} resp={resp}")
                    except Exception as exc:
                        logging.error("Upload followup json failed: %s", json_path, exc_info=True)
                        print(f"[Upload] followup json failed: {json_path} err={exc}")
            else:
                # followup 未執行或無模型時，上傳原版 JSON
                try:
                    resp = upload_json_aiteam(json_file_n)
                    logging.info("Upload original json: %s resp=%s", json_file_n, resp)
                    print(f"[Upload] original json={json_file_n} resp={resp}")
                except Exception as exc:
                    logging.error("Upload original json failed: %s", json_file_n, exc_info=True)
                    print(f"[Upload] original json failed: {json_file_n} err={exc}")

            # 若有 followup，就 copy followup JSON 到 output 並以相同檔名保存（覆蓋原版）
            try:
                json_src_for_output = json_file_n
                if has_model and outputs:
                    case_date_key = ""
                    try:
                        parts = str(ID).split("_")
                        if len(parts) >= 2 and len(parts[1]) == 8 and parts[1].isdigit():
                            case_date_key = parts[1]
                    except Exception:
                        case_date_key = ""
                    chosen = None
                    if case_date_key:
                        for p in outputs:
                            if str(getattr(p, "name", "")).endswith(f"_{case_date_key}.json"):
                                chosen = p
                                break
                    if chosen is None and outputs:
                        chosen = outputs[0]
                    if chosen is not None:
                        json_src_for_output = str(chosen)

                shutil.copy(json_src_for_output, os.path.join(path_output, "Pred_WMH_platform_json.json"))
                logging.info("Copy platform json to output: src=%s", json_src_for_output)
            except Exception:
                logging.warning("Failed to copy platform json to output.", exc_info=True)

            logging.info('!!! ' + ID +  ' post_wmh finish.')


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
    parser.add_argument('--Inputs', type=str, nargs='+', default = ['/data/4TB1/pipeline/chuan/example_input/01550089_20251117_MR_21411150023/T2FLAIR_AXI.nii.gz'], help='用於輸入的檔案')
    parser.add_argument('--DicomDir', type=str, nargs='+', default = ['/data/4TB1/pipeline/chuan/example_inputDicom/01550089_20251117_MR_21411150023/T2FLAIR_AXI/'], help='用於輸入的檔案')
    parser.add_argument('--Output_folder', type=str, default = '/data/4TB1/pipeline/chuan/example_output/01550089_20251117_MR_21411150023/',help='用於輸出結果的資料夾')    
    parser.add_argument('--input_json', type=str, default = '', help='followup-v3 平台輸入 json 內容或檔案路徑')
    parser.add_argument('--gpu_n', type=int, default = 0, help='使用哪一顆gpu')
    args = parser.parse_args()

    ID = str(args.ID)
    Inputs = args.Inputs  # 將列表合併為字符串，保留順序
    DicomDirs = args.DicomDir #對應的dicom資料夾，用來做dicom-seg
    path_output = str(args.Output_folder)
    if "--Output_folder" not in sys.argv:
        path_output = _env_path("RADX_OUTPUT_ROOT", path_output)
    input_json = str(args.input_json)
    print('DicomDirs:', DicomDirs)

    #下面設定各個路徑
    path_code = _env_path("RADX_CODE_ROOT", "/data/4TB1/pipeline/chuan/code/")
    path_process = _env_path("RADX_PROCESS_ROOT", "/data/4TB1/pipeline/chuan/process/")  #前處理dicom路徑(test case)
    path_nnunet_model = _env_path(
        "RADX_WMH_MODEL",
        "/data/4TB1/pipeline/chuan/code/nnUNet/nnUNet_results/Dataset015_DeepLacune/nnUNetTrainer__nnUNetPlans__2d",
    )

    path_processModel = os.path.join(path_process, 'Deep_WMH')  #前處理dicom路徑(test case)
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
    pipeline_wmh(ID, Inputs, DicomDirs, path_output, path_code, path_nnunet_model, path_processModel, path_json, path_log, gpu_n, input_json)

    # #最後再讀取json檔結果
    # with open(json_path_name) as f:
    #     data = json.load(f)

    # logging.info('Json!!! ' + str(data))