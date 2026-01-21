import argparse
import dataclasses
import os
import pathlib
from typing import Union, List
import itertools
import pydicom

from code_ai.pipeline.rdx.build.aneurysm import AneurysmDetectionBuilder
from code_ai.pipeline.dicomseg import utils

# @dataclasses
# class AneurysmPath:
#     id                       : str
#     intput_path              : List[Union[str, pathlib.Path,]]
#     output_folder_path       : pathlib.Path
#     intput_dicom_folder_path : pathlib.Path
#     process_model_path       : pathlib.Path
#     process_model_path_id    : pathlib.Path
#     path_dcms                : pathlib.Path  = process_model_path_id.joinpath("Dicom")
#     path_nii                 : pathlib.Path  = process_model_path_id.joinpath("Image_nii")
#     path_reslice_nii         : pathlib.Path  = process_model_path_id.joinpath("Image_reslice")
#     path_dcmseg              : pathlib.Path  = path_dcms.joinpath("Dicom-Seg")


def execute_rdx_platform_json(_id: int, path_root: pathlib.Path,
                              model_id: str = '924d1538-597c-41d6-bc27-4b0b359111cf'):
    """
    執行 RDX 平台 JSON 生成流程，處理動脈瘤檢測的 DICOM 和預測結果

    Args:
        _id: 患者或案例 ID
        path_root: 資料根目錄路徑
        model_id: 模型識別碼，預設為動脈瘤檢測模型 ID
    """

    # 定義所有必要的資料夾路徑
    path_dict = dict(
        path_dcms=path_root.joinpath("Dicom"),  # DICOM 原始檔案目錄
        path_nii=path_root.joinpath("Image_nii"),  # NIfTI 格式影像目錄
        path_reslice_nii=path_root.joinpath("Image_reslice"),  # 重切影像目錄
        path_dcmseg=path_root.joinpath("Dicom", "Dicom-Seg")  # DICOM-SEG 輸出目錄
    )

    # 初始化動脈瘤檢測 JSON 建構器
    aneurysm_platform_json_builder = AneurysmDetectionBuilder()

    # 定義三個序列名稱：腦部 MRA、俯仰角 MIP、偏航角 MIP
    series_name_list = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw']

    # 建立每個序列的 DICOM 資料夾路徑列表
    path_dcms_mar_folder_path_list = [path_dict['path_dcms'].joinpath(x) for x in series_name_list]

    # 建立預測 NIfTI 檔案路徑列表
    # MRA_BRAIN 使用 Image_nii 目錄，其他使用 Image_reslice 目錄
    pred_nii_path_list = [{"series_name": x,
                           "pred_nii_path": str(path_dict['path_nii'].joinpath("Pred.nii.gz"))
                           if x == 'MRA_BRAIN'
                           else str(path_dict['path_reslice_nii'].joinpath("{}_pred.nii.gz".format(x)))
                           }
                          for x in series_name_list]

    # 載入並排序每個序列的 DICOM 檔案
    #             - sorted_dcms: List of sorted DICOM file paths
    #             - image: SimpleITK image object
    #             - first_dcm: First DICOM dataset
    #             - source_images: List of all DICOM datasets (without pixel data)
    dicom_data_list = [utils.load_and_sort_dicom_files(x) for x in path_dcms_mar_folder_path_list]

    # 提取每個序列的第一個 DICOM 檔案（索引 [2] 表示排序後的第一個檔案 first_dcm）
    series_first_dcm_data_list = [x[2] for x in dicom_data_list]

    # 為每個序列生成 DICOM-SEG 檔案和預測結果
    pred_result_list = [{"series_name": x[0],
                         "data": aneurysm_platform_json_builder.use_create_dicom_seg_file(
                             path_dict['path_nii'] if x[0] == 'MRA_BRAIN' else path_dict['path_reslice_nii'],
                             x[0],                     # sorted_dcms
                             path_dict['path_dcmseg'],
                             *x[1][1:]  # 解包 DICOM 資料（跳過第一個元素）
                         )}
                        for x in zip(series_name_list, dicom_data_list)]

    # 讀取所有生成的 DICOM-SEG 檔案
    dcm_seg_path_list = [list(map(lambda xx: pydicom.read_file(xx['dcm_seg_path']), x['data']))
                         for x in pred_result_list]

    # 從 Excel 檔案讀取預測結果並轉換為 JSON 格式
    pred_json_list = aneurysm_platform_json_builder.get_excel_to_pred_json(
        excel_file_path=str(path_dict['path_nii']
                            .parent
                            .joinpath("excel", "Aneurysm_Pred_list.xlsx")),
        intput_list=pred_nii_path_list
    )

    # 合併俯仰角和偏航角的角度資訊到主要預測結果
    pred_json_merge = aneurysm_platform_json_builder.merge_pitch_yaw_angle(pred_json_list)

    # 使用建構器模式組裝最終的平台 JSON
    aneurysm_platform_json = (aneurysm_platform_json_builder
                              .set_patient_info(series_first_dcm_data_list)                     # 設定患者資訊
                              .set_model_id(model_id)                                         # 設定模型 ID
                              .set_detections(dcm_seg_path_list[0], pred_json_merge['data'])  # 設定檢測結果
                              .build()  # 建構最終 JSON
                              )

    # 確保輸出目錄存在
    output_series_folder = path_root
    if output_series_folder.is_dir():
        output_series_folder.mkdir(exist_ok=True, parents=True)
    else:
        output_series_folder.parent.mkdir(parents=True, exist_ok=True)

    # 儲存平台 JSON 檔案
    platform_json_path = output_series_folder.joinpath(path_root.joinpath('rdx_aneurysm_pred_json.json'))
    with open(platform_json_path, 'w') as f:
        f.write(aneurysm_platform_json.model_dump_json())

    print("Processing complete!")


def main(_id='07807990_20250715_MR_21405290051',
         path_processID=pathlib.Path('/data/4TB1/pipeline/chuan/process/Deep_Aneurysm/07807990_20250715_MR_21405290051/nnUNet'),
         model_id: str = '924d1538-597c-41d6-bc27-4b0b359111cf'):
    """
    Main function to process command line arguments and execute the pipeline.

    This function:
    """
    # Parse command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--id', type=str, default='01901124_20250617_MR_21404020048',
    #                     help='目前執行的case的patient_id or study id')

    # parser.add_argument('--inputs', type=str, nargs='+',
    #                     default=['/mnt/e/pipeline/test_data/01901124_20250617_MR_21404020048/Image_nii/Pred.nii.gz '],
    #                     help='用於輸入的檔案')
    # parser.add_argument('--output_folder', type=str, default='/mnt/e/pipeline/新增資料夾/01901124_20250617_MR_21404020048/output',
    #                     help='用於輸出結果的資料夾')
    # parser.add_argument('--input_dicom_folder', type=str,
    #                     default='/mnt/e/pipeline/test_data/01901124_20250617_MR_21404020048/Dicom/MRA_BRAIN',
    #                     help='用於輸入的檔案')
    # path_processModel = "/mnt/e/pipeline/test_data"
    # args = parser.parse_args()

    # path_root = pathlib.Path(path_processModel)
    # path_processID = path_root.joinpath( args.id)
    execute_rdx_platform_json(
        _id=_id, path_root=path_processID, model_id=model_id)







if __name__ == '__main__':
    main()