import argparse
import os
import pathlib
from code_ai.pipeline.rdx.build.vessel import VesselDilatedBuilder


def main(_id='07807990_20250715_MR_21405290051',
         path_processID=pathlib.Path('/data/4TB1/pipeline/chuan/process/Deep_Aneurysm/07807990_20250715_MR_21405290051/nnUNet'),):
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

    # print('path_processID',os.environ.get('path_processID'))
    # platform_json = VesselDilatedBuilder.execute_rdx_platform_json(args.id, path_processID)
    # print('platform_json',platform_json)
    # execute_rdx_platform_json(_id=args.id,
    #                           path_root=path_processID)

    VesselDilatedBuilder.execute_rdx_platform_json(_id=_id, path_root=path_processID)








if __name__ == '__main__':
    main()