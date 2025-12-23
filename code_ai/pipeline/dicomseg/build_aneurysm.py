import json
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pydicom

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

    # 取得本次 inference 的 input series / study / patient（以 MRA_BRAIN 為準）
    mra_first_dcm = series_first_dcm_data_list[0]
    input_study_instance_uid = str(getattr(mra_first_dcm, "StudyInstanceUID", ""))
    input_series_instance_uid = str(getattr(mra_first_dcm, "SeriesInstanceUID", ""))
    patient_id = str(getattr(mra_first_dcm, "PatientID", ""))

    def _utc_iso_now_ms() -> str:
        # 例：2025-06-16T08:48:05.123Z
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _safe_int(x: Any, default: int = 0) -> int:
        try:
            return int(float(x))
        except Exception:
            return default

    def _calculate_angle_from_pred(pred: np.ndarray, mask_index: int) -> int:
        """
        以 fake 3D 的方式計算 pitch/yaw angle：
        - 只看該 mask_index 是否出現在某張 z-slice（避免像素數量造成偏移）
        - median_slice * 3 - 3
        """
        try:
            has_annotation = np.any(pred == mask_index, axis=(1, 2))
            slices_with_annotation = np.where(has_annotation)[0]
            if len(slices_with_annotation) == 0:
                return 0
            median_slice = int(np.median(slices_with_annotation))
            return int(median_slice * 3 - 3)
        except Exception:
            return 0

    # 為每個序列生成 DICOM-SEG 檔案和預測結果
    # - 每個 mask_index 會產一個 DICOM-SEG 檔案（MRA_BRAIN_A1.dcm / MIP_Pitch_A1.dcm / MIP_Yaw_A1.dcm）
    from code_ai.pipeline.dicomseg.aneurysm import use_create_dicom_seg_file, get_excel_to_pred_json  # local import 避免循環

    path_dict["path_dcmseg"].mkdir(parents=True, exist_ok=True)

    pred_result_list = []
    for series_name, dicom_data in zip(series_name_list, dicom_data_list):
        _, image, first_dcm, source_images = dicom_data
        nii_root = path_dict["path_nii"] if series_name == "MRA_BRAIN" else path_dict["path_reslice_nii"]
        result = use_create_dicom_seg_file(
            nii_root,
            series_name,
            path_dict["path_dcmseg"],
            image,
            first_dcm,
            source_images,
        )
        pred_result_list.append({"series_name": series_name, "data": result or []})

    # 讀取所有生成的 DICOM-SEG 檔案
    # 也同時保留 mask_index -> seg dataset 的對應，方便後續組 detections
    seg_ds_by_series: Dict[str, Dict[int, pydicom.FileDataset]] = {}
    for x in pred_result_list:
        series_name = x["series_name"]
        seg_ds_by_series[series_name] = {}
        for item in x["data"]:
            mask_index = int(item["mask_index"])
            dcm_seg_path = item["dcm_seg_path"]
            seg_ds_by_series[series_name][mask_index] = pydicom.dcmread(str(dcm_seg_path))

    # 從 Excel 檔案讀取預測結果並轉換為 JSON 格式
    pred_json_list = get_excel_to_pred_json(
        excel_file_path=str(path_dict['path_nii']
                            .parent
                            .joinpath("excel", "Aneurysm_Pred_list.xlsx")),
        intput_list=pred_nii_path_list
    )

    # 角度：從 MIP_Pitch/MIP_Yaw 的 pred nii 計算（若檔案不存在/沒有標註 → 0）
    pitch_pred_path = path_dict["path_reslice_nii"].joinpath("MIP_Pitch_pred.nii.gz")
    yaw_pred_path = path_dict["path_reslice_nii"].joinpath("MIP_Yaw_pred.nii.gz")
    pitch_pred: Optional[np.ndarray] = None
    yaw_pred: Optional[np.ndarray] = None
    if pitch_pred_path.exists():
        pitch_pred = utils.get_array_to_dcm_axcodes(str(pitch_pred_path))
    if yaw_pred_path.exists():
        yaw_pred = utils.get_array_to_dcm_axcodes(str(yaw_pred_path))

    # Excel pred：以 MRA_BRAIN 的那份當作主 detections 的來源（reformatted_series 會照抄它）
    excel_mra = next((x for x in pred_json_list if x.get("series_name") == "MRA_BRAIN"), None) or {"data": []}
    excel_mra_by_mask: Dict[int, Dict[str, Any]] = {
        _safe_int(d.get("mask_index")): d for d in excel_mra.get("data", [])
    }

    # 依照 MRA_BRAIN 實際生成的 seg 檔案來決定有哪些病灶
    mra_seg_map = seg_ds_by_series.get("MRA_BRAIN", {})
    mask_indices_sorted = sorted(mra_seg_map.keys())

    detections: List[Dict[str, Any]] = []
    for mask_index in mask_indices_sorted:
        seg_ds = mra_seg_map[mask_index]
        excel_row = excel_mra_by_mask.get(mask_index, {})

        pitch_angle = _calculate_angle_from_pred(pitch_pred, mask_index) if pitch_pred is not None else 0
        yaw_angle = _calculate_angle_from_pred(yaw_pred, mask_index) if yaw_pred is not None else 0

        detections.append(
            {
                "annotated_series_instance_uid": input_series_instance_uid,
                "series_instance_uid": str(getattr(seg_ds, "SeriesInstanceUID", "")),
                "sop_instance_uid": str(getattr(seg_ds, "SOPInstanceUID", "")),
                "label": str(excel_row.get("mask_name") or f"A{mask_index}"),
                "type": str(excel_row.get("type") or ""),
                "location": str(excel_row.get("location") or ""),
                "diameter": _safe_float(excel_row.get("diameter")),
                "main_seg_slice": _safe_int(excel_row.get("main_seg_slice")),
                "probability": _safe_float(excel_row.get("prob_max")),
                "pitch_angle": pitch_angle,
                "yaw_angle": yaw_angle,
                "mask_index": int(mask_index),
                "sub_location": str(excel_row.get("sub_location") or ""),
            }
        )

    def _build_reformatted_series(series_name: str, series_description: str) -> Dict[str, Any]:
        # 用 DICOM 第一張當作新生成的 pitch/yaw seriesInstanceUid
        idx = series_name_list.index(series_name)
        first_dcm = series_first_dcm_data_list[idx]
        annotated_uid = str(getattr(first_dcm, "SeriesInstanceUID", ""))

        seg_map = seg_ds_by_series.get(series_name, {})
        reformatted_dets: List[Dict[str, Any]] = []
        for d in detections:
            mask_index = int(d["mask_index"])
            seg_ds = seg_map.get(mask_index)
            reformatted_dets.append(
                {
                    "annotated_series_instance_uid": annotated_uid,
                    "series_instance_uid": str(getattr(seg_ds, "SeriesInstanceUID", "")) if seg_ds else "",
                    "sop_instance_uid": str(getattr(seg_ds, "SOPInstanceUID", "")) if seg_ds else "",
                    "label": d["label"],
                    "type": d["type"],
                    "location": d["location"],
                    "diameter": d["diameter"],
                    "main_seg_slice": d["main_seg_slice"],
                    "probability": d["probability"],
                    "mask_index": mask_index,
                    "sub_location": d["sub_location"],
                }
            )

        return {
            "series_instance_uid": annotated_uid,
            "series_description": series_description,
            "detections": reformatted_dets,
        }

    output_json: Dict[str, Any] = {
        "inference_id": str(uuid.uuid4()),
        "inference_timestamp": _utc_iso_now_ms(),
        "input_study_instance_uid": [input_study_instance_uid] if input_study_instance_uid else [],
        "input_series_instance_uid": [input_series_instance_uid] if input_series_instance_uid else [],
        "model_id": model_id,
        "patient_id": patient_id,
        "detections": detections,
        "reformatted_series": [
            _build_reformatted_series("MIP_Pitch", "mip_pitch"),
            _build_reformatted_series("MIP_Yaw", "mip_yaw"),
        ],
    }

    # 確保輸出目錄存在
    path_root.mkdir(exist_ok=True, parents=True)

    # 儲存平台 JSON 檔案
    platform_json_path = path_root.joinpath("rdx_aneurysm_pred_json.json")
    with open(platform_json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False)

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