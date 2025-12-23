import json
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pydicom

from code_ai.pipeline.dicomseg import utils


def execute_rdx_platform_json(
    _id: str,
    path_root: pathlib.Path,
    model_id: str = "289fc383-34ff-46b7-bf0a-fbb22a104a18",
) -> pathlib.Path:
    """
    產生 vessel dilated 的 DICOM-SEG + JSON（給 RADAX 流程用）。

    注意：此 repo 內沒有 `code_ai.pipeline.rdx.build.*`（只有 schema），
    所以這裡改成自包含，不再依賴不存在的 builder。
    """

    path_dict = dict(
        path_dcms=path_root.joinpath("Dicom"),
        path_nii=path_root.joinpath("Image_nii"),
        path_dcmseg=path_root.joinpath("Dicom", "Dicom-Seg"),
    )
    path_dict["path_dcmseg"].mkdir(parents=True, exist_ok=True)

    # 以 MRA_BRAIN 當作 annotated series
    mra_dicom_dir = path_dict["path_dcms"].joinpath("MRA_BRAIN")
    _, image, first_dcm, source_images = utils.load_and_sort_dicom_files(mra_dicom_dir)

    input_study_instance_uid = str(getattr(first_dcm, "StudyInstanceUID", ""))
    input_series_instance_uid = str(getattr(first_dcm, "SeriesInstanceUID", ""))
    patient_id = str(getattr(first_dcm, "PatientID", ""))

    def _utc_iso_now_ms() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    # 讀 vessel mask（預設放在 Image_nii/Vessel.nii.gz）
    vessel_path = path_dict["path_nii"].joinpath("Vessel.nii.gz")
    vessel_arr = utils.get_array_to_dcm_axcodes(str(vessel_path))
    mask = (vessel_arr > 0).astype("uint8")

    # 產生單一 segment 的 DICOM-SEG（檔名需符合 pipeline 期待：MRA_BRAIN_Vessel_A1.dcm）
    from code_ai.pipeline.dicomseg.utils.base import get_dicom_seg_template, make_dicomseg_file  # local import

    label_dict = {1: {"SegmentLabel": "Vessel_A1", "color": "red"}}
    template_json = get_dicom_seg_template("VesselDilated", label_dict)
    seg_ds = make_dicomseg_file(mask, image, first_dcm, source_images, template_json)

    dcm_seg_path = path_dict["path_dcmseg"].joinpath("MRA_BRAIN_Vessel_A1.dcm")
    if dcm_seg_path.exists():
        dcm_seg_path.unlink()
    seg_ds.save_as(dcm_seg_path)

    # 重新讀檔，拿 SOP / Series UID（避免有些欄位在 save 後才完整）
    seg_ds_read = pydicom.dcmread(str(dcm_seg_path))

    output_json: Dict[str, Any] = {
        "inference_id": str(uuid.uuid4()),
        "inference_timestamp": _utc_iso_now_ms(),
        "input_study_instance_uid": [input_study_instance_uid] if input_study_instance_uid else [],
        "input_series_instance_uid": [input_series_instance_uid] if input_series_instance_uid else [],
        "model_id": model_id,
        "patient_id": patient_id,
        "detections": [
            {
                "annotated_series_instance_uid": input_series_instance_uid,
                "series_instance_uid": str(getattr(seg_ds_read, "SeriesInstanceUID", "")),
                "sop_instance_uid": str(getattr(seg_ds_read, "SOPInstanceUID", "")),
                "label": "Vessel",
            }
        ],
    }

    json_path = path_root.joinpath("rdx_vessel_dilated_json.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False)

    return json_path


def main(
    _id: str = "07807990_20250715_MR_21405290051",
    path_processID: pathlib.Path = pathlib.Path(
        "/data/4TB1/pipeline/chuan/process/Deep_Aneurysm/07807990_20250715_MR_21405290051/nnUNet"
    ),
    model_id: str = "289fc383-34ff-46b7-bf0a-fbb22a104a18",
):
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

    execute_rdx_platform_json(_id=_id, path_root=path_processID, model_id=model_id)








if __name__ == '__main__':
    main()