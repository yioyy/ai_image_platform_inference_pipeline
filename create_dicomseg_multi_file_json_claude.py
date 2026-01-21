# -*- coding: utf-8 -*-
"""

根據手邊資料製作多組dicom-seg影像
由於國庭的cornerstone有bug，不能讀取出mask id，所以必須一個mask一個dicom-seg
範例case包含為CMB

@author: sean
"""

import argparse
import datetime
import enum
import os
import pathlib
from typing import Dict, List, Any, Union, Tuple, Optional

import numpy as np
import pydicom
import nibabel as nib
import SimpleITK as sitk
import matplotlib.colors as mcolors
import pydicom_seg
from pydantic import BaseModel, PositiveInt, field_validator
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir
from skimage.measure import regionprops_table

#from code_ai.pipeline import pipeline_parser

# 載入範例文件只需執行一次，改為在函數外
# EXAMPLE_FILE = 'SEG_20230210_160056_635_S3.dcm'

EXAMPLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'resource', 'SEG_20230210_160056_635_S3.dcm')
DCM_EXAMPLE = pydicom.dcmread(EXAMPLE_FILE)

# // 目前 Orthanc 自動同步機制的 group 應為 44
GROUP_ID = os.getenv("GROUP_ID",'47')


class SeriesTypeEnum(enum.Enum):
    MRA_BRAIN   = '1'
    TOF_MRA     = '1'
    MIP_Pitch   = '2'
    MIP_Yaw     = '3'
    T1BRAVO_AXI = '4'
    T1FLAIR_AXI = '5'
    T2FLAIR_AXI = '6'
    SWAN        = '7'
    DWI0        = '8'
    DWI1000     = '9'
    ADC         = '10'

    @classmethod
    def to_list(cls) -> List[Union[enum.Enum]]:
        return list(map(lambda c: c, cls))




class InstanceRequest(BaseModel):
    # dicom id
    sop_instance_uid :str
    # 應與 ImageOrientationPatient 和 ImagePositionPatient 有關
    projection       :str


class SeriesRequest(BaseModel):
    # dicom
    series_instance_uid :str
    instance : List[InstanceRequest]
    pass


class SortedRequest(BaseModel):
    study_instance_uid: str
    series            : List[SeriesRequest]
    pass


#用資料夾名稱比較???
class StudySeriesRequest(BaseModel):
    # dicom
    series_type         : str
    series_instance_uid : str

    @field_validator('series_type', mode='before')
    @classmethod
    def extract_series_type(cls, value):
        if value is None:
            return '1'
        if isinstance(value, str):
            series_type_enum_list: List[SeriesTypeEnum] = SeriesTypeEnum.to_list()
            for series_type_enum in series_type_enum_list:
                if value == series_type_enum.name:
                    return series_type_enum.value
        return value


class StudyRequest(BaseModel):
    group_id           :str #= GROUP_ID
    # "2017-12-25"
    study_date         :datetime.date
    # M F
    gender             :str
    # 99 56
    age                :int
    # Brain MRI
    study_name         :str
    # Mock
    patient_name       :str
    # 表幾顆腫瘤， 1.... 99
    aneurysm_lession   :PositiveInt = 0
    # 表示 AI 訓練結果 , 1:成功, 2:進行中, 3:失敗
    aneurysm_status    :PositiveInt = 1
    resolution_x       :PositiveInt = 256
    resolution_y       :PositiveInt = 256
    study_instance_uid :str
    patient_id         :str
    series             :List[StudySeriesRequest]


    @field_validator('patient_name', mode='before')
    @classmethod
    def extract_patient_name_number(cls, value):
        if value is None:
            return ''
        # 如果是從DICOM獲取的值
        if isinstance(value, str):

            return value
        else:
            return str(value)

    @field_validator('study_date', mode='before')
    @classmethod
    def parse_date(cls, value):
        try:
            if isinstance(value, str) and len(value) == 8:
                return datetime.date(int(value[:4]), int(value[4:6]), int(value[6:8]))
            else:
                return datetime.datetime.now().date()
        except:
            return datetime.datetime.now().date()

    @field_validator('age', mode='before')
    @classmethod
    def extract_age_number(cls, value):
        if value is None:
            return None

        # 如果是從DICOM獲取的值
        if isinstance(value, str):
            # 移除所有非數字字符
            digits_only = ''.join(c for c in value if c.isdigit())
            if digits_only:
                return int(digits_only)
        return value


class MaskInstanceRequest(BaseModel):
    mask_index   : str
    mask_name    : str
    # diameter,type,location,sub_location,checked 在第一次上傳後寫入 db , 之後可從前端介面手動更改
    diameter     : str
    type         : str = 'saccular'
    location     : str = 'M'
    sub_location : str = '2'
    prob_max     : str
    # bool , 0 或 1
    checked      : str = '1'
    # bool , 0 或 1
    is_ai        : str = '1'

    seg_series_instance_uid : str
    seg_sop_instance_uid    : str

    # 對應到 dicom image Key porint uid
    dicom_sop_instance_uid  : str

    main_seg_slice          : int = 1
    is_main_seg             : int = 0

    @field_validator('diameter', mode='before')
    @classmethod
    def extract_diameter_number(cls, value):
        if value is None:
            return '0.0'
        if isinstance(value, np.ndarray):
            return str(value.mean().item())
        # 如果是從DICOM獲取的值
        if isinstance(value, (float, int)):
            return str(value)
        return value

    @field_validator('prob_max', mode='before')
    @classmethod
    def extract_prob_max(cls, value):
        if value is None:
            return '1.0'
        if isinstance(value, (float, int)):
            return str(round(value, 6))
        return '1.0'

class MaskSeriseRequest(BaseModel):
    # dicom image uid
    series_instance_uid : str
    # 目前僅 Aneu 的 TOF_MRA:1 , Pitch: 2 , Yaw: 3 , 未來有其他 Series 再加
    series_type         : str
    instances           : List[MaskInstanceRequest]


    @field_validator('series_type', mode='before')
    @classmethod
    def extract_series_type(cls, value):
        if value is None:
            return '1'
        if isinstance(value, str):
            series_type_enum_list: List[SeriesTypeEnum] = SeriesTypeEnum.to_list()
            for series_type_enum in series_type_enum_list:
                if value == series_type_enum.name:
                    return series_type_enum.value
        return value

class MaskRequest(BaseModel):
    study_instance_uid : str
    group_id           : str #PositiveInt = GROUP_ID
    series             : List[MaskSeriseRequest]


class AITeamRequest(BaseModel):
    study   : Optional[StudyRequest]  = None
    sorted  : Optional[SortedRequest] = None
    mask    : Optional[MaskRequest]   = None


def get_diameter(mask_np,source_image,) -> float:
    try:
        spacing = source_image.get((0x0028, 0x0030)).value
        props = regionprops_table(mask_np, properties=('label', 'bbox'))
        # return max((props['bbox-2'] - props['bbox-0']) * spacing[0],(props['bbox-3'] - props['bbox-1']) * spacing[1])
        return (((props['bbox-2'] - props['bbox-0']) * spacing[0] + (props['bbox-3'] - props['bbox-1']) * spacing[1]) / 2.0).item()
    except:
        return 0.0


def make_mask_json(source_images:List[Union[FileDataset, DicomDir]],
                   sorted_dcms  :List[Union[str, pathlib.Path]],
                   dcm_seg_path:Union[str,pathlib.Path],
                   mask_index :int ,
                   group_id:str = GROUP_ID,) ->  MaskRequest:
    with open(dcm_seg_path, 'rb') as f:
        dcm_seg = pydicom.read_file(f)
    # Referenced Series
    referenced_series_list = dcm_seg.get((0x0008, 0x1115)).value
    # print(referenced_series_list)
    mask_np = dcm_seg.pixel_array
    # (0028,0030)	Pixel Spacing	0.9375\0.9375
    mask_dict_list = []
    mask_dict = dict()
    series_instance_uid = dcm_seg.get((0x0020, 0x000E)).value
    for index, referenced_series in enumerate(referenced_series_list):
        instance_list = referenced_series[(0x0008, 0x114a)].value
        for jj, instance in enumerate(instance_list):
            instance_dict = dict()
            sop_instance_uid    = instance.get((0x008, 0x1155)).value

            filtered_items  = next(filter(lambda x: x[1].get((0x0008, 0x0018)).value == sop_instance_uid ,
                                          enumerate(source_images)))
            mask_name    = dcm_seg.get((0x0008, 0x103E)).value
            diameter     = get_diameter(mask_np[index], source_images[filtered_items[0]])
            type         = "saccular"
            location     = "M"
            sub_location = "2"
            checked      = "1"
            prob_max     = "1"
            is_ai        = "1"
            series_type  = os.path.basename(os.path.dirname(sorted_dcms[filtered_items[0]]))
            main_seg_slice = index
            instance_dict.update(dict(
                mask_index=mask_index,
                mask_name=mask_name,
                diameter=diameter,
                type=type,
                location=location,
                sub_location=sub_location,
                checked=checked,
                prob_max=prob_max,
                is_ai=is_ai,
                series_type=series_type,
                sop_instance_uid = sop_instance_uid,
                series_instance_uid = series_instance_uid,
                main_seg_slice= main_seg_slice))
            mask_dict_list.append(instance_dict)
            break
        if index == 0:
            study_instance_uid = dcm_seg.get((0x0020, 0x000D)).value
            mask_dict.update(dict(study_instance_uid=study_instance_uid,
                                  group_id=group_id,
                                  instances=mask_dict_list))
        break

    mask_dict.update(dict(instances=mask_dict_list))
    return MaskRequest.model_validate(mask_dict)


def make_study_series_json(source_images : List[Union[FileDataset, DicomDir]],
                           sorted_dcms   : List[Union[pathlib.Path, str]]) -> List[StudySeriesRequest]:
    """
    Create a JSON representation of sorted DICOM instances for a study.

    Args:
        source_images: List of DICOM image datasets

    Returns:
        StudySeriesRequest: Validated sorted request data
    """
    instance_list = []
    series_dict = dict()
    series_instance_uid_dict = {}
    for index, dicom_ds in enumerate(source_images):
        series_instance_uid = dicom_ds.get((0x0020, 0x000E)).value
        if series_instance_uid_dict.get(series_instance_uid) is None:
            series_type = os.path.basename(os.path.dirname(sorted_dcms[index]))
            series_instance_uid_dict.update({series_instance_uid:series_type})
        else:
            continue

    for series_instance_uid, series_type in series_instance_uid_dict.items():
        instance_dict = dict()

        # Update instance dictionary
        instance_dict.update(dict(
            series_type=series_type,
            series_instance_uid=series_instance_uid
        ))
        instance_list.append(StudySeriesRequest.model_validate(instance_dict))

    # Validate and return the sorted request
    return instance_list


def make_study_json(source_images:List[Union[FileDataset, DicomDir]], study_series_request_list:List[StudySeriesRequest], group_id:str = GROUP_ID) -> AITeamRequest:
    instance_list = []
    series_list = []
    at_team_dict = dict()
    series_dict = dict()
    sorted_dict = dict()
    study_dict  = dict()
    for index, dicom_ds in enumerate(source_images):
        instance_dict = dict()
        sop_instance_uid = dicom_ds.get((0x0008, 0x0018)).value
        study_instance_uid = dicom_ds.get((0x0020, 0x000D)).value
        series_instance_uid = dicom_ds.get((0x0020, 0x000E)).value
        study_date = dicom_ds.get((0x0008, 0x0020)).value
        gender = dicom_ds.get((0x0010, 0x0040)).value
        age = dicom_ds.get((0x0010, 0x1010)).value
        study_name = dicom_ds.get((0x0008, 0x1030)).value
        patient_name = dicom_ds.get((0x0010, 0x0010)).value
        resolution_x = dicom_ds.get((0x0028, 0x0010)).value
        resolution_y = dicom_ds.get((0x0028, 0x0011)).value
        patient_id = dicom_ds.get((0x0010, 0x0020)).value
        image_osition_patient = dicom_ds.get((0x0020, 0x0032)).value[-1]
        # (0020,0032)	Image Position Patient	-14.7431\-143.248\98.5056
        instance_dict.update(dict(sop_instance_uid=sop_instance_uid,
                                  projection=str(image_osition_patient)))

        instance_list.append(instance_dict)
        if index == 0:
            series_dict.update(dict(series_instance_uid=series_instance_uid,
                                    instance=instance_list))
            series_list.append(series_dict)
            sorted_dict.update(dict(study_instance_uid=study_instance_uid,
                                    series=series_list
                                    ))
            study_dict.update(dict(group_id=group_id,
                                   study_instance_uid=study_instance_uid,
                                   study_date=study_date,
                                   gender=gender,
                                   age=age,
                                   study_name=study_name,
                                   patient_name=patient_name,
                                   resolution_x=resolution_x,
                                   resolution_y=resolution_y,
                                   patient_id=patient_id,
                                   series = study_series_request_list))
            at_team_dict.update(dict(study=study_dict,
                                     sorted=sorted_dict,
                                     mask=None))

    at_team_request = AITeamRequest.model_validate(at_team_dict)
    return at_team_request

def compute_orientation(init_axcodes, final_axcodes):
    """
    A thin wrapper around ``nib.orientations.ornt_transform``

    :param init_axcodes: Initial orientation codes
    :param final_axcodes: Target orientation codes
    :return: orientations array, start_ornt, end_ornt
    """
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)

    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)

    return ornt_transf, ornt_init, ornt_fin


def do_reorientation(data_array, init_axcodes, final_axcodes):
    """
    source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation
    Performs the reorientation (changing order of axes)

    :param data_array: 3D Array to reorient
    :param init_axcodes: Initial orientation
    :param final_axcodes: Target orientation
    :return data_reoriented: New data array in its reoriented form
    """
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array

    return nib.orientations.apply_orientation(data_array, ornt_transf)


def get_dicom_seg_template(model: str, label_dict: Dict) -> Dict:
    """創建DICOM-SEG模板"""
    unique_labels = list(label_dict.keys())

    segment_attributes = []
    for idx in unique_labels:
        name = label_dict[idx]["SegmentLabel"]
        rgb_rate = mcolors.to_rgb(label_dict[idx]["color"])
        rgb = [int(y * 255) for y in rgb_rate]

        segment_attribute = {
            "labelID": int(idx),
            "SegmentLabel": name,
            "SegmentAlgorithmType": "MANUAL",
            "SegmentAlgorithmName": "SHH",
            "SegmentedPropertyCategoryCodeSequence": {
                "CodeValue": "M-01000",
                "CodingSchemeDesignator": "SRT",
                "CodeMeaning": "Morphologically Altered Structure",
            },
            "SegmentedPropertyTypeCodeSequence": {
                "CodeValue": "M-35300",
                "CodingSchemeDesignator": "SRT",
                "CodeMeaning": "Embolus",
            },
            "recommendedDisplayRGBValue": rgb,
        }
        segment_attributes.append(segment_attribute)

    template = {
        "ContentCreatorName": "Reader1",
        "ClinicalTrialSeriesID": "Session1",
        "ClinicalTrialTimePointID": "1",
        "SeriesDescription": model,
        "SeriesNumber": "300",
        "InstanceNumber": "1",
        "segmentAttributes": [segment_attributes],
        "ContentLabel": "SEGMENTATION",
        "ContentDescription": "SHH",
        "ClinicalTrialCoordinatingCenterName": "SHH",
        "BodyPartExamined": "",
    }
    return template


def load_and_sort_dicom_files(path_dcms: str) -> tuple[List[Any], Any, FileDataset | DicomDir, list[FileDataset | DicomDir]]:
    """載入並排序DICOM檔案，只需執行一次"""
    # 讀取DICOM文件
    reader = sitk.ImageSeriesReader()
    dcms = sorted(reader.GetGDCMSeriesFileNames(path_dcms))

    # 讀取所有切片
    slices = [pydicom.dcmread(dcm) for dcm in dcms]

    # 依照切片位置排序
    slice_dcm = []
    for (slice_data, dcm_slice) in zip(slices, dcms):
        IOP = np.array(slice_data.get((0x0020, 0x0037)).value)
        IPP = np.array(slice_data.get((0x0020, 0x0032)).value)
        normal = np.cross(IOP[0:3], IOP[3:])
        projection = np.dot(IPP, normal)
        slice_dcm.append({"d": projection, "dcm": dcm_slice})

    # 排序切片
    slice_dcms = sorted(slice_dcm, key=lambda i: i['d'])
    sorted_dcms = [y['dcm'] for y in slice_dcms]

    # 讀取影像資料
    reader.SetFileNames(sorted_dcms)
    image = reader.Execute()

    # 讀取第一張DICOM影像以獲取標頭資訊
    first_dcm = pydicom.dcmread(sorted_dcms[0], force=True)

    # 預載所有DICOM檔案（但不載入像素資料以節省記憶體）
    source_images = [pydicom.dcmread(x, stop_before_pixels=True) for x in sorted_dcms]

    return sorted_dcms, image, first_dcm, source_images



def transform_mask_for_dicom_seg(mask: np.ndarray) -> np.ndarray:
    """將遮罩轉換為DICOM-SEG所需的格式"""
    # 轉換格式：(y,x,z) -> (z,x,y)
    segmentation_data = mask.transpose(2, 0, 1).astype(np.uint8)
    # 轉換格式：(z,x,y) -> (z,y,x)
    segmentation_data = np.swapaxes(segmentation_data, 1, 2)
    # 翻轉y軸和x軸以符合DICOM座標系統
    segmentation_data = np.flip(segmentation_data, 1)
    segmentation_data = np.flip(segmentation_data, 2)

    return segmentation_data


def make_dicomseg_file(mask: np.ndarray,
                       image: sitk.Image,
                       first_dcm: pydicom.FileDataset,
                       source_images: List[pydicom.FileDataset],
                       template_json: Dict) -> pydicom.FileDataset:
    """製作DICOM-SEG檔案"""
    # 創建模板
    template = pydicom_seg.template.from_dcmqi_metainfo(template_json)

    # 設定寫入器
    writer = pydicom_seg.MultiClassWriter(
        template=template,
        inplane_cropping=False,
        skip_empty_slices=True,
        skip_missing_segment=True,
    )

    # 轉換遮罩格式
    #segmentation_data = transform_mask_for_dicom_seg(mask)
    segmentation_data = mask

    # 創建SimpleITK影像
    segmentation = sitk.GetImageFromArray(segmentation_data)
    segmentation.CopyInformation(image)

    # 產生DICOM-SEG檔案
    dcm_seg = writer.write(segmentation, source_images)

    # 從示範檔案和第一張DICOM影像中複製相關資訊
    dcm_seg[0x10, 0x0010].value = first_dcm[0x10, 0x0010].value  # Patient's Name
    dcm_seg[0x20, 0x0011].value = first_dcm[0x20, 0x0011].value  # Series Number

    # 複製更多元數據
    dcm_seg[0x5200, 0x9229].value = DCM_EXAMPLE[0x5200, 0x9229].value
    dcm_seg[0x5200, 0x9229][0][0x20, 0x9116][0][0x20, 0x0037].value = first_dcm[0x20, 0x0037].value
    dcm_seg[0x5200, 0x9229][0][0x28, 0x9110][0][0x18, 0x0050].value = first_dcm[0x18, 0x0050].value
    dcm_seg[0x5200, 0x9229][0][0x28, 0x9110][0][0x18, 0x0088].value = first_dcm[0x18, 0x0088].value
    dcm_seg[0x5200, 0x9229][0][0x28, 0x9110][0][0x28, 0x0030].value = first_dcm[0x28, 0x0030].value

    return dcm_seg


def process_prediction_mask(
        pred_data: np.ndarray,
        path_dcms: str,
        series_name: str,
        output_folder: pathlib.Path
) -> Optional[AITeamRequest]:
    """處理預測遮罩並生成DICOM-SEG文件，at_team_request"""
    # 只需載入一次DICOM檔案
    sorted_dcms, image, first_dcm, source_images = load_and_sort_dicom_files(path_dcms)
    at_team_request = make_study_json(source_images)
    # # 處理每個分割區域
    mask_request_list = []
    pred_data_unique = np.unique(pred_data)
    if len(pred_data_unique) < 1:
        return
    else:
        pred_data_unique = pred_data_unique[1:]
    pred_data_unique_len = len(pred_data_unique)
    for index, i in enumerate(pred_data_unique):
    # for i in range(pred_data_max):
        # 創建單一區域的遮罩
        mask = np.zeros_like(pred_data)
        mask[pred_data == i] = 1

        # 如果遮罩中有值，才創建DICOM-SEG
        if np.sum(mask) > 0:
            # 創建標籤字典
            label_dict = {1: {'SegmentLabel': f'A{i}', 'color': 'red'}}

            # 創建模板
            template_json = get_dicom_seg_template(series_name, label_dict)

            # 產生DICOM-SEG
            dcm_seg = make_dicomseg_file(
                mask.astype('uint8'),
                image,
                first_dcm,
                source_images,
                template_json
            )
            # 保存DICOM-SEG
            dcm_seg_filename = f'{series_name}_{label_dict[1]["SegmentLabel"]}.dcm'
            dcm_seg_path = output_folder.joinpath(dcm_seg_filename)
            dcm_seg.save_as(dcm_seg_path)
            print(f" " * 100,end='\r')
            print(f"{index+1}/{pred_data_unique_len} Saved: {dcm_seg_path}",end='\r')
            temp_mask_request = make_mask_json(source_images=source_images,
                                               sorted_dcms=sorted_dcms,
                                               dcm_seg_path=dcm_seg_path,
                                               mask_index= int(i),
                                               )
            mask_request_list.append(temp_mask_request)
            # break
    first_mask_request = mask_request_list[0]
    mask_instances_list = []
    for x in mask_request_list:
        mask_instances_list.extend(x.instances)
    mask_request =  MaskRequest(study_instance_uid=first_mask_request.study_instance_uid,
                                group_id=first_mask_request.group_id,
                                instances = mask_instances_list)
    at_team_request.mask = mask_request
    at_team_request.study.aneurysm_lession = pred_data_unique.shape[0]
    print()
    return at_team_request


def main():
    """主函數"""
    parser = pipeline_parser()
    # parser.add_argument('--OutputDicomSegFolder', type=str,
    #                     default='/mnt/e/rename_nifti_20250509/',
    #                     help='用於輸出結果的資料夾')

    args = parser.parse_args()
    ID = args.ID
    path_dcms = pathlib.Path(args.InputsDicomDir)
    path_nii = pathlib.Path(args.Inputs[0])
    path_dcmseg = pathlib.Path(args.Output_folder)

    # 取得系列名稱
    series = path_nii.name.split('.')[0]

    # 載入預測數據
    pred_nii = nib.load(path_nii)
    pred_data = np.array(pred_nii.dataobj)

    pred_nii_obj_axcodes = tuple(nib.aff2axcodes(pred_nii.affine))
    new_nifti_array = do_reorientation(pred_data, pred_nii_obj_axcodes, ('S', 'P', 'L'))

    # 創建輸出子目錄
    series_folder = path_dcmseg.joinpath(f'{ID}')
    if series_folder.is_dir():
        series_folder.mkdir(exist_ok=True, parents=True)
    else:
        series_folder.parent.mkdir(parents=True, exist_ok=True)

    # 處理預測遮罩
    at_team_request :AITeamRequest = process_prediction_mask(new_nifti_array, str(path_dcms), series, series_folder)
    platform_json_path = series_folder.joinpath(path_nii.name.replace('.nii.gz', '_platform_json.json'))
    print('platform_json_path',platform_json_path)
    at_team_request.mask.instances = sorted(at_team_request.mask.instances, key=lambda instance: instance.mask_index)
    with open(platform_json_path, 'w') as f:
        f.write(at_team_request.model_dump_json())
    print("Processing complete!")


if __name__ == '__main__':
    main()
