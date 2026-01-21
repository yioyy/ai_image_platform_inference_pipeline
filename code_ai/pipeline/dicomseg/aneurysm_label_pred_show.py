# -*- coding: utf-8 -*-
"""
DICOM-SEG Generator

This script creates multiple DICOM-SEG images from available mask data.
Due to a bug in Cornerstone (a JavaScript library for medical imaging)
that prevents reading mask IDs, each mask must be stored in a separate DICOM-SEG file.
The example cases provided include CMB (Cerebral Microbleeds).

This module handles:
1. Loading DICOM series and mask data (in NIfTI format)
2. Converting mask data to DICOM-SEG format
3. Creating required JSON metadata for each mask
4. Saving DICOM-SEG files and associated metadata

@author: sean
"""
import sys
sys.path.append('/data/4TB1/pipeline/chuan/code') #在程式碼中動態加路徑（開發時方便）

import argparse
import json
import os
import pathlib
from typing import Dict, List, Any, Union, Optional
from typing_extensions import Self

import pandas as pd

import numpy as np
import pydicom
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir
import nibabel as nib
from code_ai.pipeline.dicomseg import utils
from code_ai.pipeline.dicomseg.schema.base import SeriesTypeEnum, ModelTypeEnum
from code_ai.pipeline.dicomseg.schema.base import StudyRequest, StudySeriesRequest, StudyModelRequest
from code_ai.pipeline.dicomseg.schema.aneurysm import AneurysmMaskRequest, AneurysmMaskSeriesRequest, AneurysmMaskInstanceRequest
from code_ai.pipeline.dicomseg.schema.aneurysm import AneurysmAITeamRequest, SortedRequest
from code_ai.pipeline.dicomseg.schema.aneurysm import AneurysmAITeam2Request, AneurysmMask2Request
from code_ai.pipeline.dicomseg.schema.aneurysm import AneurysmMaskSeries2Request, AneurysmMaskModel2Request


from code_ai.pipeline.dicomseg.base import PlatformJSONBuilder, ReviewBasePlatformJSONBuilder
from code_ai import load_dotenv
load_dotenv()



class AneurysmPlatformJSONBuilder(PlatformJSONBuilder[AneurysmAITeamRequest]):
    model_class = AneurysmAITeamRequest
    model_type  = ModelTypeEnum.Aneurysm.value
    series_type = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._series_dict: Optional[Dict[Union[SeriesTypeEnum,str],List[Union[FileDataset, DicomDir]]]] = {}


    def reset(self):
        super().reset()
        self._series_dict: Optional[Dict[SeriesTypeEnum, List[Union[FileDataset, DicomDir]]]] = {}

    def set_series_type(self, series_type:Union[SeriesTypeEnum,str],
                              source_images: List[Union[FileDataset, DicomDir]],
                              ) -> Self :
        if isinstance(series_type,SeriesTypeEnum):
            self._series_dict.update({series_type.name:source_images})
        else:
            self._series_dict.update({series_type: source_images})
        return self

    def build_mask_instance(self, source_images: List[Union[FileDataset, DicomDir]],
                            dcm_seg: Union[FileDataset, DicomDir], *args, **kwargs) -> Self:
        main_seg_slice = kwargs['main_seg_slice']
        mask_instance_dict = dict()
        # Extract UIDs from the DICOM-SEG and source images
        seg_sop_instance_uid = dcm_seg.get((0x008, 0x0018)).value
        seg_series_instance_uid = dcm_seg.get((0x020, 0x000E)).value
        dicom_sop_instance_uid = source_images[main_seg_slice].get((0x008, 0x0018)).value
        # Extract mask name from DICOM-SEG Series Description
        # e.g. (0008,103E) Series Description: aneurysm AI

        mask_index     = kwargs.get('mask_index', 1)
        main_seg_slice = kwargs.get('main_seg_slice', "")
        diameter       = kwargs.get('diameter', '0.0')
        _type          = kwargs.get('type', '')
        location       = kwargs.get('location', 'M')
        sub_location   = kwargs.get('sub_location', '')
        prob_max       = kwargs.get('prob_max', '1.0')

        # Create mask instance dictionary with all required metadata
        mask_instance_dict.update({
            'mask_index': mask_index,
            'mask_name': "A{}".format(mask_index),
            'diameter': diameter,
            'type': _type,
            'location': location,
            'sub_location': sub_location,
            'prob_max': prob_max,
            'checked': "1",
            'is_ai': kwargs.get('is_ai', "1"),
            'seg_sop_instance_uid': seg_sop_instance_uid,
            'seg_series_instance_uid': seg_series_instance_uid,
            'dicom_sop_instance_uid': dicom_sop_instance_uid,
            'main_seg_slice': main_seg_slice,
            'is_main_seg': "1"
        })
        return AneurysmMaskInstanceRequest.model_validate(mask_instance_dict)

    def build_mask_series(self,
                          source_images: List[Union[FileDataset, DicomDir]],
                          reslut_list: List[Dict[str, Any]],
                          pred_json_list: List[Dict[str, Any]],
                          *args, **kwargs) -> Union[Dict[str,Any],Any]:

        """
            one cmb lesion is one mask series

        """
        series_instance_uid = source_images[0].get((0x0020, 0x000E)).value
        if self.series_type:
            mask_series_dict = {
                'series_instance_uid': series_instance_uid,
                'series_type': self.series_type,
                'model_type' : self.model_type
            }
        else:
            series_type     = kwargs.get('series_type')
            if series_type:
                mask_series_dict = {
                    'series_instance_uid': series_instance_uid,
                    'series_type': series_type,
                    'model_type': self.model_type
                }
            else:
                raise ValueError('series_type is None')

        mask_instance_list = []
        for index, reslut in enumerate(reslut_list):
            filter_cmd_data = list(filter(lambda x: x['mask_index'] == str(reslut['mask_index']), pred_json_list))
            if filter_cmd_data:
                pass
                filter_cmd: Dict[str, Any] = filter_cmd_data[0]
            else:
                continue
            dcm_seg_path = reslut['dcm_seg_path']
            with open(dcm_seg_path, 'rb') as f:
                dcm_seg = pydicom.read_file(f)
            kwargs.update({'diameter': filter_cmd['diameter'],
                           'type': filter_cmd['type'],
                           'location': filter_cmd['location'],
                           'sub_location': filter_cmd['sub_location'] if filter_cmd['sub_location'] else "" ,
                           'prob_max': filter_cmd['prob_max'],
                           'is_ai': filter_cmd['is_ai'],
                           'main_seg_slice' : filter_cmd['main_seg_slice'],
                           'mask_index' : reslut['mask_index']
                           })
            # Create a mask instance for this series
            mask_instance = self.build_mask_instance(source_images,
                                                     dcm_seg, *args, **kwargs)
            mask_instance_list.append(mask_instance)

        # Add the instances to the series dictionary
        mask_series_dict.update({'instances': mask_instance_list})
        return AneurysmMaskSeriesRequest.model_validate(mask_series_dict)

    # result_list = [{"series_name": "MRA_BRAIN",
    #                 "result_list":mar_brain_result_list},{},{}]
    # pred_json_list = [{"series_name": "MRA_BRAIN",
    #     #               "pred_list":pred_list},{},{}]
    def set_mask(self, source_images: List[Union[FileDataset, DicomDir]],
                 result_list: List[Dict[str, Any]],
                 pred_json_list: List[Dict[str, Any]],
                 group_id: int) -> Self:

        study_instance_uid = source_images[0].get((0x0020, 0x000D)).value
        mask_dict = {
            'study_instance_uid': study_instance_uid,
            'group_id': group_id
        }
        series_list = []
        for index,result in enumerate(result_list):

            mask_series = self.build_mask_series(source_images = self._series_dict[result['series_name']],
                                                 reslut_list    = result['result_list'],
                                                 pred_json_list = pred_json_list[index]['pred_json'],
                                                 **{'series_type' : result['series_name']}
                                                 )

            series_list.append(mask_series)


        mask_dict.update({'series':series_list})
        self._mask_request = AneurysmMaskRequest.model_validate(mask_dict)
        return self

    def set_sorted(self, source_images: List[Union[FileDataset, DicomDir]]) -> Self:
        sorted_dict = dict()
        series_list = []
        for series_name,series_source_images in self._series_dict.items():
            instance_list = []
            series_dict = dict()
            # Process each DICOM image
            for index, dicom_ds in enumerate(series_source_images):
                instance_dict = dict()
                # Extract SOP Instance UID and Image Position Patient
                sop_instance_uid = dicom_ds.get((0x0008, 0x0018)).value
                image_osition_patient = str(dicom_ds.get((0x0020, 0x0032)).value[-1])
                # Format: (0020,0032) Image Position Patient -14.7431\-143.248\98.5056

                # Update instance dictionary
                instance_dict.update(dict(
                    sop_instance_uid=sop_instance_uid,
                    projection=str(image_osition_patient)
                ))
                instance_list.append(instance_dict)

                # If this is the first image, add study and series information
                if index == 0:
                    study_instance_uid = dicom_ds.get((0x0020, 0x000D)).value
                    series_instance_uid = dicom_ds.get((0x0020, 0x000E)).value
                    series_dict.update(dict(
                        series_instance_uid=series_instance_uid,
                        instance=instance_list
                    ))
                    series_list.append(series_dict)
                    sorted_dict.update({'study_instance_uid': study_instance_uid})

            # Add the series list to the sorted dictionary
        sorted_dict.update({'series': series_list})

         # Validate and return the sorted request
        self._sorted_request = SortedRequest.model_validate(sorted_dict)
        return self


    def set_study(self, source_images: List[Union[FileDataset, DicomDir]],
                  result_list: List[Dict[str, Any]],
                  pred_json_list: List[Dict[str, Any]], group_id: int) -> Self:
        study_dict = self.build_study_basic_info(source_images=source_images,
                                                 group_id=group_id)
        study_series_list = []
        study_model_list  = []
        for index, (series_name, series_source_images) in enumerate(self._series_dict.items()):
            study_series = self.build_study_series(source_images=series_source_images,series_type=series_name)
            study_series_list.extend(study_series)
            study_model = self.build_study_model(pred_json_list=pred_json_list[index]['pred_json'],series_type=series_name)
            study_model_list.extend(study_model)


        study_dict.update({'series':study_series_list,
                           'model':study_model_list})
        self._study_request = StudyRequest.model_validate(study_dict)

        return self



    def build_study_series(self, source_images: List[Union[FileDataset, DicomDir]]
                           , *args, **kwargs) -> Union[Dict[str,Any],Any]:
        if self.series_type:
            series_type = self.series_type
        else:
            series_type = kwargs.get('series_type')

        series_instance_uid_dict = {}
        for index, dicom_ds in enumerate(source_images):
            series_instance_uid = dicom_ds.get((0x0020, 0x000E)).value
            if series_instance_uid_dict.get(series_instance_uid) is None:

                # 取得影像解析度
                resolution_x = dicom_ds.get((0x0028, 0x0010)).value
                resolution_y = dicom_ds.get((0x0028, 0x0011)).value

                series_instance_uid_dict.update({series_instance_uid: {
                    'series_type': series_type,
                    'resolution_x': resolution_x,
                    'resolution_y': resolution_y
                }})
            else:
                break
        series_list = []
        for series_instance_uid, series_info in series_instance_uid_dict.items():
            instance_dict = dict()

            # Update instance dictionary
            instance_dict.update(dict(
                series_type=series_info['series_type'],
                series_instance_uid=series_instance_uid,
                resolution_x=series_info['resolution_x'],
                resolution_y=series_info['resolution_y']
            ))
            series_list.append(StudySeriesRequest.model_validate(instance_dict))

        # Validate and return the sorted request
        return series_list

    def build_study_model(self, pred_json_list: List[Dict[str, Any]]
                          , *args, **kwargs) -> Union[Dict[str,Any],Any]:
        if self.series_type:
            study_model = dict(lession     = len(pred_json_list),
                               series_type = self.series_type,
                               model_type  = self.model_type,
                               status = "1",
                               report = "",)
        else:
            series_type = kwargs.get('series_type')
            study_model = dict(lession=len(pred_json_list),
                               series_type=series_type,
                               model_type=self.model_type,
                               status="1",
                               report="", )
        return [StudyModelRequest.model_validate(study_model)]



class ReviewAneurysmPlatformJSONBuilder(ReviewBasePlatformJSONBuilder):
    model_type  = ModelTypeEnum.Aneurysm.value
    MaskInstanceClass = AneurysmMaskInstanceRequest
    AITeamClass     = AneurysmAITeam2Request
    MaskClass       = AneurysmMask2Request
    MaskSeriesClass = AneurysmMaskSeries2Request
    MaskModelClass  = AneurysmMaskModel2Request

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group_id = kwargs.get('group_id',51)

    def get_mask_instance(self, source_images: List[Union[FileDataset, DicomDir]],
                          series_type: SeriesTypeEnum,
                          dicom_seg_result:Dict[str,Any],
                          pred_json :Dict[str,Any],
                          *args, **kwargs) -> List["MaskInstanceClass"]:
        result_data_list    = dicom_seg_result['data']
        pred_json_data_list = pred_json['data']
        mask_instance_list = []
        for index, result in enumerate(result_data_list):
            mask_instance_dict = dict()
            filter_cmd_data = list(filter(lambda x: str(x['mask_index']) == str(result['mask_index']), pred_json_data_list))
            if filter_cmd_data :
                filter_cmd = filter_cmd_data[0]
            else:
                continue
            dcm_seg_path = result['dcm_seg_path']
            with open(dcm_seg_path, 'rb') as f:
                dcm_seg = pydicom.read_file(f)
            # Extract UIDs from the DICOM-SEG and source images
            seg_sop_instance_uid = dcm_seg.get((0x008, 0x0018)).value
            seg_series_instance_uid = dcm_seg.get((0x020, 0x000E)).value
            dicom_sop_instance_uid = source_images[int(result['main_seg_slice'])].get((0x008, 0x0018)).value
            # mask_index = kwargs.get('mask_index', 1)
            # main_seg_slice = kwargs.get('main_seg_slice', "")
            # diameter = kwargs.get('diameter', '0.0')
            # _type = kwargs.get('type', '')
            # location = kwargs.get('location', 'M')
            # sub_location = kwargs.get('sub_location', '')
            # prob_max = kwargs.get('prob_max', '1.0')
            mask_instance_dict.update({'diameter'       : filter_cmd['diameter'],
                                       'type'           : filter_cmd['type'],
                                       'location'       : filter_cmd['location'],
                                       'sub_location'   : filter_cmd['sub_location'],
                                       'prob_max'       : filter_cmd['prob_max'],
                                       'is_ai'          : filter_cmd['is_ai'],
                                       #'main_seg_slice' : filter_cmd['main_seg_slice'],
                                       'main_seg_slice' : len(source_images) - result['main_seg_slice'],
                                       'mask_index'     : result['mask_index'],
                                       'mask_name': "A{}".format(result['mask_index']),
                                       'seg_sop_instance_uid': seg_sop_instance_uid,
                                       'seg_series_instance_uid': seg_series_instance_uid,
                                       'dicom_sop_instance_uid': dicom_sop_instance_uid,
                                       'is_main_seg': "1",
                                       'checked': "1",
                                       })
            mask_instance_list.append(self.MaskInstanceClass.model_validate(mask_instance_dict))
        return mask_instance_list


    def get_study_model(self, series_type: SeriesTypeEnum, pred_data: Dict[str, Any],
                        *args, **kwargs) -> "StudyModelClass":
        pred_json_list = pred_data['data']
        study_model = dict(lession=len(pred_json_list),
                           series_type=series_type.value,
                           model_type=self.model_type,
                           status="1",
                           report="", )
        return self.StudyModelClass.model_validate(study_model)



    def get_mask_series(self,source_images:List[Union[FileDataset, DicomDir]],
                             series_type: SeriesTypeEnum,
                             dicom_seg_result,
                             pred_json,
                            *args, **kwargs) -> "MaskSeriesClass":
        """
            dicom_seg_result : {"series_type": {}, "data":{} }
            dicom_seg_result : {"series_type": {}, "data":{} }
        """
        series_instance_uid = source_images[0].get((0x0020, 0x000E)).value
        mask_series_dict = {
            'series_instance_uid': series_instance_uid,
            'series_type': series_type,
        }
        mask_instance = self.get_mask_instance(source_images = source_images,
                                               series_type = series_type,
                                               dicom_seg_result = dicom_seg_result,
                                               pred_json = pred_json,
                                               *args, **kwargs)
        mask_series_dict.update({'instances': mask_instance})


        return self.MaskSeriesClass.model_validate(mask_series_dict)


    def get_mask_model(self, source_images: List[Union[FileDataset, DicomDir]], series_type: SeriesTypeEnum,
                          dicom_seg_result: Dict[str, Any], pred_json: Dict[str, Any], *args, **kwargs) -> MaskModelClass:
        mask_model_dict = {}
        mask_series = self.get_mask_series(source_images=source_images,
                                           series_type=series_type,
                                           dicom_seg_result=dicom_seg_result,
                                           pred_json=pred_json)
        mask_model_dict.update({"series":[mask_series],
                                "model_type":self.model_type
                                })
        return self.MaskModelClass.model_validate(mask_model_dict)


    def build_mask(self,dicom_seg_result_list,pred_json_list,*args, **kwargs) -> Self:
        mask_dict = {}
        if all((self._series_dict is not None,self.group_id is not None)):
            for index, (series_name, series_source_images) in enumerate(self._series_dict.items()):
                if index == 0:
                    study_instance_uid = series_source_images[0].get((0x0020, 0x000D)).value
                    mask_dict.update( {
                        'study_instance_uid': study_instance_uid,
                        'group_id': self.group_id
                    })
                    break
        else:
            raise ValueError('self._series_dict or self.group_id is None')
        mask_model_series_list = []
        mask_model_dict = {}
        for index, (series_name, series_source_images) in enumerate(self._series_dict.items()):
            mask_model_series:AneurysmMaskModel2Request = self.get_mask_model(source_images    = series_source_images,
                                                    dicom_seg_result = dicom_seg_result_list[index],
                                                    pred_json        = pred_json_list[index],
                                                    series_type      = series_name,
                                                    *args, **kwargs)
            mask_model_series_list.extend(mask_model_series.series)
            if index == 0:
                mask_model_dict.update({"model_type":mask_model_series.model_type})

        mask_model_dict.update({"series":mask_model_series_list})
        mask_dict.update({'model': [mask_model_dict]})
        self._mask_request = self.MaskClass.model_validate(mask_dict)
        return self

    def build_study(self, *args, **kwargs) -> Self:
        study_series_list = []
        study_model_list = []
        study_dict = {}
        pred_json_list = kwargs.get('pred_json_list')
        for index, (series_name, series_source_images) in enumerate(self._series_dict.items()):
            if index==0:
                study_dict.update(self.get_study_basic_info(source_images=series_source_images,
                                                            group_id=self.group_id))
                study_model = self.get_study_model(series_type=series_name, pred_data=pred_json_list[index],
                                                   *args, **kwargs)
                study_model_list.append(study_model)
            else:
                study_model:StudyModelRequest = study_model_list[0]
                new_study_model = self.get_study_model(series_type=series_name, pred_data=pred_json_list[index],
                                                   *args, **kwargs)
                
                if not isinstance(study_model.series_type, list):
                    study_model.series_type = [study_model.series_type]
                if not isinstance(new_study_model.series_type, list):
                    new_study_model.series_type = [new_study_model.series_type]
                study_model.series_type.extend(new_study_model.series_type)

            study_series = self.get_study_series(source_images=series_source_images, series_type=series_name)
            study_series_list.append(study_series)

        study_dict.update({'series': study_series_list,
                           'model': study_model_list})
        self._study_request = self.StudyClass.model_validate(study_dict)
        return self


def use_create_dicom_seg_file(path_nii:pathlib.Path,
                              series_name: str,
                              output_folder: pathlib.Path,
                              image: Any,
                              first_dcm: FileDataset | DicomDir,
                              source_images: List[FileDataset | DicomDir],):
    # Load prediction data from NIfTI file
    if series_name == "MRA_BRAIN":
        new_nifti_array = utils.get_array_to_dcm_axcodes(path_nii.joinpath('Pred.nii.gz'))
    else:
        new_nifti_array = utils.get_array_to_dcm_axcodes(path_nii.joinpath(f'{series_name}_pred.nii.gz'))
    pred_data_unique = np.unique(new_nifti_array)
    if len(pred_data_unique) < 1:
        return None
    else:
        pred_data_unique = pred_data_unique[1:]  # Exclude background value (0)
    result_list = utils.create_dicom_seg_file(pred_data_unique,
                                              new_nifti_array,
                                              series_name,
                                              output_folder,
                                              image,
                                              first_dcm,
                                              source_images)
    return result_list


def use_create_combined_dicom_seg_file(label_nii_path: str,
                                        pred_nii_path: str,
                                        series_name: str,
                                        output_folder: pathlib.Path,
                                        image: Any,
                                        first_dcm: FileDataset | DicomDir,
                                        source_images: List[FileDataset | DicomDir],
                                        label_count: int):
    """
    分別處理 Label 和 Pred 並創建 DICOM-SEG，保持兩者完整性不破壞
    
    Args:
        label_nii_path: Label NIfTI 檔案路徑
        pred_nii_path: Pred NIfTI 檔案路徑
        series_name: Series 名稱
        output_folder: 輸出資料夾
        image: DICOM 影像
        first_dcm: 第一個 DICOM 檔案
        source_images: 來源 DICOM 影像列表
        label_count: Label 的動脈瘤數量
    
    Returns:
        result_list: 合併的 DICOM-SEG 結果列表
    """
    # 讀取 Label 和 Pred NIfTI
    label_array = utils.get_array_to_dcm_axcodes(label_nii_path)
    pred_array = utils.get_array_to_dcm_axcodes(pred_nii_path)
    
    result_list = []
    
    # 處理 Label（保持原編號）
    label_unique = np.unique(label_array)
    label_unique = label_unique[label_unique > 0]  # 排除背景 (0)
    
    if len(label_unique) > 0:
        # 創建 Label 的 DICOM-SEG
        label_result_list = utils.create_dicom_seg_file(
            label_unique,
            label_array,
            series_name,
            output_folder,
            image,
            first_dcm,
            source_images
        )
        if label_result_list:
            result_list.extend(label_result_list)
    
    # 處理 Pred（重新編號接續 Label）
    pred_unique = np.unique(pred_array)
    pred_unique = pred_unique[pred_unique > 0]  # 排除背景 (0)
    
    if len(pred_unique) > 0:
        # 重新編號 Pred array（接續 Label 編號）
        pred_renumbered = np.zeros_like(pred_array)
        for pred_idx, pred_value in enumerate(pred_unique, start=1):
            new_pred_value = label_count + pred_idx  # Pred 編號接續 Label
            pred_renumbered[pred_array == pred_value] = new_pred_value
        
        # 取得重新編號後的唯一值
        pred_renumbered_unique = np.unique(pred_renumbered)
        pred_renumbered_unique = pred_renumbered_unique[pred_renumbered_unique > 0]
        
        # 創建 Pred 的 DICOM-SEG
        pred_result_list = utils.create_dicom_seg_file(
            pred_renumbered_unique,
            pred_renumbered,
            series_name,
            output_folder,
            image,
            first_dcm,
            source_images
        )
        if pred_result_list:
            result_list.extend(pred_result_list)
    
    return result_list if len(result_list) > 0 else None


# pred_json_list = [{"series_name": "MRA_BRAIN",
# "pred_json":pred_json},
#                       {"series_name": "MIP_Pitch",
#                        "pred_json": pred_json},
#                       {"series_name": "MIP_Yaw",
#                        "pred_json": pred_json},
#                       ]
def get_excel_to_pred_json(excel_file_path:str,
                           intput_list:List[Dict[str,Any]]):
    """
    intput_list:Dict[str,str] -> {"series_name": "MIP_Pitch",
         #                        "pred_nii_path": "*.nii.gz"}
    """
    pred_json_list = []
    df = pd.read_excel(excel_file_path)
    aneurysm_number_count = df['Aneurysm_Number'].iloc()[0]
    for intput_dict in intput_list:
        series_name_pred = []
        for aneurysm_number in range(1,aneurysm_number_count+1):
            pred = utils.get_array_to_dcm_axcodes(intput_dict['pred_nii_path'])
            z_l, y_l, x_l = pred.shape
            have_labels = np.where(pred == aneurysm_number)[0]
            # have_labels = np.where(np.sum(pred_one, axis=(1,2)) > 0)[0]
            sub_location = df[f"{aneurysm_number}_SubLocation"].iloc()[0]
            segment_attribute = {
                "mask_index": str(aneurysm_number),
                "mask_name": 'A' + str(aneurysm_number),
                "diameter": str(round(df[f"{aneurysm_number}_size"].iloc()[0], 1)),
                "type": 'saccular',
                "location": str(df[f"{aneurysm_number}_Location"].iloc()[0]),
                "sub_location": str(sub_location) if pd.notna(sub_location) else "",
                "checked": "1",
                "prob_max": round(df[f"{aneurysm_number}_Prob_max"].iloc()[0], 2),
                "is_ai": "1",
                "main_seg_slice": int(pred.shape[0] - np.median(have_labels)),
                "is_main_seg": "1"
            }
            series_name_pred.append(segment_attribute)
        pred_json_list.append({"series_name":intput_dict['series_name'],
                               "data":series_name_pred
                               })
            # break
        # break
    return pred_json_list


def get_excel_to_combined_json(label_excel_path: str,
                                pred_excel_path: str,
                                label_nii_path_list: List[Dict[str, Any]],
                                pred_nii_path_list: List[Dict[str, Any]]):
    """
    整合 Label 和 Pred 的 JSON 資料，Label 優先編號，Pred 接續編號
    
    Args:
        label_excel_path: Label Excel 檔案路徑
        pred_excel_path: Pred Excel 檔案路徑
        label_nii_path_list: Label NIfTI 檔案路徑列表
        pred_nii_path_list: Pred NIfTI 檔案路徑列表
    
    Returns:
        combined_json_list: 整合後的 JSON 列表
    """
    combined_json_list = []
    
    # 讀取 Label Excel
    df_label = pd.read_excel(label_excel_path)
    label_count = df_label['Aneurysm_Number'].iloc[0]
    
    # 讀取 Pred Excel
    df_pred = pd.read_excel(pred_excel_path)
    pred_count = df_pred['Aneurysm_Number'].iloc[0]
    
    for label_dict, pred_dict in zip(label_nii_path_list, pred_nii_path_list):
        series_name = label_dict['series_name']
        series_combined_data = []
        
        # 讀取 NIfTI 檔案（避免在迴圈中重複讀取）
        label_nii = utils.get_array_to_dcm_axcodes(label_dict['nii_path'])
        pred_nii = utils.get_array_to_dcm_axcodes(pred_dict['nii_path'])
        
        # 處理 Label 資料（優先編號）
        for aneurysm_number in range(1, label_count + 1):
            have_labels = np.where(label_nii == aneurysm_number)[0]
            
            if len(have_labels) > 0:  # 只處理有資料的標籤
                sub_location = df_label[f"{aneurysm_number}_SubLocation"].iloc[0]
                segment_attribute = {
                    "mask_index": str(aneurysm_number),
                    "mask_name": 'A' + str(aneurysm_number),  # Label 用 A 前綴
                    "diameter": str(round(df_label[f"{aneurysm_number}_size"].iloc[0], 1)),
                    "type": 'saccular',
                    "location": str(df_label[f"{aneurysm_number}_Location"].iloc[0]),
                    "sub_location": str(sub_location) if pd.notna(sub_location) else "",
                    "checked": "1",
                    "prob_max": round(df_label[f"{aneurysm_number}_Prob_max"].iloc[0], 2) if f"{aneurysm_number}_Prob_max" in df_label.columns else 1.0,
                    "is_ai": "0",  # Label 不是 AI 預測
                    "main_seg_slice": int(label_nii.shape[0] - np.median(have_labels)),
                    "is_main_seg": "1"
                }
                series_combined_data.append(segment_attribute)
        
        # 處理 Pred 資料（接續 Label 編號）
        for aneurysm_number in range(1, pred_count + 1):
            have_labels = np.where(pred_nii == aneurysm_number)[0]
            
            if len(have_labels) > 0:  # 只處理有資料的預測
                new_mask_index = label_count + aneurysm_number  # Pred 編號接續 Label
                sub_location = df_pred[f"{aneurysm_number}_SubLocation"].iloc[0]
                segment_attribute = {
                    "mask_index": str(new_mask_index),
                    "mask_name": 'A' + str(new_mask_index),  # Pred 用 A 前綴，編號接續 Label
                    "diameter": str(round(df_pred[f"{aneurysm_number}_size"].iloc[0], 1)),
                    "type": 'saccular',
                    "location": str(df_pred[f"{aneurysm_number}_Location"].iloc[0]),
                    "sub_location": str(sub_location) if pd.notna(sub_location) else "",
                    "checked": "1",
                    "prob_max": round(df_pred[f"{aneurysm_number}_Prob_max"].iloc[0], 2),
                    "is_ai": "1",  # Pred 是 AI 預測
                    "main_seg_slice": int(pred_nii.shape[0] - np.median(have_labels)),
                    "is_main_seg": "1"
                }
                series_combined_data.append(segment_attribute)
        
        combined_json_list.append({
            "series_name": series_name,
            "data": series_combined_data
        })
    
    return combined_json_list



def execute_dicomseg_platform_json(_id:int,root_path:str,group_id:int):
    path_root = pathlib.Path(root_path)
    path_dcms = path_root.joinpath("Dicom")
    path_nii = path_root.joinpath("Image_nii")
    path_reslice_nii = path_root.joinpath("Image_reslice")
    path_dcmseg = path_dcms.joinpath("Dicom-Seg")

    # Create output directory
    output_series_folder = path_root
    if output_series_folder.is_dir():
        output_series_folder.mkdir(exist_ok=True, parents=True)
    else:
        output_series_folder.parent.mkdir(parents=True, exist_ok=True)

    mar_brain_path_dcms = path_dcms.joinpath("MRA_BRAIN")
    mip_pitch_path_dcms = path_dcms.joinpath("MIP_Pitch")
    mip_yaw_path_dcms = path_dcms.joinpath("MIP_Yaw")

    # 處理 Label 檔案
    label_nii_path_list = [{"series_name": "MRA_BRAIN",
                            "nii_path": str(path_nii.joinpath("Label.nii.gz"))},
                           {"series_name": "MIP_Pitch",
                            "nii_path": str(path_reslice_nii.joinpath("MIP_Pitch_label.nii.gz"))},
                           {"series_name": "MIP_Yaw",
                            "nii_path": str(path_reslice_nii.joinpath("MIP_Yaw_label.nii.gz"))}
                           ]
    
    # 處理 Pred 檔案
    pred_nii_path_list = [{"series_name": "MRA_BRAIN",
                           "nii_path": str(path_nii.joinpath("Pred.nii.gz"))},
                          {"series_name": "MIP_Pitch",
                           "nii_path": str(path_reslice_nii.joinpath("MIP_Pitch_pred.nii.gz"))},
                          {"series_name": "MIP_Yaw",
                           "nii_path": str(path_reslice_nii.joinpath("MIP_Yaw_pred.nii.gz"))}
                          ]
    
    # 讀取 Label 和 Pred 的 Excel 並整合
    label_excel_path = str(path_nii.parent.joinpath("excel", "Aneurysm_Label_list.xlsx"))
    pred_excel_path = str(path_nii.parent.joinpath("excel", "Aneurysm_Pred_list.xlsx"))
    
    combined_json_list = get_excel_to_combined_json(
        label_excel_path=label_excel_path,
        pred_excel_path=pred_excel_path,
        label_nii_path_list=label_nii_path_list,
        pred_nii_path_list=pred_nii_path_list
    )
    
    # 讀取 Label 數量用於合併編號
    df_label = pd.read_excel(label_excel_path)
    label_count = df_label['Aneurysm_Number'].iloc[0]

    mar_brain_sorted_dcms, mar_brain_image, mar_brain_first_dcm, mar_brain_source_images = utils.load_and_sort_dicom_files(
        str(mar_brain_path_dcms))
    # Create combined DICOM-SEG files (Label + Pred)
    mar_brain_result_list = use_create_combined_dicom_seg_file(
        label_nii_path=str(path_nii.joinpath("Label.nii.gz")),
        pred_nii_path=str(path_nii.joinpath("Pred.nii.gz")),
        series_name="MRA_BRAIN",
        output_folder=path_dcmseg,
        image=mar_brain_image,
        first_dcm=mar_brain_first_dcm,
        source_images=mar_brain_source_images,
        label_count=label_count
    )
    
    mip_pitch_sorted_dcms, mip_pitch_image, mip_pitch_first_dcm, mip_pitch_source_images = utils.load_and_sort_dicom_files(
        str(mip_pitch_path_dcms))
    mip_pitch_result_list = use_create_combined_dicom_seg_file(
        label_nii_path=str(path_reslice_nii.joinpath("MIP_Pitch_label.nii.gz")),
        pred_nii_path=str(path_reslice_nii.joinpath("MIP_Pitch_pred.nii.gz")),
        series_name="MIP_Pitch",
        output_folder=path_dcmseg,
        image=mip_pitch_image,
        first_dcm=mip_pitch_first_dcm,
        source_images=mip_pitch_source_images,
        label_count=label_count
    )

    mip_yaw_sorted_dcms, mip_yaw_image, mip_yaw_first_dcm, mip_yaw_source_images = utils.load_and_sort_dicom_files(
        str(mip_yaw_path_dcms))
    mip_yaw_result_list = use_create_combined_dicom_seg_file(
        label_nii_path=str(path_reslice_nii.joinpath("MIP_Yaw_label.nii.gz")),
        pred_nii_path=str(path_reslice_nii.joinpath("MIP_Yaw_pred.nii.gz")),
        series_name="MIP_Yaw",
        output_folder=path_dcmseg,
        image=mip_yaw_image,
        first_dcm=mip_yaw_first_dcm,
        source_images=mip_yaw_source_images,
        label_count=label_count
    )
    result_list = [{"series_name": "MRA_BRAIN",
                    "data": mar_brain_result_list},
                   {"series_name": "MIP_Pitch",
                    "data": mip_pitch_result_list},
                   {"series_name": "MIP_Yaw",
                    "data": mip_yaw_result_list},
                   ]
    # pred_json_list = [{"series_name": "MRA_BRAIN",
    #                    "pred_json":pred_json},
    #                   {"series_name": "MIP_Pitch",
    #                    "pred_json": pred_json},
    #                   {"series_name": "MIP_Yaw",
    #                    "pred_json": pred_json},
    #                   ]
    # # Create platform JSON
    # aneurysm_platform_json_builder = AneurysmPlatformJSONBuilder()
    # aneurysm_platform_json = (aneurysm_platform_json_builder.set_series_type(SeriesTypeEnum.MRA_BRAIN, mar_brain_source_images)
    #                                                         .set_series_type(SeriesTypeEnum.MIP_Pitch, mip_pitch_source_images)
    #                                                         .set_series_type(SeriesTypeEnum.MIP_Yaw, mip_yaw_source_images)
    #                                                         .set_mask(source_images=mar_brain_source_images,
    #                                                                   result_list =result_list,
    #                                                                   pred_json_list=pred_json_list,
    #                                                                   group_id = group_id)
    #                                                         .set_sorted(source_images=mar_brain_source_images)
    #                                                         .set_study(source_images=mar_brain_source_images,
    #                                                                    result_list =result_list,
    #                                                                    pred_json_list=pred_json_list,
    #                                                                    group_id = group_id)
    #                                                         .build())

    aneurysm_platform_json_builder = ReviewAneurysmPlatformJSONBuilder()
    aneurysm_platform_json = (aneurysm_platform_json_builder.set_group_id(group_id=group_id)
                              .set_series_type(SeriesTypeEnum.MRA_BRAIN, mar_brain_source_images)
                              .set_series_type(SeriesTypeEnum.MIP_Pitch, mip_pitch_source_images)
                              .set_series_type(SeriesTypeEnum.MIP_Yaw, mip_yaw_source_images)
                              .build_mask(dicom_seg_result_list=result_list,
                                          pred_json_list=combined_json_list)
                              .build_sorted()
                              .build_study(pred_json_list=combined_json_list, )
                              .build())

    platform_json_path = output_series_folder.joinpath(
        path_root.joinpath("JSON", str(_id) + '_platform_json.json'))
    with open(platform_json_path, 'w') as f:
        f.write(aneurysm_platform_json.model_dump_json())
    print("Processing complete!")




def main(_id='07807990_20250715_MR_21405290051',
         path_processID=pathlib.Path('/data/4TB1/pipeline/chuan/process/Deep_Aneurysm/07807990_20250715_MR_21405290051/nnUNet'),
         group_id='54'):
    """
    Main function to process command line arguments and execute the pipeline.

    This function:
    """
    # Parse command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--ID', type=str, default='01901124_20250617_MR_21404020048',
    #                     help='目前執行的case的patient_id or study id')

    # parser.add_argument('--Inputs', type=str, nargs='+',
    #                     default=['/mnt/e/pipeline/新增資料夾/01901124_20250617_MR_21404020048/Image_nii/Pred.nii.gz '],
    #                     help='用於輸入的檔案')
    # parser.add_argument('--Output_folder', type=str, default='/mnt/e/pipeline/新增資料夾/',
    #                     help='用於輸出結果的資料夾')
    # parser.add_argument('--InputsDicomDir', type=str,
    #                     default='/mnt/e/pipeline/新增資料夾/01901124_20250617_MR_21404020048/Dicom/MRA_BRAIN',
    #                     help='用於輸入的檔案')
    # args = parser.parse_args()
    # # path_process      = os.getenv("PATH_PROCESS")
    # # path_processModel = os.path.join(path_process, 'Deep_Aneurysm')
    # path_processModel = "/mnt/e/pipeline/p1"
    # path_processID = os.path.join(path_processModel, args.ID)

    execute_dicomseg_platform_json(
        _id=_id, root_path=path_processID, group_id=group_id)



if __name__ == '__main__':
    main()
