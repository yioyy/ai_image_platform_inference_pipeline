# -*- coding: utf-8 -*-
"""
DICOM-SEG Generator

@author: sean
"""
import abc
import os
import pathlib
from typing import Dict, List, Any, Union, Optional, TypeVar, Generic
from typing_extensions import Self
from pydantic import BaseModel
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir
from code_ai.pipeline.dicomseg.schema.base import SortedRequest
from code_ai.pipeline.dicomseg.schema.base import StudyRequest
from code_ai.pipeline.dicomseg.schema.base import MaskRequest, MaskSeriesRequest
from code_ai import load_dotenv

load_dotenv()
GROUP_ID = os.getenv("GROUP_ID_CMB", 44)

T = TypeVar('T', bound=BaseModel)


class PlatformJSONBuilder(Generic[T], metaclass=abc.ABCMeta):
    model_class: type[T]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ai_team_dict: Dict[str, Any] = {}
        self.reset()

    def reset(self):
        self._study_request: Optional[StudyRequest] = None
        self._sorted_request: Optional[SortedRequest] = None
        self._mask_request: Optional[MaskRequest] = None
        self._ai_team_dict: Dict[str, Any] = {}


    @abc.abstractmethod
    def build_mask_instance(self, source_images: List[Union[FileDataset, DicomDir]],
                            dcm_seg: Union[FileDataset, DicomDir],
                            *args, **kwargs) -> Union[Dict[str,Any],Any]:
        pass

    @abc.abstractmethod
    def build_mask_series(self,
                          source_images: List[Union[FileDataset, DicomDir]],
                          reslut_list: List[Dict[str, Any]],
                          pred_json_list: List[Dict[str, Any]],
                          # dcm_seg: Union[FileDataset, DicomDir],
                          *args, **kwargs) -> Union[Dict[str,Any],Any]:
        pass

    @abc.abstractmethod
    def set_mask(self,
                 source_images: List[Union[FileDataset, DicomDir]],
                 reslut_list: List[Dict[str, Any]],
                 pred_json_list: List[Dict[str, Any]],
                 group_id: int) -> Self:
        pass

    def set_sorted(self, source_images: List[Union[FileDataset, DicomDir]]):
        instance_list = []
        series_list = []
        series_dict = dict()
        sorted_dict = dict()

        # Process each DICOM image
        for index, dicom_ds in enumerate(source_images):
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

    def build_study_basic_info(self, source_images: List[Union[FileDataset, DicomDir]],
                               group_id: int = GROUP_ID) -> Dict[str,Any]:
        dicom_ds = source_images[0]
        # Extract study metadata from the first DICOM image
        study_instance_uid = dicom_ds.get((0x0020, 0x000D)).value
        study_date = dicom_ds.get((0x0008, 0x0020)).value
        gender = dicom_ds.get((0x0010, 0x0040)).value
        age = dicom_ds.get((0x0010, 0x1010)).value
        study_name = dicom_ds.get((0x0008, 0x1030)).value
        patient_name = dicom_ds.get((0x0010, 0x0010)).value
        # resolution_x = dicom_ds.get((0x0028, 0x0010)).value
        # resolution_y = dicom_ds.get((0x0028, 0x0011)).value
        patient_id = dicom_ds.get((0x0010, 0x0020)).value

        # Update study dictionary with metadata
        return dict(
            group_id=group_id,
            study_instance_uid=study_instance_uid,
            study_date=study_date,
            gender=gender,
            age=age,
            study_name=study_name,
            patient_name=patient_name,
            patient_id=patient_id, )

    @abc.abstractmethod
    def build_study_series(self, source_images: List[Union[FileDataset, DicomDir]],
                           *args, **kwargs) -> Self:
        pass

    @abc.abstractmethod
    def build_study_model(self, pred_json_list: List[Dict[str, Any]],
                          *args, **kwargs) -> Self:
        pass


    @abc.abstractmethod
    def set_study(self,source_images: List[Union[FileDataset, DicomDir]],
                  reslut_list: List[Dict[str, Any]],
                  pred_json_list: List[Dict[str, Any]],
                  group_id: int) -> Self:
        pass

    def build(self) -> T:
        self._ai_team_dict.update({"study":self._study_request if self._study_request else None,
                                   "sorted": self._sorted_request if self._sorted_request else None,
                                   "mask": self._mask_request if self._mask_request else None,})
        instance = self.model_class.model_validate(self._ai_team_dict)
        self.reset()
        return instance
