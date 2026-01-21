import datetime
import os
from typing import List,  Union,  Optional
import numpy as np
from pydantic import BaseModel, PositiveInt, field_validator, field_serializer, Field
from code_ai import load_dotenv



from .base import ModelTypeEnum, SeriesTypeEnum, SortedRequest

load_dotenv()
# // 目前 Orthanc 自動同步機制的 group 應為 44
GROUP_ID = os.getenv("GROUP_ID",44)


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
    group_id           :int = GROUP_ID
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
                return datetime.datetime.strptime(value,'%Y-%m-%d').date()
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
    mask_index   : PositiveInt = 1
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


class MaskSeriesRequest(BaseModel):
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
    group_id           : PositiveInt = GROUP_ID
    series             : List[MaskSeriesRequest]




class CMBAITeamRequest(BaseModel):
    study   : Optional[StudyRequest]   = Field(None)
    sorted  : Optional[SortedRequest]  = Field(None)
    mask    : Optional[MaskRequest]    = Field(None)
