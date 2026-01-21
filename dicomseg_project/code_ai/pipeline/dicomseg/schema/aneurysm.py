
from typing import  List,  Optional
from pydantic import Field
from .base import MaskRequest,MaskSeriesRequest, MaskInstanceRequest, SeriesTypeEnum
from .base import AITeamRequest, StudyRequest,SortedRequest,StudyModelRequest
from pydantic import field_validator


class AneurysmMaskInstanceRequest(MaskInstanceRequest):
    sub_location: Optional[str] = Field(None)  # 標記為排除


class AneurysmMaskSeriesRequest(MaskSeriesRequest):
    instances: List[AneurysmMaskInstanceRequest]


class AneurysmMaskRequest(MaskRequest):
    series: List[AneurysmMaskSeriesRequest]


class AneurysmStudyModelRequest(StudyModelRequest):
    series_type : List[str]

    @field_validator('series_type', mode='before')
    @classmethod
    def extract_series_type(cls, value):
        if value is None:
            return ['1']
        if isinstance(value, str):
            series_type_list = []
            series_type_enum_list: List[SeriesTypeEnum] = SeriesTypeEnum.to_list()
            for series_type_enum in series_type_enum_list:
                if value == series_type_enum.name:
                    series_type_list.append(series_type_enum.value)
            return series_type_list
        else:
            return list(value)

class AneurysmStudyRequest(StudyRequest):
    model : List[AneurysmStudyModelRequest]

class AneurysmAITeamRequest(AITeamRequest):
    study   : Optional[AneurysmStudyRequest] = Field(None)
    sorted  : Optional[SortedRequest]       = Field(None)
    mask    : Optional[AneurysmMaskRequest] = Field(None)
