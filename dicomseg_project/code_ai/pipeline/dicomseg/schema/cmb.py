
from typing import  List,  Optional
from pydantic import Field
from .base import MaskRequest,MaskSeriesRequest, MaskInstanceRequest
from .base import AITeamRequest, StudyRequest,SortedRequest


class CMBMaskInstanceRequest(MaskInstanceRequest):
    sub_location: Optional[str] = Field(None, exclude=True)  # 標記為排除


class CMBMaskSeriesRequest(MaskSeriesRequest):
    instances: List[CMBMaskInstanceRequest]


class CMBMaskRequest(MaskRequest):
    series: List[CMBMaskSeriesRequest]


class CMBAITeamRequest(AITeamRequest):
    study   : Optional[StudyRequest]   = Field(None)
    sorted  : Optional[SortedRequest]  = Field(None)
    mask    : Optional[CMBMaskRequest] = Field(None)
