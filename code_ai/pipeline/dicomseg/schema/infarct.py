from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field  # type: ignore

from code_ai.pipeline.dicomseg.schema.base import (
    AITeamRequest,
    MaskSeriesRequest,
    SortedRequest,
    StudyRequest,
)


class InfarctMaskInstanceRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    mask_index: int
    mask_name: str
    volume: str
    type: str = ""
    location: str
    mean_adc: str
    prob_max: str
    prob_mean: str
    checked: str = "1"
    is_ai: str = "1"
    seg_series_instance_uid: str
    seg_sop_instance_uid: str
    dicom_sop_instance_uid: str
    main_seg_slice: int = 1
    is_main_seg: int = 1


class InfarctMaskSeriesRequest(MaskSeriesRequest):
    model_config = ConfigDict(from_attributes=True)
    model_type: Optional[str] = Field(default=None, exclude=True)
    instances: List[InfarctMaskInstanceRequest]


class InfarctMaskModelRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    model_type: str
    series: List[InfarctMaskSeriesRequest]


class InfarctMaskRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    study_instance_uid: str
    group_id: int
    model: List[InfarctMaskModelRequest]


class InfarctAITeamRequest(AITeamRequest):
    study: Optional[StudyRequest] = Field(None)
    sorted: Optional[SortedRequest] = Field(None)
    mask: Optional[InfarctMaskRequest] = Field(None)

