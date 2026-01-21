import datetime

from typing import List,  Optional, TypeVar, Generic
from pydantic import BaseModel, field_validator, field_serializer, ConfigDict, Field
from code_ai import load_dotenv



load_dotenv()


class DetectionsBaseResponse(BaseModel):
    series_instance_uid : str = Field(...)
    sop_instance_uid    : str = Field(...)
    label               : str = Field(...)



DetectionT = TypeVar('DetectionT', bound=DetectionsBaseResponse)


class PredictionBaseResponse(BaseModel, Generic[DetectionT]):
    model_config = ConfigDict(from_attributes=True)
    inference_timestamp : datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    patient_id          : str               = Field(...)
    study_instance_uid  : str               = Field(...)
    series_instance_uid : str               = Field(...)
    model_id            : str               = Field(...)
    detections          :List[Optional[DetectionT]] = Field(...)

    @field_validator('inference_timestamp')
    @classmethod
    def ensure_utc(cls, v: datetime.datetime) -> datetime.datetime:
        """確保 timestamp 為 UTC"""
        if v.tzinfo is None:
            # 如果沒有時區資訊，假設為 UTC
            return v.replace(tzinfo=datetime.timezone.utc)
        elif v.tzinfo != datetime.timezone.utc:
            # 如果不是 UTC，轉換為 UTC
            return v.astimezone(datetime.timezone.utc)
        return v

    @field_serializer('inference_timestamp')
    def serialize_timestamp(self, value: datetime.datetime) -> str:
        """序列化為 ISO 8601 格式的 UTC 字串"""
        # 確保是 UTC 並轉換為 ISO 格式
        utc_time = value.astimezone(datetime.timezone.utc)
        return utc_time.isoformat()


PredictionT = TypeVar('PredictionT', bound=PredictionBaseResponse)


class InferenceCompleteRequest(BaseModel):
    studyInstanceUid  : str = Field(...)
    seriesInstanceUid : str = Field(...)
