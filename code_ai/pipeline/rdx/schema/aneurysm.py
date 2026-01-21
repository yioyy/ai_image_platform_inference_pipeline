from pydantic import field_validator, field_serializer, ConfigDict, Field, PositiveFloat, PositiveInt, confloat
from .base import DetectionsBaseResponse, PredictionBaseResponse



class AneurysmDetectionItem(DetectionsBaseResponse):
    type           : str = Field(...)
    location       : str = Field(...)
    diameter       : PositiveFloat = Field(...)
    main_seg_slice : PositiveInt   = Field(...)
    probability    : confloat(ge=0.0, le=1.0, allow_inf_nan=True) = Field(...)
    pitch_angle    : int           = Field(...)
    yaw_angle      : int           = Field(...)
    mask_index     : int           = Field(...)
    sub_location   : str           = Field(...)

    @field_serializer('diameter')
    def serialize_diameter(self, value: float) -> float:
        """序列化時保留 4 位小數"""
        return round(value, 4)


class AneurysmDetectionResponse(PredictionBaseResponse[AneurysmDetectionItem]):
    pass
