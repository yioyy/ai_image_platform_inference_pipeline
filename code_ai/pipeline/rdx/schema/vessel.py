from .base import DetectionsBaseResponse, PredictionBaseResponse


class VesselDilatedDetectionItem(DetectionsBaseResponse):
    pass


class VesselDilatedDetectionResponse(PredictionBaseResponse[VesselDilatedDetectionItem]):
    pass
