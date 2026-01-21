"""
RADAX Aneurysm Detection Response Schema

這個模組定義了 RADAX 顱內動脈瘤檢測的回應資料結構。
遵循 Pydantic 模型設計，支持 JSON 序列化/反序列化。

@author: AI Team
"""
from datetime import datetime
from typing import List, Optional
from pydantic import Field, ConfigDict, BaseModel


class RadaxDetection(BaseModel):
    """單個動脈瘤檢測結果"""
    model_config = ConfigDict(from_attributes=True)
    
    series_instance_uid: str = Field(..., description="DICOM-SEG 的 SeriesInstanceUID")
    sop_instance_uid: str = Field(..., description="DICOM-SEG 的 SOPInstanceUID")
    label: str = Field(..., description="分割標籤，顯示在 viewer 上 (如 A1, A2)")
    type: str = Field(..., description="動脈瘤類型 (如 saccular)")
    location: str = Field(..., description="解剖位置 (如 ICA, MCA)")
    diameter: float = Field(..., description="直徑大小 (mm)")
    main_seg_slice: int = Field(..., description="2D 最佳角度位於第幾張切片")
    probability: float = Field(..., description="檢測置信度 (0-1)")
    pitch_angle: Optional[int] = Field(None, description="3D 最佳角度 - pitch")
    yaw_angle: Optional[int] = Field(None, description="3D 最佳角度 - yaw")
    mask_index: int = Field(..., description="Mask 索引編號")
    sub_location: Optional[str] = Field("", description="次級解剖位置")


class RadaxAneurysmResponse(BaseModel):
    """RADAX 動脈瘤檢測完整回應"""
    model_config = ConfigDict(from_attributes=True)
    
    inference_timestamp: str = Field(..., description="推理時間戳 (UTC ISO format)")
    patient_id: str = Field(..., description="患者 ID")
    study_instance_uid: str = Field(..., description="StudyInstanceUID")
    series_instance_uid: str = Field(..., description="來源 SeriesInstanceUID")
    model_id: str = Field(..., description="模型 UUID")
    detections: List[RadaxDetection] = Field(default_factory=list, description="檢測結果列表")
    
    @classmethod
    def create(
        cls,
        patient_id: str,
        study_instance_uid: str,
        series_instance_uid: str,
        model_id: str,
        detections: Optional[List[RadaxDetection]] = None,
        inference_timestamp: Optional[str] = None
    ) -> "RadaxAneurysmResponse":
        """工廠方法：創建回應實例"""
        if inference_timestamp is None:
            inference_timestamp = datetime.utcnow().isoformat() + "Z"
        
        return cls(
            inference_timestamp=inference_timestamp,
            patient_id=patient_id,
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
            model_id=model_id,
            detections=detections or []
        )
    
    def add_detection(self, detection: RadaxDetection) -> None:
        """添加單個檢測結果"""
        self.detections.append(detection)
    
    def get_detection_count(self) -> int:
        """獲取檢測到的動脈瘤數量"""
        return len(self.detections)
    
    def to_json(self) -> str:
        """序列化為 JSON 字符串"""
        return self.model_dump_json(indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "RadaxAneurysmResponse":
        """從 JSON 字符串反序列化"""
        return cls.model_validate_json(json_str)


class RadaxDetectionBuilder:
    """Detection 建構器 - 簡化創建流程"""
    
    def __init__(self):
        self._data = {}
    
    def set_series_uid(self, uid: str) -> "RadaxDetectionBuilder":
        """設置 SeriesInstanceUID"""
        self._data["series_instance_uid"] = uid
        return self
    
    def set_sop_uid(self, uid: str) -> "RadaxDetectionBuilder":
        """設置 SOPInstanceUID"""
        self._data["sop_instance_uid"] = uid
        return self
    
    def set_label(self, label: str) -> "RadaxDetectionBuilder":
        """設置標籤"""
        self._data["label"] = label
        return self
    
    def set_type(self, aneurysm_type: str) -> "RadaxDetectionBuilder":
        """設置類型"""
        self._data["type"] = aneurysm_type
        return self
    
    def set_location(self, location: str, sub_location: Optional[str] = None) -> "RadaxDetectionBuilder":
        """設置位置"""
        self._data["location"] = location
        if sub_location:
            self._data["sub_location"] = sub_location
        return self
    
    def set_measurements(self, diameter: float, probability: float) -> "RadaxDetectionBuilder":
        """設置測量值"""
        self._data["diameter"] = diameter
        self._data["probability"] = probability
        return self
    
    def set_slice_info(self, main_seg_slice: int, mask_index: int) -> "RadaxDetectionBuilder":
        """設置切片資訊"""
        self._data["main_seg_slice"] = main_seg_slice
        self._data["mask_index"] = mask_index
        return self
    
    def set_angles(self, pitch: Optional[int] = None, yaw: Optional[int] = None) -> "RadaxDetectionBuilder":
        """設置角度"""
        if pitch is not None:
            self._data["pitch_angle"] = pitch
        if yaw is not None:
            self._data["yaw_angle"] = yaw
        return self
    
    def build(self) -> RadaxDetection:
        """建構 RadaxDetection 實例"""
        return RadaxDetection.model_validate(self._data)


# 便捷的型別別名
RadaxResponse = RadaxAneurysmResponse
Detection = RadaxDetection

