"""
Schema 功用示範

展示有 Schema 和沒有 Schema 的差異
"""

# ❌ 沒有 Schema 的情況
def process_detection_without_schema(data: dict):
    """沒有驗證，容易出錯"""
    # 問題 1: 不知道應該有哪些欄位
    label = data['label']  # 如果 key 拼錯了呢？data['lable']?
    
    # 問題 2: 不知道類型
    diameter = data['diameter']  # 可能是字串 "3.5" 而非數字 3.5
    
    # 問題 3: 不知道哪些是必填
    sub_location = data.get('sub_location', '')  # 每次都要手動處理預設值
    
    # 問題 4: 沒有文檔
    # 其他開發者不知道這個 dict 應該有什麼內容
    
    return f"處理: {label}, {diameter}mm"


# ✅ 有 Schema 的情況
from pydantic import BaseModel, Field
from typing import Optional

class DetectionSchema(BaseModel):
    """動脈瘤檢測結果 - 自帶文檔！"""
    label: str = Field(..., description="標籤，如 A1")
    diameter: float = Field(..., description="直徑 (mm)")
    probability: float = Field(..., ge=0.0, le=1.0, description="置信度 0-1")
    sub_location: Optional[str] = Field("", description="次級位置")

def process_detection_with_schema(data: dict):
    """有驗證，安全可靠"""
    # Pydantic 自動驗證和轉換
    detection = DetectionSchema.model_validate(data)
    
    # 類型安全，IDE 有自動補全
    return f"處理: {detection.label}, {detection.diameter}mm"


# 範例演示
if __name__ == '__main__':
    print("=" * 60)
    print("Schema 功用示範")
    print("=" * 60)
    
    # 測試 1: 正確的資料
    print("\n【測試 1】正確的資料")
    correct_data = {
        "label": "A1",
        "diameter": 3.5,
        "probability": 0.87
    }
    
    try:
        result = process_detection_with_schema(correct_data)
        print(f"✓ 成功: {result}")
    except Exception as e:
        print(f"✗ 失敗: {e}")
    
    # 測試 2: 型別錯誤（自動轉換）
    print("\n【測試 2】型別自動轉換")
    type_convert_data = {
        "label": "A1",
        "diameter": "3.5",      # 字串，會自動轉成 float
        "probability": 0.87
    }
    
    try:
        detection = DetectionSchema.model_validate(type_convert_data)
        print(f"✓ 成功: diameter={detection.diameter}, type={type(detection.diameter)}")
    except Exception as e:
        print(f"✗ 失敗: {e}")
    
    # 測試 3: 缺少必填欄位
    print("\n【測試 3】缺少必填欄位")
    missing_field_data = {
        "label": "A1",
        # 缺少 diameter 和 probability
    }
    
    try:
        detection = DetectionSchema.model_validate(missing_field_data)
        print(f"✓ 成功: {detection}")
    except Exception as e:
        print(f"✗ 失敗（預期）: {str(e)[:100]}...")
    
    # 測試 4: 數值超出範圍
    print("\n【測試 4】數值驗證")
    invalid_range_data = {
        "label": "A1",
        "diameter": 3.5,
        "probability": 1.5  # 超過 1.0！
    }
    
    try:
        detection = DetectionSchema.model_validate(invalid_range_data)
        print(f"✓ 成功: {detection}")
    except Exception as e:
        print(f"✗ 失敗（預期）: {str(e)[:100]}...")
    
    # 測試 5: JSON 序列化
    print("\n【測試 5】JSON 序列化/反序列化")
    detection = DetectionSchema(
        label="A1",
        diameter=3.5,
        probability=0.87,
        sub_location="proximal"
    )
    
    # 轉成 JSON
    json_str = detection.model_dump_json()
    print(f"JSON: {json_str}")
    
    # 從 JSON 還原
    restored = DetectionSchema.model_validate_json(json_str)
    print(f"還原: {restored.label}, {restored.diameter}mm")
    
    print("\n" + "=" * 60)
    print("總結：Schema 提供驗證、轉換、文檔、類型安全！")
    print("=" * 60)

