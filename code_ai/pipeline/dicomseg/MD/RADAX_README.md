# RADAX Aneurysm Detection JSON Generator

## 概述

這個模組將顱內動脈瘤檢測結果轉換為 RADAX 平台所需的 JSON 格式。

## 架構設計

### 設計原則 (Linus 風格)

✅ **Good Taste 特徵:**
- **扁平邏輯**: 無深度嵌套，最多 3 層縮排
- **Early Return**: 提前驗證，快速失敗
- **資料結構驅動**: 使用 Pydantic 模型和 Builder 模式
- **簡潔實用**: 避免過度抽象

## 文件結構

```
code_ai/pipeline/dicomseg/
├── schema/
│   └── aneurysm_radax.py          # Pydantic 資料模型
├── aneurysm_radax.py               # 主要處理邏輯
├── example_radax_usage.py          # 使用範例
└── RADAX_README.md                 # 本文件
```

## 核心組件

### 1. 資料模型 (`schema/aneurysm_radax.py`)

#### RadaxDetection
單個動脈瘤檢測結果的資料模型。

**欄位說明:**
- `series_instance_uid`: DICOM-SEG 的 SeriesInstanceUID
- `sop_instance_uid`: DICOM-SEG 的 SOPInstanceUID
- `label`: 分割標籤 (如 "A1", "A2")
- `type`: 動脈瘤類型 (如 "saccular")
- `location`: 解剖位置 (如 "ICA", "MCA")
- `diameter`: 直徑大小 (mm)
- `main_seg_slice`: 2D 最佳角度切片編號
- `probability`: 檢測置信度 (0-1)
- `pitch_angle`: 3D 最佳角度 - pitch (可選)
- `yaw_angle`: 3D 最佳角度 - yaw (可選)
- `mask_index`: Mask 索引編號
- `sub_location`: 次級解剖位置 (可選)

#### RadaxAneurysmResponse
完整的檢測回應資料模型。

**欄位說明:**
- `inference_timestamp`: 推理時間戳 (UTC ISO format)
- `patient_id`: 患者 ID
- `study_instance_uid`: StudyInstanceUID
- `series_instance_uid`: 來源 SeriesInstanceUID
- `model_id`: 模型 UUID
- `detections`: 檢測結果列表

### 2. Builder 模式

#### RadaxDetectionBuilder
使用 Fluent API 建構檢測結果。

**優勢:**
- 鏈式調用，可讀性高
- 類型安全，自動驗證
- 靈活設置，可選參數處理良好

### 3. JSON 生成器 (`aneurysm_radax.py`)

#### RadaxJSONGenerator
主要的 JSON 生成邏輯。

**核心方法:**
- `create_response()`: 創建回應實例
- `add_detection_from_result()`: 從結果添加檢測
- `process_series_results()`: 批次處理結果
- `save_json()`: 保存為 JSON 文件

## 使用方式

### 方式 1: 快速使用 (推薦)

```python
from code_ai.pipeline.dicomseg.aneurysm_radax import execute_radax_json_generation
import pathlib

# 一行代碼完成生成
result_path = execute_radax_json_generation(
    patient_id="00165585",
    root_path=pathlib.Path("/path/to/patient/data"),
    model_version="aneurysm_v1"
)

if result_path:
    print(f"成功: {result_path}")
```

### 方式 2: 手動建構 (細粒度控制)

```python
from code_ai.pipeline.dicomseg.aneurysm_radax import RadaxJSONGenerator, RadaxDetectionBuilder

# 創建生成器
generator = RadaxJSONGenerator(model_version="aneurysm_v1")

# 載入 DICOM 並創建回應
response = generator.create_response(source_images)

# 手動建構檢測
detection = (RadaxDetectionBuilder()
    .set_series_uid("1.2.826.0...")
    .set_sop_uid("1.2.840.113619...")
    .set_label("A1")
    .set_type("saccular")
    .set_location("ICA")
    .set_measurements(diameter=3.5, probability=0.87)
    .set_slice_info(main_seg_slice=79, mask_index=1)
    .set_angles(pitch=39, yaw=39)
    .build()
)

response.add_detection(detection)

# 保存
generator.save_json(pathlib.Path("output.json"))
```

### 方式 3: 批次處理

```python
patients = [
    ("patient_001", "/path/to/patient1"),
    ("patient_002", "/path/to/patient2"),
]

for patient_id, root_path in patients:
    result = execute_radax_json_generation(
        patient_id=patient_id,
        root_path=pathlib.Path(root_path)
    )
    print(f"{patient_id}: {'✓' if result else '✗'}")
```

## JSON 輸出格式

```json
{
  "inference_timestamp": "2025-06-16T08:48:05.123Z",
  "patient_id": "00165585",
  "study_instance_uid": "1.2.840.113820.7004543577.563.821209120092.8",
  "series_instance_uid": "1.2.840.113619.2.408.5554020.7697731.26195.1694478601.690",
  "model_id": "924d1538-597c-41d6-bc27-4b0b359111cf",
  "detections": [
    {
      "series_instance_uid": "1.2.826.0.1.3680043.8.498...",
      "sop_instance_uid": "1.2.840.113619.2.408...",
      "label": "A1",
      "type": "saccular",
      "location": "ICA",
      "diameter": 3.5,
      "main_seg_slice": 79,
      "probability": 0.87,
      "pitch_angle": 39,
      "yaw_angle": 39,
      "mask_index": 1,
      "sub_location": ""
    }
  ]
}
```

## 目錄結構要求

輸入資料應遵循以下結構:

```
patient_root/
├── Dicom/
│   ├── MRA_BRAIN/          # 原始 DICOM 序列
│   └── Dicom-Seg/          # DICOM-SEG 文件
│       ├── aneurysm_1.dcm
│       ├── aneurysm_2.dcm
│       └── ...
├── Image_nii/
│   └── Pred.nii.gz         # 預測 NIfTI
├── excel/
│   └── Aneurysm_Pred_list.xlsx  # 預測資料 Excel
└── JSON/                   # 輸出目錄
    └── {patient_id}_radax_aneurysm.json
```

## Excel 格式要求

`Aneurysm_Pred_list.xlsx` 應包含以下欄位:

| 欄位名稱 | 說明 | 範例 |
|---------|------|------|
| `Aneurysm_Number` | 動脈瘤總數 | 3 |
| `{N}_size` | 第 N 個動脈瘤的大小 | 3.5 |
| `{N}_Location` | 第 N 個動脈瘤的位置 | ICA |
| `{N}_SubLocation` | 第 N 個動脈瘤的次位置 | "" |
| `{N}_Prob_max` | 第 N 個動脈瘤的置信度 | 0.87 |

其中 `{N}` 為動脈瘤編號 (1, 2, 3, ...)

## 模型版本配置

在 `aneurysm_radax.py` 中配置模型 ID:

```python
MODEL_ID_MAP = {
    "aneurysm_v1": "924d1538-597c-41d6-bc27-4b0b359111cf",
    "aneurysm_v2": "另一個-UUID",
}
```

## 錯誤處理

所有關鍵函數都使用 **Early Return** 模式:

```python
def process_data(data):
    # 驗證 1
    if not data:
        return None
    
    # 驗證 2
    if not data.is_valid():
        return None
    
    # 主要邏輯
    result = do_processing(data)
    return result
```

**優勢:**
- 扁平邏輯，無嵌套
- 快速失敗，易於除錯
- 符合 Linus 風格

## 測試

```python
# 執行範例
python code_ai/pipeline/dicomseg/example_radax_usage.py

# 執行主程式
python code_ai/pipeline/dicomseg/aneurysm_radax.py
```

## 擴展

### 添加新的檢測屬性

1. 在 `RadaxDetection` 模型中添加欄位:
```python
class RadaxDetection(BaseModel):
    # ... 現有欄位 ...
    new_field: Optional[str] = Field(None, description="新欄位")
```

2. 在 `RadaxDetectionBuilder` 中添加設置方法:
```python
def set_new_field(self, value: str) -> "RadaxDetectionBuilder":
    self._data["new_field"] = value
    return self
```

### 支援新的影像序列類型

在 `execute_radax_json_generation` 中添加處理邏輯:

```python
# 載入新序列
path_new_series = root_path / "Dicom" / "NEW_SERIES"
if path_new_series.exists():
    _, _, _, new_series_images = utils.load_and_sort_dicom_files(str(path_new_series))
    # 處理邏輯...
```

## 最佳實踐

1. **使用 Builder 模式**: 提高可讀性和可維護性
2. **Early Return**: 提前驗證，避免嵌套
3. **類型提示**: 使用完整的類型註解
4. **錯誤處理**: 返回 `None` 而非拋出異常（在適當情況下）
5. **文檔字符串**: 簡潔明確的說明

## 相關文件

- `aneurysm.py`: 原始的 Platform JSON 生成器
- `schema/base.py`: 基礎資料模型
- `utils.py`: DICOM 處理工具函數

## 維護者

AI Team

## 更新日誌

### v1.0.0 (2025-01-17)
- 初始版本
- 實現 RADAX JSON 格式生成
- 支援 Builder 模式
- 完整的類型提示和文檔

