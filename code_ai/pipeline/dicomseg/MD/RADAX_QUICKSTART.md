# RADAX JSON 快速開始指南

## 5 分鐘快速上手

### 步驟 1: 準備資料

確保您的資料遵循以下結構:

```
/path/to/patient_data/
├── Dicom/
│   ├── MRA_BRAIN/              # DICOM 原始檔案
│   └── Dicom-Seg/              # DICOM-SEG 結果
│       ├── aneurysm_1.dcm
│       ├── aneurysm_2.dcm
│       └── ...
├── Image_nii/
│   └── Pred.nii.gz             # 預測 NIfTI
└── excel/
    └── Aneurysm_Pred_list.xlsx # 預測資料
```

### 步驟 2: 一行代碼生成 JSON

```python
from code_ai.pipeline.dicomseg.aneurysm_radax import execute_radax_json_generation
import pathlib

# 執行生成
result = execute_radax_json_generation(
    patient_id="00165585",
    root_path=pathlib.Path("/path/to/patient_data"),
    model_version="aneurysm_v1"
)

# 檢查結果
if result:
    print(f"✓ 成功! JSON 已保存至: {result}")
else:
    print("✗ 生成失敗，請檢查資料")
```

### 步驟 3: 查看結果

生成的 JSON 將保存在:
```
/path/to/patient_data/JSON/{patient_id}_radax_aneurysm.json
```

## 範例輸出

```json
{
  "inference_timestamp": "2025-06-16T08:48:05.123Z",
  "patient_id": "00165585",
  "study_instance_uid": "1.2.840.113820...",
  "series_instance_uid": "1.2.840.113619...",
  "model_id": "924d1538-597c-41d6-bc27-4b0b359111cf",
  "detections": [
    {
      "series_instance_uid": "1.2.826.0.1.3680043...",
      "sop_instance_uid": "1.2.840.113619...",
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

## 進階使用

### 批次處理多個患者

```python
patients = [
    ("patient_001", "/data/patient_001"),
    ("patient_002", "/data/patient_002"),
    ("patient_003", "/data/patient_003"),
]

for patient_id, path in patients:
    result = execute_radax_json_generation(
        patient_id=patient_id,
        root_path=pathlib.Path(path)
    )
    status = "✓" if result else "✗"
    print(f"{status} {patient_id}")
```

### 手動建構 (細粒度控制)

```python
from code_ai.pipeline.dicomseg.aneurysm_radax import (
    RadaxJSONGenerator,
    RadaxDetectionBuilder
)

# 創建生成器
generator = RadaxJSONGenerator(model_version="aneurysm_v1")

# 載入 DICOM 並創建回應
response = generator.create_response(source_images)

# 手動添加檢測
detection = (RadaxDetectionBuilder()
    .set_series_uid("1.2.826.0.1...")
    .set_sop_uid("1.2.840.113619...")
    .set_label("A1")
    .set_type("saccular")
    .set_location("ICA")
    .set_measurements(3.5, 0.87)
    .set_slice_info(79, 1)
    .set_angles(39, 39)
    .build()
)

response.add_detection(detection)

# 保存
output_path = pathlib.Path("output.json")
generator.save_json(output_path)
```

## 常見問題

### Q1: 缺少 Excel 檔案怎麼辦?

**A:** Excel 檔案必須存在且包含必要欄位:
- `Aneurysm_Number`: 動脈瘤總數
- `{N}_size`: 大小
- `{N}_Location`: 位置
- `{N}_Prob_max`: 置信度

### Q2: DICOM-SEG 檔案命名規則?

**A:** 檔案名稱格式應為: `aneurysm_{mask_index}.dcm`

範例:
```
aneurysm_1.dcm
aneurysm_2.dcm
aneurysm_3.dcm
```

### Q3: 如何修改模型 ID?

**A:** 在 `aneurysm_radax.py` 中修改:

```python
MODEL_ID_MAP = {
    "aneurysm_v1": "您的-模型-UUID",
    "aneurysm_v2": "另一個-UUID",
}
```

### Q4: JSON 生成失敗怎麼辦?

**A:** 檢查以下項目:
1. ✅ DICOM 檔案是否存在
2. ✅ Excel 檔案是否存在且格式正確
3. ✅ DICOM-SEG 檔案是否存在
4. ✅ 路徑是否正確

### Q5: 如何添加新的欄位?

**A:** 修改 `schema/aneurysm_radax.py`:

```python
class RadaxDetection(BaseModel):
    # ... 現有欄位 ...
    new_field: Optional[str] = Field(None, description="新欄位")
```

然後在 Builder 中添加:

```python
def set_new_field(self, value: str) -> "RadaxDetectionBuilder":
    self._data["new_field"] = value
    return self
```

## 檢查清單

在運行之前，確保:

- [ ] 資料結構正確
- [ ] DICOM 檔案存在
- [ ] Excel 檔案存在
- [ ] DICOM-SEG 檔案存在
- [ ] 輸出目錄有寫入權限

## 更多資訊

- 完整文檔: `RADAX_README.md`
- 使用範例: `example_radax_usage.py`
- 實現總結: `RADAX_IMPLEMENTATION_SUMMARY.md`

## 支援

遇到問題? 聯絡 AI Team。

---

**提示:** 這是根據 `aneurysm.py` 的架構規則創建的新模組，遵循 Linus 風格設計原則。

