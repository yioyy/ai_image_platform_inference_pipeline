"""
RADAX Aneurysm JSON 生成範例

展示如何使用 RadaxJSONGenerator 生成符合 RADAX 格式的 JSON。
"""
import pathlib
from code_ai.pipeline.dicomseg.aneurysm_radax import (
    RadaxJSONGenerator,
    RadaxDetectionBuilder,
    execute_radax_json_generation
)


def example_simple_usage():
    """範例 1: 簡單使用 - 完整流程"""
    print("=== 範例 1: 完整流程 ===")
    
    patient_id = "00165585"
    root_path = "/path/to/patient/data"
    
    result_path = execute_radax_json_generation(
        patient_id=patient_id,
        root_path=pathlib.Path(root_path),
        model_version="aneurysm_v1"
    )
    
    if result_path:
        print(f"✓ 成功生成: {result_path}")
    else:
        print("✗ 生成失敗")


def example_manual_construction():
    """範例 2: 手動建構 - 細粒度控制"""
    print("\n=== 範例 2: 手動建構 ===")
    
    # 創建生成器
    generator = RadaxJSONGenerator(model_version="aneurysm_v1")
    
    # 假設已載入 DICOM source_images
    # source_images = load_dicom_files(...)
    
    # 創建回應
    # response = generator.create_response(source_images)
    
    # 手動建構單個檢測
    detection = (RadaxDetectionBuilder()
        .set_series_uid("1.2.826.0.1.3680043.8.498.19510627188178662465079463369223727988")
        .set_sop_uid("1.2.840.113619.2.408.5554020.7697731.24791.1694478765.944")
        .set_label("A1")
        .set_type("saccular")
        .set_location("ICA", "")
        .set_measurements(diameter=3.5, probability=0.87)
        .set_slice_info(main_seg_slice=79, mask_index=1)
        .set_angles(pitch=39, yaw=39)
        .build()
    )
    
    print(f"✓ 建構檢測: {detection.label} at {detection.location}")
    print(f"  直徑: {detection.diameter}mm, 置信度: {detection.probability}")


def example_json_operations():
    """範例 3: JSON 操作"""
    print("\n=== 範例 3: JSON 序列化與反序列化 ===")
    
    from code_ai.pipeline.dicomseg.schema.aneurysm_radax import RadaxAneurysmResponse
    
    # 創建回應
    response = RadaxAneurysmResponse.create(
        patient_id="00165585",
        study_instance_uid="1.2.840.113820.7004543577.563.821209120092.8",
        series_instance_uid="1.2.840.113619.2.408.5554020.7697731.26195.1694478601.690",
        model_id="924d1538-597c-41d6-bc27-4b0b359111cf"
    )
    
    # 添加檢測
    detection = (RadaxDetectionBuilder()
        .set_series_uid("1.2.826.0.1.3680043.8.498.19510627188178662465079463369223727988")
        .set_sop_uid("1.2.840.113619.2.408.5554020.7697731.24791.1694478765.944")
        .set_label("A1")
        .set_type("saccular")
        .set_location("ICA")
        .set_measurements(3.5, 0.87)
        .set_slice_info(79, 1)
        .set_angles(39, 39)
        .build()
    )
    
    response.add_detection(detection)
    
    # 序列化為 JSON
    json_str = response.to_json()
    print("✓ JSON 輸出:")
    print(json_str[:200] + "...")
    
    # 反序列化
    restored = RadaxAneurysmResponse.from_json(json_str)
    print(f"\n✓ 反序列化成功: {restored.get_detection_count()} 個檢測")


def example_batch_processing():
    """範例 4: 批次處理多個患者"""
    print("\n=== 範例 4: 批次處理 ===")
    
    patients = [
        ("00165585", "/path/to/patient1"),
        ("00165586", "/path/to/patient2"),
        ("00165587", "/path/to/patient3"),
    ]
    
    results = []
    
    for patient_id, root_path in patients:
        result = execute_radax_json_generation(
            patient_id=patient_id,
            root_path=pathlib.Path(root_path),
            model_version="aneurysm_v1"
        )
        
        if result:
            results.append((patient_id, result))
            print(f"✓ {patient_id}: 成功")
        else:
            print(f"✗ {patient_id}: 失敗")
    
    print(f"\n總計: {len(results)}/{len(patients)} 成功")


def main():
    """執行所有範例"""
    print("RADAX Aneurysm JSON 生成範例\n")
    
    # 注意: 這些範例需要實際的 DICOM 檔案才能執行
    # 這裡僅展示 API 使用方式
    
    try:
        example_manual_construction()
        example_json_operations()
        # example_simple_usage()  # 需要實際檔案
        # example_batch_processing()  # 需要實際檔案
    except Exception as e:
        print(f"\n注意: 某些範例需要實際的 DICOM 檔案")
        print(f"錯誤: {e}")


if __name__ == '__main__':
    main()

