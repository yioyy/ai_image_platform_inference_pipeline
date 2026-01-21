"""
計算最佳觀察角度示範

展示 calculate_best_angles 函數的運作原理
"""
import pathlib
import numpy as np


def explain_angle_calculation():
    """說明角度計算邏輯"""
    print("=" * 60)
    print("最佳觀察角度計算說明")
    print("=" * 60)
    
    print("\n【算法說明】")
    print("1. 讀取 MIP_Pitch_pred.nii.gz 和 MIP_Yaw_pred.nii.gz")
    print("2. 針對特定動脈瘤編號 (mask_index)，找出所有標註的張數")
    print("3. 計算張數的中位數")
    print("4. 角度 = (張數中位數) * 3 - 3")
    
    print("\n【範例計算】")
    print("-" * 60)
    
    # 模擬範例：動脈瘤編號 1 在 Pitch 方向的標註
    print("\n動脈瘤 #1 - Pitch 方向:")
    pitch_slices = np.array([10, 11, 12, 13, 14, 15])  # 有標註的張數
    print(f"  標註張數: {pitch_slices}")
    median_pitch = int(np.median(pitch_slices))
    print(f"  中位數: {median_pitch}")
    pitch_angle = median_pitch * 3 - 3
    print(f"  Pitch 角度 = {median_pitch} * 3 - 3 = {pitch_angle}°")
    
    # 模擬範例：動脈瘤編號 1 在 Yaw 方向的標註
    print("\n動脈瘤 #1 - Yaw 方向:")
    yaw_slices = np.array([8, 9, 10, 11, 12])  # 有標註的張數
    print(f"  標註張數: {yaw_slices}")
    median_yaw = int(np.median(yaw_slices))
    print(f"  中位數: {median_yaw}")
    yaw_angle = median_yaw * 3 - 3
    print(f"  Yaw 角度 = {median_yaw} * 3 - 3 = {yaw_angle}°")
    
    print("\n" + "-" * 60)
    print(f"結果: {{'pitch': {pitch_angle}, 'yaw': {yaw_angle}}}")
    
    # 另一個動脈瘤的範例
    print("\n" + "=" * 60)
    print("\n動脈瘤 #2 - Pitch 方向:")
    pitch_slices_2 = np.array([20, 21, 22, 23, 24, 25, 26])
    print(f"  標註張數: {pitch_slices_2}")
    median_pitch_2 = int(np.median(pitch_slices_2))
    print(f"  中位數: {median_pitch_2}")
    pitch_angle_2 = median_pitch_2 * 3 - 3
    print(f"  Pitch 角度 = {median_pitch_2} * 3 - 3 = {pitch_angle_2}°")
    
    print("\n動脈瘤 #2 - Yaw 方向:")
    yaw_slices_2 = np.array([15, 16, 17, 18, 19, 20])
    print(f"  標註張數: {yaw_slices_2}")
    median_yaw_2 = int(np.median(yaw_slices_2))
    print(f"  中位數: {median_yaw_2}")
    yaw_angle_2 = median_yaw_2 * 3 - 3
    print(f"  Yaw 角度 = {median_yaw_2} * 3 - 3 = {yaw_angle_2}°")
    
    print("\n" + "-" * 60)
    print(f"結果: {{'pitch': {pitch_angle_2}, 'yaw': {yaw_angle_2}}}")
    
    print("\n" + "=" * 60)


def explain_file_structure():
    """說明檔案結構"""
    print("\n【檔案結構要求】")
    print("-" * 60)
    print("""
patient_root/
├── Image_nii/
│   └── Pred.nii.gz                  # 主要預測檔案
└── Image_reslice/                    # 重採樣目錄
    ├── MIP_Pitch_pred.nii.gz        # Pitch 方向 MIP 預測
    └── MIP_Yaw_pred.nii.gz          # Yaw 方向 MIP 預測

每個 NIfTI 檔案中：
- 值為 0: 背景
- 值為 1: 動脈瘤 #1 的標註
- 值為 2: 動脈瘤 #2 的標註
- 值為 3: 動脈瘤 #3 的標註
- ...以此類推
    """)


def explain_implementation():
    """說明實現細節"""
    print("\n【實現細節】")
    print("-" * 60)
    print("""
函數簽名：
    calculate_best_angles(pred_nii_path, mask_index) -> Dict[str, int]

參數：
    - pred_nii_path: Pred.nii.gz 的路徑
    - mask_index: 動脈瘤編號 (1, 2, 3, ...)

返回：
    {"pitch": pitch_angle, "yaw": yaw_angle}

流程：
1. 從 pred_nii_path 定位 Image_reslice 目錄
   pred_nii_path.parent.parent / "Image_reslice"

2. 讀取 MIP 預測檔案
   - MIP_Pitch_pred.nii.gz
   - MIP_Yaw_pred.nii.gz

3. 對每個方向：
   a. 使用 np.where() 找出所有值等於 mask_index 的張數索引
   b. 計算中位數：np.median(slices)
   c. 計算角度：median * 3 - 3

4. 錯誤處理：
   - 檔案不存在 → 返回 {"pitch": 0, "yaw": 0}
   - 沒有標註 → 角度為 0
   - 讀取錯誤 → 角度為 0
    """)


def example_usage():
    """使用範例"""
    print("\n【使用範例】")
    print("-" * 60)
    print("""
from code_ai.pipeline.dicomseg.aneurysm_radax import calculate_best_angles
import pathlib

# 預測檔案路徑
pred_nii_path = pathlib.Path("/path/to/patient/Image_nii/Pred.nii.gz")

# 計算動脈瘤 #1 的最佳角度
angles_1 = calculate_best_angles(pred_nii_path, mask_index=1)
print(f"動脈瘤 #1: {angles_1}")
# 輸出: {'pitch': 30, 'yaw': 27}

# 計算動脈瘤 #2 的最佳角度
angles_2 = calculate_best_angles(pred_nii_path, mask_index=2)
print(f"動脈瘤 #2: {angles_2}")
# 輸出: {'pitch': 66, 'yaw': 48}

# 在主流程中使用
predictions = [
    {"mask_index": "1", ...},
    {"mask_index": "2", ...},
    {"mask_index": "3", ...},
]

angles_list = [
    calculate_best_angles(pred_nii_path, int(pred["mask_index"]))
    for pred in predictions
]

# angles_list = [
#     {'pitch': 30, 'yaw': 27},
#     {'pitch': 66, 'yaw': 48},
#     {'pitch': 54, 'yaw': 39},
# ]
    """)


def explain_edge_cases():
    """說明邊界情況"""
    print("\n【邊界情況處理】")
    print("-" * 60)
    print("""
情況 1: 檔案不存在
    → 返回 {"pitch": 0, "yaw": 0}

情況 2: 特定動脈瘤沒有標註（找不到對應的 mask_index）
    → 該方向角度為 0

情況 3: 只有一個張數有標註
    → 中位數就是該張數
    → 角度 = 張數 * 3 - 3

情況 4: 讀取檔案時發生錯誤
    → 捕獲異常，返回角度 0

範例：
    # 情況 3: 只有第 5 張有標註
    slices = [5]
    median = 5
    angle = 5 * 3 - 3 = 12°
    
    # 情況 2: 完全沒有標註
    slices = []
    angle = 0°
    """)


def main():
    """主函數"""
    explain_angle_calculation()
    explain_file_structure()
    explain_implementation()
    example_usage()
    explain_edge_cases()
    
    print("\n" + "=" * 60)
    print("說明完畢！")
    print("=" * 60)


if __name__ == '__main__':
    main()

