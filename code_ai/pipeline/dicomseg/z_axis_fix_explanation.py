"""
Z 軸（張數）標註中位數計算 - 修正說明

展示修正前後的差異
"""
import numpy as np


def demonstrate_the_issue():
    """展示原始問題"""
    print("=" * 70)
    print("問題：標註張數的中位數應該只看 z 軸有沒有標註")
    print("=" * 70)
    
    # 模擬 3D 陣列 (z, y, x) = (30, 256, 256)
    # 假設動脈瘤 mask_index=1 在以下張數有標註：
    print("\n【情境】動脈瘤 #1 的標註情況：")
    print("-" * 70)
    print("第 10 張: 有 500 個像素標註")
    print("第 11 張: 有 300 個像素標註")
    print("第 12 張: 有 100 個像素標註")
    print("第 13 張: 有 200 個像素標註")
    print("第 14 張: 有 400 個像素標註")
    
    # ❌ 錯誤的做法（修正前）
    print("\n" + "=" * 70)
    print("❌ 錯誤做法（修正前）")
    print("=" * 70)
    
    # 模擬 np.where(array == mask_index)[0] 的結果
    # 會包含所有標註像素的 z 索引（有重複）
    wrong_slices = (
        [10] * 500 +  # 第 10 張的 500 個像素
        [11] * 300 +  # 第 11 張的 300 個像素
        [12] * 100 +  # 第 12 張的 100 個像素
        [13] * 200 +  # 第 13 張的 200 個像素
        [14] * 400    # 第 14 張的 400 個像素
    )
    wrong_slices = np.array(wrong_slices)
    
    print(f"\n取得的張數陣列（包含重複）:")
    print(f"  長度: {len(wrong_slices)}")
    print(f"  內容: [10, 10, 10, ..., 11, 11, 11, ..., 12, ...]")
    print(f"  （第 10 張重複 500 次，第 11 張重複 300 次...）")
    
    wrong_median = int(np.median(wrong_slices))
    wrong_angle = wrong_median * 3 - 3
    
    print(f"\n計算結果:")
    print(f"  中位數: {wrong_median}")
    print(f"  角度: {wrong_median} × 3 - 3 = {wrong_angle}°")
    print(f"\n⚠️  問題：中位數偏向標註像素多的張數（第 10 張和第 14 張）")
    
    # ✅ 正確的做法（修正後）
    print("\n" + "=" * 70)
    print("✅ 正確做法（修正後）")
    print("=" * 70)
    
    # 只看哪些張數有標註（不管每張有多少像素）
    correct_slices = np.array([10, 11, 12, 13, 14])
    
    print(f"\n取得的張數陣列（去重）:")
    print(f"  長度: {len(correct_slices)}")
    print(f"  內容: {correct_slices}")
    print(f"  說明: 只記錄「有標註的張數」，不管每張有多少像素")
    
    correct_median = int(np.median(correct_slices))
    correct_angle = correct_median * 3 - 3
    
    print(f"\n計算結果:")
    print(f"  中位數: {correct_median}")
    print(f"  角度: {correct_median} × 3 - 3 = {correct_angle}°")
    print(f"\n✓ 正確：真正的「張數」中位數")
    
    print("\n" + "=" * 70)
    print(f"差異對比：")
    print(f"  錯誤結果: 角度 = {wrong_angle}° (中位數 {wrong_median})")
    print(f"  正確結果: 角度 = {correct_angle}° (中位數 {correct_median})")
    print(f"  差異: {abs(wrong_angle - correct_angle)}°")
    print("=" * 70)


def explain_the_fix():
    """說明修正方法"""
    print("\n\n" + "=" * 70)
    print("修正方法說明")
    print("=" * 70)
    
    print("\n【修正前】")
    print("-" * 70)
    print("""
pitch_slices = np.where(pitch_array == mask_index)[0]
median_slice = int(np.median(pitch_slices))

問題：
  - np.where() 返回所有符合條件的索引（包含重複）
  - 如果某張有很多像素，該張數會重複很多次
  - 中位數會偏向像素多的張數
    """)
    
    print("\n【修正後】")
    print("-" * 70)
    print("""
# 找出哪些 z 切片（張數）有標註
# 對 y, x 軸做 any 操作，只看 z 軸是否有標註
has_annotation = np.any(pitch_array == mask_index, axis=(1, 2))
slices_with_annotation = np.where(has_annotation)[0]
median_slice = int(np.median(slices_with_annotation))

優點：
  ✓ 只看「哪些張數有標註」（去重）
  ✓ 不受每張標註像素數量影響
  ✓ 真正的「張數」中位數
    """)


def demonstrate_np_any():
    """展示 np.any 的運作方式"""
    print("\n\n" + "=" * 70)
    print("np.any() 運作原理")
    print("=" * 70)
    
    # 模擬簡化的 3D 陣列
    print("\n假設有一個簡化的 3D 陣列 (5 張, 4x4 像素):")
    print("-" * 70)
    
    # 創建示範陣列
    demo_array = np.zeros((5, 4, 4), dtype=int)
    
    # 第 1 張（索引 1）有標註
    demo_array[1, 1:3, 1:3] = 1  # 中間 2x2 區域標註為 1
    
    # 第 3 張（索引 3）有標註
    demo_array[3, 0:2, 0:2] = 1  # 左上角 2x2 區域標註為 1
    
    print("\n陣列內容（mask_index=1 的標註）:")
    for z in range(5):
        has_label = np.any(demo_array[z] == 1)
        print(f"  第 {z} 張: {'有標註' if has_label else '無標註'}")
        if has_label:
            count = np.sum(demo_array[z] == 1)
            print(f"         （{count} 個像素）")
    
    print("\n執行 np.any():")
    print("-" * 70)
    
    # 方法 1: 使用 np.any
    has_annotation = np.any(demo_array == 1, axis=(1, 2))
    print(f"has_annotation = np.any(array == 1, axis=(1, 2))")
    print(f"結果: {has_annotation}")
    print(f"說明: True 表示該張有標註，False 表示沒有")
    
    # 找出有標註的張數
    slices_with_annotation = np.where(has_annotation)[0]
    print(f"\nslices_with_annotation = np.where(has_annotation)[0]")
    print(f"結果: {slices_with_annotation}")
    print(f"說明: 只有第 {slices_with_annotation} 張有標註")
    
    # 計算中位數
    if len(slices_with_annotation) > 0:
        median = int(np.median(slices_with_annotation))
        print(f"\n中位數: {median}")
    
    print("\n" + "=" * 70)
    print("總結: np.any(axis=(1,2)) 會沿著 y, x 軸做 OR 運算")
    print("      只保留 z 軸資訊（哪些張數有標註）")
    print("=" * 70)


def show_code_comparison():
    """程式碼對比"""
    print("\n\n" + "=" * 70)
    print("程式碼對比")
    print("=" * 70)
    
    print("\n【修正前】")
    print("-" * 70)
    print("""
# ❌ 會包含重複的張數索引
pitch_slices = np.where(pitch_array == mask_index)[0]

if len(pitch_slices) == 0:
    pitch_angle = 0
else:
    median_slice = int(np.median(pitch_slices))  # 受像素數量影響
    pitch_angle = median_slice * 3 - 3
    """)
    
    print("\n【修正後】")
    print("-" * 70)
    print("""
# ✅ 只看張數是否有標註（去重）
has_annotation = np.any(pitch_array == mask_index, axis=(1, 2))
slices_with_annotation = np.where(has_annotation)[0]

if len(slices_with_annotation) == 0:
    pitch_angle = 0
else:
    median_slice = int(np.median(slices_with_annotation))  # 純粹的張數中位數
    pitch_angle = median_slice * 3 - 3
    """)


def main():
    """主函數"""
    demonstrate_the_issue()
    explain_the_fix()
    demonstrate_np_any()
    show_code_comparison()
    
    print("\n\n" + "=" * 70)
    print("✅ 修正完成！")
    print("=" * 70)
    print("\n現在程式正確地只看「z 軸（張數）有沒有標註」")
    print("不受每張標註像素數量的影響。")
    print("=" * 70)


if __name__ == '__main__':
    main()

