"""
修正版：AUC 反向推論
更精確的數學分析
"""

import numpy as np
from typing import Dict


def analyze_auc_decomposition():
    """
    精確分析 AUC 分解問題
    
    已知：
    - 29% 用模型 + 71% 用預設 -> AUC = 0.62431
    - 100% 用預設 -> AUC = 0.53441
    
    求：29% 真實模型 AUC
    """
    
    print("=" * 80)
    print("AUC 反向推論 - 精確分析")
    print("=" * 80)
    
    # 參數設置
    p_29 = 0.29  # 29% 使用模型
    p_71 = 0.71  # 71% 使用預設
    pos_ratio_29 = 0.443
    pos_ratio_71 = 0.43
    
    # 計算整體正樣本比例
    pos_ratio_all = p_29 * pos_ratio_29 + p_71 * pos_ratio_71
    
    print(f"\n[問題設定]")
    print(f"29% 數據: 正樣本比例 = {pos_ratio_29:.3f}, 使用模型預測")
    print(f"71% 數據: 正樣本比例 = {pos_ratio_71:.2f}, 使用預設值")
    print(f"整體正樣本比例: {pos_ratio_all:.3f}")
    
    # 觀測值
    auc_mixed = 0.62431  # 29% 模型 + 71% 預設
    auc_default = 0.53441  # 100% 預設
    
    print(f"\n[觀測到的 AUC]")
    print(f"混合策略 (29% 模型 + 71% 預設): {auc_mixed:.5f}")
    print(f"純預設值策略: {auc_default:.5f}")
    print(f"差異: {auc_mixed - auc_default:.5f}")
    
    # === 方法 1: 簡化線性模型 ===
    print(f"\n{'='*80}")
    print(f"[方法 1: 線性近似 - 上界估計]")
    print(f"{'='*80}")
    
    print(f"\n假設: AUC 可以近似為各部分的加權平均")
    print(f"      mixed_auc ~= p_29 * auc_29 + p_71 * auc_71")
    print(f"\n其中:")
    print(f"  - auc_29 = 29% 數據的真實模型 AUC (未知)")
    print(f"  - auc_71 = 71% 數據使用預設值的 AUC")
    print(f"\n關鍵洞察:")
    print(f"  當 71% 使用預設值時，這部分的 AUC 應該接近全預設的 AUC")
    print(f"  因為預設值對所有樣本給出相同的預測分布")
    
    # 假設 71% 用預設值時，AUC 接近 auc_default
    auc_71_estimate = auc_default
    
    # 解出 auc_29
    auc_29_linear = (auc_mixed - p_71 * auc_71_estimate) / p_29
    
    print(f"\n計算:")
    print(f"  {auc_mixed:.5f} = {p_29:.2f} * auc_29 + {p_71:.2f} * {auc_71_estimate:.5f}")
    print(f"  auc_29 = ({auc_mixed:.5f} - {p_71:.2f} * {auc_71_estimate:.5f}) / {p_29:.2f}")
    print(f"  auc_29 = {auc_29_linear:.5f}")
    
    print(f"\n結論: 29% 數據真實模型 AUC ~= {auc_29_linear:.5f}")
    print(f"\n注意: 這是線性近似，給出的是理論上界")
    
    # === 方法 2: 基於 AUC 定義的精確分析 ===
    print(f"\n{'='*80}")
    print(f"[方法 2: 基於排序對的精確分析]")
    print(f"{'='*80}")
    
    print(f"\nAUC 的定義: 隨機正樣本排在隨機負樣本前面的機率")
    print(f"\n將數據分為 4 個區域:")
    print(f"  A: 29% 中的正樣本 (比例: {p_29 * pos_ratio_29:.3f})")
    print(f"  B: 29% 中的負樣本 (比例: {p_29 * (1 - pos_ratio_29):.3f})")
    print(f"  C: 71% 中的正樣本 (比例: {p_71 * pos_ratio_71:.3f})")
    print(f"  D: 71% 中的負樣本 (比例: {p_71 * (1 - pos_ratio_71):.3f})")
    
    n_A = p_29 * pos_ratio_29
    n_B = p_29 * (1 - pos_ratio_29)
    n_C = p_71 * pos_ratio_71
    n_D = p_71 * (1 - pos_ratio_71)
    
    print(f"\n所有正負樣本對:")
    print(f"  (A, B): 29% 內部的正負對")
    print(f"  (A, D): 29% 正樣本 vs 71% 負樣本")
    print(f"  (C, B): 71% 正樣本 vs 29% 負樣本")
    print(f"  (C, D): 71% 內部的正負對")
    
    print(f"\n混合策略下各對的貢獻:")
    print(f"  P(A > B) = auc_29 (未知)")
    print(f"  P(A > D) = ? (取決於 29% 模型分數 vs 71% 預設分數的分布)")
    print(f"  P(C > B) = ? (取決於 71% 預設分數 vs 29% 模型分數的分布)")
    print(f"  P(C > D) = {auc_default:.5f} (71% 都用預設值)")
    
    print(f"\n關鍵假設:")
    print(f"  如果 29% 和 71% 的分數分布相互獨立")
    print(f"  則 P(A > D) ~= P(C > B) ~= 0.5")
    
    # 基於獨立性假設
    # total_auc = [n_A*n_B * auc_29 + n_A*n_D * 0.5 + n_C*n_B * 0.5 + n_C*n_D * auc_default] / total_pairs
    total_pos = n_A + n_C
    total_neg = n_B + n_D
    total_pairs = total_pos * total_neg
    
    # auc_mixed = [n_A*n_B * auc_29 + n_A*n_D * 0.5 + n_C*n_B * 0.5 + n_C*n_D * auc_default] / total_pairs
    # 解出 auc_29
    
    known_contribution = (n_A * n_D * 0.5 + n_C * n_B * 0.5 + n_C * n_D * auc_default) / total_pairs
    auc_29_contribution_weight = (n_A * n_B) / total_pairs
    
    auc_29_independence = (auc_mixed - known_contribution) / auc_29_contribution_weight
    
    print(f"\n計算（獨立性假設）:")
    print(f"  正負對總數: {total_pairs:.4f}")
    print(f"  29% 內部對權重: {auc_29_contribution_weight:.4f}")
    print(f"  已知貢獻: {known_contribution:.4f}")
    print(f"  auc_29 = ({auc_mixed:.5f} - {known_contribution:.5f}) / {auc_29_contribution_weight:.4f}")
    print(f"  auc_29 = {auc_29_independence:.5f}")
    
    print(f"\n結論: 29% 數據真實模型 AUC ~= {auc_29_independence:.5f}")
    
    # === 方法 3: 考慮你的假設 ===
    print(f"\n{'='*80}")
    print(f"[方法 3: 基於分離假設的分析]")
    print(f"{'='*80}")
    
    print(f"\n你的假設: 「29% 模型認為是正樣本的值 > 71% 所有值」")
    print(f"\n這意味著:")
    print(f"  P(A > D) = 1.0 (29% 正樣本完全大於 71% 負樣本)")
    print(f"  P(A > C) = 1.0 (29% 正樣本甚至大於 71% 正樣本)")
    
    print(f"\n但這會導致:")
    separation_contribution = (n_A * n_D * 1.0 + n_A * n_C * 1.0 + n_C * n_B * 0.0 + n_C * n_D * auc_default) / total_pairs
    auc_29_separation_needed = (auc_mixed - separation_contribution) / auc_29_contribution_weight
    
    print(f"  理論混合 AUC = {separation_contribution + auc_29_contribution_weight * 1.0:.5f}")
    print(f"  (如果 29% 內部也是完美的)")
    
    print(f"\n但實際觀測到的混合 AUC = {auc_mixed:.5f}")
    print(f"說明: 完全分離假設與觀測不符")
    
    # === 綜合結論 ===
    print(f"\n{'='*80}")
    print(f"[最終結論與建議]")
    print(f"{'='*80}")
    
    # 取方法 1 和方法 2 的平均作為最佳估計
    best_estimate = (auc_29_linear + auc_29_independence) / 2
    
    print(f"\n[!] 29% 數據真實模型 AUC 估計:")
    print(f"    方法 1 (線性近似):      {auc_29_linear:.5f}")
    print(f"    方法 2 (獨立性假設):    {auc_29_independence:.5f}")
    print(f"    綜合估計 (平均):        {best_estimate:.5f}")
    
    improvement = best_estimate - auc_default
    print(f"\n[!] 相比預設值的提升:")
    print(f"    絕對提升: +{improvement:.5f}")
    print(f"    相對提升: +{(improvement / auc_default * 100):.1f}%")
    
    # 性能評估
    print(f"\n[!] 性能評估:")
    if best_estimate < 0.60:
        level = "較弱"
        recommendation = "建議重點檢查數據質量、特徵工程和模型選擇"
    elif best_estimate < 0.70:
        level = "中等"
        recommendation = "有提升空間，考慮 ensemble、特徵優化或超參數調整"
    elif best_estimate < 0.80:
        level = "良好"
        recommendation = "表現不錯，確保模型的泛化能力和穩定性"
    else:
        level = "優秀"
        recommendation = "表現優異，注意避免過擬合"
    
    print(f"    等級: {level}")
    print(f"    建議: {recommendation}")
    
    # 關於假設的討論
    print(f"\n[!] 關於「29% 正樣本 > 71% 所有值」的假設:")
    print(f"    如果此假設成立:")
    print(f"      - 理論混合 AUC 會遠高於 {auc_mixed:.5f}")
    print(f"      - 估計會達到 0.75 以上")
    print(f"    ")
    print(f"    實際情況:")
    print(f"      - 混合 AUC 只有 {auc_mixed:.5f}")
    print(f"      - 說明 29% 和 71% 之間有顯著重疊")
    print(f"      - 完全分離的假設不成立")
    
    # Kaggle 策略
    print(f"\n[!] Kaggle 競賽策略:")
    print(f"    當前 Public LB: {auc_mixed:.5f} (29% 模型 + 71% 預設)")
    print(f"    ")
    print(f"    如果能在 71% 數據上應用模型:")
    print(f"      - 樂觀情況 (達到 29% 表現): AUC ~= {best_estimate:.3f}")
    print(f"      - 保守情況 (考慮泛化損失): AUC ~= {best_estimate - 0.05:.3f}")
    print(f"    ")
    print(f"    改進方向:")
    print(f"      1. 目標: 提升模型在 29% 上的 AUC 至 0.75+")
    print(f"      2. 分析: 為何只比預設值好 {improvement:.3f}？")
    print(f"      3. 檢查: 特徵是否有足夠的預測力？")
    
    # 視覺化
    print(f"\n{'='*80}")
    print(f"AUC 對比視覺化")
    print(f"{'='*80}\n")
    
    def print_bar(label, value, width=50):
        bar_length = int((value - 0.5) * width / 0.5)  # 0.5-1.0 映射到 0-50
        bar = '=' * bar_length
        print(f"{label:25s} |{bar:<{width}}| {value:.5f}")
    
    print_bar("預設值 AUC", auc_default)
    print_bar("混合策略 AUC", auc_mixed)
    print_bar("29% 真實模型 AUC", best_estimate)
    print_bar("目標 AUC", 0.75)
    print(f"\n{'':25s}  0.50{'':10s}0.60{'':10s}0.70{'':10s}0.80{'':10s}0.90{'':5s}1.00")
    
    print(f"\n{'='*80}\n")
    
    return {
        'auc_29_linear': auc_29_linear,
        'auc_29_independence': auc_29_independence,
        'best_estimate': best_estimate,
        'improvement': improvement
    }


if __name__ == "__main__":
    result = analyze_auc_decomposition()

