"""
AUC 反向推論工具
從混合預測結果反推部分數據的真實模型表現

問題設定：
- 29% 用模型預測 + 71% 用分布預設值 -> 總 AUC = 0.62431
- 100% 全用分布預設值 -> AUC = 0.53441
- 反推：29% 資料的真實模型 AUC 是多少？
"""

import numpy as np
from typing import Tuple, Dict
from scipy.optimize import minimize_scalar


class AUCReverseInference:
    """AUC 反向推論分析器"""
    
    def __init__(
        self,
        known_ratio: float = 0.29,
        known_pos_ratio: float = 0.443,
        unknown_pos_ratio: float = 0.43,
        mixed_auc: float = 0.62431,
        default_auc: float = 0.53441
    ):
        """
        初始化
        
        Args:
            known_ratio: 使用模型的數據比例（29%）
            known_pos_ratio: 29% 數據中正樣本比例
            unknown_pos_ratio: 71% 數據中正樣本比例
            mixed_auc: 混合策略的 AUC（29% 模型 + 71% 預設）
            default_auc: 全用預設值的 AUC
        """
        self.known_ratio = known_ratio
        self.known_pos_ratio = known_pos_ratio
        self.unknown_pos_ratio = unknown_pos_ratio
        self.unknown_ratio = 1 - known_ratio
        self.mixed_auc = mixed_auc
        self.default_auc = default_auc
        
        # 計算樣本數（假設 10000）
        self.total_samples = 10000
        self.known_samples = int(self.total_samples * self.known_ratio)
        self.unknown_samples = self.total_samples - self.known_samples
        
        self.known_pos = int(self.known_samples * self.known_pos_ratio)
        self.known_neg = self.known_samples - self.known_pos
        self.unknown_pos = int(self.unknown_samples * self.unknown_pos_ratio)
        self.unknown_neg = self.unknown_samples - self.unknown_pos
        
        # 整體正樣本比例
        total_pos = self.known_pos + self.unknown_pos
        self.overall_pos_ratio = total_pos / self.total_samples
    
    def calculate_auc_manual(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """手動計算 AUC"""
        order = np.argsort(-y_scores)
        y_true_sorted = y_true[order]
        
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        rank_sum = 0
        for i, label in enumerate(y_true_sorted):
            if label == 1:
                rank_sum += i + 1
        
        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return auc
    
    def simulate_mixed_scenario(self, model_auc_29: float) -> float:
        """
        模擬混合場景
        
        Args:
            model_auc_29: 假設的 29% 資料真實模型 AUC
        
        Returns:
            模擬得到的混合 AUC
        """
        # 生成 29% 數據（使用模型）
        y_true_29 = np.concatenate([
            np.ones(self.known_pos),
            np.zeros(self.known_neg)
        ])
        
        # 根據假設的 AUC 生成分數
        pos_mean = 0.5 + (model_auc_29 - 0.5)
        neg_mean = 0.5 - (model_auc_29 - 0.5)
        
        y_scores_29_pos = np.random.normal(pos_mean, 0.15, self.known_pos)
        y_scores_29_neg = np.random.normal(neg_mean, 0.15, self.known_neg)
        y_scores_29 = np.concatenate([y_scores_29_pos, y_scores_29_neg])
        y_scores_29 = np.clip(y_scores_29, 0, 1)
        
        # 生成 71% 數據（使用預設值 = 隨機）
        y_true_71 = np.concatenate([
            np.ones(self.unknown_pos),
            np.zeros(self.unknown_neg)
        ])
        
        # 預設值：接近 0.5 的隨機分數（對應 default_auc = 0.534）
        y_scores_71 = np.random.uniform(0.3, 0.7, self.unknown_samples)
        
        # 組合所有數據
        y_true_all = np.concatenate([y_true_29, y_true_71])
        y_scores_all = np.concatenate([y_scores_29, y_scores_71])
        
        # 計算整體 AUC
        mixed_auc_sim = self.calculate_auc_manual(y_true_all, y_scores_all)
        return mixed_auc_sim
    
    def find_true_model_auc(self, n_iterations: int = 50) -> Dict:
        """
        通過數值搜索找到真實的 29% 模型 AUC
        
        Returns:
            包含推斷結果的字典
        """
        results = []
        
        # 測試不同的 AUC 值
        test_auc_values = np.linspace(0.55, 0.95, n_iterations)
        
        for test_auc in test_auc_values:
            # 多次模擬取平均
            sim_aucs = []
            for _ in range(20):
                sim_auc = self.simulate_mixed_scenario(test_auc)
                sim_aucs.append(sim_auc)
            
            avg_sim_auc = np.mean(sim_aucs)
            error = abs(avg_sim_auc - self.mixed_auc)
            
            results.append({
                'test_model_auc': test_auc,
                'simulated_mixed_auc': avg_sim_auc,
                'error': error
            })
        
        # 找到誤差最小的
        best_result = min(results, key=lambda x: x['error'])
        
        return {
            'estimated_model_auc': best_result['test_model_auc'],
            'simulated_mixed_auc': best_result['simulated_mixed_auc'],
            'error': best_result['error'],
            'all_results': results
        }
    
    def analytical_lower_bound(self) -> float:
        """
        理論下界分析
        
        如果假設 29% 和 71% 完全獨立，可以用分解公式
        """
        # 簡化模型：假設 AUC 可以近似分解
        # mixed_auc ≈ w1 * auc_29 + w2 * auc_71 + interaction_term
        # 其中 auc_71 ≈ default_auc（因為用的是預設值）
        
        # 最簡單的線性近似
        # 0.62431 ≈ 0.29 * auc_29 + 0.71 * 0.534
        # auc_29 ≈ (0.62431 - 0.71 * 0.534) / 0.29
        
        estimated_auc_29 = (self.mixed_auc - self.unknown_ratio * self.default_auc) / self.known_ratio
        return estimated_auc_29
    
    def separation_assumption_bound(self) -> Tuple[float, float]:
        """
        基於分離假設的界限
        
        假設：29% 正樣本 > 71% 所有樣本
        """
        # 上界：如果 29% 內部完美分類
        # 29% 的所有正樣本排在最前面，然後是 29% 負樣本，最後是 71%
        
        # 計算跨區域的貢獻
        # 29% 正樣本 vs 71% 所有樣本：完全正確
        cross_contribution_1 = self.known_pos * self.unknown_samples
        
        # 29% 負樣本 vs 71% 正樣本：這取決於 71% 內部的排序
        # 如果 71% 內部 AUC = 0.534，說明有些混亂
        cross_contribution_2 = self.known_neg * self.unknown_pos * self.default_auc
        
        # 29% 負樣本 vs 71% 負樣本：假設隨機
        cross_contribution_3 = self.known_neg * self.unknown_neg * 0.5
        
        # 29% 內部的貢獻
        # 假設 29% 內部 AUC = x
        # 總正確配對數 = x * known_pos * known_neg + cross_contributions
        
        # 這個計算比較複雜，我們用模擬來估計
        return None, None
    
    def monte_carlo_estimation(self, n_sim: int = 100) -> Dict:
        """蒙特卡羅估計"""
        auc_estimates = []
        
        for _ in range(n_sim):
            result = self.find_true_model_auc(n_iterations=30)
            auc_estimates.append(result['estimated_model_auc'])
        
        return {
            'mean': np.mean(auc_estimates),
            'median': np.median(auc_estimates),
            'std': np.std(auc_estimates),
            'min': np.min(auc_estimates),
            'max': np.max(auc_estimates),
            'q25': np.percentile(auc_estimates, 25),
            'q75': np.percentile(auc_estimates, 75)
        }


def main():
    """主函數"""
    print("=" * 80)
    print("AUC 反向推論分析")
    print("從混合預測結果反推真實模型表現")
    print("=" * 80)
    
    analyzer = AUCReverseInference(
        known_ratio=0.29,
        known_pos_ratio=0.443,
        unknown_pos_ratio=0.43,
        mixed_auc=0.62431,
        default_auc=0.53441
    )
    
    print(f"\n[數據配置]")
    print(f"29% 數據使用模型預測")
    print(f"  - 正樣本比例: {analyzer.known_pos_ratio:.3f}")
    print(f"\n71% 數據使用分布預設值")
    print(f"  - 正樣本比例: {analyzer.unknown_pos_ratio:.2f}")
    print(f"\n觀測到的 AUC:")
    print(f"  - 混合策略 (29% 模型 + 71% 預設): {analyzer.mixed_auc:.5f}")
    print(f"  - 全用預設值: {analyzer.default_auc:.5f}")
    
    # 理論下界（線性近似）
    print(f"\n{'='*80}")
    print(f"[1. 理論分析 - 線性近似]")
    print(f"{'='*80}")
    
    linear_estimate = analyzer.analytical_lower_bound()
    print(f"\n假設 AUC 可以線性分解:")
    print(f"  mixed_auc = 0.29 * auc_29 + 0.71 * auc_71")
    print(f"  0.62431 = 0.29 * auc_29 + 0.71 * {analyzer.default_auc:.5f}")
    print(f"\n解出: auc_29 = {linear_estimate:.5f}")
    print(f"\n注意: 這是簡化的線性近似，實際 AUC 不完全滿足線性關係")
    
    # 數值搜索
    print(f"\n{'='*80}")
    print(f"[2. 數值模擬反推]")
    print(f"{'='*80}")
    print(f"\n正在進行數值搜索...")
    
    search_result = analyzer.find_true_model_auc(n_iterations=50)
    
    print(f"\n通過模擬找到的最佳匹配:")
    print(f"  29% 數據真實模型 AUC: {search_result['estimated_model_auc']:.5f}")
    print(f"  對應的模擬混合 AUC: {search_result['simulated_mixed_auc']:.5f}")
    print(f"  與實際混合 AUC 的誤差: {search_result['error']:.5f}")
    
    # 置信區間估計
    print(f"\n正在進行蒙特卡羅估計（可能需要 1-2 分鐘）...")
    # mc_result = analyzer.monte_carlo_estimation(n_sim=20)
    
    # 由於計算時間較長，這裡用快速估計
    quick_estimates = []
    for _ in range(10):
        result = analyzer.find_true_model_auc(n_iterations=30)
        quick_estimates.append(result['estimated_model_auc'])
    
    mc_result = {
        'mean': np.mean(quick_estimates),
        'std': np.std(quick_estimates),
        'min': np.min(quick_estimates),
        'max': np.max(quick_estimates)
    }
    
    print(f"\n置信區間估計（10 次獨立模擬）:")
    print(f"  平均估計: {mc_result['mean']:.5f} +/- {mc_result['std']:.5f}")
    print(f"  範圍: [{mc_result['min']:.5f}, {mc_result['max']:.5f}]")
    
    # 關鍵結論
    print(f"\n{'='*80}")
    print(f"[3. 關鍵結論]")
    print(f"{'='*80}")
    
    best_estimate = search_result['estimated_model_auc']
    improvement = best_estimate - analyzer.default_auc
    
    print(f"\n[!] 29% 數據的真實模型表現:")
    print(f"    AUC = {best_estimate:.5f}")
    print(f"\n[!] 相比預設值的提升:")
    print(f"    Delta AUC = {improvement:.5f}")
    print(f"    提升幅度 = {(improvement / analyzer.default_auc * 100):.1f}%")
    
    print(f"\n[!] 解讀:")
    if best_estimate < 0.65:
        print(f"    模型在 29% 數據上的表現較弱")
        print(f"    建議: 重點檢查特徵工程和模型選擇")
    elif best_estimate < 0.75:
        print(f"    模型表現中等，有提升空間")
        print(f"    建議: 考慮 ensemble 或優化超參數")
    else:
        print(f"    模型表現良好！")
        print(f"    建議: 確保泛化能力，避免過擬合")
    
    # 假設驗證
    print(f"\n{'='*80}")
    print(f"[4. 關於你的假設]")
    print(f"{'='*80}")
    
    print(f"\n假設: 「29% 模型認為是正樣本的值 > 71% 所有值」")
    print(f"\n這個假設的影響:")
    print(f"  - 如果此假設成立，說明模型有很強的分辨能力")
    print(f"  - 29% 的正樣本會完全排在 71% 數據之前")
    print(f"  - 這會顯著提高整體 AUC")
    
    print(f"\n但從計算結果看:")
    print(f"  - 混合 AUC (0.624) 只比預設值 (0.534) 高 0.09")
    print(f"  - 這暗示 29% 和 71% 之間可能有較大重疊")
    print(f"  - 完全分離的假設可能過於樂觀")
    
    print(f"\n更合理的情境:")
    print(f"  - 29% 模型預測的正樣本整體偏高")
    print(f"  - 但與 71% 數據有部分重疊")
    print(f"  - 建議: 如果能獲得 71% 數據的反饋，可驗證此假設")
    
    # 實務建議
    print(f"\n{'='*80}")
    print(f"[5. Kaggle 競賽策略建議]")
    print(f"{'='*80}")
    
    print(f"\n[!] Public LB 分析:")
    print(f"    當前分數: 0.624 (29% 模型 + 71% 預設)")
    print(f"    29% 真實模型: ~{best_estimate:.3f}")
    
    print(f"\n[!] Private LB 預期:")
    print(f"    如果能在 71% 數據上達到類似表現:")
    print(f"    預期 AUC: {best_estimate:.3f} - 0.05 = {best_estimate - 0.05:.3f} ~ {best_estimate:.3f}")
    print(f"    (考慮可能的泛化損失)")
    
    print(f"\n[!] 改進方向:")
    print(f"    1. 目標: 提升 29% 數據的模型 AUC 至 0.75+")
    print(f"    2. 方法: 特徵工程、模型融合、超參數優化")
    print(f"    3. 驗證: 確保本地 CV 與 Public LB 一致")
    
    print(f"\n[!] 風險提示:")
    print(f"    - 如果 29% 和 71% 分布不同（data drift）")
    print(f"    - Public/Private LB 可能出現 shake-up")
    print(f"    - 建議: 建立穩健的本地驗證策略")
    
    print(f"\n{'='*80}")
    print(f"分析完成！")
    print(f"{'='*80}\n")
    
    # 視覺化結果
    print(f"\nAUC 對比視覺化:")
    print(f"-" * 60)
    print(f"\n預設值 AUC:    |{'='*20}| {analyzer.default_auc:.3f}")
    print(f"混合策略 AUC:  |{'='*25}| {analyzer.mixed_auc:.3f}")
    print(f"29% 真實 AUC:  |{'='*int((best_estimate-0.5)*50)}| {best_estimate:.3f}")
    print(f"\n                0.50      0.60      0.70      0.80      0.90")


if __name__ == "__main__":
    main()

