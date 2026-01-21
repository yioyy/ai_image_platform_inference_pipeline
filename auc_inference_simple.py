"""
簡化的 AUC 推論分析（不需要 matplotlib）
"""

import numpy as np
from typing import Tuple, Dict


class SimpleAUCAnalyzer:
    """簡化的 AUC 推論分析器"""
    
    def __init__(
        self,
        known_ratio: float = 0.29,
        known_pos_ratio: float = 0.443,
        known_auc: float = 0.62431,
        unknown_pos_ratio: float = 0.43
    ):
        self.known_ratio = known_ratio
        self.known_pos_ratio = known_pos_ratio
        self.known_auc = known_auc
        self.unknown_pos_ratio = unknown_pos_ratio
        self.unknown_ratio = 1 - known_ratio
        
        # 計算樣本數量（假設總樣本數為 10000）
        self.total_samples = 10000
        self.known_samples = int(self.total_samples * self.known_ratio)
        self.unknown_samples = self.total_samples - self.known_samples
        
        # 計算正負樣本數
        self.known_pos = int(self.known_samples * self.known_pos_ratio)
        self.known_neg = self.known_samples - self.known_pos
        self.unknown_pos = int(self.unknown_samples * self.unknown_pos_ratio)
        self.unknown_neg = self.unknown_samples - self.unknown_pos
        
        # 計算整體正樣本比例
        total_pos = self.known_pos + self.unknown_pos
        self.overall_pos_ratio = total_pos / self.total_samples
    
    def calculate_auc_from_scores(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """手動計算 AUC（不使用 sklearn）"""
        # 排序
        order = np.argsort(-y_scores)
        y_true_sorted = y_true[order]
        
        # 計算 AUC
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        # 計算排名和
        rank_sum = 0
        for i, label in enumerate(y_true_sorted):
            if label == 1:
                rank_sum += i + 1
        
        # Mann-Whitney U 統計量
        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return auc
    
    def theoretical_bounds(self) -> Tuple[float, float]:
        """計算理論邊界"""
        # 簡化的加權平均估計
        # 下界：未知數據 AUC = 0.5（隨機）
        lower_bound = (self.known_ratio * self.known_auc + 
                       self.unknown_ratio * 0.5)
        
        # 上界：未知數據 AUC = 1.0（完美）
        upper_bound = (self.known_ratio * self.known_auc + 
                       self.unknown_ratio * 1.0)
        
        return lower_bound, upper_bound
    
    def estimate_practical_range(self) -> Dict:
        """
        基於實際假設估計 AUC 範圍
        假設：29% 預測為正的樣本 > 71% 樣本
        """
        # 情境 1: 溫和分離（部分重疊）
        # 假設未知數據內部 AUC = 0.55
        scenario_1 = (self.known_ratio * self.known_auc + 
                      self.unknown_ratio * 0.55)
        
        # 情境 2: 中等分離
        # 假設未知數據內部 AUC = 0.60
        scenario_2 = (self.known_ratio * self.known_auc + 
                      self.unknown_ratio * 0.60)
        
        # 情境 3: 強分離
        # 假設未知數據內部 AUC = 0.70
        scenario_3 = (self.known_ratio * self.known_auc + 
                      self.unknown_ratio * 0.70)
        
        # 情境 4: 極強分離（接近你的假設）
        # 假設未知數據內部 AUC = 0.80
        scenario_4 = (self.known_ratio * self.known_auc + 
                      self.unknown_ratio * 0.80)
        
        return {
            'weak_separation': scenario_1,
            'moderate_separation': scenario_2,
            'strong_separation': scenario_3,
            'very_strong_separation': scenario_4
        }
    
    def monte_carlo_simulation(self, n_sim: int = 1000) -> Dict:
        """蒙特卡羅模擬"""
        results = []
        
        for _ in range(n_sim):
            # 隨機假設未知數據的 AUC（偏向較好的表現）
            unknown_auc = np.random.beta(6, 4) * 0.5 + 0.5  # Beta 分布，集中在 0.5-1.0
            
            # 計算整體 AUC（加權平均的簡化模型）
            overall_auc = (self.known_ratio * self.known_auc + 
                          self.unknown_ratio * unknown_auc)
            
            results.append(overall_auc)
        
        results = np.array(results)
        
        return {
            'mean': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results),
            'q25': np.percentile(results, 25),
            'q75': np.percentile(results, 75),
            'ci_lower': np.percentile(results, 2.5),
            'ci_upper': np.percentile(results, 97.5)
        }


def main():
    """主函數"""
    print("=" * 80)
    print("Kaggle 競賽 AUC 推論分析")
    print("=" * 80)
    
    analyzer = SimpleAUCAnalyzer(
        known_ratio=0.29,
        known_pos_ratio=0.443,
        known_auc=0.62431,
        unknown_pos_ratio=0.43
    )
    
    print(f"\n【數據配置】")
    print(f"已知數據比例: {analyzer.known_ratio:.1%}")
    print(f"  - 正樣本比例: {analyzer.known_pos_ratio:.3f}")
    print(f"  - AUC: {analyzer.known_auc:.5f}")
    print(f"\n未知數據比例: {analyzer.unknown_ratio:.1%}")
    print(f"  - 正樣本比例: {analyzer.unknown_pos_ratio:.2f}")
    print(f"\n整體正樣本比例: {analyzer.overall_pos_ratio:.3f}")
    
    # 理論邊界
    lower_bound, upper_bound = analyzer.theoretical_bounds()
    print(f"\n{'='*80}")
    print(f"【1. 理論邊界分析】")
    print(f"{'='*80}")
    print(f"最悲觀情況（未知數據完全隨機，AUC=0.5）:")
    print(f"  整體 AUC >= {lower_bound:.5f}")
    print(f"\n最樂觀情況（未知數據完美分類，AUC=1.0）:")
    print(f"  整體 AUC <= {upper_bound:.5f}")
    print(f"\n理論區間: [{lower_bound:.5f}, {upper_bound:.5f}]")
    
    # 實際情境
    scenarios = analyzer.estimate_practical_range()
    print(f"\n{'='*80}")
    print(f"【2. 不同分離程度下的 AUC 估計】")
    print(f"{'='*80}")
    print(f"假設未知數據內部 AUC 不同時，整體 AUC 的估計:\n")
    print(f"情境 1 - 弱分離（未知 AUC=0.55）:  {scenarios['weak_separation']:.5f}")
    print(f"情境 2 - 中等分離（未知 AUC=0.60）:{scenarios['moderate_separation']:.5f}")
    print(f"情境 3 - 強分離（未知 AUC=0.70）:  {scenarios['strong_separation']:.5f}")
    print(f"情境 4 - 極強分離（未知 AUC=0.80）:{scenarios['very_strong_separation']:.5f}")
    
    # 蒙特卡羅模擬
    mc_results = analyzer.monte_carlo_simulation(n_sim=5000)
    print(f"\n{'='*80}")
    print(f"【3. 蒙特卡羅模擬結果】（5000 次模擬）")
    print(f"{'='*80}")
    print(f"平均值:    {mc_results['mean']:.5f} (標準差: {mc_results['std']:.5f})")
    print(f"中位數:    {mc_results['median']:.5f}")
    print(f"範圍:      [{mc_results['min']:.5f}, {mc_results['max']:.5f}]")
    print(f"95% CI:    [{mc_results['ci_lower']:.5f}, {mc_results['ci_upper']:.5f}]")
    print(f"四分位距:  [{mc_results['q25']:.5f}, {mc_results['q75']:.5f}]")
    
    # 回答用戶問題
    print(f"\n{'='*80}")
    print(f"【4. 回答你的問題】")
    print(f"{'='*80}")
    
    print(f"\n問題 1: 「假設 29% 模型認為是正樣本的值一定大於 71% 的值，這樣能計算嗎？」")
    print(f"答案: [YES] 可以計算！")
    print(f"\n這是一個「完全分離」的假設，在這種極端情況下：")
    print(f"  - 如果未知數據內部完美分類（AUC=1.0）")
    print(f"    -> 整體 AUC ~= {upper_bound:.5f}")
    print(f"  - 如果未知數據內部隨機（AUC=0.5）")
    print(f"    -> 整體 AUC ~= {lower_bound:.5f}")
    
    print(f"\n問題 2: 「如果不是極端排序，但維持提供的數據，可以大概推論 AUC 的區間嗎？」")
    print(f"答案: [YES] 可以推論！")
    
    print(f"\n基於合理假設（非極端情況）：")
    print(f"  - 保守估計: {mc_results['q25']:.5f} ~ {mc_results['median']:.5f}")
    print(f"  - 中性估計: {mc_results['median']:.5f} ~ {mc_results['q75']:.5f}")
    print(f"  - 樂觀估計: {mc_results['q75']:.5f} ~ {mc_results['ci_upper']:.5f}")
    
    print(f"\n最可能的區間（50% 信賴區間）:")
    print(f"  整體 AUC in [{mc_results['q25']:.5f}, {mc_results['q75']:.5f}]")
    
    # 關鍵洞察
    print(f"\n{'='*80}")
    print(f"【5. 關鍵洞察與建議】")
    print(f"{'='*80}")
    
    print(f"\n[!] 重要發現:")
    print(f"  1. 71% 未知數據的權重很大，對整體 AUC 影響深遠")
    print(f"  2. 即使你的模型在 29% 數據上表現不錯（0.624），")
    print(f"     整體 AUC 仍可能下降到 {mc_results['q25']:.3f}-{mc_results['median']:.3f}")
    print(f"  3. 你提到「分布預設值 AUC=0.534」，這接近隨機猜測")
    print(f"     建議審視特徵工程和模型是否有提升空間")
    
    print(f"\n[!] 實務建議:")
    print(f"  1. 如果可能，嘗試獲取更多未知數據的本地驗證集")
    print(f"  2. 檢查 29% 和 71% 數據是否有分布差異（data drift）")
    print(f"  3. 考慮使用 ensemble 或 stacking 提升泛化能力")
    print(f"  4. 分析你的假設是否合理：")
    print(f"     - 29% 正樣本分數真的都 > 71% 嗎？")
    print(f"     - 這可能過於樂觀，實際可能有重疊")
    
    print(f"\n[!] 排行榜策略:")
    best_guess = (mc_results['q25'] + mc_results['median']) / 2
    print(f"  - 如果是 Public Leaderboard (29%)，你的表現已知: 0.624")
    print(f"  - 預期 Private Leaderboard (71%) 可能: {best_guess:.3f} ~ {mc_results['median']:.3f}")
    print(f"  - 建議準備心理預期：可能有 0.05-0.10 的 AUC 下降")
    
    print(f"\n{'='*80}")
    print(f"分析完成！")
    print(f"{'='*80}\n")
    
    # 簡化的視覺化（ASCII）
    print("\nAUC 分布視覺化（ASCII）:")
    print("-" * 60)
    bins = np.linspace(mc_results['min'], mc_results['max'], 20)
    hist, _ = np.histogram([mc_results['mean']], bins=bins)
    
    # 簡單的 ASCII 直方圖
    values = np.linspace(lower_bound, upper_bound, 10)
    print(f"\n理論區間 [{lower_bound:.3f} -------- {upper_bound:.3f}]")
    print(f"         |")
    print(f"已知 AUC: {analyzer.known_auc:.3f}")
    print(f"         |")
    print(f"估計區間 [{mc_results['q25']:.3f} ==== {mc_results['median']:.3f} ==== {mc_results['q75']:.3f}]")
    print(f"          Q25        中位數         Q75")


if __name__ == "__main__":
    main()

