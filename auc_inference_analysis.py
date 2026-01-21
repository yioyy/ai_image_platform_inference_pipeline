"""
AUC 推論分析工具
分析從部分數據的 AUC 推論整體 AUC 的可能範圍

問題設定：
- 已推論 29% 資料：正樣本比例 0.443，AUC = 0.62431
- 剩餘 71% 資料：正樣本比例 0.43
- 假設：29% 預測為正的樣本分數 > 71% 的所有分數
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple, Dict
import pandas as pd


class AUCInferenceAnalyzer:
    """AUC 推論分析器"""
    
    def __init__(
        self,
        known_ratio: float = 0.29,
        known_pos_ratio: float = 0.443,
        known_auc: float = 0.62431,
        unknown_pos_ratio: float = 0.43
    ):
        """
        初始化分析器
        
        Args:
            known_ratio: 已知數據比例 (29%)
            known_pos_ratio: 已知數據中正樣本比例 (0.443)
            known_auc: 已知數據的 AUC (0.62431)
            unknown_pos_ratio: 未知數據中正樣本比例 (0.43)
        """
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
        
        print(f"數據分布分析：")
        print(f"已知數據：{self.known_samples} 樣本 (正: {self.known_pos}, 負: {self.known_neg})")
        print(f"未知數據：{self.unknown_samples} 樣本 (正: {self.unknown_pos}, 負: {self.unknown_neg})")
        print(f"整體正樣本比例：{self.overall_pos_ratio:.4f}")
    
    def generate_known_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成符合已知 AUC 的模擬數據
        
        Returns:
            y_true: 真實標籤
            y_scores: 預測分數
        """
        # 生成標籤
        y_true = np.concatenate([
            np.ones(self.known_pos),
            np.zeros(self.known_neg)
        ])
        
        # 生成分數以達到目標 AUC
        # 使用正態分布，調整均值差異來控制 AUC
        pos_mean = 0.6
        neg_mean = 0.4
        
        # 迭代調整以達到目標 AUC
        best_diff = float('inf')
        best_scores = None
        
        for diff in np.linspace(0.1, 0.5, 50):
            pos_scores = np.random.normal(0.5 + diff/2, 0.2, self.known_pos)
            neg_scores = np.random.normal(0.5 - diff/2, 0.2, self.known_neg)
            
            y_scores = np.concatenate([pos_scores, neg_scores])
            y_scores = np.clip(y_scores, 0, 1)
            
            current_auc = roc_auc_score(y_true, y_scores)
            
            if abs(current_auc - self.known_auc) < best_diff:
                best_diff = abs(current_auc - self.known_auc)
                best_scores = y_scores.copy()
        
        return y_true, best_scores
    
    def calculate_extreme_case_auc(self, separation_type: str = "perfect") -> float:
        """
        計算極端情況下的 AUC
        
        Args:
            separation_type: "perfect" (完全分離) 或 "worst" (最差情況)
        
        Returns:
            估計的整體 AUC
        """
        # 生成已知數據
        y_true_known, y_scores_known = self.generate_known_data()
        
        if separation_type == "perfect":
            # 假設：已知數據的正樣本分數 > 未知數據的所有分數
            # 未知數據中，給予完美排序（AUC = 1.0）
            max_known_score = np.max(y_scores_known[y_true_known == 1])
            
            # 未知數據分數都小於已知正樣本的最大分數
            unknown_pos_scores = np.random.uniform(0, max_known_score * 0.9, self.unknown_pos)
            unknown_neg_scores = np.random.uniform(0, max_known_score * 0.8, self.unknown_neg)
            
        elif separation_type == "worst":
            # 最差情況：未知數據隨機
            min_known_score = np.min(y_scores_known)
            unknown_pos_scores = np.random.uniform(0, min_known_score, self.unknown_pos)
            unknown_neg_scores = np.random.uniform(0, min_known_score, self.unknown_neg)
        
        # 組合所有數據
        y_true_all = np.concatenate([
            y_true_known,
            np.ones(self.unknown_pos),
            np.zeros(self.unknown_neg)
        ])
        
        y_scores_all = np.concatenate([
            y_scores_known,
            unknown_pos_scores,
            unknown_neg_scores
        ])
        
        overall_auc = roc_auc_score(y_true_all, y_scores_all)
        return overall_auc
    
    def estimate_auc_range(self, n_simulations: int = 1000) -> Dict:
        """
        通過蒙特卡羅模擬估計 AUC 的可能範圍
        
        Args:
            n_simulations: 模擬次數
        
        Returns:
            包含 AUC 統計信息的字典
        """
        auc_results = []
        
        for _ in range(n_simulations):
            # 生成已知數據
            y_true_known, y_scores_known = self.generate_known_data()
            
            # 對未知數據，假設不同程度的分數分離
            # 隨機選擇分離程度
            separation_factor = np.random.uniform(0.3, 0.9)
            
            # 未知數據的分數分布
            unknown_auc = np.random.uniform(0.5, 0.8)  # 未知數據內部的 AUC
            
            # 生成未知數據分數
            pos_mean = 0.5 + (unknown_auc - 0.5)
            neg_mean = 0.5 - (unknown_auc - 0.5)
            
            unknown_pos_scores = np.random.normal(pos_mean, 0.15, self.unknown_pos)
            unknown_neg_scores = np.random.normal(neg_mean, 0.15, self.unknown_neg)
            
            # 應用分離：未知數據分數整體偏低
            shift = -separation_factor * 0.3
            unknown_pos_scores = np.clip(unknown_pos_scores + shift, 0, 1)
            unknown_neg_scores = np.clip(unknown_neg_scores + shift, 0, 1)
            
            # 組合數據
            y_true_all = np.concatenate([
                y_true_known,
                np.ones(self.unknown_pos),
                np.zeros(self.unknown_neg)
            ])
            
            y_scores_all = np.concatenate([
                y_scores_known,
                unknown_pos_scores,
                unknown_neg_scores
            ])
            
            overall_auc = roc_auc_score(y_true_all, y_scores_all)
            auc_results.append(overall_auc)
        
        auc_results = np.array(auc_results)
        
        return {
            'mean': np.mean(auc_results),
            'median': np.median(auc_results),
            'std': np.std(auc_results),
            'min': np.min(auc_results),
            'max': np.max(auc_results),
            'q25': np.percentile(auc_results, 25),
            'q75': np.percentile(auc_results, 75),
            'ci_lower': np.percentile(auc_results, 2.5),
            'ci_upper': np.percentile(auc_results, 97.5),
            'all_results': auc_results
        }
    
    def theoretical_bounds(self) -> Tuple[float, float]:
        """
        計算理論上的 AUC 邊界
        
        Returns:
            (下界, 上界)
        """
        # 下界：未知數據完全隨機 (AUC = 0.5)
        # 上界：未知數據完美分類 (AUC = 1.0)
        
        # 加權平均的下界
        lower_bound = (self.known_ratio * self.known_auc + 
                       self.unknown_ratio * 0.5)
        
        # 加權平均的上界
        upper_bound = (self.known_ratio * self.known_auc + 
                       self.unknown_ratio * 1.0)
        
        return lower_bound, upper_bound
    
    def visualize_auc_distribution(self, results: Dict):
        """視覺化 AUC 分布"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. AUC 直方圖
        ax1 = axes[0, 0]
        ax1.hist(results['all_results'], bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(results['mean'], color='red', linestyle='--', 
                    label=f"平均: {results['mean']:.4f}")
        ax1.axvline(results['median'], color='green', linestyle='--', 
                    label=f"中位數: {results['median']:.4f}")
        ax1.axvline(self.known_auc, color='blue', linestyle='--', 
                    label=f"已知 AUC: {self.known_auc:.4f}")
        ax1.set_xlabel('AUC')
        ax1.set_ylabel('頻率')
        ax1.set_title('整體 AUC 分布（蒙特卡羅模擬）')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. 箱型圖
        ax2 = axes[0, 1]
        ax2.boxplot([results['all_results']], labels=['整體 AUC'])
        ax2.axhline(self.known_auc, color='blue', linestyle='--', 
                    label=f"已知 AUC: {self.known_auc:.4f}")
        ax2.set_ylabel('AUC')
        ax2.set_title('AUC 箱型圖')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. 累積分布函數
        ax3 = axes[1, 0]
        sorted_auc = np.sort(results['all_results'])
        cumulative = np.arange(1, len(sorted_auc) + 1) / len(sorted_auc)
        ax3.plot(sorted_auc, cumulative, linewidth=2)
        ax3.axvline(results['ci_lower'], color='red', linestyle='--', 
                    label=f"95% CI 下界: {results['ci_lower']:.4f}")
        ax3.axvline(results['ci_upper'], color='red', linestyle='--', 
                    label=f"95% CI 上界: {results['ci_upper']:.4f}")
        ax3.set_xlabel('AUC')
        ax3.set_ylabel('累積機率')
        ax3.set_title('累積分布函數')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. 統計摘要
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        統計摘要
        ═══════════════════════
        
        已知數據 (29%):
        - 樣本數: {self.known_samples}
        - 正樣本比例: {self.known_pos_ratio:.4f}
        - AUC: {self.known_auc:.4f}
        
        未知數據 (71%):
        - 樣本數: {self.unknown_samples}
        - 正樣本比例: {self.unknown_pos_ratio:.4f}
        
        整體 AUC 估計:
        - 平均: {results['mean']:.4f} ± {results['std']:.4f}
        - 中位數: {results['median']:.4f}
        - 範圍: [{results['min']:.4f}, {results['max']:.4f}]
        - 95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]
        - Q25-Q75: [{results['q25']:.4f}, {results['q75']:.4f}]
        
        理論邊界:
        - 下界 (隨機): {self.theoretical_bounds()[0]:.4f}
        - 上界 (完美): {self.theoretical_bounds()[1]:.4f}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('auc_inference_analysis.png', dpi=300, bbox_inches='tight')
        print("\n圖表已保存為 'auc_inference_analysis.png'")
        plt.show()
    
    def generate_report(self) -> pd.DataFrame:
        """生成詳細報告"""
        print("\n" + "="*70)
        print("AUC 推論分析報告")
        print("="*70)
        
        # 理論邊界
        lower_bound, upper_bound = self.theoretical_bounds()
        print(f"\n1. 理論邊界分析：")
        print(f"   最悲觀情況 (未知數據隨機): AUC ≈ {lower_bound:.4f}")
        print(f"   最樂觀情況 (未知數據完美): AUC ≈ {upper_bound:.4f}")
        
        # 蒙特卡羅模擬
        print(f"\n2. 蒙特卡羅模擬 (1000 次)：")
        results = self.estimate_auc_range(n_simulations=1000)
        
        print(f"   整體 AUC 估計:")
        print(f"   - 均值: {results['mean']:.4f} (標準差: {results['std']:.4f})")
        print(f"   - 中位數: {results['median']:.4f}")
        print(f"   - 95% 信賴區間: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
        print(f"   - 四分位距: [{results['q25']:.4f}, {results['q75']:.4f}]")
        
        # 不同假設下的情況
        print(f"\n3. 不同分離假設下的 AUC：")
        
        scenarios = []
        for separation in ['perfect', 'worst']:
            auc = self.calculate_extreme_case_auc(separation)
            scenario_name = "完全分離" if separation == 'perfect' else "最差情況"
            print(f"   - {scenario_name}: {auc:.4f}")
            scenarios.append({'情況': scenario_name, 'AUC': auc})
        
        # 實際建議
        print(f"\n4. 實際推論建議：")
        print(f"   基於你的假設（29% 預測正樣本 > 71% 所有樣本）：")
        print(f"   - 如果完全成立，整體 AUC 可能在 {results['ci_lower']:.4f} 到 {results['ci_upper']:.4f}")
        print(f"   - 但這是極端假設，實際情況可能更保守")
        print(f"   - 建議估計區間：[{results['q25']:.4f}, {results['median']:.4f}]")
        
        print(f"\n5. 關鍵結論：")
        print(f"   ✓ 你的假設是可計算的")
        print(f"   ✓ 在非極端排序下，整體 AUC 很可能在 {results['mean']-results['std']:.4f} 到 {results['mean']+results['std']:.4f}")
        print(f"   ✓ 71% 未知數據的表現對整體 AUC 影響很大（占比高）")
        print(f"   ✓ 建議：在實際提交前，盡可能獲取更多未知數據的反饋")
        
        print("="*70 + "\n")
        
        # 視覺化
        self.visualize_auc_distribution(results)
        
        # 返回摘要
        summary_df = pd.DataFrame({
            '指標': ['已知 AUC', '估計平均', '估計中位數', 'CI 下界', 'CI 上界', 
                   '理論下界', '理論上界'],
            '值': [
                self.known_auc,
                results['mean'],
                results['median'],
                results['ci_lower'],
                results['ci_upper'],
                lower_bound,
                upper_bound
            ]
        })
        
        return summary_df


def main():
    """主函數"""
    print("Kaggle 競賽 AUC 推論分析")
    print("-" * 70)
    
    # 初始化分析器
    analyzer = AUCInferenceAnalyzer(
        known_ratio=0.29,
        known_pos_ratio=0.443,
        known_auc=0.62431,
        unknown_pos_ratio=0.43
    )
    
    # 生成報告
    summary_df = analyzer.generate_report()
    
    print("\n摘要表格：")
    print(summary_df.to_string(index=False))
    
    # 保存結果
    summary_df.to_csv('auc_inference_summary.csv', index=False, encoding='utf-8-sig')
    print("\n結果已保存至 'auc_inference_summary.csv'")


if __name__ == "__main__":
    main()

