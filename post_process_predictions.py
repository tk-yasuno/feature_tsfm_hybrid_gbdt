"""
Post-Processing for Predictions
予測の後処理による誤検知削減

時系列の連続性を考慮した平滑化処理により、
単発の誤検知を削減しPrecisionを向上させる。

手法:
1. 連続アラート判定（N日連続で異常予測が出た場合のみアラート）
2. 指数移動平均による平滑化
3. 多数決フィルタリング
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import (
    RESULTS_ROOT,
    FORECAST_HORIZONS
)


class PredictionPostProcessor:
    """予測後処理クラス"""
    
    def __init__(self, predictions_df: pd.DataFrame):
        """
        初期化
        
        Args:
            predictions_df: 予測結果DataFrame
        """
        self.predictions_df = predictions_df.copy()
        self.processed_df = None
    
    def consecutive_alert_filter(
        self,
        column: str,
        window: int = 3,
        threshold: float = 0.5
    ) -> pd.Series:
        """
        連続アラート判定フィルタ
        
        連続してwindow日間異常予測が出た場合のみTrueにする
        
        Args:
            column: 予測確率のカラム名
            window: 連続日数
            threshold: 異常判定の閾値
            
        Returns:
            フィルタ後の予測（Boolean）
        """
        probs = self.predictions_df[column].values
        filtered = np.zeros(len(probs), dtype=bool)
        
        for i in range(len(probs)):
            if i < window - 1:
                # 最初のwindow-1日は判定できない
                filtered[i] = False
            else:
                # 過去window日間すべて閾値を超えている
                if np.all(probs[i-window+1:i+1] > threshold):
                    filtered[i] = True
        
        return pd.Series(filtered, index=self.predictions_df.index)
    
    def exponential_moving_average(
        self,
        column: str,
        alpha: float = 0.3
    ) -> pd.Series:
        """
        指数移動平均による平滑化
        
        Args:
            column: 予測確率のカラム名
            alpha: 平滑化パラメータ（0-1、大きいほど最新値を重視）
            
        Returns:
            平滑化後の予測確率
        """
        probs = self.predictions_df[column].values
        smoothed = [probs[0]]
        
        for p in probs[1:]:
            smoothed.append(alpha * p + (1 - alpha) * smoothed[-1])
        
        return pd.Series(smoothed, index=self.predictions_df.index)
    
    def majority_vote_filter(
        self,
        column: str,
        window: int = 5,
        threshold: float = 0.5
    ) -> pd.Series:
        """
        多数決フィルタ
        
        前後window日間の多数決で判定
        
        Args:
            column: 予測確率のカラム名
            window: ウィンドウサイズ
            threshold: 異常判定の閾値
            
        Returns:
            フィルタ後の予測（Boolean）
        """
        probs = self.predictions_df[column].values
        binary = (probs > threshold).astype(int)
        filtered = np.zeros(len(probs), dtype=bool)
        
        half_window = window // 2
        
        for i in range(len(probs)):
            start = max(0, i - half_window)
            end = min(len(probs), i + half_window + 1)
            
            # ウィンドウ内の多数決
            votes = binary[start:end]
            if np.mean(votes) > 0.5:
                filtered[i] = True
        
        return pd.Series(filtered, index=self.predictions_df.index)
    
    def adaptive_threshold_adjustment(
        self,
        column: str,
        base_threshold: float = 0.5,
        recent_window: int = 30
    ) -> pd.Series:
        """
        適応的な閾値調整
        
        最近の予測確率の分布に応じて閾値を調整
        
        Args:
            column: 予測確率のカラム名
            base_threshold: ベース閾値
            recent_window: 最近の日数
            
        Returns:
            調整後の予測（Boolean）
        """
        probs = self.predictions_df[column].values
        filtered = np.zeros(len(probs), dtype=bool)
        
        for i in range(len(probs)):
            if i < recent_window:
                # 十分なデータがない場合はベース閾値を使用
                threshold = base_threshold
            else:
                # 最近の予測確率の統計量から閾値を調整
                recent_probs = probs[i-recent_window:i]
                mean_prob = np.mean(recent_probs)
                std_prob = np.std(recent_probs)
                
                # 平均 + 1σ を閾値とする（異常値の検出）
                threshold = min(mean_prob + std_prob, 0.9)
                threshold = max(threshold, base_threshold)
            
            filtered[i] = probs[i] > threshold
        
        return pd.Series(filtered, index=self.predictions_df.index)
    
    def apply_all_filters(
        self,
        method: str = 'consecutive',
        **kwargs
    ):
        """
        全ホライズンに対してフィルタを適用
        
        Args:
            method: フィルタ手法
                - 'consecutive': 連続アラート判定
                - 'ema': 指数移動平均
                - 'majority': 多数決フィルタ
                - 'adaptive': 適応的閾値調整
            **kwargs: フィルタのパラメータ
        """
        print(f"\n🔧 Applying {method} filter...")
        
        self.processed_df = self.predictions_df.copy()
        
        for horizon in FORECAST_HORIZONS:
            pred_col = f'pred_{horizon}d'
            
            if pred_col not in self.processed_df.columns:
                print(f"  ⚠ Column {pred_col} not found, skipping")
                continue
            
            if method == 'consecutive':
                filtered = self.consecutive_alert_filter(pred_col, **kwargs)
                # 確率を0/1に変換
                self.processed_df[f'{pred_col}_filtered'] = filtered.astype(float)
            
            elif method == 'ema':
                alpha = kwargs.get('alpha', 0.3)
                threshold = kwargs.get('threshold', 0.5)
                smoothed = self.exponential_moving_average(pred_col, alpha=alpha)
                self.processed_df[f'{pred_col}_smoothed'] = smoothed
                # 閾値適用
                self.processed_df[f'{pred_col}_filtered'] = (smoothed > threshold).astype(float)
            
            elif method == 'majority':
                filtered = self.majority_vote_filter(pred_col, **kwargs)
                self.processed_df[f'{pred_col}_filtered'] = filtered.astype(float)
            
            elif method == 'adaptive':
                filtered = self.adaptive_threshold_adjustment(pred_col, **kwargs)
                self.processed_df[f'{pred_col}_filtered'] = filtered.astype(float)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            print(f"  ✓ Processed {pred_col}")
        
        print("✓ Filter applied to all horizons")
    
    def evaluate_filtered_predictions(self):
        """フィルタ適用後の評価"""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix
        )
        
        print("\n📊 Evaluation of filtered predictions:")
        print("-" * 70)
        
        results = []
        
        for horizon in FORECAST_HORIZONS:
            label_col = f'label_{horizon}d'
            pred_col = f'pred_{horizon}d'
            filtered_col = f'{pred_col}_filtered'
            
            if label_col not in self.processed_df.columns:
                continue
            if filtered_col not in self.processed_df.columns:
                continue
            
            y_true = self.processed_df[label_col]
            y_pred_original = (self.processed_df[pred_col] > 0.5).astype(int)
            y_pred_filtered = self.processed_df[filtered_col].astype(int)
            
            # 元の予測の評価
            orig_prec = precision_score(y_true, y_pred_original, zero_division=0)
            orig_rec = recall_score(y_true, y_pred_original, zero_division=0)
            orig_f1 = f1_score(y_true, y_pred_original, zero_division=0)
            
            # フィルタ後の評価
            filt_prec = precision_score(y_true, y_pred_filtered, zero_division=0)
            filt_rec = recall_score(y_true, y_pred_filtered, zero_division=0)
            filt_f1 = f1_score(y_true, y_pred_filtered, zero_division=0)
            
            print(f"\n{horizon}d horizon:")
            print(f"  Original  - Precision: {orig_prec:.3f}, Recall: {orig_rec:.3f}, F1: {orig_f1:.3f}")
            print(f"  Filtered  - Precision: {filt_prec:.3f}, Recall: {filt_rec:.3f}, F1: {filt_f1:.3f}")
            
            # 改善率
            if orig_prec > 0:
                prec_improvement = (filt_prec - orig_prec) / orig_prec * 100
                print(f"  Improvement - Precision: {prec_improvement:+.1f}%")
            
            results.append({
                'Horizon': f'{horizon}d',
                'Original_Precision': orig_prec,
                'Filtered_Precision': filt_prec,
                'Original_Recall': orig_rec,
                'Filtered_Recall': filt_rec,
                'Original_F1': orig_f1,
                'Filtered_F1': filt_f1
            })
        
        print("-" * 70)
        
        return pd.DataFrame(results)
    
    def save_processed_predictions(self, output_path: str = None):
        """
        処理後の予測を保存
        
        Args:
            output_path: 出力ファイルパス
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RESULTS_ROOT / f"predictions_filtered_{timestamp}.csv"
        else:
            output_path = Path(output_path)
        
        self.processed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 Filtered predictions saved to: {output_path}")


def main():
    """メイン処理"""
    print("="*70)
    print("予測後処理による誤検知削減")
    print("="*70)
    
    # 最新の予測結果を読み込み
    results_dir = RESULTS_ROOT
    prediction_files = sorted(results_dir.glob("predictions_viz_*/test_predictions.csv"))
    
    if not prediction_files:
        print("⚠ No prediction files found.")
        print("Please run visualize_predictions.py first.")
        return
    
    latest_file = prediction_files[-1]
    print(f"\n📂 Loading predictions from: {latest_file}")
    
    predictions_df = pd.read_csv(latest_file)
    print(f"✓ Loaded {len(predictions_df):,} predictions")
    
    # 後処理クラス作成
    processor = PredictionPostProcessor(predictions_df)
    
    # 複数の手法を試す
    methods = [
        ('consecutive', {'window': 3, 'threshold': 0.5}),
        ('ema', {'alpha': 0.3, 'threshold': 0.5}),
        ('majority', {'window': 5, 'threshold': 0.5}),
        ('adaptive', {'base_threshold': 0.5, 'recent_window': 30})
    ]
    
    results_summary = []
    
    for method, params in methods:
        print(f"\n{'='*70}")
        print(f"Method: {method.upper()}")
        print(f"Parameters: {params}")
        print('='*70)
        
        # フィルタ適用
        processor.apply_all_filters(method=method, **params)
        
        # 評価
        eval_results = processor.evaluate_filtered_predictions()
        eval_results['Method'] = method
        results_summary.append(eval_results)
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_ROOT / f"predictions_filtered_{method}_{timestamp}.csv"
        processor.save_processed_predictions(output_path)
    
    # サマリー保存
    summary_df = pd.concat(results_summary, ignore_index=True)
    summary_path = RESULTS_ROOT / f"filtering_methods_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print("✅ 完了!")
    print(f"💾 Comparison saved to: {summary_path}")
    print("="*70)


if __name__ == "__main__":
    main()
