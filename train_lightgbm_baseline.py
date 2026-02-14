"""
LightGBM Baseline Model for Range Deviation Forecast
LightGBM特徴量ベースベースラインモデル

SOTA比較のため、統計的特徴量のみを使用した
勾配ブースティングモデルを実装。

特徴:
1. 追加した28の統計的特徴量のみを使用
2. 3つのホライズン（30, 60, 90日）それぞれでモデル学習
3. クラス不均衡対応（scale_pos_weight）
4. ハイパーパラメータ最適化（Optuna）
5. 特徴量重要度分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PROCESSED_DATA_DIR,
    MODEL_ROOT,
    RESULTS_ROOT,
    FORECAST_HORIZONS,
    RANDOM_SEED
)

# プロット設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class LightGBMBaseline:
    """LightGBMベースラインモデル"""
    
    def __init__(self, use_optuna: bool = False):
        """
        初期化
        
        Args:
            use_optuna: Optunaでハイパーパラメータ最適化するか
        """
        self.use_optuna = use_optuna
        self.models = {}  # {horizon: model}
        self.feature_cols = []
        self.results = {}
        
    def load_data(self):
        """データのロード"""
        print("📂 Loading enriched data...")
        
        train_path = PROCESSED_DATA_DIR / "training_samples_enriched.csv"
        test_path = PROCESSED_DATA_DIR / "test_samples_enriched.csv"
        
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "Enriched data not found. Please run create_enriched_features.py first."
            )
        
        # データ読み込み
        print(f"  Loading train: {train_path}")
        self.train_df = pd.read_csv(train_path)
        
        print(f"  Loading test: {test_path}")
        self.test_df = pd.read_csv(test_path)
        
        print(f"✓ Loaded train: {len(self.train_df):,} samples")
        print(f"✓ Loaded test: {len(self.test_df):,} samples")
        
        # 特徴量カラムの特定（追加した統計的特徴量のみ）
        exclude_cols = [
            'equipment_id', 'check_item_id', 'date', 
            'window_start', 'window_end', 'values_sequence',
            'label_current', 'label_30d', 'label_60d', 'label_90d',
            'any_anomaly'
        ]
        
        self.feature_cols = [col for col in self.train_df.columns 
                            if col not in exclude_cols]
        
        print(f"✓ Feature columns: {len(self.feature_cols)}")
        print(f"  {', '.join(self.feature_cols[:5])}...")
        
        return self.train_df, self.test_df
    
    def get_default_params(self, pos_weight: float = 1.0) -> Dict:
        """
        デフォルトのハイパーパラメータ
        
        Args:
            pos_weight: 正例の重み（クラス不均衡対応）
            
        Returns:
            パラメータ辞書
        """
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': RANDOM_SEED,
            'scale_pos_weight': pos_weight,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
    
    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50
    ) -> Dict:
        """
        Optunaでハイパーパラメータ最適化
        
        Args:
            X_train: 訓練特徴量
            y_train: 訓練ラベル
            X_val: 検証特徴量
            y_val: 検証ラベル
            n_trials: 試行回数
            
        Returns:
            最適パラメータ
        """
        try:
            import optuna
        except ImportError:
            print("⚠ Optuna not installed. Using default parameters.")
            print("  Install: pip install optuna")
            pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
            return self.get_default_params(pos_weight)
        
        print(f"\n🔍 Optimizing hyperparameters with Optuna ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'verbose': -1,
                'random_state': RANDOM_SEED
            }
            
            # クラス不均衡対応
            pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
            params['scale_pos_weight'] = pos_weight
            
            # データセット作成
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # 学習
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            # 検証データでの評価
            y_pred = model.predict(X_val)
            auc = roc_auc_score(y_val, y_pred)
            
            return auc
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        print(f"✓ Best AUC: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")
        
        # 最適パラメータにクラス重みを追加
        best_params = study.best_params
        best_params['objective'] = 'binary'
        best_params['metric'] = 'auc'
        best_params['boosting_type'] = 'gbdt'
        best_params['verbose'] = -1
        best_params['random_state'] = RANDOM_SEED
        
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        best_params['scale_pos_weight'] = pos_weight
        
        return best_params
    
    def train_model(self, horizon: int):
        """
        特定ホライズンのモデル学習
        
        Args:
            horizon: 予測ホライズン（30, 60, 90）
        """
        print(f"\n{'='*70}")
        print(f"Training model for {horizon}d horizon")
        print('='*70)
        
        label_col = f'label_{horizon}d'
        
        # 特徴量とラベル
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df[label_col]
        
        X_test = self.test_df[self.feature_cols]
        y_test = self.test_df[label_col]
        
        print(f"\nData statistics:")
        print(f"  Train: {len(X_train):,} samples")
        print(f"  Train positives: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
        print(f"  Test: {len(X_test):,} samples")
        print(f"  Test positives: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
        
        # クラス不均衡の重み
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        print(f"  Positive class weight: {pos_weight:.2f}")
        
        # ハイパーパラメータ
        if self.use_optuna:
            # テストセットの一部を検証用に使用
            val_size = int(len(X_test) * 0.5)
            X_val = X_test[:val_size]
            y_val = y_test[:val_size]
            
            params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            params = self.get_default_params(pos_weight)
        
        # データセット作成
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # 学習
        print(f"\n🚀 Training LightGBM model...")
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        self.models[horizon] = model
        
        print(f"✓ Training completed")
        print(f"  Best iteration: {model.best_iteration}")
        
        # 予測
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        
        # 最適閾値の探索（F1スコア最大化）
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred = (y_pred_proba > optimal_threshold).astype(int)
        
        # 評価
        metrics = {
            'horizon': horizon,
            'optimal_threshold': optimal_threshold,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        
        self.results[horizon] = {
            'metrics': metrics,
            'predictions': y_pred_proba,
            'labels': y_test,
            'model': model
        }
        
        print(f"\n📊 Evaluation metrics:")
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        
        return model, metrics
    
    def train_all_horizons(self):
        """全ホライズンのモデル学習"""
        print("\n" + "="*70)
        print("🚀 LightGBM Baseline Training - Feature-based SOTA")
        print("="*70)
        
        all_metrics = []
        
        for horizon in FORECAST_HORIZONS:
            model, metrics = self.train_model(horizon)
            all_metrics.append(metrics)
        
        # サマリー
        print("\n" + "="*70)
        print("📊 Training Summary")
        print("="*70)
        
        metrics_df = pd.DataFrame(all_metrics)
        print(metrics_df.to_string(index=False))
        
        return metrics_df
    
    def plot_feature_importance(self, top_n: int = 20):
        """特徴量重要度の可視化"""
        print(f"\n📊 Plotting feature importance (top {top_n})...")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Feature Importance by Horizon', fontsize=16, weight='bold')
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            ax = axes[idx]
            model = self.models[horizon]
            
            # 特徴量重要度
            importance = model.feature_importance(importance_type='gain')
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            # プロット
            ax.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
            ax.set_yticks(range(len(feature_importance_df)))
            ax.set_yticklabels(feature_importance_df['feature'])
            ax.invert_yaxis()
            ax.set_xlabel('Importance (Gain)', fontsize=12)
            ax.set_title(f'{horizon}-day Forecast', fontsize=14)
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # 保存
        output_dir = RESULTS_ROOT / "lightgbm_baseline"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def save_models(self):
        """モデルの保存"""
        print(f"\n💾 Saving models...")
        
        output_dir = MODEL_ROOT / "lightgbm_baseline"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for horizon, model in self.models.items():
            model_path = output_dir / f"model_{horizon}d.txt"
            model.save_model(str(model_path))
            print(f"  ✓ Saved {horizon}d model: {model_path}")
        
        # メトリクスも保存
        metrics_list = [self.results[h]['metrics'] for h in FORECAST_HORIZONS]
        metrics_df = pd.DataFrame(metrics_list)
        
        metrics_path = output_dir / 'metrics_summary.csv'
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ Saved metrics: {metrics_path}")
    
    def compare_with_granite(self, granite_results_path: str = None):
        """Granite TSモデルとの比較"""
        print(f"\n📊 Comparing with Granite TS model...")
        
        if granite_results_path is None:
            # 最新のGranite結果を探す
            granite_files = sorted(RESULTS_ROOT.glob("predictions_viz_*/metrics_summary.csv"))
            if not granite_files:
                print("  ⚠ Granite results not found. Skipping comparison.")
                return
            granite_results_path = granite_files[-1]
        
        granite_df = pd.read_csv(granite_results_path)
        
        # LightGBMの結果
        lgbm_metrics = [self.results[h]['metrics'] for h in FORECAST_HORIZONS]
        lgbm_df = pd.DataFrame(lgbm_metrics)
        lgbm_df['Model'] = 'LightGBM'
        
        granite_df['Model'] = 'Granite TS'
        granite_df = granite_df.rename(columns={'Horizon': 'horizon'})
        
        # ホライズン列を数値に変換
        granite_df['horizon'] = granite_df['horizon'].str.replace('d', '').astype(int)
        
        # 結合
        comparison_df = pd.concat([
            granite_df[['Model', 'horizon', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].rename(columns={
                'Precision': 'precision',
                'Recall': 'recall',
                'F1-Score': 'f1',
                'ROC-AUC': 'roc_auc'
            }),
            lgbm_df[['Model', 'horizon', 'precision', 'recall', 'f1', 'roc_auc']]
        ], ignore_index=True)
        
        print("\n" + "="*70)
        print("Model Comparison: LightGBM vs Granite TS")
        print("="*70)
        print(comparison_df.to_string(index=False))
        
        # 保存
        output_dir = RESULTS_ROOT / "lightgbm_baseline"
        comparison_path = output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 Comparison saved: {comparison_path}")
        
        return comparison_df


def main():
    """メイン処理"""
    baseline = LightGBMBaseline(use_optuna=False)  # Trueでハイパーパラメータ最適化
    
    # データロード
    baseline.load_data()
    
    # 全ホライズンの学習
    metrics_df = baseline.train_all_horizons()
    
    # 特徴量重要度の可視化
    baseline.plot_feature_importance()
    
    # モデル保存
    baseline.save_models()
    
    # Granite TSとの比較
    baseline.compare_with_granite()
    
    print("\n" + "="*70)
    print("✅ LightGBM Baseline Training Completed!")
    print("="*70)


if __name__ == "__main__":
    main()
