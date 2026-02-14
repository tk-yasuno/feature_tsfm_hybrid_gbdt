"""
V2.0 Hybrid Model Predictions Visualization
ハイブリッドモデル（Granite TS埋め込み + 統計特徴量 + LightGBM）の予測結果可視化

機能:
1. テストデータのロード
2. 訓練済みLightGBMモデルのロード
3. テスト予測の実行
4. 詳細な可視化:
   - 混同行列
   - ROC曲線・PR曲線
   - 確率分布
   - サンプル予測プロット
   - 特徴量重要度
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from config import (
    PROCESSED_DATA_DIR,
    RESULTS_ROOT,
    MODEL_ROOT,
    FORECAST_HORIZONS
)

# プロット設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class HybridModelVisualizer:
    """ハイブリッドモデル予測可視化クラス"""
    
    def __init__(self, model_dir: str = None):
        """
        初期化
        
        Args:
            model_dir: モデルディレクトリのパス
        """
        if model_dir is None:
            model_dir = MODEL_ROOT / "hybrid_model"
        
        self.model_dir = Path(model_dir)
        self.models = {}
        self.test_data = None
        self.predictions = {}
        self.metrics = {}
        
    def load_test_data(self):
        """テストデータのロード"""
        print("📂 Loading test data...")
        
        # まず保存済みのテストデータをチェック
        test_features_path = PROCESSED_DATA_DIR / "test_features_hybrid.csv"
        test_labels_path = PROCESSED_DATA_DIR / "test_labels.csv"
        
        if test_features_path.exists() and test_labels_path.exists():
            # 保存済みデータをロード
            self.X_test = pd.read_csv(test_features_path)
            self.y_test = pd.read_csv(test_labels_path)
            print(f"✓ Loaded saved test data:")
        else:
            # フォールバック: test_samples_enriched.csvから読み込む
            print("  ⚠ Saved test data not found, loading from test_samples_enriched.csv...")
            
            # Test samples enrichedをロード
            enriched_path = PROCESSED_DATA_DIR / "test_samples_enriched.csv"
            if not enriched_path.exists():
                # さらにフォールバック: training_samples_enriched.csvから分割
                enriched_path = PROCESSED_DATA_DIR / "training_samples_enriched.csv"
                if not enriched_path.exists():
                    raise FileNotFoundError(
                        f"No enriched data found.\n"
                        "Please run create_enriched_features.py or train_hybrid_model.py first."
                    )
                
                print(f"  Loading from {enriched_path.name}...")
                df = pd.read_csv(enriched_path)
                
                # データ分割（train_hybrid_model.pyと同じロジック）
                from sklearn.model_selection import train_test_split
                
                df['any_anomaly'] = ((df['label_30d'] == 1) | 
                                     (df['label_60d'] == 1) | 
                                     (df['label_90d'] == 1)).astype(int)
                
                _, test_df = train_test_split(
                    df,
                    test_size=0.15,
                    stratify=df['any_anomaly'],
                    random_state=42
                )
            else:
                print(f"  Loading from {enriched_path.name}...")
                test_df = pd.read_csv(enriched_path)
            
            print(f"  Loaded {len(test_df):,} test samples")
            
            # 特徴量カラムを選択
            exclude_cols = [
                'equipment_id', 'check_item_id', 'date', 
                'window_start', 'window_end', 'values_sequence',
                'label_current', 'label_30d', 'label_60d', 'label_90d',
                'any_anomaly'
            ]
            
            feature_cols = [col for col in test_df.columns if col not in exclude_cols]
            
            self.X_test = test_df[feature_cols]
            self.y_test = test_df[[f'label_{h}d' for h in FORECAST_HORIZONS]]
            
            print(f"✓ Loaded test data from enriched file:")
        
        print(f"  Features shape: {self.X_test.shape}")
        print(f"  Labels shape: {self.y_test.shape}")
        
        # 異常サンプル数を表示
        for horizon in FORECAST_HORIZONS:
            label_col = f'label_{horizon}d'
            if label_col in self.y_test.columns:
                n_anomalies = self.y_test[label_col].sum()
                anomaly_rate = n_anomalies / len(self.y_test) * 100
                print(f"  {horizon}d anomalies: {n_anomalies} ({anomaly_rate:.1f}%)")
        
        return self.X_test, self.y_test
    
    def load_models(self):
        """訓練済みモデルのロード"""
        print(f"\n📂 Loading models from: {self.model_dir}")
        
        for horizon in FORECAST_HORIZONS:
            # 新しいファイル名と古いファイル名の両方を試す
            model_paths = [
                self.model_dir / f"lgbm_hybrid_{horizon}d.txt",
                self.model_dir / f"model_{horizon}d.txt"
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if model_path.exists():
                    # LightGBMモデルのロード
                    self.models[horizon] = lgb.Booster(model_file=str(model_path))
                    print(f"  ✓ Loaded {horizon}d model from {model_path.name}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                print(f"  ⚠ Model not found for {horizon}d horizon")
        
        if not self.models:
            raise FileNotFoundError(
                f"No models found in {self.model_dir}\n"
                "Please run train_hybrid_model.py first."
            )
        
        print(f"✓ Loaded {len(self.models)} models")
        
        return self.models
    
    def predict_all(self):
        """全ホライズンで予測"""
        print(f"\n🔮 Running predictions...")
        
        for horizon in FORECAST_HORIZONS:
            if horizon not in self.models:
                print(f"  ⚠ Skipping {horizon}d (model not loaded)")
                continue
            
            # 予測
            y_pred_proba = self.models[horizon].predict(self.X_test)
            
            # 最適閾値の探索
            label_col = f'label_{horizon}d'
            if label_col not in self.y_test.columns:
                print(f"  ⚠ Skipping {horizon}d (labels not found)")
                continue
            
            y_true = self.y_test[label_col].values
            
            # F1スコア最大化
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            y_pred = (y_pred_proba > optimal_threshold).astype(int)
            
            # メトリクス計算
            self.metrics[horizon] = {
                'optimal_threshold': optimal_threshold,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'pr_auc': average_precision_score(y_true, y_pred_proba)
            }
            
            # 予測結果を保存
            self.predictions[horizon] = {
                'y_true': y_true,
                'y_pred_proba': y_pred_proba,
                'y_pred': y_pred
            }
            
            print(f"  ✓ {horizon}d predictions: Precision={self.metrics[horizon]['precision']:.4f}, "
                  f"Recall={self.metrics[horizon]['recall']:.4f}, F1={self.metrics[horizon]['f1']:.4f}")
        
        print("✓ All predictions completed")
        
        return self.predictions, self.metrics
    
    def plot_confusion_matrices(self, save_dir: str = None):
        """混同行列のプロット"""
        if save_dir is None:
            save_dir = self.create_output_dir()
        else:
            save_dir = Path(save_dir)
        
        print(f"\n📊 Creating confusion matrices...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Hybrid Model - Confusion Matrix by Horizon', fontsize=16, weight='bold')
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            if horizon not in self.predictions:
                continue
            
            ax = axes[idx]
            
            y_true = self.predictions[horizon]['y_true']
            y_pred = self.predictions[horizon]['y_pred']
            
            # 混同行列
            cm = confusion_matrix(y_true, y_pred)
            
            # ヒートマップ
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'],
                       cbar=True, annot_kws={'size': 16, 'weight': 'bold'})
            
            threshold = self.metrics[horizon]['optimal_threshold']
            ax.set_title(f'{horizon}-day Forecast\n(threshold={threshold:.3f})', fontsize=14)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('Actual Label', fontsize=12)
            
            # メトリクス表示
            metrics = self.metrics[horizon]
            metrics_text = (
                f"Accuracy: {metrics['accuracy']:.3f}\n"
                f"Precision: {metrics['precision']:.3f}\n"
                f"Recall: {metrics['recall']:.3f}\n"
                f"F1-Score: {metrics['f1']:.3f}"
            )
            
            ax.text(0.5, -0.35, metrics_text,
                   transform=ax.transAxes, ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存
        save_path = save_dir / 'hybrid_confusion_matrices.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path.name}")
    
    def plot_roc_curves(self, save_dir: str = None):
        """ROC曲線のプロット"""
        if save_dir is None:
            save_dir = self.create_output_dir()
        else:
            save_dir = Path(save_dir)
        
        print(f"\n📊 Creating ROC curves...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Hybrid Model - ROC Curves by Horizon', fontsize=16, weight='bold')
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            if horizon not in self.predictions:
                continue
            
            ax = axes[idx]
            
            y_true = self.predictions[horizon]['y_true']
            y_pred_proba = self.predictions[horizon]['y_pred_proba']
            
            # ROC曲線
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            roc_auc = self.metrics[horizon]['roc_auc']
            
            # 最適閾値の位置
            optimal_threshold = self.metrics[horizon]['optimal_threshold']
            optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
            
            # プロット
            ax.plot(fpr, tpr, color='darkorange', lw=3,
                   label=f'Hybrid Model (AUC = {roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                   label='Random Classifier')
            
            # 最適閾値をマーカーで表示
            ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12,
                   label=f'Optimal Point (th={optimal_threshold:.3f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'{horizon}-day Forecast', fontsize=14)
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        save_path = save_dir / 'hybrid_roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path.name}")
    
    def plot_pr_curves(self, save_dir: str = None):
        """Precision-Recall曲線のプロット"""
        if save_dir is None:
            save_dir = self.create_output_dir()
        else:
            save_dir = Path(save_dir)
        
        print(f"\n📊 Creating Precision-Recall curves...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Hybrid Model - Precision-Recall Curves by Horizon', fontsize=16, weight='bold')
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            if horizon not in self.predictions:
                continue
            
            ax = axes[idx]
            
            y_true = self.predictions[horizon]['y_true']
            y_pred_proba = self.predictions[horizon]['y_pred_proba']
            
            # PR曲線
            precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = self.metrics[horizon]['pr_auc']
            
            # 最適閾値の位置
            optimal_threshold = self.metrics[horizon]['optimal_threshold']
            if len(thresholds_pr) > 0:
                optimal_idx = np.argmin(np.abs(thresholds_pr - optimal_threshold))
            else:
                optimal_idx = 0
            
            # プロット
            ax.plot(recall, precision, color='darkgreen', lw=3,
                   label=f'Hybrid Model (AP = {pr_auc:.4f})')
            
            # 最適閾値をマーカーで表示
            ax.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=12,
                   label=f'Optimal Point (th={optimal_threshold:.3f})')
            
            # ベースライン（ランダム分類器）
            baseline = y_true.sum() / len(y_true)
            ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, 
                   linestyle='--', label=f'Baseline ({baseline:.3f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(f'{horizon}-day Forecast', fontsize=14)
            ax.legend(loc="lower left", fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        save_path = save_dir / 'hybrid_pr_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path.name}")
    
    def plot_probability_distributions(self, save_dir: str = None):
        """予測確率分布のプロット"""
        if save_dir is None:
            save_dir = self.create_output_dir()
        else:
            save_dir = Path(save_dir)
        
        print(f"\n📊 Creating probability distributions...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Hybrid Model - Prediction Probability Distribution', fontsize=16, weight='bold')
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            if horizon not in self.predictions:
                continue
            
            ax = axes[idx]
            
            y_true = self.predictions[horizon]['y_true']
            y_pred_proba = self.predictions[horizon]['y_pred_proba']
            
            # 正常と異常のデータ分離
            normal_probs = y_pred_proba[y_true == 0]
            anomaly_probs = y_pred_proba[y_true == 1]
            
            # ヒストグラム
            ax.hist(normal_probs, bins=50, alpha=0.6, label='Normal', color='green', density=True)
            ax.hist(anomaly_probs, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
            
            # 最適閾値ライン
            optimal_threshold = self.metrics[horizon]['optimal_threshold']
            ax.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, 
                      label=f'Optimal Threshold ({optimal_threshold:.3f})')
            
            ax.set_title(f'{horizon}-day Forecast', fontsize=14)
            ax.set_xlabel('Prediction Probability', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 統計情報を追加
            stats_text = (
                f"Normal: μ={normal_probs.mean():.3f}, σ={normal_probs.std():.3f}\n"
                f"Anomaly: μ={anomaly_probs.mean():.3f}, σ={anomaly_probs.std():.3f}"
            )
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, ha='left', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存
        save_path = save_dir / 'hybrid_probability_distributions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path.name}")
    
    def plot_feature_importance(self, top_n: int = 20, save_dir: str = None):
        """特徴量重要度のプロット"""
        if save_dir is None:
            save_dir = self.create_output_dir()
        else:
            save_dir = Path(save_dir)
        
        print(f"\n📊 Creating feature importance plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Hybrid Model - Top 20 Feature Importance', fontsize=16, weight='bold')
        
        feature_names = self.X_test.columns.tolist()
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            if horizon not in self.models:
                continue
            
            ax = axes[idx]
            
            # 特徴量重要度を取得
            importance = self.models[horizon].feature_importance(importance_type='gain')
            
            # DataFrameに変換してソート
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(top_n)
            
            # 横棒グラフ
            colors = ['red' if 'emb_' in f else 'blue' for f in fi_df['Feature']]
            ax.barh(range(len(fi_df)), fi_df['Importance'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(fi_df)))
            ax.set_yticklabels(fi_df['Feature'], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Importance (Gain)', fontsize=12)
            ax.set_title(f'{horizon}-day Forecast', fontsize=14)
            ax.grid(True, alpha=0.3, axis='x')
            
            # 凡例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.7, label='TS Embedding'),
                Patch(facecolor='blue', alpha=0.7, label='Statistical Feature')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        # 保存
        save_path = save_dir / 'hybrid_feature_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path.name}")
    
    def create_metrics_summary(self, save_dir: str = None):
        """メトリクスサマリーの作成"""
        if save_dir is None:
            save_dir = self.create_output_dir()
        else:
            save_dir = Path(save_dir)
        
        print(f"\n📊 Creating metrics summary...")
        
        # メトリクスをDataFrameに変換
        metrics_data = []
        for horizon in FORECAST_HORIZONS:
            if horizon not in self.metrics:
                continue
            
            metrics = self.metrics[horizon]
            metrics_data.append({
                'Horizon': f'{horizon}d',
                'Optimal Threshold': f"{metrics['optimal_threshold']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'PR-AUC': f"{metrics['pr_auc']:.4f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # テーブルプロット
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=metrics_df.values,
                        colLabels=metrics_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # ヘッダーのスタイル
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#2196F3')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 0:
                    cell.set_facecolor('#E3F2FD')
                else:
                    cell.set_facecolor('#F5F5F5')
        
        plt.title('Hybrid Model - Performance Metrics Summary', 
                 fontsize=16, pad=20, weight='bold')
        
        # 保存
        save_path = save_dir / 'hybrid_metrics_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # CSV保存
        csv_path = save_dir / 'hybrid_metrics_summary.csv'
        
        # 数値形式で保存
        metrics_df_numeric = pd.DataFrame([{
            'Horizon': f'{horizon}d',
            **self.metrics[horizon]
        } for horizon in FORECAST_HORIZONS if horizon in self.metrics])
        
        metrics_df_numeric.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"  ✓ Saved: {save_path.name}")
        print(f"  ✓ Saved: {csv_path.name}")
        
        # コンソール表示
        print("\n" + "="*80)
        print("Hybrid Model - Performance Metrics Summary")
        print("="*80)
        print(metrics_df.to_string(index=False))
        print("="*80)
    
    def save_predictions_csv(self, save_dir: str = None):
        """予測結果をCSVに保存"""
        if save_dir is None:
            save_dir = self.create_output_dir()
        else:
            save_dir = Path(save_dir)
        
        print(f"\n💾 Saving predictions to CSV...")
        
        # 各ホライズンの予測を結合
        predictions_df = pd.DataFrame()
        
        for horizon in FORECAST_HORIZONS:
            if horizon not in self.predictions:
                continue
            
            predictions_df[f'label_{horizon}d'] = self.predictions[horizon]['y_true']
            predictions_df[f'pred_proba_{horizon}d'] = self.predictions[horizon]['y_pred_proba']
            predictions_df[f'pred_{horizon}d'] = self.predictions[horizon]['y_pred']
        
        # 保存
        save_path = save_dir / 'hybrid_test_predictions.csv'
        predictions_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"  ✓ Saved: {save_path.name}")
        print(f"    Total samples: {len(predictions_df):,}")
    
    def create_output_dir(self):
        """出力ディレクトリの作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RESULTS_ROOT / f"hybrid_viz_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def visualize_all(self, save_dir: str = None):
        """すべての可視化を実行"""
        if save_dir is None:
            save_dir = self.create_output_dir()
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Starting visualization...")
        print(f"  Output directory: {save_dir}")
        
        # すべての可視化を実行
        self.plot_confusion_matrices(save_dir)
        self.plot_roc_curves(save_dir)
        self.plot_pr_curves(save_dir)
        self.plot_probability_distributions(save_dir)
        self.plot_feature_importance(top_n=20, save_dir=save_dir)
        self.create_metrics_summary(save_dir)
        self.save_predictions_csv(save_dir)
        
        print(f"\n✅ All visualizations completed!")
        print(f"📁 Results saved to: {save_dir}")
        
        return save_dir


def main():
    """メイン処理"""
    print("=" * 80)
    print("V2.0 Hybrid Model - Predictions Visualization")
    print("Granite TS Embeddings + Statistical Features + LightGBM")
    print("=" * 80)
    
    # Visualizer作成
    visualizer = HybridModelVisualizer()
    
    # データロード
    try:
        visualizer.load_test_data()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Please run the following command first:")
        print("   python train_hybrid_model.py")
        return
    
    # モデルロード
    try:
        visualizer.load_models()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Please run the following command first:")
        print("   python train_hybrid_model.py")
        return
    
    # 予測
    visualizer.predict_all()
    
    # 可視化
    output_dir = visualizer.visualize_all()
    
    print("\n" + "=" * 80)
    print("✅ Visualization completed successfully!")
    print(f"📁 All results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
