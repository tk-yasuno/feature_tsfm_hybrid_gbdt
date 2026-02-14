"""
Visualize Predictions for Test Samples
テストサンプルの予測値可視化スクリプト

機能:
1. テストサンプルのロード
2. モデルによる予測
3. 30日、60日、90日ごとの可視化
4. 時系列データ + 予測確率 + 実際のラベルの表示
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
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
    MODEL_ROOT,
    RESULTS_ROOT,
    FORECAST_HORIZONS,
    LOOKBACK_DAYS,
    TRAINING_CONFIG,
    RANDOM_SEED
)

from granite_ts_model import GraniteTimeSeriesClassifier

# プロット設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class PredictionVisualizer:
    """予測結果可視化クラス"""
    
    def __init__(self, model_path: str = None):
        """
        初期化
        
        Args:
            model_path: モデルのパス
        """
        if model_path is None:
            model_path = str(MODEL_ROOT / "granite_pump_lora" / "best_model")
        
        self.model_path = model_path
        self.model = None
        self.test_df = None
        self.predictions = None
        self.optimal_thresholds = {}  # 各ホライズンの最適閾値
        
    def load_test_data(self):
        """テストデータのロード"""
        print("📂 Loading test data...")
        
        # 学習時に保存されたテストデータを読み込む
        test_path = PROCESSED_DATA_DIR / "test_samples.csv"
        
        if not test_path.exists():
            raise FileNotFoundError(
                f"Test samples not found: {test_path}\n"
                "Please run train.py first to generate test set."
            )
        
        self.test_df = pd.read_csv(test_path)
        print(f"✓ Loaded {len(self.test_df):,} test samples")
        
        # any_anomalyカラムが無い場合は作成
        if 'any_anomaly' not in self.test_df.columns:
            self.test_df['any_anomaly'] = ((self.test_df['label_30d'] == 1) | 
                                           (self.test_df['label_60d'] == 1) | 
                                           (self.test_df['label_90d'] == 1)).astype(int)
        
        print(f"  Anomaly samples: {self.test_df['any_anomaly'].sum()} "
              f"({self.test_df['any_anomaly'].sum()/len(self.test_df)*100:.1f}%)")
        
        return self.test_df
    
    def load_model(self):
        """モデルのロード"""
        print(f"\n📂 Loading model from: {self.model_path}")
        
        # モデル作成
        self.model = GraniteTimeSeriesClassifier()
        
        # 重み読み込み
        try:
            self.model.load_model(self.model_path)
            self.model.eval()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load trained weights: {e}")
            print("  Using untrained model for testing")
    
    def predict_batch(self):
        """バッチ予測"""
        print(f"\n🔮 Running predictions on {len(self.test_df)} test samples...")
        
        predictions_list = []
        
        with torch.no_grad():
            for idx, row in self.test_df.iterrows():
                # 系列データ取得
                if isinstance(row['values_sequence'], str):
                    import ast
                    values = ast.literal_eval(row['values_sequence'])
                else:
                    values = row['values_sequence']
                
                sequence = np.array(values, dtype=np.float32)
                
                # 長さ調整
                if len(sequence) < LOOKBACK_DAYS:
                    sequence = np.pad(sequence, (LOOKBACK_DAYS - len(sequence), 0), mode='edge')
                elif len(sequence) > LOOKBACK_DAYS:
                    sequence = sequence[-LOOKBACK_DAYS:]
                
                # 正規化
                seq_mean = np.mean(sequence)
                seq_std = np.std(sequence)
                if seq_std > 0:
                    sequence = (sequence - seq_mean) / seq_std
                
                # Tensor変換
                seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
                seq_tensor = seq_tensor.to(self.model.device)
                
                # 予測
                outputs = self.model(seq_tensor)
                
                # 確率値取得（モデルは辞書を返す）
                pred_dict = {}
                for h in FORECAST_HORIZONS:
                    prob = outputs[f'prob_{h}d'].item()
                    pred_dict[f'pred_{h}d'] = prob
                
                predictions_list.append(pred_dict)
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(self.test_df)} samples...")
        
        # 予測結果をDataFrameに追加
        self.predictions = pd.DataFrame(predictions_list)
        self.test_df = pd.concat([self.test_df.reset_index(drop=True), 
                                   self.predictions], axis=1)
        
        print("✓ Predictions completed")
        
        return self.test_df
    
    def find_optimal_thresholds(self):
        """
        各ホライズンの最適閾値を探索
        
        2つの方法で閾値を計算：
        1. Youden's Index (J = TPR - FPR) が最大になる閾値
        2. F1スコアが最大になる閾値
        """
        print(f"\n🔍 Finding optimal thresholds...")
        
        threshold_info = []
        
        for horizon in FORECAST_HORIZONS:
            label_col = f'label_{horizon}d'
            pred_col = f'pred_{horizon}d'
            
            y_true = self.test_df[label_col]
            y_score = self.test_df[pred_col]
            
            # ROC曲線でYouden's Indexを計算
            fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
            youden_index = tpr - fpr
            optimal_idx_youden = np.argmax(youden_index)
            optimal_threshold_youden = thresholds_roc[optimal_idx_youden]
            
            # F1スコアが最大になる閾値を探索
            precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
            # F1 = 2 * (precision * recall) / (precision + recall)
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
            optimal_idx_f1 = np.argmax(f1_scores)
            optimal_threshold_f1 = thresholds_pr[optimal_idx_f1]
            
            # F1スコアが最大になる閾値を採用
            self.optimal_thresholds[horizon] = optimal_threshold_f1
            
            threshold_info.append({
                'Horizon': f'{horizon}d',
                'Youden_Threshold': optimal_threshold_youden,
                'Youden_TPR': tpr[optimal_idx_youden],
                'Youden_FPR': fpr[optimal_idx_youden],
                'F1_Threshold': optimal_threshold_f1,
                'F1_Score': f1_scores[optimal_idx_f1],
                'Optimal_Threshold': optimal_threshold_f1  # F1ベースを採用
            })
            
            print(f"\n  {horizon}d horizon:")
            print(f"    Youden's Index: threshold={optimal_threshold_youden:.4f}, "
                  f"TPR={tpr[optimal_idx_youden]:.3f}, FPR={fpr[optimal_idx_youden]:.3f}")
            print(f"    F1-based: threshold={optimal_threshold_f1:.4f}, F1={f1_scores[optimal_idx_f1]:.3f}")
            print(f"    → Selected threshold: {optimal_threshold_f1:.4f}")
        
        # 閾値情報を保存
        self.threshold_info_df = pd.DataFrame(threshold_info)
        
        print("\n✓ Optimal thresholds found")
        
        return self.optimal_thresholds
    
    def plot_sample_predictions(
        self, 
        num_samples: int = 9,
        save_dir: str = None
    ):
        """
        サンプルの予測結果を可視化
        
        Args:
            num_samples: 可視化するサンプル数（各ホライズンごと）
            save_dir: 保存先ディレクトリ
        """
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = RESULTS_ROOT / f"predictions_viz_{timestamp}"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📊 Visualizing predictions...")
        print(f"  Save directory: {save_dir}")
        
        # 各ホライズンごとにグラフ作成
        for horizon in FORECAST_HORIZONS:
            self._plot_horizon_predictions(
                horizon=horizon,
                num_samples=num_samples,
                save_dir=save_dir
            )
        
        print(f"✓ Visualization completed. Results saved to: {save_dir}")
    
    def _plot_horizon_predictions(
        self,
        horizon: int,
        num_samples: int,
        save_dir: Path
    ):
        """
        特定ホライズンの予測結果を可視化
        
        Args:
            horizon: 予測ホライズン（30, 60, 90）
            num_samples: サンプル数
            save_dir: 保存先ディレクトリ
        """
        print(f"\n  Creating visualizations for {horizon}d horizon...")
        
        label_col = f'label_{horizon}d'
        pred_col = f'pred_{horizon}d'
        
        # ラベルごとにサンプル取得
        normal_samples = self.test_df[self.test_df[label_col] == 0].head(num_samples // 2)
        anomaly_samples = self.test_df[self.test_df[label_col] == 1].head(num_samples // 2)
        
        sample_df = pd.concat([normal_samples, anomaly_samples]).reset_index(drop=True)
        
        # グリッドプロット作成
        n_cols = 3
        n_rows = (len(sample_df) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle(f'Prediction Results - {horizon}-day Forecast', fontsize=16, y=0.995)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (_, row) in enumerate(sample_df.iterrows()):
            row_idx = idx // n_cols
            col_idx = idx % n_cols
            ax = axes[row_idx, col_idx]
            
            # 系列データ取得
            if isinstance(row['values_sequence'], str):
                import ast
                values = ast.literal_eval(row['values_sequence'])
            else:
                values = row['values_sequence']
            
            sequence = np.array(values, dtype=np.float32)
            
            # 長さ調整
            if len(sequence) < LOOKBACK_DAYS:
                sequence = np.pad(sequence, (LOOKBACK_DAYS - len(sequence), 0), mode='edge')
            elif len(sequence) > LOOKBACK_DAYS:
                sequence = sequence[-LOOKBACK_DAYS:]
            
            # プロット
            days = np.arange(-LOOKBACK_DAYS, 0)
            ax.plot(days, sequence, color='steelblue', linewidth=1.5, alpha=0.8, label='Measured Value')
            
            # 予測結果
            true_label = int(row[label_col])
            pred_prob = row[pred_col]
            
            # タイトル
            label_text = "Anomaly" if true_label == 1 else "Normal"
            pred_text = f"{pred_prob:.3f}"
            
            # 背景色
            if true_label == 1:
                ax.set_facecolor('#ffeeee')  # 赤系
            else:
                ax.set_facecolor('#eeffee')  # 緑系
            
            ax.set_title(
                f'Equip: {row["equipment_id"]}, Item: {row["check_item_id"]}\n'
                f'Actual: {label_text} | Pred Prob: {pred_text}',
                fontsize=10
            )
            ax.set_xlabel('Days in Past', fontsize=9)
            ax.set_ylabel('Measured Value', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper left')
            
            # 予測確率をテキストで表示
            ax.text(
                0.98, 0.02,
                f'Anomaly Prob: {pred_prob:.1%}',
                transform=ax.transAxes,
                fontsize=11,
                ha='right',
                va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        # 余分なサブプロットを非表示
        for idx in range(len(sample_df), n_rows * n_cols):
            row_idx = idx // n_cols
            col_idx = idx % n_cols
            axes[row_idx, col_idx].axis('off')
        
        plt.tight_layout()
        
        # 保存
        save_path = save_dir / f'predictions_{horizon}d.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {save_path.name}")
    
    def plot_probability_distributions(self, save_dir: str = None):
        """
        予測確率の分布をホライズンごとにプロット
        
        Args:
            save_dir: 保存先ディレクトリ
        """
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = RESULTS_ROOT / f"predictions_viz_{timestamp}"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Creating probability distribution plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Prediction Probability Distribution by Horizon', fontsize=16)
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            ax = axes[idx]
            
            label_col = f'label_{horizon}d'
            pred_col = f'pred_{horizon}d'
            
            # 正常と異常のデータ分離
            normal_probs = self.test_df[self.test_df[label_col] == 0][pred_col]
            anomaly_probs = self.test_df[self.test_df[label_col] == 1][pred_col]
            
            # ヒストグラム
            ax.hist(normal_probs, bins=50, alpha=0.6, label='Normal', color='green')
            ax.hist(anomaly_probs, bins=50, alpha=0.6, label='Anomaly', color='red')
            
            # 閾値ライン
            ax.axvline(0.5, color='black', linestyle='--', linewidth=2, 
                      label='Threshold (0.5)')
            
            ax.set_title(f'{horizon}-day Forecast', fontsize=14)
            ax.set_xlabel('Prediction Probability', fontsize=12)
            ax.set_ylabel('Number of Samples', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        save_path = save_dir / 'probability_distributions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path.name}")
    
    def plot_confusion_matrices(self, save_dir: str = None):
        """
        混同行列をホライズンごとにプロット
        
        Args:
            save_dir: 保存先ディレクトリ
        """
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = RESULTS_ROOT / f"predictions_viz_{timestamp}"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Creating confusion matrices...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Confusion Matrix by Horizon (Optimal Threshold)', fontsize=16)
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            ax = axes[idx]
            
            label_col = f'label_{horizon}d'
            pred_col = f'pred_{horizon}d'
            
            # 予測ラベル（最適閾値）
            threshold = self.optimal_thresholds.get(horizon, 0.5)
            y_true = self.test_df[label_col]
            y_pred = (self.test_df[pred_col] > threshold).astype(int)
            
            # 混同行列
            cm = confusion_matrix(y_true, y_pred)
            
            # ヒートマップ
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'],
                       cbar=True, annot_kws={'size': 14})
            
            ax.set_title(f'{horizon}-day Forecast (th={threshold:.3f})', fontsize=14)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('Actual Label', fontsize=12)
            
            # メトリクス追加
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            ax.text(0.5, -0.25, 
                   f'Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}',
                   transform=ax.transAxes, ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # 保存
        save_path = save_dir / 'confusion_matrices.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path.name}")
    
    def plot_roc_curves(self, save_dir: str = None):
        """
        ROC曲線をホライズンごとにプロット
        
        Args:
            save_dir: 保存先ディレクトリ
        """
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = RESULTS_ROOT / f"predictions_viz_{timestamp}"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Creating ROC curves...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('ROC Curves by Horizon', fontsize=16)
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            ax = axes[idx]
            
            label_col = f'label_{horizon}d'
            pred_col = f'pred_{horizon}d'
            
            y_true = self.test_df[label_col]
            y_score = self.test_df[pred_col]
            
            # ROC曲線
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = roc_auc_score(y_true, y_score)
            
            # 最適閾値の位置を見つける
            optimal_threshold = self.optimal_thresholds.get(horizon, 0.5)
            optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
            
            # プロット
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                   label='Random')
            
            # 最適閾値をマーカーで表示
            ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
                   label=f'Optimal (th={optimal_threshold:.3f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'{horizon}-day Forecast', fontsize=14)
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        save_path = save_dir / 'roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path.name}")
    
    def plot_precision_recall_curves(self, save_dir: str = None):
        """
        Precision-Recall曲線をホライズンごとにプロット
        
        Args:
            save_dir: 保存先ディレクトリ
        """
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = RESULTS_ROOT / f"predictions_viz_{timestamp}"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Creating Precision-Recall curves...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Precision-Recall Curves by Horizon', fontsize=16)
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            ax = axes[idx]
            
            label_col = f'label_{horizon}d'
            pred_col = f'pred_{horizon}d'
            
            y_true = self.test_df[label_col]
            y_score = self.test_df[pred_col]
            
            # PR曲線
            precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            
            # 最適閾値の位置を見つける
            optimal_threshold = self.optimal_thresholds.get(horizon, 0.5)
            # thresholds_prは長さがprecision/recallより1少ない
            if len(thresholds_pr) > 0:
                optimal_idx = np.argmin(np.abs(thresholds_pr - optimal_threshold))
            else:
                optimal_idx = 0
            
            # プロット
            ax.plot(recall, precision, color='darkgreen', lw=2,
                   label=f'PR curve (AP = {ap:.3f})')
            
            # 最適閾値をマーカーで表示
            ax.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=10,
                   label=f'Optimal (th={optimal_threshold:.3f})')
            
            # ベースライン（ランダム分類器）
            baseline = y_true.sum() / len(y_true)
            ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, 
                   linestyle='--', label=f'Baseline ({baseline:.3f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(f'{horizon}-day Forecast', fontsize=14)
            ax.legend(loc="lower left", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        save_path = save_dir / 'precision_recall_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path.name}")
    
    def create_metrics_summary(self, save_dir: str = None):
        """
        メトリクスサマリーテーブルを作成
        
        Args:
            save_dir: 保存先ディレクトリ
        """
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = RESULTS_ROOT / f"predictions_viz_{timestamp}"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Creating metrics summary...")
        
        metrics_data = []
        
        for horizon in FORECAST_HORIZONS:
            label_col = f'label_{horizon}d'
            pred_col = f'pred_{horizon}d'
            
            threshold = self.optimal_thresholds.get(horizon, 0.5)
            y_true = self.test_df[label_col]
            y_score = self.test_df[pred_col]
            y_pred = (y_score > threshold).astype(int)
            
            # メトリクス計算
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_true, y_score)
            pr_auc = average_precision_score(y_true, y_score)
            
            metrics_data.append({
                'Horizon': f'{horizon}d',
                'Threshold': threshold,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'PR-AUC': pr_auc
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # テーブル用に値をフォーマット
        table_data = []
        for _, row in metrics_df.iterrows():
            formatted_row = [row['Horizon']]
            for col in ['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']:
                formatted_row.append(f"{row[col]:.3f}")
            table_data.append(formatted_row)
        
        # テーブルプロット
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=table_data,
                        colLabels=metrics_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # ヘッダーのスタイル
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 0:
                    cell.set_facecolor('#E8F5E9')
                else:
                    cell.set_facecolor('#F5F5F5')
        
        plt.title('Performance Metrics Summary (Optimal Threshold)', 
                 fontsize=14, pad=20, weight='bold')
        
        # 保存
        save_path = save_dir / 'metrics_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # CSV保存
        csv_path = save_dir / 'metrics_summary.csv'
        metrics_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 閾値情報もCSV保存
        if hasattr(self, 'threshold_info_df'):
            threshold_path = save_dir / 'optimal_thresholds.csv'
            self.threshold_info_df.to_csv(threshold_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ Saved: {threshold_path.name}")
        
        print(f"  ✓ Saved: {save_path.name}")
        print(f"  ✓ Saved: {csv_path.name}")
        
        # コンソールにも表示
        print("\n" + "="*70)
        print("Performance Metrics Summary (Optimal Threshold)")
        print("="*70)
        print(metrics_df.to_string(index=False))
        print("="*70)
    
    def save_predictions(self, save_dir: str = None):
        """
        予測結果をCSVに保存
        
        Args:
            save_dir: 保存先ディレクトリ
        """
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = RESULTS_ROOT / f"predictions_viz_{timestamp}"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果保存
        output_cols = ['equipment_id', 'check_item_id']
        for h in FORECAST_HORIZONS:
            output_cols.extend([f'label_{h}d', f'pred_{h}d'])
        
        output_df = self.test_df[output_cols]
        save_path = save_dir / 'test_predictions.csv'
        output_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 Predictions saved to: {save_path}")


def main():
    """メイン処理"""
    print("=" * 70)
    print("テストサンプル予測値可視化")
    print("=" * 70)
    
    # Visualizer作成
    visualizer = PredictionVisualizer()
    
    # テストデータロード
    test_df = visualizer.load_test_data()
    
    # モデルロード
    visualizer.load_model()
    
    # 予測
    test_df = visualizer.predict_batch()
    
    # 最適閾値の探索
    optimal_thresholds = visualizer.find_optimal_thresholds()
    
    # 可視化
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RESULTS_ROOT / f"predictions_viz_{timestamp}"
    
    # サンプル予測結果プロット
    visualizer.plot_sample_predictions(num_samples=9, save_dir=save_dir)
    
    # 確率分布プロット
    visualizer.plot_probability_distributions(save_dir=save_dir)
    
    # 実績 vs 予測の対比
    visualizer.plot_confusion_matrices(save_dir=save_dir)
    visualizer.plot_roc_curves(save_dir=save_dir)
    visualizer.plot_precision_recall_curves(save_dir=save_dir)
    visualizer.create_metrics_summary(save_dir=save_dir)
    
    # 予測結果保存
    visualizer.save_predictions(save_dir=save_dir)
    
    print("\n" + "=" * 70)
    print("✅ 完了!")
    print("=" * 70)


if __name__ == "__main__":
    main()
