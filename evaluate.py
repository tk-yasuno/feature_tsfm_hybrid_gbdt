"""
Evaluation Script for Pump Range Deviation Forecast
評価スクリプト

機能:
1. 予測結果の評価
2. ROC-AUC, PR-AUC計算
3. リードタイム分析
4. 混同行列の生成
5. 評価結果の可視化
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

from config import (
    RESULTS_ROOT,
    FORECAST_HORIZONS,
    EVALUATION_METRICS,
    LEADTIME_THRESHOLDS
)


class ModelEvaluator:
    """モデル評価クラス"""
    
    def __init__(self, results_df: pd.DataFrame):
        """
        初期化
        
        Args:
            results_df: 予測結果DataFrame
        """
        self.results_df = results_df
        self.metrics = {}
        
    def calculate_metrics(self) -> Dict:
        """
        評価指標の計算
        
        Returns:
            メトリクスの辞書
        """
        print("📊 Calculating evaluation metrics...")
        
        metrics = {}
        
        for h in FORECAST_HORIZONS:
            print(f"\n  Horizon: {h} days")
            
            prob_col = f'predictions.prob_{h}d'
            label_col = f'true_label_{h}d'
            
            if prob_col not in self.results_df.columns or label_col not in self.results_df.columns:
                print(f"    ⚠ Missing columns for {h}d horizon")
                continue
            
            y_true = self.results_df[label_col].values
            y_pred_proba = self.results_df[prob_col].values
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            horizon_metrics = {}
            
            # ROC-AUC
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                horizon_metrics['roc_auc'] = float(roc_auc)
                print(f"    ROC-AUC: {roc_auc:.4f}")
            except:
                horizon_metrics['roc_auc'] = None
            
            # PR-AUC
            try:
                pr_auc = average_precision_score(y_true, y_pred_proba)
                horizon_metrics['pr_auc'] = float(pr_auc)
                print(f"    PR-AUC: {pr_auc:.4f}")
            except:
                horizon_metrics['pr_auc'] = None
            
            # 分類レポート
            report = classification_report(
                y_true, y_pred,
                target_names=['Normal', 'Anomalous'],
                output_dict=True
            )
            
            horizon_metrics['precision'] = report['Anomalous']['precision']
            horizon_metrics['recall'] = report['Anomalous']['recall']
            horizon_metrics['f1_score'] = report['Anomalous']['f1-score']
            horizon_metrics['accuracy'] = report['accuracy']
            
            print(f"    Precision: {horizon_metrics['precision']:.4f}")
            print(f"    Recall: {horizon_metrics['recall']:.4f}")
            print(f"    F1-Score: {horizon_metrics['f1_score']:.4f}")
            
            # 混同行列
            cm = confusion_matrix(y_true, y_pred)
            horizon_metrics['confusion_matrix'] = cm.tolist()
            
            metrics[f'{h}d'] = horizon_metrics
        
        self.metrics = metrics
        print("\n✓ Metrics calculation complete")
        
        return metrics
    
    def analyze_leadtime(self) -> Dict:
        """
        リードタイム分析
        
        Returns:
            リードタイム分析結果
        """
        print("\n⏱️  Analyzing lead time performance...")
        
        leadtime_analysis = {}
        
        for h in FORECAST_HORIZONS:
            prob_col = f'predictions.prob_{h}d'
            label_col = f'true_label_{h}d'
            
            if prob_col not in self.results_df.columns or label_col not in self.results_df.columns:
                continue
            
            # 実際に異常だったケース
            anomalous_cases = self.results_df[self.results_df[label_col] == 1]
            
            if len(anomalous_cases) == 0:
                continue
            
            horizon_leadtime = {}
            
            for threshold in LEADTIME_THRESHOLDS:
                # 閾値以上で予測できたケース
                detected = anomalous_cases[anomalous_cases[prob_col] >= threshold]
                
                detection_rate = len(detected) / len(anomalous_cases)
                horizon_leadtime[f'threshold_{threshold}'] = {
                    'detection_rate': float(detection_rate),
                    'detected': len(detected),
                    'total_anomalous': len(anomalous_cases)
                }
            
            leadtime_analysis[f'{h}d'] = horizon_leadtime
            
            print(f"  {h}d horizon:")
            print(f"    Total anomalous: {len(anomalous_cases)}")
            for threshold in LEADTIME_THRESHOLDS:
                rate = horizon_leadtime[f'threshold_{threshold}']['detection_rate']
                print(f"    Detection rate @ {threshold}: {rate:.2%}")
        
        print("✓ Lead time analysis complete")
        
        return leadtime_analysis
    
    def plot_roc_curves(self, save_path: Path = None):
        """
        ROC曲線のプロット
        
        Args:
            save_path: 保存先パス
        """
        print("\n📈 Plotting ROC curves...")
        
        fig, axes = plt.subplots(1, len(FORECAST_HORIZONS), figsize=(15, 5))
        
        if len(FORECAST_HORIZONS) == 1:
            axes = [axes]
        
        for idx, h in enumerate(FORECAST_HORIZONS):
            prob_col = f'predictions.prob_{h}d'
            label_col = f'true_label_{h}d'
            
            if prob_col not in self.results_df.columns or label_col not in self.results_df.columns:
                continue
            
            y_true = self.results_df[label_col].values
            y_pred_proba = self.results_df[prob_col].values
            
            # ROC曲線計算
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            
            # プロット
            axes[idx].plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
            axes[idx].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[idx].set_xlabel('False Positive Rate')
            axes[idx].set_ylabel('True Positive Rate')
            axes[idx].set_title(f'{h}-day Horizon')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curves saved: {save_path}")
        
        plt.close()
    
    def plot_pr_curves(self, save_path: Path = None):
        """
        Precision-Recall曲線のプロット
        
        Args:
            save_path: 保存先パス
        """
        print("\n📈 Plotting PR curves...")
        
        fig, axes = plt.subplots(1, len(FORECAST_HORIZONS), figsize=(15, 5))
        
        if len(FORECAST_HORIZONS) == 1:
            axes = [axes]
        
        for idx, h in enumerate(FORECAST_HORIZONS):
            prob_col = f'predictions.prob_{h}d'
            label_col = f'true_label_{h}d'
            
            if prob_col not in self.results_df.columns or label_col not in self.results_df.columns:
                continue
            
            y_true = self.results_df[label_col].values
            y_pred_proba = self.results_df[prob_col].values
            
            # PR曲線計算
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = average_precision_score(y_true, y_pred_proba)
            
            # プロット
            axes[idx].plot(recall, precision, label=f'PR (AUC={pr_auc:.3f})')
            axes[idx].set_xlabel('Recall')
            axes[idx].set_ylabel('Precision')
            axes[idx].set_title(f'{h}-day Horizon')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ PR curves saved: {save_path}")
        
        plt.close()
    
    def plot_confusion_matrices(self, save_path: Path = None):
        """
        混同行列のプロット
        
        Args:
            save_path: 保存先パス
        """
        print("\n📊 Plotting confusion matrices...")
        
        fig, axes = plt.subplots(1, len(FORECAST_HORIZONS), figsize=(15, 5))
        
        if len(FORECAST_HORIZONS) == 1:
            axes = [axes]
        
        for idx, h in enumerate(FORECAST_HORIZONS):
            prob_col = f'predictions.prob_{h}d'
            label_col = f'true_label_{h}d'
            
            if prob_col not in self.results_df.columns or label_col not in self.results_df.columns:
                continue
            
            y_true = self.results_df[label_col].values
            y_pred_proba = self.results_df[prob_col].values
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            
            # ヒートマップ
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Normal', 'Anomalous'],
                yticklabels=['Normal', 'Anomalous'],
                ax=axes[idx]
            )
            axes[idx].set_title(f'{h}-day Horizon')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrices saved: {save_path}")
        
        plt.close()
    
    def save_evaluation_report(self, metrics: Dict, leadtime: Dict):
        """
        評価レポートの保存
        
        Args:
            metrics: メトリクス
            leadtime: リードタイム分析結果
        """
        print("\n💾 Saving evaluation report...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'num_samples': len(self.results_df),
            'metrics': metrics,
            'leadtime_analysis': leadtime
        }
        
        report_path = RESULTS_ROOT / "evaluation_metrics.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Evaluation report saved: {report_path}")


def load_latest_results():
    """最新の推論結果をロード"""
    print("📂 Loading latest inference results...")
    
    # 最新の結果ファイルを検索
    result_files = list(RESULTS_ROOT.glob("inference_results_*.csv"))
    
    if not result_files:
        raise FileNotFoundError("No inference results found. Please run inference.py first.")
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    
    print(f"✓ Loading: {latest_file.name}")
    
    results_df = pd.read_csv(latest_file)
    
    print(f"✓ Loaded {len(results_df):,} predictions")
    
    return results_df


def main():
    """メイン実行"""
    print("="*60)
    print("📊 Pump Range Deviation Forecast - Evaluation")
    print("="*60)
    
    # 結果ロード
    results_df = load_latest_results()
    
    # 評価器作成
    evaluator = ModelEvaluator(results_df)
    
    # メトリクス計算
    metrics = evaluator.calculate_metrics()
    
    # リードタイム分析
    leadtime = evaluator.analyze_leadtime()
    
    # 可視化
    evaluator.plot_roc_curves(RESULTS_ROOT / "roc_curves.png")
    evaluator.plot_pr_curves(RESULTS_ROOT / "pr_curves.png")
    evaluator.plot_confusion_matrices(RESULTS_ROOT / "confusion_matrices.png")
    
    # レポート保存
    evaluator.save_evaluation_report(metrics, leadtime)
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
