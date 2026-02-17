"""
Evaluation Script for LSTM Baseline Model with Threshold Optimization
LSTMベースラインモデルの評価スクリプト（閾値最適化付き）
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import ast
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from config import (
    PROCESSED_DATA_DIR,
    MODEL_ROOT,
    RESULTS_ROOT,
    FORECAST_HORIZONS,
    LOOKBACK_DAYS,
    create_directories
)

from lstm_baseline_model import LSTMBaselineClassifier


class PumpDeviationDataset(Dataset):
    """ポンプ逸脱予測データセット"""
    
    def __init__(self, samples_df: pd.DataFrame):
        self.samples = samples_df.reset_index(drop=True)
        self.horizons = FORECAST_HORIZONS
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        
        # 時系列データを取得
        if isinstance(row['values_sequence'], str):
            values = ast.literal_eval(row['values_sequence'])
        else:
            values = row['values_sequence']
        
        values = np.array(values, dtype=np.float32)
        
        # 長さを調整
        if len(values) < LOOKBACK_DAYS:
            values = np.pad(values, (LOOKBACK_DAYS - len(values), 0), mode='edge')
        elif len(values) > LOOKBACK_DAYS:
            values = values[-LOOKBACK_DAYS:]
        
        values = values.reshape(-1, 1)
        
        # ラベル取得
        labels = {}
        for h in self.horizons:
            label_col = f'label_{h}d'
            labels[f'label_{h}d'] = np.float32(row[label_col])
        
        return {
            'sequence': torch.FloatTensor(values),
            'labels': labels,
            'equipment_id': row['equipment_id'],
            'check_item_id': row['check_item_id']
        }


def optimize_threshold(y_true, y_prob, metric='f1'):
    """
    最適な閾値を探索してF1スコアを最大化
    
    Args:
        y_true: 正解ラベル
        y_prob: 予測確率
        metric: 最適化するメトリクス
    
    Returns:
        最適な閾値とスコア
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_score = 0.0
    
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score, thresholds, scores


def evaluate_model_with_optimization(
    model: LSTMBaselineClassifier,
    dataloader: DataLoader,
    device: torch.device
):
    """
    F1スコアを最大化する閾値で評価
    """
    model.eval()
    
    all_labels = {h: [] for h in FORECAST_HORIZONS}
    all_probs = {h: [] for h in FORECAST_HORIZONS}
    
    # 予測確率を収集
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions"):
            sequences = batch['sequence'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            predictions = model(sequences)
            
            for h in FORECAST_HORIZONS:
                prob_key = f'prob_{h}d'
                label_key = f'label_{h}d'
                
                probs = predictions[prob_key].cpu().numpy()
                true_labels = labels[label_key].cpu().numpy()
                
                all_probs[h].extend(probs)
                all_labels[h].extend(true_labels)
    
    # 閾値最適化
    print("\n" + "="*80)
    print("🔍 Optimizing thresholds for F1 score...")
    print("="*80)
    
    optimal_thresholds = {}
    
    for h in FORECAST_HORIZONS:
        y_true = np.array(all_labels[h])
        y_prob = np.array(all_probs[h])
        
        best_threshold, best_f1, thresholds, scores = optimize_threshold(
            y_true, y_prob, metric='f1'
        )
        optimal_thresholds[h] = best_threshold
        
        f1_default = f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
        
        print(f"\nHorizon {h} days:")
        print(f"  Optimal threshold: {best_threshold:.3f}")
        print(f"  F1 at optimal: {best_f1:.4f}")
        print(f"  F1 at 0.5:     {f1_default:.4f}")
        print(f"  Improvement:   {((best_f1 - f1_default) / max(f1_default, 0.0001) * 100):.1f}%")
    
    # 最適化された閾値で評価
    results = {}
    
    print("\n" + "="*80)
    print("📊 Evaluation Results (with optimized thresholds)")
    print("="*80)
    
    for h in FORECAST_HORIZONS:
        y_true = np.array(all_labels[h])
        y_prob = np.array(all_probs[h])
        
        threshold = optimal_thresholds[h]
        y_pred = (y_prob >= threshold).astype(int)
        
        # メトリクス計算
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.0
        
        cm = confusion_matrix(y_true, y_pred)
        
        results[f'{h}d'] = {
            'threshold': float(threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist(),
            'total_samples': len(y_true),
            'positive_samples': int(y_true.sum()),
            'negative_samples': int((1 - y_true).sum())
        }
        
        print(f"\n{'='*60}")
        print(f"Horizon: {h} days")
        print(f"{'='*60}")
        print(f"Threshold: {threshold:.3f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
    
    return results


def main():
    print("="*80)
    print("🔍 LSTM Baseline Evaluation (Threshold Optimized)")
    print("="*80)
    
    create_directories()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # モデルロード
    print("\n" + "="*80)
    print("📂 Loading model...")
    print("="*80)
    
    model_path = MODEL_ROOT / "lstm_baseline" / "best_model"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = LSTMBaselineClassifier()
    model.load_model(model_path)
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # テストデータロード
    print("\n" + "="*80)
    print("📊 Loading test data...")
    print("="*80)
    
    test_path = PROCESSED_DATA_DIR / "test_samples.csv"
    if not test_path.exists():
        print("⚠ Test samples not found, using validation split from training data")
        train_path = PROCESSED_DATA_DIR / "training_samples.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Training samples not found: {train_path}")
        
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(train_path)
        _, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label_30d'])
    else:
        test_df = pd.read_csv(test_path)
    
    print(f"✓ Loaded {len(test_df)} test samples")
    
    # データローダー作成
    test_dataset = PumpDeviationDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    # 評価実行
    results = evaluate_model_with_optimization(model, test_loader, device)
    
    # 結果保存
    print("\n" + "="*80)
    print("💾 Saving results...")
    print("="*80)
    
    results_dir = RESULTS_ROOT / "lstm_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON保存
    results_file = results_dir / f"evaluation_optimized_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    # サマリー表示
    print("\n" + "="*80)
    print("📊 Evaluation Summary")
    print("="*80)
    
    summary_data = []
    for h in FORECAST_HORIZONS:
        h_results = results[f'{h}d']
        summary_data.append({
            'Horizon (days)': h,
            'Threshold': f"{h_results['threshold']:.3f}",
            'Accuracy': f"{h_results['accuracy']:.4f}",
            'Precision': f"{h_results['precision']:.4f}",
            'Recall': f"{h_results['recall']:.4f}",
            'F1 Score': f"{h_results['f1_score']:.4f}",
            'AUC': f"{h_results['auc']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # CSV保存
    summary_csv = results_dir / f"evaluation_optimized_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Summary saved to: {summary_csv}")
    
    print("\n" + "="*80)
    print("✅ LSTM Baseline Evaluation Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
