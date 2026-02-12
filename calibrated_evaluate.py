"""
Evaluation with Calibrated Model
キャリブレーション済みモデルの評価
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from config import PROJECT_ROOT, RESULTS_ROOT, FORECAST_HORIZONS


def load_latest_results(pattern="calibrated_inference_results_*.csv"):
    """Load the most recent calibrated inference results"""
    results_files = list(RESULTS_ROOT.glob(pattern))
    if not results_files:
        raise FileNotFoundError(f"No results files found matching {pattern}")
    
    latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
    return pd.read_csv(latest_file), latest_file.name


def calculate_metrics(y_true, y_pred_proba, y_pred_binary):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    else:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
    
    # Classification metrics
    metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred_binary, zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Additional metrics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Specificity and sensitivity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


def compare_calibrated_vs_raw(df):
    """Compare calibrated vs raw predictions"""
    print("\n📊 Comparing Calibrated vs Raw Predictions")
    print("=" * 60)
    
    comparison = {}
    
    for h in FORECAST_HORIZONS:
        print(f"\n{h}d Horizon:")
        print("-" * 40)
        
        # Get data
        true_labels = df[f'true_label_{h}d'].values
        raw_probs = df[f'raw_prob_{h}d'].values
        calibrated_probs = df[f'calibrated_prob_{h}d'].values
        binary_preds = df[f'binary_pred_{h}d'].values
        
        # Filter out unknown labels (-1)
        valid_mask = true_labels != -1
        true_labels = true_labels[valid_mask]
        raw_probs = raw_probs[valid_mask]
        calibrated_probs = calibrated_probs[valid_mask]
        binary_preds = binary_preds[valid_mask]
        
        if len(true_labels) == 0:
            print("  ⚠️  No valid labels for this horizon")
            continue
        
        # Calculate metrics for raw predictions
        raw_binary = (raw_probs >= 0.5).astype(int)
        raw_metrics = calculate_metrics(true_labels, raw_probs, raw_binary)
        
        # Calculate metrics for calibrated predictions
        calibrated_metrics = calculate_metrics(true_labels, calibrated_probs, binary_preds)
        
        # Display comparison
        print(f"\n  RAW Predictions (threshold=0.5):")
        print(f"    ROC-AUC: {raw_metrics['roc_auc']:.4f}")
        print(f"    PR-AUC: {raw_metrics['pr_auc']:.4f}")
        print(f"    Precision: {raw_metrics['precision']:.4f}")
        print(f"    Recall: {raw_metrics['recall']:.4f}")
        print(f"    F1-Score: {raw_metrics['f1_score']:.4f}")
        print(f"    TP: {raw_metrics['true_positives']}, FP: {raw_metrics['false_positives']}")
        print(f"    FN: {raw_metrics['false_negatives']}, TN: {raw_metrics['true_negatives']}")
        
        print(f"\n  CALIBRATED Predictions (optimal threshold):")
        print(f"    ROC-AUC: {calibrated_metrics['roc_auc']:.4f}")
        print(f"    PR-AUC: {calibrated_metrics['pr_auc']:.4f}")
        print(f"    Precision: {calibrated_metrics['precision']:.4f}")
        print(f"    Recall: {calibrated_metrics['recall']:.4f}")
        print(f"    F1-Score: {calibrated_metrics['f1_score']:.4f}")
        print(f"    TP: {calibrated_metrics['true_positives']}, FP: {calibrated_metrics['false_positives']}")
        print(f"    FN: {calibrated_metrics['false_negatives']}, TN: {calibrated_metrics['true_negatives']}")
        
        # Calculate improvement
        improvements = {}
        for metric in ['precision', 'recall', 'f1_score']:
            if raw_metrics[metric] > 0:
                improvement = ((calibrated_metrics[metric] - raw_metrics[metric]) / 
                              raw_metrics[metric] * 100)
            else:
                improvement = float('inf') if calibrated_metrics[metric] > 0 else 0
            improvements[metric] = improvement
        
        print(f"\n  IMPROVEMENTS:")
        for metric, improvement in improvements.items():
            if improvement == float('inf'):
                print(f"    {metric.capitalize()}: +∞%")
            else:
                symbol = "+" if improvement >= 0 else ""
                print(f"    {metric.capitalize()}: {symbol}{improvement:.1f}%")
        
        comparison[h] = {
            'raw': raw_metrics,
            'calibrated': calibrated_metrics,
            'improvements': improvements
        }
    
    return comparison


def plot_probability_distributions(df, save_path):
    """Plot probability distributions before and after calibration"""
    print("\n📊 Plotting probability distributions...")
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    for idx, h in enumerate(FORECAST_HORIZONS):
        # Get data
        true_labels = df[f'true_label_{h}d'].values
        raw_probs = df[f'raw_prob_{h}d'].values
        calibrated_probs = df[f'calibrated_prob_{h}d'].values
        
        # Filter valid labels
        valid_mask = true_labels != -1
        true_labels = true_labels[valid_mask]
        raw_probs = raw_probs[valid_mask]
        calibrated_probs = calibrated_probs[valid_mask]
        
        if len(true_labels) == 0:
            continue
        
        # Raw probabilities
        ax_raw = axes[idx, 0]
        for label in [0, 1]:
            mask = true_labels == label
            if mask.sum() > 0:
                ax_raw.hist(raw_probs[mask], bins=30, alpha=0.5, 
                           label=f'Class {label}', density=True)
        ax_raw.set_xlabel('Raw Probability')
        ax_raw.set_ylabel('Density')
        ax_raw.set_title(f'{h}d Horizon - Raw Predictions')
        ax_raw.legend()
        ax_raw.grid(True, alpha=0.3)
        
        # Calibrated probabilities
        ax_cal = axes[idx, 1]
        for label in [0, 1]:
            mask = true_labels == label
            if mask.sum() > 0:
                ax_cal.hist(calibrated_probs[mask], bins=30, alpha=0.5,
                           label=f'Class {label}', density=True)
        ax_cal.set_xlabel('Calibrated Probability')
        ax_cal.set_ylabel('Density')
        ax_cal.set_title(f'{h}d Horizon - Calibrated Predictions')
        ax_cal.legend()
        ax_cal.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Probability distributions saved: {save_path}")


def plot_confusion_matrices_comparison(df, save_path):
    """Plot confusion matrices for raw vs calibrated"""
    print("\n📊 Plotting confusion matrices comparison...")
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    
    for idx, h in enumerate(FORECAST_HORIZONS):
        # Get data
        true_labels = df[f'true_label_{h}d'].values
        raw_probs = df[f'raw_prob_{h}d'].values
        calibrated_preds = df[f'binary_pred_{h}d'].values
        
        # Filter valid
        valid_mask = true_labels != -1
        true_labels = true_labels[valid_mask]
        raw_probs = raw_probs[valid_mask]
        calibrated_preds = calibrated_preds[valid_mask]
        
        if len(true_labels) == 0:
            continue
        
        # Raw confusion matrix
        raw_binary = (raw_probs >= 0.5).astype(int)
        cm_raw = confusion_matrix(true_labels, raw_binary)
        
        ax_raw = axes[idx, 0]
        sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax_raw)
        ax_raw.set_xlabel('Predicted')
        ax_raw.set_ylabel('Actual')
        ax_raw.set_title(f'{h}d - Raw (threshold=0.5)')
        
        # Calibrated confusion matrix
        cm_cal = confusion_matrix(true_labels, calibrated_preds)
        
        ax_cal = axes[idx, 1]
        sns.heatmap(cm_cal, annot=True, fmt='d', cmap='Greens', ax=ax_cal)
        ax_cal.set_xlabel('Predicted')
        ax_cal.set_ylabel('Actual')
        ax_cal.set_title(f'{h}d - Calibrated (optimal threshold)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved: {save_path}")


def analyze_alert_distribution(df):
    """Analyze alert level distribution"""
    print("\n🚨 Alert Level Distribution:")
    print("=" * 60)
    
    alert_counts = df['alert_level'].value_counts()
    total = len(df)
    
    for level in ['NORMAL', 'CAUTION', 'WARNING', 'CRITICAL']:
        count = alert_counts.get(level, 0)
        percentage = count / total * 100
        print(f"  {level:10s}: {count:4d} ({percentage:5.1f}%)")
    
    return alert_counts.to_dict()


def main():
    print("=" * 60)
    print("📊 Calibrated Model Evaluation")
    print("=" * 60)
    
    # Load latest results
    print("\n📂 Loading latest calibrated inference results...")
    df, filename = load_latest_results()
    print(f"✓ Loaded: {filename}")
    print(f"✓ {len(df)} samples")
    
    # Compare calibrated vs raw
    comparison = compare_calibrated_vs_raw(df)
    
    # Analyze alert distribution
    alert_dist = analyze_alert_distribution(df)
    
    # Plot probability distributions
    prob_dist_path = RESULTS_ROOT / 'calibrated_probability_distributions.png'
    plot_probability_distributions(df, prob_dist_path)
    
    # Plot confusion matrices comparison
    cm_path = RESULTS_ROOT / 'calibrated_confusion_matrices_comparison.png'
    plot_confusion_matrices_comparison(df, cm_path)
    
    # Save evaluation report
    print("\n💾 Saving evaluation report...")
    report = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(df),
        'comparison': {
            str(h): {
                'raw': {k: v for k, v in comp['raw'].items() 
                       if k != 'confusion_matrix'},
                'calibrated': {k: v for k, v in comp['calibrated'].items() 
                              if k != 'confusion_matrix'},
                'improvements': comp['improvements']
            }
            for h, comp in comparison.items()
        },
        'alert_distribution': alert_dist
    }
    
    report_path = RESULTS_ROOT / 'calibrated_evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Evaluation report saved: {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ Calibrated Evaluation Complete!")
    print("=" * 60)
    
    # Summary
    print("\n📋 Summary:")
    for h in FORECAST_HORIZONS:
        if h in comparison:
            cal_f1 = comparison[h]['calibrated']['f1_score']
            print(f"  {h}d horizon F1-Score: {cal_f1:.4f}")


if __name__ == "__main__":
    main()
