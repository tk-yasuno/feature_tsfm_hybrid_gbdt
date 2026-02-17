"""
Model Comparison Script
LSTMベースライン vs Granite TSモデルの比較

Usage:
    python compare_models.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import RESULTS_ROOT, FORECAST_HORIZONS

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(model_name: str):
    """
    モデルの評価結果をロード
    
    Args:
        model_name: モデル名 ('lstm_baseline', 'granite_ts', etc.)
    
    Returns:
        評価結果の辞書
    """
    results_dir = RESULTS_ROOT / model_name
    
    if not results_dir.exists():
        print(f"⚠ Warning: Results directory not found for {model_name}")
        return None
    
    # 最新の評価結果を探す
    result_files = list(results_dir.glob("evaluation_results_*.json"))
    
    if not result_files:
        print(f"⚠ Warning: No evaluation results found for {model_name}")
        return None
    
    # 最新のファイルを選択
    latest_file = sorted(result_files)[-1]
    print(f"✓ Loading results from: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results


def compare_metrics(results_dict: dict):
    """
    複数モデルのメトリクスを比較
    
    Args:
        results_dict: {model_name: results} の辞書
    """
    print("\n" + "="*80)
    print("📊 Model Comparison - Metrics")
    print("="*80)
    
    # メトリクス名
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    comparison_data = []
    
    for horizon in FORECAST_HORIZONS:
        h_key = f'{horizon}d'
        
        print(f"\n{'='*80}")
        print(f"Horizon: {horizon} days")
        print(f"{'='*80}")
        
        for model_name, results in results_dict.items():
            if results is None:
                continue
            
            h_results = results.get(h_key, {})
            
            row = {
                'Horizon': horizon,
                'Model': model_name
            }
            
            print(f"\n{model_name}:")
            for metric in metrics:
                value = h_results.get(metric, 0.0)
                row[metric.capitalize()] = value
                print(f"  {metric.capitalize()}: {value:.4f}")
            
            comparison_data.append(row)
    
    # DataFrameに変換
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df


def plot_comparison(comparison_df: pd.DataFrame, output_dir: Path):
    """
    比較グラフを作成
    
    Args:
        comparison_df: 比較データのDataFrame
        output_dir: 出力ディレクトリ
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_score', 'Auc']
    
    # 各メトリクスごとにグラフを作成
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # モデルごとにプロット
        for model in comparison_df['Model'].unique():
            model_data = comparison_df[comparison_df['Model'] == model]
            ax.plot(
                model_data['Horizon'],
                model_data[metric],
                marker='o',
                label=model,
                linewidth=2,
                markersize=8
            )
        
        ax.set_xlabel('Forecast Horizon (days)', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(comparison_df['Horizon'].unique())
        
        # 0-1の範囲に制限
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        output_file = output_dir / f'comparison_{metric.lower()}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    # 全メトリクスを一つのグラフに
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for model in comparison_df['Model'].unique():
            model_data = comparison_df[comparison_df['Model'] == model]
            ax.plot(
                model_data['Horizon'],
                model_data[metric],
                marker='o',
                label=model,
                linewidth=2,
                markersize=6
            )
        
        ax.set_xlabel('Horizon (days)', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(comparison_df['Horizon'].unique())
        ax.set_ylim([0, 1.05])
    
    # 最後のサブプロットを非表示
    axes[-1].axis('off')
    
    plt.suptitle('Model Comparison - All Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / 'comparison_all_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def create_summary_table(comparison_df: pd.DataFrame, output_dir: Path):
    """
    サマリーテーブルを作成
    
    Args:
        comparison_df: 比較データのDataFrame
        output_dir: 出力ディレクトリ
    """
    # Pivot形式に変換
    pivot_tables = {}
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1_score', 'Auc']:
        pivot = comparison_df.pivot(
            index='Horizon',
            columns='Model',
            values=metric
        )
        pivot_tables[metric] = pivot
    
    # Excelに保存
    output_file = output_dir / 'comparison_summary.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for metric, pivot in pivot_tables.items():
            pivot.to_excel(writer, sheet_name=metric)
    
    print(f"✓ Summary table saved: {output_file}")
    
    # CSV形式も保存
    csv_file = output_dir / 'comparison_summary.csv'
    comparison_df.to_csv(csv_file, index=False)
    print(f"✓ CSV saved: {csv_file}")


def main():
    """メイン関数"""
    print("="*80)
    print("🔍 Model Comparison")
    print("="*80)
    
    # 比較するモデル
    models_to_compare = [
        'lstm_baseline',
        'granite_ts',
        'hybrid_model',
        'lightgbm_baseline'
    ]
    
    # 結果をロード
    print("\n📂 Loading results...")
    results_dict = {}
    
    for model_name in models_to_compare:
        print(f"\n{model_name}:")
        results = load_results(model_name)
        if results is not None:
            results_dict[model_name] = results
    
    if not results_dict:
        print("\n⚠ No results found. Please run evaluation first.")
        return
    
    print(f"\n✓ Loaded results for {len(results_dict)} models")
    
    # メトリクス比較
    comparison_df = compare_metrics(results_dict)
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_ROOT / f"model_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Output directory: {output_dir}")
    
    # グラフ作成
    print("\n📊 Creating comparison plots...")
    plot_comparison(comparison_df, output_dir)
    
    # サマリーテーブル作成
    print("\n📄 Creating summary table...")
    create_summary_table(comparison_df, output_dir)
    
    print("\n" + "="*80)
    print("✅ Model Comparison Complete!")
    print("="*80)
    print(f"📂 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
