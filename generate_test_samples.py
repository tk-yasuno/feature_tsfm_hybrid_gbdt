"""
Generate Test Samples
テストサンプルの生成（学習時と同じ分割）

このスクリプトは学習を再実行せずに、train.pyと同じロジックで
テストサンプルを生成します。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

from config import (
    PROCESSED_DATA_DIR,
    TRAINING_CONFIG,
    RANDOM_SEED
)


def generate_test_samples():
    """テストサンプルの生成"""
    print("="*60)
    print("テストサンプル生成")
    print("="*60)
    
    # training_samples.csvの読み込み
    samples_path = PROCESSED_DATA_DIR / "training_samples.csv"
    
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Training samples not found: {samples_path}\n"
            "Please run range_definition.py first."
        )
    
    print(f"\n📂 Loading training samples from: {samples_path}")
    samples_df = pd.read_csv(samples_path)
    print(f"✓ Loaded {len(samples_df):,} samples")
    
    # Create composite label for stratification
    samples_df['any_anomaly'] = ((samples_df['label_30d'] == 1) | 
                                   (samples_df['label_60d'] == 1) | 
                                   (samples_df['label_90d'] == 1)).astype(int)
    
    anomaly_count = samples_df['any_anomaly'].sum()
    print(f"✓ Anomaly samples: {anomaly_count} ({anomaly_count/len(samples_df)*100:.1f}%)")
    
    # データ分割 (train.pyと同じロジック)
    print("\n🔀 Splitting data (same as train.py)...")
    
    train_ratio = TRAINING_CONFIG['train_ratio']
    val_ratio = TRAINING_CONFIG['val_ratio']
    test_ratio = TRAINING_CONFIG['test_ratio']
    
    print(f"  Train: {train_ratio:.1%}")
    print(f"  Val: {val_ratio:.1%}")
    print(f"  Test: {test_ratio:.1%}")
    print(f"  Random seed: {RANDOM_SEED}")
    
    # First split: train+val vs test (stratified)
    train_val_df, test_df = train_test_split(
        samples_df,
        test_size=test_ratio,
        stratify=samples_df['any_anomaly'],
        random_state=RANDOM_SEED
    )
    
    print(f"\n✓ Split completed:")
    print(f"  Train+Val: {len(train_val_df):,} samples")
    print(f"  Test: {len(test_df):,} samples")
    print(f"  Test anomalies: {test_df['any_anomaly'].sum()} "
          f"({test_df['any_anomaly'].sum()/len(test_df)*100:.1f}%)")
    
    # テストデータの保存
    test_df_path = PROCESSED_DATA_DIR / "test_samples.csv"
    test_df.to_csv(test_df_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Test samples saved: {test_df_path}")
    
    # 統計情報の表示
    print("\n📊 Test set statistics:")
    print(f"  Total samples: {len(test_df):,}")
    print(f"  Unique equipment: {test_df['equipment_id'].nunique()}")
    print(f"  Unique check items: {test_df['check_item_id'].nunique()}")
    
    for h in [30, 60, 90]:
        label_col = f'label_{h}d'
        if label_col in test_df.columns:
            anomaly_count = test_df[label_col].sum()
            print(f"  {h}d anomalies: {anomaly_count} ({anomaly_count/len(test_df)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ 完了!")
    print("="*60)


if __name__ == "__main__":
    generate_test_samples()
