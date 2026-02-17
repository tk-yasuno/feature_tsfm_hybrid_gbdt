"""
Training Script for LSTM Baseline Model
LSTMベースラインモデルのトレーニングスクリプト

Granite TSモデルと比較するためのLSTMベースライン
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import ast
import warnings
from typing import Dict
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from config import (
    PROCESSED_DATA_DIR,
    MODEL_ROOT,
    RESULTS_ROOT,
    TRAINING_CONFIG,
    FORECAST_HORIZONS,
    LOOKBACK_DAYS,
    RANDOM_SEED,
    create_directories
)

from lstm_baseline_model import LSTMBaselineClassifier


class PumpDeviationDataset(Dataset):
    """ポンプ逸脱予測データセット"""
    
    def __init__(self, samples_df: pd.DataFrame):
        """
        初期化
        
        Args:
            samples_df: 学習サンプルDataFrame
        """
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
        
        # 長さを調整（パディングまたはトリミング）
        if len(values) < LOOKBACK_DAYS:
            # 前方パディング
            values = np.pad(values, (LOOKBACK_DAYS - len(values), 0), mode='edge')
        elif len(values) > LOOKBACK_DAYS:
            # 最新データを使用
            values = values[-LOOKBACK_DAYS:]
        
        # reshape
        values = values.reshape(-1, 1)  # [seq_len, 1]
        
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


class FocalLoss(nn.Module):
    """Focal Loss（クラス不均衡対応）"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 予測確率 [batch_size]
            targets: 正解ラベル [batch_size]
        """
        bce_loss = nn.functional.binary_cross_entropy(
            inputs, targets, reduction='none'
        )
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


def train_one_epoch(
    model: LSTMBaselineClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict:
    """
    1エポックのトレーニング
    
    Args:
        model: モデル
        dataloader: データローダー
        optimizer: オプティマイザ
        criterion: 損失関数
        device: デバイス
        epoch: エポック番号
    
    Returns:
        損失と精度の辞書
    """
    model.train()
    
    total_loss = 0.0
    horizon_losses = {h: 0.0 for h in FORECAST_HORIZONS}
    horizon_correct = {h: 0 for h in FORECAST_HORIZONS}
    horizon_total = {h: 0 for h in FORECAST_HORIZONS}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        sequences = batch['sequence'].to(device)
        labels = {k: v.to(device) for k, v in batch['labels'].items()}
        
        # 順伝播
        predictions = model(sequences)
        
        # 各ホライズンの損失計算
        loss = 0
        for h in FORECAST_HORIZONS:
            prob_key = f'prob_{h}d'
            label_key = f'label_{h}d'
            
            h_loss = criterion(predictions[prob_key], labels[label_key])
            loss += h_loss
            horizon_losses[h] += h_loss.item()
            
            # 精度計算
            preds_binary = (predictions[prob_key] > 0.5).float()
            correct = (preds_binary == labels[label_key]).sum().item()
            horizon_correct[h] += correct
            horizon_total[h] += labels[label_key].size(0)
        
        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # プログレスバー更新
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # エポック平均
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    
    metrics = {
        'loss': avg_loss,
        'horizons': {}
    }
    
    for h in FORECAST_HORIZONS:
        metrics['horizons'][f'{h}d'] = {
            'loss': horizon_losses[h] / num_batches,
            'accuracy': horizon_correct[h] / horizon_total[h] if horizon_total[h] > 0 else 0.0
        }
    
    return metrics


def validate(
    model: LSTMBaselineClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict:
    """
    検証
    
    Args:
        model: モデル
        dataloader: データローダー
        criterion: 損失関数
        device: デバイス
    
    Returns:
        損失と精度の辞書
    """
    model.eval()
    
    total_loss = 0.0
    horizon_losses = {h: 0.0 for h in FORECAST_HORIZONS}
    horizon_correct = {h: 0 for h in FORECAST_HORIZONS}
    horizon_total = {h: 0 for h in FORECAST_HORIZONS}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            sequences = batch['sequence'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            # 順伝播
            predictions = model(sequences)
            
            # 各ホライズンの損失計算
            loss = 0
            for h in FORECAST_HORIZONS:
                prob_key = f'prob_{h}d'
                label_key = f'label_{h}d'
                
                h_loss = criterion(predictions[prob_key], labels[label_key])
                loss += h_loss
                horizon_losses[h] += h_loss.item()
                
                # 精度計算
                preds_binary = (predictions[prob_key] > 0.5).float()
                correct = (preds_binary == labels[label_key]).sum().item()
                horizon_correct[h] += correct
                horizon_total[h] += labels[label_key].size(0)
            
            total_loss += loss.item()
    
    # 平均
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    
    metrics = {
        'loss': avg_loss,
        'horizons': {}
    }
    
    for h in FORECAST_HORIZONS:
        metrics['horizons'][f'{h}d'] = {
            'loss': horizon_losses[h] / num_batches,
            'accuracy': horizon_correct[h] / horizon_total[h] if horizon_total[h] > 0 else 0.0
        }
    
    return metrics


def main():
    """メイン関数"""
    print("="*80)
    print("🚀 LSTM Baseline Training (Focal Loss gamma=3.0)")
    print("="*80)
    
    # ディレクトリ作成
    create_directories()
    
    # 出力ディレクトリ（gamma=3専用）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = MODEL_ROOT / "lstm_baseline_gamma3"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = RESULTS_ROOT / "lstm_baseline_gamma3"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Output directory: {output_dir}")
    print(f"📂 Results directory: {results_dir}")
    
    # デバイス設定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # データロード
    print("\n" + "="*80)
    print("📊 Loading data...")
    print("="*80)
    
    samples_path = PROCESSED_DATA_DIR / "training_samples.csv"
    if not samples_path.exists():
        raise FileNotFoundError(f"Training samples not found: {samples_path}")
    
    df = pd.read_csv(samples_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # クラス不均衡の確認
    print(f"\n📊 Class Distribution:")
    for h in FORECAST_HORIZONS:
        label_col = f'label_{h}d'
        pos_count = df[label_col].sum()
        neg_count = len(df) - pos_count
        pos_ratio = pos_count / len(df) * 100
        imbalance_ratio = neg_count / max(pos_count, 1)
        print(f"  {h}d: Positive={int(pos_count)} ({pos_ratio:.2f}%), Negative={int(neg_count)} ({100-pos_ratio:.2f}%), Imbalance Ratio={imbalance_ratio:.2f}:1")
    
    # データ分割
    train_df, val_df = train_test_split(
        df,
        test_size=TRAINING_CONFIG.get('val_split', 0.2),
        random_state=RANDOM_SEED,
        stratify=df['label_30d']  # 30日ラベルで層化分割
    )
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    
    # データセット作成
    train_dataset = PumpDeviationDataset(train_df)
    val_dataset = PumpDeviationDataset(val_df)
    
    # データローダー作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # モデル作成
    print("\n" + "="*80)
    print("🏗️  Building model...")
    print("="*80)
    
    model = LSTMBaselineClassifier()
    model.to(device)
    
    # オプティマイザ
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG.get('weight_decay', 0.01)
    )
    
    # 学習率スケジューラ
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 損失関数（クラス不均衡に強く対応: gamma=3.0）
    print(f"📊 Loss Function: Focal Loss (alpha={0.25}, gamma={3.0})")
    print(f"   Higher gamma focuses more on hard-to-classify examples")
    criterion = FocalLoss(alpha=0.25, gamma=3.0)
    
    # トレーニング
    print("\n" + "="*80)
    print("🔥 Training...")
    print("="*80)
    
    history = {
        'train': [],
        'val': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = TRAINING_CONFIG.get('patience', 10)
    
    for epoch in range(1, TRAINING_CONFIG['num_epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{TRAINING_CONFIG['num_epochs']}")
        print(f"{'='*80}")
        
        # トレーニング
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        history['train'].append(train_metrics)
        
        print(f"\n[Train] Loss: {train_metrics['loss']:.4f}")
        for h in FORECAST_HORIZONS:
            h_metrics = train_metrics['horizons'][f'{h}d']
            print(f"  {h}d - Loss: {h_metrics['loss']:.4f}, Acc: {h_metrics['accuracy']:.4f}")
        
        # 検証
        val_metrics = validate(model, val_loader, criterion, device)
        history['val'].append(val_metrics)
        
        print(f"\n[Val] Loss: {val_metrics['loss']:.4f}")
        for h in FORECAST_HORIZONS:
            h_metrics = val_metrics['horizons'][f'{h}d']
            print(f"  {h}d - Loss: {h_metrics['loss']:.4f}, Acc: {h_metrics['accuracy']:.4f}")
        
        # 学習率スケジューラ
        scheduler.step(val_metrics['loss'])
        
        # Early Stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # ベストモデル保存
            model.save_model(output_dir / "best_model")
            print(f"\n💾 Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"\n⏳ Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                break
    
    # 学習履歴保存
    print("\n" + "="*80)
    print("💾 Saving training history...")
    print("="*80)
    
    with open(results_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✓ Training history saved to: {results_dir / 'training_history.json'}")
    
    print("\n" + "="*80)
    print("✅ LSTM Baseline Training Complete!")
    print("="*80)
    print(f"📂 Best model saved to: {output_dir / 'best_model'}")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
