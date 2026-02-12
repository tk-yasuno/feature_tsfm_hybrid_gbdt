"""
Training Script for Granite TS Pump Deviation Forecast
Granite TSモデルのトレーニングスクリプト

機能:
1. 学習データのロードと分割
2. データローダー構築
3. モデルのトレーニング
4. Early Stopping
5. 学習履歴の保存
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import ast
import warnings
warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE
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

from granite_ts_model import GraniteTimeSeriesClassifier


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
            labels[f'label_{h}d'] = float(row[label_col])
        
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
        bce_loss = nn.functional.binary_cross_entropy(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class Trainer:
    """トレーニングクラス"""
    
    def __init__(
        self,
        model: GraniteTimeSeriesClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict = TRAINING_CONFIG
    ):
        """
        初期化
        
        Args:
            model: モデル
            train_loader: 訓練データローダー
            val_loader: 検証データローダー
            config: トレーニング設定
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # オプティマイザ
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 損失関数
        if config.get('use_class_weights', False):
            self.criterion = FocalLoss(gamma=config.get('focal_loss_gamma', 2.0))
        else:
            self.criterion = nn.BCELoss()
        
        # 学習率スケジューラ
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Early Stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = config['patience']
        
        # 履歴
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> float:
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            sequences = batch['sequence'].to(self.model.device)
            labels = {k: v.float().to(self.model.device) for k, v in batch['labels'].items()}
            
            # 順伝播
            predictions = self.model(sequences)
            
            # 損失計算（各ホライズンの平均）
            loss = 0
            for h in FORECAST_HORIZONS:
                pred_key = f'prob_{h}d'
                label_key = f'label_{h}d'
                loss += self.criterion(predictions[pred_key], labels[label_key])
            
            loss = loss / len(FORECAST_HORIZONS)
            
            # 勾配降下
            self.optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['max_grad_norm']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """検証"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                sequences = batch['sequence'].to(self.model.device)
                labels = {k: v.float().to(self.model.device) for k, v in batch['labels'].items()}
                
                # 順伝播
                predictions = self.model(sequences)
                
                # 損失計算
                loss = 0
                for h in FORECAST_HORIZONS:
                    pred_key = f'prob_{h}d'
                    label_key = f'label_{h}d'
                    loss += self.criterion(predictions[pred_key], labels[label_key])
                
                loss = loss / len(FORECAST_HORIZONS)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int = None):
        """
        トレーニング実行
        
        Args:
            num_epochs: エポック数
        """
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        print("="*60)
        print("🚀 Starting Training")
        print("="*60)
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Device: {self.model.device}")
        print("="*60)
        
        import time
        start_time = time.time()
        epoch_times = []
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"📊 Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # 訓練
            train_loss = self.train_epoch()
            
            # 検証
            val_loss = self.validate()
            
            # エポック時間計算
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = num_epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_minutes = eta_seconds / 60
            
            # 学習率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 履歴記録
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # ログ出力（より詳細に）
            print(f"\n📈 Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {current_lr:.2e}")
            print(f"  Time:       {epoch_time:.1f}s")
            print(f"  ETA:        {eta_minutes:.1f}min ({remaining_epochs} epochs left)")
            
            # スケジューラ更新
            self.scheduler.step(val_loss)
            
            # Early Stopping チェック
            if val_loss < self.best_val_loss - self.config['min_delta']:
                improvement = self.best_val_loss - val_loss
                print(f"\n✅ Val loss improved: {self.best_val_loss:.4f} -> {val_loss:.4f} (Δ {improvement:.4f})")
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # ベストモデル保存
                model_save_path = MODEL_ROOT / "granite_pump_lora" / "best_model"
                model_save_path.mkdir(parents=True, exist_ok=True)
                self.model.save_model(str(model_save_path))
                print(f"💾 Best model saved to: {model_save_path}")
            else:
                print(f"\n⚠️  No improvement (patience: {self.patience_counter + 1}/{self.patience})")
                self.patience_counter += 1
            
            # 途中経過保存（5エポックごと）
            if (epoch + 1) % 5 == 0:
                history_path = RESULTS_ROOT / "training_history.json"
                with open(history_path, 'w') as f:
                    json.dump(self.history, f, indent=2)
                print(f"💾 Training history saved (Epoch {epoch+1})")
            
            # Early Stopping
            if self.patience_counter >= self.patience:
                print(f"\n{'='*60}")
                print(f"⏹️  Early stopping triggered at epoch {epoch+1}")
                print(f"   Best Val Loss: {self.best_val_loss:.4f}")
                print(f"{'='*60}")
                break
        
        # トレーニング完了サマリー
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)
        print(f"📊 Training Summary:")
        print(f"  Total Epochs:     {epoch+1}/{num_epochs}")
        print(f"  Total Time:       {total_time/60:.1f} minutes")
        print(f"  Best Val Loss:    {self.best_val_loss:.4f}")
        print(f"  Final Train Loss: {train_loss:.4f}")
        print(f"  Avg Epoch Time:   {sum(epoch_times)/len(epoch_times):.1f}s")
        print("="*60)


def apply_smote_to_training_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    トレーニングデータにSMOTEを適用して異常サンプルを増加
    
    Args:
        train_df: トレーニングデータフレーム
        
    Returns:
        SMOTE適用後のデータフレーム
    """
    print(f"  Before SMOTE: {len(train_df)} samples")
    print(f"    Normal: {(train_df['any_anomaly'] == 0).sum()}")
    print(f"    Anomaly: {(train_df['any_anomaly'] == 1).sum()}")
    
    # 時系列データを2D配列に変換
    sequences = []
    for idx, row in train_df.iterrows():
        if isinstance(row['values_sequence'], str):
            values = ast.literal_eval(row['values_sequence'])
        else:
            values = row['values_sequence']
        
        # 長さを統一
        values = np.array(values, dtype=np.float32)
        if len(values) < LOOKBACK_DAYS:
            values = np.pad(values, (LOOKBACK_DAYS - len(values), 0), mode='edge')
        elif len(values) > LOOKBACK_DAYS:
            values = values[-LOOKBACK_DAYS:]
        
        sequences.append(values)
    
    X = np.array(sequences)  # [num_samples, lookback_days]
    y = train_df['any_anomaly'].values.astype(int)
    
    # SMOTEを適用（異常サンプルを正常サンプルの50%まで増加）
    try:
        # k_neighbors='auto'は最小クラスのサンプル数-1を使用
        min_samples = min(np.sum(y == 0), np.sum(y == 1))
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        
        smote = SMOTE(
            sampling_strategy=0.5,  # 少数クラスを多数クラスの50%に
            k_neighbors=k_neighbors,
            random_state=RANDOM_SEED
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"  After SMOTE: {len(X_resampled)} samples")
        print(f"    Normal: {np.sum(y_resampled == 0)}")
        print(f"    Anomaly: {np.sum(y_resampled == 1)}")
        
        # 新しいDataFrameを作成（効率化版）
        new_rows = []
        
        # オリジナルサンプルと異常サンプルのインデックスを取得
        anomaly_indices = np.where(train_df['any_anomaly'] == 1)[0]
        
        for i in range(len(X_resampled)):
            if i < len(train_df):
                # オリジナルサンプル
                new_rows.append(train_df.iloc[i].to_dict())
            else:
                # SMOTEで生成されたサンプル
                # ランダムに異常サンプルからメタデータをコピー
                if len(anomaly_indices) > 0:
                    random_idx = np.random.choice(anomaly_indices)
                    row_dict = train_df.iloc[random_idx].copy().to_dict()
                else:
                    row_dict = train_df.iloc[0].copy().to_dict()
                
                # 時系列データを更新
                row_dict['values_sequence'] = X_resampled[i].tolist()
                row_dict['any_anomaly'] = int(y_resampled[i])
                
                # ラベルも更新
                if y_resampled[i] == 1:
                    for h in FORECAST_HORIZONS:
                        row_dict[f'label_{h}d'] = 1.0
                else:
                    for h in FORECAST_HORIZONS:
                        row_dict[f'label_{h}d'] = 0.0
                
                new_rows.append(row_dict)
        
        resampled_df = pd.DataFrame(new_rows)
        print(f"  ✓ SMOTE applied successfully")
        
        return resampled_df
        
    except Exception as e:
        print(f"  ⚠️  SMOTE failed: {e}")
        print(f"  Continuing with original data")
        return train_df


def load_and_split_data():
    """データのロードと分割"""
    print("📂 Loading training samples...")
    
    samples_path = PROCESSED_DATA_DIR / "training_samples.csv"
    
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Training samples not found: {samples_path}\n"
            "Please run range_definition.py first."
        )
    
    samples_df = pd.read_csv(samples_path)
    print(f"✓ Loaded {len(samples_df):,} samples")
    
    # Create composite label for stratification
    samples_df['any_anomaly'] = ((samples_df['label_30d'] == 1) | 
                                   (samples_df['label_60d'] == 1) | 
                                   (samples_df['label_90d'] == 1)).astype(int)
    anomaly_count = samples_df['any_anomaly'].sum()
    print(f"✓ Anomaly samples: {anomaly_count} ({anomaly_count/len(samples_df)*100:.1f}%)")
    
    # 訓練/検証/テスト分割 with stratification
    from sklearn.model_selection import train_test_split
    
    train_ratio = TRAINING_CONFIG['train_ratio']
    val_ratio = TRAINING_CONFIG['val_ratio']
    test_ratio = TRAINING_CONFIG['test_ratio']
    
    # First split: train+val vs test (stratified)
    train_val_df, test_df = train_test_split(
        samples_df,
        test_size=test_ratio,
        stratify=samples_df['any_anomaly'],
        random_state=RANDOM_SEED
    )
    
    # Second split: train vs val (stratified)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio / (train_ratio + val_ratio),
        stratify=train_val_df['any_anomaly'],
        random_state=RANDOM_SEED
    )
    
    print(f"✓ Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"  Train anomalies: {train_df['any_anomaly'].sum()} ({train_df['any_anomaly'].sum()/len(train_df)*100:.1f}%)")
    print(f"  Val anomalies: {val_df['any_anomaly'].sum()} ({val_df['any_anomaly'].sum()/len(val_df)*100:.1f}%)")
    
    # Apply SMOTE to training data only
    print("\n🔧 Applying SMOTE to balance training data...")
    train_df = apply_smote_to_training_data(train_df)
    
    # Create datasets from dataframes
    train_dataset = PumpDeviationDataset(train_df)
    val_dataset = PumpDeviationDataset(val_df)
    test_dataset = PumpDeviationDataset(test_df)
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    """データローダー作成"""
    batch_size = TRAINING_CONFIG['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def save_training_history(history: dict):
    """学習履歴の保存"""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    
    history_path = RESULTS_ROOT / "training_history.json"
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"💾 Training history saved: {history_path}")


def main():
    """メイン実行"""
    print("="*60)
    print("🚀 Pump Range Deviation Forecast - Training")
    print("="*60)
    
    # 再現性
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # ディレクトリ作成
    create_directories()
    
    # データロード
    train_dataset, val_dataset, test_dataset = load_and_split_data()
    
    # データローダー作成
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # モデル作成
    print("\n🏗️  Creating model...")
    model = GraniteTimeSeriesClassifier()
    model.to(model.device)
    
    # トレーナー作成
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TRAINING_CONFIG
    )
    
    # トレーニング実行
    trainer.train()
    
    # 履歴保存
    save_training_history(trainer.history)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
