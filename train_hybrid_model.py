"""
Hybrid Model: Granite TS Embeddings + Statistical Features
ハイブリッドモデル - 時系列埋め込み + 統計的特徴量

アーキテクチャ:
1. Granite TS Encoder: 時系列データ → 埋め込みベクトル（64次元）
2. 統計的特徴量: 28個の手作り特徴量
3. 特徴結合: 埋め込み + 統計特徴 → LightGBM
4. 予測: 異常確率（30日、60日、90日）

期待される改善:
- LightGBM baseline: Precision 79-87%
- Hybrid model: Precision 90%+ を目指す
"""

import sys
import os

# Granite TS用の回避策：torchvisionをスキップ
sys.modules['torchvision'] = None
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PROCESSED_DATA_DIR,
    MODEL_ROOT,
    RESULTS_ROOT,
    FORECAST_HORIZONS,
    RANDOM_SEED,
    LOOKBACK_DAYS,
    USE_GPU,
    GPU_ID
)

# Granite TSモデル
from granite_ts_model import GraniteTimeSeriesClassifier

# プロット設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class HybridDataset(Dataset):
    """ハイブリッドモデル用のデータセット"""
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        """
        初期化
        
        Args:
            df: データフレーム（enrichedデータ）
            feature_cols: 統計的特徴量のカラム名リスト
        """
        self.df = df
        self.feature_cols = feature_cols
        
        # 時系列データの解析
        self.sequences = []
        for seq_str in df['values_sequence'].values:
            # 文字列をリストに変換（角カッコ付きのリスト形式）
            import ast
            try:
                values = ast.literal_eval(seq_str)
            except:
                # フォールバック: カンマ区切り
                values = [float(x.strip('[] ')) for x in seq_str.split(',') if x.strip()]
            
            # LOOKBACK_DAYS日分にパディング/トリミング
            if len(values) < LOOKBACK_DAYS:
                values = [values[0]] * (LOOKBACK_DAYS - len(values)) + values
            elif len(values) > LOOKBACK_DAYS:
                values = values[-LOOKBACK_DAYS:]
            self.sequences.append(values)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        
        # 統計的特徴量
        self.features = df[feature_cols].values.astype(np.float32)
        
        # ラベル
        self.labels = {
            f'label_{h}d': df[f'label_{h}d'].values.astype(np.int64)
            for h in FORECAST_HORIZONS
        }
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 時系列: [seq_len, 1]
        sequence = self.sequences[idx].reshape(-1, 1)
        
        # 統計的特徴量: [num_features]
        features = self.features[idx]
        
        # ラベル
        labels = {k: v[idx] for k, v in self.labels.items()}
        
        return {
            'sequence': torch.from_numpy(sequence),
            'features': torch.from_numpy(features),
            'labels': labels
        }


class HybridModel:
    """ハイブリッドモデル: Granite TS埋め込み + LightGBM"""
    
    def __init__(self, use_gpu: bool = USE_GPU):
        """
        初期化
        
        Args:
            use_gpu: GPU使用フラグ
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{GPU_ID}' if self.use_gpu else 'cpu')
        
        # Granite TS Encoder
        self.ts_encoder = None
        self.embedding_dim = 64  # TinyTimeMixerのd_model
        
        # LightGBMモデル（各ホライズンごと）
        self.lgbm_models = {}
        
        # 結果保存
        self.results = {}
        self.feature_cols = []
        
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
        
        # 特徴量カラムの特定
        exclude_cols = [
            'equipment_id', 'check_item_id', 'date', 
            'window_start', 'window_end', 'values_sequence',
            'label_current', 'label_30d', 'label_60d', 'label_90d',
            'any_anomaly'
        ]
        
        self.feature_cols = [col for col in self.train_df.columns 
                            if col not in exclude_cols]
        
        print(f"✓ Statistical feature columns: {len(self.feature_cols)}")
        
        return self.train_df, self.test_df
    
    def build_ts_encoder(self):
        """Granite TS Encoderの構築"""
        print("\n🏗️  Building Granite TS Encoder...")
        
        try:
            # Granite TSモデルをロード（分類ヘッドは使わない）
            model = GraniteTimeSeriesClassifier(
                num_horizons=len(FORECAST_HORIZONS),
                device=self.device
            )
            
            # エンコーダー部分のみを使用
            if hasattr(model, 'base_model'):
                self.ts_encoder = model.base_model
            elif hasattr(model, 'model'):
                self.ts_encoder = model.model.base_model  # PEFTモデルの場合
            elif hasattr(model, 'lstm'):
                # フォールバックLSTMの場合
                self.ts_encoder = nn.Sequential(
                    model.lstm,
                    nn.LayerNorm(model.hidden_size)
                )
                self.embedding_dim = model.hidden_size
            else:
                raise ValueError("Could not extract encoder from Granite TS model")
            
            self.ts_encoder.to(self.device)
            self.ts_encoder.eval()  # 推論モード（埋め込み抽出のみ）
            
            print(f"✓ Granite TS Encoder ready (embedding_dim={self.embedding_dim})")
            print(f"  Device: {self.device}")
            
        except Exception as e:
            print(f"⚠ Warning: Could not build Granite TS Encoder: {e}")
            print("  Falling back to simple statistical features only...")
            self.ts_encoder = None
    
    def extract_embeddings(
        self,
        df: pd.DataFrame,
        batch_size: int = 256
    ) -> np.ndarray:
        """
        時系列データから埋め込みベクトルを抽出
        
        Args:
            df: データフレーム
            batch_size: バッチサイズ
            
        Returns:
            埋め込みベクトル [num_samples, embedding_dim]
        """
        if self.ts_encoder is None:
            print("  ⚠ No encoder available, returning empty embeddings")
            return np.zeros((len(df), self.embedding_dim), dtype=np.float32)
        
        print(f"  Extracting embeddings from {len(df):,} samples...")
        
        # データセット作成
        dataset = HybridDataset(df, self.feature_cols)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        embeddings = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                sequences = batch['sequence'].to(self.device)  # [B, seq_len, 1]
                
                # Granite TS Encoderで埋め込み抽出
                try:
                    # LSTMかGranite TSかで処理を分岐
                    if hasattr(self.ts_encoder, '__getitem__') and hasattr(self.ts_encoder[0], 'forward'):
                        # Sequential（LSTM）の場合
                        lstm_out, (h, c) = self.ts_encoder[0](sequences)
                        hidden = h[-1]  # 最後の層の隠れ状態 [B, embed_dim]
                    else:
                        # Granite TS TinyTimeMixer の場合
                        # past_values: [B, seq_len, num_channels]
                        outputs = self.ts_encoder(
                            past_values=sequences,
                            output_hidden_states=True,
                            return_dict=True
                        )
                        
                        # backbone_hidden_state を使用: [B, 1, num_patches, d_model]
                        if hasattr(outputs, 'backbone_hidden_state') and outputs.backbone_hidden_state is not None:
                            backbone_hidden = outputs.backbone_hidden_state  # [B, 1, 11, 64]
                            # [B, 1, 11, 64] -> [B, 64] に変換（パッチ次元を平均化）
                            hidden = backbone_hidden.squeeze(1).mean(dim=1)  # [B, 64]
                        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                            # フォールバック: 最後の隠れ層を使用
                            last_hidden = outputs.hidden_states[-1]  # [B, 1, 11, 8]
                            hidden = last_hidden.squeeze(1).mean(dim=1)  # [B, 8]
                        else:
                            # さらなるフォールバック
                            hidden = torch.mean(sequences, dim=1).squeeze()  # [B]
                            if len(hidden.shape) == 1:
                                hidden = hidden.unsqueeze(-1)  # [B, 1]
                    
                    embeddings.append(hidden.cpu().numpy())
                    
                except Exception as e:
                    print(f"    ⚠ Error in batch {i}: {e}")
                    # フォールバック: ゼロベクトル
                    embeddings.append(np.zeros((sequences.size(0), self.embedding_dim), dtype=np.float32))
                
                if (i + 1) % 10 == 0:
                    print(f"    Progress: {(i + 1) * batch_size:,} / {len(df):,}")
        
        embeddings = np.vstack(embeddings)
        print(f"  ✓ Extracted embeddings: {embeddings.shape}")
        
        return embeddings
    
    def prepare_hybrid_features(self):
        """ハイブリッド特徴量の準備"""
        print("\n🔧 Preparing hybrid features...")
        
        # 訓練データの埋め込み抽出
        print("  Extracting train embeddings...")
        train_embeddings = self.extract_embeddings(self.train_df)
        
        # テストデータの埋め込み抽出
        print("  Extracting test embeddings...")
        test_embeddings = self.extract_embeddings(self.test_df)
        
        # 統計的特徴量と結合
        train_stat_features = self.train_df[self.feature_cols].values
        test_stat_features = self.test_df[self.feature_cols].values
        
        # 結合: [埋め込み | 統計的特徴]
        self.X_train_hybrid = np.hstack([train_embeddings, train_stat_features])
        self.X_test_hybrid = np.hstack([test_embeddings, test_stat_features])
        
        print(f"✓ Hybrid features prepared:")
        print(f"  Train shape: {self.X_train_hybrid.shape}")
        print(f"  Test shape: {self.X_test_hybrid.shape}")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Statistical features: {len(self.feature_cols)}")
        print(f"  Total features: {self.X_train_hybrid.shape[1]}")
        
        return self.X_train_hybrid, self.X_test_hybrid
    
    def get_lgbm_params(self, pos_weight: float) -> Dict:
        """LightGBMパラメータ"""
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
    
    def train_horizon(self, horizon: int):
        """特定ホライズンのモデル学習"""
        print(f"\n{'='*70}")
        print(f"Training hybrid model for {horizon}d horizon")
        print('='*70)
        
        label_col = f'label_{horizon}d'
        
        # ラベル
        y_train = self.train_df[label_col].values
        y_test = self.test_df[label_col].values
        
        print(f"\nData statistics:")
        print(f"  Train: {len(y_train):,} samples")
        print(f"  Train positives: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
        print(f"  Test: {len(y_test):,} samples")
        print(f"  Test positives: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
        
        # クラス不均衡の重み
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        print(f"  Positive class weight: {pos_weight:.2f}")
        
        # パラメータ
        params = self.get_lgbm_params(pos_weight)
        
        # データセット作成
        train_data = lgb.Dataset(self.X_train_hybrid, label=y_train)
        test_data = lgb.Dataset(self.X_test_hybrid, label=y_test, reference=train_data)
        
        # 学習
        print(f"\n🚀 Training LightGBM on hybrid features...")
        
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
        
        self.lgbm_models[horizon] = model
        
        print(f"✓ Training completed")
        print(f"  Best iteration: {model.best_iteration}")
        
        # 予測
        y_pred_proba = model.predict(self.X_test_hybrid, num_iteration=model.best_iteration)
        
        # 最適閾値の探索
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
        """全ホライズンの学習"""
        print("\n" + "="*70)
        print("🚀 Hybrid Model Training - Granite TS + Statistical Features")
        print("="*70)
        
        all_metrics = []
        
        for horizon in FORECAST_HORIZONS:
            model, metrics = self.train_horizon(horizon)
            all_metrics.append(metrics)
        
        # サマリー
        print("\n" + "="*70)
        print("📊 Training Summary")
        print("="*70)
        
        metrics_df = pd.DataFrame(all_metrics)
        print(metrics_df.to_string(index=False))
        
        return metrics_df
    
    def save_models(self):
        """モデルの保存"""
        print(f"\n💾 Saving hybrid models...")
        
        output_dir = MODEL_ROOT / "hybrid_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for horizon, model in self.lgbm_models.items():
            model_path = output_dir / f"lgbm_hybrid_{horizon}d.txt"
            model.save_model(str(model_path))
            print(f"  ✓ Saved {horizon}d model: {model_path}")
        
        # メトリクスも保存
        metrics_list = [self.results[h]['metrics'] for h in FORECAST_HORIZONS]
        metrics_df = pd.DataFrame(metrics_list)
        
        metrics_path = output_dir / 'metrics_summary.csv'
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ Saved metrics: {metrics_path}")
        
        # テストデータも保存（可視化用）
        test_features_path = PROCESSED_DATA_DIR / 'test_features_hybrid.csv'
        test_labels_path = PROCESSED_DATA_DIR / 'test_labels.csv'
        
        # 特徴量を保存
        test_features_df = pd.DataFrame(
            self.X_test_hybrid,
            columns=[f'emb_{i}' for i in range(self.embedding_dim)] + self.feature_cols
        )
        test_features_df.to_csv(test_features_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ Saved test features: {test_features_path}")
        
        # ラベルを保存
        test_labels_df = self.test_df[[f'label_{h}d' for h in FORECAST_HORIZONS]]
        test_labels_df.to_csv(test_labels_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ Saved test labels: {test_labels_path}")
    
    def compare_models(self):
        """3つのモデルを比較"""
        print(f"\n📊 Comparing all models...")
        
        # LightGBM baseline
        lgbm_path = MODEL_ROOT / "lightgbm_baseline" / "metrics_summary.csv"
        if lgbm_path.exists():
            lgbm_df = pd.read_csv(lgbm_path)
            lgbm_df['Model'] = 'LightGBM'
        else:
            lgbm_df = pd.DataFrame()
        
        # Hybrid model
        hybrid_metrics = [self.results[h]['metrics'] for h in FORECAST_HORIZONS]
        hybrid_df = pd.DataFrame(hybrid_metrics)
        hybrid_df['Model'] = 'Hybrid'
        
        # Granite TS（元のモデル）
        granite_files = sorted(RESULTS_ROOT.glob("predictions_viz_*/metrics_summary.csv"))
        if granite_files:
            granite_df = pd.read_csv(granite_files[-1])
            granite_df['Model'] = 'Granite TS'
            granite_df = granite_df.rename(columns={'Horizon': 'horizon'})
            granite_df['horizon'] = granite_df['horizon'].str.replace('d', '').astype(int)
            granite_df = granite_df.rename(columns={
                'Precision': 'precision',
                'Recall': 'recall',
                'F1-Score': 'f1',
                'ROC-AUC': 'roc_auc'
            })
        else:
            granite_df = pd.DataFrame()
        
        # 結合
        comparison_dfs = []
        if not granite_df.empty:
            comparison_dfs.append(granite_df[['Model', 'horizon', 'precision', 'recall', 'f1', 'roc_auc']])
        if not lgbm_df.empty:
            comparison_dfs.append(lgbm_df[['Model', 'horizon', 'precision', 'recall', 'f1', 'roc_auc']])
        comparison_dfs.append(hybrid_df[['Model', 'horizon', 'precision', 'recall', 'f1', 'roc_auc']])
        
        comparison_df = pd.concat(comparison_dfs, ignore_index=True)
        
        print("\n" + "="*70)
        print("Model Comparison: Granite TS vs LightGBM vs Hybrid")
        print("="*70)
        print(comparison_df.to_string(index=False))
        
        # 保存
        output_dir = RESULTS_ROOT / "hybrid_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_path = output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 Comparison saved: {comparison_path}")
        
        return comparison_df


def main():
    """メイン処理"""
    hybrid = HybridModel()
    
    # データロード
    hybrid.load_data()
    
    # Granite TS Encoderの構築
    hybrid.build_ts_encoder()
    
    # ハイブリッド特徴量の準備（埋め込み + 統計特徴）
    hybrid.prepare_hybrid_features()
    
    # 全ホライズンの学習
    metrics_df = hybrid.train_all_horizons()
    
    # モデル保存
    hybrid.save_models()
    
    # 全モデルの比較
    hybrid.compare_models()
    
    print("\n" + "="*70)
    print("✅ Hybrid Model Training Completed!")
    print("="*70)


if __name__ == "__main__":
    main()
