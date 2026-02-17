"""
LSTM Baseline Model for HVAC Range Deviation Forecast
LSTMベースラインモデル実装

Granite TSモデルと比較するためのシンプルなLSTMベースライン
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import numpy as np

from config import (
    FORECAST_HORIZONS,
    LOOKBACK_DAYS,
    USE_GPU,
    GPU_ID
)


class LSTMBaselineClassifier(nn.Module):
    """
    LSTMベースライン逸脱予測分類器
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_horizons: int = len(FORECAST_HORIZONS),
        device: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            hidden_size: LSTM隠れ層次元
            num_layers: LSTM層数
            num_horizons: 予測ホライズン数
            device: 計算デバイス
        """
        super().__init__()
        
        # デバイス設定
        if device is None:
            if USE_GPU and torch.cuda.is_available():
                self.device = torch.device(f'cuda:{GPU_ID}')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  LSTM Baseline - Using device: {self.device}")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_horizons = num_horizons
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        
        # 正規化層
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Attention層（簡易）
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # 分類ヘッド追加
        self._add_classification_heads()
        
        print(f"✓ LSTM Baseline model built:")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Num layers: {self.num_layers}")
        print(f"  - Forecast horizons: {FORECAST_HORIZONS}")
        
        # パラメータ数をカウント
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
    
    def _add_classification_heads(self):
        """分類ヘッド追加（マルチホライズン）"""
        
        # 各ホライズンに対する分類ヘッド
        self.classification_heads = nn.ModuleDict({
            f"head_{h}d": nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_size // 4, 1),  # バイナリ分類
                nn.Sigmoid()
            )
            for h in FORECAST_HORIZONS
        })
        
        print(f"  ✓ Classification heads added for horizons: {FORECAST_HORIZONS}")
    
    def forward(
        self,
        input_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        順伝播
        
        Args:
            input_sequence: 入力系列 [batch_size, seq_len, 1]
            attention_mask: アテンションマスク [batch_size, seq_len] (未使用)
            
        Returns:
            各ホライズンの予測確率の辞書
        """
        # LSTM処理
        lstm_out, (hidden, cell) = self.lstm(input_sequence)
        # lstm_out: [batch_size, seq_len, hidden_size]
        
        # 最終時刻の出力を使用
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # 正規化
        normalized_output = self.layer_norm(last_output)
        
        # Attention（セルフアテンション）
        attn_output, _ = self.attention(
            lstm_out,
            lstm_out,
            lstm_out
        )
        pooled_output = attn_output[:, -1, :]  # 最終時刻 [batch_size, hidden_size]
        
        # 各ホライズンの予測
        predictions = {}
        for h in FORECAST_HORIZONS:
            head_name = f"head_{h}d"
            pred = self.classification_heads[head_name](pooled_output)
            predictions[f"prob_{h}d"] = pred.squeeze(-1)  # [batch_size]
        
        return predictions
    
    def predict(
        self,
        input_sequence: np.ndarray,
        return_probs: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        推論
        
        Args:
            input_sequence: 入力系列 [batch_size, seq_len] or [seq_len]
            return_probs: 確率を返すか（Falseの場合はバイナリラベル）
            
        Returns:
            予測結果の辞書
        """
        self.eval()
        
        # 次元調整
        if input_sequence.ndim == 1:
            input_sequence = input_sequence[np.newaxis, :, np.newaxis]  # [1, seq_len, 1]
        elif input_sequence.ndim == 2:
            input_sequence = input_sequence[:, :, np.newaxis]  # [batch_size, seq_len, 1]
        
        # Tensorに変換
        input_tensor = torch.FloatTensor(input_sequence).to(self.device)
        
        # 推論
        with torch.no_grad():
            predictions = self.forward(input_tensor)
        
        # CPU・NumPyに変換
        results = {}
        for key, value in predictions.items():
            probs = value.cpu().numpy()
            
            if return_probs:
                results[key] = probs
            else:
                # 0.5を閾値にバイナリ化
                results[key.replace('prob', 'label')] = (probs > 0.5).astype(int)
        
        return results
    
    def save_model(self, save_path: str):
        """
        モデル保存
        
        Args:
            save_path: 保存先パス
        """
        from pathlib import Path
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving LSTM baseline model to: {save_path}")
        
        # 全体を保存
        torch.save(self.state_dict(), save_path / "lstm_baseline.pt")
        
        # モデル設定も保存
        config = {
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_horizons': self.num_horizons,
            'forecast_horizons': FORECAST_HORIZONS,
            'lookback_days': LOOKBACK_DAYS
        }
        
        import json
        with open(save_path / "model_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✓ LSTM baseline model saved successfully")
    
    def load_model(self, load_path: str):
        """
        モデル読み込み
        
        Args:
            load_path: 読み込み元パス
        """
        from pathlib import Path
        load_path = Path(load_path)
        
        print(f"📂 Loading LSTM baseline model from: {load_path}")
        
        state_dict = torch.load(load_path / "lstm_baseline.pt", map_location=self.device)
        self.load_state_dict(state_dict)
        
        print("✓ LSTM baseline model loaded")
        
        self.to(self.device)


def test_model():
    """モデルのテスト"""
    print("="*60)
    print("🧪 Testing LSTM Baseline Classifier")
    print("="*60)
    
    # モデル作成
    model = LSTMBaselineClassifier()
    model.to(model.device)
    
    # ダミーデータでテスト
    batch_size = 4
    seq_len = 90
    
    dummy_input = torch.randn(batch_size, seq_len, 1).to(model.device)
    
    print(f"\n📊 Testing with input shape: {dummy_input.shape}")
    
    # 順伝播テスト
    predictions = model(dummy_input)
    
    print("\n✓ Forward pass successful")
    print("Predictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape} -> {value[0].item():.4f}")
    
    # NumPy入力でのテスト
    numpy_input = np.random.randn(seq_len)
    results = model.predict(numpy_input)
    
    print("\n✓ Prediction successful")
    print("Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ LSTM Baseline model test complete!")
    print("="*60)


if __name__ == "__main__":
    test_model()
