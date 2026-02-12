"""
Inference Script for Pump Range Deviation Forecast
推論スクリプト

機能:
1. 訓練済みモデルのロード
2. テストデータでの推論
3. 逸脱確率の予測
4. アラート判定
5. 結果の可視化
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from config import (
    PROCESSED_DATA_DIR,
    MODEL_ROOT,
    RESULTS_ROOT,
    INFERENCE_CONFIG,
    FORECAST_HORIZONS,
    LOOKBACK_DAYS
)

from granite_ts_model import GraniteTimeSeriesClassifier


class PumpDeviationInference:
    """ポンプ逸脱予測推論クラス"""
    
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
        self.alert_threshold_warning = INFERENCE_CONFIG['alert_threshold_warning']
        self.alert_threshold_critical = INFERENCE_CONFIG['alert_threshold_critical']
        
    def load_model(self):
        """モデルのロード"""
        print(f"📂 Loading model from: {self.model_path}")
        
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
    
    def predict_single(
        self,
        sequence: np.ndarray,
        equipment_id: str = None,
        check_item_id: str = None
    ) -> Dict:
        """
        単一系列の予測
        
        Args:
            sequence: 時系列データ [seq_len]
            equipment_id: 設備ID
            check_item_id: チェック項目ID
            
        Returns:
            予測結果の辞書
        """
        # 予測
        predictions = self.model.predict(sequence, return_probs=True)
        
        # アラートレベル判定
        alerts = {}
        for h in FORECAST_HORIZONS:
            prob_key = f'prob_{h}d'
            prob = predictions[prob_key][0]
            
            if prob >= self.alert_threshold_critical:
                alert_level = "CRITICAL"
            elif prob >= self.alert_threshold_warning:
                alert_level = "WARNING"
            else:
                alert_level = "NORMAL"
            
            alerts[f'alert_{h}d'] = alert_level
        
        result = {
            'equipment_id': equipment_id,
            'check_item_id': check_item_id,
            'predictions': {k: float(v[0]) for k, v in predictions.items()},
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(
        self,
        test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        バッチ予測
        
        Args:
            test_df: テストデータDataFrame
            
        Returns:
            予測結果DataFrame
        """
        print(f"\n🔮 Running batch inference on {len(test_df)} samples...")
        
        results = []
        
        for idx, row in test_df.iterrows():
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
            
            # 予測
            result = self.predict_single(
                sequence,
                equipment_id=row['equipment_id'],
                check_item_id=row['check_item_id']
            )
            
            # 真のラベルを追加
            for h in FORECAST_HORIZONS:
                label_col = f'label_{h}d'
                if label_col in row:
                    result[f'true_label_{h}d'] = int(row[label_col])
            
            results.append(result)
        
        # DataFrame化
        results_df = pd.json_normalize(results)
        
        print(f"✓ Batch inference complete")
        
        return results_df
    
    def generate_alerts(
        self,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        アラート生成
        
        Args:
            results_df: 予測結果DataFrame
            
        Returns:
            アラートDataFrame
        """
        print("\n🚨 Generating alerts...")
        
        alerts = []
        
        for _, row in results_df.iterrows():
            for h in FORECAST_HORIZONS:
                alert_col = f'alerts.alert_{h}d'
                prob_col = f'predictions.prob_{h}d'
                
                if alert_col in row and row[alert_col] != "NORMAL":
                    alert = {
                        'equipment_id': row['equipment_id'],
                        'check_item_id': row['check_item_id'],
                        'horizon': f'{h} days',
                        'probability': row[prob_col],
                        'alert_level': row[alert_col],
                        'timestamp': row['timestamp']
                    }
                    alerts.append(alert)
        
        alerts_df = pd.DataFrame(alerts)
        
        if len(alerts_df) > 0:
            # 確率でソート
            alerts_df = alerts_df.sort_values('probability', ascending=False)
            
            print(f"✓ Generated {len(alerts_df)} alerts")
            print(f"  CRITICAL: {(alerts_df['alert_level'] == 'CRITICAL').sum()}")
            print(f"  WARNING: {(alerts_df['alert_level'] == 'WARNING').sum()}")
        else:
            print("✓ No alerts generated (all normal)")
        
        return alerts_df
    
    def save_results(
        self,
        results_df: pd.DataFrame,
        alerts_df: pd.DataFrame,
        output_prefix: str = "inference"
    ):
        """
        結果の保存
        
        Args:
            results_df: 予測結果DataFrame
            alerts_df: アラートDataFrame
            output_prefix: 出力ファイル名のプレフィックス
        """
        print("\n💾 Saving inference results...")
        
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        
        # タイムスタンプ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 予測結果
        results_path = RESULTS_ROOT / f"{output_prefix}_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"✓ Results saved: {results_path}")
        
        # アラート
        if len(alerts_df) > 0:
            alerts_path = RESULTS_ROOT / f"{output_prefix}_alerts_{timestamp}.csv"
            alerts_df.to_csv(alerts_path, index=False, encoding='utf-8-sig')
            print(f"✓ Alerts saved: {alerts_path}")
        
        # サマリー統計
        summary = {
            'timestamp': timestamp,
            'num_samples': len(results_df),
            'num_alerts': len(alerts_df),
            'alert_rate': len(alerts_df) / len(results_df) if len(results_df) > 0 else 0,
            'thresholds': {
                'warning': self.alert_threshold_warning,
                'critical': self.alert_threshold_critical
            }
        }
        
        summary_path = RESULTS_ROOT / f"{output_prefix}_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved: {summary_path}")


def load_test_data():
    """テストデータのロード"""
    print("📂 Loading test data...")
    
    samples_path = PROCESSED_DATA_DIR / "training_samples.csv"
    
    if not samples_path.exists():
        raise FileNotFoundError(f"Training samples not found: {samples_path}")
    
    samples_df = pd.read_csv(samples_path)
    
    # テストセットとして最新20%を使用
    test_size = int(len(samples_df) * 0.2)
    test_df = samples_df.iloc[-test_size:].reset_index(drop=True)
    
    print(f"✓ Loaded {len(test_df):,} test samples")
    
    return test_df


def main():
    """メイン実行"""
    print("="*60)
    print("🔮 Pump Range Deviation Forecast - Inference")
    print("="*60)
    
    # 推論器作成
    inference = PumpDeviationInference()
    
    # モデルロード
    inference.load_model()
    
    # テストデータロード
    test_df = load_test_data()
    
    # バッチ予測
    results_df = inference.predict_batch(test_df)
    
    # アラート生成
    alerts_df = inference.generate_alerts(results_df)
    
    # 結果保存
    inference.save_results(results_df, alerts_df)
    
    # トップ10アラート表示
    if len(alerts_df) > 0:
        print("\n🚨 Top 10 High-Risk Alerts:")
        print(alerts_df.head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ Inference Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
