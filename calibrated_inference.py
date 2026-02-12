"""
Calibrated Inference with Optimal Thresholds
キャリブレーション済みモデルでの推論
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch

from config import (
    PROJECT_ROOT, MODEL_ROOT, RESULTS_ROOT, PROCESSED_DATA_DIR,
    FORECAST_HORIZONS, LOOKBACK_DAYS
)
from granite_ts_model import GraniteTimeSeriesClassifier


class CalibratedPredictor:
    """Predictor with probability calibration and optimal thresholds"""
    
    def __init__(self, model_path, calibration_dir, device='cpu'):
        """
        Args:
            model_path: Path to trained model
            calibration_dir: Directory containing calibrators and thresholds
            device: Device to use (cpu/cuda)
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.calibrators = {}
        self.optimal_thresholds = {}
        self._load_calibration(calibration_dir)
    
    def _load_model(self, model_path):
        """Load trained model"""
        print(f"📂 Loading model from: {model_path}")
        model = GraniteTimeSeriesClassifier(device=self.device)
        model.load_model(model_path)
        model.eval()
        print("✓ Model loaded successfully")
        return model
    
    def _load_calibration(self, calibration_dir):
        """Load calibrators and optimal thresholds"""
        calibration_dir = Path(calibration_dir)
        
        # Load calibrators
        calibrator_path = calibration_dir / 'calibrators.pkl'
        if calibrator_path.exists():
            with open(calibrator_path, 'rb') as f:
                self.calibrators = pickle.load(f)
            print(f"✓ Calibrators loaded: {len(self.calibrators)} horizons")
        else:
            print(f"⚠️  No calibrators found at {calibrator_path}")
        
        # Load optimal thresholds
        thresholds_path = calibration_dir / 'optimal_thresholds.json'
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                self.optimal_thresholds = json.load(f)
                # Convert string keys to int
                self.optimal_thresholds = {int(k): v for k, v in self.optimal_thresholds.items()}
            print(f"✓ Optimal thresholds loaded:")
            for h, threshold in self.optimal_thresholds.items():
                print(f"    {h}d: {threshold:.3f}")
        else:
            print(f"⚠️  No optimal thresholds found at {thresholds_path}")
            # Use default threshold
            self.optimal_thresholds = {h: 0.5 for h in FORECAST_HORIZONS}
    
    def predict_single(self, sequence, equipment_id, check_item_id):
        """
        Predict with calibration and optimal thresholds
        
        Args:
            sequence: Time series sequence (numpy array)
            equipment_id: Equipment ID
            check_item_id: Check item ID
            
        Returns:
            Dictionary with predictions and alert level
        """
        # Ensure correct length
        if len(sequence) < LOOKBACK_DAYS:
            sequence = np.pad(sequence, (LOOKBACK_DAYS - len(sequence), 0), mode='edge')
        elif len(sequence) > LOOKBACK_DAYS:
            sequence = sequence[-LOOKBACK_DAYS:]
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        # Get raw predictions
        with torch.no_grad():
            raw_predictions = self.model(sequence_tensor)
        
        # Apply calibration and optimal thresholds
        result = {
            'equipment_id': equipment_id,
            'check_item_id': check_item_id,
            'predictions': {},
            'calibrated_predictions': {},
            'binary_predictions': {},
            'alert_level': 'NORMAL',
            'max_probability': 0.0
        }
        
        max_prob = 0.0
        
        for h in FORECAST_HORIZONS:
            prob_key = f'prob_{h}d'
            horizon_key = f'{h}d'
            raw_prob = raw_predictions[prob_key][0].item()
            
            # Apply calibration if available
            if h in self.calibrators:
                calibrator = self.calibrators[h]
                if hasattr(calibrator, 'predict_proba'):
                    calibrated_prob = calibrator.predict_proba([[raw_prob]])[0, 1]
                else:
                    calibrated_prob = calibrator.predict([raw_prob])[0]
            else:
                calibrated_prob = raw_prob
            
            # Apply optimal threshold
            threshold = self.optimal_thresholds.get(h, 0.5)
            binary_pred = 1 if calibrated_prob >= threshold else 0
            
            result['predictions'][horizon_key] = raw_prob
            result['calibrated_predictions'][horizon_key] = float(calibrated_prob)
            result['binary_predictions'][horizon_key] = binary_pred
            
            if calibrated_prob > max_prob:
                max_prob = calibrated_prob
        
        result['max_probability'] = max_prob
        
        # Determine alert level based on calibrated probabilities and optimal thresholds
        # Use more nuanced alert levels
        if max_prob >= 0.9:
            result['alert_level'] = 'CRITICAL'
        elif max_prob >= 0.7:
            result['alert_level'] = 'WARNING'
        elif max_prob >= 0.5:
            result['alert_level'] = 'CAUTION'
        else:
            result['alert_level'] = 'NORMAL'
        
        return result
    
    def predict_batch(self, test_df):
        """
        Batch prediction with calibration
        
        Args:
            test_df: Test dataframe with values_sequence column
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for idx, row in test_df.iterrows():
            # Parse sequence
            values = eval(row['values_sequence'])
            values = np.array(values, dtype=np.float32)
            
            # Get prediction
            result = self.predict_single(
                values,
                row['equipment_id'],
                row['check_item_id']
            )
            
            # Add true labels if available
            for h in FORECAST_HORIZONS:
                label_col = f'label_{h}d'
                if label_col in row:
                    result[f'true_label_{h}d'] = int(row[label_col]) if row[label_col] != -1 else -1
            
            predictions.append(result)
        
        return predictions
    
    def generate_alerts(self, predictions):
        """Generate alerts from predictions"""
        alerts = []
        
        for pred in predictions:
            if pred['alert_level'] != 'NORMAL':
                for h in FORECAST_HORIZONS:
                    horizon_key = f'{h}d'
                    if pred['binary_predictions'][horizon_key] == 1:
                        alert = {
                            'equipment_id': pred['equipment_id'],
                            'check_item_id': pred['check_item_id'],
                            'horizon': f'{h}d',
                            'raw_probability': pred['predictions'][horizon_key],
                            'calibrated_probability': pred['calibrated_predictions'][horizon_key],
                            'threshold': self.optimal_thresholds.get(h, 0.5),
                            'alert_level': pred['alert_level'],
                            'timestamp': datetime.now().isoformat()
                        }
                        alerts.append(alert)
        
        return alerts


def main():
    print("=" * 60)
    print("🔮 Calibrated Pump Range Deviation Forecast - Inference")
    print("=" * 60)
    
    # Paths
    model_path = MODEL_ROOT / "granite_pump_lora" / "best_model"
    calibration_dir = MODEL_ROOT / "granite_pump_lora" / "calibration"
    test_data_path = PROCESSED_DATA_DIR / "training_samples.csv"
    
    # Check files exist
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    if not calibration_dir.exists():
        print(f"❌ Error: Calibration data not found at {calibration_dir}")
        print("Please run calibrate_model.py first!")
        return
    
    # Initialize predictor
    predictor = CalibratedPredictor(model_path, calibration_dir, device='cpu')
    
    # Load test data
    print("\n📂 Loading test data...")
    df = pd.read_csv(test_data_path)
    
    # Create composite label for stratification
    df['any_anomaly'] = ((df['label_30d'] == 1) | (df['label_60d'] == 1) | (df['label_90d'] == 1)).astype(int)
    
    # Use same stratified test split as training
    from sklearn.model_selection import train_test_split
    from config import TRAINING_CONFIG
    
    train_val_df, test_df = train_test_split(
        df,
        test_size=TRAINING_CONFIG['test_ratio'],
        stratify=df['any_anomaly'],
        random_state=42
    )
    
    print(f"✓ Loaded {len(test_df)} test samples")
    anomaly_count = test_df['any_anomaly'].sum()
    print(f"✓ Test anomalies: {anomaly_count} ({anomaly_count/len(test_df)*100:.1f}%)")
    
    # Run batch inference
    print(f"\n🔮 Running calibrated batch inference on {len(test_df)} samples...")
    predictions = predictor.predict_batch(test_df)
    print("✓ Batch inference complete")
    
    # Generate alerts
    print("\n🚨 Generating alerts...")
    alerts = predictor.generate_alerts(predictions)
    
    if len(alerts) == 0:
        print("✓ No alerts generated (all normal)")
    else:
        print(f"⚠️  Generated {len(alerts)} alerts")
        for alert in alerts[:10]:  # Show first 10
            print(f"  - Equipment {alert['equipment_id']}, "
                  f"Item {alert['check_item_id']}, "
                  f"{alert['horizon']}: "
                  f"{alert['calibrated_probability']:.3f} "
                  f"[{alert['alert_level']}]")
        if len(alerts) > 10:
            print(f"  ... and {len(alerts) - 10} more")
    
    # Save results
    print("\n💾 Saving inference results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions
    results_df = pd.DataFrame([
        {
            'equipment_id': pred['equipment_id'],
            'check_item_id': pred['check_item_id'],
            **{f'raw_prob_{h}d': pred['predictions'][f'{h}d'] for h in FORECAST_HORIZONS},
            **{f'calibrated_prob_{h}d': pred['calibrated_predictions'][f'{h}d'] for h in FORECAST_HORIZONS},
            **{f'binary_pred_{h}d': pred['binary_predictions'][f'{h}d'] for h in FORECAST_HORIZONS},
            **{f'true_label_{h}d': pred.get(f'true_label_{h}d', -1) for h in FORECAST_HORIZONS},
            'alert_level': pred['alert_level'],
            'max_probability': pred['max_probability']
        }
        for pred in predictions
    ])
    
    results_path = RESULTS_ROOT / f"calibrated_inference_results_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"✓ Results saved: {results_path}")
    
    # Save alerts
    if len(alerts) > 0:
        alerts_df = pd.DataFrame(alerts)
        alerts_path = RESULTS_ROOT / f"calibrated_alerts_{timestamp}.csv"
        alerts_df.to_csv(alerts_path, index=False)
        print(f"✓ Alerts saved: {alerts_path}")
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'num_samples': len(test_df),
        'num_alerts': len(alerts),
        'alert_breakdown': {
            level: sum(1 for a in alerts if a['alert_level'] == level)
            for level in ['CRITICAL', 'WARNING', 'CAUTION']
        },
        'optimal_thresholds': predictor.optimal_thresholds
    }
    
    summary_path = RESULTS_ROOT / f"calibrated_inference_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved: {summary_path}")
    
    print("\n" + "=" * 60)
    print("✅ Calibrated Inference Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
