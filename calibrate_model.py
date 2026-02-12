"""
Probability Calibration and Threshold Optimization
確率キャリブレーションと閾値最適化
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    precision_recall_curve, 
    f1_score, 
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    PROJECT_ROOT, MODEL_ROOT, RESULTS_ROOT, PROCESSED_DATA_DIR,
    FORECAST_HORIZONS, LOOKBACK_DAYS, TRAINING_CONFIG
)
from granite_ts_model import GraniteTimeSeriesClassifier


class CalibrationDataset(Dataset):
    """Dataset for calibration"""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Parse values_sequence
        values = eval(row['values_sequence'])
        values = np.array(values, dtype=np.float32)
        
        # Pad or truncate
        if len(values) < LOOKBACK_DAYS:
            values = np.pad(values, (LOOKBACK_DAYS - len(values), 0), mode='edge')
        elif len(values) > LOOKBACK_DAYS:
            values = values[-LOOKBACK_DAYS:]
        
        sequence = torch.FloatTensor(values).unsqueeze(-1)  # [seq_len, 1]
        
        # Labels
        labels = {}
        for h in FORECAST_HORIZONS:
            col = f'label_{h}d'
            if col in row and row[col] != -1:
                labels[f'{h}d'] = float(row[col])
            else:
                labels[f'{h}d'] = 0.0  # Default to normal
        
        return sequence, labels


class ProbabilityCalibrator:
    """Probability calibration using Platt Scaling or Isotonic Regression"""
    
    def __init__(self, model_path, device='cpu'):
        """
        Args:
            model_path: Path to trained model
            device: Device to use (cpu/cuda)
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.calibrators = {}  # {horizon: calibrator}
        self.optimal_thresholds = {}  # {horizon: threshold}
        
    def _load_model(self, model_path):
        """Load trained model"""
        print(f"📂 Loading model from: {model_path}")
        model = GraniteTimeSeriesClassifier(device=self.device)
        model.load_model(model_path)
        model.eval()
        return model
    
    def _get_predictions(self, dataloader):
        """Get raw predictions from model"""
        all_predictions = {h: [] for h in FORECAST_HORIZONS}
        all_labels = {h: [] for h in FORECAST_HORIZONS}
        
        with torch.no_grad():
            for sequences, labels in tqdm(dataloader, desc="Getting predictions"):
                sequences = sequences.to(self.device)
                # Use forward pass directly
                predictions = self.model(sequences)
                
                for h in FORECAST_HORIZONS:
                    prob_key = f'prob_{h}d'
                    all_predictions[h].extend(predictions[prob_key].cpu().numpy())
                    all_labels[h].extend([labels[f'{h}d'][i].item() 
                                         for i in range(len(labels[f'{h}d']))])
        
        return all_predictions, all_labels
    
    def calibrate_platt_scaling(self, val_loader, method='sigmoid'):
        """
        Calibrate using Platt Scaling (logistic regression)
        
        Args:
            val_loader: Validation data loader
            method: 'sigmoid' for Platt Scaling, 'isotonic' for Isotonic Regression
        """
        print(f"\n🔧 Calibrating with method: {method}")
        
        # Get predictions on validation set
        predictions, labels = self._get_predictions(val_loader)
        
        # Train calibrator for each horizon
        for h in FORECAST_HORIZONS:
            print(f"\n  Calibrating {h}d horizon...")
            
            y_true = np.array(labels[h])
            y_pred = np.array(predictions[h])
            
            # Check if we have both classes
            if len(np.unique(y_true)) < 2:
                print(f"    ⚠️  Warning: Only one class in validation set for {h}d")
                continue
            
            # Fit calibrator
            if method == 'sigmoid':
                # Platt Scaling
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression()
                calibrator.fit(y_pred.reshape(-1, 1), y_true)
            elif method == 'isotonic':
                # Isotonic Regression
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(y_pred, y_true)
            
            self.calibrators[h] = calibrator
            
            # Calculate calibrated predictions
            if method == 'sigmoid':
                y_calibrated = calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
            else:
                y_calibrated = calibrator.predict(y_pred)
            
            # Show improvement
            print(f"    Original probability range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
            print(f"    Calibrated probability range: [{y_calibrated.min():.4f}, {y_calibrated.max():.4f}]")
            print(f"    ✓ Calibrator trained")
    
    def find_optimal_thresholds(self, val_loader, metric='f1'):
        """
        Find optimal threshold for each horizon
        
        Args:
            val_loader: Validation data loader
            metric: Optimization metric ('f1', 'balanced_accuracy', 'youden')
        """
        print(f"\n🎯 Finding optimal thresholds (optimizing {metric})...")
        
        # Get predictions
        predictions, labels = self._get_predictions(val_loader)
        
        for h in FORECAST_HORIZONS:
            print(f"\n  Optimizing threshold for {h}d horizon...")
            
            y_true = np.array(labels[h])
            y_pred = np.array(predictions[h])
            
            # Apply calibration if available
            if h in self.calibrators:
                calibrator = self.calibrators[h]
                if hasattr(calibrator, 'predict_proba'):
                    y_pred = calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
                else:
                    y_pred = calibrator.predict(y_pred)
            
            # Skip if only one class
            if len(np.unique(y_true)) < 2:
                print(f"    ⚠️  Skipping: only one class present")
                self.optimal_thresholds[h] = 0.5
                continue
            
            # Calculate metric for different thresholds
            thresholds = np.linspace(0, 1, 101)
            scores = []
            
            for threshold in thresholds:
                y_binary = (y_pred >= threshold).astype(int)
                
                if metric == 'f1':
                    score = f1_score(y_true, y_binary, zero_division=0)
                elif metric == 'balanced_accuracy':
                    score = accuracy_score(y_true, y_binary)
                elif metric == 'youden':
                    # Youden's J statistic = Sensitivity + Specificity - 1
                    tn = np.sum((y_true == 0) & (y_binary == 0))
                    fp = np.sum((y_true == 0) & (y_binary == 1))
                    fn = np.sum((y_true == 1) & (y_binary == 0))
                    tp = np.sum((y_true == 1) & (y_binary == 1))
                    
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    score = sensitivity + specificity - 1
                
                scores.append(score)
            
            # Find optimal threshold
            optimal_idx = np.argmax(scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_score = scores[optimal_idx]
            
            self.optimal_thresholds[h] = float(optimal_threshold)
            
            print(f"    Optimal threshold: {optimal_threshold:.3f}")
            print(f"    {metric.upper()} score: {optimal_score:.4f}")
            
            # Calculate metrics at optimal threshold
            y_binary = (y_pred >= optimal_threshold).astype(int)
            precision = precision_score(y_true, y_binary, zero_division=0)
            recall = recall_score(y_true, y_binary, zero_division=0)
            f1 = f1_score(y_true, y_binary, zero_division=0)
            
            print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    def plot_calibration_curves(self, val_loader, save_path):
        """Plot calibration curves (reliability diagrams)"""
        print("\n📊 Plotting calibration curves...")
        
        predictions, labels = self._get_predictions(val_loader)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, h in enumerate(FORECAST_HORIZONS):
            ax = axes[idx]
            
            y_true = np.array(labels[h])
            y_pred = np.array(predictions[h])
            
            if len(np.unique(y_true)) < 2:
                ax.text(0.5, 0.5, 'Insufficient data', 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{h}d Horizon')
                continue
            
            # Original predictions
            fraction_of_positives, mean_predicted_value = \
                self._calibration_curve(y_true, y_pred, n_bins=10)
            
            ax.plot(mean_predicted_value, fraction_of_positives, 
                   's-', label='Original', alpha=0.7)
            
            # Calibrated predictions
            if h in self.calibrators:
                calibrator = self.calibrators[h]
                if hasattr(calibrator, 'predict_proba'):
                    y_calibrated = calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
                else:
                    y_calibrated = calibrator.predict(y_pred)
                
                fraction_of_positives_cal, mean_predicted_value_cal = \
                    self._calibration_curve(y_true, y_calibrated, n_bins=10)
                
                ax.plot(mean_predicted_value_cal, fraction_of_positives_cal,
                       'o-', label='Calibrated', alpha=0.7)
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'{h}d Horizon')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Calibration curves saved: {save_path}")
    
    def _calibration_curve(self, y_true, y_prob, n_bins=10):
        """Calculate calibration curve"""
        bins = np.linspace(0, 1, n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        bin_sums = np.bincount(binids, weights=y_prob, minlength=n_bins)
        bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
        bin_total = np.bincount(binids, minlength=n_bins)
        
        nonzero = bin_total != 0
        
        fraction_of_positives = np.zeros(n_bins)
        mean_predicted_value = np.zeros(n_bins)
        
        fraction_of_positives[nonzero] = bin_true[nonzero] / bin_total[nonzero]
        mean_predicted_value[nonzero] = bin_sums[nonzero] / bin_total[nonzero]
        
        return fraction_of_positives[nonzero], mean_predicted_value[nonzero]
    
    def save(self, save_dir):
        """Save calibrators and optimal thresholds"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save calibrators
        calibrator_path = save_dir / 'calibrators.pkl'
        with open(calibrator_path, 'wb') as f:
            pickle.dump(self.calibrators, f)
        print(f"💾 Calibrators saved: {calibrator_path}")
        
        # Save optimal thresholds
        thresholds_path = save_dir / 'optimal_thresholds.json'
        with open(thresholds_path, 'w') as f:
            json.dump(self.optimal_thresholds, f, indent=2)
        print(f"💾 Optimal thresholds saved: {thresholds_path}")
    
    def load(self, load_dir):
        """Load calibrators and optimal thresholds"""
        load_dir = Path(load_dir)
        
        # Load calibrators
        calibrator_path = load_dir / 'calibrators.pkl'
        if calibrator_path.exists():
            with open(calibrator_path, 'rb') as f:
                self.calibrators = pickle.load(f)
            print(f"✓ Calibrators loaded: {calibrator_path}")
        
        # Load optimal thresholds
        thresholds_path = load_dir / 'optimal_thresholds.json'
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                self.optimal_thresholds = json.load(f)
                # Convert string keys to int
                self.optimal_thresholds = {int(k): v for k, v in self.optimal_thresholds.items()}
            print(f"✓ Optimal thresholds loaded: {thresholds_path}")


def main():
    print("=" * 60)
    print("🔧 Probability Calibration & Threshold Optimization")
    print("=" * 60)
    
    # Paths
    model_path = MODEL_ROOT / "granite_pump_lora" / "best_model"
    calibration_dir = MODEL_ROOT / "granite_pump_lora" / "calibration"
    training_samples_path = PROCESSED_DATA_DIR / "training_samples.csv"
    
    # Check files exist
    if not training_samples_path.exists():
        print(f"❌ Error: Training samples not found at {training_samples_path}")
        return
    
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    # Load data
    print("\n📂 Loading training samples...")
    df = pd.read_csv(training_samples_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Create composite label for stratification (any anomaly in any horizon)
    df['any_anomaly'] = ((df['label_30d'] == 1) | (df['label_60d'] == 1) | (df['label_90d'] == 1)).astype(int)
    anomaly_count = df['any_anomaly'].sum()
    print(f"✓ Anomaly samples: {anomaly_count} ({anomaly_count/len(df)*100:.1f}%)")
    
    # Split into train/val with stratification
    from sklearn.model_selection import train_test_split
    train_ratio = TRAINING_CONFIG['train_ratio']
    val_ratio = TRAINING_CONFIG['val_ratio']
    
    # First split: train+val vs test (stratified)
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TRAINING_CONFIG['test_ratio'],
        stratify=df['any_anomaly'],
        random_state=42
    )
    
    # Second split: train vs val (stratified)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio / (train_ratio + val_ratio),
        stratify=train_val_df['any_anomaly'],
        random_state=42
    )
    
    print(f"✓ Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"  Train anomalies: {train_df['any_anomaly'].sum()} ({train_df['any_anomaly'].sum()/len(train_df)*100:.1f}%)")
    print(f"  Val anomalies: {val_df['any_anomaly'].sum()} ({val_df['any_anomaly'].sum()/len(val_df)*100:.1f}%)")
    print(f"  Val 30d: {(val_df['label_30d'] == 1).sum()}, 60d: {(val_df['label_60d'] == 1).sum()}, 90d: {(val_df['label_90d'] == 1).sum()}")
    
    # Create validation dataset
    val_dataset = CalibrationDataset(val_df)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False
    )
    
    # Initialize calibrator
    calibrator = ProbabilityCalibrator(model_path, device='cpu')
    
    # Calibrate using Platt Scaling
    calibrator.calibrate_platt_scaling(val_loader, method='sigmoid')
    
    # Find optimal thresholds
    calibrator.find_optimal_thresholds(val_loader, metric='f1')
    
    # Plot calibration curves
    plot_path = RESULTS_ROOT / 'calibration_curves.png'
    calibrator.plot_calibration_curves(val_loader, plot_path)
    
    # Save calibrators and thresholds
    calibrator.save(calibration_dir)
    
    print("\n" + "=" * 60)
    print("✅ Calibration Complete!")
    print("=" * 60)
    print(f"\nOptimal Thresholds:")
    for h, threshold in calibrator.optimal_thresholds.items():
        print(f"  {h}d horizon: {threshold:.3f}")
    print(f"\n💾 Calibration artifacts saved to: {calibration_dir}")


if __name__ == "__main__":
    main()
