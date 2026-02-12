"""
Range Definition and Label Generation
正常レンジ定義とラベル生成スクリプト

機能:
1. 統計ベースの正常レンジ定義（5th-95th percentile）
2. 現在ラベル生成: y_t^(s)
3. 将来ラベル生成: y_{t,h}^(s) for h ∈ {30, 60, 90}
4. ラベル付きデータセット構築
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from config import (
    PROCESSED_DATA_DIR,
    RANGES_DATA_DIR,
    LOWER_PERCENTILE,
    UPPER_PERCENTILE,
    NORMAL_PERIOD_START,
    NORMAL_PERIOD_END,
    FORECAST_HORIZONS,
    LOOKBACK_DAYS,
    RANDOM_SEED,
    create_directories
)


class RangeDefinitionLabeler:
    """正常レンジ定義とラベル生成クラス"""
    
    def __init__(self):
        """初期化"""
        self.time_series_df = None
        self.range_definitions = {}
        self.labeled_dataset = None
        
    def load_processed_data(self) -> pd.DataFrame:
        """
        処理済みデータの読み込み
        
        Returns:
            処理済みDataFrame
        """
        print("📂 Loading processed time series data...")
        
        input_path = PROCESSED_DATA_DIR / "processed_time_series.csv"
        
        if not input_path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {input_path}\n"
                "Please run data_preprocessing.py first."
            )
        
        df = pd.read_csv(input_path)
        
        # 日付型に変換
        if 'index' in df.columns:
            df['date'] = pd.to_datetime(df['index'])
        
        print(f"✓ Loaded {len(df):,} rows")
        print(f"  Equipment IDs: {df['equipment_id'].nunique()}")
        print(f"  Check Items: {df['check_item_id'].nunique()}")
        
        self.time_series_df = df
        return df
    
    def define_ranges_statistical(
        self,
        df: pd.DataFrame,
        use_normal_period: bool = True
    ) -> Dict[Tuple[int, str], Dict[str, float]]:
        """
        統計ベースの正常レンジ定義
        
        Args:
            df: 時系列DataFrame
            use_normal_period: 正常期間のみ使用するか
            
        Returns:
            レンジ定義の辞書 {(equipment_id, check_item): {lower, upper, mean, std}}
        """
        print("\n📏 Defining normal ranges (statistical method)...")
        print(f"  Lower Percentile: {LOWER_PERCENTILE}th")
        print(f"  Upper Percentile: {UPPER_PERCENTILE}th")
        
        range_defs = {}
        
        # 正常期間フィルタ
        if use_normal_period and NORMAL_PERIOD_START and NORMAL_PERIOD_END:
            print(f"  Using normal period: {NORMAL_PERIOD_START} to {NORMAL_PERIOD_END}")
            df_filtered = df[
                (df['date'] >= NORMAL_PERIOD_START) &
                (df['date'] <= NORMAL_PERIOD_END)
            ]
        else:
            df_filtered = df.copy()
        
        # 設備×測定項目でグループ化
        grouped = df_filtered.groupby(['equipment_id', 'check_item_id'])
        
        for (eq_id, check_item_id), group in grouped:
            values = group['value'].dropna()
            
            if len(values) < 10:  # 最低10データポイント必要
                continue
            
            # 分位点計算
            lower_bound = np.percentile(values, LOWER_PERCENTILE)
            upper_bound = np.percentile(values, UPPER_PERCENTILE)
            
            # レンジ幅が0または小さすぎる場合は標準偏差ベースで拡張
            range_width = upper_bound - lower_bound
            std_val = np.std(values)
            mean_val = np.mean(values)
            
            if range_width < 1e-6 or range_width < 0.1 * std_val:
                # IQR（四分位範囲）ベースの外れ値検出に切り替え
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                
                if iqr > 1e-6:
                    # IQR x 1.5の範囲（Tukey's fences）
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                else:
                    # 標準偏差ベース（±2σ）
                    lower_bound = mean_val - 2 * std_val
                    upper_bound = mean_val + 2 * std_val
                
                print(f"  ⚠️ Equipment {eq_id}: Range width too small, using IQR/std method")
                print(f"     Original: [{np.percentile(values, LOWER_PERCENTILE):.4f}, {np.percentile(values, UPPER_PERCENTILE):.4f}]")
                print(f"     Adjusted: [{lower_bound:.4f}, {upper_bound:.4f}]")
            
            # 統計量
            range_defs[(eq_id, check_item_id)] = {
                'lower': float(lower_bound),
                'upper': float(upper_bound),
                'mean': float(mean_val),
                'std': float(std_val),
                'n_samples': len(values)
            }
        
        print(f"✓ Defined ranges for {len(range_defs)} time series")
        
        self.range_definitions = range_defs
        return range_defs
    
    def generate_current_labels(
        self,
        df: pd.DataFrame,
        range_defs: Dict
    ) -> pd.DataFrame:
        """
        現在ラベル生成: y_t^(s)
        
        Args:
            df: 時系列DataFrame
            range_defs: レンジ定義
            
        Returns:
            現在ラベル付きDataFrame
        """
        print("\n🏷️  Generating current labels (y_t)...")
        
        df = df.copy()
        df['label_current'] = 0  # デフォルトは正常
        
        # 設備×測定項目ごとにラベル付け
        for (eq_id, check_item_id), range_def in range_defs.items():
            lower = range_def['lower']
            upper = range_def['upper']
            
            # 該当する行を抽出
            mask = (df['equipment_id'] == eq_id) & (df['check_item_id'] == check_item_id)
            
            # レンジ外を異常(1)とラベル
            df.loc[mask & (df['value'] < lower), 'label_current'] = 1
            df.loc[mask & (df['value'] > upper), 'label_current'] = 1
        
        # 統計情報
        n_normal = (df['label_current'] == 0).sum()
        n_anomalous = (df['label_current'] == 1).sum()
        anomaly_rate = n_anomalous / len(df) * 100
        
        print(f"✓ Current labels generated:")
        print(f"  Normal: {n_normal:,} ({100-anomaly_rate:.2f}%)")
        print(f"  Anomalous: {n_anomalous:,} ({anomaly_rate:.2f}%)")
        
        return df
    
    def generate_future_labels(
        self,
        df: pd.DataFrame,
        range_defs: Dict,
        horizons: List[int] = FORECAST_HORIZONS
    ) -> pd.DataFrame:
        """
        将来ラベル生成: y_{t,h}^(s) for h ∈ {30, 60, 90}
        
        Args:
            df: 時系列DataFrame
            range_defs: レンジ定義
            horizons: 予測ホライズンのリスト
            
        Returns:
            将来ラベル付きDataFrame
        """
        print(f"\n🔮 Generating future labels for horizons: {horizons} days...")
        
        df = df.copy()
        
        # 設備×測定項目ごとに処理
        grouped = df.groupby(['equipment_id', 'check_item_id'])
        
        all_data = []
        
        for (eq_id, check_item_id), group in grouped:
            # レンジ定義取得
            if (eq_id, check_item_id) not in range_defs:
                continue
            
            range_def = range_defs[(eq_id, check_item_id)]
            lower = range_def['lower']
            upper = range_def['upper']
            
            group = group.sort_values('date').reset_index(drop=True)
            
            # 各ホライズンのラベル生成
            for h in horizons:
                label_col = f'label_future_{h}d'
                
                # h日後の値を参照
                group[f'value_future_{h}d'] = group['value'].shift(-h)
                
                # レンジ外判定
                future_values = group[f'value_future_{h}d']
                group[label_col] = 0  # デフォルト正常
                group.loc[(future_values < lower) | (future_values > upper), label_col] = 1
                group.loc[future_values.isnull(), label_col] = -1  # 未来データなし
            
            all_data.append(group)
        
        df_labeled = pd.concat(all_data, ignore_index=True)
        
        # 統計情報
        print(f"✓ Future labels generated:")
        for h in horizons:
            label_col = f'label_future_{h}d'
            valid_mask = df_labeled[label_col] != -1
            n_valid = valid_mask.sum()
            n_anomalous = (df_labeled.loc[valid_mask, label_col] == 1).sum()
            anomaly_rate = n_anomalous / n_valid * 100 if n_valid > 0 else 0
            print(f"  {h}d horizon: {n_anomalous:,}/{n_valid:,} anomalous ({anomaly_rate:.2f}%)")
        
        self.labeled_dataset = df_labeled
        return df_labeled
    
    def create_training_samples(
        self,
        df: pd.DataFrame,
        lookback: int = LOOKBACK_DAYS,
        horizons: List[int] = FORECAST_HORIZONS
    ) -> pd.DataFrame:
        """
        学習用サンプル作成（スライディングウィンドウ）
        
        Args:
            df: ラベル付きDataFrame
            lookback: 過去参照長
            horizons: 予測ホライズン
            
        Returns:
            学習サンプルDataFrame
        """
        print(f"\n🔄 Creating training samples (lookback={lookback} days)...")
        
        samples = []
        
        # 設備×測定項目ごとに処理
        grouped = df.groupby(['equipment_id', 'check_item_id'])
        
        for (eq_id, check_item_id), group in grouped:
            group = group.sort_values('date').reset_index(drop=True)
            
            # ウィンドウをスライドさせてサンプル作成
            for i in range(lookback, len(group)):
                # 過去データウィンドウ
                window_data = group.iloc[i-lookback:i]
                
                # 現在時点のデータ
                current_data = group.iloc[i]
                
                # 将来ラベルをチェック
                valid_sample = True
                future_labels = {}
                for h in horizons:
                    label_col = f'label_future_{h}d'
                    if label_col in current_data and current_data[label_col] != -1:
                        future_labels[f'label_{h}d'] = int(current_data[label_col])
                    else:
                        valid_sample = False
                        break
                
                if not valid_sample:
                    continue
                
                # サンプル作成
                sample = {
                    'equipment_id': eq_id,
                    'check_item_id': check_item_id,
                    'date': current_data['date'],
                    'window_start': window_data['date'].iloc[0],
                    'window_end': window_data['date'].iloc[-1],
                    'values_sequence': window_data['value_normalized'].tolist(),
                    'label_current': int(current_data['label_current']),
                    **future_labels
                }
                
                samples.append(sample)
        
        samples_df = pd.DataFrame(samples)
        
        print(f"✓ Created {len(samples_df):,} training samples")
        
        return samples_df
    
    def save_ranges_and_labels(
        self,
        range_defs: Dict,
        labeled_df: pd.DataFrame,
        samples_df: pd.DataFrame
    ):
        """
        レンジ定義とラベル付きデータの保存
        
        Args:
            range_defs: レンジ定義
            labeled_df: ラベル付きDataFrame
            samples_df: 学習サンプルDataFrame
        """
        print("\n💾 Saving ranges and labeled data...")
        
        # ディレクトリ作成
        RANGES_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. レンジ定義をJSON保存
        range_defs_serializable = {
            f"{eq_id}_{check_item_id}": range_def
            for (eq_id, check_item_id), range_def in range_defs.items()
        }
        
        range_path = RANGES_DATA_DIR / "range_definitions.json"
        with open(range_path, 'w', encoding='utf-8') as f:
            json.dump(range_defs_serializable, f, indent=2, ensure_ascii=False)
        print(f"✓ Range definitions saved: {range_path}")
        
        # 2. ラベル付きデータ保存
        labeled_path = PROCESSED_DATA_DIR / "labeled_time_series.csv"
        labeled_df.to_csv(labeled_path, index=False, encoding='utf-8-sig')
        print(f"✓ Labeled time series saved: {labeled_path}")
        
        # 3. 学習サンプル保存
        samples_path = PROCESSED_DATA_DIR / "training_samples.csv"
        samples_df.to_csv(samples_path, index=False, encoding='utf-8-sig')
        print(f"✓ Training samples saved: {samples_path}")
        
        # 4. 統計情報保存
        stats = {
            'num_ranges_defined': len(range_defs),
            'total_labeled_points': len(labeled_df),
            'total_training_samples': len(samples_df),
            'lookback_days': LOOKBACK_DAYS,
            'forecast_horizons': FORECAST_HORIZONS,
            'lower_percentile': LOWER_PERCENTILE,
            'upper_percentile': UPPER_PERCENTILE
        }
        
        stats_path = RANGES_DATA_DIR / "labeling_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Statistics saved: {stats_path}")
    
    def run_pipeline(self):
        """
        レンジ定義・ラベル生成パイプライン実行
        """
        print("="*60)
        print("🚀 Starting Range Definition & Label Generation Pipeline")
        print("="*60)
        
        # ディレクトリ作成
        create_directories()
        
        # 1. データ読み込み
        df = self.load_processed_data()
        
        # 2. 正常レンジ定義
        range_defs = self.define_ranges_statistical(df)
        
        # 3. 現在ラベル生成
        df = self.generate_current_labels(df, range_defs)
        
        # 4. 将来ラベル生成
        labeled_df = self.generate_future_labels(df, range_defs)
        
        # 5. 学習サンプル作成
        samples_df = self.create_training_samples(labeled_df)
        
        # 6. 保存
        self.save_ranges_and_labels(range_defs, labeled_df, samples_df)
        
        print("\n" + "="*60)
        print("✅ Range Definition & Label Generation Complete!")
        print("="*60)
        
        return range_defs, labeled_df, samples_df


def main():
    """メイン実行"""
    # 再現性のためのシード設定
    np.random.seed(RANDOM_SEED)
    
    # パイプライン実行
    labeler = RangeDefinitionLabeler()
    range_defs, labeled_df, samples_df = labeler.run_pipeline()
    
    print(f"\n🎉 Successfully created {len(samples_df):,} training samples!")


if __name__ == "__main__":
    main()
