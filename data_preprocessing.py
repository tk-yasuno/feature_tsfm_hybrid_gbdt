"""
Data Preprocessing for Pump Range Deviation Forecast
ポンプ設備データの前処理スクリプト

機能:
1. CSVデータの読み込み（エンコーディング処理）
2. 日時パース処理
3. 欠損値補間（前値保持）
4. 日次集計
5. 設備×測定項目ごとの時系列データ構築
6. 正規化（z-score）
7. 処理済みデータの保存
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import (
    SOURCE_CSV_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    CSV_ENCODING,
    COLUMNS,
    MIN_DATA_POINTS,
    AGGREGATION_METHOD,
    TARGET_EQUIPMENT_IDS,
    RANDOM_SEED,
    create_directories
)


class PumpDataPreprocessor:
    """ポンプ設備データの前処理クラス"""
    
    def __init__(self):
        """初期化"""
        self.raw_df = None
        self.processed_df = None
        self.time_series_dict = {}
        self.metadata_dict = {}
        
    def load_raw_data(self, csv_path: Path = SOURCE_CSV_PATH) -> pd.DataFrame:
        """
        生データの読み込み
        
        Args:
            csv_path: CSVファイルのパス
            
        Returns:
            読み込んだDataFrame
        """
        print(f"📂 Loading data from: {csv_path}")
        
        try:
            # エンコーディング指定で読み込み
            df = pd.read_csv(csv_path, encoding=CSV_ENCODING)
            print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # カラム名の表示
            print(f"Columns: {list(df.columns)}")
            
            self.raw_df = df
            return df
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise
    
    def parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        日時パース処理
        
        Args:
            df: 入力DataFrame
            
        Returns:
            日時パース済みDataFrame
        """
        print("\n🕐 Parsing datetime columns...")
        
        # 実際のCSVカラム名: "実施日時" (例: "2024/3/1 5:04")
        datetime_col = COLUMNS["datetime"]
        
        if datetime_col not in df.columns:
            print(f"❌ Column '{datetime_col}' not found. Available columns:")
            for col in df.columns:
                print(f"    - {col}")
            raise ValueError(f"Required column '{datetime_col}' not found")
        
        # 日時パース
        df['datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')
        
        # 日付のみ抽出
        df['date'] = df['datetime'].dt.date
        
        # パースできなかった行数をカウント
        null_count = df['datetime'].isnull().sum()
        if null_count > 0:
            print(f"⚠ Warning: {null_count} rows have invalid datetime")
        
        print(f"✓ Datetime parsed. Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        日次集計処理
        
        Args:
            df: 入力DataFrame
            
        Returns:
            日次集計済みDataFrame
        """
        print(f"\n📊 Aggregating to daily data (method: {AGGREGATION_METHOD})...")
        
        # 必要なカラム名
        equipment_col = COLUMNS["equipment_id"]
        check_item_col = COLUMNS["check_item_id"]
        value_col = COLUMNS["value"]  # 実施結果の値
        
        # 数値変換
        df['value'] = pd.to_numeric(df[value_col], errors='coerce')
        
        # 集計
        if AGGREGATION_METHOD == "mean":
            agg_func = 'mean'
        elif AGGREGATION_METHOD == "median":
            agg_func = 'median'
        elif AGGREGATION_METHOD == "last":
            agg_func = 'last'
        else:
            agg_func = 'mean'
        
        # グループ化して集計
        daily_df = df.groupby([
            equipment_col,
            check_item_col,
            'date'
        ]).agg({
            'value': agg_func
        }).reset_index()
        
        # カラム名変更
        daily_df.columns = ['equipment_id', 'check_item_id', 'date', 'value']
        
        print(f"✓ Daily aggregation complete: {len(daily_df):,} rows")
        
        return daily_df
    
    def create_time_series(self, daily_df: pd.DataFrame) -> dict:
        """
        設備×測定項目ごとの時系列データ構築
        
        Args:
            daily_df: 日次集計済みDataFrame
            
        Returns:
            時系列データの辞書 {(equipment_id, check_item): DataFrame}
        """
        print("\n🔄 Creating time series per equipment × check_item...")
        
        # 設備フィルタ適用
        if TARGET_EQUIPMENT_IDS is not None:
            before_count = len(daily_df)
            daily_df = daily_df[daily_df['equipment_id'].isin(TARGET_EQUIPMENT_IDS)].copy()
            after_count = len(daily_df)
            print(f"  ✓ Equipment filter applied: {TARGET_EQUIPMENT_IDS}")
            print(f"    {before_count} → {after_count} rows ({len(TARGET_EQUIPMENT_IDS)} equipment)")
        
        time_series_dict = {}
        
        # 設備×測定項目でグループ化
        grouped = daily_df.groupby(['equipment_id', 'check_item_id'])
        
        for (eq_id, check_item_id), group in grouped:
            # 日付でソート
            ts_df = group.sort_values('date').copy()
            
            # 最小データポイント数フィルタ
            if len(ts_df) < MIN_DATA_POINTS:
                continue
            
            # 日付を連続化（欠損日を補完）
            date_range = pd.date_range(
                start=ts_df['date'].min(),
                end=ts_df['date'].max(),
                freq='D'
            )
            
            # リインデックス
            ts_df = ts_df.set_index('date')
            ts_df = ts_df.reindex(date_range.date)
            
            # 欠損値を前値保持で補間
            ts_df['value'] = ts_df['value'].ffill()
            
            # まだ欠損がある場合は後値保持
            ts_df['value'] = ts_df['value'].bfill()
            
            # キーとして保存
            key = (eq_id, check_item_id)
            time_series_dict[key] = ts_df.reset_index()
        
        print(f"✓ Created {len(time_series_dict)} time series")
        
        return time_series_dict
    
    def normalize_time_series(self, time_series_dict: dict) -> dict:
        """
        時系列データの正規化（z-score）
        
        Args:
            time_series_dict: 時系列データの辞書
            
        Returns:
            正規化済み時系列データの辞書
        """
        print("\n📏 Normalizing time series (z-score)...")
        
        normalized_dict = {}
        
        for key, ts_df in time_series_dict.items():
            # z-score正規化
            values = ts_df['value'].values
            mean = np.mean(values)
            std = np.std(values)
            
            # 標準偏差が0の場合は正規化しない
            if std > 0:
                normalized_values = (values - mean) / std
            else:
                normalized_values = values - mean
            
            # 正規化データを追加
            ts_df_norm = ts_df.copy()
            ts_df_norm['value_normalized'] = normalized_values
            ts_df_norm['value_mean'] = mean
            ts_df_norm['value_std'] = std
            
            normalized_dict[key] = ts_df_norm
        
        print(f"✓ Normalized {len(normalized_dict)} time series")
        
        return normalized_dict
    
    def save_processed_data(self, time_series_dict: dict):
        """
        処理済みデータの保存
        
        Args:
            time_series_dict: 時系列データの辞書
        """
        print("\n💾 Saving processed data...")
        
        # ディレクトリ作成
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # 全データを1つのDataFrameに統合
        all_data = []
        for (eq_id, check_item_id), ts_df in time_series_dict.items():
            ts_df = ts_df.copy()
            ts_df['equipment_id'] = eq_id
            ts_df['check_item_id'] = check_item_id
            all_data.append(ts_df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # CSV保存
        output_path = PROCESSED_DATA_DIR / "processed_time_series.csv"
        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ Saved to: {output_path}")
        
        # 統計情報保存
        stats = {
            'num_equipment': len(set([k[0] for k in time_series_dict.keys()])),
            'num_check_items': len(set([k[1] for k in time_series_dict.keys()])),
            'num_time_series': len(time_series_dict),
            'total_data_points': len(combined_df),
            'date_range': f"{combined_df['date'].min()} to {combined_df['date'].max()}" if 'date' in combined_df.columns else "N/A"
        }
        
        stats_path = PROCESSED_DATA_DIR / "processing_stats.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✓ Statistics saved to: {stats_path}")
        print("\n📊 Processing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    def run_pipeline(self):
        """
        前処理パイプライン実行
        """
        print("="*60)
        print("🚀 Starting Data Preprocessing Pipeline")
        print("="*60)
        
        # ディレクトリ作成
        create_directories()
        
        # 1. データ読み込み
        df = self.load_raw_data()
        
        # 2. 日時パース
        df = self.parse_datetime(df)
        
        # 3. 日次集計
        daily_df = self.aggregate_daily(df)
        
        # 4. 時系列データ構築
        self.time_series_dict = self.create_time_series(daily_df)
        
        # 5. 正規化
        self.time_series_dict = self.normalize_time_series(self.time_series_dict)
        
        # 6. 保存
        self.save_processed_data(self.time_series_dict)
        
        print("\n" + "="*60)
        print("✅ Data Preprocessing Complete!")
        print("="*60)
        
        return self.time_series_dict


def main():
    """メイン実行"""
    # 再現性のためのシード設定
    np.random.seed(RANDOM_SEED)
    
    # 前処理実行
    preprocessor = PumpDataPreprocessor()
    time_series_dict = preprocessor.run_pipeline()
    
    print(f"\n🎉 Processed {len(time_series_dict)} time series successfully!")


if __name__ == "__main__":
    main()
