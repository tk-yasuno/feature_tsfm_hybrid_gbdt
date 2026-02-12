"""
設備スケールアップ: v1.1向けに64設備を選定
全320設備の20%に相当する64設備を、データ品質と変動性に基づいて選定する
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# データソース
DATA_SOURCE = Path(r"C:\Users\yasun\RL\ner-equipment-granite\data_source\251217_チェック項目_実施結果.csv")

print("="*80)
print("🎯 v1.1: 64設備選定 (全320設備の20%)")
print("="*80)

# データ読み込み
print(f"\n📂 Loading data from: {DATA_SOURCE.name}")
df = pd.read_csv(DATA_SOURCE, encoding='cp932')

print(f"   Total records: {len(df):,}")
print(f"   Columns: {list(df.columns)}")

# カラム名確認
print(f"\n   First 5 rows:")
print(df.head())

# 設備ID列を特定
equipment_col = None
for col in df.columns:
    if '設備' in col and ('id' in col.lower() or 'ID' in col):
        equipment_col = col
        break

if equipment_col is None:
    print("   ❌ 設備IDカラムが見つかりません")
    print(f"   Available columns: {list(df.columns)}")
    exit(1)

print(f"\n✓ Equipment ID column: '{equipment_col}'")

# 値カラムを特定（実施結果の値）
value_col = None
for col in df.columns:
    if '実施結果' in col and '値' in col:
        value_col = col
        break

if value_col is None:
    # フォールバック: '最新の値'や'値'など
    for col in df.columns:
        if '最新' in col and '値' in col:
            value_col = col
            break

if value_col is None:
    print("   ❌ 測定値カラムが見つかりません")
    print(f"   Available columns: {list(df.columns)}")
    exit(1)

print(f"✓ Value column: '{value_col}'")

# チェック項目カラムを特定
check_item_col = None
for col in df.columns:
    if 'チェック項目' in col and ('id' in col.lower() or 'ID' in col):
        check_item_col = col
        break

# 日時カラムを特定
datetime_col = None
for col in df.columns:
    if '日時' in col or '日付' in col:
        datetime_col = col
        break

print(f"✓ Check item column: '{check_item_col}'")
print(f"✓ Datetime column: '{datetime_col}'")

# 空調設備フィルタ（共通分類コード == 3）
category_col = None
for col in df.columns:
    if '共通分類' in col or '分類コード' in col:
        category_col = col
        break

if category_col:
    print(f"\n🔍 Filtering by equipment category...")
    print(f"   Category column: '{category_col}'")
    print(f"   Unique categories: {df[category_col].unique()}")
    
    # 空調設備のみに絞る
    df_hvac = df[df[category_col] == 3].copy()
    print(f"   HVAC equipment records: {len(df_hvac):,} / {len(df):,}")
else:
    print(f"\n⚠️  Category column not found, using all equipment")
    df_hvac = df.copy()

# 設備ごとに分析
print(f"\n📊 Analyzing equipment...")
equipment_stats = []

for eq_id in df_hvac[equipment_col].unique():
    eq_data = df_hvac[df_hvac[equipment_col] == eq_id]
    
    # 測定値を数値化
    values = pd.to_numeric(eq_data[value_col], errors='coerce').dropna()
    
    if len(values) < 180:  # 最低180日分のデータ
        continue
    
    # 日時データがあれば期間を計算
    date_range_days = None
    if datetime_col:
        try:
            dates = pd.to_datetime(eq_data[datetime_col], errors='coerce').dropna()
            if len(dates) > 0:
                date_range_days = (dates.max() - dates.min()).days
        except:
            pass
    
    # 統計計算
    stats = {
        'equipment_id': eq_id,
        'data_points': len(values),
        'unique_values': values.nunique(),
        'mean': values.mean(),
        'std': values.std(),
        'cv': values.std() / values.mean() if values.mean() != 0 else 0,  # 変動係数
        'min': values.min(),
        'max': values.max(),
        'range': values.max() - values.min(),
        'date_range_days': date_range_days,
        'check_items_count': eq_data[check_item_col].nunique() if check_item_col else 1
    }
    
    # 変動性スコア（データ品質を考慮）
    # スコア = 変動係数 × データポイント数の対数 × データ期間の対数
    score = stats['cv'] * np.log1p(stats['data_points'])
    if date_range_days:
        score *= np.log1p(date_range_days)
    stats['quality_score'] = score
    
    equipment_stats.append(stats)

# DataFrame化
stats_df = pd.DataFrame(equipment_stats)

print(f"\n✓ Total equipment analyzed: {len(stats_df)}")
print(f"   Equipment with sufficient data (≥180 points): {len(stats_df)}")

# 変動がある設備のみ（std > 0）
stats_df = stats_df[stats_df['std'] > 0].copy()
print(f"   Equipment with variation (std > 0): {len(stats_df)}")

# 品質スコアでソート
stats_df = stats_df.sort_values('quality_score', ascending=False)

# TOP 64設備を選定
top_64 = stats_df.head(64)

print(f"\n" + "="*80)
print(f"✅ Selected TOP 64 Equipment (20% of 320)")
print("="*80)

print(f"\n📈 Statistics of selected equipment:")
print(f"   Data points:   {top_64['data_points'].min():.0f} - {top_64['data_points'].max():.0f} (median: {top_64['data_points'].median():.0f})")
print(f"   Std deviation: {top_64['std'].min():.4f} - {top_64['std'].max():.4f} (median: {top_64['std'].median():.4f})")
print(f"   CV (coeff):    {top_64['cv'].min():.4f} - {top_64['cv'].max():.4f} (median: {top_64['cv'].median():.4f})")
if top_64['date_range_days'].notna().any():
    print(f"   Date range:    {top_64['date_range_days'].min():.0f} - {top_64['date_range_days'].max():.0f} days (median: {top_64['date_range_days'].median():.0f})")

print(f"\n🏆 TOP 10 Equipment by Quality Score:")
print(top_64.head(10)[['equipment_id', 'data_points', 'std', 'cv', 'quality_score']].to_string(index=False))

# v1.0の5設備が含まれているか確認
v1_equipment = [265706, 265707, 265708, 265709, 265710]
overlap = set(v1_equipment) & set(top_64['equipment_id'].values)
print(f"\n🔍 v1.0 equipment overlap: {len(overlap)}/5")
if overlap:
    print(f"   Included from v1.0: {sorted(overlap)}")
missing = set(v1_equipment) - overlap
if missing:
    print(f"   Missing from v1.0: {sorted(missing)}")

# 結果保存
output_path = Path("data/processed/selected_64_equipment.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

equipment_list = top_64['equipment_id'].tolist()

result = {
    "version": "v1.1",
    "selection_date": "2026-02-12",
    "total_equipment": len(stats_df),
    "selected_equipment": 64,
    "selection_criteria": {
        "min_data_points": 180,
        "std_threshold": "std > 0",
        "ranking_method": "quality_score = cv × log(data_points) × log(date_range_days)"
    },
    "equipment_ids": equipment_list,
    "statistics": {
        "data_points": {
            "min": int(top_64['data_points'].min()),
            "max": int(top_64['data_points'].max()),
            "median": int(top_64['data_points'].median())
        },
        "std_deviation": {
            "min": float(top_64['std'].min()),
            "max": float(top_64['std'].max()),
            "median": float(top_64['std'].median())
        },
        "cv": {
            "min": float(top_64['cv'].min()),
            "max": float(top_64['cv'].max()),
            "median": float(top_64['cv'].median())
        }
    }
}

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n💾 Saved to: {output_path}")

# CSV形式でも保存
csv_path = Path("data/processed/selected_64_equipment.csv")
top_64.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"💾 Details saved to: {csv_path}")

print(f"\n" + "="*80)
print(f"✅ Equipment selection complete!")
print(f"   Next step: Update config.py with TARGET_EQUIPMENT_IDS")
print("="*80)

# config.py更新用のコードを出力
print(f"\n📝 Copy this to config.py:")
print(f"\nTARGET_EQUIPMENT_IDS = {equipment_list}")
