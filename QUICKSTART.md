# Quick Start Guide - Pump Range Deviation Forecast MVP

## 📋 前提条件

- Python 3.8以上
- CUDA対応GPU（推奨、CPUでも動作可）
- 10GB以上の空きディスク容量

## 🚀 セットアップ

### 1. 環境構築

```powershell
# プロジェクトディレクトリに移動
cd PumpRange_Deviation_Forecast

# 仮想環境作成（推奨）
python -m venv venv
.\venv\Scripts\Activate.ps1

# 依存パッケージインストール
pip install -r requirements.txt

# Granite Time Series（オプション）
pip install git+https://github.com/ibm-granite/granite-tsfm.git
```

### 2. ディレクトリ初期化

```powershell
# 設定ファイルで必要なディレクトリを作成
python config.py
```

## 📊 MVPワークフロー

### Step 1: データ前処理

```powershell
python data_preprocessing.py
```

**実行内容:**
- CSVデータ読み込み（エンコーディング処理）
- 日時パース
- 日次集計
- 欠損値補間（前値保持）
- 設備×測定項目ごとの時系列構築
- z-score正規化

**出力:**
- `data/processed/processed_time_series.csv`
- `data/processed/processing_stats.txt`

**所要時間:** 約5-10分（13,000件のデータ）

---

### Step 2: 正常レンジ定義とラベル生成

```powershell
python range_definition.py
```

**実行内容:**
- 統計ベースレンジ設定（5th-95th percentile）
- 現在ラベル生成: y_t^(s)
- 将来ラベル生成: y_{t,h}^(s) for h ∈ {30, 60, 90}
- スライディングウィンドウで学習サンプル作成

**出力:**
- `data/ranges/range_definitions.json`
- `data/processed/labeled_time_series.csv`
- `data/processed/training_samples.csv`

**所要時間:** 約3-5分

---

### Step 3: モデル訓練

```powershell
python train.py
```

**実行内容:**
- Granite TSモデル読み込み（Fallback: LSTM）
- LoRAアダプター適用
- バイナリクロスエントロピー損失で学習
- Early Stopping

**出力:**
- `models/granite_pump_lora/best_model/`
- `results/training_history.json`

**所要時間:** 約30-60分（GPU使用時）

**注意:**
- Granite TSモデルが利用できない場合は、自動的にLSTMフォールバックモデルが使用されます
- 初回実行時はモデルのダウンロードに時間がかかる場合があります

---

### Step 4: 推論

```powershell
python inference.py
```

**実行内容:**
- 訓練済みモデルロード
- テストデータで逸脱確率予測
- アラート判定（WARNING/CRITICAL）

**出力:**
- `results/inference_results_YYYYMMDD_HHMMSS.csv`
- `results/inference_alerts_YYYYMMDD_HHMMSS.csv`
- `results/inference_summary_YYYYMMDD_HHMMSS.json`

**所要時間:** 約3-5分

---

### Step 5: 評価

```powershell
python evaluate.py
```

**実行内容:**
- ROC-AUC, PR-AUC計算
- リードタイム分析
- 混同行列生成
- 可視化

**出力:**
- `results/evaluation_metrics.json`
- `results/roc_curves.png`
- `results/pr_curves.png`
- `results/confusion_matrices.png`

**所要時間:** 約2-3分

---

## 🔧 トラブルシューティング

### Granite TSモデルが見つからない場合

モデルが自動的にLSTMフォールバックモードに切り替わります。MVPテストには十分です。

```
⚠ Warning: Could not load Granite TS model
  Using fallback LSTM model for MVP testing...
```

### エンコーディングエラー

CSVファイルが読み込めない場合は、`config.py`の`CSV_ENCODING`を変更してください。

```python
CSV_ENCODING = "utf-8"  # または "shift-jis", "cp932"
```

### GPUメモリ不足

バッチサイズを減らしてください。

```python
# config.py
TRAINING_CONFIG = {
    "batch_size": 16,  # 32 → 16に変更
    ...
}
```

### データが少なすぎる警告

`MIN_DATA_POINTS`を調整してください。

```python
# config.py
MIN_DATA_POINTS = 90  # 180 → 90に変更
```

---

## 📈 結果の確認

### トレーニング履歴

```powershell
python -c "import json; print(json.dumps(json.load(open('results/training_history.json')), indent=2))"
```

### アラート確認

```powershell
python -c "import pandas as pd; df = pd.read_csv('results/inference_alerts_*.csv'); print(df.head(10))"
```

### 評価メトリクス

```powershell
python -c "import json; print(json.dumps(json.load(open('results/evaluation_metrics.json')), indent=2))"
```

---

## 🎯 次のステップ

1. **ハイパーパラメータチューニング**
   - `config.py`でLORA_CONFIG、TRAINING_CONFIGを調整
   
2. **データ拡張**
   - より長い期間のデータを追加
   - 他の測定項目を追加
   
3. **モデル改善**
   - Granite TSの完全版を導入
   - アンサンブル手法の検討
   
4. **本番展開**
   - リアルタイム推論API構築
   - ダッシュボード開発

---

## 📚 参考資料

- [README.md](README.md) - プロジェクト概要
- [config.py](config.py) - 設定パラメータ
- Granite Time Series: https://github.com/ibm-granite/granite-tsfm

---

## 💡 Tips

### 全パイプライン一括実行

```powershell
python data_preprocessing.py && python range_definition.py && python train.py && python inference.py && python evaluate.py
```

### モデルテスト（小規模データ）

```python
# config.py で設定変更
MIN_DATA_POINTS = 30
LOOKBACK_DAYS = 30
TRAINING_CONFIG['num_epochs'] = 5
```

### GPU確認

```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}')"
```

---

**Last Updated:** 2026-02-12
