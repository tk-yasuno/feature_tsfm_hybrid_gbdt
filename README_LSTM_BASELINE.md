# LSTM Baseline Model - README

## 概要
Granite TSモデルと比較するためのLSTMベースラインモデルです。

## モデル構成
- **アーキテクチャ**: LSTM + Multi-head Attention
- **Hidden Size**: 128
- **LSTM Layers**: 2
- **Classification Heads**: 3つ（30日、60日、90日）
- **総パラメータ数**: 296,579

## ファイル構成
```
lstm_baseline_model.py       # LSTMモデル定義
train_lstm_baseline.py       # トレーニングスクリプト
evaluate_lstm_baseline.py    # 評価スクリプト
compare_models.py            # モデル比較スクリプト
```

## 使用方法

### 1. モデルのテスト
```bash
python lstm_baseline_model.py
```

### 2. トレーニング
```bash
python train_lstm_baseline.py
```

**トレーニング設定**:
- Batch Size: 32
- Learning Rate: 0.001
- Optimizer: AdamW
- Loss Function: Focal Loss (α=0.25, γ=2.0)
- Early Stopping: Patience=10
- 学習率スケジューラ: ReduceLROnPlateau

**出力**:
- モデル: `models/lstm_baseline/best_model/`
- トレーニング履歴: `results/lstm_baseline/training_history.json`

### 3. 評価
```bash
python evaluate_lstm_baseline.py
```

**評価メトリクス**:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC
- Confusion Matrix

**出力**:
- 評価結果: `results/lstm_baseline/evaluation_results_YYYYMMDD_HHMMSS.json`
- サマリー: `results/lstm_baseline/evaluation_summary_YYYYMMDD_HHMMSS.csv`

### 4. モデル比較
```bash
python compare_models.py
```

他のモデル（Granite TS、Hybrid、LightGBMなど）との比較を行います。

**出力**:
- 比較グラフ: `results/model_comparison_YYYYMMDD_HHMMSS/`
- サマリーテーブル: `comparison_summary.xlsx`, `comparison_summary.csv`

## モデルの特徴

### 長所
1. **シンプルな実装**: 理解しやすく、デバッグが容易
2. **高速な学習**: 約30万パラメータで軽量
3. **時系列の文脈考慮**: LSTMによる過去情報の保持
4. **Attention機構**: 重要な時点への注目

### 短所
1. **長期依存性**: 非常に長い系列では勾配消失の可能性
2. **並列化**: RNNベースのため、並列計算が制限される
3. **表現力**: Transformerベースのモデルと比較して表現力が低い

## Granite TSモデルとの比較

| 項目 | LSTM Baseline | Granite TS |
|------|---------------|------------|
| アーキテクチャ | LSTM + Attention | TinyTimeMixer (Transformer) |
| パラメータ数 | ~300K | ~数百万（ベース） |
| 学習速度 | 高速 | 中速 |
| 推論速度 | 高速 | 中速 |
| 表現力 | 中 | 高 |
| 事前学習 | なし | あり |
| LoRA適用 | なし | あり |

## トレーニング結果

トレーニングが完了すると、以下の情報が記録されます：

### Training History
```json
{
  "train": [
    {
      "loss": 0.xxx,
      "horizons": {
        "30d": {"loss": 0.xxx, "accuracy": 0.xxx},
        "60d": {"loss": 0.xxx, "accuracy": 0.xxx},
        "90d": {"loss": 0.xxx, "accuracy": 0.xxx}
      }
    }
  ],
  "val": [...]
}
```

### Evaluation Results
```json
{
  "30d": {
    "accuracy": 0.xxx,
    "precision": 0.xxx,
    "recall": 0.xxx,
    "f1_score": 0.xxx,
    "auc": 0.xxx,
    "confusion_matrix": [[TN, FP], [FN, TP]]
  }
}
```

## 改良の可能性

1. **双方向LSTM**: 過去と未来の両方の文脈を考慮
2. **GRU**: より軽量な代替アーキテクチャ
3. **より深いネットワーク**: 層数を増やして表現力を向上
4. **アンサンブル**: 複数のLSTMモデルを組み合わせ

## 参考文献

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
2. Vaswani, A., et al. (2017). Attention is all you need. NIPS 2017.
3. Lin, T. Y., et al. (2017). Focal loss for dense object detection. ICCV 2017.

## 問題が発生した場合

### CUDA/GPU関連エラー
```bash
# CPUで実行する場合、config.pyで設定
USE_GPU = False
```

### メモリ不足
```bash
# バッチサイズを減らす
# config.pyまたはコマンドライン引数で調整
```

### トレーニングの中断
```bash
# 最後のチェックポイントから再開
# （実装予定）
```

## ライセンス
プロジェクトのライセンスに従います。

## 更新履歴
- 2026-02-17: 初版作成
