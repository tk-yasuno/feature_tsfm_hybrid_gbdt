# Pump Range Deviation Forecast MVP - Project Summary

**プロジェクト:** Granite TSを使った時系列分類モデルのMVP実装  
**対象データ:** ポンプ設備データ（CSV_260106_Pump_Equipment_13K.csv）  
**実装日:** 2026年2月12日  
**ステータス:** ✅ MVP完成

---

## 📋 プロジェクト概要

### 目的

ポンプ設備の測定値が正常レンジから逸脱する確率を、Granite Time Series Foundation ModelとLoRAを用いて予測するMVPシステムの構築。

### 予測タスク

1. **現在時点の状態分類**
   - 測定値が正常レンジ内にあるかを判定
   
2. **将来逸脱確率予測**
   - 30日先、60日先、90日先の逸脱確率を予測
   - 式: `P(X_{t+h} ∉ Range | X_{t-89:t})`

---

## 🏗️ システムアーキテクチャ

```
データフロー:
CSV_260106_Pump_Equipment_13K.csv (13,000件)
  ↓
[1] data_preprocessing.py
  - 日次集計
  - 欠損値補間
  - z-score正規化
  ↓
processed_time_series.csv
  ↓
[2] range_definition.py
  - 統計ベースレンジ定義 (5th-95th percentile)
  - 現在・将来ラベル生成
  - スライディングウィンドウサンプリング
  ↓
training_samples.csv + range_definitions.json
  ↓
[3] train.py
  - Granite TS + LoRA
  - バイナリクロスエントロピー損失
  - Early Stopping
  ↓
granite_pump_lora/best_model/
  ↓
[4] inference.py
  - 逸脱確率予測
  - アラート判定 (WARNING: 0.7+, CRITICAL: 0.9+)
  ↓
inference_results.csv + inference_alerts.csv
  ↓
[5] evaluate.py
  - ROC-AUC, PR-AUC計算
  - リードタイム分析
  - 可視化
  ↓
evaluation_metrics.json + 各種グラフ
```

---

## 📁 プロジェクト構造

```
PumpRange_Deviation_Forecast/
├── README.md                  # プロジェクト概要
├── QUICKSTART.md             # クイックスタートガイド
├── PROJECT_SUMMARY.md        # このファイル
├── requirements.txt          # 依存パッケージ
├── config.py                 # 設定ファイル
├── setup.py                  # セットアップスクリプト
│
├── data_preprocessing.py     # データ前処理
├── range_definition.py      # レンジ定義・ラベル生成
├── granite_ts_model.py      # Granite TSモデル実装
├── train.py                 # トレーニング
├── inference.py             # 推論
├── evaluate.py              # 評価
│
├── data/
│   ├── raw/                 # 生データ
│   ├── processed/           # 前処理済みデータ
│   │   ├── processed_time_series.csv
│   │   ├── labeled_time_series.csv
│   │   └── training_samples.csv
│   └── ranges/              # レンジ定義
│       ├── range_definitions.json
│       └── labeling_stats.json
│
├── models/
│   └── granite_pump_lora/   # 訓練済みモデル
│       └── best_model/
│
├── results/                  # 実験結果
│   ├── training_history.json
│   ├── inference_results_*.csv
│   ├── inference_alerts_*.csv
│   ├── evaluation_metrics.json
│   ├── roc_curves.png
│   ├── pr_curves.png
│   └── confusion_matrices.png
│
└── notebooks/
    └── mvp_demo.ipynb       # デモノートブック
```

---

## 🔧 技術スタック

### コアテクノロジー

- **Granite Time Series:** IBM製時系列Foundation Model
- **LoRA (Low-Rank Adaptation):** 軽量ファインチューニング手法
- **PyTorch:** ディープラーニングフレームワーク
- **Transformers/PEFT:** HuggingFaceライブラリ

### データ処理

- **Pandas:** データ処理
- **NumPy:** 数値計算
- **Scikit-learn:** 評価指標

### 可視化

- **Matplotlib/Seaborn:** グラフ作成
- **Plotly:** インタラクティブ可視化（オプション）

---

## 📊 主要パラメータ

### データパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| LOOKBACK_DAYS | 90 | 過去参照長（日） |
| FORECAST_HORIZONS | [30, 60, 90] | 予測ホライズン（日） |
| MIN_DATA_POINTS | 180 | 最小データポイント数 |
| LOWER_PERCENTILE | 5 | 正常レンジ下限（分位点） |
| UPPER_PERCENTILE | 95 | 正常レンジ上限（分位点） |

### モデルパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| LoRA r | 8 | LoRAランク |
| LoRA alpha | 16 | スケーリング係数 |
| Target Modules | ["q_proj", "v_proj"] | LoRA適用層 |

### トレーニングパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| Batch Size | 32 | バッチサイズ |
| Learning Rate | 5e-5 | 学習率 |
| Num Epochs | 20 | 最大エポック数 |
| Patience | 5 | Early Stopping耐性 |
| Train/Val/Test | 0.7/0.15/0.15 | データ分割比率 |

---

## 🎯 実装された機能

### ✅ データ処理パイプライン

- [x] CSV読み込み（エンコーディング処理）
- [x] 日時パース処理
- [x] 日次集計
- [x] 欠損値補間（前値/後値保持）
- [x] 設備×測定項目ごとの時系列構築
- [x] z-score正規化
- [x] 統計情報出力

### ✅ レンジ定義・ラベル生成

- [x] 分位点ベースレンジ定義（5th-95th）
- [x] 現在ラベル生成（y_t）
- [x] 将来ラベル生成（y_{t,h} for h=30,60,90）
- [x] スライディングウィンドウサンプリング
- [x] ラベル分布分析

### ✅ モデル実装

- [x] Granite TSベースモデル統合
- [x] LoRA適用
- [x] LSTMフォールバックモデル
- [x] マルチホライズン分類ヘッド
- [x] Focal Loss（クラス不均衡対応）
- [x] モデル保存・読み込み

### ✅ トレーニング

- [x] データローダー構築
- [x] 訓練/検証/テスト分割
- [x] Early Stopping
- [x] 学習率スケジューリング
- [x] 勾配クリッピング
- [x] 学習履歴保存

### ✅ 推論

- [x] 訓練済みモデル読み込み
- [x] バッチ予測
- [x] 逸脱確率計算
- [x] アラート判定（WARNING/CRITICAL）
- [x] 結果CSV出力

### ✅ 評価

- [x] ROC-AUC計算
- [x] PR-AUC計算（希少イベント重視）
- [x] Precision/Recall/F1-Score
- [x] 混同行列生成
- [x] リードタイム分析
- [x] ROC/PR曲線プロット
- [x] 評価レポート出力

### ✅ ドキュメント

- [x] README.md
- [x] QUICKSTART.md
- [x] PROJECT_SUMMARY.md（このファイル）
- [x] デモノートブック
- [x] コード内ドキュメント

---

## 🚀 クイックスタート

### 環境セットアップ

```powershell
cd PumpRange_Deviation_Forecast
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python setup.py
```

### パイプライン実行

```powershell
# 全パイプライン実行
python data_preprocessing.py
python range_definition.py
python train.py
python inference.py
python evaluate.py
```

### ノートブックでの確認

```powershell
jupyter notebook notebooks/mvp_demo.ipynb
```

---

## 📈 期待される成果物

### データ成果物

- 処理済み時系列データ（CSV）
- 正常レンジ定義（JSON）
- 学習サンプルデータ（CSV）

### モデル成果物

- 訓練済みLoRAアダプター
- 学習履歴（JSON）
- モデルチェックポイント

### 評価成果物

- 推論結果（CSV）
- アラート一覧（CSV）
- 評価メトリクス（JSON）
- ROC/PR曲線（PNG）
- 混同行列（PNG）

---

## 🔍 評価指標

### 分類性能

- **ROC-AUC:** 全体的な分類性能指標
- **PR-AUC:** 希少イベント（異常）検出性能（重視）
- **Precision:** 異常と予測したうちの正解率
- **Recall:** 実際の異常のうち検出できた割合
- **F1-Score:** Precision/Recallの調和平均

### リードタイム分析

各予測ホライズン（30/60/90日）について：
- 閾値0.5/0.7/0.9での検出率
- 実際の異常を何日前に予測できたか

---

## 💡 技術的工夫

### 1. クラス不均衡対応

- **Focal Loss:** 正常データが多い問題に対応
- **Early Stopping:** 過学習防止
- **PR-AUC重視:** 希少イベント評価

### 2. スケーラビリティ

- **LoRA:** 大規模モデルの軽量ファインチューニング
- **バッチ処理:** 効率的な推論
- **モジュール設計:** 拡張しやすい構造

### 3. 実用性

- **フォールバックモデル:** Granite TS未導入でもLSTMで動作
- **段階的実行:** 各ステップを独立して実行可能
- **可視化:** 豊富なグラフで結果を理解しやすく

---

## 🎓 学習したこと

### データサイエンス

- 時系列データの正常レンジ定義方法
- 分位点ベース統計手法
- スライディングウィンドウサンプリング
- クラス不均衡データの扱い方

### ディープラーニング

- Foundation Modelの活用
- LoRAによる転移学習
- マルチタスク学習（マルチホライズン）
- Early Stoppingとハイパーパラメータ調整

### エンジニアリング

- モジュール化されたパイプライン設計
- 設定ファイルによる柔軟な管理
- エラーハンドリングとフォールバック
- ドキュメント駆動開発

---

## 🔮 今後の展開

### Phase 2: モデル改善

- [ ] Granite TSの完全版導入
- [ ] ハイパーパラメータチューニング（Optuna）
- [ ] アンサンブル手法の検討
- [ ] Attention可視化

### Phase 3: 機能拡張

- [ ] リアルタイム推論API（FastAPI）
- [ ] ダッシュボード開発（Streamlit/Dash）
- [ ] アラート通知システム
- [ ] A/Bテスト機能

### Phase 4: ツインエンジン統合

- [ ] Engine 2（物理モデル/GNN）の実装
- [ ] 2エンジンの予測結果統合
- [ ] 信頼度スコアリング
- [ ] 説明可能性の向上

### Phase 5: 本番展開

- [ ] コンテナ化（Docker）
- [ ] CI/CD構築
- [ ] モニタリング（MLflow）
- [ ] オートスケーリング

---

## 📚 参考資料

### 技術論文・ドキュメント

- Granite Time Series: https://github.com/ibm-granite/granite-tsfm
- LoRA: Low-Rank Adaptation of Large Language Models
- Focal Loss for Dense Object Detection

### 関連プロジェクト

- equipment_ner_mvp/ - Granite NERによる設備名抽出
- PumpRange_Deviation_Forecast/ - このプロジェクト

---

## 👥 コントリビューター

- **実装:** GitHub Copilot (Claude Sonnet 4.5)
- **設計:** User (yasun)
- **日付:** 2026年2月12日

---

## 📝 ライセンス

MIT License

---

## 🙏 謝辞

- IBM Granite Time Series チーム
- HuggingFace Transformers チーム
- PyTorch コミュニティ

---

**Last Updated:** 2026-02-12  
**Status:** ✅ MVP Complete and Ready for Testing
