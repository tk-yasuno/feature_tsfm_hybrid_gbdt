# hvac_tsfm_lora

**HVAC Time Series Foundation Model with LoRA Fine-tuning**

設備異常予測システム - IBM Granite Time Series + LoRA + SMOTE + Focal Lossによる高精度な予防保全モデル

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-v1.1%20Production%20Ready-brightgreen.svg)](HVACRange_Deviation_Forecast/README.md)

## 🎯 主要成果

**v1.1 (64設備 - 最終完成版)**
- 60日先予測: **検出率 99.7%**
- 90日先予測: **検出率 97.4%**
- 訓練時間: 15.1分（NVIDIA GPU）
- モデルサイズ: 29,504パラメータ（22.1% trainable）

## 📊 実験プロセスと教訓

| Version | 設備数 | Val Loss | 60d検出率 | 90d検出率 | Status |
|---------|--------|----------|-----------|-----------|--------|
| v1.0 | 5 | 0.0063 | - | **99.6%** | ✅ 成功 |
| v1.1 | 64 | **0.0136** | **99.7%** | **97.4%** | ✅ **最終版** 🏆 |
| v1.3 | 64 | 0.0126 | 0% | 0% | ❌ 過学習 |
| v2.0 | 112 | 0.0130 | 0% | 0% | ❌ スケール限界 |

**重要な教訓:**
- ✅ v1.1 (64設備) が最適バランス
- ❌ モデル容量拡大は逆効果（v1.3失敗）
- ❌ 単純なスケールアップは性能劣化（v2.0失敗）
- 🔑 **データ品質 > データ量**

## 🚀 Quick Start

### インストール
```bash
git clone https://github.com/YOUR_USERNAME/hvac_tsfm_lora.git
cd hvac_tsfm_lora/HVACRange_Deviation_Forecast
pip install -r requirements.txt
```

### データ前処理
```bash
python data_preprocessing.py
python range_definition.py
```

### トレーニング
```bash
python train.py  # v1.1構成で約15分
```

### 評価
```bash
python inference.py
python evaluate.py
```

## 📁 プロジェクト構造

```
hvac_tsfm_lora/
└── HVACRange_Deviation_Forecast/
    ├── README.md                    # 詳細ドキュメント
    ├── config.py                    # v1.1最適構成
    ├── requirements.txt
    ├── data_preprocessing.py        # データパイプライン
    ├── range_definition.py          # 正常レンジ定義
    ├── granite_ts_model.py          # Granite TS + LoRA
    ├── train.py                     # トレーニング
    ├── inference.py                 # 推論
    ├── evaluate.py                  # 評価
    ├── hvac_64equip_Lesson.md       # v1.1実験レポート
    └── hvac_top5_Lesson.md          # v1.0実験レポート
```

## 🔑 技術スタック

- **Foundation Model**: IBM Granite Time Series (TinyTimeMixer-512-96)
- **Fine-tuning**: LoRA (r=8, α=16) - パラメータ効率的な学習
- **Data Augmentation**: SMOTE - クラス不均衡対応
- **Loss Function**: Focal Loss (γ=3.0) - 難しいサンプル重視
- **Framework**: PyTorch 2.8.0, Transformers, PEFT

## 📚 詳細ドキュメント

- [詳細README](HVACRange_Deviation_Forecast/README.md) - 完全な実験結果と技術詳細
- [v1.1実験レポート](HVACRange_Deviation_Forecast/hvac_64equip_Lesson.md) - スケーリング成功と失敗の教訓
- [v1.0実験レポート](HVACRange_Deviation_Forecast/hvac_top5_Lesson.md) - 初期実験とキャリブレーション

## 🎓 主要な学び

### 成功要因
1. **LoRA Fine-tuning**: 22.1%のパラメータで高精度達成
2. **SMOTE + Focal Loss**: クラス不均衡を効果的に解決
3. **64設備が最適**: データ品質と量のバランス

### 失敗から得た教訓
1. **v1.3**: モデル容量を増やせば良いわけではない
2. **v2.0**: 設備数を増やせば良いわけではない
3. **原則**: Val Loss改善 ≠ Test性能向上（過学習に注意）

## 🏆 本番適用推奨

**v1.1 (64設備) を本番構成として推奨**
- 60日先/90日先の異常を高精度で予測
- 検出率97%以上で実用レベル
- 予防保全による設備ダウンタイム削減に貢献

## 📄 License

MIT License

## 👥 Contact

プロジェクトに関する質問や提案は、Issueまでお願いします。

---

**Status**: ✅ **Production Ready** - v1.1は即座に本番適用可能
