# GitHub登録チェックリスト - hvac_tsfm_lora

## ✅ 完了した準備作業

### ドキュメント
- [x] README.md更新（v2.0失敗を追記、v1.1最終版を明記）
- [x] .gitignore作成（v2.0実験ファイル除外設定）
- [x] GITHUB_README.md作成（リポジトリトップ用サマリー）
- [x] hvac_64equip_Lesson.md（v1.1詳細レポート）
- [x] hvac_top5_Lesson.md（v1.0詳細レポート）

### コア構成
- [x] config.py（v1.1最適構成）
- [x] requirements.txt
- [x] LICENSE（MIT）

## 📁 GitHub登録対象ファイル（v1.1 Production）

### 必須ファイル
```
HVACRange_Deviation_Forecast/
├── README.md                      # メインドキュメント（更新済み）
├── .gitignore                     # 新規作成
├── LICENSE                        # MIT License
├── requirements.txt               # 依存パッケージ
├── config.py                      # v1.1構成
├── hvac_64equip_Lesson.md         # v1.1実験レポート
├── hvac_top5_Lesson.md            # v1.0実験レポート
└── GITHUB_README.md               # リポジトリトップ用（→README.mdとしてコピー）
```

### コアスクリプト（v1.1用）
```
├── data_preprocessing.py          # データパイプライン
├── range_definition.py            # 正常レンジ定義
├── granite_ts_model.py            # Granite TS + LoRA
├── train.py                       # トレーニング
├── inference.py                   # 推論
├── evaluate.py                    # 評価
└── select_64_equipment.py         # 64設備選定ロジック
```

### オプションスクリプト（キャリブレーション）
```
├── calibrate_model.py             # 確率キャリブレーション
├── calibrated_inference.py        # キャリブレーション推論
└── calibrated_evaluate.py         # キャリブレーション評価
```

### サポートファイル
```
├── setup.py                       # セットアップスクリプト
└── QUICKSTART.md                  # クイックスタートガイド（あれば）
```

## 🚫 除外するファイル（.gitignoreで設定済み）

### v2.0実験ファイル（失敗版）
```
✗ config_v2.py
✗ data_preprocessing_v2.py
✗ range_definition_v2.py
✗ train_v2.py
✗ evaluate_v2.py
✗ select_216_equipment.py
✗ check_full_data.py
```

### 一時/テストファイル
```
✗ check_*.py
✗ debug_*.py
✗ test_*.py
✗ analyze_*.py
✗ inspect_*.py
✗ investigate_*.py
✗ verify_*.py
✗ find_*.py
```

### 大容量データ/モデル
```
✗ data/**/*.csv                    # データファイルは除外
✗ models/**/*.safetensors          # モデル重みは除外
✗ results/**/*.png                 # 結果画像は除外
```

### Keep設定（JSONは含める）
```
✓ models/**/adapter_config.json    # LoRA設定のみ含める
✓ models/**/training_history.json  # 学習履歴のみ含める
✓ data/processed/selected_64_equipment.json  # 設備リスト含める
```

## 🔧 GitHub登録手順

### 1. ローカルリポジトリ初期化
```bash
cd C:\Users\yasun\RL\ner-equipment-granite\HVACRange_Deviation_Forecast
git init
```

### 2. README配置
```bash
# GITHUB_README.mdをルートREADME.mdとしてコピー
copy GITHUB_README.md ..\README.md
```

### 3. ファイル追加
```bash
git add .
git status  # 除外ファイルを確認
```

### 4. 初回コミット
```bash
git commit -m "v1.1 Production Ready - HVAC Time Series Foundation Model with LoRA"
```

### 5. リモートリポジトリ接続
```bash
# GitHubで手動作成したリポジトリ「hvac_tsfm_lora」に接続
git remote add origin https://github.com/YOUR_USERNAME/hvac_tsfm_lora.git
git branch -M main
```

### 6. プッシュ
```bash
git push -u origin main
```

## 📝 リポジトリ説明文（GitHub用）

**Title**: hvac_tsfm_lora

**Description**:
```
HVAC Time Series Foundation Model with LoRA Fine-tuning - 設備異常予測システム

IBM Granite Time Series + LoRA + SMOTE + Focal Lossによる高精度な予防保全モデル（v1.1 - 64設備, 検出率97%以上）
```

**Tags**:
```
time-series, anomaly-detection, predictive-maintenance, lora, granite, 
foundation-model, pytorch, transformers, hvac, equipment-monitoring
```

## ✅ 完成版としての確認事項

- [x] README.mdにv1.1が最終版と明記
- [x] v1.3/v2.0失敗を明確に文書化
- [x] config.pyがv1.1構成（r=8, α=16, γ=3.0）
- [x] .gitignoreでv2.0ファイル除外
- [x] LICENSEファイル存在確認
- [x] requirements.txt更新確認

## 🎯 次のアクション

1. **GitHub手動作成**: リポジトリ「hvac_tsfm_lora」を作成
2. **ローカル初期化**: 上記手順1-4を実行
3. **リモート接続**: 上記手順5-6を実行
4. **公開確認**: GitHubでファイル構造とREADME表示を確認

---

**Status**: ✅ GitHub登録準備完了 - v1.1 Production Ready
