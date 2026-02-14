# Precision向上戦略
## HVAC Range Deviation Forecast - 実用的改善手法

**現状の課題**
- Precision: 9-11%（非常に低い - 誤検知が多い）
- Recall: 46-95%（高い - 異常は検出できている）
- 実用性: 10回アラートを出して1回しか当たらない状態

---

## 🎯 改善戦略マップ（効果・難易度・実装時間）

| 手法 | 期待効果 | 実装難易度 | 実装時間 | 優先度 |
|------|---------|-----------|---------|-------|
| **1. 特徴量エンジニアリング** | ⭐⭐⭐⭐ | 低 | 1-2日 | **最優先** |
| **2. クラス不均衡対策の改善** | ⭐⭐⭐⭐ | 中 | 1日 | **最優先** |
| **3. 時系列平滑化・後処理** | ⭐⭐⭐ | 低 | 半日 | 高 |
| **4. 設備・項目別閾値最適化** | ⭐⭐⭐ | 中 | 1日 | 高 |
| **5. アンサンブル学習** | ⭐⭐⭐ | 中 | 2-3日 | 中 |
| **6. 深い分類ヘッド** | ⭐⭐ | 低 | 半日 | 中 |
| **7. Focal Loss導入** | ⭐⭐⭐ | 低 | 半日 | 中 |
| **8. データ品質向上** | ⭐⭐⭐⭐ | 高 | 3-5日 | 低（長期） |

---

## 📊 戦略1: 特徴量エンジニアリング（最優先）

### なぜ効果的か
現在は「生の時系列データのみ」を入力している。統計的特徴量を追加することで：
- 異常のパターンをより明示的に学習できる
- モデルが「どこが変か」を理解しやすくなる
- 誤検知を減らせる

### 実装すべき特徴量

#### A. 統計的特徴量（過去90日間）
```python
1. 平均値（mean）
2. 標準偏差（std）
3. 最大値・最小値
4. 四分位数（25%, 50%, 75%）
5. 歪度（skewness）- 分布の非対称性
6. 尖度（kurtosis）- 分布の尖り具合
```

#### B. トレンド特徴量
```python
1. 線形回帰の傾き（上昇/下降トレンド）
2. 最近30日 vs 過去60日の平均値比
3. 最近の変化率（最後の10日間の傾き）
```

#### C. レンジ関連特徴量
```python
1. 上限までの距離の平均
2. 下限までの距離の平均
3. レンジ内滞在率
4. レンジ逸脱回数（過去90日間）
5. 連続してレンジ内にいる日数
```

#### D. 変動性特徴量
```python
1. ローリング標準偏差（7日、14日、30日）
2. 変化率の標準偏差
3. 最大ドローダウン
```

#### E. 時間特徴量
```python
1. 日次の測定回数
2. 欠損値の割合
3. 最終測定からの経過日数
```

**実装**: `create_enriched_features.py` を作成

---

## 🔧 戦略2: クラス不均衡対策の改善

### 現状の問題
現在はSMOTEを使用しているが、時系列データには最適ではない。

### 改善策

#### A. クラスウェイトの動的調整
```python
# 損失関数にクラスウェイトを適用
class_weights = {
    0: 1.0,  # 正常
    1: 10.0  # 異常（より重視）
}
```

#### B. Focal Loss の導入
```python
# 難しい例（誤分類されやすい例）に焦点を当てる
# 簡単な例（正しく分類される例）の重みを下げる
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
```

#### C. SMOTE-NCの代わりにADASYNを使用
```python
# ADASYNは少数クラスの境界領域により多くのサンプルを生成
from imblearn.over_sampling import ADASYN
```

**実装**: `train_with_focal_loss.py` を作成

---

## 🔍 戦略3: 時系列平滑化・後処理（即効性高）

### なぜ効果的か
単発のアラートは誤検知の可能性が高い。時間的な連続性を考慮する。

### 実装方法

#### A. 連続アラート判定
```python
# 連続して2-3日異常判定が出た場合のみアラート
def smooth_predictions(predictions, window=3):
    """
    連続してwindow日間異常予測が出た場合のみTrueにする
    """
    smoothed = []
    for i in range(len(predictions)):
        if i < window - 1:
            smoothed.append(False)
        else:
            # 過去window日間すべて異常予測
            if all(predictions[i-window+1:i+1]):
                smoothed.append(True)
            else:
                smoothed.append(False)
    return smoothed
```

#### B. 指数移動平均による平滑化
```python
# 予測確率を平滑化
def ema_smoothing(probabilities, alpha=0.3):
    smoothed = [probabilities[0]]
    for p in probabilities[1:]:
        smoothed.append(alpha * p + (1 - alpha) * smoothed[-1])
    return smoothed
```

**実装**: `post_process_predictions.py` を作成

---

## 🎛️ 戦略4: 設備・項目別閾値最適化

### なぜ効果的か
現在は全設備・全項目で同じ閾値を使用。しかし：
- 設備によって異常の出やすさが異なる
- 項目（温度、圧力など）によって特性が異なる

### 実装方法

```python
# 設備ID × チェック項目IDごとに最適閾値を計算
optimal_thresholds = {}

for equip_id in unique_equipment:
    for item_id in unique_items:
        subset = test_df[
            (test_df['equipment_id'] == equip_id) & 
            (test_df['check_item_id'] == item_id)
        ]
        
        if len(subset) > 10:  # 最低サンプル数
            # F1最大化
            threshold = find_optimal_threshold(
                subset['true_label'], 
                subset['pred_prob']
            )
            optimal_thresholds[(equip_id, item_id)] = threshold
```

**実装**: `optimize_thresholds_per_equipment.py` を作成

---

## 🤖 戦略5: アンサンブル学習

### なぜ効果的か
複数のモデルの予測を組み合わせることで、誤検知を減らせる。

### 実装方法

#### A. モデルアンサンブル
```python
# 異なるモデルを組み合わせ
models = [
    GraniteTimeSeriesClassifier(),  # 現在のモデル
    LSTMClassifier(),               # LSTM
    XGBoostClassifier()             # XGBoost（特徴量ベース）
]

# 投票または平均
ensemble_pred = np.mean([model.predict(X) for model in models], axis=0)
```

#### B. Stacking
```python
# レベル1: 複数のベースモデル
# レベル2: メタモデルがベースモデルの予測を統合
```

**実装**: `ensemble_training.py` を作成

---

## 🏗️ 戦略6: モデルアーキテクチャの改善

### A. より深い分類ヘッド
現在の分類ヘッドは3層。これを深くして表現力を向上。

```python
self.classification_heads = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.BatchNorm1d(hidden_size),
    nn.Dropout(0.3),
    
    nn.Linear(hidden_size, hidden_size // 2),
    nn.ReLU(),
    nn.BatchNorm1d(hidden_size // 2),
    nn.Dropout(0.3),
    
    nn.Linear(hidden_size // 2, hidden_size // 4),
    nn.ReLU(),
    nn.BatchNorm1d(hidden_size // 4),
    nn.Dropout(0.2),
    
    nn.Linear(hidden_size // 4, 1),
    nn.Sigmoid()
)
```

### B. Attention機構の追加
```python
# 時系列のどの部分が重要かを学習
attention_weights = F.softmax(attention_scores, dim=1)
context = torch.sum(attention_weights * features, dim=1)
```

---

## 🎲 戦略7: Focal Loss導入

### なぜ効果的か
- 簡単に分類できるサンプル（正常と明らかにわかる）の重みを下げる
- 難しいサンプル（境界領域）に集中して学習
- クラス不均衡に強い

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
```

**実装**: `focal_loss.py` を作成

---

## 📈 戦略8: データ品質向上（長期施策）

### A. ラベルの再定義
現在の逸脱定義を見直す：
```python
# 現在: レンジ外に1回でも出たら異常
# 改善案1: 連続してX回レンジ外に出たら異常
# 改善案2: レンジ外滞在率がY%以上で異常
# 改善案3: レンジからの距離がZ以上で異常
```

### B. 外れ値の除去
```python
# IQR法で外れ値を除去
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
outliers = (data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)
```

### C. 設備選定の見直し
異常が極端に少ない設備は学習データから除外。

---

## 🚀 推奨実装順序（クイックウィン戦略）

### Week 1: 即効性の高い施策（Precision 10% → 20-25%目標）

#### Day 1-2: 特徴量エンジニアリング
- [ ] 統計的特徴量の追加
- [ ] レンジ関連特徴量の追加
- [ ] モデル再学習

#### Day 3: 後処理の実装
- [ ] 時系列平滑化
- [ ] 連続アラート判定
- [ ] 評価

#### Day 4: クラス不均衡対策
- [ ] Focal Loss実装
- [ ] クラスウェイト調整
- [ ] 再学習・評価

#### Day 5: 閾値最適化
- [ ] 設備・項目別閾値計算
- [ ] 評価・調整

### Week 2: 中期施策（Precision 25% → 35-40%目標）

- [ ] より深い分類ヘッド
- [ ] アンサンブル学習
- [ ] データ拡張

### Week 3+: 長期施策（Precision 40%+ 目標）

- [ ] ラベル定義の見直し
- [ ] データ品質向上
- [ ] マルチタスク学習

---

## 📊 効果測定方法

各施策実施後、以下を記録：

```python
metrics = {
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred),
    'roc_auc': roc_auc_score(y_true, y_score),
    'false_positive_rate': FP / (FP + TN),
    'false_alarm_reduction': (baseline_FP - current_FP) / baseline_FP
}
```

---

## 🎯 目標設定

| フェーズ | Precision目標 | Recall維持 | 実用性 |
|---------|--------------|-----------|-------|
| 現状 | 10% | 75% | 10回に1回的中 |
| Phase 1 | 20-25% | 70%以上 | 5回に1回的中 |
| Phase 2 | 30-40% | 65%以上 | 3回に1回的中 ✅ |
| Phase 3 | 50%+ | 60%以上 | 2回に1回的中 🎯 |

---

## 💡 追加の最適化アイデア

### ビジネス要件ベースの最適化
```python
# 重要度に応じて閾値を変える
if equipment_category == "重要設備":
    threshold = 0.3  # 低めで感度高く
else:
    threshold = 0.6  # 高めで誤検知を減らす
```

### コスト考慮型学習
```python
# False Positive（誤検知）のコストが高い場合
cost_matrix = {
    'TP': 100,   # 正しく異常検知: +100
    'TN': 0,     # 正しく正常判定: 0
    'FP': -50,   # 誤検知: -50（対応コスト）
    'FN': -200   # 見逃し: -200（設備停止コスト）
}
```

---

## 📝 実装チェックリスト

すぐに実装可能な施策：
- [ ] `create_enriched_features.py` - 特徴量エンジニアリング
- [ ] `train_with_focal_loss.py` - Focal Loss
- [ ] `post_process_predictions.py` - 時系列平滑化
- [ ] `optimize_thresholds_per_equipment.py` - 個別閾値最適化

これらを順次実装・評価することで、段階的にPrecisionを向上できます。
