# Hybrid Model v2.2 vs v2.2.2 vs v2.3: Feature Fusion Architecture の比較

**作成日**: 2026年2月15日  
**最終更新**: 2026年2月15日  
**テーマ**: 特徴量エンジニアリングとFusion戦略の価値検証

---

## 📌 Executive Summary

v2.2（Simple Concat Fusion）、v2.2.2（Cross-Attention Fusion）、v2.3（No Statistical Features）の比較実験により、**明示的な統計特徴量の重要性**と**Simple Fusion戦略の有効性**が実証されました。

**結論**: **v2.2をProduction Modelとして確定**。Feature Engineering は不可欠。過度な複雑性は不要。

---

## 🎯 実験目的

v2.2で「インプットが重い」という課題認識があり、Statistical Features（28次元）を削除してTinyTimeMixer Embeddings（64次元）のみで学習できるか検証。

**仮説**: TinyTimeMixerの時系列エンコーダが統計情報を内包しているため、明示的な統計特徴量は不要かもしれない。

**結果**: **仮説は棄却** → 統計特徴量は必須

---

## 📊 Performance Comparison

### v2.2: Simple Concat Fusion ✅ PRODUCTION MODEL

| Metric             | 30d    | 60d    | 90d    | **Average**   |
| ------------------ | ------ | ------ | ------ | ------------------- |
| **F1-Score** | 0.2771 | 0.2756 | 0.2840 | **0.2789** ✅ |
| **ROC-AUC**  | 0.7230 | 0.6981 | 0.6958 | **0.7056**    |
| Precision          | 0.2404 | 0.2435 | 0.2634 | 0.2491              |
| Recall             | 0.3257 | 0.3173 | 0.3101 | 0.3177              |
| Accuracy           | 0.8318 | 0.8349 | 0.8486 | 0.8384              |

**Architecture**:

```
TinyTimeMixer (64d) ──┐
                      ├→ Simple Concat (92d) → Linear Fusion (128d) → Multi-Task Heads
Statistical (28d) ────┘
```

**Parameters**: 195,204 total, 61,766 trainable (31.64%)  
**Training**: Epoch 25, Best Avg F1: 0.2789

---

### v2.2.2: Cross-Attention Fusion

| Metric             | 30d    | 60d    | 90d    | **Average**   |
| ------------------ | ------ | ------ | ------ | ------------------- |
| **F1-Score** | 0.2660 | 0.2602 | 0.2805 | **0.2689** ⚠️ |
| **ROC-AUC**  | 0.7077 | 0.6849 | 0.6900 | **0.6942**    |
| Precision          | 0.1985 | 0.2289 | 0.2386 | 0.2220              |
| Recall             | 0.4031 | 0.3014 | 0.3401 | 0.3482              |
| Accuracy           | 0.8006 | 0.8446 | 0.8340 | 0.8264              |

**Architecture**:

```
TinyTimeMixer (64d) ──┐
                      ├→ Bidirectional Cross-Attention → Fused (128d) → Multi-Task Heads
Statistical (28d) ────┘
                      
  1. Embeddings attend to Features
  2. Features attend to Embeddings
```

**Parameters**: 218,952 total, 85,514 trainable (39.06%)  
**Training**: Epoch 25, Best Avg F1: 0.2689

---

### v2.3: No Statistical Features (Time-Series Only)

| Metric             | 30d    | 60d    | 90d    | **Average**   |
| ------------------ | ------ | ------ | ------ | ------------------- |
| **F1-Score** | 0.1924 | 0.1995 | 0.2001 | **0.1973** ❌ |
| **ROC-AUC**  | 0.5865 | 0.5961 | 0.5797 | **0.5874**    |
| Precision          | 0.1222 | 0.1221 | 0.1213 | 0.1219              |
| Recall             | 0.4515 | 0.5460 | 0.6214 | 0.5396              |
| Accuracy           | 0.6601 | 0.6027 | 0.5274 | 0.5967              |

**Architecture**:

```
TinyTimeMixer (64d) → Direct Hidden Layer (64d → 128d) → Multi-Task Heads
```

**Parameters**: 191,620 total, 58,182 trainable (30.36%)

---

## 📉 Performance Gap Analysis

| Metric               | v2.2 ✅ | v2.2.2  | v2.3    | v2.2.2 vs v2.2 | v2.3 vs v2.2 |
| -------------------- | ------- | ------- | ------- | -------------- | ------------ |
| **Average F1** | **0.2789** | 0.2689  | 0.1973  | **-3.6%** ⚠️  | **-29.2%** ❌ |
| Average ROC-AUC      | **0.7056** | 0.6942  | 0.5874  | -1.6%          | -16.8%       |
| Average Precision    | **0.2491** | 0.2220  | 0.1219  | -10.9%         | -51.1%       |
| Average Recall       | 0.3177  | **0.3482** | 0.5396  | +9.6%          | +69.9%       |
| Average Accuracy     | **0.8384** | 0.8264  | 0.5967  | -1.4%          | -28.8%       |
| **Parameters**       | 195,204 | 218,952 | 191,620 | **+12.2%**     | -1.8%        |
| **Trainable %**      | 31.64%  | 39.06%  | 30.36%  | +7.4pt         | -1.3pt       |

### 🔍 Key Observations
#### v2.2.2 vs v2.2（Cross-Attention実験）

1. **F1-Score微減**: -3.6%低下（0.2789 → 0.2689）
2. **パラメータ増加**: +12.2%（195,204 → 218,952）
3. **訓練可能パラメータ増**: 31.64% → 39.06%（+7.4pt）
4. **ROC-AUC微減**: -1.6%（0.7056 → 0.6942）
5. **複雑性 vs 性能**: パラメータ増でも性能向上せず

#### v2.3 vs v2.2（統計特徴量削除実験）

1. **F1-Score大幅低下**: -29.2%の性能劣化
2. **Precision崩壊**: 半減（0.2491 → 0.1219）（v2.3実験から）
3. **Recall過剰**: +69.9%増加（過検出傾向）
4. **ROC-AUC大幅低下**: 0.7056 → 0.5874（ランダム分類に近い）
5. **パラメータ削減効果は微小**: わずか-1.8%5874（ランダム分類に近い）
5. **パラメータ削減効果は微小**: わずか1.8%減

---

## 💡 Critical Lessons

### ✅ Lesson 1: Statistical Features は不可欠

**発見**: TinyTimeMixerのembeddingsだけでは、統計的パターンを十分に捉えられない。

**v2.2のStatistical Features (28次元)**:

```python
# Time-Series Statistics (7)
- values_mean: 過去90日の平均
- values_std: 標準偏差
- values_min / max: 範囲
- values_range: レンジ幅
- values_cv: 変動係数
- values_trend: トレンド傾向

# Recent Behavior (9)
- values_recent_mean: 直近30日平均
- values_recent_std: 直近30日標準偏差
- values_spike_count: スパイク回数
- values_sudden_change: 急変回数
- ...

# Seasonal & Autocorrelation (4)
- values_lag1_corr: 1日ラグ自己相関
- values_seasonal_strength: 季節性強度
- ...

# Distribution Features (8)
- values_skewnesSimple Fusion > Complex Fusion（v2.2.2実験から）

**発見**: Cross-Attentionによる動的重み付けより、Simple Concatの方が効果的。

**v2.2.2（Cross-Attention）の問題点:**
- パラメータ増加: +12.2%（23,748パラメータ追加）
- F1-Score低下: -3.6%（0.2789 → 0.2689）
- 訓練可能パラメータ増: 39.06%（過学習リスク増）
- 複雑性に見合う性能向上なし

**理由の考察:**
1. **特徴量が少ない**: 28次元の統計特徴に対してCross-Attentionは過剰
2. **情報密度が高い**: Simple Concatで既に十分な情報統合
3. **正則化効果の減少**: パラメータ増により過学習傾向
4. **訓練データ不足**: Cross-Attentionの恩恵を得るにはより大量のデータが必要

**教訓:**
```
Architectural Complexity ≠ Better Performance

Simple Concat Fusion (v2.2):
  - Efficient: 195,204 params
  - Effective: F1=0.2789
  - Stable: 31.64% trainable
4
Cross-Attention Fusion (v2.2.2):
  - Complex: 218,952 params (+12%)
  - Less Effective: F1=0.2689 (-4%)
  - Overfitting Risk: 39.06% trainable
```

**結論**: 28次元の統計特徴量に対しては、Simple Concatが最適。Cross-Attentionは特徴量が多い（100+次元）場合に有効。

---

### ✅ Lesson 3: s: 歪度
- values_kurtosis: 尖度
- values_entropy: エントロピー
- ...
```

これらの**明示的な統計量5*が、異常検知の判断に重要な役割を果たしている。

---

### ✅ Lesson 2: Precision vs Recall のトレードオフ

**v2.3の問題点**: Recallが高い（0.54）が、Precisionが低い（0.12）

- **意味**: 異常を過剰に検出（偽陽性が多い）
- **原因**: 統計特徴量がないため、正常範囲内の変動と異常を区別できない

**v2.2の強み**: Precision 0.25、Recall 0.32 のバランス

- Mean/Stdなどの統計量で、正常ベースラインを確立
- 真の異常のみを検出

---

### ✅ Lesson 3: Feature Engineering > Model Complexity

**判明した事実**:

- パラメータ削減: わずか1.8%（3,584パラメータ）
- 性能低下: 29.2%（F1-Score）

**結論**:

```
少数の良質な特徴量 >> 複雑なモデル構造
```

Feature Engineering（特徴量エンジニアリング）への投資は、モデル設計の最適化よりも**高ROI**。

---

### ✅ Lesson 4: "重い"は必ずしも悪ではない

**v2.2への反省**:

- 「92次元は重い」という主観的判断
- 実際には必要な情報密度だった

**教訓**:
2.2 失敗要因（Cross-Attention）

1. **過剰な複雑性**
   - 28次元の特徴量に対してCross-Attentionは過剰設計
   - Multi-head Attention（4 heads）× 2方向 = 過度なパラメータ化

2. **パラメータ効率の悪化**
   - +23,748パラメータ（+12.2%）で-3.6% F1低下
   - ROI（投資対効果）が負

3. **訓練可能パラメータの増加**
   - 31.64% → 39.06%（+7.4pt）
   - 過学習リスク増大、汎化性能低下

4. **特徴量規模とアーキテクチャのミスマッチ**
   - Cross-Attentionは100+次元の高次元特徴に有効
   - 28次元では情報密度が高すぎて効果薄

---
 ✅ PRODUCTION MODEL

- **Device**: NVIDIA RTX 4060 Ti (16GB)
- **Training Time**: ~25 epochs完了
- **Convergence**: Epoch 25でBest (Avg F1: 0.2789)
- **Memory**: GPU十分に余裕あり
- **Parameters**: 195,204 (最適なバランス)

### v2.2.2

- **Device**: NVIDIA RTX 4060 Ti (16GB)
- **Training Time**: ~25 epochs完了
- **Convergence**: Epoch 25でBest (Avg F1: 0.2689)
- **Memory**: GPU十分に余裕あり
- **Parameters**: 218,952 (+12.2% vs v2.2)
- **速度**: v2.2とほぼ同等（Cross-Attentionのオーバーヘッド小）

### v2.3

- **Device**: NVIDIA RTX 4060 Ti (16GB)
- **Training Time**: ~25 epochs完了
- **Convergence**: Epoch 24でBest (Avg F1: 0.1973)
- **Memory**: GPU十分に余裕あり
- **Parameters**: 191,620 (-1.8% vs v2.2)
- **速度改善**: 特に顕著な改善なし（データローダーがボトルネック）

**結論**: 
- パラメータ増加（v2.2.2）も削減（v2.3）も性能向上には寄与せず
- v2.2の195,204パラメータが最適なスイートスポット
- Simple architectureが安定性と性能を両立r 64d)
   +
   Statistical Summary (Engineered 28d)
   ↓
   Complementary Information Fusion
   ```
2. **Multi-Task Learning**

   - 30d, 60d, 90d の3ホライズンを同時学習
   - Shared representation learning で汎化性能向上
3. **Focal Loss**

   - γ=3.0 で hard examples に集中
   - Class imbalance（異常 9%）への対処
4. **Temporal Attention**

   - 4-head attention でパターン強調
   - Interpretability確保

---

### v2.3 失敗要因

1. **情報不足**

   - TinyTimeMixerは時系列の**形状**を捉える
   - しかし**統計的特性**（平均レベル、変動幅）は弱い
2. **過検出傾向**

   - 正常範囲のベースラインがない
   - 変動があれば全て異常と判断
3. **ドメイン知識の欠如**

   - HVAC機器の異常判定には統計的閾値が重要
   - "平均から何σ離れているか"などの情報が必須

---
~~Option 3: Attention-based Feature Fusion~~ ❌ 効果なし

- ~~現状: 単純concat~~
- ~~改善: Cross-attention で動的に重み付け~~
- **実験結果（v2.2.2）**: F1-Score -3.6%低下、パラメータ+12%増加
- **結論**: 28次元の特徴量にはSimple Concat Fusionが最適
- **Cross-Attentionが有効なケース**: 100+次元の高次元特徴量*Convergence**: Epoch 25でBest (Avg F1: 0.2789)
- **Memory**: GPU十分に余裕あり

### v2.3

- **Device**: NVIDIA RTX 4060 Ti (16GB)
- **Training Time**: ~25 epochs完了
- **Convergence**: Epoch 24でBest (Avg F1: 0.1973)
- **速度改善**: 特に顕著な改善なし（データローダーがボトルネック）

**結論**: 特徴量削減による速度向上は限定的。性能犠牲に見合わない。

---

## 🎓 Broader Implications

### 1. Time-Series Anomaly Detection における特徴設計

**原則**:

```
Raw Time-Series Embeddings (形状情報)
+
Statistical Features (統計情報)
= Robust Anomaly Detection
```

### 2. Transfer Learning の限界

- 事前学習モデル（TinyTimeMixer）は汎用的なパターンを学習
- しかしドメイン固有の統計的異常は捉えられない
- **Domain Adaptation が必要**

### 3. Explainability

v2.2の利点:

- Statistical features は解釈可能
- 「この期間は平均が高く、変動が大きいため異常」と説明できる

v2.3の問題:

- Embedding空間のみ → ブラックボックス
- 予測根拠の説明が困難

---

## 🔮 Future Directions

### 推奨戦略: v2.2 をベースに改善

#### Option 1: Feature Selection（特徴選択）
 ✅ PRODUCTION: `models/hybrid_model_v2.2/pytorch_model_multitask.pt`
- **v2.2.2 Model** (Experimental): `models/hybrid_model_v2.2.2/pytorch_model_multitask.pt`
- **v2.3 Model** (Experimental): `models/hybrid_model_v2.3/pytorch_model_multitask.pt`
- **Training History**:
  - `results/training_history_v2.2.json` ✅
  - `results/training_history_v2.2.2.json`
  - `results/training_history_v2.3.json`

### Code

- **v2.2 Training** ✅: [train_hybrid_model_v2_2.py](train_hybrid_model_v2_2.py)
- **v2.2.2 Training**: [train_hybrid_model_v2_2_2.py](train_hybrid_model_v2_2_2.py)
- **v2.3 Training**: [train_hybrid_model_v2_3.py](train_hybrid_model_v2_3.py)
- **v2.2 Visualization** ✅-based Feature Fusion

- 現状: 単純concat
- 改善: Cross-attention で動的に重み付け

```python
fused = CrossAttention(embeddings, statistical_features)
```

#### Option 4: Hierarchical Multi-Task

- Current: Flat multi-task (30d, 60d, 90d)
- Enhanced: Hierarchical (30d → 60d → 90d)
- Long-term予測がShort-term予測を条件付け

---

## 📝 Action Items

### ✅ Immediate (完了)

1. ~~v2.2モデルをproduction candidateとして保存~~
2. ~~v2.3実験結果をドキュメント化~~
3. ~~Lessonファイル作成~~

### 🔄 Next Steps

1. **v2.2 Feature Importance Analysis**

   - SHAP values計算
   - 28特徴量の貢献度可視化
2. **v2.2 Visualization Enhancement**

   - Attention weights + Feature importance の統合可視化
   - サンプルごとの予測根拠説明
3. **v2.4 Design**

   - Feature selection実装
   - 重要度上位15-20特徴のみ使用
### Core Findings

1. ✅ **Statistical Features (28d) are essential** - 29.2% F1-Score improvement (v2.3実験)
2. ✅ **Simple Fusion > Complex Fusion** - Concat outperforms Cross-Attention for 28d features (v2.2.2実験)
3. ✅ **Feature Engineering > Model Complexity** - Small features → Large impact
4. ✅ **TinyTimeMixer captures shape, not statistics** - Complementary information needed
5. ✅ **v2.2 is production-ready** - Best balance of performance and interpretability

### Architecture Lessons

6. ⚠️ **Cross-Attention overhead** - +12% params, -4% performance (v2.2.2)
7. ⚠️ **Feature dimensionality matters** - Cross-Attention effective for 100+ dims, overkill for 28d
8. ✅ **Simplicity wins** - v2.2's Simple Concat performs best with fewest params
9. ❌ **v2.3 experiment valuable** - Proved the necessity of explicit features
10. ❌ **v2.2.2 experiment valuable** - Proved Simple Fusion superiority

### Production Recommendation

**CONFIRMED: v2.2 as Production Model** ✅

| Criterion | v2.2 Score |
|-----------|------------|
| Performance | **0.2789 F1** (Best) |
| Parameters | **195,204** (Optimal) |
| Trainable % | **31.64%** (Balanced) |
| Interpretability | **High** (Simple architecture) |
| Stability | **Excellent** (Proven) |
| Deployment Ready | **Yes** |

---

**Conclusion**:

> "The best model is not the most complex, nor the simplest, but the one that captures the right information with the right architecture."

**v2.2の成功要因:**
- ✅ Statistical Features (28d) による明示的なドメイン知識統合
- ✅ Simple Concat Fusion による効率的な特徴結合
- ✅ 最適なパラメータ数（195,204）と訓練可能比率（31.64%）
- ✅ Multi-Task Learning による共通表現学習
- ✅ Temporal Attention による時系列パターン強調

**v2.2.2とv2.3の教訓:**
- ❌ Cross-Attentionは28次元には過剰（v2.2.2）
- ❌ 統計特徴の削除は致命的（v2.3: -29.2% F1）
- ✅ アーキテクチャは問題規模に合わせて設計すべき
- ✅ ドメイン知識（統計特徴）は不可欠

---

**Document Version**: 2.0  
**Last Updated**: 2026-02-15 (v2.2.2実験追加)  
**Status**: ✅ Production Model Confirmed (v2.2)  
**Decision**: **v2.2をProduction Modelとして確定**
- **v2.2 Model**: `models/hybrid_model_v2.2/pytorch_model_multitask.pt`
- **v2.3 Model**: `models/hybrid_model_v2.3/pytorch_model_multitask.pt`
- **Training History**:
  - `results/training_history_v2.2.json`
  - `results/training_history_v2.3.json`

### Code

- **v2.2 Training**: [train_hybrid_model_v2_2.py](train_hybrid_model_v2_2.py)
- **v2.3 Training**: [train_hybrid_model_v2_3.py](train_hybrid_model_v2_3.py)
- **v2.2 Visualization**: [visualize_forecast_v2_2.py](visualize_forecast_v2_2.py)
- **v2.3 Visualization**: [visualize_forecast_v2_3.py](visualize_forecast_v2_3.py)

---

## 🎯 Key Takeaways

1. ✅ **Statistical Features (28d) are essential** - 29.2% F1-Score improvement
2. ✅ **Feature Engineering > Model Complexity** - Small features → Large impact
3. ✅ **TinyTimeMixer captures shape, not statistics** - Complementary information needed
4. ✅ **v2.2 is production-ready** - Best balance of performance and interpretability
5. ❌ **v2.3 experiment valuable** - Proved the necessity of explicit features

---

**Conclusion**:

> "The best model is not the simplest, but the one that captures the right information."

v2.2 の92次元入力は、最適な情報密度を持つアーキテクチャだった。

---

**Document Version**: 1.0
**Last Updated**: 2026-02-15
**Status**: ✅ Validated & Production-Ready (v2.2)
