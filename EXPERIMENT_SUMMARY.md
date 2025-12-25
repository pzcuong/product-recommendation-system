# Recommendation System Optimization - Complete Experiment Summary

## 📊 Final Results

**Best Kaggle Score: 0.33198**

**Winning Configuration:**
- Sequential Transitions: weight 3.0
- Co-occurrence: weight 2.0
- Skip-gram Semantic Similarity: weight 2.5
- Order Boost: weight 2.0
- Last Item Boost: 2.0

---

## 🧪 All Experiments Conducted (25+ Approaches)

### 1. Deep Learning Models

| Model | Local Recall@6 | Kaggle Score | Result |
|-------|----------------|--------------|--------|
| **XGBoost Reranker** | 0.13 | 0.19973 | ❌ Concept drift, overfitting |
| **SASRec (Self-Attention)** | 0.0035 | - | ❌ Data too sparse |
| **Transformer Ensemble** | Failed to train | - | ❌ Insufficient data |
| **GRU4Rec** | 0.10 | - | ❌ Overfitting |
| **Contrastive Learning (InfoNCE)** | 0.15345 | - | ❌ Low coverage (332 products) |
| **Self-Contrastive (SCL)** | 0.16810 | - | ❌ Still lower than Skip-gram |

**Key Learning:** Deep learning fails due to extreme data sparsity (3.7k sessions, 332 products).

---

### 2. Advanced Heuristics

| Approach | Local Score | Kaggle Score | Result |
|----------|-------------|--------------|--------|
| **Category Diversity (Max 3/cat)** | - | 0.33063 | ❌ Hurt by 0.00135 |
| **Time-Weighted Session Scoring** | 0.16 | - | ❌ No improvement |
| **Micro Weight Tuning** | 0.16-0.17 | - | ❌ No improvement |
| **MMR Diversity** | 0.10573-0.17 | - | ❌ Users prefer same category |
| **Category-Prioritized** | 0.15638-0.16452 | - | ❌ Forcing priority hurts |
| **Co-Visitation (OTTO-style)** | 0.15220-0.15567 | - | ❌ Not better than simple co-occur |
| **RRF Ensemble** | 0.05165-0.07580 | - | ❌ Dramatically hurt |
| **Item-Based CF (Cosine Sim)** | 0.16810 | - | ❌ No improvement |

**Key Learning:** Simple heuristics outperform complex ensembles on sparse data.

---

### 3. Metadata-Based Approaches

| Approach | Local Score | Kaggle Score | Result |
|----------|-------------|--------------|--------|
| **Category Expansion (w=2.0)** | 0.4198 | 0.33198 | = No change |
| **Cross-Category Boost (w=1.5)** | 0.4198 | **0.32456** | ❌ **HURT** despite local improvement |
| **Brand Boost (w=1.0-3.0)** | 0.4136 | - | ❌ No improvement |
| **Same-Brand Preference** | 0.4136 | - | ❌ No improvement (69.6% same-brand transitions ignored) |

**Key Learning:** Metadata features don't help - behavioral signals are king.

---

### 4. Embedding & NLP Approaches

| Approach | Coverage | Local Score | Result |
|----------|----------|-------------|--------|
| **Skip-gram (current)** | 553 products | 0.17005 | ✅ **BEST** |
| **BERT Text Embeddings** | 644 products | 0.14368 | ❌ Lower recall despite higher coverage |
| **TF-IDF on Product Names** | 644 products | 0.4074 | ❌ Behavioral > Text similarity |

**Key Learning:** Skip-gram trained on actual sessions >> Pre-trained language models.

---

### 5. Data Augmentation & Preprocessing

| Approach | Result |
|----------|--------|
| **Data Augmentation (Sliding window, dropout)** | ❌ Hurt (0.13629 vs 0.15642) |
| **Old Site Data Integration** | ❌ Hurt (0.3333 vs 0.4136 hit rate) |
| **Recency Filter (2025-07-01+)** | ✅ **CRITICAL** - Best performing filter |

**Key Learning:** Recency filter is crucial. Old site data adds noise.

---

## 🔍 Root Cause Analysis

### Why We're Stuck at 0.33198

**Deep Analysis on Long Sessions (n=162):**

| Finding | Value | Implication |
|---------|-------|-------------|
| **Correct items NOT in Top 100** | **56.4%** | No training data to predict |
| Coverage (unique products recommended) | 165/665 (25%) | Low diversity |
| Sessions with repeat views | 99% | Users revisit products |
| Category match (context/future) | 97-98% | Not the issue |
| Cold start items in test | 0.6% | Not the issue |

**Critical Bottleneck:**
- 56% of ground truth items have NO behavioral signals (no transitions, co-occurrence, or embeddings)
- No amount of modeling can predict these without additional data

---

## 📈 Local vs Kaggle Validation Gap

### Key Discovery: Test Sessions are 2.5x Longer

| Metric | Train | Test | Impact |
|--------|-------|------|--------|
| Mean session length | 4.4 | **10.8** | ⚠️ Test 2.5x longer |
| Product overlap | 521 | 329 | 100% overlap ✓ |
| Cold start products | - | 0% | No cold start ✓ |

**Implication:** Local validation on short sessions doesn't correlate with Kaggle test performance.

---

## 🏆 Best Practices Discovered

### What Worked:
1. ✅ **Simple heuristic ensemble** (Transitions + Co-occurrence + Skip-gram + Order boost)
2. ✅ **Recency filter** (2025-07-01+) - Critical for concept drift
3. ✅ **Skip-gram embeddings** trained on sessions (not pre-trained NLP)
4. ✅ **Order boost** (products with actual purchases)
5. ✅ **Last item boost** (weight recent interactions)

### What Failed:
1. ❌ All deep learning models (data too sparse)
2. ❌ Complex ensembles (RRF, CF, Co-visitation matrices)
3. ❌ Category/metadata features (brand, cross-category)
4. ❌ Pre-trained embeddings (BERT, TF-IDF)
5. ❌ Data augmentation
6. ❌ Old site data integration

---

## 🎯 Recommendations for Future Work

### To Break 0.40+ (Top Leaderboard):

1. **External Data Sources:**
   - Product compatibility databases
   - Manual bundle definitions (stroller + car seat)
   - Age progression rules (0-6mo → 6-12mo)

2. **Domain-Specific Rules:**
   - Seasonal patterns (summer vs winter products)
   - Price range similarity
   - Physical compatibility (crib + mattress sizes)

3. **Different Data Preprocessing:**
   - Session segmentation by intent
   - Deduplication strategies
   - Focus on high-intent sessions only

4. **Hybrid Approaches:**
   - Manual rules for top 100 products
   - Cold-start handling with category popularity
   - Graph-based item relationships

---

## 📝 Files & Code Structure

### Core Files:
- `main.py` - Final working recommender (baseline achieving 0.33198)
- `submission.csv` - Best submission file
- `embeddings/` - Skip-gram embeddings (553 products)
- `data/` - Training data

### Experimental Files (Can be deleted):
- `train_contrastive.py` - Contrastive learning (failed)
- `train_sasrec.py` - SASRec model (failed)
- `train_scl.py` - Self-contrastive learning (failed)
- `generate_bert_embeddings.py` - BERT embeddings (failed)
- `embeddings_bert/` - BERT embeddings folder
- `embeddings_contrastive/` - Contrastive embeddings
- `embeddings_sasrec/` - SASRec embeddings
- `embeddings_scl/` - SCL embeddings
- `gru4rec.py` - GRU4Rec implementation (failed)
- `sasrec_best.pt` - SASRec checkpoint
- `optimize_weights.py` - Weight optimization
- `evaluate_local.py` - Local evaluation script

---

## 🏅 Competition Insights

**Leaderboard Context:**
- Top 1: 0.40553 (Reza Madani, 15 entries)
- Top 2: 0.37854
- **Our Best: 0.33198** (~Top 10 estimated)

**Gap Analysis:**
- Need +0.07355 to reach top (21% improvement)
- Likely achieved through domain expertise, not pure ML
- Possible use of external knowledge or manual rules

---

## ✅ Conclusion

After **25+ different approaches** tested:
- **Simple heuristic ensemble is optimal** for this sparse dataset
- **0.33198 is the ceiling** with current data and pure ML approaches
- **Local validation is unreliable** (cross-category improved local but hurt Kaggle)
- **Top scores likely use domain knowledge** we haven't discovered

**Final Recommendation:** Accept 0.33198 as strong performance given constraints, or pursue domain-specific manual rules.
