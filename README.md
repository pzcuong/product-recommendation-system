# Baby Product Recommendation System

**Competition:** Rental Product Recommendation System  
**Best Kaggle Score:** 0.33198 (Top ~10)  
**Task:** Next-item prediction with Recall@6

---

## 📁 Project Structure

```
recommended-system/
├── main.py                    # Main recommender (best model)
├── submission.csv             # Best submission (Score: 0.33198)
├── EXPERIMENT_SUMMARY.md      # Complete analysis of all experiments
├── embeddings/                # Skip-gram embeddings (553 products)
├── data/                      # Training data
└── train_embeddings.py        # Script to train Skip-gram embeddings
```

---

## 🚀 Quick Start

### Generate Submission:
```bash
python main.py
```

This will:
1. Train all models on recent data (2025-07-01+)
2. Generate predictions for test set
3. Save to `submission.csv`

---

## 🏆 Best Model Configuration

**Ensemble Heuristic Approach:**
- **Sequential Transitions:** weight 3.0  
- **Co-occurrence:** weight 2.0  
- **Skip-gram Semantic:** weight 2.5  
- **Order Boost:** weight 2.0  
- **Last Item Boost:** 2.0  

**Key Features:**
- ✅ Recency filter (2025-07-01+) - CRITICAL
- ✅ No deep learning (data too sparse)
- ✅ Simple beats complex on sparse data

---

## 📊 Results Summary

| Approach | Kaggle Score | Status |
|----------|--------------|--------|
| **Baseline Heuristic** | **0.33198** | ✅ **BEST** |
| XGBoost Reranker | 0.19973 | ❌ Overfitting |
| Deep Learning (SASRec, GRU) | - | ❌ Failed |
| Category Expansion | 0.33198 | = Same |
| Cross-Category Boost | 0.32456 | ❌ Hurt |
| BERT Embeddings | - | ❌ Lower recall |
| TF-IDF NLP | - | ❌ Lower recall |

**Total experiments:** 25+  
**All failed to beat baseline**

See [EXPERIMENT_SUMMARY.md](./EXPERIMENT_SUMMARY.md) for complete details.

---

## 🔍 Key Insights

### Why Simple Heuristic Wins:
1. **Data Sparsity:** Only 3.7k sessions, 332 products
2. **Test Sessions 2.5x Longer:** Mean 10.8 vs 4.4 in train
3. **56% Missing Data:** Correct items have no training signal
4. **Behavioral > Text:** Skip-gram on sessions beats NLP

### What Works:
- ✅ Recency filtering (concept drift)
- ✅ Sequential transitions (A→B patterns)
- ✅ Order boost (actual purchases)
- ✅ Skip-gram embeddings trained on sessions

### What Doesn't Work:
- ❌ Deep learning (data too sparse)
- ❌ Pre-trained embeddings (BERT, TF-IDF)
- ❌ Complex ensembles (RRF, CF)
- ❌ Metadata features (brand, category)
- ❌ Old site data (adds noise)

---

## 📈 Performance Analysis

**Root Cause of 0.33198 Ceiling:**
- 56.4% of ground truth items NOT in Top 100 candidates
- These items have zero training data (no transitions, co-occurrence)
- No ML can predict without signals

**To Break 0.40+ (Top Leaderboard):**
- Likely needs external data or domain knowledge
- Manual product bundles (stroller + car seat)
- Age progression rules (0-6mo → 6-12mo)
- Physical compatibility databases

---

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
tqdm
gensim (for Skip-gram)
```

---

## 📝 License

Competition code - for educational purposes

---

## 🙏 Acknowledgments

- Kaggle OTTO competition (inspiration for co-visitation)
- RecSys literature on sparse data handling
- Baby product rental domain insights
