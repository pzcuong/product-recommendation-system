# Rental Product Recommendation System

## Best Score: 0.33198 (Kaggle Recall@6)

---

## 🏆 Best Pipeline

### 1. Data Filtering
```
RECENCY_CUTOFF = '2025-07-01'
```
- **Chỉ sử dụng data từ 2025-07-01 trở đi**
- Test data từ cùng thời kỳ → alignment tốt
- Old data (2021-2024) **HURT** performance badly

### 2. Training Models

#### Model 1: Sequential Transitions (weight: 3.0)
```python
transitions[product_A][product_B] += 1  # A → B patterns
```
- Học từ sequences: user xem A rồi xem B
- Boost cho items gần cuối session (last_item_boost)

#### Model 2: Co-occurrence (weight: 2.0)
```python
cooccur[A][B] += 1  # A và B xuất hiện cùng session
```
- Học từ co-occurrence trong cùng session
- Không quan tâm thứ tự

#### Model 3: Semantic Similarity (weight: 2.5)
```python
# Pre-trained embeddings via Skip-gram
similarity = dot(embed[viewed], embed[candidate])
```
- Embeddings học từ session sequences
- Tìm products tương tự về semantic

#### Model 4: Order Boost (weight: 2.0)
```python
# Boost candidates that were actually purchased
if pid in candidate_scores:
    candidate_scores[pid] += 2.0 * min(order_count / 10.0, 1.0)
```
- **CHỈ boost existing candidates** (không add new!)
- Products đã được mua → signal tốt

### 3. Scoring Pipeline

```
For each test session:
1. Get viewed products from session
2. Score candidates:
   - Transitions: 3.0 × position_weight × decay
   - Co-occurrence: 2.0 × decay
   - Semantic: 2.5 × decay
   - Order boost: 2.0 × (chỉ existing candidates)
3. Fallback nếu thiếu:
   - Category popularity (0.5)
   - Global popularity
4. Return top 6
```

### 4. Key Insights

| Factor | Impact |
|--------|--------|
| Recency filter | **CRITICAL** - old data hurts |
| Order boost | +0.8% improvement |
| Add ordered as candidates | **HURT** (-8.4%) |
| Use ALL data | **HURT** (-13.6%) |
| Transformer | Không tốt bằng heuristic |

---

## 📁 File Structure

```
recommended-system/
├── main.py              # Main pipeline (BEST CONFIG)
├── train_embeddings.py  # Train Skip-gram embeddings
├── gru4rec.py           # GRU4Rec model (unused)
├── evaluate_local.py    # Local validation
├── optimize_weights.py  # Weight optimization
├── embeddings/          # Pre-trained embeddings
├── data/                # Input data
├── submission.csv       # Best submission (0.33198)
└── overview.md          # This file
```

---

## 🚀 How to Run

```bash
# 1. Train embeddings (optional - already provided)
python train_embeddings.py --epochs 10 --dim 64

# 2. Generate submission
python main.py

# Output: submission.csv
```

---

## ⚠️ What NOT to Do

1. **KHÔNG** dùng data trước 2025-07-01 cho training
2. **KHÔNG** thêm ordered products trực tiếp vào candidates
3. **KHÔNG** trust local validation 100% (correlation ~50%)
4. **KHÔNG** dùng Transformer/Deep Learning với data ít

---

## 📊 Experiments Summary

| Experiment | Local | Kaggle | Note |
|------------|-------|--------|------|
| Baseline | 0.157 | 0.324 | - |
| + Recency filter | 0.157 | 0.326 | ✓ |
| + Order boost 2.0 | 0.170 | **0.332** | ✓ Best |
| + Grid search | 0.176 | 0.331 | ✗ |
| + Co-purchase | 0.170 | 0.331 | ✗ |
| Hybrid (all data) | 0.173 | 0.196 | ✗ Bad |
| Cold-start fix | 0.170 | 0.248 | ✗ Bad |
