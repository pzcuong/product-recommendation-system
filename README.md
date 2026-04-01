# CL-GRU4Rec+RP: Product Recommendation System

**Contrastive Learning Enhanced GRU4Rec with Re-Purchase Awareness**

---

## 🎯 Method Overview

CL-GRU4Rec+RP combines three novel components for session-based product recommendation:

1. **GRU4Rec**: Sequential pattern learning with clean PyTorch implementation
2. **Contrastive Learning**: Item similarity discovery via session co-occurrence
3. **Re-Purchase Awareness**: Behavioral signal for repeated product interactions

With **Adaptive Two-Stage Fusion** for dataset-specific optimization.

---

## 📊 Datasets Supported

- **Kaggle Rental Product**: Rental product recommendation
- **Synerise RecSys 2025**: E-commerce session-based recommendation

---

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch pandas numpy tqdm
```

### Training & Evaluation

```bash
# Kaggle Rental Product
python cl_gru4rec_rp_unified.py --dataset rental

# Synerise RecSys 2025
python cl_gru4rec_rp_unified.py --dataset synerise
```

### Interactive Demo

```bash
# Run demo on Kaggle Rental data
python demo.py --dataset rental

# Run demo on Synerise data
python demo.py --dataset synerise
```

The demo showcases:
- **Single Session Recommendation**: Full explanation of each recommendation
- **Session Progression**: How recommendations evolve as user interacts
- **Cold Start Handling**: Fallback strategies for new users

---

## 📁 Project Structure

```
product-recommendation-system/
├── cl_gru4rec_rp_unified.py    # Main model (unified PyTorch)
├── demo.py                       # Interactive demo script
├── data/                         # Kaggle rental data
│   ├── metrika_hits.csv
│   ├── metrika_visits.csv
│   └── ...
├── synerise_dataset/             # Synerise RecSys data
│   ├── add_to_cart.parquet
│   ├── product_buy.parquet
│   └── ...
├── synerise_final.pkl            # Cached Synerise data
├── docs/                         # Academic documentation
│   ├── BAO_CAO_HOC_LUAN.md       # Pure academic report (recommended)
│   ├── BAO_CAO_DO_AN_CL_GRU4REC_RP.md
│   ├── SLIDE_DECK.md
│   └── README.md
└── submission.csv                # Final predictions
```

---

## 🔬 Architecture

### Component 1: GRU4Rec

```python
class GRU4RecModel(nn.Module):
    - Embedding dim: 128
    - Hidden dim: 200
    - Dropout: 0.15
    - Loss: Cross-entropy
    - Ensemble: 3 seeds (42, 123, 456)
```

### Component 2: Contrastive Learning

```python
class ContrastiveItemModel(nn.Module):
    - Embedding dim: 64
    - Temperature: 0.07
    - Loss: InfoNCE
    - Positive pairs: Session co-occurrence
    - Negative samples: 256 per batch
```

### Component 3: Re-Purchase Awareness

```python
# Stage 1: RP scoring (dominant for repeat-purchase data)
for item, event in user_history:
    weight = 5.0 if event == "buy" else 2.0
    recency_boost = 1.0 + (position / len(history))
    rp_score[item] += weight * recency_boost
```

### Adaptive Two-Stage Fusion

```
Stage 1: RP fills slots (buy-boosted + recency)
    ↓
Stage 2: Discovery fills remaining slots
    - Co-occurrence patterns
    - CL similarity embeddings
    - GRU sequential scores
```

---

## 📈 Performance

### Kaggle Rental Product

- **Recall@6**: 0.0XXX
- **NDCG@6**: 0.0XXX
- **Hit Rate@6**: 0.0XXX

### Synerise RecSys 2025

- **Recall@6**: 0.0XXX
- **NDCG@6**: 0.0XXX
- **Hit Rate@6**: 0.0XXX

---

## 🎓 Key Innovations

1. **Separate Training**: GRU and CL trained independently, combined at inference
2. **Adaptive Fusion**: Dataset-specific signal combination (RP for Synerise, GRU for Rental)
3. **Session-Adaptive**: Fallback strategies for cold-start sessions
4. **Per-User Split**: Academic-standard 80/20 per-user evaluation (not time-based)

---

## 📝 Documentation

See `docs/` folder for:

- **BAO_CAO_HOC_LUAN.md**: Pure academic report (30-40 pages, no code)
- **SLIDE_DECK.md**: 20-slide presentation for defense

---

## ⚙️ Configuration

Edit `cl_gru4rec_rp_unified.py` to modify:

```python
# GRU config
GRU_EMBED_DIM = 128
GRU_HIDDEN_DIM = 200
GRU_DROPOUT = 0.15
GRU_EPOCHS = 25

# CL config
CL_EMBED_DIM = 64
CL_EPOCHS = 25
CL_TEMP = 0.07

# Fusion
K = 6  # Number of recommendations
```

---

## 🔄 Reproducibility

- Fixed seeds: 42, 123, 456
- Deterministic CUDA operations
- Version-controlled data splits

---

## 📊 Evaluation Metrics

- **Recall@K**: Fraction of ground truth items in top-K
- **NDCG@K**: Normalized discounted cumulative gain
- **Hit Rate@K**: Binary success metric

For Synerise, extended metrics:

- **Novelty**: 1 - popularity^100 (competition formula)
- **Diversity**: Entropy of recommendation distribution
- **Coverage**: Catalog coverage percentage

---

## 🚀 Future Work

1. **Explainable AI**: Interpretability for fusion decisions
2. **Real-time API**: Production-ready inference
3. **Seasonal Modeling**: Time-aware recommendations
4. **Multi-Behavior**: Extend beyond cart/buy events

---

## 📄 License

Academic project for RecSys 2025 competition.

---

**Last Updated**: April 2025
**Model**: CL-GRU4Rec+RP v4 (Unified PyTorch)
