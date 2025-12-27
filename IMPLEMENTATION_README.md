# Advanced Recommendation System Implementation

## Overview

This implementation follows the strategic framework document to build a state-of-the-art session-based recommendation system using transfer learning, BERT embeddings, and rental-specific features.

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Evaluate Baseline

The current heuristic system achieves **0.33616 Recall@6**:

```bash
python evaluate.py --submission submission.csv --ground_truth "95)submission.csv"
```

### 3. Generate BERT Embeddings (Coming Soon)

```bash
python generate_embeddings.py --output embeddings_bert/products.parquet
```

### 4. Train UniSRec Model (Coming Soon)

```bash
python train_unisrec.py --config config.yaml --epochs 20
```

### 5. Generate Advanced Submission (Coming Soon)

```bash
python submit.py --model unisrec --output submission_advanced.csv
```

## Project Structure

```
product-recommendation-system/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
│
├── src/                        # Core library
│   ├── utils.py               # Utilities (metrics, helpers)
│   ├── text_encoder.py        # BERT encoding & projection
│   ├── data_processor.py      # Data loading & preprocessing
│   ├── candidate_retrieval.py # Hybrid candidate generation
│   └── models/
│       └── unisrec.py         # UniSRec transformer model
│
├── evaluate.py                 # Evaluation script
├── train_unisrec.py           # Training script (TODO)
├── submit.py                  # Submission generator (TODO)
│
├── embeddings_bert/           # BERT embeddings cache
├── data/                      # Training data
└── tests/                     # Unit tests (TODO)
```

## Implementation Status

### ✅ Completed (Phases 1-2)

1. **Configuration Management**
   - YAML-based configuration
   - Centralized hyperparameters

2. **Utilities**
   - Metric calculation (Recall@K, NDCG@K, Coverage)
   - Early stopping
   - Submission formatting

3. **BERT Text Encoder**
   - Multilingual DistilBERT support
   - Embedding caching (parquet format)
   - Learnable projection layers
   - Mixture-of-Experts adapter for domain adaptation

4. **Data Processor**
   - Old/New site product mappings
   - Session sequence building
   - Rental feature extraction
   - Co-occurrence & transition matrices
   - Time-based train/val splits

5. **Candidate Retrieval**
   - FAISS semantic search
   - Co-visitation matrices
   - Sequential transitions
   - V-SKNN session similarity
   - Hybrid retrieval with configurable weights
   - Reciprocal Rank Fusion

6. **UniSRec Model**
   - Transformer architecture with causal attention
   - MoE projection for BERT embeddings
   - Positional encoding
   - Multi-head self-attention
   - Sampled softmax loss

7. **Evaluation Framework**
   - Ground truth comparison
   - Recall@6, NDCG@6, Coverage metrics
   - Failure analysis
   - Multi-model comparison

### 🚧 In Progress (Phase 3)

- UniSRec training pipeline
- BERT embedding generation for all products
- Negative sampling implementation

### 📝 Planned (Phases 4-6)

- LightGBM reranker
- Ensemble optimization
- Final submission generation
- Kaggle submission

## Key Design Decisions

### 1. Addressing Data Sparsity

**Problem:** Previous deep learning attempts failed (SASRec: 0.0035 Recall@6)

**Solutions:**
- Shallow transformer (2 blocks instead of 4-8)
- High dropout (0.3)
- Small batch size with gradient accumulation
- Strong regularization (weight decay 0.01)
- Sampled softmax with 2048 negatives
- Early stopping (patience=5)

### 2. Domain Adaptation (Old → New Site)

**Problem:** Test data is from new site, training includes old site

**Solutions:**
- BERT embeddings (transfer semantic knowledge)
- Mixture-of-Experts projection (learns site-specific patterns)
- Recency filter (2025-07-01+) for training
- Validation on new site sessions only

### 3. Rental-Specific Features

Implemented but not yet integrated into training:
- Days since return (temporal urgency)
- Rental duration intent (travel vs developmental)
- Age progression delta (biological appropriateness)
- Seasonal matching (outdoor items in summer)

### 4. Safety Net Strategy

Keep proven heuristics as fallback:
- Baseline submission available
- Hybrid ensemble planned
- RRF for combining rankings

## Current Performance

| Model | Recall@6 | NDCG@6 | Coverage | Status |
|-------|----------|---------|----------|---------|
| **Baseline Heuristic** | **0.33616** | 0.21942 | 1.28 | ✓ Working |
| UniSRec (planned) | TBD | TBD | TBD | 🚧 In progress |
| Hybrid Ensemble (planned) | TBD | TBD | TBD | 📝 Planned |

## Next Steps

1. **Generate BERT Embeddings**
   - Run on all 665 products (old + new sites)
   - Cache to `embeddings_bert/products.parquet`
   - Estimated time: 2-3 minutes

2. **Implement Training Pipeline**
   - Data loader for sessions
   - Negative sampling (random + hard)
   - Training loop with validation
   - Checkpoint saving

3. **Train UniSRec**
   - 20 epochs with early stopping
   - Monitor validation Recall@6
   - Target: > 0.20 local recall

4. **Evaluate & Compare**
   - Run evaluation
   - Compare with baseline
   - Analyze failures

5. **Generate Submissions**
   - Baseline (sanity check)
   - Advanced (UniSRec)
   - Hybrid (ensemble)

## Known Challenges

1. **Data Sparsity**: Only 3.7k sessions, 332 products
   - Mitigation: Regularization, transfer learning

2. **Missing Signals**: 56.4% of ground truth items not in Top-100 candidates
   - Mitigation: Better semantic understanding via BERT

3. **Local-Kaggle Gap**: Test sessions 2.5x longer than train
   - Mitigation: Validate on similar long sessions

4. **Computational Cost**: BERT embeddings + FAISS + Transformer
   - Mitigation: Caching, quantization, batch processing

## References

- Strategic Framework Document (provided)
- UniSRec Paper: "Towards Universal Sequence Representation Learning" 
- SASRec Paper: "Self-Attentive Sequential Recommendation"
- Kaggle OTTO Competition (co-visitation approach)
- EXPERIMENT_SUMMARY.md (25+ previous attempts documented)

## Contact

For questions or issues, refer to the implementation plan and task breakdown artifacts.
