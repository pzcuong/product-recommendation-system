# CEARF-N: Bounded Out-of-Fit Dynamic Rank Allocation for Sparse Next-Item Recommendation

CEARF-N is a memory–neural rank-fusion system for session-based next-item
recommendation. It combines three retrieval memories (transition, session,
popularity) with a neural residual (PASGR, a 1-layer GRU) via a bounded linear
gate that predicts the per-query memory/neural blend coefficient β from 3
target-free features (session length, last-item frequency, tail flag), trained
on out-of-fit (OOF) training data without validation labels for mixing.

## Method

```
┌────────────────────────────────────────────────────────────────┐
│ TRAINING (per domain)                                          │
├────────────────────────────────────────────────────────────────┤
│ 1. Build CEARF memory index                                    │
│    • transition (recency-weighted, window=4)                   │
│    • session neighbours (IDF-weighted, 120 candidates)         │
│    • popularity fallback                                       │
│ 2. Train PASGR neural residual                                 │
│    • 1-layer GRU, dim=64, prototype-aligned teacher            │
│ 3. OOF holdout split (source-disjoint, 20%)                    │
│ 4. Learn bounded gate on OOF data                              │
│    • features: log(context_len), log(freq), tail_flag          │
│    • output: β_q ∈ [0,1]                                       │
│ 5. Freeze gate + experts                                       │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ INFERENCE                                                     │
├────────────────────────────────────────────────────────────────┤
│ 1. Memory ranking  M (CEARF, 120 items)                        │
│ 2. Neural ranking  N (PASGR, 120 items)                        │
│ 3. β_q = gate(features)                                        │
│ 4. Fuse: score(x) = (1-β_q)/(20+rank_M) + β_q/(20+rank_N)      │
│ 5. Return top-20                                               │
└────────────────────────────────────────────────────────────────┘
```

**Primary outcome:** `U = 0.5·R@6 + 0.5·R@20`.

## Results (Recall@20, 3 seeds)

| Domain | CEARF-N | Memory | Neural | NARM | SR-GNN | GRU4Rec |
|---|---:|---:|---:|---:|---:|---:|
| Video Games | **.146** | .119 | .126 | .137 | .123 | .104 |
| Baby Products | **.054** | .039 | .045 | .030 | .052 | .045 |
| Diginetica | **.490** | — | — | .483 | .432 | .417 |

CEARF-N achieves the best R@20 on all three domains, with paired bootstrap
CIs (20,000 reps, query-level) excluding zero against every ID-only baseline
and both constituent endpoints on Amazon. On Diginetica, the continuous
router is detectably better than both the regime and bucketed routers
(p < .001).

## Repository layout

```
sparse_bench/
├── cearf.py                    # Memory index (transition/session/popularity)
├── pasgr.py                    # Neural residual
├── run_cearfn_evidence.py      # CEARF-N evidence suite (Amazon)
├── run_cearfn_v2.py            # CEARF-N v2 runner (Diginetica routers)
├── run_paper_baselines.py      # Baselines (GRU4Rec/NARM/SASRec/SR-GNN/SIGMA)
├── plot_rq_figures.py          # Publication figures per RQ
├── demo/                       # Interactive demo (FastAPI + HTML)
├── artifacts_paper/            # Frozen result archive (model outputs)
│   ├── per_query_ranks/        # 72 per-query rank arrays
│   ├── per_seed_metrics.json   # R@6/10/20/NDCG/U per method × domain × seed
│   ├── real_paired_cis.json    # Paired bootstrap CIs
│   ├── manifest.json           # Checksums, shapes, query counts
│   ├── verify_artifacts.py     # Artifact verifier
│   └── README.md
└── paper/                      # LNCS paper source (.tex)
```

## Verify artifacts

```bash
cd sparse_bench/artifacts_paper
python verify_artifacts.py
# OK: 72 arrays verified (monotonic, nDCG<=R@20, utility, JSON, SHA-256)
```

## Reproduce experiments

```bash
# CEARF-N evidence suite (Amazon)
PYTHONPATH=sparse_bench python sparse_bench/run_cearfn_evidence.py \
  Video_Games Baby_Products --seeds 42 43 44 123 456

# CEARF-N v2 (Diginetica routers)
PYTHONPATH=sparse_bench python sparse_bench/run_cearfn_v2.py \
  Diginetica_HID --seeds 42 123 456

# Baselines
PYTHONPATH=sparse_bench python sparse_bench/run_paper_baselines.py \
  Video_Games Baby_Products Diginetica_HID --seeds 42 123 456

# Figures
PYTHONPATH=sparse_bench python sparse_bench/plot_rq_figures.py
```

## Run the demo

```bash
cd sparse_bench/demo
PYTHONPATH=.. python3 app.py
# Open http://localhost:8000
```

The demo loads the datasets, computes memory + neural rankings in real time,
and fuses them with the bounded β gate. Try adding items to the cart — the
session grows and recommendations update.

## Citation

```bibtex
@misc{cearfn2026,
  author = {Cuong, Pham Quoc and Minh, Nguyen Ngoc and Minh, Le Quang and An, Nguyen Hoai Nguyet},
  title = {{CEARF-N}: Bounded Out-of-Fit Dynamic Rank Allocation for Sparse Next-Item Recommendation},
  year = {2026}
}
```
