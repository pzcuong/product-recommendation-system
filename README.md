# CEARF-N: Query-Conditioned Cross-Evidence Weighting

A memory–neural rank-fusion system for sparse session recommendation. CEARF-N
combines three retrieval memories (transition, session, popularity) with a
neural residual (GRU4Rec/PASGR) via a **query-conditioned β gate** — a
lightweight MLP that predicts the optimal per-query memory/neural blend
coefficient from 14 interpretable features.

## Key results (Recall@20)

| Dataset | Uniform β | k-Means | Feature bucketed | **Query-conditioned** |
|---|---:|---:|---:|---:|
| Video Games | .137 | .138 | .136 | **.147** |
| Baby Products | .052 | .053 | .051 | **.056** |
| Diginetica | .456 | .457 | .455 | **.490** |

The query-conditioned gate achieves the best R@20 on all three domains, with
no sign-flip across seeds. Cross-expert agreement is the most informative
gate feature.

## Architecture

```
Training sessions → K-fold cross-fitting → OOF predictions (memory+neural)
                                          → 14 target-free features
                                          → dynamic β gate (MLP 14→16→21)
                                          → freeze → test once
```

Fusion formula: `F_q(x) = (1-β_q)/(20+rank_M(x)) + β_q/(20+rank_N(x))`

## Repository layout

```
sparse_bench/
├── cearf.py                    # Memory index (transition/session/popularity)
├── pasgr.py                    # Neural residual
├── dynamic_beta.py             # Query-conditioned β gate
├── run_cearfn_v2.py            # Main runner
├── run_paper_baselines.py      # Baselines (GRU4Rec/SASRec/NARM/SR-GNN/SIGMA)
├── plot_rq_figures.py          # Publication figures per RQ
├── plot_latency.py             # Latency comparison chart
├── demo/                       # Interactive demo (FastAPI + HTML)
│   ├── app.py                  # Backend
│   └── templates/index.html    # Frontend (cart/purchase/history)
└── paper/                      # Paper source (LaTeX, LNCS)
```

## Run the demo

```bash
cd sparse_bench/demo
PYTHONPATH=.. python3 app.py
# Open http://localhost:8000
```

The demo loads 3 datasets (Baby Products, Video Games, Diginetica), computes
memory + neural rankings in real time, and fuses them with a query-conditioned
β. Try adding items to cart — the session grows and recommendations update.

## Reproduce results

```bash
# CEARF-N v2 multi-seed
PYTHONPATH=sparse_bench python sparse_bench/run_cearfn_v2.py \
  Video_Games Baby_Products Diginetica_HID --seeds 42 123 456

# Baselines
PYTHONPATH=sparse_bench python sparse_bench/run_paper_baselines.py \
  Video_Games Baby_Products

# Figures
PYTHONPATH=sparse_bench python sparse_bench/plot_rq_figures.py
```

## Citation

If you use this work, please cite:

```bibtex
@misc{cearfn2026,
  author = {Cuong, Pham Quoc and Minh, Nguyen Ngoc and Minh, Le Quang and An, Nguyen Hoai Nguyet},
  title = {{CEARF-N}: Query-Conditioned Cross-Evidence Weighting for Sparse Session Recommendation},
  year = {2026}
}
```
