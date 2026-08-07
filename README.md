# CEARF-N: Query-Conditioned Cross-Evidence Weighting

A memory–neural rank-fusion system for sparse session recommendation. CEARF-N
combines three retrieval memories (transition, session, popularity) with a
neural residual (PASGR, a 1-layer GRU) via a **bounded linear gate** that
predicts the per-query memory/neural blend coefficient β from 3 target-free
features (session length, last-item frequency, tail flag), trained on
out-of-fold (OOF) training data without validation labels for mixing.

## Key results (Recall@20)

| Dataset | Uniform β | k-Means | Feature bucketed | Equal mixing | OOF global | **Query-cond.** |
|---|---:|---:|---:|---:|---:|---:|
| Video Games | .137 | .138 | .136 | .1447 | .1465 | **.147** |
| Baby Products | .052 | .053 | .051 | .0539 | .0555 | **.056** |
| Diginetica | .456 | .457 | .455 | .4716 | .4850 | **.490** |

The gate beats equal mixing on **3/3 domains** (all paired CI > 0). Against
the training-learned OOF global baseline, the gate is significantly better on
Diginetica (+.005, p=.003) and comparable on Amazon (n.s.).

## Method

- **Gate**: bounded linear, 3 features → β_q ∈ [0,1]
- **Fusion**: `F_q(x) = (1-β_q)/(20+rank_M(x)) + β_q/(20+rank_N(x))`
- **Training**: OOF holdout (deterministic source-disjoint split)
- **Memory**: CEARF (transition + session + popularity, 120-width)
- **Neural**: PASGR (1-layer GRU, dim=64, prototype-aligned)

## Repository layout

```
sparse_bench/
├── cearf.py                    # Memory index
├── pasgr.py                    # Neural residual
├── run_cearfn_v2.py            # Main runner (legacy validation-router)
├── run_paper_baselines.py      # Baselines (GRU4Rec/NARM/SR-GNN/SIGMA)
├── plot_rq_figures.py          # Publication figures per RQ
├── demo/                       # Interactive demo (FastAPI + HTML)
├── artifacts_paper/            # Frozen result archive
│   ├── per_query_ranks/        # 90 NPZ rank arrays
│   ├── per_seed_metrics.json   # R@6/10/20/NDCG/U per method × domain × seed
│   ├── narm_minilm_results.json # Paired bootstrap CIs
│   ├── method_config.json      # Array → method mapping
│   ├── minilm_config.json      # Semantic-teacher provenance
│   ├── manifest.json           # SHA-256, shapes, query counts
│   ├── verify_artifacts.py     # Artifact verifier
│   └── README.md
└── paper/                      # LNCS paper source (.tex)
```

## Verify artifacts

```bash
cd sparse_bench/artifacts_paper
python verify_artifacts.py
# Should print: OK: 90 arrays verified
```

## Run the demo

```bash
cd sparse_bench/demo
PYTHONPATH=.. python3 app.py
# Open http://localhost:8000
```

## Citation

```bibtex
@misc{cearfn2026,
  author = {Cuong, Pham Quoc and Minh, Nguyen Ngoc and Minh, Le Quang and An, Nguyen Hoai Nguyet},
  title = {{CEARF-N}: Query-Conditioned Cross-Evidence Weighting for Sparse Session Recommendation},
  year = {2026}
}
```
