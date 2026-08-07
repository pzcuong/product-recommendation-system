# CEARF-N Paper Artifacts

Per-query rank arrays from model inference outputs.

## Contents

- `per_query_ranks/*.npz` — per-query target ranks (0=miss, 1..20=hit), 72 arrays
- `per_seed_metrics.json` — R@6/R@10/R@20/nDCG@20/U per method × domain × seed
- `real_paired_cis.json` — paired bootstrap CIs (query-level, 20,000 reps)
- `manifest.json` — rank-array checksums, shapes, query counts
- `verify_artifacts.py` — artifact verifier

## Sources

| Source | Methods | Seeds |
|---|---|---|
| `cearfn_evidence_artifacts` | CEARF-N, memory-only, neural-only | 3 matched |
| `cearfn_v2_artifacts` | continuous / regime / bucketed routers | 3 |
| `paper_baseline_artifacts` | GRU4Rec, NARM, SASRec, SR-GNN, SIGMA | 3 |

## Key numbers (R@20, mean over 3 seeds)

| Domain | CEARF-N | Memory | Neural | NARM | SR-GNN | GRU4Rec |
|---|---:|---:|---:|---:|---:|---:|
| Video Games | **.1460** | .1190 | .1263 | .1371 | .1227 | .1037 |
| Baby Products | **.0539** | .0388 | .0450 | .0299 | .0521 | .0450 |
| Diginetica | **.4903** | — | — | .4827 | .4322 | .4172 |

## Paired CIs (CEARF-N vs baselines, query-level bootstrap)

| Domain | vs NARM | vs SR-GNN | vs GRU4Rec | vs SASRec | vs SIGMA |
|---|---:|---:|---:|---:|---:|
| Video Games | +.0089 SIG | +.0233 SIG | +.0423 SIG | +.1080 SIG | +.0595 SIG |
| Baby Products | +.0240 SIG | +.0018 SIG | +.0089 SIG | +.0223 SIG | +.0292 SIG |
| Diginetica | +.0076 SIG | +.0581 SIG | +.0749 SIG | +.4882 SIG | +.1169 SIG |

All CIs exclude zero (see `real_paired_cis.json`).

### Constituent endpoints (Amazon)

| Domain | vs Memory | vs Neural |
|---|---:|---:|
| Video Games | +.0270 SIG | +.0197 SIG |
| Baby Products | +.0151 SIG | +.0089 SIG |

CEARF-N beats both constituent endpoints (memory-only, neural-only) on the
two Amazon domains.

### Allocation (Diginetica, v2 routers)

| Comparison | Δ R@20 | 95% CI |
|---|---:|---|
| continuous vs regime | +.0028 | [+.0023, +.0033] |
| continuous vs bucketed | +.0027 | [+.0022, +.0032] |

On Diginetica, the available real allocation controls are the continuous /
regime / bucketed routers; the continuous router is detectably better than
both alternatives.

## Statistical caveat (Diginetica)

Diginetica prefix queries do not retain original session identifiers; paired
intervals are query-level, not session-clustered, and do not model residual
within-session dependence.

## Verification

```bash
python verify_artifacts.py
# OK: 72 arrays verified
```

- Monotonicity R@6 ≤ R@10 ≤ R@20: checked against JSON.
- nDCG@20 ≤ R@20: checked (single-target, misses contribute 0).
- Point estimates are the center of reported CIs.
- `real_paired_cis.json` cross-checked against arrays.
