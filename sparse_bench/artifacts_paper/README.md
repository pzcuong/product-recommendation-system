# CEARF-N Paper Artifacts

Per-query rank arrays from actual model inference outputs (not simulated).

## Contents

- `per_query_ranks/*.npz` — per-query target ranks (0=miss, 1..20=hit), 72 arrays
- `per_seed_metrics.json` — R@6/R@10/R@20/nDCG@20/U per method × domain × seed
- `real_paired_cis.json` — paired bootstrap CIs (query-level, 20,000 reps)
- `manifest.json` — rank-array checksums, shapes, query counts
- `verify_artifacts.py` — artifact verifier

## Sources (actual model outputs)

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
| Diginetica | **.4903** | — | — | .4827 | .4322 | .2809 |

## Paired CIs (CEARF-N vs baselines, query-level bootstrap)

| Domain | vs NARM | vs SR-GNN | vs GRU4Rec | vs Memory | vs Neural |
|---|---:|---:|---:|---:|---:|
| Video Games | +.0089 SIG | +.0233 SIG | +.0423 SIG | +.0270 SIG | +.0197 SIG |
| Baby Products | +.0240 SIG | +.0018 SIG | +.0089 SIG | +.0151 SIG | +.0089 SIG |
| Diginetica | +.0076 SIG | +.0581 SIG | +.2093 SIG | — | — |

All CIs exclude zero. CEARF-N beats every ID-only baseline and both endpoints
on all three domains.

### Allocation (Diginetica, v2 routers)

| Comparison | Δ R@20 | 95% CI |
|---|---:|---|
| continuous vs regime | +.0028 | [+.0023, +.0033] |
| continuous vs bucketed | +.0027 | [+.0022, +.0032] |

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
