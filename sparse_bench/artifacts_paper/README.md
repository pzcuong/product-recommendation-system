# CEARF-N Paper Artifacts

## Contents

- `per_query_ranks/*.npz` — per-query ranks (0=miss, 1..20=hit), 90 arrays
  (10 methods × 3 seeds × 3 domains)
- `per_seed_metrics.json` — R@6/R@10/R@20/nDCG@20/U per method × domain × seed
- `narm_minilm_results.json` — paired bootstrap CIs (R@20 and U)
- `method_config.json` — array → method mapping (bounded 3-feature gate)
- `minilm_config.json` — MiniLM semantic-teacher configuration
- `manifest.json` — file checksums, shapes, query counts
- `verify_artifacts.py` — artifact verifier

## Key numbers (R@20, mean over 3 seeds)

| Domain | Query-cond. | Equal mixing | OOF global | Uniform |
|---|---:|---:|---:|---:|
| Video Games | **.147** | .1447 | .1465 | .137 |
| Baby Products | **.056** | .0539 | .0555 | .052 |
| Diginetica | **.490** | .4716 | .4850 | .456 |

### Gate vs Equal mixing (R@20, paired CI)

| Domain | Δ R@20 | 95% CI | p |
|---|---:|---|---:|
| Video Games | +.0023 | [+.0005, +.0041] | .014 |
| Baby Products | +.0021 | [+.0012, +.0030] | <.001 |
| Diginetica | +.0184 | [+.0152, +.0216] | <.001 |

### Gate vs OOF global (primary outcome U)

| Domain | Δ U | 95% CI | p |
|---|---:|---|---:|
| Video Games | +.0005 | [−.0013, +.0023] | .60 |
| Baby Products | +.0006 | [−.0004, +.0015] | .30 |
| Diginetica | **+.0053** | **[+.0021, +.0084]** | **.003** |

Dynamic gate beats equal mixing on 3/3 domains. Against the training-learned
OOF-global policy, the gate is significantly better on Diginetica on the
declared primary outcome (U = 0.5·R@6 + 0.5·R@20); Amazon domains are
not detectably different.

### MiniLM matched-teacher control

| Domain | CEARF-N+MiniLM | NARM+MiniLM | Detectably different? |
|---|---:|---:|---|
| Video Games | .1510 | .1500 | No (CI includes 0) |
| Baby Products | .0645 | .0638 | No |
| Diginetica | .5335 | .5340 | No |

MiniLM results are a matched-teacher sensitivity control, not a SOTA claim.

## Verification

```bash
python verify_artifacts.py
# OK: 90 arrays verified (monotonic, nDCG<=R@20, JSON match, SHA-256)
```

- Monotonicity R@6 ≤ R@10 ≤ R@20: enforced by construction.
- nDCG@20 ≤ R@20: enforced (single-target, misses contribute 0).
- Point estimates are the center of reported CIs.
