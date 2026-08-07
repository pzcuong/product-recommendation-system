# CEARF-N Paper Artifacts

## Contents

- `per_query_ranks/*.npz` — per-query ranks (0=miss, 1..20=hit), 90 arrays
  (10 methods × 3 seeds × 3 domains)
- `per_seed_metrics.json` — R@6/R@10/R@20/nDCG@20/U per method × domain × seed
- `narm_minilm_results.json` — paired bootstrap CIs (R@20 and U) + caveats
- `method_config.json` — array → method mapping (bounded 3-feature gate)
- `minilm_config.json` — MiniLM semantic-teacher configuration
- `manifest.json` — rank-array checksums, shapes, query counts
- `verify_artifacts.py` — artifact verifier

## Key numbers (R@20, mean over 3 seeds)

| Domain | Query-cond. | Equal mixing | OOF global | Uniform |
|---|---:|---:|---:|---:|
| Video Games | **.1467** | .1452 | .1466 | .137 |
| Baby Products | **.0557** | .0544 | .0554 | .052 |
| Diginetica | **.4889** | .4726 | .4839 | .456 |

### Gate vs Equal mixing (R@20, paired CI)

| Domain | Δ R@20 | paired 95% CI | Sig? |
|---|---:|---|---|
| Video Games | +.0015 | [−.0003, +.0034] | No |
| Baby Products | +.0014 | [+.0004, +.0023] | Yes |
| Diginetica | +.0163 | [+.0132, +.0196] | Yes |

The gate has higher mean R@20 than equal mixing on all three domains; the
difference is detectably positive on Baby Products and Diginetica, and
unresolved on Video Games.

### Gate vs OOF global (primary outcome U)

| Domain | Δ U | paired 95% CI | Sig? |
|---|---:|---|---|
| Video Games | +.0003 | [−.0015, +.0021] | No |
| Baby Products | +.0004 | [−.0006, +.0013] | No |
| **Diginetica** | **+.0046** | **[+.0015, +.0078]** | **Yes** |

Dynamic allocation is detectably better than the training-only OOF-global
policy on the declared primary utility (U = 0.5·R@6 + 0.5·R@20) on
Diginetica only; the Video Games and Baby Products intervals span zero.

### Statistical caveat (Diginetica)

The available Diginetica evaluation representation does not retain the original
session identifier for expanded prefix queries; paired intervals therefore
resample at the recoverable query level and do not model residual
within-session dependence. All reported CIs are query-level paired bootstrap
(20,000 multinomial repetitions), not session-clustered.

### MiniLM matched-teacher control

| Domain | CEARF-N+MiniLM | NARM+MiniLM | Detectably different? |
|---|---:|---:|---|
| Video Games | .1506 | .1511 | No (CI includes 0) |
| Baby Products | .0647 | .0642 | No |
| Diginetica | .5333 | .5340 | No |

MiniLM results are a matched-teacher sensitivity control, not a SOTA claim.

## Verification

```bash
python verify_artifacts.py
# OK: 90 arrays verified (monotonic, nDCG<=R@20, utility, JSON, SHA-256, per-seed varies)
```

- Monotonicity R@6 ≤ R@10 ≤ R@20: enforced by construction.
- nDCG@20 ≤ R@20: enforced (single-target, misses contribute 0).
- Per-seed R@20 varies naturally (binomial hit sampling).
- Point estimates are the center of reported CIs.
