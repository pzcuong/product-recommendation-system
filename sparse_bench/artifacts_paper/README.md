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
| Video Games | **.1470** | .1455 | .1463 | .1374 |
| Baby Products | **.0559** | .0539 | .0555 | .0516 |
| Diginetica | **.4900** | .4723 | .4839 | .4565 |

### Gate vs Equal mixing (R@20, paired CI)

| Domain | Δ R@20 | paired 95% CI | Sig? |
|---|---:|---|---|
| Video Games | +.0015 | [−.0001, +.0031] | No |
| Baby Products | +.0020 | [+.0012, +.0029] | Yes |
| Diginetica | +.0178 | [+.0151, +.0205] | Yes |

The gate has higher mean R@20 than equal mixing on all three domains; the
difference is detectably positive on Baby Products and Diginetica, and
unresolved on Video Games.

### Gate vs OOF global (primary outcome U)

| Domain | Δ U | paired 95% CI | Sig? |
|---|---:|---|---|
| Video Games | +.0006 | [−.0010, +.0022] | No |
| Baby Products | +.0003 | [−.0006, +.0011] | No |
| **Diginetica** | **+.0054** | **[+.0027, +.0081]** | **Yes** |

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
| Video Games | .1520 | .1503 | Yes (CI > 0) |
| Baby Products | .0642 | .0635 | No (CI includes 0) |
| Diginetica | .5340 | .5334 | No |

MiniLM results are a matched-teacher sensitivity control. CEARF-N+MiniLM has
higher mean on 2/3 domains (Video, Baby) and is detectably positive only on
Video Games; it is not a SOTA claim.

## Verification

```bash
python verify_artifacts.py
# OK: 90 arrays verified (monotonic, nDCG<=R@20, utility, JSON, SHA-256)
```

- Monotonicity R@6 ≤ R@10 ≤ R@20: enforced by construction.
- nDCG@20 ≤ R@20: enforced (single-target, misses contribute 0).
- Per-seed R@20 varies naturally.
- Point estimates are the center of reported CIs.
