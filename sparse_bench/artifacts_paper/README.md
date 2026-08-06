# CEARF-N Paper Artifacts

## Contents

- `per_query_ranks/*.npz` — per-query ranks (0=miss, 1..20=hit), 81 arrays
- `per_seed_metrics.json` — R@6/R@10/R@20/nDCG@20/U per method × domain × seed
- `narm_minilm_results.json` — paired bootstrap CIs
- `minilm_config.json` — MiniLM semantic-teacher configuration
- `manifest.json` — file checksums, shapes, query counts

## Key numbers (R@20, mean over 3 seeds)

| Domain | CEARF-N+MiniLM | NARM+MiniLM | Query-cond | Equal mixing |
|---|---:|---:|---:|---:|
| Video Games | **.1510** | .1500 | **.1470** | .1447 |
| Baby Products | **.0645** | .0638 | **.0560** | .0539 |
| Diginetica | .5335 | **.5340** | **.4900** | .4716 |

- Dynamic gate beats equal mixing on 3/3 domains (all paired CI > 0).
- CEARF-N+MiniLM has higher mean on 2/3; differences are not detectably
  significant (paired CI includes 0). Diginetica NARM+MiniLM is marginally
  higher. MiniLM results are a matched-teacher sensitivity control, not a
  SOTA claim.

## Verification

- Monotonicity R@6 ≤ R@10 ≤ R@20: enforced by construction.
- nDCG@20 ≤ R@20: enforced (single-target, misses contribute 0).
- Point estimates are the center of reported CIs.
