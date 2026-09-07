# E2 — Matched-information comparison

## Same-teacher conditions (recorded; TF-IDF condition)

| Condition | Init | Video | Baby |
|---|---|---|---|
| NARM (ID-only) | random | 0.13705 | 0.02990 |
| CEARF-N no-meta | random | 0.13208 | 0.05055 |
| NARM+sem | TF-IDF | 0.15387 | 0.06628 |
| CEARF-N (TF-IDF condition) | TF-IDF | 0.14700 | 0.05600 |

## Expert swap — does fusion help a STRONG expert too? (recorded, 3 seeds)

| Domain | NARM-only | Gate fusion | Gain |
|---|---|---|---|
| Video | 0.13763 | 0.14854 | +0.01091 |
| Baby | 0.03880 | 0.05001 | +0.01121 |
| Diginetica | 0.53178 | 0.53939 | +0.00761 |

NOTE: the rescue/damage counts (4451/1766, 3889/1191, 3076/1693) belong to the PRIMARY comparison (dynamic fusion vs memory-only) and satisfy (rescue−damage)/N = ΔR@20 vs memory exactly; they are NOT a decomposition of the NARM-swap gain.