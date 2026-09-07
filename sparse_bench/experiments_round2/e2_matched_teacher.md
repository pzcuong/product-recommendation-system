# E2 — Matched-information comparison

## Same-teacher conditions (recorded, tab:metadata)

| Condition | Init | Video | Baby |
|---|---|---|---|
| NARM (ID-only) | random | 0.13705 | 0.02990 |
| CEARF-N no-meta | random | 0.13208 | 0.05055 |
| NARM+sem | TF-IDF | 0.15387 | 0.06628 |
| CEARF-N full | TF-IDF | 0.14700 | 0.05600 |

## Expert swap — does fusion help a STRONG expert too? (recorded, 3 seeds)

| Domain | NARM-only | Gate fusion | Gain | PASGR-only (ref) | Gain over PASGR |
|---|---|---|---|---|---|
| Video | 0.13763 | 0.14854 | +0.01091 | - | +0.02164 |
| Baby | 0.03880 | 0.05001 | +0.01121 | - | +0.00796 |
| Diginetica | 0.53178 | 0.53939 | +0.00761 | - | +0.06849 |

**Findings:** (1) metadata, not the fusion mechanism, drives most of the Amazon gain — NARM+sem beats CEARF-N full on both domains (+.0069 Video, +.0103 Baby). (2) Fusion is expert-strength-robust: on Diginetica NARM is the *stronger* expert (.53178 vs PASGR .44852) yet the same gate still adds +.00761. (3) Rescue/damage decomposition (recorded): Video 4451/1766, Baby 3889/1191, Diginetica 3076/1693 net rescues — fusion's gains come from concrete query-level rescues, not metric artifacts.