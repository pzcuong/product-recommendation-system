# E3 — Cost-quality audit

R@20/MRR@20 recomputed from frozen arrays; budgets recorded.

| CEARF-N | n/a | 0.1460 | 0.0539 | 0.4903 | 0.1629 |
| v2 bucketed router | 3 router cells | -- | -- | 0.4876 | 0.1593 |
| GRU4Rec | <=12 ep. | 0.1037 | 0.0450 | 0.4154 | 0.1605 |
| NARM | <=12 ep. | 0.1371 | 0.0299 | 0.4827 | 0.1683 |
| v2 regime router | 3 router cells | -- | -- | 0.4875 | 0.1593 |
| SASRec | <=12 ep. | 0.0379 | 0.0316 | 0.0020 | 0.0003 |

**CEARF-N serving decomposition (recorded, Video):** gate 1.171 µs + fusion 85.89 µs + expert 3.71 ms = 3.796 ms/query — gate+fusion overhead is 2.3% of total; expert retrieval dominates.

**Note:** per-baseline latency cannot be re-measured after the incident (datasets lost); the recorded figure `figures/fig_latency.pdf` remains the reference chart.