# E3 — Cost-quality audit (same lineage as Tables 1-2)

| Method | Budget | R@20 V | R@20 B | R@20 D | nDCG@20 D |
|---|---|---|---|---|---|
| Transition-only | det. | 0.04223 | 0.00735 | 0.40102 | 0.16687 |
| V-SKNN | 8 | 0.11936 | 0.05038 | 0.50506 | 0.25184 |
| STAN | 16 | 0.12115 | 0.04944 | 0.51502 | 0.25952 |
| GRU4Rec | <=12 ep. | 0.10368 | 0.04501 | 0.41785 | 0.19557 |
| NARM | <=12 ep. | 0.13705 | 0.02990 | 0.53406 | 0.26321 |
| SR-GNN | <=5 ep. | 0.12267 | 0.05208 | 0.49267 | 0.23599 |
| SIGMA-compat. | <=10 ep. | 0.08644 | 0.02473 | 0.37219 | 0.17374 |
| CEARF-N (equal $.5$) | 0 (alloc.) | 0.14610 | 0.05555 | 0.51430 | 0.25658 |
| CEARF-N (OOF global) | 0 (alloc.) | 0.14719 | 0.05671 | 0.51702 | 0.25786 |
| CEARF-N (dynamic gate) | 0 (alloc.) | 0.14735 | 0.05669 | 0.51701 | 0.25792 |

**CEARF-N serving decomposition (recorded, Video):** gate 1.171 µs + fusion 85.89 µs + expert 3.71 ms = 3.796 ms/query — gate+fusion overhead is 2.3% of total; expert retrieval dominates.

**Note:** per-baseline latency cannot be re-measured after the incident (datasets lost); the recorded figure `figures/fig_latency.pdf` remains the reference chart.

v2-lineage MRR@20 (recomputed from frozen arrays) lives in `experiments_round2/mrr_from_arrays.json` — different lineage, excluded from the main tables deliberately.