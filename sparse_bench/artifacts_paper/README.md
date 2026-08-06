# CEARF-N Paper Artifacts

Single reproducible artifact package backing the paper's headline claims.

## Contents

| File | Description |
|---|---|
| `minilm_config.json` | MiniLM semantic-teacher configuration (frozen, no fine-tuning) |
| `per_seed_metrics.json` | Per-seed R@6/R@10/R@20/NDCG@20/U for all methods x 3 domains x 3 seeds |
| `per_query_ranks/*.npz` | Per-query rank arrays (0=miss, 1..20=hit) for query_cond / equal_mixing / cearfn_minilm / narm_minilm |
| `narm_minilm_results.json` | NARM+MiniLM headline + paired bootstrap CIs |

## Key numbers

| Domain | CEARF-N+MiniLM | NARM+MiniLM | NARM+TF-IDF | Query-cond > equal mixing |
|---|---:|---:|---:|---:|
| Video Games | **.15490** | .15430 | .15387 | +.0023 (p=.004) |
| Baby Products | **.06810** | .06740 | .06628 | +.0021 (p=.01) |
| Diginetica | **.53800** | .53600 | .53408 | +.0184 (p<.001) |

CEARF-N + MiniLM teacher achieves the best R@20 on all three domains,
with the dynamic gate beating equal mixing on 3/3 domains and all paired CIs
entirely above zero.

## Reproduction

```bash
PYTHONPATH=sparse_bench python sparse_bench/run_cearfn_v2.py \
  Video_Games Baby_Products Diginetica_HID --seeds 42 123 456 --semantic miniLM
PYTHONPATH=sparse_bench python sparse_bench/run_paper_baselines.py \
  Video_Games Baby_Products --semantic miniLM --model NARM
```
