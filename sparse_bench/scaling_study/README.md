# Controlled GRU–SSM data-scaling study

This directory implements all eight phases of the reviewer-response plan without
changing the legacy benchmark. Every trained run stores its config, checkpoint,
rankings, per-query metrics, learning history, parameter count and provenance.

## Phase map

| Phase | Implementation |
|---|---|
| 0. Protocol/audit | immutable JSON manifests with dataset checksum; `audit` command |
| 1. Model ablation | `gru4rec`, `pure_ssm`, `fe_gru`, `fe_gru_ssm`; shared head/loss |
| 2. Coverage/metrics | coverage JSON plus Recall@5/10/20, NDCG@20, seen-target Recall@20 |
| 3. Pilot/decision | factorial `suite`; explicit A/B/C `decision` gate |
| 4. Fair tuning | equal trial budget, validation-only `tune` command |
| 5. Cross-over | multiple draw manifests, aggregation and per-draw interpolation |
| 6. SASRec | causal SASRec in the same runner and tuning budget |
| 7. Paper | outcome-dependent LNAI source skeleton in `paper/` |
| 8. Reproducibility | raw artifacts, tests and completeness audit |

Run commands from the repository root:

```bash
python -m sparse_bench.scaling_study.cli manifests \
  --scales 10000,15000,30000 --draws 42

python -m sparse_bench.scaling_study.cli suite \
  --manifests sparse_bench/scaling_study/artifacts/manifests \
  --config sparse_bench/scaling_study/configs/default.json \
  --variants gru4rec,pure_ssm,fe_gru,fe_gru_ssm --seeds 42,123,456

python -m sparse_bench.scaling_study.cli decision \
  --results sparse_bench/scaling_study/artifacts/runs
```

After the pilot, generate 10K/12K/15K manifests for draws `42,123,456`, run only
`gru4rec` and the scientifically supported challenger with seeds `42,123`, then:

```bash
python -m sparse_bench.scaling_study.cli analyze \
  --results sparse_bench/scaling_study/artifacts/runs \
  --challenger pure_ssm

python -m sparse_bench.scaling_study.cli audit \
  --results sparse_bench/scaling_study/artifacts/runs
```

Tuning writes a validation-ranked `selection.json`. Test metrics are stored for
audit but are never used by the selector:

```bash
python -m sparse_bench.scaling_study.cli tune \
  --manifest sparse_bench/scaling_study/artifacts/manifests/scale_15000/draw_42.json \
  --search-space sparse_bench/scaling_study/configs/search_space.json \
  --max-trials 9
```

Do not mix pilot runs and final tuned runs in the same results directory. The
cross-over output is descriptive: interpolation is emitted only when the paired
model difference changes sign between evaluated scales. It is not a confidence
interval and must not be presented as one.
