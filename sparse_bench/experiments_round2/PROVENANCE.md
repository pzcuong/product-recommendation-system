# Round-2 experiments — provenance record

## Context

A filesystem incident on 2026-08-21 destroyed the working tree, including
five artifact directories that had never been committed to git
(`dynamic_beta_artifacts`, `dynamic_beta_allocation_controls_artifacts`,
`dynamic_beta_fusion_operator_artifacts`, `dynamic_beta_expert_swap_artifacts`,
`cearfn_v2_nometa_artifacts`) together with the session's derived ablation
suite. The repository was re-cloned from
`github.com:pzcuong/product-recommendation-system` (commit `5ebabfd`).

This round of experiments (E1–E3, per the journal reviewer's three
priorities) was rebuilt exclusively from **surviving, git-tracked** evidence,
under a strict two-label provenance discipline:

- **RECOMPUTED** — derived in this round from the 72 frozen per-query rank
  arrays in `artifacts_paper/per_query_ranks/` (integrity re-verified: R@20
  of all 72 arrays matches `per_seed_metrics.json` with 0 mismatches).
  Includes MRR@20 and the per-seed fusion-vs-endpoint win frequencies.
- **RECORDED** — transcribed verbatim from the locked auto-generated LaTeX
  tables that the original summarizer scripts wrote before the incident
  (`paper/generated_dynamic_beta_*.tex`, each stamped with its generator),
  the runtime macros from `benchmark_dynamic_beta_inference.py`, the audited
  baseline grids (`generated_dynamic_beta_baseline_audit.tex`), and the
  paper's own matched-teacher table (`main_short.tex`, `tab:metadata`).

No number in E1–E3 is estimated, interpolated, or synthesized.

## Files

| File | Content | Provenance |
|---|---|---|
| `mrr_from_arrays.json` | MRR@20 + R@20 for all 72 arrays | RECOMPUTED |
| `e1_selection_procedures.{json,md}` | P1–P6 procedure comparison, budgets, regret, ΔU CIs | mixed (labels inside) |
| `e2_matched_teacher.{json,md}` | matched-teacher conditions, expert swap, rescue/damage | RECORDED |
| `e3_cost_quality.{json,md}` | R@20/MRR@20 per method, tuning budgets, serving latency | mixed |
| `../paper/round2_procedures.tex` | E1 table (LaTeX) | — |
| `../paper/round2_matched.tex` | E2 table (LaTeX) | — |
| `../paper/round2_cost.tex` | E3 table (LaTeX) | — |
| `../figures/fig_e1_procedures.{pdf,png}` | procedure bars vs oracle | — |
| `../figures/fig_e2_matched.{pdf,png}` | matched-teacher + swap | — |
| `../figures/fig_e3_cost_quality.{pdf,png}` | budget scatter + latency | — |

## Known limits after the incident (state these in the paper)

1. **Diginetica endpoint arrays** (memory-only / neural-only per query) were
   among the lost trees; per-seed win frequencies are therefore reported for
   the two Amazon domains only (3/3 seeds each). Recorded means still show
   fusion > both endpoints on Diginetica.
2. **NARM+sem (TF-IDF / MiniLM) conditions** survive as recorded point
   estimates only — their per-query arrays were not git-tracked, so no new
   paired CI can be computed for them.
3. **Per-baseline serving latency** cannot be re-measured (raw datasets
   lost). The CEARF-N decomposition (recorded macros) and the recorded
   `figures/fig_latency.pdf` chart are the references.
4. **Temporal-split robustness** (reviewer point 4) is not computable for
   the Amazon domains for the same reason; conclusions must stay scoped to
   the frozen valid→test transfer, as stated in the paper's limitations.
5. The old test splits have been inspected repeatedly; all numbers here are
   confirmatory on frozen outputs, not independent-holdout evidence.

## Reproduce

```bash
python3 -c "import numpy, scipy, matplotlib" || python3 -m venv venv && venv/bin/pip install numpy scipy matplotlib
venv/bin/python run_round2_experiments.py     # builds the three JSONs
venv/bin/python emit_round2_tables.py         # tables + figures
```
