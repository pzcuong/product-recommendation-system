# sparse_bench — internal multi-domain benchmark

Internal, domain-agnostic reconstruction of a repository-local DualTwin V3.2
configuration. This is not a reproduction of a published CoDT paper and must
not be cited or described as an established external method. It combines the
repository configuration that reached Recall@6 ≈ 0.43 on the hidden Rental
test set with an experimental **Long-Tail-Aware Reranker (LTAR)** and a
**cold-start vs learnable-tail** grouped-evaluation diagnostic. See
`ANALYSIS.md` for the internal write-up.

## Files

| File | Purpose |
|---|---|
| `codt_core.py` | Domain-agnostic CoDT: PGSA-Rec ensemble, M-CL, co-visitation/PMI, V3.2 fusion, MMR. `train_codt_assets()` trains once per domain; `predict_codt()` scores any variant. |
| `ltar.py` | Long-Tail-Aware Reranker (candidate injection + inverse-popularity bias). |
| `loaders.py` | Unified loaders: `load_rental`, `load_rental_visit`, `load_amazon`, `load_retailrocket`. |
| `grouped_eval.py` | 12-group evaluation + cold-start/learnable-tail split. |
| `baselines.py` | MostPop, ItemKNN, SKNN, GRU4Rec, SASRec (apple-to-apple). |
| `run_codt_multi.py` | Orchestrator: all models × domains × variants, ablations, CI. |
| `ANALYSIS.md` | Internal method notes, results, and gate-check diagnostics. |

## Quick run

```bash
# Full config on Rental (visit-level anchor) — reproduces HR@10 ≈ 0.52
python run_codt_multi.py Rental_visit

# Fast modest pass over all domains (1 seed, subsampled)
python run_codt_multi.py --fast Rental_visit RetailRocket Baby_Products Video_Games
```

## CEARF-N ADMA reproduction: current dynamic-beta protocol

The current short-paper method is **CEARF-N: Contextual Expert-Allocation Rank
Fusion with a Neural residual**. Its fusion coefficient is not a
validation grid value and is not produced by the retired validation router.
A continuous global prior is learned from out-of-fit training ranks, then a
bounded three-feature linear gate assigns

```text
delta_eff = min(0.10, beta_OOF, 1 - beta_OOF)
beta_q = beta_OOF + delta_eff * tanh(w^T z_q + b)
```

for each query. The allocator has five optimized coefficients in total: the
global-prior logit plus three weights and one bias. Its effective residual is clipped at
the simplex boundary, so both `beta_q` and the induced fused-score change are
bounded by construction. More sharply, each pair changes by
`(beta_q - beta_OOF) * ((rho_N[a]-rho_M[a]) - (rho_N[b]-rho_M[b]))`;
this certifies pairs that cannot reverse. Because the allocator consumes ranks
only, it is invariant to monotone raw-score transformations that preserve each
expert's top-120 ordering.

Before allocation fitting, a stable hash declares exactly
5,000 validation queries from each loader's candidate pool. Those declared
source IDs are ineligible for the OOF holdout; on Diginetica their target
events and incoming transitions are removed from tuning sessions. Unselected
Diginetica targets remain training events. Amazon validation targets are
external to training, while source histories outside the declared subset
remain OOF-eligible. Neither declared-validation nor test labels are arguments
to either beta fitter.

```bash
# Unit and protocol tests.
python -m unittest test_dynamic_beta.py \
  test_dynamic_beta_allocation_controls.py \
  test_dynamic_beta_expert_swap.py \
  test_summarize_dynamic_beta.py

# Three matched seeds. The 5,000 OOF sources are split into 1,000
# profile-lock and 4,000 gate-calibration queries.
python run_dynamic_beta.py Video_Games Baby_Products Diginetica_HID \
  --seeds 42 123 456 \
  --oof-cap 5000 --profile-cap 1000 --valid-cap 5000 \
  --output dynamic_beta_trainonly_v2_results.json \
  --artifact-dir dynamic_beta_trainonly_v2_artifacts

# Paired query-level intervals and generated LaTeX tables.
python summarize_dynamic_beta.py --bootstrap-repetitions 20000

# Allocation-only controls reuse the frozen 120-wide OOF/test expert ranks.
# They include coarse OOF policies, capacity/feature ablations, residual
# bounds, and a same-multiset permutation of beta_q across test queries.
# This performs no expert training or prediction.
python run_dynamic_beta_allocation_controls.py \
  Video_Games Baby_Products Diginetica_HID --seeds 42 123 456

# Exact gate/RRF runtime replay and external-expert portability audit.
python benchmark_dynamic_beta_inference.py
python run_dynamic_beta_fusion_control.py \
  Video_Games Baby_Products Diginetica_HID --seeds 42 123 456
python summarize_dynamic_beta_fusion_control.py
python run_dynamic_beta_expert_swap.py \
  Video_Games Baby_Products Diginetica_HID \
  --seeds 42 123 456 \
  --dynamic-artifact-dir dynamic_beta_trainonly_v2_artifacts \
  --primary-results dynamic_beta_trainonly_v2_results.json
python summarize_dynamic_beta_expert_swap.py

# Data-driven paper/slide charts (SVG and PNG).
python slide_graphs/generate_dynamic_beta_charts.py
```

Primary files:

| Artifact | Contents |
|---|---|
| `dynamic_beta.py` | continuous scalar/gate objectives, target-free features, and weighted RRF |
| `run_dynamic_beta.py` | declared-validation-disjoint OOF split, manifests, allocation/capacity/RRF ablations |
| `dynamic_beta_trainonly_v2_results.json` | per-seed metrics for the corrected train-only protocol |
| `dynamic_beta_trainonly_v2_artifacts/` | frozen manifests, matched ranks, beta arrays, and gate states |
| `summarize_dynamic_beta.py` | seed aggregation and paired query-level inference |
| `audit_dynamic_beta_provenance.py` | independent exact reconstruction of declared splits, memory ranks, PASGR checkpoints, and predictions |
| `run_dynamic_beta_allocation_controls.py` | frozen-rank coarse-policy, residual-bound, feature, same-beta-multiset assignment, and regularization controls with a pre-test manifest |
| `dynamic_beta_allocation_controls_results.json` | per-seed R@6/10/20, nDCG@6/10/20, and utility for every allocation control |
| `paper/generated_dynamic_beta_allocation_controls.tex` | generated mean (sample SD) supplementary tables for all allocation controls |
| `run_dynamic_beta_expert_swap.py` | PASGR-to-NARM expert substitution with fresh OOF and complete-training NARM refits at a pre-locked epoch budget |
| `run_dynamic_beta_fusion_control.py` | normalized CombSUM versus weighted RRF under frozen dynamic beta and equal beta=.5, with exact checkpoint replay |
| `benchmark_dynamic_beta_inference.py` | exact gate and RRF replay/timing |
| `slide_graphs/generate_dynamic_beta_charts.py` | charts generated from JSON/NPZ artifacts, without embedded result values |

The eight-page source is `paper/main_dynamic.tex`; the separate technical
supplement is `paper/supplementary_dynamic.tex`.

## Historical validation-gated protocol (not the current paper method)

The ADMA evidence pipeline uses three matched model seeds (`42 123 456`) and
never uses test labels for model or hyperparameter selection.  The principal
commands are:

```bash
# External training-free baselines; V-SKNN/STAN grids are validation-selected.
python run_neighborhood_baselines.py Video_Games Baby_Products Diginetica_HID

# 2x2x2x2 PASGR component sweep, including Diginetica.
python run_validation_gated_pasgr.py Diginetica_HID

# Final gated method and constituent ranks for all matched seeds.
python run_cearfn_v2.py Video_Games Baby_Products Diginetica_HID \
  --seeds 42 123 456

# Query-clustered paired inference (queries are clusters; seeds are repeats).
python run_v2_paired.py
python run_diginetica_paired.py
```

Router-family selection is nested inside validation: 80% of validation queries
fit/calibrate the regime, bucketed, and continuous alternatives; the disjoint
20% chooses the family by the same Recall@6/Recall@20 utility, with simpler
routers winning exact ties.  The chosen family is refit on all validation
queries before test inference.  Component-sweep ties are likewise resolved
only with validation metrics and then by fewer enabled switches; test metrics
are recorded after selection and never participate in ordering cells.
On Diginetica, validation queries are reconstructed from training sessions.
The held target and incoming transition are therefore removed from each source
session before building the validation memory or fitting PASGR/neighbourhood
models.  Full training sessions are restored only for the single post-selection
test fit.  Cache names containing `nested_valid`/`nested_test` identify this
leakage-safe protocol; older unqualified Diginetica validation caches must not
be used for paper results.

Neural baselines on Diginetica follow the same leakage-safe train/validation
split, but the primary paper numbers evaluate the validation-selected
checkpoint directly rather than refitting from scratch on full sessions.
That choice is deliberate: a full-session refit produced a catastrophic NARM
collapse despite a healthy selected checkpoint, so the locked manuscript
numbers use the selected checkpoint consistently across GRU4Rec, SASRec, NARM,
SR-GNN, and SIGMA-compatible.  The refit-from-scratch path remains available
only as a sensitivity check and must not replace the paper table.

`run_neighborhood_baselines.py` follows the public `rn5l/session-rec`
V-SKNN/STAN scoring equations.  Because the unified loaders retain item order
but not original wall-clock session timestamps, STAN uses deterministic loader
session order as its recency coordinate; validation may select no inter-session
time decay.  V-SKNN, STAN, and transition retrieval are deterministic, so their
three reported seed rows are identical by design rather than stochastic
replications.

For paired inference, each test query contributes the mean hit difference over
the three matched seeds.  The 20,000-repetition bootstrap resamples queries,
thereby preserving correlation among the three predictions for the same query.
The aggregate randomization p-value uses query-cluster sign flips; exact
McNemar tests are retained separately for each seed.  This is especially
important on Diginetica, where multiple prefix queries can originate from the
same reconstructed source session; the current HID artifact does not retain a
source-session identifier, so clustering above the query level is not possible
and is disclosed as a limitation.

Timing measurements are from an Apple M2 Pro MacBook Pro (12 CPU cores, 32 GB
RAM).  PASGR training/inference uses PyTorch MPS where supported; the memory
path is single-process CPU code.  Report timings per domain rather than as an
unqualified cross-domain range.

### Locked ADMA artifacts and results

The submission tables use the leakage-safe `nested` artifacts, never the
historical unqualified Diginetica files:

| Artifact | Contents |
|---|---|
| `pasgr_config_per_domain.json` | 16-cell validation sweep selections for all three domains |
| `neighborhood_baseline_results.json` | validation-tuned V-SKNN, STAN, and transition metrics plus identical deterministic seed rows |
| `cearfn_v2_nested_results.json` | matched-seed router and constituent results |
| `cearfn_v2_nested_constituent_summary.json` | paper-ready means, standard deviations, and per-seed values |
| `external_baseline_paired_nested.json` | query-cluster bootstrap, sign-flip, and per-seed McNemar tests |
| `cearfn_v2_nometa_nested_results.json` | Diginetica no-metadata attribution rerun |
| `paper_baseline_digi_nested.json` | leakage-safe Diginetica neural baseline aggregates and per-seed checkpoint metadata |
| `diginetica_neural_paired_nested.json` | query-clustered paired tests of CEARF-N vs Diginetica neural baselines |
| `semantic_init_results.json` | semantic-initialized GRU4Rec/NARM reruns for metadata-fairness checks |
| `semantic_init_paired_amazon.json` | paired Amazon tests of CEARF-N vs semantic-initialized baselines |

Selected PASGR switches are Video Games `(graph=.35, prototype=off,
contrastive=.15, in-batch=.10)`, Baby Products `(0, off, .15, 0)`, and
Diginetica `(.35, off, 0, .10)`.  Their validation utilities are `.1369`,
`.0488`, and `.4554`, respectively.  The Diginetica sweep selects
`beta=.35`; its single locked post-selection test run has Recall@20 `.518666`.

Three-seed constituent Recall@20 means (standard deviations) are:

| Domain | Memory-only | Neural-only | Validation-selected fusion |
|---|---:|---:|---:|
| Video Games | .119014 (.000000) | .125708 (.000068) | .146845 (.000385) |
| Baby Products | .038792 (.000000) | .048732 (.000495) | .053461 (.003974) |
| Diginetica | .495646 (.000000) | .448525 (.003015) | .516810 (.002684) |

External V-SKNN/STAN/transition Recall@20 is `.119362/.121146/.042232` on
Video Games, `.050379/.049437/.007349` on Baby Products, and
`.505061/.515019/.401015` on Diginetica.  CEARF-N is statistically tied with
STAN on Diginetica: difference `.001791`, query-cluster 95% CI
`[-.003692, .007301]`, sign-flip `p=.526`.  The Baby selected-fusion variance
is caused by validation choosing the continuous family for seed 42; test
results are not used to replace that choice.

Leakage-safe Diginetica neural Recall@20 means are GRU4Rec `.417847`, SASRec
`.270970`, NARM `.534057`, SR-GNN `.492666`, and SIGMA-compatible `.372194`.
CEARF-N exceeds GRU4Rec, SASRec, SR-GNN, and SIGMA-compatible, is tied with
STAN, and trails NARM by `.01725` with query-cluster 95% CI
`[-.02012, -.01435]`.

The metadata-fairness rerun materially weakens the Amazon superiority claim.
With the same semantic initialization used by PASGR, GRU4Rec reaches Recall@20
`.134277` on Video Games and `.059474` on Baby Products, while NARM reaches
`.153870` and `.066279`. Paired Amazon tests show CEARF-N loses to
semantic-initialized NARM on both domains: Video Games `-.007025`
with 95% CI `[-.008470, -.005589]`, and Baby Products `-.012818`
with 95% CI `[-.013775, -.011845]`. The evidence therefore supports a
metadata-aware systems claim, not a metadata-neutral claim of architectural
superiority over external neural baselines.

## Key results (see ANALYSIS.md)

- **Rental (visit-level):** CoDT-DT-FullFusion R@10 = **0.517** versus the
  repository-local artifact value 0.459. This is an internal gate check, not a
  comparison with a published reference.
- **Fusion ablation:** PGSA-only R@6=0.405 → +fusion R@6=0.413 (fusion lifts the
  top-6/top-10 positions — resolves the repo's internal "fusion hurts" note,
  which was an under-trained single-seed ablation).
- **Tail diagnostic:** `tgt_tail=0` on Rental is caused by **cold-start targets**
  (train freq = 0) — information-theoretically unrecoverable by any collaborative
  method, not a CoDT artifact.
