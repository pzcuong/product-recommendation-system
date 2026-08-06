# Canonical CEARF-N Submission Lineage

This manifest is the source-of-truth boundary for the ADMA 2026 short paper.

## Included

- Core CEARF-N: `cearfn_v2_nested_results.json`
- Locked PASGR family: `pasgr_config_per_domain.json`
- Three-seed constituents: `cearfn_v2_nested_constituent_summary.json`
- External 15-state gate: `narm_expert_matched_r6r20.json`
- Gate versus validation-best singleton: `hard_gate_singleton_audit.json`
- ID-only paired controls: `baseline_paired_analysis.json` and
  `external_baseline_paired_nested*.json`
- Matched semantic teachers: `narm_tfidf_fairness_nested_results.json`,
  `narm_minilm_fairness_results.json`, and `cearfn_v2_minilm_results.json`

All reported main runs use seeds 42, 123, and 456. The core and hard gate use
the declared utility `0.5 * Recall@6 + 0.5 * Recall@20`. The hard gate contains
all 15 non-empty subsets of CEARF-N, STAN, V-SKNN, and NARM, including the four
singletons.

## Canonical Diginetica numbers

- CEARF-N v2: Recall@20 = **0.51681**
- ID-only NARM: Recall@20 = **0.53406**
- Validation-selected hard gate: Recall@20 = **0.53790**

Diginetica uses the official HID session split, repeat consumption
(`exclude_seen=False`), leakage-safe validation memory, locked per-seed router
selection, and matched NARM checkpoints.

## Explicitly excluded

The `.49028` CEARF-N and `.48265` NARM values in `plot_results.py` belong to an
earlier continuous-router exploration. They are retained only for historical
diagnostics and must not be used in submission tables, figures, or claims.

The legacy `cearfn_evidence_results.json` and exploratory `cearfn_v2_results`
plots are likewise not sources for the locked main tables.
