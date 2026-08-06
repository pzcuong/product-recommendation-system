#!/usr/bin/env python3
"""Paired bootstrap analysis of CEARF-N vs published baselines on matched seeds.

Runs a per-query McNemar exact test and a paired bootstrap CI for the Recall@20
gap between CEARF-N and each of the 5 baselines, restricted to the seeds both
sides share (42, 123, 456). Also emits a fair comparison table that restricts
CEARF-N to the same three seeds so the mean/std reported for CEARF-N matches
the baselines exactly in seed selection.

Reads:
  sparse_bench/cearfn_evidence_artifacts/{ds}_full_seed{S}_ranks.npz
  sparse_bench/paper_baseline_artifacts/{ds}_full_{model}_seed{S}_ranks.npz
  sparse_bench/paper_baseline_results.json
  sparse_bench/cearfn_evidence_results.json

Writes:
  sparse_bench/baseline_paired_analysis.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
DOMAINS = {
    "Video_Games": "video_games",
    "Baby_Products": "baby_products",
}
BASELINES = ("GRU4Rec", "SASRec", "NARM", "SR-GNN", "SIGMA-compatible")
FILE_BASELINES = {
    "GRU4Rec": "gru4rec",
    "SASRec": "sasrec",
    "NARM": "narm",
    "SR-GNN": "sr_gnn",
    "SIGMA-compatible": "sigma_compatible",
}
SHARED_SEEDS = (42, 123, 456)
K = 20


def paired_recall_test(challenger: np.ndarray, baseline: np.ndarray,
                       k: int = K, reps: int = 20000, seed: int = 20260720) -> dict:
    hit_a = (challenger > 0) & (challenger <= k)
    hit_b = (baseline > 0) & (baseline <= k)
    positive = int(np.sum(hit_a & ~hit_b))
    negative = int(np.sum(~hit_a & hit_b))
    unchanged = int(len(hit_a) - positive - negative)
    rng = np.random.default_rng(seed)
    probabilities = np.asarray([positive, negative, unchanged], dtype=float) / len(hit_a)
    draws = rng.multinomial(len(hit_a), probabilities, size=reps)
    delta = (draws[:, 0] - draws[:, 1]) / len(hit_a)
    discordant = positive + negative
    pvalue = (float(binomtest(positive, discordant, 0.5).pvalue)
              if discordant else 1.0)
    return {
        "k": k,
        "challenger_recall": float(hit_a.mean()),
        "baseline_recall": float(hit_b.mean()),
        "difference": float(hit_a.mean() - hit_b.mean()),
        "relative_improvement_pct": float(
            (hit_a.mean() - hit_b.mean()) / max(hit_b.mean(), 1e-12) * 100.0),
        "paired_bootstrap_ci95": [float(x) for x in np.quantile(delta, [.025, .975])],
        "challenger_only": positive,
        "baseline_only": negative,
        "mcnemar_exact_p": pvalue,
        "bootstrap_repetitions": reps,
        "n": int(len(hit_a)),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


def seed_ci_half_width(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(student_t.ppf(.975, len(arr) - 1) * arr.std(ddof=1) / math.sqrt(len(arr)))


def ranks_at_20_from_rank(rank: np.ndarray) -> np.ndarray:
    """uint8 rank (1..20, 0 = miss) is already a recall-position array."""
    return rank.astype(np.int32)


def main() -> None:
    out: dict = {"K": K, "shared_seeds": list(SHARED_SEEDS), "domains": {}}
    for ds_key, ds_label in DOMAINS.items():
        per_domain: dict = {"baselines": {}}
        print(f"\n=== {ds_key} ===", flush=True)

        # Load CEARF-N ranks for the shared seeds only (fair comparison)
        cearfn_ranks: dict[int, np.ndarray] = {}
        for seed in SHARED_SEEDS:
            path = HERE / "cearfn_evidence_artifacts" / f"{ds_label}_full_seed{seed}_ranks.npz"
            with np.load(path) as data:
                cearfn_ranks[seed] = data["cearfn_rank"].astype(np.int32)
        cearfn_r20 = [float((ranks > 0).mean()) for ranks in cearfn_ranks.values()]
        cearfn_r20_mean, cearfn_r20_std = mean_std(cearfn_r20)
        per_domain["cearfn_r20_matched"] = {
            "per_seed": dict(zip(SHARED_SEEDS, cearfn_r20)),
            "mean": cearfn_r20_mean,
            "std": cearfn_r20_std,
            "seed_ci95_half_width": seed_ci_half_width(cearfn_r20),
        }
        print(f"CEARF-N (3 matched seeds) R@20: {cearfn_r20_mean:.5f} ± {cearfn_r20_std:.5f}",
              flush=True)

        for baseline in BASELINES:
            file_key = FILE_BASELINES[baseline]
            base_ranks: dict[int, np.ndarray] = {}
            for seed in SHARED_SEEDS:
                path = (HERE / "paper_baseline_artifacts"
                        / f"{ds_label}_full_{file_key}_seed{seed}_ranks.npz")
                with np.load(path) as data:
                    base_ranks[seed] = data["ranks"].astype(np.int32)

            base_r20 = [float((ranks > 0).mean()) for ranks in base_ranks.values()]
            base_mean, base_std = mean_std(base_r20)

            # Per-seed paired tests (each test set is the same, so per-seed pairing
            # captures ranking noise on identical queries).
            per_seed_tests: dict[int, dict] = {}
            for seed in SHARED_SEEDS:
                per_seed_tests[seed] = paired_recall_test(
                    cearfn_ranks[seed], base_ranks[seed])

            # Pooled sign test: concatenate per-seed discordant counts and run a
            # single exact binomial sign test on the summed positive/negative.
            # Mean recall over the three seeds is reported separately so the
            # pooled test reflects direction across the matched-seed ensemble.
            total_positive = sum(per_seed_tests[s]["challenger_only"] for s in SHARED_SEEDS)
            total_negative = sum(per_seed_tests[s]["baseline_only"] for s in SHARED_SEEDS)
            discordant = total_positive + total_negative
            pooled_p = (float(binomtest(total_positive, discordant, 0.5).pvalue)
                        if discordant else 1.0)
            pooled_diff_per_query = (
                (cearfn_r20_mean - base_mean)  # mean-of-seeds gap in recall
            )
            # Conservative bootstrap on the per-seed mean gap.
            rng = np.random.default_rng(20260720)
            per_seed_diffs = np.asarray([
                per_seed_tests[s]["challenger_recall"] - per_seed_tests[s]["baseline_recall"]
                for s in SHARED_SEEDS
            ], dtype=float)
            boot = rng.choice(per_seed_diffs, size=(20000, len(SHARED_SEEDS)), replace=True).mean(axis=1)
            pooled = {
                "method": "exact binomial sign test over discordant counts summed across seeds",
                "challenger_recall_mean": cearfn_r20_mean,
                "baseline_recall_mean": base_mean,
                "difference": float(pooled_diff_per_query),
                "relative_improvement_pct": float(
                    pooled_diff_per_query / max(base_mean, 1e-12) * 100.0),
                "paired_bootstrap_ci95": [float(x) for x in np.quantile(boot, [.025, .975])],
                "challenger_only_total": total_positive,
                "baseline_only_total": total_negative,
                "mcnemar_exact_p": pooled_p,
                "bootstrap_repetitions": 20000,
                "n_seeds": len(SHARED_SEEDS),
            }

            per_domain["baselines"][baseline] = {
                "r20_per_seed": dict(zip(SHARED_SEEDS, base_r20)),
                "r20_mean": base_mean,
                "r20_std": base_std,
                "r20_seed_ci95_half_width": seed_ci_half_width(base_r20),
                "paired_vs_cearfn_per_seed": {
                    str(seed): per_seed_tests[seed] for seed in SHARED_SEEDS},
                "paired_vs_cearfn_pooled": pooled,
            }
            print(
                f"  {baseline:18s} R@20: {base_mean:.5f} ± {base_std:.5f} | "
                f"Δ R@20 vs CEARF-N (pooled): {pooled['difference']:+.5f} "
                f"CI95=[{pooled['paired_bootstrap_ci95'][0]:+.5f},"
                f"{pooled['paired_bootstrap_ci95'][1]:+.5f}] "
                f"p={pooled['mcnemar_exact_p']:.2e}",
                flush=True,
            )

        out["domains"][ds_key] = per_domain

    output_path = HERE / "baseline_paired_analysis.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
