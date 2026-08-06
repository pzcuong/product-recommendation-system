#!/usr/bin/env python3
"""Paired inference: CEARF-N v2 vs semantic-init neural baselines on Amazon."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from paired_statistics import cluster_paired_recall

HERE = Path(__file__).resolve().parent
SEEDS = (42, 123, 456)
K = 20
REPS = 20000
DOMAINS = {
    "Video_Games": "video_games",
    "Baby_Products": "baby_products",
}
DISPLAY = {
    "GRU4Rec": "GRU4Rec+sem",
    "NARM": "NARM+sem",
}


def paired_test(challenger: np.ndarray, baseline: np.ndarray,
                k: int = K, reps: int = REPS, seed: int = 20260725) -> dict:
    hit_a = (challenger > 0) & (challenger <= k)
    hit_b = (baseline > 0) & (baseline <= k)
    pos = int(np.sum(hit_a & ~hit_b))
    neg = int(np.sum(~hit_a & hit_b))
    disc = pos + neg
    pvalue = float(binomtest(pos, disc, 0.5).pvalue) if disc else 1.0
    rng = np.random.default_rng(seed)
    probs = np.asarray([pos, neg, int(len(hit_a) - pos - neg)], dtype=float) / len(hit_a)
    draws = rng.multinomial(len(hit_a), probs, size=reps)
    delta = (draws[:, 0] - draws[:, 1]) / len(hit_a)
    return {
        "challenger_recall": float(hit_a.mean()),
        "baseline_recall": float(hit_b.mean()),
        "difference": float(hit_a.mean() - hit_b.mean()),
        "ci95": [float(x) for x in np.quantile(delta, [.025, .975])],
        "pos": pos,
        "neg": neg,
        "mcnemar_p": pvalue,
        "n": int(len(hit_a)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", type=Path,
                        default=HERE / "cearfn_v2_nested_artifacts")
    parser.add_argument("--semantic-dir", type=Path,
                        default=HERE / "semantic_init_artifacts")
    parser.add_argument("--models", nargs="+", default=["GRU4Rec", "NARM"])
    parser.add_argument("--output", type=Path,
                        default=HERE / "semantic_init_paired_amazon.json")
    args = parser.parse_args()

    results = {}
    for domain_key, domain_stem in DOMAINS.items():
        domain_results = {"baselines": {}}
        for model in args.models:
            per_seed = {}
            model_stem = model.lower()
            c_stack = []
            b_stack = []
            for seed in SEEDS:
                v2_path = args.v2_dir / f"{domain_stem}_v2_seed{seed}_ranks.npz"
                baseline_path = args.semantic_dir / (
                    f"{domain_stem}_{model_stem}_sem_seed{seed}_ranks.npz"
                )
                if not v2_path.exists() or not baseline_path.exists():
                    continue
                with np.load(v2_path) as z:
                    rank_key = "selected_rank" if "selected_rank" in z.files else "regime_rank"
                    challenger = z[rank_key].astype(np.int32)
                with np.load(baseline_path) as z:
                    baseline = z["ranks"].astype(np.int32)
                per_seed[str(seed)] = paired_test(challenger, baseline)
                c_stack.append(challenger)
                b_stack.append(baseline)
            if not per_seed:
                continue
            total_pos = sum(row["pos"] for row in per_seed.values())
            total_neg = sum(row["neg"] for row in per_seed.values())
            disc = total_pos + total_neg
            sign_p = float(binomtest(total_pos, disc, 0.5).pvalue) if disc else 1.0
            diffs = [row["difference"] for row in per_seed.values()]
            domain_results["baselines"][model] = {
                "per_seed": per_seed,
                "aggregate": {
                    "mean_difference": float(np.mean(diffs)),
                    "sign_test_p": sign_p,
                    "total_pos": total_pos,
                    "total_neg": total_neg,
                },
                "cluster_aggregate": cluster_paired_recall(c_stack, b_stack, k=K, reps=REPS),
            }
        results[domain_key] = domain_results

    args.output.write_text(json.dumps(results, indent=2))
    print("\n=== CEARF-N v2 vs semantic-init baselines ===")
    for domain_key, domain_results in results.items():
        print(f"\n[{domain_key}]")
        for model in args.models:
            if model not in domain_results["baselines"]:
                continue
            row = domain_results["baselines"][model]
            agg = row["aggregate"]
            cluster = row["cluster_aggregate"]
            print(
                f"  vs {DISPLAY.get(model, model):12s}: "
                f"Δ={agg['mean_difference']:+.5f} "
                f"CI95=[{cluster['cluster_bootstrap_ci95'][0]:+.5f},"
                f"{cluster['cluster_bootstrap_ci95'][1]:+.5f}] "
                f"p={cluster['cluster_sign_flip_p']:.2e} "
                f"(CEARF-N={cluster['challenger_recall_mean']:.5f}, "
                f"{DISPLAY.get(model, model)}={cluster['baseline_recall_mean']:.5f})"
            )
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
