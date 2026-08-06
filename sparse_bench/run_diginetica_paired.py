#!/usr/bin/env python3
"""Paired inference: CEARF-N v2 vs neural baselines on Diginetica.

Uses the rank artifacts already produced by run_cearfn_v2.py and
run_paper_baselines.py. No model training required.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from paired_statistics import cluster_paired_recall

HERE = Path(__file__).resolve().parent

SEEDS = (42, 123, 456)
K = 20
REPS = 20000


def artifact_stem(model: str) -> str:
    return model.lower().replace("-", "_")


def paired_recall_test(challenger: np.ndarray, baseline: np.ndarray,
                       k: int = K, reps: int = REPS, seed: int = 20260721) -> dict:
    hit_a = (challenger > 0) & (challenger <= k)
    hit_b = (baseline > 0) & (baseline <= k)
    positive = int(np.sum(hit_a & ~hit_b))
    negative = int(np.sum(~hit_a & hit_b))
    discordant = positive + negative
    pvalue = (float(binomtest(positive, discordant, 0.5).pvalue)
              if discordant else 1.0)
    rng = np.random.default_rng(seed)
    probs = np.asarray([positive, negative, int(len(hit_a) - positive - negative)],
                       dtype=float) / len(hit_a)
    draws = rng.multinomial(len(hit_a), probs, size=reps)
    delta = (draws[:, 0] - draws[:, 1]) / len(hit_a)
    return {
        "k": k,
        "challenger_recall": float(hit_a.mean()),
        "baseline_recall": float(hit_b.mean()),
        "difference": float(hit_a.mean() - hit_b.mean()),
        "paired_bootstrap_ci95": [float(x) for x in np.quantile(delta, [.025, .975])],
        "challenger_only": positive,
        "baseline_only": negative,
        "mcnemar_exact_p": pvalue,
        "n": int(len(hit_a)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", type=Path,
                        default=HERE / "cearfn_v2_nested_artifacts")
    parser.add_argument("--baseline-dir", type=Path,
                        default=HERE / "paper_baseline_digi_nested_artifacts")
    parser.add_argument("--models", nargs="+", default=[
        "gru4rec", "sasrec", "narm", "sr-gnn", "sigma-compatible"])
    parser.add_argument("--output", type=Path,
                        default=HERE / "diginetica_neural_paired_nested.json")
    args = parser.parse_args()
    v2_dir = args.v2_dir
    base_dir = args.baseline_dir
    results = {}

    for seed in SEEDS:
        # Load CEARF-N v2 ranks (regime + continuous).
        v2_path = v2_dir / f"diginetica_hid_v2_seed{seed}_ranks.npz"
        if not v2_path.exists():
            print(f"MISSING v2 seed={seed}: {v2_path}")
            continue
        v2 = np.load(v2_path)
        selected_key = "selected_rank" if "selected_rank" in v2.files else "continuous_rank"
        cearfn_cont = v2[selected_key].astype(np.int32)
        cearfn_regime = v2["regime_rank"].astype(np.int32)

        results[seed] = {"selected_vs_regime":
                         paired_recall_test(cearfn_cont, cearfn_regime)}
        for model in args.models:
            model_stem = artifact_stem(model)
            path = base_dir / f"diginetica_hid_full_{model_stem}_seed{seed}_ranks.npz"
            if path.exists():
                with np.load(path) as z:
                    baseline = z["ranks"].astype(np.int32)
                results[seed][f"vs_{model}"] = paired_recall_test(
                    cearfn_cont, baseline)
            else:
                print(f"MISSING {model} seed={seed}: {path}")

    # Aggregate across seeds.
    aggregate = {}
    complete_tests = sorted(set.intersection(
        *(set(row) for row in results.values()))) if results else []
    for test_name in complete_tests:
        total_pos = sum(r[test_name]["challenger_only"] for r in results.values())
        total_neg = sum(r[test_name]["baseline_only"] for r in results.values())
        discordant = total_pos + total_neg
        agg_p = (float(binomtest(total_pos, discordant, 0.5).pvalue)
                 if discordant else 1.0)
        # Mean difference across seeds.
        diffs = [r[test_name]["difference"] for r in results.values()]
        aggregate[test_name] = {
            "mean_difference": float(np.mean(diffs)),
            "total_challenger_only": total_pos,
            "total_baseline_only": total_neg,
            "sign_test_p": agg_p,
            "per_seed": {str(s): r[test_name] for s, r in results.items()},
        }

    out = {"dataset": "Diginetica_HID", "k": K, "seeds": list(SEEDS),
           "aggregate": aggregate}
    cont = []; regime = []
    for run_seed in SEEDS:
        with np.load(v2_dir / f"diginetica_hid_v2_seed{run_seed}_ranks.npz") as z:
            key = "selected_rank" if "selected_rank" in z.files else "continuous_rank"
            cont.append(z[key].astype(np.int32))
            regime.append(z["regime_rank"].astype(np.int32))
    out["cluster_aggregate"] = {
        "selected_vs_regime": cluster_paired_recall(
            cont, regime, k=K, reps=REPS)}
    for model in args.models:
        model_stem = artifact_stem(model)
        paths = [base_dir /
                 f"diginetica_hid_full_{model_stem}_seed{s}_ranks.npz"
                 for s in SEEDS]
        if all(path.exists() for path in paths):
            rows = []
            for path in paths:
                with np.load(path) as z:
                    rows.append(z["ranks"].astype(np.int32))
            out["cluster_aggregate"][f"vs_{model}"] = cluster_paired_recall(
                cont, rows, k=K, reps=REPS)
    path = args.output
    path.write_text(json.dumps(out, indent=2))

    # Print summary.
    print("\n=== Diginetica paired inference (k=20) ===")
    for test_name in complete_tests:
        agg = aggregate[test_name]
        ci = [
            np.mean([r[test_name]["paired_bootstrap_ci95"][0]
                     for r in results.values()]),
            np.mean([r[test_name]["paired_bootstrap_ci95"][1]
                     for r in results.values()]),
        ]
        print(f"{test_name}: Δ={agg['mean_difference']:+.5f} "
              f"CI95=[{ci[0]:+.5f},{ci[1]:+.5f}] "
              f"sign_p={agg['sign_test_p']:.2e}")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
