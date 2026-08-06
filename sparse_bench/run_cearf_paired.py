#!/usr/bin/env python3
"""Paired CEARF evaluation against standard non-parametric baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import baselines
import cearf
import loaders


HERE = Path(__file__).resolve().parent


def hit_vector(predictions, queries, k):
    keys = sorted(queries)
    return np.asarray([
        bool(set(queries[uid].get("targets", [])).intersection(
            predictions.get(str(uid), ())[:k]))
        for uid in keys
    ], dtype=np.int8)


def paired_test(challenger, baseline, queries, k, samples=20000, seed=42):
    left = hit_vector(challenger, queries, k)
    right = hit_vector(baseline, queries, k)
    difference = left - right
    rng = np.random.default_rng(seed + k)
    indices = rng.integers(0, len(difference), size=(samples, len(difference)))
    boot = difference[indices].mean(axis=1)
    challenger_only = int(np.sum((left == 1) & (right == 0)))
    baseline_only = int(np.sum((left == 0) & (right == 1)))
    discordant = challenger_only + baseline_only
    p_mcnemar = (float(binomtest(challenger_only, discordant, .5).pvalue)
                 if discordant else 1.0)
    return {
        "k": k,
        "challenger": float(left.mean()),
        "baseline": float(right.mean()),
        "difference": float(difference.mean()),
        "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "challenger_only": challenger_only,
        "baseline_only": baseline_only,
        "mcnemar_exact_p": p_mcnemar,
        "n": len(difference),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=["Rental_visit", "RetailRocket"])
    parser.add_argument("--output", type=Path,
                        default=HERE / "cearf_paired_results.json")
    args = parser.parse_args()
    output = {}
    config = cearf.CEARFConfig()
    for domain in args.domains:
        data = loaders.ALL_LOADERS[domain]()
        sessions = data["train_sessions"]
        validation = data.get("valid_queries") or {}
        if validation:
            tune_sessions = sessions
        else:
            tune_sessions, validation = cearf.make_validation_split(
                sessions, config.validation_fraction, config.validation_cap)
        tune_index = cearf.CEARFIndex(tune_sessions, data["n_items"], config)
        profiles, tuning = cearf.tune_profiles(tune_index, validation)
        del tune_index
        index = cearf.CEARFIndex(sessions, data["n_items"], config)
        queries = data["test_queries"]
        challenger = index.predict(queries, profiles, 20)
        domain_result = {"tuning": tuning, "metrics": cearf.ranking_metrics(
            challenger, queries), "comparisons": {}}
        for name in ("ItemKNN", "SKNN"):
            print(f"[paired] {domain} baseline={name}", flush=True)
            baseline = baselines.run_nonparametric(
                name, sessions, queries, data["n_items"])
            domain_result["comparisons"][name] = {
                str(k): paired_test(challenger, baseline, queries, k)
                for k in (6, 10, 20)
            }
        output[domain] = domain_result
        args.output.write_text(json.dumps(output, indent=2))
        print(json.dumps({domain: domain_result}, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
