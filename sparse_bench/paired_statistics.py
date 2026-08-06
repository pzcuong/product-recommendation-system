"""Cluster-aware paired inference for repeated-seed recommendation runs."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def cluster_paired_recall(challenger_by_seed, baseline_by_seed, k=20,
                          reps=20000, seed=20260722):
    """Treat the query as the bootstrap/permutation unit and seeds as repeats.

    Each query contributes its mean hit difference across matched seeds.  This
    preserves within-query correlation instead of pretending that the same
    query repeated for three seeds is three independent observations.
    """
    a = np.asarray(challenger_by_seed)
    b = np.asarray(baseline_by_seed)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("expected equal arrays shaped (seeds, queries)")
    hit_a = ((a > 0) & (a <= k)).astype(np.int8)
    hit_b = ((b > 0) & (b <= k)).astype(np.int8)
    per_query = (hit_a - hit_b).mean(axis=0)
    values, counts = np.unique(per_query, return_counts=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(per_query), counts / counts.sum(), size=reps)
    boot = draws @ values / len(per_query)
    # Query-cluster sign-flip randomization test for the null of exchangeability.
    perm_sum = np.zeros(reps, dtype=float)
    for value, count in zip(values, counts):
        if value:
            positive = rng.binomial(int(count), .5, size=reps)
            perm_sum += value * (2 * positive - count)
    observed = float(per_query.mean())
    permutation_p = float((np.sum(np.abs(perm_sum / len(per_query)) >=
                                  abs(observed)) + 1) / (reps + 1))
    per_seed = {}
    for row in range(a.shape[0]):
        ha, hb = hit_a[row].astype(bool), hit_b[row].astype(bool)
        pos = int(np.sum(ha & ~hb)); neg = int(np.sum(~ha & hb))
        per_seed[str(row)] = {
            "difference": float(ha.mean() - hb.mean()),
            "challenger_only": pos, "baseline_only": neg,
            "mcnemar_exact_p": float(binomtest(pos, pos + neg, .5).pvalue)
            if pos + neg else 1.0}
    return {
        "unit": "test query (seed-averaged matched-seed hit difference)",
        "n_queries": int(a.shape[1]), "n_seeds": int(a.shape[0]),
        "challenger_recall_mean": float(hit_a.mean(axis=1).mean()),
        "baseline_recall_mean": float(hit_b.mean(axis=1).mean()),
        "difference": observed,
        "cluster_bootstrap_ci95": [float(x) for x in np.quantile(boot, [.025, .975])],
        "cluster_sign_flip_p": permutation_p,
        "bootstrap_repetitions": reps, "per_seed_mcnemar": per_seed}
