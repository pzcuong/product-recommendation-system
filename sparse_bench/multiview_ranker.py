"""Validation-calibrated multi-view candidate fusion for DualTwin.

This module fuses factual graph retrieval, exposure popularity, semantic
retrieval and the counterfactual DualTwin ranking.  Hyperparameters are fit on
validation queries only and the frozen policy is then applied to test queries.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


def _rrf(experts: Mapping[str, Mapping[str, Sequence[int]]], uid: str,
         weights: Mapping[str, float], rrf_k: int, limit: int = 200) -> list[int]:
    scores: Dict[int, float] = {}
    votes: Dict[int, int] = {}
    best_rank: Dict[int, int] = {}
    for name, predictions in experts.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        for rank, item in enumerate(predictions.get(uid, [])[:limit], 1):
            item = int(item)
            scores[item] = scores.get(item, 0.0) + weight / (rrf_k + rank)
            votes[item] = votes.get(item, 0) + 1
            best_rank[item] = min(best_rank.get(item, rank), rank)
    # Consensus is a deterministic, very small tie-break rather than another
    # tuned weight. It favors actions supported by independent twin views.
    return sorted(scores, key=lambda x: (-scores[x], -votes[x], best_rank[x], x))


def _utility(ranking: Sequence[int], targets: Sequence[int]) -> float:
    target = set(targets)
    for rank, item in enumerate(ranking[:20], 1):
        if item in target:
            # Primary endpoint R@6, secondary endpoint graded rank quality.
            return (4.0 if rank <= 6 else 0.25) + 1.0 / math.log2(rank + 1)
    return 0.0


def fit_rank_fusion(experts: Mapping[str, Mapping[str, Sequence[int]]],
                    valid_queries: Mapping[str, Mapping[str, Sequence[int]]]) -> dict:
    """Fit a compact RRF policy on validation labels, never test labels."""
    required = ("SKNN", "MostPop", "Semantic", "PASGR", "DualTwin")
    names = [x for x in required if x in experts]
    if "SKNN" not in names:
        raise ValueError("SKNN factual view is required")
    # Fix factual weight to one to remove scale non-identifiability.  The grid
    # includes zero for optional views, so harmful modalities can be rejected.
    optional = [x for x in names if x != "SKNN"]
    values = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    def evaluate(weights, rrf_k):
        total = 0.0
        recall6 = 0
        for uid, query in valid_queries.items():
            ranking = _rrf(experts, uid, weights, rrf_k)
            total += _utility(ranking, query.get("targets", []))
            recall6 += bool(set(ranking[:6]) & set(query.get("targets", [])))
        return (total / max(len(valid_queries), 1),
                recall6 / max(len(valid_queries), 1),
                -sum(weights.values()), -rrf_k)

    # Coordinate search is deliberate: the validation set has thousands of
    # queries and each semantic list has 200 candidates. It reaches the same
    # compact policy family without an exponential grid over modalities.
    weights = {"SKNN": 1.0, **{name: 0.0 for name in optional}}
    rrf_k = 10
    best = (evaluate(weights, rrf_k), dict(weights), rrf_k)
    for _ in range(3):
        for name in optional:
            for value in values:
                candidate = dict(weights); candidate[name] = value
                key = evaluate(candidate, rrf_k)
                if key > best[0]:
                    best = (key, candidate, rrf_k)
            weights = dict(best[1])
        for candidate_k in (1, 5, 10, 20, 40, 80):
            key = evaluate(weights, candidate_k)
            if key > best[0]:
                best = (key, dict(weights), candidate_k)
        weights, rrf_k = dict(best[1]), best[2]
    return {"weights": best[1], "rrf_k": best[2],
            "valid_utility": best[0][0], "valid_recall@6": best[0][1]}


def apply_rank_fusion(policy: dict,
                      experts: Mapping[str, Mapping[str, Sequence[int]]],
                      queries: Mapping[str, Mapping[str, Sequence[int]]],
                      top_k: int = 200) -> Dict[str, list[int]]:
    return {uid: _rrf(experts, uid, policy["weights"], policy["rrf_k"])[:top_k]
            for uid in queries}


def evaluate_rank_fusion(policy: dict,
                         experts: Mapping[str, Mapping[str, Sequence[int]]],
                         queries: Mapping[str, Mapping[str, Sequence[int]]]) -> dict:
    """Evaluate a frozen policy on labels not used to fit that policy."""
    utility = 0.0
    recall6 = 0
    for uid, query in queries.items():
        ranking = _rrf(experts, uid, policy["weights"], policy["rrf_k"])
        utility += _utility(ranking, query.get("targets", []))
        recall6 += bool(set(ranking[:6]) & set(query.get("targets", [])))
    n = max(len(queries), 1)
    return {"utility": utility / n, "recall@6": recall6 / n, "n": len(queries)}
