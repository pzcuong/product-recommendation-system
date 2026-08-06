"""Validation-trained Verified Pareto Twin Router (VPTR).

The router chooses among already-trained ranking experts per observable context
stratum.  It never sees test targets.  A conservative shrinkage rule requires
enough validation evidence and positive Pareto utility before departing from
the factual expert.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _bucket(context: Sequence[int], head: set[int]) -> Tuple[str, str]:
    n = len(context)
    length = "short" if n <= 3 else "medium" if n <= 7 else "long"
    head_share = sum(x in head for x in context) / max(n, 1)
    exposure = "head" if head_share >= 0.5 else "nonhead"
    return length, exposure


def _utility(ranking: Sequence[int], targets: Iterable[int]) -> float:
    """Pareto proxy: prioritize R@6, then graded rank quality and coverage."""
    target = set(targets)
    for rank, item in enumerate(ranking[:20], 1):
        if item in target:
            return (2.0 if rank <= 6 else 0.5) + 1.0 / math.log2(rank + 1)
    return 0.0


def select_factual(expert_predictions: Mapping[str, Mapping[str, Sequence[int]]],
                   valid_queries: Mapping[str, Mapping[str, Sequence[int]]],
                   candidates: Sequence[str] = ("SKNN", "MostPop")) -> str:
    """Select the production anchor on validation only, with stable tie-break."""
    available = [e for e in candidates if e in expert_predictions]
    if not available:
        raise ValueError("no factual candidate predictions")
    scores = {}
    for expert in available:
        scores[expert] = sum(
            _utility(expert_predictions[expert].get(uid, []), q.get("targets", []))
            for uid, q in valid_queries.items()) / max(len(valid_queries), 1)
    return max(available, key=lambda e: (scores[e], -available.index(e)))


def fit_router(expert_predictions: Mapping[str, Mapping[str, Sequence[int]]],
               valid_queries: Mapping[str, Mapping[str, Sequence[int]]],
               item_freq: Counter, factual_expert: str = "SKNN",
               min_samples: int = 40, shrinkage: float = 80.0) -> dict:
    if factual_expert not in expert_predictions:
        raise ValueError(f"missing factual expert {factual_expert}")
    ranked = [x for x, _ in item_freq.most_common()]
    head = set(ranked[:max(1, int(len(ranked) * 0.2))])
    sums = defaultdict(lambda: defaultdict(float))
    paired = defaultdict(lambda: defaultdict(list))
    global_paired = defaultdict(list)
    counts = Counter()
    global_sums = Counter()
    n_global = 0
    for uid, query in valid_queries.items():
        key = _bucket(query.get("context", []), head)
        counts[key] += 1
        n_global += 1
        rewards = {}
        for expert, predictions in expert_predictions.items():
            reward = _utility(predictions.get(uid, []), query.get("targets", []))
            rewards[expert] = reward
            sums[key][expert] += reward
            global_sums[expert] += reward
        factual_reward = rewards[factual_expert]
        for expert, reward in rewards.items():
            paired[key][expert].append(reward - factual_reward)
            global_paired[expert].append(reward - factual_reward)
    global_mean = {e: global_sums[e] / max(n_global, 1) for e in expert_predictions}
    factual_global = global_mean.get(factual_expert, 0.0)

    def lower_bound(values):
        if len(values) < 2:
            return float("-inf")
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return mean - 1.96 * math.sqrt(var / len(values))

    global_lcb = {e: lower_bound(v) for e, v in global_paired.items()}
    policy = {}
    diagnostics = {}
    for key, n in counts.items():
        factual_local = sums[key][factual_expert] / n
        best, best_score = factual_expert, factual_local
        for expert in expert_predictions:
            local = sums[key][expert] / n
            # Empirical-Bayes shrinkage prevents tiny strata from selecting a
            # noisy intervention policy.
            shrunk = (n * local + shrinkage * global_mean[expert]) / (n + shrinkage)
            factual_shrunk = (n * factual_local + shrinkage * factual_global) / (n + shrinkage)
            local_lcb = lower_bound(paired[key][expert])
            # Two-level verifier: an intervention must beat factual ranking
            # with paired 95% confidence both globally and in this stratum.
            if (n >= min_samples and global_lcb.get(expert, float("-inf")) > 0.0
                    and local_lcb > 0.0 and shrunk > factual_shrunk
                    and shrunk > best_score):
                best, best_score = expert, shrunk
        policy["|".join(key)] = best
        diagnostics["|".join(key)] = {"n": n, "selected": best,
                                         "scores": dict(sums[key]),
                                         "lcb": {e: lower_bound(v)
                                                 for e, v in paired[key].items()}}
    return {"policy": policy, "head": sorted(head), "factual": factual_expert,
            "diagnostics": diagnostics, "global_lcb": global_lcb}


def route(router: dict, expert_predictions: Mapping[str, Mapping[str, Sequence[int]]],
          queries: Mapping[str, Mapping[str, Sequence[int]]]) -> Dict[str, List[int]]:
    head = set(router["head"])
    factual = router["factual"]
    output = {}
    for uid, query in queries.items():
        key = "|".join(_bucket(query.get("context", []), head))
        expert = router["policy"].get(key, factual)
        output[uid] = list(expert_predictions.get(expert, {}).get(uid,
                           expert_predictions[factual].get(uid, [])))
    return output
