"""Statistically guarded candidate selection (DESIGN_CASM.md §3).

Replaces argmax validation gating with a declared rule: paired per-query
one-sided tests of every candidate against the validation-best, BH-FDR across
the candidate family, then selection of the least complex candidate not
detectably worse. Also implements the two comparator rules (argmax, 1-SE) so
all three run on identical per-query artifacts.

Inputs are per-query binary hit vectors (and reciprocal ranks for the
utility) on validation queries only; the module never touches test labels.
Everything returned is a plain JSON-serialisable dict (the audit record).

Importing this module has no side effects; dependencies are numpy + scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import binom

ROUTER_RANK = {None: 0, "": 0, "constant": 0, "regime": 1,
               "bucketed": 2, "continuous": 3}
DEFAULT_Q = 0.10


@dataclass(frozen=True)
class Candidate:
    """One serving configuration in the frozen family (§3.1, §5).

    ``hits`` are per-query binary outcomes (hit@20) on validation queries;
    ``utilities`` are per-query utility values 0.5*(hit@6 + hit@20) used for
    the comparator ranking and the 1-SE bootstrap. ``active_components`` is
    the nonzero-profile-slot count (+1 if PASGR on), ``router`` the family
    name (constant < regime < bucketed < continuous), ``total_weights`` the
    number of nonzero profile weights across both regimes.
    """
    name: str
    hits: tuple[int, ...]
    utilities: tuple[float, ...]
    active_components: int
    router: str | None = None
    total_weights: int = 0
    position: int = 0          # order in the declared candidate list

    def complexity(self) -> tuple[int, int, int]:
        """Lexicographic complexity tuple (§3.4): smaller = simpler."""
        return (int(self.active_components),
                ROUTER_RANK.get(self.router, 3),
                int(self.total_weights))


def candidate_from_ranks(name: str, ranks: np.ndarray, *,
                         active_components: int, router: str | None = None,
                         total_weights: int = 0, position: int = 0,
                         k_hit: int = 20) -> Candidate:
    """Build a Candidate from a rank-at-20 vector (0 = miss, 1..20 = rank)."""
    ranks = np.asarray(ranks)
    hit20 = (ranks > 0) & (ranks <= k_hit)
    hit6 = (ranks > 0) & (ranks <= 6)
    utility = 0.5 * hit6.astype(np.float64) + 0.5 * hit20.astype(np.float64)
    return Candidate(name=name, hits=tuple(int(x) for x in hit20),
                     utilities=tuple(float(x) for x in utility),
                     active_components=active_components, router=router,
                     total_weights=total_weights, position=position)


def mcnemar_one_sided_p(n01: int, n10: int) -> float:
    """Exact one-sided McNemar (binomial sign test) p-value.

    H0: candidate is not worse than the comparator. n01 = comparator hits &
    candidate misses; n10 = candidate hits & comparator misses. Small n10
    relative to n01 is evidence the candidate is worse; p = P[Binom(n01+n10,
    1/2) <= n10] (lower tail). Zero discordant pairs -> p = 1 (declared
    handling, §3.2).
    """
    discordant = int(n01) + int(n10)
    if discordant == 0:
        return 1.0
    return float(binom.cdf(int(n10), discordant, 0.5))


def sign_flip_one_sided_p(differences: np.ndarray, reps: int = 20000,
                          seed: int = 20260726) -> float:
    """Paired sign-flip permutation test, one-sided (candidate worse).

    ``differences`` = per-query (candidate hit − comparator hit), possibly
    seed-averaged (fractional). Null: exchangeable signs. p = P[perm mean <=
    observed mean] with the +1 correction. Used instead of exact McNemar when
    hits are seed-averaged; the pilot is single-seed so McNemar is the
    default (§3.2).
    """
    differences = np.asarray(differences, dtype=np.float64)
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return 1.0
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(reps, len(nonzero))) * 2 - 1
    perm_means = (signs * np.abs(nonzero)).sum(axis=1) / len(differences)
    return float((np.sum(perm_means <= observed) + 1) / (reps + 1))


def bh_adjust(p_values: Sequence[float], q: float = DEFAULT_Q
              ) -> tuple[list[float], list[bool]]:
    """Benjamini–Hochberg: adjusted p-values + rejection flags at level q."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return [], []
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m, dtype=np.float64)
    running = 1.0
    for position in range(m - 1, -1, -1):
        rank = position + 1
        running = min(running, p[order[position]] * m / rank)
        adjusted[order[position]] = running
    rejected = adjusted <= q
    return [float(x) for x in adjusted], [bool(x) for x in rejected]


def by_adjust(p_values: Sequence[float], q: float = DEFAULT_Q
              ) -> tuple[list[float], list[bool]]:
    """Benjamini–Yekutieli (conservative sensitivity analysis, §3.3)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return [], []
    harmonic = float(sum(1.0 / k for k in range(1, m + 1)))
    adjusted, _ = bh_adjust(np.minimum(p * harmonic, 1.0), q)
    adjusted = [min(1.0, a) for a in adjusted]
    return adjusted, [a <= q for a in adjusted]


def _mean_utility(candidate: Candidate) -> float:
    return float(np.mean(candidate.utilities)) if candidate.utilities else 0.0


def _validation_best(candidates: Sequence[Candidate]) -> int:
    """Comparator c* = argmax validation utility; ties broken toward the
    simpler candidate, then earlier declared position (deterministic)."""
    best = None
    for index, candidate in enumerate(candidates):
        key = (_mean_utility(candidate),
               tuple(-x for x in candidate.complexity()),
               -candidate.position)
        if best is None or key > best[0]:
            best = (key, index)
    return best[1]


def select_argmax(candidates: Sequence[Candidate]) -> dict:
    """Status-quo rule: argmax utility, simplicity tie-break (§3.6a)."""
    index = _validation_best(candidates)
    return {"rule": "argmax", "selected": candidates[index].name,
            "selected_index": index,
            "utility": _mean_utility(candidates[index])}


def select_one_se(candidates: Sequence[Candidate], reps: int = 2000,
                  seed: int = 20260726) -> dict:
    """1-SE rule: simplest candidate within one standard error of the best;
    SE via query-level bootstrap of the mean utility of c* (§3.6b)."""
    best_index = _validation_best(candidates)
    best = candidates[best_index]
    utilities = np.asarray(best.utilities, dtype=np.float64)
    n = len(utilities)
    if n > 1:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n, size=(reps, n))
        se = float(np.std(utilities[draws].mean(axis=1), ddof=1))
    else:
        se = 0.0
    threshold = _mean_utility(best) - se
    eligible = [i for i, c in enumerate(candidates)
                if _mean_utility(c) >= threshold]
    chosen = min(eligible, key=lambda i: (candidates[i].complexity(),
                                          -_mean_utility(candidates[i]),
                                          candidates[i].position))
    return {"rule": "one_se", "selected": candidates[chosen].name,
            "selected_index": chosen, "se": se, "threshold": threshold,
            "eligible": [candidates[i].name for i in eligible],
            "utility": _mean_utility(candidates[chosen])}


def guarded_selection(candidates: Sequence[Candidate], q: float = DEFAULT_Q,
                      method: str = "mcnemar", reps: int = 20000,
                      seed: int = 20260726) -> dict:
    """FDR-guarded selection (§3.2–3.4). Returns a full audit record.

    method: 'mcnemar' (exact, binary single-seed hits) or 'sign_flip'
    (paired permutation on possibly seed-averaged hit differences).
    """
    if not candidates:
        raise ValueError("empty candidate family")
    lengths = {len(c.hits) for c in candidates}
    if len(lengths) != 1:
        raise ValueError(f"candidates disagree on n_queries: {lengths}")
    n_queries = lengths.pop()
    if n_queries == 0:
        raise ValueError("no validation queries")

    best_index = _validation_best(candidates)
    best = candidates[best_index]
    best_hits = np.asarray(best.hits, dtype=np.float64)

    tests: list[dict] = []
    order_tested: list[int] = []
    for index, candidate in enumerate(candidates):
        if index == best_index:
            continue
        hits = np.asarray(candidate.hits, dtype=np.float64)
        n01 = int(np.sum((best_hits > 0.5) & (hits <= 0.5)))
        n10 = int(np.sum((hits > 0.5) & (best_hits <= 0.5)))
        if method == "mcnemar":
            p_value = mcnemar_one_sided_p(n01, n10)
        elif method == "sign_flip":
            p_value = sign_flip_one_sided_p(hits - best_hits, reps=reps,
                                            seed=seed)
        else:
            raise ValueError(f"unknown method {method!r}")
        tests.append({"candidate": candidate.name, "n01": n01, "n10": n10,
                      "discordant": n01 + n10, "p_value": p_value})
        order_tested.append(index)

    p_values = [t["p_value"] for t in tests]
    bh_adjusted, bh_rejected = bh_adjust(p_values, q)
    by_adjusted, by_rejected = by_adjust(p_values, q)
    for test, adj, rej, adj2, rej2 in zip(tests, bh_adjusted, bh_rejected,
                                          by_adjusted, by_rejected):
        test["bh_adjusted_p"] = adj
        test["detectably_worse_bh"] = rej
        test["by_adjusted_p"] = adj2
        test["detectably_worse_by"] = rej2

    eligible = [best_index] + [
        index for index, rejected in zip(order_tested, bh_rejected)
        if not rejected]
    chosen = min(eligible, key=lambda i: (candidates[i].complexity(),
                                          -_mean_utility(candidates[i]),
                                          candidates[i].position))
    return {
        "rule": "guarded",
        "q": float(q),
        "method": method,
        "n_queries": int(n_queries),
        "comparator": best.name,
        "comparator_index": int(best_index),
        "comparator_utility": _mean_utility(best),
        "tests": tests,
        "eligible": [candidates[i].name for i in sorted(eligible)],
        "selected": candidates[chosen].name,
        "selected_index": int(chosen),
        "selected_utility": _mean_utility(candidates[chosen]),
        "selected_complexity": list(candidates[chosen].complexity()),
        "complexity_ranks": {c.name: list(c.complexity())
                             for c in candidates},
    }


def compare_selection_rules(candidates: Sequence[Candidate],
                            q: float = DEFAULT_Q, method: str = "mcnemar",
                            seed: int = 20260726) -> dict:
    """Run argmax, 1-SE, and guarded on identical inputs (§3.6)."""
    return {
        "argmax": select_argmax(candidates),
        "one_se": select_one_se(candidates, seed=seed),
        "guarded": guarded_selection(candidates, q=q, method=method,
                                     seed=seed),
    }
