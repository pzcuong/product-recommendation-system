"""Per-query adaptive β router for CEARF-N.

Two routers, both trained on VALIDATION queries only and applied to test queries
via deterministic features (no test-label access):

1. ``BucketedRouter`` — discretises each query into one of 12 buckets formed
   from session-length × last-item-popularity × component-agreement, then
   selects one β per bucket from the existing 21-point grid. Empty buckets
   fall back to the coarse regime β.

2. ``ContinuousRouter`` — regresses a continuous β from the same features using
   a regularised ridge head trained on the per-validation-query target
   ``min β that places the target in the fused top-20``. Router-family
   selection is performed on a disjoint validation holdout by the runner.

Both routers reuse the raw memory + neural candidate lists that the regime
router already produces; they only change the final β-blend step.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from run_cearfn import BETAS, fuse


# ---------------------------------------------------------------------------
# Feature extraction — shared by both routers.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QueryFeatures:
    length: int
    log_last_pop: float
    last_is_tail: int
    agreement: int            # 0..3 components agreeing on top-5
    transition_mass_norm: float
    session_mass_norm: float


def _agreement_top5(rankings: Sequence[Sequence[int]]) -> int:
    """Count components whose top-5 share the modal item."""
    if not rankings:
        return 0
    votes: dict[int, int] = defaultdict(int)
    for ranking in rankings:
        for item in ranking[:5]:
            if int(item) > 0:
                votes[int(item)] += 1
    if not votes:
        return 0
    return max(votes.values())


def extract_features(context: Sequence[int], memory_components: Sequence[Sequence[int]],
                     item_freq, head_items: set[int] | None = None) -> QueryFeatures:
    """Build router features from a single query.

    `memory_components` is the (transition, session, popularity) triple returned
    by `CEARFIndex.component_rankings`. `head_items` is the popularity head set
    from `popularity_partition`; if None the tail flag is 0 by convention.
    """
    context = [int(x) for x in context if int(x) > 0]
    length = len(context)
    if context:
        last = context[-1]
        log_last_pop = math.log1p(item_freq.get(last, 0))
        last_is_tail = 0 if (head_items is None or last in head_items) else 1
    else:
        last = 0
        log_last_pop = 0.0
        last_is_tail = 1
    agreement = _agreement_top5(memory_components)
    # Mass diagnostics: how peaked are the per-component score distributions.
    # We approximate with the reciprocal-rank mass of the top-5 of each
    # component, normalised so values live in [0, 1].
    def mass(ranking: Sequence[int]) -> float:
        top = [int(x) for x in ranking[:5] if int(x) > 0]
        if not top:
            return 0.0
        return float(sum(1.0 / (20.0 + rank) for rank in range(1, len(top) + 1))) / 5.0

    trans_mass = mass(memory_components[0]) if len(memory_components) > 0 else 0.0
    sess_mass = mass(memory_components[1]) if len(memory_components) > 1 else 0.0
    return QueryFeatures(
        length=length, log_last_pop=log_last_pop, last_is_tail=last_is_tail,
        agreement=agreement, transition_mass_norm=trans_mass,
        session_mass_norm=sess_mass)


def feature_vector(f: QueryFeatures) -> np.ndarray:
    """Numeric feature row consumed by the continuous router."""
    length_bucket_mid = 0.0 if f.length <= 2 else (1.0 if f.length <= 7 else 2.0)
    return np.asarray([
        f.length,
        math.log1p(f.length),
        f.log_last_pop,
        f.last_is_tail,
        length_bucket_mid,
        f.agreement / 3.0,
        f.transition_mass_norm,
        f.session_mass_norm,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Bucketed router.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BucketKey:
    length_bucket: str       # "short" | "mid" | "long"
    pop_bucket: str          # "head" | "tail"
    agreement_bucket: str    # "low" | "high"


def bucket_of(features: QueryFeatures) -> BucketKey:
    if features.length <= 2:
        length_bucket = "short"
    elif features.length <= 7:
        length_bucket = "mid"
    else:
        length_bucket = "long"
    pop_bucket = "tail" if features.last_is_tail else "head"
    agreement_bucket = "high" if features.agreement >= 2 else "low"
    return BucketKey(length_bucket, pop_bucket, agreement_bucket)


class BucketedRouter:
    """Selects one β per (length, popularity, agreement) bucket on validation."""

    def __init__(self, fallback_beta: dict[str, float] | None = None):
        self.beta_per_bucket: dict[BucketKey, float] = {}
        self.report_per_bucket: dict[BucketKey, dict] = {}
        # Fallback is the coarse regime β keyed by "short"/"long".
        self.fallback_beta = fallback_beta or {"short": 0.0, "long": 0.0}

    def fit(self, queries: dict, memory: dict[str, list[int]],
            neural: dict[str, list[int]], keys: list[str], features: dict[str, QueryFeatures]) -> dict:
        """Select β per bucket using validation queries only."""
        buckets: dict[BucketKey, list[str]] = defaultdict(list)
        for uid in keys:
            buckets[bucket_of(features[uid])].append(uid)
        for bucket, members in buckets.items():
            best = None
            for beta in BETAS:
                hits6 = 0
                hits20 = 0
                for uid in members:
                    target = int(queries[uid]["targets"][0])
                    pred = fuse(memory[uid], neural[uid], beta)
                    rank = next((r for r, item in enumerate(pred, 1) if item == target), None)
                    if rank is not None:
                        hits20 += 1
                        if rank <= 6:
                            hits6 += 1
                n = max(len(members), 1)
                utility = 0.5 * hits6 / n + 0.5 * hits20 / n
                candidate = (utility, hits20 / n, hits6 / n, -beta, beta)
                if best is None or candidate[:4] > best[:4]:
                    best = candidate
            assert best is not None
            self.beta_per_bucket[bucket] = best[-1]
            self.report_per_bucket[bucket] = {
                "beta": best[-1], "n": len(members),
                "utility": best[0], "recall@6": best[2], "recall@20": best[1],
            }
        return {f"{k.length_bucket}_{k.pop_bucket}_{k.agreement_bucket}": v
                for k, v in self.report_per_bucket.items()}

    def beta_for(self, features: QueryFeatures) -> float:
        bucket = bucket_of(features)
        if bucket in self.beta_per_bucket:
            return self.beta_per_bucket[bucket]
        # Empty bucket at fit time: fall back to the coarse regime β.
        regime = "short" if features.length <= 2 else "long"
        return self.fallback_beta.get(regime, 0.0)


# ---------------------------------------------------------------------------
# Continuous router.
# ---------------------------------------------------------------------------
class ContinuousRouter:
    """Closed-form ridge regression from features → min-β-for-hit@20.

    Target semantics: for each validation query we find the smallest β in the
    grid such that the fused top-20 contains the target; if no β hits, the
    target is set to the largest grid value (1.0) to express "trust the neural
    side as a last resort"; if every β hits, the target is 0.0 ("memory alone
    suffices").
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    @staticmethod
    def _target_for_query(memory_row, neural_row, target: int) -> float:
        for beta in BETAS:
            pred = fuse(memory_row, neural_row, beta)
            if any(int(x) == target for x in pred):
                return float(beta)
        return 1.0

    def fit(self, queries: dict, memory: dict[str, list[int]],
            neural: dict[str, list[int]], keys: list[str],
            features: dict[str, QueryFeatures]) -> dict:
        X = np.stack([feature_vector(features[uid]) for uid in keys], axis=0)
        y = np.asarray([self._target_for_query(
            memory[uid], neural[uid], int(queries[uid]["targets"][0]))
            for uid in keys], dtype=np.float32)
        # Closed-form ridge with bias column.
        X_aug = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)
        reg = self.alpha * np.eye(X_aug.shape[1], dtype=np.float32)
        reg[-1, -1] = 0.0  # do not penalise the intercept
        self.coef_ = np.linalg.solve(
            X_aug.T @ X_aug + reg, X_aug.T @ y)
        self.intercept_ = float(self.coef_[-1])
        self.coef_ = self.coef_[:-1]
        return {
            "n_train": int(len(keys)),
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
            "alpha": float(self.alpha),
        }

    def beta_for(self, features: QueryFeatures) -> float:
        if self.coef_ is None:
            return 0.0
        x = feature_vector(features)
        beta = float(x @ self.coef_ + self.intercept_)
        return max(0.0, min(1.0, beta))
