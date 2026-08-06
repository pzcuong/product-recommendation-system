"""Cross-Evidence Adaptive Rank Fusion (CEARF).

CEARF is an explicit retrieval model for sparse session recommendation. It
combines three independently useful memories without test-label access:

* directed, recency-weighted transition memory;
* IDF-weighted neighbour-session memory;
* a conservative popularity fallback.

The caller supplies the holdout used to select fusion profiles separately for
short and long contexts. In the current dynamic-beta protocol that holdout is
a training-only profile-lock subset disjoint from gate calibration.
Component scores are converted to ranks before fusion, avoiding incomparable
score scales. This module intentionally has no dependency on the
repository-local CoDT/DualTwin experiments.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import math
from typing import Dict, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CEARFConfig:
    window: int = 4
    transition_topn: int = 200
    candidate_sessions: int = 120
    component_topn: int = 120
    rrf_constant: float = 20.0
    consensus_bonus: float = 0.12
    short_context: int = 2
    validation_fraction: float = 0.10
    validation_cap: int = 5000
    exclude_seen: bool = True


PROFILES: dict[str, tuple[float, float, float]] = {
    "transition": (1.0, 0.0, 0.0),
    "session": (0.0, 1.0, 0.0),
    "balanced": (0.50, 0.40, 0.10),
    "transition_session": (0.60, 0.40, 0.0),
    "session_transition": (0.35, 0.60, 0.05),
    "short_safe": (0.65, 0.20, 0.15),
}


def _stable_fraction(key: str) -> float:
    digest = hashlib.blake2b(str(key).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64 - 1)


def make_validation_split(
    sessions: Mapping[str, Sequence[int]], fraction: float = 0.10,
    cap: int = 5000,
) -> tuple[dict[str, list[int]], dict[str, dict[str, list[int]]]]:
    """Create a deterministic leave-last-out validation set.

    Held sessions contribute only their prefix to the tuning index. The target
    event and its incoming transition are absent, while the context remains
    available as realistic retrieval evidence. The final test index is rebuilt
    from all training sessions.
    """
    ordered = sorted(sessions, key=lambda x: _stable_fraction(str(x)))
    wanted = min(cap, max(1, int(len(ordered) * fraction)))
    held = set(ordered[:wanted])
    train: dict[str, list[int]] = {}
    valid: dict[str, dict[str, list[int]]] = {}
    for key, seq0 in sessions.items():
        seq = [int(x) for x in seq0 if int(x) > 0]
        if key in held and len(seq) >= 2:
            valid[str(key)] = {"context": seq[:-1], "targets": [seq[-1]]}
            train[str(key)] = seq[:-1]
        else:
            train[str(key)] = seq
    return train, valid


class CEARFIndex:
    def __init__(self, sessions: Mapping[str, Sequence[int]], n_items: int,
                 config: CEARFConfig = CEARFConfig()):
        self.n_items = int(n_items)
        self.config = config
        self.sessions: list[tuple[int, ...]] = []
        self.postings: dict[int, list[int]] = defaultdict(list)
        self.freq: Counter[int] = Counter()
        transition: dict[int, Counter[int]] = defaultdict(Counter)

        for seq0 in sessions.values():
            seq = tuple(int(x) for x in seq0 if 0 < int(x) < n_items)
            if not seq:
                continue
            sid = len(self.sessions)
            self.sessions.append(seq)
            unique = set(seq)
            self.freq.update(seq)
            for item in unique:
                self.postings[item].append(sid)
            for right in range(1, len(seq)):
                target = seq[right]
                for distance in range(1, min(config.window, right) + 1):
                    source = seq[right - distance]
                    if source != target or not config.exclude_seen:
                        transition[source][target] += 1.0 / distance

        n_sessions = max(len(self.sessions), 1)
        self.idf = {
            item: math.log1p(n_sessions / max(len(posting), 1))
            for item, posting in self.postings.items()
        }
        self.transition = {
            source: dict(counter.most_common(config.transition_topn))
            for source, counter in transition.items()
        }
        self.pop_rank = [item for item, _ in self.freq.most_common()
                         if 0 < item < n_items]

    def _transition_scores(self, context: Sequence[int]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        tail = [int(x) for x in context if 0 < int(x) < self.n_items][-8:]
        for age, source in enumerate(reversed(tail)):
            recency = math.exp(-0.35 * age)
            for item, count in self.transition.get(source, {}).items():
                scores[item] += recency * math.log1p(count) / math.sqrt(
                    1.0 + self.freq.get(item, 0))
        return scores

    def _session_scores(self, context: Sequence[int]) -> dict[int, float]:
        ctx = [int(x) for x in context if 0 < int(x) < self.n_items][-8:]
        ctx_set = set(ctx)
        candidate: dict[int, float] = defaultdict(float)
        for age, item in enumerate(reversed(ctx)):
            weight = self.idf.get(item, 0.0) * math.exp(-0.20 * age)
            for sid in self.postings.get(item, ()):
                candidate[sid] += weight
        best = heapq.nlargest(self.config.candidate_sessions,
                              candidate.items(), key=lambda x: x[1])
        scores: dict[int, float] = defaultdict(float)
        for sid, overlap_score in best:
            seq = self.sessions[sid]
            norm = overlap_score / math.sqrt(max(len(set(seq)), 1))
            # Prefer items occurring after the most recent matching context
            # item, while retaining a weaker whole-session vote.
            last_match = max((i for i, x in enumerate(seq) if x in ctx_set),
                             default=-1)
            for pos, item in enumerate(seq):
                if (self.config.exclude_seen and item in ctx_set) or item <= 0:
                    continue
                direction = 1.0 if pos > last_match else 0.25
                distance = max(pos - last_match, 1)
                scores[item] += norm * direction / math.sqrt(distance)
        return scores

    @staticmethod
    def _rank(scores: Mapping[int, float], blocked: set[int], topn: int) -> list[int]:
        eligible = ((float(score), int(item)) for item, score in scores.items()
                    if item not in blocked and item > 0 and np.isfinite(score))
        return [item for _, item in heapq.nlargest(topn, eligible)]

    def component_rankings(self, context: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
        blocked = ({int(x) for x in context if int(x) > 0}
                   if self.config.exclude_seen else set())
        n = self.config.component_topn
        transition = self._rank(self._transition_scores(context), blocked, n)
        session = self._rank(self._session_scores(context), blocked, n)
        popularity = [x for x in self.pop_rank if x not in blocked][:n]
        return transition, session, popularity

    def fuse_rankings(self, context: Sequence[int],
                      rankings: tuple[list[int], list[int], list[int]],
                      profile: tuple[float, float, float], topk: int = 20) -> list[int]:
        fused: dict[int, float] = defaultdict(float)
        votes: Counter[int] = Counter()
        for weight, ranking in zip(profile, rankings):
            if weight <= 0:
                continue
            for rank, item in enumerate(ranking, 1):
                fused[item] += weight / (self.config.rrf_constant + rank)
                votes[item] += 1
        for item, count in votes.items():
            if count >= 2:
                fused[item] *= 1.0 + self.config.consensus_bonus * (count - 1)
        blocked = ({int(x) for x in context if int(x) > 0}
                   if self.config.exclude_seen else set())
        ranking = self._rank(fused, blocked, max(topk, self.config.component_topn))
        if len(ranking) < topk:
            ranked = set(ranking)
            ranking.extend(x for x in self.pop_rank
                           if x not in blocked and x not in ranked)
        return ranking[:topk]

    def predict_one(self, context: Sequence[int], profile: tuple[float, float, float],
                    topk: int = 20) -> list[int]:
        return self.fuse_rankings(context, self.component_rankings(context),
                                  profile, topk)

    def predict(self, queries: Mapping[str, Mapping[str, Sequence[int]]],
                profiles: Mapping[str, tuple[float, float, float]],
                topk: int = 20, progress: str | None = None) -> dict[str, list[int]]:
        output = {}
        total = len(queries)
        for row, (uid, query) in enumerate(queries.items(), 1):
            context = query.get("context", [])
            regime = "short" if len(context) <= self.config.short_context else "long"
            output[str(uid)] = self.predict_one(context, profiles[regime], topk)
            if progress and row % 10000 == 0:
                print(f"[CEARF] {progress} predicted={row}/{total}", flush=True)
        return output


def recall_at(predictions: Mapping[str, Sequence[int]],
              queries: Mapping[str, Mapping[str, Sequence[int]]], k: int) -> float:
    if not queries:
        return 0.0
    hits = 0
    for uid, query in queries.items():
        targets = set(int(x) for x in query.get("targets", []))
        hits += bool(targets.intersection(predictions.get(str(uid), ())[:k]))
    return hits / len(queries)


def ranking_metrics(predictions: Mapping[str, Sequence[int]],
                    queries: Mapping[str, Mapping[str, Sequence[int]]],
                    cutoffs: Sequence[int] = (6, 10, 20)) -> dict[str, float]:
    output: dict[str, float] = {"n": len(queries)}
    for k in cutoffs:
        hits = 0
        ndcg = 0.0
        for uid, query in queries.items():
            targets = set(int(x) for x in query.get("targets", []))
            rank = next((rank for rank, item in enumerate(
                predictions.get(str(uid), ())[:k], 1) if item in targets), None)
            if rank is not None:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 1)
        output[f"recall@{k}"] = hits / max(len(queries), 1)
        output[f"ndcg@{k}"] = ndcg / max(len(queries), 1)
    return output


def tune_profiles(index: CEARFIndex,
                  validation: Mapping[str, Mapping[str, Sequence[int]]]
                  ) -> tuple[dict[str, tuple[float, float, float]], dict]:
    chosen: dict[str, tuple[float, float, float]] = {}
    report: dict = {}
    for regime in ("short", "long"):
        subset = {str(uid): q for uid, q in validation.items()
                  if (len(q.get("context", [])) <= index.config.short_context) ==
                  (regime == "short")}
        if not subset:
            name = "short_safe" if regime == "short" else "session_transition"
            chosen[regime] = PROFILES[name]
            report[regime] = {"profile": name, "n": 0, "score": None}
            continue
        cached = {
            uid: index.component_rankings(q.get("context", []))
            for uid, q in subset.items()
        }
        best = None
        for name, profile in PROFILES.items():
            predictions = {uid: index.fuse_rankings(
                q.get("context", []), cached[uid], profile, 20)
                for uid, q in subset.items()}
            r6 = recall_at(predictions, subset, 6)
            r20 = recall_at(predictions, subset, 20)
            # Early precision and breadth receive equal weight.
            score = 0.5 * r6 + 0.5 * r20
            candidate = (score, r20, r6, name, profile)
            if best is None or candidate > best:
                best = candidate
        assert best is not None
        chosen[regime] = best[-1]
        report[regime] = {"profile": best[-2], "n": len(subset),
                          "score": best[0], "recall@6": best[2],
                          "recall@20": best[1]}
    return chosen, report
