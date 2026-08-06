"""Faithful, protocol-adapted V-SKNN and STAN neighborhood baselines.

The scoring equations follow the public ``rn5l/session-rec`` reference
implementation.  The benchmark loaders do not retain wall-clock session
timestamps, so STAN receives deterministic session-order timestamps; callers
must disclose this protocol adaptation when reporting the result.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class NeighborhoodConfig:
    method: str = "vsknn"
    k: int = 100
    sample_size: int = 5000
    weighting: str = "quadratic"
    score_weighting: str = "div"
    lambda_spw: float = 1.02
    lambda_snh: float | None = None
    lambda_inh: float = 2.05
    exclude_seen: bool = True


class NeighborhoodIndex:
    def __init__(self, sessions: Mapping[str, Sequence[int]], n_items: int):
        self.n_items = int(n_items)
        self.sessions: list[tuple[int, ...]] = []
        self.session_sets: list[frozenset[int]] = []
        self.postings: dict[int, list[int]] = defaultdict(list)
        self.popularity: dict[int, int] = defaultdict(int)
        for seq0 in sessions.values():
            seq = tuple(int(x) for x in seq0 if 0 < int(x) < self.n_items)
            if not seq:
                continue
            sid = len(self.sessions)
            self.sessions.append(seq)
            item_set = frozenset(seq)
            self.session_sets.append(item_set)
            for item in item_set:
                self.postings[item].append(sid)
            for item in seq:
                self.popularity[item] += 1
        self.pop_rank = [x for x, _ in sorted(
            self.popularity.items(), key=lambda p: (-p[1], p[0]))]

    @staticmethod
    def _vsknn_weight(pos: int, length: int, name: str) -> float:
        if name == "same":
            return 1.0
        if name == "linear":
            return max(0.0, 1.0 - .1 * (length - pos))
        if name == "div":
            return pos / max(length, 1)
        if name == "log":
            return 1.0 / math.log10((length - pos) + 1.7)
        if name == "quadratic":
            return (pos / max(length, 1)) ** 2
        raise KeyError(name)

    @staticmethod
    def _score_decay(step: int, name: str) -> float:
        if name == "same":
            return 1.0
        if name == "linear":
            return max(0.0, 1.0 - .1 * step)
        if name == "div":
            return 1.0 / step
        if name == "log":
            return 1.0 / math.log10(step + 1.7)
        if name == "quadratic":
            return 1.0 / (step * step)
        raise KeyError(name)

    def _candidates(self, context: Sequence[int], sample_size: int) -> set[int]:
        candidates: set[int] = set()
        for item in context:
            candidates.update(self.postings.get(int(item), ()))
        if sample_size > 0 and len(candidates) > sample_size:
            # The reference implementation's ``recent`` sampler.  Session IDs
            # preserve loader order and act as deterministic recency ranks.
            candidates = set(heapq.nlargest(sample_size, candidates))
        return candidates

    def _vsknn_similarity(self, context: Sequence[int], sid: int,
                           cfg: NeighborhoodConfig) -> float:
        pos = {int(item): self._vsknn_weight(i, len(context), cfg.weighting)
               for i, item in enumerate(context, 1)}
        return sum(pos.get(item, 0.0) for item in self.session_sets[sid]) / max(len(pos), 1)

    def _stan_similarity(self, context: Sequence[int], sid: int,
                          cfg: NeighborhoodConfig) -> float:
        pos = {int(item): math.exp((i - len(context)) / cfg.lambda_spw)
               for i, item in enumerate(context, 1)}
        dot = sum(pos.get(item, 0.0) for item in self.session_sets[sid])
        denom = math.sqrt(sum(x * x for x in pos.values())) * math.sqrt(
            max(len(self.session_sets[sid]), 1))
        similarity = dot / denom if denom else 0.0
        if cfg.lambda_snh is not None:
            age = max(len(self.sessions) - sid, 0)
            similarity *= math.exp(-age / cfg.lambda_snh)
        return similarity

    def predict_one(self, context0: Sequence[int], cfg: NeighborhoodConfig,
                    topk: int = 20) -> list[int]:
        context = [int(x) for x in context0 if 0 < int(x) < self.n_items]
        blocked = set(context) if cfg.exclude_seen else set()
        candidates = self._candidates(context, cfg.sample_size)
        if cfg.method == "vsknn":
            # Positional weights depend only on the query.  The original
            # implementation rebuilt this dictionary once per candidate
            # session, which is exactly equivalent but dominates full-catalog
            # audit time.
            pos = {
                int(item): self._vsknn_weight(i, len(context), cfg.weighting)
                for i, item in enumerate(context, 1)
            }
            denom = max(len(pos), 1)
            neighbors = heapq.nlargest(
                cfg.k,
                ((sum(pos.get(item, 0.0)
                      for item in self.session_sets[sid]) / denom, sid)
                 for sid in candidates),
                key=lambda p: (p[0], p[1]),
            )
        elif cfg.method == "stan":
            pos = {
                int(item): math.exp((i - len(context)) / cfg.lambda_spw)
                for i, item in enumerate(context, 1)
            }
            context_norm = math.sqrt(sum(value * value for value in pos.values()))

            def stan_similarity(sid: int) -> float:
                dot = sum(pos.get(item, 0.0) for item in self.session_sets[sid])
                denom = context_norm * math.sqrt(
                    max(len(self.session_sets[sid]), 1)
                )
                similarity = dot / denom if denom else 0.0
                if cfg.lambda_snh is not None:
                    age = max(len(self.sessions) - sid, 0)
                    similarity *= math.exp(-age / cfg.lambda_snh)
                return similarity

            neighbors = heapq.nlargest(
                cfg.k,
                ((stan_similarity(sid), sid) for sid in candidates),
                key=lambda p: (p[0], p[1]),
            )
        else:
            raise KeyError(cfg.method)
        scores: dict[int, float] = defaultdict(float)
        ctx_set = set(context)
        for similarity, sid in neighbors:
            if similarity <= 0:
                continue
            sequence = self.sessions[sid]
            if cfg.method == "vsknn":
                step = next((i for i, x in enumerate(reversed(context), 1)
                             if x in self.session_sets[sid]), len(context) + 1)
                decay = self._score_decay(step, cfg.score_weighting)
                for item in self.session_sets[sid]:
                    if item not in blocked:
                        scores[item] += similarity * decay
            else:
                last_match = max((i + 1 for i, x in enumerate(sequence)
                                  if x in ctx_set), default=1)
                last_pos: dict[int, int] = {}
                for i, item in enumerate(sequence, 1):
                    last_pos[item] = i
                for item, item_pos in last_pos.items():
                    if item not in blocked:
                        scores[item] += similarity * math.exp(
                            -abs(item_pos - last_match) / cfg.lambda_inh)
        ranking = [item for item, _ in heapq.nlargest(
            topk, scores.items(), key=lambda p: (p[1], -p[0]))]
        if len(ranking) < topk:
            chosen = set(ranking)
            ranking.extend(x for x in self.pop_rank
                           if x not in blocked and x not in chosen)
        return ranking[:topk]

    def predict(self, queries: Mapping[str, Mapping[str, Sequence[int]]],
                cfg: NeighborhoodConfig, topk: int = 20) -> dict[str, list[int]]:
        return {str(uid): self.predict_one(q.get("context", ()), cfg, topk)
                for uid, q in queries.items()}
