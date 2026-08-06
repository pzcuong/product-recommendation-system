"""Variable-order intent memory for top-rank session recommendation."""
from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import math
from typing import Mapping, Sequence


class MultiGranularIntentIndex:
    """Backoff suffix and segmented co-occurrence evidence.

    Orders 1/2/3 capture immediate, short-pattern, and longer-pattern intent.
    A recency-weighted segment view supplies candidates when an exact suffix is
    sparse. All statistics are train-only and preserve repeated-item targets.
    """

    def __init__(self, sessions: Mapping[str, Sequence[int]], n_items: int,
                 max_order: int = 3, segment: int = 4):
        self.n_items = int(n_items)
        self.max_order = int(max_order)
        self.segment = int(segment)
        self.ngrams = [defaultdict(Counter)
                       for _ in range(self.max_order + 1)]
        self.segment_next = defaultdict(Counter)
        self.popularity = Counter()
        for sequence0 in sessions.values():
            sequence = [int(x) for x in sequence0
                        if 0 < int(x) < self.n_items]
            self.popularity.update(sequence)
            for position in range(1, len(sequence)):
                target = sequence[position]
                for order in range(1, min(self.max_order, position) + 1):
                    key = tuple(sequence[position - order:position])
                    self.ngrams[order][key][target] += 1
                for distance in range(1, min(self.segment, position) + 1):
                    source = sequence[position - distance]
                    self.segment_next[source][target] += 1.0 / distance
        self.pop_rank = [item for item, _ in self.popularity.most_common()]

    def scores(self, context0: Sequence[int]):
        context = [int(x) for x in context0
                   if 0 < int(x) < self.n_items]
        scores = defaultdict(float)
        evidence = defaultdict(int)
        # Reliability-weighted variable-order Markov backoff. Higher orders
        # receive more weight only when their observed support is sufficient.
        for order in range(min(self.max_order, len(context)), 0, -1):
            counter = self.ngrams[order].get(tuple(context[-order:]))
            if not counter:
                continue
            support = sum(counter.values())
            reliability = support / (support + 4.0 * order)
            scale = order * reliability
            for item, count in counter.items():
                probability = count / support
                scores[item] += scale * math.log1p(20.0 * probability)
                evidence[item] += 1
        # Segmented co-occurrence view with explicit recency decay.
        for age, source in enumerate(reversed(context[-self.segment:])):
            counter = self.segment_next.get(source)
            if not counter:
                continue
            support = sum(counter.values())
            scale = 0.35 * math.exp(-0.45 * age)
            for item, count in counter.items():
                scores[item] += scale * math.log1p(count) / math.sqrt(support)
                evidence[item] += 1
        for item, views in evidence.items():
            if views >= 2:
                scores[item] *= 1.0 + 0.06 * (views - 1)
        return scores

    def predict_one(self, context: Sequence[int], topk: int = 120):
        ranking = [item for _, item in heapq.nlargest(
            topk, ((score, item) for item, score in self.scores(context).items()),
            key=lambda pair: (pair[0], -pair[1]))]
        if len(ranking) < topk:
            seen = set(ranking)
            ranking.extend(item for item in self.pop_rank if item not in seen)
        return ranking[:topk]

    def predict(self, queries, topk: int = 120, progress: str | None = None):
        output = {}
        total = len(queries)
        for row, (uid, query) in enumerate(queries.items(), 1):
            output[str(uid)] = self.predict_one(query.get("context", []), topk)
            if progress and row % 10000 == 0:
                print(f"[MGIR] {progress} predicted={row}/{total}", flush=True)
        return output


def fuse_three(memory, neural, granular, weights, topk=20, constant=20.0):
    score = defaultdict(float)
    for weight, ranking in zip(weights, (memory, neural, granular)):
        for rank, item in enumerate(ranking, 1):
            score[int(item)] += float(weight) / (constant + rank)
    return [item for item, _ in sorted(
        score.items(), key=lambda pair: (-pair[1], pair[0]))[:topk]]
