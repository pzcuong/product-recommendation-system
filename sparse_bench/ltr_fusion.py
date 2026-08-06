"""Leakage-aware learning-to-rank fusion for session recommendation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_NAMES = (
    "memory_rr", "neural_rr", "memory_depth", "neural_depth",
    "consensus", "log_frequency", "in_context", "recency_copy",
    "item_is_tail", "last_is_tail", "log_context_length", "is_last",
    "rank_agreement",
)


@dataclass
class CandidateMatrix:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    uids: list[str]
    candidates: list[list[int]]
    offsets: np.ndarray


def _fold(uid: str, folds: int) -> int:
    digest = hashlib.blake2b(str(uid).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % folds


def build_candidate_matrix(
    memory: Mapping[str, Sequence[int]],
    neural: Mapping[str, Sequence[int]],
    queries: Mapping[str, Mapping[str, Sequence[int]]],
    frequency: Counter,
    tail_score_indices: set[int],
    topn: int = 120,
) -> CandidateMatrix:
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    uids: list[str] = []
    candidates_by_query: list[list[int]] = []
    max_log_freq = math.log1p(max(frequency.values(), default=1))
    for uid0, query in queries.items():
        uid = str(uid0)
        mem = [int(x) for x in memory[uid][:topn]]
        neu = [int(x) for x in neural[uid][:topn]]
        candidate = list(dict.fromkeys(mem + neu))
        mem_rank = {item: rank for rank, item in enumerate(mem, 1)}
        neu_rank = {item: rank for rank, item in enumerate(neu, 1)}
        context = [int(x) for x in query.get("context", [])]
        context_set = set(context)
        target = int(query.get("targets", [-1])[0])
        last = context[-1] if context else -1
        last_tail = float(last > 0 and last - 1 in tail_score_indices)
        reverse_position = {}
        for distance, item in enumerate(reversed(context)):
            reverse_position.setdefault(item, distance)
        for item in candidate:
            mr = mem_rank.get(item, topn + 1)
            nr = neu_rank.get(item, topn + 1)
            has_m = mr <= topn
            has_n = nr <= topn
            rows.append([
                1.0 / (20.0 + mr) if has_m else 0.0,
                1.0 / (20.0 + nr) if has_n else 0.0,
                (topn + 1 - mr) / topn if has_m else 0.0,
                (topn + 1 - nr) / topn if has_n else 0.0,
                float(has_m and has_n),
                math.log1p(frequency.get(item, 0)) / max_log_freq,
                float(item in context_set),
                (1.0 / (1.0 + reverse_position[item])
                 if item in reverse_position else 0.0),
                float(item - 1 in tail_score_indices),
                last_tail,
                math.log1p(len(context)),
                float(item == last),
                (1.0 / (1.0 + abs(mr - nr)) if has_m and has_n else 0.0),
            ])
            labels.append(int(item == target))
        groups.append(len(candidate))
        uids.append(uid)
        candidates_by_query.append(candidate)
    offsets = np.concatenate(([0], np.cumsum(groups, dtype=np.int64)))
    return CandidateMatrix(
        x=np.asarray(rows, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int8),
        groups=np.asarray(groups, dtype=np.int32),
        uids=uids,
        candidates=candidates_by_query,
        offsets=offsets,
    )


class RankFusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(len(FEATURE_NAMES)),
            nn.Linear(len(FEATURE_NAMES), 48), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(48, 24), nn.GELU(),
            nn.Linear(24, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def _positive_queries(matrix: CandidateMatrix,
                      query_indices: Sequence[int]) -> list[int]:
    kept = [int(i) for i in query_indices if matrix.y[
        matrix.offsets[i]:matrix.offsets[i + 1]].any()]
    if not kept:
        raise ValueError("no training query has a positive candidate")
    return kept


def fit_ranker(matrix: CandidateMatrix, query_indices: Sequence[int],
               seed: int = 42, epochs: int = 16,
               batch_size: int = 64) -> RankFusionMLP:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    indices = _positive_queries(matrix, query_indices)
    model = RankFusionMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3,
                                  weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(indices)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            width = max(int(matrix.groups[i]) for i in batch)
            x = torch.zeros((len(batch), width, len(FEATURE_NAMES)),
                            dtype=torch.float32)
            mask = torch.zeros((len(batch), width), dtype=torch.bool)
            targets = torch.empty(len(batch), dtype=torch.long)
            for row, index in enumerate(batch):
                left, right = matrix.offsets[index:index + 2]
                count = right - left
                x[row, :count] = torch.from_numpy(matrix.x[left:right])
                mask[row, :count] = True
                targets[row] = int(np.flatnonzero(matrix.y[left:right])[0])
            scores = model(x).masked_fill(~mask, -torch.inf)
            loss = F.cross_entropy(scores, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


@torch.no_grad()
def score_ranker(model: RankFusionMLP, matrix: CandidateMatrix,
                 batch_rows: int = 65536) -> np.ndarray:
    model.eval()
    output = np.empty(len(matrix.x), dtype=np.float32)
    for start in range(0, len(matrix.x), batch_rows):
        end = min(start + batch_rows, len(matrix.x))
        output[start:end] = model(torch.from_numpy(
            matrix.x[start:end])).cpu().numpy()
    return output


def _rank_with_scores(matrix: CandidateMatrix, scores: np.ndarray,
                      query_indices: Sequence[int], topk: int = 20):
    predictions = {}
    for index in query_indices:
        start, end = matrix.offsets[index:index + 2]
        order = np.argsort(-scores[start:end], kind="stable")[:topk]
        predictions[matrix.uids[index]] = [
            matrix.candidates[index][int(row)] for row in order]
    return predictions


def cross_validated_fit(matrix: CandidateMatrix, folds: int = 5,
                        seed: int = 42):
    oof_scores = np.full(len(matrix.y), -np.inf, dtype=np.float32)
    all_indices = np.arange(len(matrix.uids))
    for fold in range(folds):
        valid = [i for i, uid in enumerate(matrix.uids)
                 if _fold(uid, folds) == fold]
        valid_set = set(valid)
        train = [i for i in all_indices if i not in valid_set]
        model = fit_ranker(matrix, train, seed + fold)
        fold_scores = score_ranker(model, matrix)
        for index in valid:
            start, end = matrix.offsets[index:index + 2]
            oof_scores[start:end] = fold_scores[start:end]
        print(f"[CEARF-LTR] OOF fold={fold + 1}/{folds} "
              f"train={len(train)} valid={len(valid)}", flush=True)
    predictions = _rank_with_scores(
        matrix, oof_scores, range(len(matrix.uids)), topk=120)
    final_model = fit_ranker(matrix, range(len(matrix.uids)), seed)
    return final_model, predictions


def predict_ranker(model: RankFusionMLP, matrix: CandidateMatrix,
                   topk: int = 20):
    scores = score_ranker(model, matrix)
    return _rank_with_scores(matrix, scores, range(len(matrix.uids)), topk)
