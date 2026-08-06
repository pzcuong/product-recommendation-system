"""Residual multi-view reranker for fixed MGCOT candidate sets."""
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
    "mgcot_z", "mgcot_rr", "mgcot_depth", "mgcot_gap",
    "neural_rr", "neural_depth", "memory_rr", "memory_depth",
    "granular_rr", "granular_depth", "view_consensus", "rank_agreement",
    "log_frequency", "item_is_tail", "in_context", "recency_copy",
    "is_last", "log_context_length",
)


@dataclass
class FixedCandidateMatrix:
    x: np.ndarray
    items: np.ndarray
    target_columns: np.ndarray
    uids: list[str]


def _rank_features(ranking, topn=120):
    ranks = {int(item): rank for rank, item in enumerate(ranking[:topn], 1)}
    return ranks


def build_matrix(items: np.ndarray, raw_scores: np.ndarray,
                 queries: Mapping[str, Mapping[str, Sequence[int]]],
                 neural, memory, granular, frequency: Counter,
                 tail_score_indices: set[int]) -> FixedCandidateMatrix:
    uids = [str(uid) for uid in queries]
    if len(uids) != len(items):
        raise ValueError("prediction/query row mismatch")
    rows, width = items.shape
    x = np.zeros((rows, width, len(FEATURE_NAMES)), dtype=np.float32)
    targets = np.full(rows, -1, dtype=np.int32)
    max_log_freq = math.log1p(max(frequency.values(), default=1))
    for row, uid in enumerate(uids):
        query = queries[uid]
        context = [int(v) for v in query["context"]]
        context_set = set(context)
        reverse = {}
        for distance, item in enumerate(reversed(context)):
            reverse.setdefault(item, distance)
        target = int(query["targets"][0])
        hits = np.flatnonzero(items[row] == target)
        if len(hits):
            targets[row] = int(hits[0])
        score = raw_scores[row].astype(np.float64)
        mean, std = score.mean(), max(score.std(), 1e-6)
        z = (score - mean) / std
        top = float(score[0])
        nr = _rank_features(neural[uid])
        mr = _rank_features(memory[uid])
        gr = _rank_features(granular[uid])
        for column, item0 in enumerate(items[row]):
            item = int(item0)
            ranks = (nr.get(item, 121), mr.get(item, 121),
                     gr.get(item, 121))
            present = [rank <= 120 for rank in ranks]
            valid_ranks = [rank for rank in ranks if rank <= 120]
            agreement = (1.0 / (1.0 + max(valid_ranks) - min(valid_ranks))
                         if len(valid_ranks) >= 2 else 0.0)
            features = [
                z[column], 1.0 / (1.0 + column), (120 - column) / 120,
                (top - float(score[column])) / max(abs(top), 1.0),
                1.0 / (20.0 + ranks[0]) if present[0] else 0.0,
                (121 - ranks[0]) / 120 if present[0] else 0.0,
                1.0 / (20.0 + ranks[1]) if present[1] else 0.0,
                (121 - ranks[1]) / 120 if present[1] else 0.0,
                1.0 / (20.0 + ranks[2]) if present[2] else 0.0,
                (121 - ranks[2]) / 120 if present[2] else 0.0,
                sum(present) / 3.0, agreement,
                math.log1p(frequency.get(item, 0)) / max_log_freq,
                float(item - 1 in tail_score_indices),
                float(item in context_set),
                1.0 / (1.0 + reverse[item]) if item in reverse else 0.0,
                float(context and item == context[-1]),
                math.log1p(len(context)) / math.log1p(70),
            ]
            x[row, column] = features
    return FixedCandidateMatrix(x=x, items=items.astype(np.int32),
                                target_columns=targets, uids=uids)


class ResidualRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = nn.Linear(len(FEATURE_NAMES), 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        base = x[..., 0] * self.log_scale.exp().clamp(0.25, 4.0)
        return base + self.residual(x).squeeze(-1)


def _device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def fit_ranker(matrix: FixedCandidateMatrix, indices, seed=42, epochs=12):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = np.asarray([i for i in indices if matrix.target_columns[i] >= 0],
                       dtype=np.int64)
    model = ResidualRanker().to(_device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3,
                                  weight_decay=2e-3)
    for _ in range(epochs):
        order = rng.permutation(train)
        for start in range(0, len(train), 64):
            rows = order[start:start + 64]
            x = torch.from_numpy(matrix.x[rows]).to(_device())
            y = torch.from_numpy(matrix.target_columns[rows]).long().to(_device())
            logits = model(x)
            residual = model.residual(x).squeeze(-1)
            loss = F.cross_entropy(logits, y) + 1e-3 * residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


@torch.no_grad()
def predict(model, matrix: FixedCandidateMatrix, indices=None, topk=20,
            entropy_threshold: float | None = None):
    indices = np.arange(len(matrix.uids)) if indices is None else np.asarray(indices)
    output = {}
    device = next(model.parameters()).device
    for start in range(0, len(indices), 512):
        rows = indices[start:start + 512]
        logits = model(torch.from_numpy(matrix.x[rows]).to(device))
        k = min(topk, logits.shape[1])
        order = torch.topk(logits, k, dim=1).indices
        if entropy_threshold is not None:
            base = torch.from_numpy(matrix.x[rows, :, 0]).to(device)
            probability = torch.softmax(base, dim=1)
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum(1)
            entropy = entropy / math.log(base.shape[1])
            uncertain = entropy >= float(entropy_threshold)
            base_order = torch.topk(base, k, dim=1).indices
            order = torch.where(uncertain[:, None], order, base_order)
        order = order.cpu().numpy()
        for row, columns in zip(rows, order):
            output[matrix.uids[int(row)]] = [
                int(matrix.items[int(row), column]) for column in columns]
    return output


def _fold(uid, folds=5):
    digest = hashlib.blake2b(uid.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % folds


def oof_predictions(matrix: FixedCandidateMatrix, folds=5, seed=42):
    output = {}
    all_rows = np.arange(len(matrix.uids))
    for fold in range(folds):
        valid = np.asarray([i for i, uid in enumerate(matrix.uids)
                            if _fold(uid, folds) == fold])
        mask = np.ones(len(all_rows), dtype=bool)
        mask[valid] = False
        model = fit_ranker(matrix, all_rows[mask], seed + fold)
        output.update(predict(model, matrix, valid, 20))
        print(f"[MGCOT-RERANK] fold={fold + 1}/{folds} valid={len(valid)}",
              flush=True)
    final_model = fit_ranker(matrix, all_rows, seed)
    return final_model, output


def oof_gated_predictions(matrix: FixedCandidateMatrix, thresholds,
                          folds=5, seed=42):
    """Cross-fitted predictions for uncertainty thresholds.

    A higher threshold means fewer queries are changed, making threshold
    selection an explicit intervention-budget control rather than a test-time
    heuristic.
    """
    outputs = {float(threshold): {} for threshold in thresholds}
    all_rows = np.arange(len(matrix.uids))
    for fold in range(folds):
        valid = np.asarray([i for i, uid in enumerate(matrix.uids)
                            if _fold(uid, folds) == fold])
        mask = np.ones(len(all_rows), dtype=bool)
        mask[valid] = False
        model = fit_ranker(matrix, all_rows[mask], seed + fold)
        for threshold in outputs:
            outputs[threshold].update(predict(
                model, matrix, valid, 20, entropy_threshold=threshold))
        print(f"[MGCOT-RERANK] gated-fold={fold + 1}/{folds} "
              f"valid={len(valid)}", flush=True)
    return fit_ranker(matrix, all_rows, seed), outputs
