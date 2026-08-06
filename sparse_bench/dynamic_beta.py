"""Training-only query-conditioned beta for CEARF-N rank fusion.

The gate is deliberately separated from validation. It learns a continuous
``beta_q`` directly from out-of-fold training prefixes with a differentiable
pairwise rank loss. It does not enumerate beta candidates, and validation
labels are never passed to :meth:`TrainOnlyDynamicBeta.fit`.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from run_cearfn import fuse


FEATURE_NAMES = (
    "log_context_length",
    "log_last_item_frequency",
    "last_item_is_tail",
    "memory_component_agreement_top5",
    "memory_neural_top1_agreement",
    "memory_neural_jaccard_top5",
    "memory_neural_jaccard_top20",
    "memory_neural_rr_overlap_top20",
    "memory_top1_in_neural_top5",
    "neural_top1_in_memory_top5",
    "memory_top1_component_support",
    "neural_top1_component_support",
    "memory_component_pairwise_jaccard_top10",
    "neural_novelty_outside_memory_top120",
    "log_effective_transition_branches",
    "top1_transition_share",
)


FEATURE_GROUPS = {
    "context": (0, 1, 2),
    "cross_expert": (4, 5, 6, 7, 8, 9, 13),
    "memory_certainty": (3, 10, 11, 12, 14, 15),
}


def _positive(row: Sequence[int], width: int) -> list[int]:
    return [int(item) for item in row[:width] if int(item) > 0]


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def _rr_overlap(left: Sequence[int], right: Sequence[int]) -> float:
    """Symmetric reciprocal-rank overlap, normalized to [0, 1]."""
    left_rank = {int(item): rank for rank, item in enumerate(left, 1)}
    right_rank = {int(item): rank for rank, item in enumerate(right, 1)}
    shared = left_rank.keys() & right_rank.keys()
    raw = sum(
        1.0 / math.sqrt(left_rank[item] * right_rank[item])
        for item in shared
    )
    normalizer = sum(1.0 / rank for rank in range(1, 21))
    return float(raw / normalizer) if normalizer else 0.0


def _component_agreement_top5(components: Sequence[Sequence[int]]) -> int:
    votes: Counter[int] = Counter()
    for ranking in components:
        votes.update(_positive(ranking, 5))
    return max(votes.values(), default=0)


def _component_support(item: int, components: Sequence[Sequence[int]]) -> float:
    if item <= 0:
        return 0.0
    return float(
        sum(item in set(_positive(row, 20)) for row in components)
        / max(len(components), 1)
    )


def _mean_pairwise_jaccard(
        components: Sequence[Sequence[int]], width: int = 10) -> float:
    values = []
    for left in range(len(components)):
        for right in range(left + 1, len(components)):
            values.append(_jaccard(
                _positive(components[left], width),
                _positive(components[right], width),
            ))
    return float(np.mean(values)) if values else 0.0


def _transition_diagnostics(index, context: Sequence[int]) -> tuple[float, float]:
    """Effective branching and dominant-transition share for the last item."""
    if not context:
        return 0.0, 0.0
    outgoing = index.transition.get(int(context[-1]), {})
    if not outgoing:
        return 0.0, 0.0
    weights = np.asarray(list(outgoing.values()), dtype=np.float64)
    probabilities = weights / weights.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    effective_branches = math.exp(entropy)
    return math.log1p(effective_branches), float(probabilities.max())


def feature_matrix(
        queries: Mapping[str, Mapping[str, Sequence[int]]],
        keys: Sequence[str],
        memory_arrays: Mapping[str, np.ndarray],
        neural_rankings: np.ndarray,
        item_freq: Mapping[int, int],
        head_items: set[int],
        index,
) -> np.ndarray:
    """Return deterministic target-free gate features in ``keys`` order."""
    output = np.zeros((len(keys), len(FEATURE_NAMES)), dtype=np.float32)
    component_names = ("transition", "session", "popularity")
    for row, uid0 in enumerate(keys):
        uid = str(uid0)
        context = [
            int(item) for item in queries[uid].get("context", ())
            if int(item) > 0
        ]
        last = context[-1] if context else 0
        components = [
            _positive(memory_arrays[name][row], len(memory_arrays[name][row]))
            for name in component_names
        ]
        memory = _positive(memory_arrays["selected"][row], 120)
        neural = _positive(neural_rankings[row], 120)
        memory_set = set(memory)
        memory5, neural5 = memory[:5], neural[:5]
        memory20, neural20 = memory[:20], neural[:20]
        memory_top1 = memory[0] if memory else 0
        neural_top1 = neural[0] if neural else 0
        log_branches, transition_top1 = _transition_diagnostics(
            index, context)
        output[row] = np.asarray([
            math.log1p(len(context)),
            math.log1p(item_freq.get(last, 0)),
            float(bool(last) and last not in head_items),
            _component_agreement_top5(components) / 3.0,
            float(bool(memory_top1) and memory_top1 == neural_top1),
            _jaccard(memory5, neural5),
            _jaccard(memory20, neural20),
            _rr_overlap(memory20, neural20),
            float(memory_top1 in set(neural5)) if memory_top1 else 0.0,
            float(neural_top1 in set(memory5)) if neural_top1 else 0.0,
            _component_support(memory_top1, components),
            _component_support(neural_top1, components),
            _mean_pairwise_jaccard(components),
            float(sum(item not in memory_set for item in neural20)
                  / max(len(neural20), 1)),
            log_branches,
            transition_top1,
        ], dtype=np.float32)
    return output


def rank_evidence_training_arrays(
        memory: np.ndarray,
        neural: np.ndarray,
        targets: np.ndarray,
        hard_negatives: int = 32,
        constant: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build target/negative reciprocal-rank evidence for direct beta learning.

    Returns target evidence ``[n, 2]``, negative evidence ``[n, h, 2]``,
    a negative mask and an actionable-query mask. Evidence columns are memory
    and neural. Queries whose target is absent from both candidate lists are
    excluded because no beta value can rescue them.
    """
    if not (len(memory) == len(neural) == len(targets)):
        raise ValueError("training arrays must share row count")
    target_evidence = np.zeros((len(targets), 2), dtype=np.float32)
    negative_evidence = np.zeros(
        (len(targets), hard_negatives, 2), dtype=np.float32)
    negative_mask = np.zeros(
        (len(targets), hard_negatives), dtype=np.float32)
    actionable = np.zeros(len(targets), dtype=bool)

    for row, target0 in enumerate(targets):
        target = int(target0)
        memory_rank: dict[int, int] = {}
        neural_rank: dict[int, int] = {}
        for rank, item0 in enumerate(memory[row], 1):
            item = int(item0)
            if item > 0 and item not in memory_rank:
                memory_rank[item] = rank
        for rank, item0 in enumerate(neural[row], 1):
            item = int(item0)
            if item > 0 and item not in neural_rank:
                neural_rank[item] = rank

        if target in memory_rank:
            target_evidence[row, 0] = (constant + 1.0) / (
                constant + memory_rank[target])
        if target in neural_rank:
            target_evidence[row, 1] = (constant + 1.0) / (
                constant + neural_rank[target])
        actionable[row] = bool(target_evidence[row].max() > 0)

        candidates = (memory_rank.keys() | neural_rank.keys()) - {target}
        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                -max(
                    (constant + 1.0) / (constant + memory_rank[item])
                    if item in memory_rank else 0.0,
                    (constant + 1.0) / (constant + neural_rank[item])
                    if item in neural_rank else 0.0,
                ),
                item,
            ),
        )[:hard_negatives]
        for column, item in enumerate(ranked_candidates):
            if item in memory_rank:
                negative_evidence[row, column, 0] = (constant + 1.0) / (
                    constant + memory_rank[item])
            if item in neural_rank:
                negative_evidence[row, column, 1] = (constant + 1.0) / (
                    constant + neural_rank[item])
            negative_mask[row, column] = 1.0
    return (
        target_evidence,
        negative_evidence,
        negative_mask,
        actionable,
    )


def _pairwise_loss(
        beta: torch.Tensor,
        target_evidence: torch.Tensor,
        negative_evidence: torch.Tensor,
        negative_mask: torch.Tensor,
        temperature: float,
) -> torch.Tensor:
    target_score = (
        (1.0 - beta) * target_evidence[:, 0]
        + beta * target_evidence[:, 1]
    )
    negative_score = (
        (1.0 - beta[:, None]) * negative_evidence[:, :, 0]
        + beta[:, None] * negative_evidence[:, :, 1]
    )
    pairwise = F.softplus(
        (negative_score - target_score[:, None]) / temperature)
    per_query = (
        (pairwise * negative_mask).sum(dim=1)
        / negative_mask.sum(dim=1).clamp_min(1.0)
    )
    return per_query.mean()


class _BetaNetwork(nn.Module):
    def __init__(
            self,
            n_features: int,
            hidden: int,
            initial_beta: float,
            max_residual: float,
    ):
        super().__init__()
        if hidden > 0:
            self.network = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
        else:
            self.network = nn.Linear(n_features, 1)
        final = (
            self.network[-1]
            if isinstance(self.network, nn.Sequential)
            else self.network
        )
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.initial_beta = float(
            np.clip(initial_beta, 1e-4, 1.0 - 1e-4))
        self.max_residual = float(min(
            max_residual,
            self.initial_beta,
            1.0 - self.initial_beta,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.network(x).squeeze(-1))
        return self.initial_beta + self.max_residual * residual


@dataclass
class TrainOnlyGlobalBeta:
    """One continuous beta learned on OOF training ranks, never validation."""

    temperature: float = 0.10
    epochs: int = 80
    learning_rate: float = 0.02
    admission_cost: float = 1e-3
    initial_beta: float = 0.35
    hard_negatives: int = 32
    rrf_constant: float = 20.0
    seed: int = 42

    def __post_init__(self) -> None:
        self.beta_: float | None = None

    def fit(
            self,
            memory: np.ndarray,
            neural: np.ndarray,
            targets: np.ndarray,
    ) -> dict:
        target, negative, negative_mask, actionable = (
            rank_evidence_training_arrays(
                memory, neural, targets, self.hard_negatives,
                self.rrf_constant))
        if not actionable.any():
            raise ValueError("no calibration target appears in either expert")
        torch.manual_seed(self.seed)
        initial = float(np.clip(self.initial_beta, 1e-4, 1.0 - 1e-4))
        logit = nn.Parameter(torch.tensor(
            math.log(initial / (1.0 - initial)), dtype=torch.float32))
        optimizer = torch.optim.AdamW(
            [logit], lr=self.learning_rate, weight_decay=0.0)
        target_tensor = torch.from_numpy(target[actionable])
        negative_tensor = torch.from_numpy(negative[actionable])
        mask_tensor = torch.from_numpy(negative_mask[actionable])
        for _ in range(self.epochs):
            beta = torch.sigmoid(logit).expand(len(target_tensor))
            loss = _pairwise_loss(
                beta, target_tensor, negative_tensor, mask_tensor,
                self.temperature)
            loss = loss + self.admission_cost * beta.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.beta_ = float(torch.sigmoid(logit).detach())
        return {
            "beta": self.beta_,
            "n_calibration_queries": int(len(targets)),
            "n_actionable_queries": int(actionable.sum()),
            "actionable_share": float(actionable.mean()),
            "training_uses_validation_labels": False,
            "beta_search": False,
            "objective": "pairwise reciprocal-rank loss",
        }

    def predict(self, n_queries: int) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError("fit must be called before predict")
        return np.full(n_queries, self.beta_, dtype=np.float32)


@dataclass
class TrainOnlyDynamicBeta:
    """Small continuous gate trained on OOF training prefixes only."""

    temperature: float = 0.10
    hidden: int = 16
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    admission_cost: float = 1e-3
    prior_penalty: float = 1e-3
    initial_beta: float = 0.35
    max_residual: float = 0.10
    hard_negatives: int = 32
    rrf_constant: float = 20.0
    seed: int = 42

    def __post_init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.model_: _BetaNetwork | None = None

    def fit(
            self,
            features: np.ndarray,
            memory: np.ndarray,
            neural: np.ndarray,
            targets: np.ndarray,
    ) -> dict:
        """Fit the gate without a beta grid or validation labels."""
        if not (len(features) == len(memory) == len(neural) == len(targets)):
            raise ValueError("calibration arrays must share row count")
        target, negative, negative_mask, actionable = (
            rank_evidence_training_arrays(
                memory, neural, targets, self.hard_negatives,
                self.rrf_constant))
        if not actionable.any():
            raise ValueError("no calibration target appears in either expert")

        x = np.asarray(features[actionable], dtype=np.float32)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        x = (x - self.mean_) / self.scale_

        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        self.model_ = _BetaNetwork(
            x.shape[1], self.hidden, self.initial_beta,
            self.max_residual)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        x_tensor = torch.from_numpy(x)
        target_tensor = torch.from_numpy(target[actionable])
        negative_tensor = torch.from_numpy(negative[actionable])
        mask_tensor = torch.from_numpy(negative_mask[actionable])
        batch_size = min(512, len(x_tensor))
        for _ in range(self.epochs):
            order = torch.randperm(
                len(x_tensor), generator=generator)
            for start in range(0, len(order), batch_size):
                rows = order[start:start + batch_size]
                beta = self.model_(x_tensor[rows])
                loss = _pairwise_loss(
                    beta,
                    target_tensor[rows],
                    negative_tensor[rows],
                    mask_tensor[rows],
                    self.temperature,
                )
                loss = (
                    loss
                    + self.admission_cost * beta.mean()
                    + self.prior_penalty
                    * torch.square(beta - self.initial_beta).mean()
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        calibration_beta = self.predict(features)
        return {
            "n_calibration_queries": int(len(features)),
            "n_actionable_queries": int(actionable.sum()),
            "actionable_share": float(actionable.mean()),
            "predicted_beta_mean": float(calibration_beta.mean()),
            "predicted_beta_std": float(calibration_beta.std()),
            "max_residual": float(self.model_.max_residual),
            "architecture": (
                "linear" if self.hidden <= 0 else f"mlp-{self.hidden}"),
            "training_uses_validation_labels": False,
            "beta_search": False,
            "objective": "pairwise reciprocal-rank loss",
            "feature_names": list(FEATURE_NAMES),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before predict")
        x = (np.asarray(features, dtype=np.float32) - self.mean_) / self.scale_
        with torch.no_grad():
            beta = self.model_(torch.from_numpy(x))
        return beta.numpy().astype(np.float32)

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        if self.model_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before serialization")
        payload = {
            "feature_mean": self.mean_.astype(np.float32),
            "feature_scale": self.scale_.astype(np.float32),
        }
        for name, value in self.model_.state_dict().items():
            payload[f"model::{name}"] = value.detach().cpu().numpy()
        return payload


def fuse_with_dynamic_beta(
        memory: np.ndarray,
        neural: np.ndarray,
        betas: np.ndarray,
        topk: int = 20,
        constant: float = 20.0,
) -> np.ndarray:
    if not (len(memory) == len(neural) == len(betas)):
        raise ValueError("fusion arrays must share row count")
    output = np.zeros((len(betas), topk), dtype=np.int32)
    for row, beta in enumerate(betas):
        ranking = fuse(
            memory[row], neural[row], float(beta),
            topk=topk, constant=constant)
        output[row, :len(ranking)] = ranking
    return output
