"""
Neural + ItemKNN ensemble via Reciprocal Rank Fusion (RRF).

Hypothesis: neural models (CoDT/HACL/SR-GNN) rank head items well but collapse
on the tail; ItemKNN ranks tail items well (no training needed) but is weaker
on head. RRF fuses their rankings rank-by-rank and captures both strengths.

RRF score for item i = sum over rankers of 1/(k + rank_i). Standard, parameter-
free (k=60 default). This is the same fusion used in CoDT's adaptive_rrf.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np


def rrf_fuse(rank_lists: List[List[int]], k: int = 60, top_n: int = 50) -> List[int]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    rank_lists: list of ranked item-id lists (each from a different ranker).
    Returns a single fused ranked list of length top_n.
    """
    scores: Dict[int, float] = {}
    for ranked in rank_lists:
        for rank, item in enumerate(ranked[:top_n * 2]):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return [it for it, _ in sorted(scores.items(), key=lambda x: -x[1])][:top_n]


def weighted_rrf_fuse(rank_lists: List[List[int]], weights: List[float],
                      k: int = 60, top_n: int = 50) -> List[int]:
    """Weighted RRF: each ranker contributes with its own weight."""
    scores: Dict[int, float] = {}
    for ranked, w in zip(rank_lists, weights):
        for rank, item in enumerate(ranked[:top_n * 2]):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return [it for it, _ in sorted(scores.items(), key=lambda x: -x[1])][:top_n]


def ensemble_predict(neural_preds: Dict[str, List[int]],
                     itemknn_preds: Dict[str, List[int]],
                     method: str = "rrf",
                     neural_weight: float = 1.0,
                     itemknn_weight: float = 1.0,
                     k: int = 60,
                     top_n: int = 50) -> Dict[str, List[int]]:
    """Fuse per-user neural predictions with ItemKNN predictions.

    method:
      'rrf'            : equal-weight RRF (default, parameter-free)
      'weighted_rrf'   : neural_weight / itemknn_weight control the blend
      'neural_first'   : neural top-N, then ItemKNN fills the rest (greedy)
    """
    out = {}
    uids = set(neural_preds) & set(itemknn_preds)
    for uid in uids:
        n_list = neural_preds[uid]
        k_list = itemknn_preds[uid]
        if method == "rrf":
            out[uid] = rrf_fuse([n_list, k_list], k=k, top_n=top_n)
        elif method == "weighted_rrf":
            out[uid] = weighted_rrf_fuse([n_list, k_list],
                                         [neural_weight, itemknn_weight],
                                         k=k, top_n=top_n)
        elif method == "neural_first":
            seen = set()
            merged = []
            for it in n_list:
                if it not in seen:
                    merged.append(it); seen.add(it)
            for it in k_list:
                if len(merged) >= top_n:
                    break
                if it not in seen:
                    merged.append(it); seen.add(it)
            out[uid] = merged[:top_n]
        else:
            raise ValueError(method)
    return out


# =============================================================================
# Confidence-routed fusion (the novelty).
# =============================================================================
def _softmax_entropy(scores: List[float]) -> float:
    """Entropy of the softmax-normalized top-K scores. High entropy = the
    neural head is uncertain (typically when the target is a tail item it
    can't discriminate); low entropy = confident (head items)."""
    if not scores:
        return 0.0
    s = np.asarray(scores, dtype=np.float64)
    s = s - s.max()
    e = np.exp(s)
    p = e / e.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def confidence_routed_fuse(neural_scores: Dict[str, List[tuple]],
                           itemknn_preds: Dict[str, List[int]],
                           top_n: int = 50,
                           entropy_low: float = 1.5,
                           entropy_high: float = 3.0,
                           k_rrf: int = 60) -> Dict[str, List[int]]:
    """Confidence-routed neural + ItemKNN fusion (HACL's novel contribution).

    neural_scores : {uid: [(item, score), ...]} the neural model's ranked
                    candidates with raw scores (used to compute confidence).
    itemknn_preds : {uid: [item, ...]} ItemKNN's ranked candidates.
    entropy_low   : below this entropy the neural head is trusted fully
                    (neural_weight = 1).
    entropy_high  : above this entropy the head is treated as unreliable
                    (neural_weight = 0, ItemKNN-only).
    Between the two, the neural weight interpolates linearly.

    Rationale: high entropy of the neural softmax over its top-K = the model
    cannot discriminate the target → typically a tail item it has underfit.
    Routing to ItemKNN there recovers tail coverage without sacrificing the
    confident-head rankings.
    """
    out = {}
    for uid in set(neural_scores) & set(itemknn_preds):
        n_pairs = neural_scores[uid]
        n_list = [it for it, _ in n_pairs]
        k_list = itemknn_preds[uid]
        ent = _softmax_entropy([s for _, s in n_pairs[:top_n]])
        if ent <= entropy_low:
            w_n = 1.0
        elif ent >= entropy_high:
            w_n = 0.0
        else:
            w_n = 1.0 - (ent - entropy_low) / (entropy_high - entropy_low)
        out[uid] = weighted_rrf_fuse([n_list, k_list], [w_n, 1.0],
                                     k=k_rrf, top_n=top_n)
    return out
