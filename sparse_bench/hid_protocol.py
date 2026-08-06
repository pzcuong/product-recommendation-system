"""Adapter and metrics for the official AAAI-2026 HID Diginetica split."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import pickle

import numpy as np


REPO = Path(__file__).resolve().parent.parent
HID_DATA = REPO / "reference_repos" / "Code4HID" / "datasets" / "diginetica-2"


def _reconstruct_sessions(contexts, targets):
    """Invert the official prefix expansion into complete train sessions."""
    sessions = {}
    previous = None
    for row, (context0, target) in enumerate(zip(contexts, targets)):
        context = [int(x) for x in context0]
        same_block = (previous is not None and len(context) < len(previous)
                      and context == previous[:len(context)])
        if not same_block:
            sessions[f"hid_train_{row}"] = context + [int(target)]
        previous = context
    return sessions


def load_hid_diginetica() -> dict:
    train_x, train_y = pickle.load(open(HID_DATA / "train.txt", "rb"))
    test_x, test_y = pickle.load(open(HID_DATA / "test.txt", "rb"))
    tail, head = pickle.load(open(HID_DATA / "TailHead.pkl", "rb"))
    item_att = pickle.load(open(HID_DATA / "item_att.txt", "rb"))
    sessions = _reconstruct_sessions(train_x, train_y)
    queries = {
        f"hid_test_{row}": {"context": [int(x) for x in context],
                            "targets": [int(target)]}
        for row, (context, target) in enumerate(zip(test_x, test_y))
    }
    return {
        "domain": "HID_Diginetica",
        "n_items": 43098,
        "train_sessions": sessions,
        "test_queries": queries,
        "tail_score_indices": set(int(x) for x in tail),
        "head_score_indices": set(int(x) for x in head),
        "item_attributes": {int(k): int(v) for k, v in item_att.items()},
        "official_examples": len(train_x),
    }


def attribute_semantic_matrix(data: dict, dim: int = 64, seed: int = 42) -> np.ndarray:
    """Deterministic attribute teacher without test interactions."""
    rng = np.random.default_rng(seed)
    attributes = data["item_attributes"]
    unique = sorted(set(attributes.values()))
    vectors = rng.standard_normal((len(unique), dim)).astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    lookup = {attribute: vectors[row] for row, attribute in enumerate(unique)}
    matrix = np.zeros((data["n_items"], dim), dtype=np.float32)
    for item, attribute in attributes.items():
        if 0 < item < data["n_items"]:
            matrix[item] = lookup[attribute]
    return matrix


def official_metrics(predictions: dict[str, list[int]], data: dict, k: int = 20) -> dict:
    """Match the official HID code's zero-based score-index convention."""
    tail = data["tail_score_indices"]
    queries = data["test_queries"]
    hit = 0
    reciprocal = 0.0
    ndcg = 0.0
    tail_hit = 0
    tail_reciprocal = 0.0
    tail_ndcg = 0.0
    tail_targets = 0
    recommended_tail = set()
    tail_slots = 0
    for uid, query in queries.items():
        target_index = int(query["targets"][0]) - 1
        ranking = [int(item) - 1 for item in predictions.get(uid, ())[:k]]
        if target_index in tail:
            tail_targets += 1
        rank = next((r for r, item in enumerate(ranking, 1)
                     if item == target_index), None)
        if rank is not None:
            hit += 1
            reciprocal += 1.0 / rank
            ndcg += 1.0 / math.log2(rank + 1)
            if target_index in tail:
                tail_hit += 1
                tail_reciprocal += 1.0 / rank
                tail_ndcg += 1.0 / math.log2(rank + 1)
        for item in ranking:
            if item in tail:
                recommended_tail.add(item)
                tail_slots += 1
    n = max(len(queries), 1)
    nt = max(tail_targets, 1)
    return {
        "HR@20": hit / n,
        "MRR@20": reciprocal / n,
        "NDCG@20": ndcg / n,
        "tHR@20": tail_hit / nt,
        "tMRR@20": tail_reciprocal / nt,
        "tNDCG@20": tail_ndcg / nt,
        "tCov@20": len(recommended_tail) / max(len(tail), 1),
        "Tail@20": tail_slots / (n * k),
        "n": len(queries),
        "n_tail_targets": tail_targets,
    }

