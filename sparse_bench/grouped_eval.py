"""
Grouped evaluation generalized for every domain.

Same 12 groups as the existing Rental grouped evaluator
(compute_grouped_metrics.py), but built from the unified loader output:

    len_1_2, len_3, len_4_7, len_8_plus   (by context length)
    single_visit, multi_visit              (by user visit count)
    tgt_head, tgt_mid, tgt_tail            (by target-item popularity 20/60/20)
    same_category, cross_category          (target vs last context item category)

Metrics: Recall@{5,10,20}, NDCG@{5,10,20}, HR@6  (single-target -> HR@k == Recall@k).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

K_EVAL = [5, 6, 10, 20]


# -----------------------------------------------------------------------------
# Group construction
# -----------------------------------------------------------------------------
def build_groups(data: dict) -> Tuple[dict, dict, list]:
    """Build the 12 groups from a loader's output dict.

    Returns (groups, test_items, grp_order) where
      groups[name] = list[ (uid, tgt, ctx_len) ]
      test_items[uid] = tgt (int)

    If data["reference_groups"] is provided (e.g. the visit-level Rental loader
    supplies the exact groups from multi_domain_eval), those groups are used
    verbatim so we compare apple-to-apple with the published reference numbers.
    """
    train_sessions = data["train_sessions"]
    test_queries = data["test_queries"]
    item_freq = data["item_freq"]
    item_categories = data["item_categories"]
    visit_counts = data.get("visit_counts", {u: 1 for u in train_sessions})

    grp_order = [
        "overall",
        "len_1_2", "len_3", "len_4_7", "len_8_plus",
        "single_visit", "multi_visit",
        "tgt_head", "tgt_mid", "tgt_tail",
        "tgt_tail_learnable", "tgt_tail_coldstart",
        "same_category", "cross_category",
    ]

    # ---- Optional: use loader-provided reference groups verbatim. ----
    ref = data.get("reference_groups")
    if ref:
        groups: Dict[str, List[Tuple[str, int, int]]] = {g: [] for g in grp_order}
        test_items: Dict[str, int] = {}
        for g in grp_order:
            for e in ref.get(g, []):
                uid = str(e["vid"])
                tgt = int(e["target"])
                ctx_len = int(e.get("len", len(test_queries.get(uid, {}).get("context", []))))
                test_items[uid] = tgt
                groups[g].append((uid, tgt, ctx_len))
        # Add the cold/learnable tail split from training frequency.
        fr = data["item_freq"]
        for e in ref.get("tgt_tail", []):
            uid = str(e["vid"]); tgt = int(e["target"])
            entry = (uid, tgt, int(e.get("len", 0)))
            if fr.get(tgt, 0) >= 1:
                groups["tgt_tail_learnable"].append(entry)
            else:
                groups["tgt_tail_coldstart"].append(entry)
        return groups, test_items, grp_order

    # Popularity tiers (20/60/20 over TRAIN item frequency, over the whole vocab).
    n = len(item_freq)
    if n:
        sorted_items = sorted(item_freq.items(), key=lambda x: x[1], reverse=True)
        head = set(i for i, _ in sorted_items[: max(1, int(n * 0.2))])
        tail_start = max(1, int(n * 0.8))
        tail = set(i for i, _ in sorted_items[tail_start:])
    else:
        head, tail = set(), set()

    groups = {g: [] for g in grp_order}
    test_items = {}

    for uid, q in test_queries.items():
        ctx = q["context"]
        tgt = q["targets"][0]
        ctx_len = len(ctx)
        test_items[uid] = tgt
        entry = (uid, tgt, ctx_len)

        groups["overall"].append(entry)
        if ctx_len <= 2:
            groups["len_1_2"].append(entry)
        elif ctx_len == 3:
            groups["len_3"].append(entry)
        elif ctx_len <= 7:
            groups["len_4_7"].append(entry)
        else:
            groups["len_8_plus"].append(entry)

        vc = visit_counts.get(uid, 1)
        (groups["single_visit"] if vc <= 1 else groups["multi_visit"]).append(entry)

        if tgt in head:
            groups["tgt_head"].append(entry)
        elif tgt in tail:
            groups["tgt_tail"].append(entry)
            # cold-start vs learnable tail split
            if item_freq.get(tgt, 0) >= 1:
                groups["tgt_tail_learnable"].append(entry)
            else:
                groups["tgt_tail_coldstart"].append(entry)
        else:
            groups["tgt_mid"].append(entry)

        # Category transition (last context item vs target)
        if ctx and item_categories:
            last_ctx = ctx[-1]
            tcat = item_categories.get(tgt)
            ccat = item_categories.get(last_ctx)
            if tcat is not None and ccat is not None:
                if tcat == ccat:
                    groups["same_category"].append(entry)
                else:
                    groups["cross_category"].append(entry)

    return groups, test_items, grp_order


# -----------------------------------------------------------------------------
# Metric computation (domain-agnostic, model-agnostic)
# -----------------------------------------------------------------------------
def compute_metrics_for_group(predictions: Dict[str, List[int]],
                              group_entries: List[Tuple[str, int, int]],
                              k_values: List[int] = None) -> Dict[str, float]:
    """Mean Recall@k, NDCG@k, HR@k, Precision@k, MRR@k over a group.

    Single-target setting: HR@k == Recall@k, Precision@k = hit/20 (the SR-GNN
    paper's "P@20"), MRR@k = 1/(rank+1) for the hit (the SR-GNN "MRR@20").
    These let us compare directly against published SR-GNN/GCE-GNN numbers.
    """
    if k_values is None:
        k_values = K_EVAL
    results = {f"recall@{k}": [] for k in k_values}
    results.update({f"ndcg@{k}": [] for k in k_values})

    for uid, tgt, _ in group_entries:
        pred = predictions.get(uid, [])
        for k in k_values:
            topk = pred[:k]
            hit = 1 if tgt in topk else 0
            results[f"recall@{k}"].append(hit)
            if hit:
                rank = topk.index(tgt)
                results[f"ndcg@{k}"].append(1.0 / math.log2(rank + 2))
            else:
                results[f"ndcg@{k}"].append(0.0)

    if not results[f"recall@{k_values[0]}"]:
        return {k: 0.0 for k in results}
    out = {k: float(np.mean(v)) for k, v in results.items()}
    # Add SR-GNN-standard metrics: P@20 (Precision@20) and MRR@20.
    # For single-target: Precision@k = hit/k (SR-GNN reports P@20 = hit/20),
    # MRR@k = 1/(rank+1) for the hit else 0.
    for k in k_values:
        p_at_k = []
        mrr_at_k = []
        for uid, tgt, _ in group_entries:
            pred = predictions.get(uid, [])
            topk = pred[:k]
            if tgt in topk:
                rank = topk.index(tgt)
                p_at_k.append(1.0 / k)
                mrr_at_k.append(1.0 / (rank + 1))
            else:
                p_at_k.append(0.0)
                mrr_at_k.append(0.0)
        out[f"precision@{k}"] = float(np.mean(p_at_k)) if p_at_k else 0.0
        out[f"mrr@{k}"] = float(np.mean(mrr_at_k)) if mrr_at_k else 0.0
    return out


def evaluate_all_groups(predictions: Dict[str, List[int]], data: dict,
                        k_values: List[int] = None) -> Dict[str, dict]:
    """Compute per-group metrics. Returns {group_name: {metric: value, 'n': count}}."""
    groups, _, grp_order = build_groups(data)
    out = {}
    for g in grp_order:
        entries = groups[g]
        m = compute_metrics_for_group(predictions, entries, k_values)
        m["n"] = len(entries)
        out[g] = m
    return out


def format_group_table(group_metrics: dict, k_show: int = 6) -> str:
    grp_order = [
        "overall",
        "len_1_2", "len_3", "len_4_7", "len_8_plus",
        "single_visit", "multi_visit",
        "tgt_head", "tgt_mid", "tgt_tail",
        "tgt_tail_learnable", "tgt_tail_coldstart",
        "same_category", "cross_category",
    ]
    lines = [f"  {'group':15} {'N':>5} {'R@6':>8} {'R@20':>8} {'NDCG@6':>8} {'NDCG@20':>8}"]
    lines.append("  " + "-" * 60)
    for g in grp_order:
        m = group_metrics.get(g, {})
        if m.get("n", 0) == 0:
            lines.append(f"  {g:15} {0:>5} {'-':>8}")
            continue
        lines.append(f"  {g:15} {m['n']:>5} {m.get(f'recall@{k_show}',0):>8.4f} "
                     f"{m.get('recall@20',0):>8.4f} {m.get(f'ndcg@{k_show}',0):>8.4f} "
                     f"{m.get('ndcg@20',0):>8.4f}")
    return "\n".join(lines)
