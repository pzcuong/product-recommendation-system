from __future__ import annotations

import math
import numpy as np


def per_query(rankings: dict[str, list[int]], queries: dict[str, dict], train_items=None):
    rows = []
    for uid, q in queries.items():
        target = q["targets"][0]
        pred = rankings.get(uid, [])
        rank = pred.index(target) + 1 if target in pred else None
        row = {"query_id": uid, "target": target, "rank": rank,
               "target_seen": target in train_items if train_items is not None else None}
        for k in (5, 10, 20):
            row[f"recall@{k}"] = int(rank is not None and rank <= k)
            row[f"ndcg@{k}"] = 1 / math.log2(rank + 1) if rank is not None and rank <= k else 0
        rows.append(row)
    return rows


def aggregate(rows):
    metrics = {}
    for key in ("recall@5", "recall@10", "recall@20", "ndcg@20"):
        metrics[key] = float(np.mean([r[key] for r in rows])) if rows else 0
    seen = [r for r in rows if r["target_seen"]]
    metrics["seen_target_recall@20"] = float(np.mean([r["recall@20"] for r in seen])) if seen else 0
    metrics["n_queries"] = len(rows)
    metrics["n_seen_targets"] = len(seen)
    return metrics


def paired_bootstrap(rows_a, rows_b, metric="recall@20", samples=2000, seed=2026):
    a = {r["query_id"]: r[metric] for r in rows_a}
    b = {r["query_id"]: r[metric] for r in rows_b}
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("no paired queries")
    diffs = np.asarray([a[i] - b[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(samples)])
    return {"difference": float(diffs.mean()), "ci_low": float(np.quantile(boot, .025)),
            "ci_high": float(np.quantile(boot, .975)), "p_two_sided": float(2 * min(
                np.mean(boot <= 0), np.mean(boot >= 0))), "n": len(ids), "samples": samples}
