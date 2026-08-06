#!/usr/bin/env python3
"""Evaluate exported MGCOT predictions on masked Rental LOO."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def metrics(predictions, queries, k=6):
    hits, reciprocal, ndcg = 0, 0.0, 0.0
    for uid, query in queries.items():
        target = int(query["targets"][0])
        rank = next((rank for rank, item in enumerate(
            predictions[str(uid)][:k], 1) if item == target), None)
        if rank is not None:
            hits += 1
            reciprocal += 1.0 / rank
            ndcg += 1.0 / math.log2(rank + 1)
    n = max(len(queries), 1)
    return {f"HR@{k}": hits / n, f"Recall@{k}": hits / n,
            f"MRR@{k}": reciprocal / n, f"NDCG@{k}": ndcg / n,
            "n": len(queries)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--queries", type=Path,
                        default=Path("rental_intent_bench/split_loo_masked/queries.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("sparse_bench/mgcot_rental_results.json"))
    args = parser.parse_args()
    queries = json.loads(args.queries.read_text())
    bundle = np.load(args.npz)
    expected = np.asarray([q["targets"][0] for q in queries.values()])
    if not np.array_equal(expected, bundle["targets"]):
        raise ValueError("prediction rows do not align with Rental queries")
    predictions, candidate_hits = {}, 0
    for row, (uid, query) in enumerate(queries.items()):
        seen = set(map(int, query["context"]))
        ranking = [int(item) for item in bundle["items"][row]
                   if int(item) not in seen]
        predictions[str(uid)] = ranking
        candidate_hits += int(int(query["targets"][0]) in ranking)
    result = {
        "protocol": "Rental split_loo_masked; train-only adjacency; nested-selected 13 epochs",
        "method": "MGCOT-MPS Rental",
        "metrics": metrics(predictions, queries, 6),
        "candidate_recall@120_after_mask": candidate_hits / len(queries),
        "local_comparators": {
            "DT-RRF": {"Recall@6": 0.268752},
            "SKNN": {"Recall@6": 0.269456},
            "ItemKNN": {"Recall@6": 0.230739},
            "MostPop": {"Recall@6": 0.229175},
        },
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
