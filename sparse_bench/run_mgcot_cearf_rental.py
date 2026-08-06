#!/usr/bin/env python3
"""Validation-selected MGCOT + CEARF fusion on masked Rental LOO."""
from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path

import numpy as np

import cearf
from evaluate_mgcot_rental import metrics


def masked_npz(bundle, queries):
    expected = np.asarray([q["targets"][0] for q in queries.values()])
    if not np.array_equal(expected, bundle["targets"]):
        raise ValueError("MGCOT rows do not align with queries")
    output = {}
    for row, (uid, query) in enumerate(queries.items()):
        seen = set(map(int, query["context"]))
        output[str(uid)] = [int(x) for x in bundle["items"][row]
                            if int(x) not in seen]
    return output


def fuse(left, right, left_weight, constant, topk=120):
    score = defaultdict(float)
    for weight, ranking in ((left_weight, left),
                            (1.0 - left_weight, right)):
        for rank, item in enumerate(ranking[:120], 1):
            score[int(item)] += weight / (constant + rank)
    return [item for item, _ in sorted(
        score.items(), key=lambda pair: (-pair[1], pair[0]))[:topk]]


def candidate_recall(predictions, queries, k=120):
    return sum(int(q["targets"][0]) in predictions[str(uid)][:k]
               for uid, q in queries.items()) / len(queries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested-npz", type=Path, required=True)
    parser.add_argument("--full-npz", type=Path, required=True)
    parser.add_argument("--split", type=Path,
                        default=Path("rental_intent_bench/split_loo_masked"))
    parser.add_argument("--output", type=Path,
                        default=Path("sparse_bench/mgcot_cearf_rental_results.json"))
    args = parser.parse_args()
    vocab = json.loads((args.split / "vocab.json").read_text())
    sequence_map = json.loads((args.split / "train_seqs.json").read_text())
    full_queries = json.loads((args.split / "queries.json").read_text())
    full_sessions = {str(uid): list(map(int, sequence))
                     for uid, sequence in sequence_map.items()}
    nested_sessions, nested_queries = {}, {}
    for uid, sequence in full_sessions.items():
        if len(sequence) >= 2:
            nested_sessions[uid] = sequence[:-1]
            nested_queries[uid] = {
                "context": sequence[:-1], "targets": [sequence[-1]]}
        else:
            nested_sessions[uid] = sequence

    config = cearf.CEARFConfig(
        exclude_seen=True, component_topn=120, transition_topn=200,
        candidate_sessions=120)
    nested_index = cearf.CEARFIndex(
        nested_sessions, int(vocab["n_items"]), config)
    profiles, profile_report = cearf.tune_profiles(
        nested_index, nested_queries)
    memory_nested = nested_index.predict(nested_queries, profiles, 120)
    mgcot_nested = masked_npz(np.load(args.nested_npz), nested_queries)
    best = None
    curve = []
    for constant in (5.0, 20.0, 80.0):
        for left_weight in np.linspace(0.0, 1.0, 11):
            predictions = {
                uid: fuse(mgcot_nested[uid], memory_nested[uid],
                          float(left_weight), constant)
                for uid in nested_queries
            }
            result = metrics(predictions, nested_queries, 6)
            objective = result["HR@6"] + result["MRR@6"]
            candidate = (objective, result["HR@6"], result["MRR@6"],
                         -constant, -abs(float(left_weight) - 0.5))
            curve.append({"mgcot_weight": float(left_weight),
                          "constant": constant, "metrics": result})
            if best is None or candidate > best[0]:
                best = (candidate, float(left_weight), constant, result)

    full_index = cearf.CEARFIndex(
        full_sessions, int(vocab["n_items"]), config)
    memory_full = full_index.predict(full_queries, profiles, 120)
    mgcot_full = masked_npz(np.load(args.full_npz), full_queries)
    fused_full = {
        uid: fuse(mgcot_full[uid], memory_full[uid], best[1], best[2])
        for uid in full_queries
    }
    result = {
        "protocol": "Rental split_loo_masked; nested validation selection",
        "method": "MGCOT-CEARF rank fusion",
        "profile_report": profile_report,
        "selected": {"mgcot_weight": best[1], "constant": best[2]},
        "validation": {
            "MGCOT": metrics(mgcot_nested, nested_queries, 6),
            "CEARF": metrics(memory_nested, nested_queries, 6),
            "fusion": best[3],
            "union_candidate_recall@120": candidate_recall({
                uid: list(dict.fromkeys(mgcot_nested[uid] + memory_nested[uid]))
                for uid in nested_queries}, nested_queries),
        },
        "test": {
            "MGCOT": metrics(mgcot_full, full_queries, 6),
            "CEARF": metrics(memory_full, full_queries, 6),
            "fusion": metrics(fused_full, full_queries, 6),
            "union_candidate_recall@120": candidate_recall({
                uid: list(dict.fromkeys(mgcot_full[uid] + memory_full[uid]))
                for uid in full_queries}, full_queries),
        },
        "comparators": {"DT-RRF": 0.268752, "SKNN": 0.269456},
        "validation_curve": curve,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "validation_curve"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
