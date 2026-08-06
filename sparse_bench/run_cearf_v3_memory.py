#!/usr/bin/env python3
"""Leakage-safe memory-only CEARF-v3 evaluation.

This runner is intentionally limited to the cheap M4/M5 memory experiment.
It holds validation targets out of the tuning index, selects profiles only on
validation, rebuilds one full index for test, and persists per-query ranks for
paired analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cearf
import loaders
from cearf_v3_ext import CEARFIndexV3, SemanticMemory, tune_profiles_v3
from run_cearfn_evidence import (
    metrics_from_ranks,
    query_fingerprint,
    ranks_at_20,
    targets_for,
)
from validation_protocol import hold_out_validation_targets


def prediction_matrix(predictions: dict[str, list[int]],
                      keys: list[str]) -> np.ndarray:
    return np.asarray([predictions[key] for key in keys], dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True,
                        choices=("Video_Games", "Baby_Products",
                                 "Diginetica_HID", "Tmall"))
    parser.add_argument("--semantic-matrix", type=Path)
    parser.add_argument("--use-repeat", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank-artifact", required=True, type=Path)
    parser.add_argument("--validation-cap", type=int, default=5000)
    args = parser.parse_args()

    data = loaders.ALL_LOADERS[args.domain]()
    exclude_seen = args.domain not in {"Diginetica_HID", "Tmall"}
    if args.use_repeat and exclude_seen:
        raise ValueError("--use-repeat is meaningful only for repeat protocols")

    valid_keys = sorted(data["valid_queries"], key=cearf._stable_fraction)[
        :args.validation_cap]
    validation = {key: data["valid_queries"][key] for key in valid_keys}
    tune_sessions = hold_out_validation_targets(
        data["train_sessions"], validation)

    semantic = None
    if args.semantic_matrix:
        semantic = SemanticMemory(np.load(args.semantic_matrix))
    config = cearf.CEARFConfig(exclude_seen=exclude_seen)
    tune_index = CEARFIndexV3(
        tune_sessions, data["n_items"], config,
        semantic=semantic, use_repeat=args.use_repeat)
    profiles, profile_report = tune_profiles_v3(tune_index, validation)
    valid_predictions = tune_index.predict(
        validation, profiles, progress=f"{args.domain}-valid-v3")

    test_index = CEARFIndexV3(
        data["train_sessions"], data["n_items"], config,
        semantic=semantic, use_repeat=args.use_repeat)
    test_predictions = test_index.predict(
        data["test_queries"], profiles, progress=f"{args.domain}-test-v3")

    valid_order = sorted(validation)
    test_order = sorted(data["test_queries"])
    valid_matrix = prediction_matrix(valid_predictions, valid_order)
    test_matrix = prediction_matrix(test_predictions, test_order)
    valid_ranks = ranks_at_20(
        valid_matrix, targets_for(valid_order, validation))
    test_ranks = ranks_at_20(
        test_matrix, targets_for(test_order, data["test_queries"]))

    args.rank_artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.rank_artifact,
        valid_keys=np.asarray(valid_order, dtype=str),
        test_keys=np.asarray(test_order, dtype=str),
        valid_top20=valid_matrix,
        test_top20=test_matrix,
        valid_ranks=valid_ranks,
        test_ranks=test_ranks,
        validation_fingerprint=np.asarray(query_fingerprint(validation)),
        test_fingerprint=np.asarray(query_fingerprint(data["test_queries"])),
    )
    result = {
        "domain": args.domain,
        "protocol": {
            "exclude_seen": exclude_seen,
            "use_repeat": args.use_repeat,
            "semantic_matrix": (str(args.semantic_matrix)
                                if args.semantic_matrix else None),
            "validation_queries": len(validation),
            "test_queries": len(data["test_queries"]),
            "selection_uses_test_labels": False,
        },
        "selected_profiles": profile_report,
        "validation": metrics_from_ranks(valid_ranks),
        "test": metrics_from_ranks(test_ranks),
        "rank_artifact": str(args.rank_artifact),
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
