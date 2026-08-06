#!/usr/bin/env python3
"""Validation-gated residual reranking of fixed MGCOT candidates.

The official test set is touched only after five-fold out-of-fold validation
improves the locked MGCOT baseline.  Candidate generation stays unchanged;
the learned model may only reorder the top-120 candidates exported by MGCOT.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

import cearf
from graph_narm_mps import load_graph_narm
import hid_protocol
from mgcot_residual_ranker import (
    FEATURE_NAMES, build_matrix, oof_gated_predictions, predict,
)
from multigranular import MultiGranularIntentIndex
from narm_mps import predict_narm
from run_cearfn_hid_protocol import recurrence_rerank, tune_recurrence


HERE = Path(__file__).resolve().parent
ENTROPY_THRESHOLDS = (0.00, 0.70, 0.74, 0.77, 0.80, 0.82,
                      0.84, 0.86, 0.88, 0.90, 1.01)


def npz_predictions(bundle, uids, topk=20):
    # NpzFile.__getitem__ decompresses the complete member on every access.
    # Materialize once before iterating; otherwise this becomes O(n_queries)
    # full decompressions (hours on the official test split).
    items = bundle["items"]
    return {
        str(uid): [int(x) for x in items[row, :topk]]
        for row, uid in enumerate(uids)
    }


def candidate_recall(bundle):
    items = bundle["items"]
    targets = bundle["targets"]
    return float(np.mean(np.any(items == targets[:, None], axis=1)))


def assert_alignment(bundle, queries, label):
    targets = np.asarray(
        [int(query["targets"][0]) for query in queries.values()],
        dtype=np.int32,
    )
    if len(targets) != len(bundle["targets"]):
        raise ValueError(f"{label}: row mismatch")
    if not np.array_equal(targets, bundle["targets"]):
        mismatch = int(np.flatnonzero(targets != bundle["targets"])[0])
        raise ValueError(f"{label}: target mismatch at row {mismatch}")


def auxiliary_views(sessions, queries, data, config, profiles=None,
                    gammas=None, checkpoint=None, progress=None):
    memory_index = cearf.CEARFIndex(sessions, data["n_items"], config)
    if profiles is None:
        profiles, profile_report = cearf.tune_profiles(memory_index, queries)
    else:
        profile_report = None
    memory = memory_index.predict(queries, profiles, 120, progress=progress)
    if gammas is None:
        gammas, recurrence_report, memory = tune_recurrence(
            memory, queries, config.short_context)
    else:
        recurrence_report = None
        memory = {
            str(uid): recurrence_rerank(
                memory[str(uid)], query["context"],
                gammas["short" if len(query["context"]) <=
                       config.short_context else "long"],
            )
            for uid, query in queries.items()
        }
    granular = MultiGranularIntentIndex(
        sessions, data["n_items"]).predict(queries, 120, progress=progress)
    neural_model, neural_history = load_graph_narm(checkpoint)
    neural = predict_narm(neural_model, queries, topk=120)
    return {
        "memory": memory,
        "granular": granular,
        "neural": neural,
        "frequency": memory_index.freq,
        "profiles": profiles,
        "profile_report": profile_report,
        "gammas": gammas,
        "recurrence_report": recurrence_report,
        "neural_history": neural_history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-npz", type=Path,
                        default=HERE / "mgcot_nested_valid_top120.npz")
    parser.add_argument("--test-npz", type=Path,
                        default=HERE / "mgcot_full_test_top120.npz")
    parser.add_argument("--validation-checkpoint", type=Path,
                        default=HERE / "graph_narm_validation_2ep.pt")
    parser.add_argument("--full-checkpoint", type=Path,
                        default=HERE / "graph_narm_hid_2ep.pt")
    parser.add_argument("--output", type=Path,
                        default=HERE / "mgcot_residual_hid_results.json")
    parser.add_argument("--model-output", type=Path,
                        default=HERE / "mgcot_residual_ranker.pt")
    args = parser.parse_args()
    started = time.time()
    data = hid_protocol.load_hid_diginetica()
    config = cearf.CEARFConfig(validation_cap=5000, exclude_seen=False)
    tune_sessions, validation = cearf.make_validation_split(
        data["train_sessions"], config.validation_fraction,
        config.validation_cap)
    valid_bundle = np.load(args.validation_npz)
    assert_alignment(valid_bundle, validation, "validation")
    print("[MGCOT-RERANK] building validation views", flush=True)
    views = auxiliary_views(
        tune_sessions, validation, data, config,
        checkpoint=args.validation_checkpoint)
    matrix = build_matrix(
        valid_bundle["items"], valid_bundle["scores"], validation,
        views["neural"], views["memory"], views["granular"],
        views["frequency"], data["tail_score_indices"],
    )
    baseline_valid = npz_predictions(valid_bundle, validation, 20)
    valid_data = dict(data)
    valid_data["test_queries"] = validation
    baseline_metrics = hid_protocol.official_metrics(
        baseline_valid, valid_data)
    print(f"[MGCOT-RERANK] baseline-valid={baseline_metrics} "
          f"candidate-recall@120={candidate_recall(valid_bundle):.6f}",
          flush=True)
    final_model, gated_oof = oof_gated_predictions(
        matrix, ENTROPY_THRESHOLDS, folds=5)
    gated_metrics = {
        threshold: hid_protocol.official_metrics(predictions, valid_data)
        for threshold, predictions in gated_oof.items()
    }
    baseline_objective = baseline_metrics["HR@20"] + baseline_metrics["MRR@20"]
    safe = [
        threshold for threshold, metrics in gated_metrics.items()
        if metrics["HR@20"] >= baseline_metrics["HR@20"]
        and metrics["MRR@20"] >= baseline_metrics["MRR@20"]
        and metrics["HR@20"] + metrics["MRR@20"] > baseline_objective
    ]
    # Maximal threshold = minimum fraction of queries modified.
    selected_threshold = max(safe) if safe else None
    oof_metrics = (gated_metrics[selected_threshold]
                   if selected_threshold is not None else baseline_metrics)
    gate_passed = selected_threshold is not None
    report = {
        "protocol": "validation-gated fixed-candidate reranking",
        "method": "MGCOT-RR: residual multi-view linear reranker",
        "features": list(FEATURE_NAMES),
        "validation": {
            "baseline": baseline_metrics,
            "oof_reranked": oof_metrics,
            "candidate_recall@120": candidate_recall(valid_bundle),
            "gate_objective": "HR@20 + MRR@20",
            "selection_rule": "highest entropy threshold with non-decreasing "
                              "OOF HR and MRR",
            "selected_entropy_threshold": selected_threshold,
            "threshold_metrics": {
                str(k): v for k, v in gated_metrics.items()
            },
            "gate_passed": gate_passed,
        },
        "limitations": [
            "The nested checkpoint uses the public MGCOT adjacency whose "
            "held-target provenance cannot be proven absent.",
            "The reranker reorders MGCOT top-120 candidates and cannot recover "
            "targets outside that set.",
        ],
    }
    print(f"[MGCOT-RERANK] oof-valid={oof_metrics} gate={gate_passed}",
          flush=True)
    if not gate_passed:
        report["test"] = "SKIPPED: out-of-fold validation gate failed"
        report["seconds"] = time.time() - started
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return

    print("[MGCOT-RERANK] validation gate passed; building locked test views",
          flush=True)
    test_bundle = np.load(args.test_npz)
    assert_alignment(test_bundle, data["test_queries"], "test")
    full_views = auxiliary_views(
        data["train_sessions"], data["test_queries"], data, config,
        profiles=views["profiles"], gammas=views["gammas"],
        checkpoint=args.full_checkpoint, progress="test-views")
    test_matrix = build_matrix(
        test_bundle["items"], test_bundle["scores"], data["test_queries"],
        full_views["neural"], full_views["memory"],
        full_views["granular"], full_views["frequency"],
        data["tail_score_indices"],
    )
    baseline_test = npz_predictions(test_bundle, data["test_queries"], 20)
    reranked_test = predict(
        final_model, test_matrix, topk=20,
        entropy_threshold=selected_threshold)
    report["test"] = {
        "baseline": hid_protocol.official_metrics(baseline_test, data),
        "reranked": hid_protocol.official_metrics(reranked_test, data),
        "candidate_recall@120": candidate_recall(test_bundle),
    }
    report["published"] = {
        "HID": {"HR@20": 0.5422, "MRR@20": 0.1918},
        "MGCOT": {"HR@20": 0.6831, "MRR@20": 0.2979},
    }
    report["seconds"] = time.time() - started
    torch.save({
        "state_dict": final_model.cpu().state_dict(),
        "features": FEATURE_NAMES,
        "validation_metrics": oof_metrics,
    }, args.model_output)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
