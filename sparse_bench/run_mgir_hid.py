#!/usr/bin/env python3
"""Validation-selected Multi-Granular Intent Ranking on official HID data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cearf
from graph_narm_mps import load_graph_narm
import hid_protocol
from multigranular import MultiGranularIntentIndex, fuse_three
from narm_mps import predict_narm
from run_cearfn_hid_protocol import recurrence_rerank, tune_recurrence


HERE = Path(__file__).resolve().parent
WEIGHTS = tuple(
    (a / 10, b / 10, (10 - a - b) / 10)
    for a in range(11) for b in range(11 - a)
)
CONSTANTS = (5.0, 20.0, 80.0)


def tune_fusion(memory, neural, granular, queries, data, short_context=2):
    selected, report = {}, {}
    for regime in ("short", "long"):
        keys = [str(uid) for uid, query in queries.items()
                if ((len(query["context"]) <= short_context)
                    == (regime == "short"))]
        subset = {uid: queries[uid] for uid in keys}
        subset_data = dict(data)
        subset_data["test_queries"] = subset
        best = None
        for constant in CONSTANTS:
            for weights in WEIGHTS:
                predictions = {
                    uid: fuse_three(memory[uid], neural[uid], granular[uid],
                                    weights, 20, constant)
                    for uid in keys
                }
                metrics = hid_protocol.official_metrics(predictions, subset_data)
                # Directly align selection with overall and tail top-rank.
                objective = 0.25 * (
                    metrics["HR@20"] + metrics["MRR@20"] +
                    metrics["tHR@20"] + metrics["tMRR@20"])
                candidate = (objective, metrics["MRR@20"], metrics["HR@20"],
                             metrics["tMRR@20"], metrics["tHR@20"],
                             -constant, weights)
                if best is None or candidate > best[0]:
                    best = (candidate, constant, weights, metrics)
        selected[regime] = {"constant": best[1], "weights": best[2]}
        report[regime] = {
            "constant": best[1], "weights": best[2], "n": len(keys),
            "objective": best[0][0], "metrics": best[3],
        }
    return selected, report


def apply_fusion(memory, neural, granular, queries, selected):
    output = {}
    for uid, query in queries.items():
        regime = "short" if len(query["context"]) <= 2 else "long"
        setting = selected[regime]
        output[str(uid)] = fuse_three(
            memory[str(uid)], neural[str(uid)], granular[str(uid)],
            setting["weights"], 20, setting["constant"])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-checkpoint", type=Path,
                        default=HERE / "graph_narm_validation_2ep.pt")
    parser.add_argument("--full-checkpoint", type=Path,
                        default=HERE / "graph_narm_hid_2ep.pt")
    parser.add_argument("--output", type=Path,
                        default=HERE / "mgir_hid_results.json")
    parser.add_argument("--locked-selection", type=Path,
                        help="Reuse a validation-locked fusion selection")
    args = parser.parse_args()
    started = time.time()
    data = hid_protocol.load_hid_diginetica()
    config = cearf.CEARFConfig(validation_cap=5000, exclude_seen=False)
    tune_sessions, validation = cearf.make_validation_split(
        data["train_sessions"], config.validation_fraction, 5000)

    memory_index = cearf.CEARFIndex(tune_sessions, data["n_items"], config)
    profiles, profile_report = cearf.tune_profiles(memory_index, validation)
    memory_valid = memory_index.predict(validation, profiles, 120)
    gammas, recurrence_report, memory_valid = tune_recurrence(
        memory_valid, validation, config.short_context)
    granular_index = MultiGranularIntentIndex(tune_sessions, data["n_items"])
    granular_valid = granular_index.predict(validation, 120)
    validation_model, validation_history = load_graph_narm(
        args.validation_checkpoint)
    neural_valid = predict_narm(validation_model, validation, topk=120)
    if args.locked_selection:
        locked = json.loads(args.locked_selection.read_text())
        selected = locked["selected"]
        selection_report = locked["report"]
        print(f"[MGIR] reused locked selection {args.locked_selection}",
              flush=True)
    else:
        selected, selection_report = tune_fusion(
            memory_valid, neural_valid, granular_valid, validation, data)
    fused_valid = apply_fusion(
        memory_valid, neural_valid, granular_valid, validation, selected)
    valid_data = dict(data)
    valid_data["test_queries"] = validation
    print(f"[MGIR] selection={selection_report} valid="
          f"{hid_protocol.official_metrics(fused_valid, valid_data)}", flush=True)

    memory_full = cearf.CEARFIndex(
        data["train_sessions"], data["n_items"], config)
    memory_test = memory_full.predict(
        data["test_queries"], profiles, 120, progress="test-memory")
    memory_test = {
        uid: recurrence_rerank(
            ranking, data["test_queries"][uid]["context"],
            gammas["short" if len(data["test_queries"][uid]["context"])
                   <= config.short_context else "long"])
        for uid, ranking in memory_test.items()
    }
    granular_full = MultiGranularIntentIndex(
        data["train_sessions"], data["n_items"])
    granular_test = granular_full.predict(
        data["test_queries"], 120, progress="test-granular")
    full_model, full_history = load_graph_narm(args.full_checkpoint)
    neural_test = predict_narm(full_model, data["test_queries"], topk=120)
    fused_test = apply_fusion(
        memory_test, neural_test, granular_test,
        data["test_queries"], selected)
    result = {
        "protocol": "Code4HID/MGCOT byte-identical Diginetica artifacts",
        "method": "MGIR: variable-order intent + CEARF + Graph-NARM",
        "profiles": profile_report, "recurrence": recurrence_report,
        "selection": selection_report,
        "validation_history": validation_history,
        "full_history": full_history,
        "validation": hid_protocol.official_metrics(fused_valid, valid_data),
        "test": {
            "MGIR": hid_protocol.official_metrics(fused_test, data),
            "granular": hid_protocol.official_metrics(granular_test, data),
            "memory": hid_protocol.official_metrics(memory_test, data),
            "neural": hid_protocol.official_metrics(neural_test, data),
        },
        "published": {
            "HID": {"HR@20": 0.5422, "MRR@20": 0.1918},
            "MGCOT": {"HR@20": 0.6831, "MRR@20": 0.2979},
        },
        "seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
