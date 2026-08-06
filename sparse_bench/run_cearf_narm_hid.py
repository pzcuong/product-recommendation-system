#!/usr/bin/env python3
"""Validation-selected CEARF + dense NARM fusion on official Diginetica."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import torch

import cearf
import hid_protocol
from graph_narm_mps import load_graph_narm
from narm_mps import (NARMConfig, expand_sessions, load_narm, predict_narm,
                      train_narm)
from run_cearfn import fuse, tune_beta
from run_cearfn_hid_protocol import recurrence_rerank, tune_recurrence


HERE = Path(__file__).resolve().parent
METRIC_BETAS = tuple(step / 40 for step in range(41))
RRF_CONSTANTS = (5.0, 10.0, 20.0, 40.0, 80.0)


def load_sequence_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "graph_gate" in payload["state_dict"]:
        return load_graph_narm(path)
    return load_narm(path)


def tune_metric_fusion(memory, neural, queries, data, short_context=2):
    """Tune RRF on a balanced head/tail ranking objective, validation only."""
    selected = {}
    report = {}
    for regime in ("short", "long"):
        keys = [str(uid) for uid, query in queries.items()
                if ((len(query["context"]) <= short_context)
                    == (regime == "short"))]
        subset_queries = {uid: queries[uid] for uid in keys}
        subset_data = dict(data)
        subset_data["test_queries"] = subset_queries
        best = None
        for constant in RRF_CONSTANTS:
            for beta in METRIC_BETAS:
                predictions = {
                    uid: fuse(memory[uid], neural[uid], beta, 20, constant)
                    for uid in keys
                }
                metrics = hid_protocol.official_metrics(
                    predictions, subset_data)
                objective = 0.25 * (metrics["HR@20"] + metrics["MRR@20"] +
                                    metrics["tHR@20"] + metrics["tMRR@20"])
                candidate = (objective, metrics["MRR@20"], metrics["HR@20"],
                             metrics["tMRR@20"], metrics["tHR@20"],
                             -constant, -beta)
                if best is None or candidate > best[0]:
                    best = (candidate, beta, constant, metrics)
        selected[regime] = {"beta": best[1], "constant": best[2]}
        report[regime] = {
            "beta": best[1], "constant": best[2], "n": len(keys),
            "objective": best[0][0], "metrics": best[3],
        }
    return selected, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--full-checkpoint", type=Path,
                        default=HERE / "narm_hid_full_results.pt")
    parser.add_argument("--validation-checkpoint", type=Path,
                        help="Reuse a validation-trained NARM checkpoint")
    parser.add_argument("--metric-tuner", action="store_true",
                        help="Tune beta and RRF constant on balanced HR/MRR/tail metrics")
    parser.add_argument("--output", type=Path,
                        default=HERE / "cearf_narm_hid_results.json")
    args = parser.parse_args()
    started = time.time()
    data = hid_protocol.load_hid_diginetica()
    config = cearf.CEARFConfig(validation_cap=5000, exclude_seen=False)
    tune_sessions, validation = cearf.make_validation_split(
        data["train_sessions"], config.validation_fraction, 5000)
    tune_index = cearf.CEARFIndex(tune_sessions, data["n_items"], config)
    profiles, profile_report = cearf.tune_profiles(tune_index, validation)
    memory_valid = tune_index.predict(validation, profiles, 120,
                                      progress="NARM-valid-memory")
    gammas, recurrence_report, memory_valid = tune_recurrence(
        memory_valid, validation, config.short_context)
    if args.validation_checkpoint:
        tune_model, history = load_sequence_checkpoint(
            args.validation_checkpoint)
        print(f"[CEARF-NARM] validation_checkpoint="
              f"{args.validation_checkpoint} validation={len(validation)}",
              flush=True)
    else:
        contexts, targets = expand_sessions(tune_sessions)
        narm_config = NARMConfig(n_items=data["n_items"], dim=100,
                                 batch_size=args.batch_size,
                                 epochs=args.epochs)
        print(f"[CEARF-NARM] validation_train={len(targets)} "
              f"validation={len(validation)}", flush=True)
        tune_model, history = train_narm(
            contexts, targets, narm_config,
            checkpoint=args.output.with_suffix(".validation.pt"))
    narm_valid = predict_narm(tune_model, validation, topk=120)
    if args.metric_tuner:
        betas, beta_report = tune_metric_fusion(
            memory_valid, narm_valid, validation, data,
            config.short_context)
    else:
        betas, beta_report = tune_beta(memory_valid, narm_valid, validation,
                                       config.short_context)
    fused_valid = {}
    for uid, query in validation.items():
        regime = "short" if len(query["context"]) <= config.short_context else "long"
        setting = betas[regime]
        beta = setting["beta"] if isinstance(setting, dict) else setting
        constant = (setting["constant"] if isinstance(setting, dict)
                    else 20.0)
        fused_valid[uid] = fuse(memory_valid[uid], narm_valid[uid],
                                beta, 20, constant)
    valid_data = dict(data)
    valid_data["test_queries"] = validation
    print(f"[CEARF-NARM] betas={beta_report} "
          f"valid={hid_protocol.official_metrics(fused_valid, valid_data)}",
          flush=True)
    del tune_model, tune_index

    index = cearf.CEARFIndex(data["train_sessions"], data["n_items"], config)
    memory_test = index.predict(data["test_queries"], profiles, 120,
                                progress="NARM-test-memory")
    memory_test = {
        uid: recurrence_rerank(
            ranking, data["test_queries"][uid]["context"],
            gammas["short" if len(data["test_queries"][uid]["context"])
                   <= config.short_context else "long"])
        for uid, ranking in memory_test.items()
    }
    full_model, full_history = load_sequence_checkpoint(args.full_checkpoint)
    narm_test = predict_narm(full_model, data["test_queries"], topk=120)
    fused_test = {}
    for uid, query in data["test_queries"].items():
        regime = "short" if len(query["context"]) <= config.short_context else "long"
        setting = betas[regime]
        beta = setting["beta"] if isinstance(setting, dict) else setting
        constant = (setting["constant"] if isinstance(setting, dict)
                    else 20.0)
        fused_test[uid] = fuse(memory_test[uid], narm_test[uid],
                               beta, 20, constant)
    result = {
        "protocol": "Code4HID/MGCOT byte-identical Diginetica artifacts",
        "method": "CEARF-NARM",
        "metric_tuner": args.metric_tuner,
        "validation_checkpoint": (str(args.validation_checkpoint)
                                  if args.validation_checkpoint else None),
        "full_checkpoint": str(args.full_checkpoint),
        "profiles": profile_report,
        "recurrence": recurrence_report,
        "betas": beta_report,
        "validation_history": history,
        "full_history": full_history,
        "validation": hid_protocol.official_metrics(fused_valid, valid_data),
        "test": {
            "CEARF-NARM": hid_protocol.official_metrics(fused_test, data),
            "CEARF": hid_protocol.official_metrics(memory_test, data),
            "NARM": hid_protocol.official_metrics(narm_test, data),
        },
        "published": {
            "HID_HR@20": 0.5422, "HID_MRR@20": 0.1918,
            "MGCOT_HR@20": 0.6831, "MGCOT_MRR@20": 0.2979,
        },
        "seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
