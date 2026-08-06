#!/usr/bin/env python3
"""Official HID run with out-of-fold LambdaMART evidence fusion."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

import cearf
import hid_protocol
import ltr_fusion
import pasgr
from run_cearfn import fuse, train_neural, tune_beta
from run_cearfn_hid_protocol import recurrence_rerank, tune_recurrence


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--validation-cap", type=int, default=5000)
    parser.add_argument("--output", type=Path,
                        default=HERE / "cearfn_ltr_hid.json")
    args = parser.parse_args()
    started = time.time()
    data = hid_protocol.load_hid_diginetica()
    config = cearf.CEARFConfig(validation_cap=args.validation_cap,
                               exclude_seen=False)
    sessions = data["train_sessions"]
    tune_sessions, validation = cearf.make_validation_split(
        sessions, config.validation_fraction, args.validation_cap)
    semantic = hid_protocol.attribute_semantic_matrix(data)
    print(f"[CEARF-LTR] sessions={len(sessions)} valid={len(validation)} "
          f"test={len(data['test_queries'])}", flush=True)

    tune_index = cearf.CEARFIndex(tune_sessions, data["n_items"], config)
    profiles, profile_report = cearf.tune_profiles(tune_index, validation)
    memory_valid = tune_index.predict(validation, profiles, 120,
                                      progress="LTR-valid-memory")
    gammas, recurrence_report, memory_valid = tune_recurrence(
        memory_valid, validation, config.short_context)
    tune_data = dict(data)
    tune_data["train_sessions"] = tune_sessions
    tune_neural = train_neural("HID_Diginetica", tune_sessions, tune_data,
                               semantic, args.epochs)
    neural_valid = pasgr.predict_pasgr(
        tune_neural, validation, data["n_items"], 120, exclude_seen=False)

    betas, beta_report = tune_beta(memory_valid, neural_valid, validation,
                                   config.short_context)
    rrf_valid = {}
    union_valid = {}
    for uid, query in validation.items():
        regime = "short" if len(query["context"]) <= config.short_context else "long"
        rrf_valid[uid] = fuse(memory_valid[uid], neural_valid[uid],
                              betas[regime], 120)
        union_valid[uid] = list(dict.fromkeys(
            memory_valid[uid] + neural_valid[uid]))
    matrix_valid = ltr_fusion.build_candidate_matrix(
        memory_valid, neural_valid, validation,
        Counter(item for seq in tune_sessions.values() for item in seq),
        data["tail_score_indices"])
    ranker, ltr_oof = ltr_fusion.cross_validated_fit(matrix_valid, folds=5)
    blend_weights, blend_report = tune_beta(
        rrf_valid, ltr_oof, validation, config.short_context)
    blend_valid = {}
    for uid, query in validation.items():
        regime = "short" if len(query["context"]) <= config.short_context else "long"
        blend_valid[uid] = fuse(rrf_valid[uid], ltr_oof[uid],
                                blend_weights[regime], 20)
    valid_data = dict(data)
    valid_data["test_queries"] = validation
    oof_metrics = hid_protocol.official_metrics(ltr_oof, valid_data)
    rrf_metrics = hid_protocol.official_metrics(rrf_valid, valid_data)
    blend_metrics = hid_protocol.official_metrics(blend_valid, valid_data)
    candidate_recall = cearf.recall_at(union_valid, validation, 240)
    model_path = args.output.with_suffix(".ranker.pt")
    torch.save({"state_dict": ranker.state_dict(),
                "feature_names": ltr_fusion.FEATURE_NAMES}, model_path)
    print(f"[CEARF-LTR] candidate_recall={candidate_recall:.6f} "
          f"OOF={oof_metrics} RRF={rrf_metrics} "
          f"blend={blend_report} metrics={blend_metrics}", flush=True)
    del tune_index, tune_neural, matrix_valid

    index = cearf.CEARFIndex(sessions, data["n_items"], config)
    memory_test = index.predict(data["test_queries"], profiles, 120,
                                progress="LTR-test-memory")
    memory_test = {
        uid: recurrence_rerank(
            ranking, data["test_queries"][uid]["context"],
            gammas["short" if len(data["test_queries"][uid]["context"])
                   <= config.short_context else "long"])
        for uid, ranking in memory_test.items()
    }
    final_neural = train_neural("HID_Diginetica", sessions, data,
                                semantic, args.epochs)
    neural_test = pasgr.predict_pasgr(
        final_neural, data["test_queries"], data["n_items"], 120,
        exclude_seen=False)
    matrix_test = ltr_fusion.build_candidate_matrix(
        memory_test, neural_test, data["test_queries"],
        Counter(item for seq in sessions.values() for item in seq),
        data["tail_score_indices"])
    ltr_test = ltr_fusion.predict_ranker(ranker, matrix_test, topk=120)
    rrf_test = {}
    blend_test = {}
    union_test = {}
    for uid, query in data["test_queries"].items():
        regime = "short" if len(query["context"]) <= config.short_context else "long"
        rrf_test[uid] = fuse(memory_test[uid], neural_test[uid],
                             betas[regime], 120)
        blend_test[uid] = fuse(rrf_test[uid], ltr_test[uid],
                               blend_weights[regime], 20)
        union_test[uid] = list(dict.fromkeys(memory_test[uid] + neural_test[uid]))

    result = {
        "protocol": "Code4HID official diginetica-2 artifacts",
        "method": "CEARF-LTR",
        "profiles": profile_report,
        "recurrence": recurrence_report,
        "betas": beta_report,
        "validation": {
            "candidate_recall@union": candidate_recall,
            "LTR-OOF": oof_metrics,
            "RRF": rrf_metrics,
            "blend_selection": blend_report,
            "RRF-LTR": blend_metrics,
        },
        "test": {
            "candidate_recall@union": cearf.recall_at(
                union_test, data["test_queries"], 240),
            "CEARF-RRF-LTR": hid_protocol.official_metrics(blend_test, data),
            "CEARF-LTR": hid_protocol.official_metrics(ltr_test, data),
            "CEARF-N-RRF": hid_protocol.official_metrics(rrf_test, data),
            "CEARF": hid_protocol.official_metrics(memory_test, data),
            "PASGR": hid_protocol.official_metrics(neural_test, data),
        },
        "published_HID_best": {
            "model": "GCE-GNN+HID, AAAI 2026 Table 1",
            "HR@20": 0.5422, "MRR@20": 0.1918,
            "tHR@20": 0.5183, "tMRR@20": 0.1837,
            "tCov@20": 0.9421, "Tail@20": 0.4667,
        },
        "ranker_model": str(model_path),
        "feature_names": list(ltr_fusion.FEATURE_NAMES),
        "seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
