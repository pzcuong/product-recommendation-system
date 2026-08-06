#!/usr/bin/env python3
"""Run CEARF-N on the official AAAI-2026 HID Diginetica protocol."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import cearf
import hid_protocol
import pasgr
from run_cearfn import BETAS, fuse, train_neural, tune_beta


HERE = Path(__file__).resolve().parent
RECURRENCE_GAMMAS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0)


def recurrence_rerank(ranking, context, gamma, topk=120, constant=20.0):
    """Fuse behavioral retrieval with a recency-ordered repeat-intent view."""
    score = defaultdict(float)
    for rank, item in enumerate(ranking, 1):
        score[int(item)] += 1.0 / (constant + rank)
    seen = set()
    copy_rank = []
    for item in reversed(context):
        item = int(item)
        if item > 0 and item not in seen:
            seen.add(item)
            copy_rank.append(item)
    for rank, item in enumerate(copy_rank, 1):
        score[item] += float(gamma) / (constant + rank)
    return [item for item, _ in sorted(
        score.items(), key=lambda pair: (-pair[1], pair[0]))[:topk]]


def _hr_mrr(predictions, queries, k=20):
    hit = 0
    mrr = 0.0
    for uid, query in queries.items():
        target = int(query["targets"][0])
        rank = next((rank for rank, item in enumerate(
            predictions[str(uid)][:k], 1) if int(item) == target), None)
        if rank is not None:
            hit += 1
            mrr += 1.0 / rank
    n = max(len(queries), 1)
    return hit / n, mrr / n


def tune_recurrence(memory, queries, short_context=2):
    selected = {}
    report = {}
    reranked = {}
    for regime in ("short", "long"):
        keys = [str(uid) for uid, query in queries.items()
                if (len(query["context"]) <= short_context) == (regime == "short")]
        subset = {uid: queries[uid] for uid in keys}
        best = None
        for gamma in RECURRENCE_GAMMAS:
            predictions = {uid: recurrence_rerank(
                memory[uid], subset[uid]["context"], gamma) for uid in keys}
            hr, mrr = _hr_mrr(predictions, subset)
            candidate = (0.5 * hr + 0.5 * mrr, hr, mrr, -gamma, gamma)
            if best is None or candidate > best:
                best = candidate
        selected[regime] = best[-1]
        report[regime] = {"gamma": best[-1], "n": len(keys),
                          "score": best[0], "HR@20": best[1],
                          "MRR@20": best[2]}
        for uid in keys:
            reranked[uid] = recurrence_rerank(
                memory[uid], subset[uid]["context"], best[-1])
    return selected, report, reranked


def evidence_regime(query, tail_score_indices, short_context=2):
    context = query.get("context", [])
    length = "short" if len(context) <= short_context else "long"
    last_is_tail = bool(context) and (int(context[-1]) - 1 in tail_score_indices)
    return f"{length}_{'tail' if last_is_tail else 'head'}"


def tune_contextual_beta(memory, neural, queries, tail_score_indices,
                         short_context=2):
    selected = {}
    report = {}
    for regime in ("short_head", "short_tail", "long_head", "long_tail"):
        keys = [str(uid) for uid, query in queries.items()
                if evidence_regime(query, tail_score_indices,
                                   short_context) == regime]
        subset = {uid: queries[uid] for uid in keys}
        best = None
        for beta in BETAS:
            predictions = {uid: fuse(memory[uid], neural[uid], beta)
                           for uid in keys}
            r6 = cearf.recall_at(predictions, subset, 6)
            r20 = cearf.recall_at(predictions, subset, 20)
            candidate = (0.5 * r6 + 0.5 * r20, r20, r6, -beta, beta)
            if best is None or candidate > best:
                best = candidate
        selected[regime] = best[-1]
        report[regime] = {"beta": best[-1], "n": len(keys),
                          "score": best[0], "recall@6": best[2],
                          "recall@20": best[1]}
    return selected, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--inbatch-weight", type=float, default=0.0)
    parser.add_argument("--validation-cap", type=int, default=5000)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--router", choices=("length", "context"),
                        default="length")
    parser.add_argument("--selection", type=Path,
                        help="Reuse a validation-only selection artifact")
    parser.add_argument("--output", type=Path,
                        default=HERE / "cearfn_hid_diginetica.json")
    args = parser.parse_args()
    started = time.time()
    data = hid_protocol.load_hid_diginetica()
    # HID's session protocol permits repeat consumption. Roughly one fifth of
    # Diginetica test targets already occur in their context, so filtering seen
    # items would change the task and impose an artificial recall ceiling.
    config = cearf.CEARFConfig(validation_cap=args.validation_cap,
                               exclude_seen=False)
    sessions = data["train_sessions"]
    tune_sessions, validation = cearf.make_validation_split(
        sessions, config.validation_fraction, args.validation_cap)
    semantic = hid_protocol.attribute_semantic_matrix(data)
    print(f"[HID protocol] reconstructed_sessions={len(sessions)} "
          f"official_examples={data['official_examples']} "
          f"valid={len(validation)} test={len(data['test_queries'])}", flush=True)

    tune_index = tune_neural = None
    if args.selection:
        selected = json.loads(args.selection.read_text())
        selected_weight = float(selected.get("inbatch_weight", 0.0))
        if selected_weight != args.inbatch_weight:
            raise ValueError("selection inbatch_weight does not match run")
        profile_report = selected["profiles"]
        recurrence_report = selected["recurrence"]
        beta_report = selected["betas"]
        profiles = {regime: cearf.PROFILES[report["profile"]]
                    for regime, report in profile_report.items()}
        gammas = {regime: float(report["gamma"])
                  for regime, report in recurrence_report.items()}
        betas = {regime: float(report["beta"])
                 for regime, report in beta_report.items()}
    else:
        tune_index = cearf.CEARFIndex(tune_sessions, data["n_items"], config)
        profiles, profile_report = cearf.tune_profiles(tune_index, validation)
        memory_valid = tune_index.predict(validation, profiles, 120,
                                          progress="HID-valid-memory")
        gammas, recurrence_report, memory_valid = tune_recurrence(
            memory_valid, validation, config.short_context)
        tune_data = dict(data)
        tune_data["train_sessions"] = tune_sessions
        tune_neural = train_neural("HID_Diginetica", tune_sessions, tune_data,
                                   semantic, args.epochs,
                                   inbatch_weight=args.inbatch_weight)
        neural_valid = pasgr.predict_pasgr(tune_neural, validation,
                                           data["n_items"], 120,
                                           exclude_seen=False)
        if args.router == "context":
            betas, beta_report = tune_contextual_beta(
                memory_valid, neural_valid, validation,
                data["tail_score_indices"], config.short_context)
        else:
            betas, beta_report = tune_beta(
                memory_valid, neural_valid, validation,
                config.short_context)
    print(f"[HID protocol] profiles={profile_report} "
          f"recurrence={recurrence_report} betas={beta_report}", flush=True)
    if args.validation_only:
        result = {
            "protocol": "Code4HID official diginetica-2 validation only",
            "profiles": profile_report,
            "recurrence": recurrence_report,
            "betas": beta_report,
            "inbatch_weight": args.inbatch_weight,
            "seconds": time.time() - started,
        }
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        return
    del tune_index, tune_neural

    index = cearf.CEARFIndex(sessions, data["n_items"], config)
    memory_test = index.predict(data["test_queries"], profiles, 120,
                                progress="HID-test-memory")
    memory_test = {
        uid: recurrence_rerank(
            ranking, data["test_queries"][uid]["context"],
            gammas["short" if len(data["test_queries"][uid]["context"])
                   <= config.short_context else "long"])
        for uid, ranking in memory_test.items()
    }
    final_neural = train_neural("HID_Diginetica", sessions, data,
                                semantic, args.epochs,
                                inbatch_weight=args.inbatch_weight)
    neural_test = pasgr.predict_pasgr(final_neural, data["test_queries"],
                                      data["n_items"], 120,
                                      exclude_seen=False)
    fused = {}
    for uid, query in data["test_queries"].items():
        contextual = evidence_regime(query, data["tail_score_indices"],
                                     config.short_context)
        regime = contextual if contextual in betas else contextual.split("_", 1)[0]
        fused[uid] = fuse(memory_test[uid], neural_test[uid], betas[regime], 20)

    result = {
        "protocol": "Code4HID official diginetica-2 artifacts",
        "profiles": profile_report,
        "recurrence": recurrence_report,
        "betas": beta_report,
        "router": ("context" if any("_" in key for key in betas)
                   else "length"),
        "CEARF-N": hid_protocol.official_metrics(fused, data),
        "CEARF": hid_protocol.official_metrics(memory_test, data),
        "PASGR": hid_protocol.official_metrics(neural_test, data),
        "published_HID_best": {
            "model": "GCE-GNN+HID, AAAI 2026 Table 1",
            "HR@20": 0.5422, "MRR@20": 0.1918,
            "tHR@20": 0.5183, "tMRR@20": 0.1837,
            "tCov@20": 0.9421, "Tail@20": 0.4667,
        },
        "seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
