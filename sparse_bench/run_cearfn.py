#!/usr/bin/env python3
"""CEARF-N: validation-routed fusion of CEARF memory and PASGR neural ranks."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import cearf
import loaders
import pasgr
from run_pasgr_full import semantic_matrix


HERE = Path(__file__).resolve().parent
BETAS = tuple(round(step * 0.05, 2) for step in range(21))


def fuse(memory, neural, beta, topk=20, constant=20.0):
    score = defaultdict(float)
    if beta < 1.0:
        for rank, item in enumerate(memory, 1):
            score[int(item)] += (1.0 - beta) / (constant + rank)
    if beta > 0.0:
        for rank, item in enumerate(neural, 1):
            score[int(item)] += beta / (constant + rank)
    return [item for item, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))[:topk]]


def tune_beta(memory, neural, queries, short_context=2):
    selected = {}
    report = {}
    for regime in ("short", "long"):
        keys = [str(uid) for uid, q in queries.items()
                if (len(q.get("context", [])) <= short_context) == (regime == "short")]
        if not keys:
            selected[regime] = 0.0
            report[regime] = {"beta": 0.0, "n": 0}
            continue
        best = None
        subset = {uid: queries[uid] for uid in keys}
        for beta in BETAS:
            pred = {uid: fuse(memory[uid], neural[uid], beta) for uid in keys}
            r6 = cearf.recall_at(pred, subset, 6)
            r20 = cearf.recall_at(pred, subset, 20)
            candidate = (0.5 * r6 + 0.5 * r20, r20, r6, -beta, beta)
            if best is None or candidate > best:
                best = candidate
        selected[regime] = best[-1]
        report[regime] = {"beta": best[-1], "n": len(keys), "score": best[0],
                          "recall@6": best[2], "recall@20": best[1]}
    return selected, report


def train_neural(domain, sessions, data, semantic, epochs, seed=42,
                 inbatch_weight=0.0):
    freq = Counter(x for seq in sessions.values() for x in seq)
    config = pasgr.PASGRConfig(
        dim=64, prototypes=min(96, max(8, data["n_items"] // 250)),
        epochs=epochs, batch_size=512, hard_negatives=32,
        top_k=120, seed=seed, inbatch_weight=inbatch_weight)
    assets = pasgr.build_prototype_graph_embeddings(
        sessions, data["n_items"], freq, semantic, config)
    return pasgr.train_pasgr(sessions, data["n_items"], freq, semantic,
                             config, prepared_assets=assets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=["Rental_visit", "RetailRocket"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--validation-cap", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=HERE / "cearfn_results.json")
    args = parser.parse_args()
    config = cearf.CEARFConfig(validation_cap=args.validation_cap)
    results = {}
    for domain in args.domains:
        started = time.time()
        data = loaders.ALL_LOADERS[domain]()
        sessions = data["train_sessions"]
        validation = data.get("valid_queries") or {}
        if validation:
            tune_sessions = sessions
            if len(validation) > args.validation_cap:
                keys = sorted(validation, key=cearf._stable_fraction)[:args.validation_cap]
                validation = {str(key): validation[key] for key in keys}
        else:
            tune_sessions, validation = cearf.make_validation_split(
                sessions, config.validation_fraction, args.validation_cap)
        semantic = semantic_matrix(domain, data)
        print(f"[CEARF-N] {domain} training validation neural", flush=True)
        tune_index = cearf.CEARFIndex(tune_sessions, data["n_items"], config)
        profiles, profile_report = cearf.tune_profiles(tune_index, validation)
        memory_valid = tune_index.predict(validation, profiles, 120)
        tune_neural = train_neural(domain, tune_sessions, data, semantic, args.epochs)
        neural_valid = pasgr.predict_pasgr(tune_neural, validation, data["n_items"], 120)
        betas, beta_report = tune_beta(memory_valid, neural_valid, validation)
        print(f"[CEARF-N] {domain} profiles={profile_report} betas={beta_report}", flush=True)
        del tune_index, tune_neural

        print(f"[CEARF-N] {domain} training final neural", flush=True)
        index = cearf.CEARFIndex(sessions, data["n_items"], config)
        memory_test = index.predict(data["test_queries"], profiles, 120,
                                    progress=f"{domain}-memory")
        final_neural = train_neural(domain, sessions, data, semantic, args.epochs)
        neural_test = pasgr.predict_pasgr(final_neural, data["test_queries"],
                                          data["n_items"], 120)
        fused = {}
        for uid, query in data["test_queries"].items():
            regime = "short" if len(query.get("context", [])) <= config.short_context else "long"
            fused[str(uid)] = fuse(memory_test[str(uid)], neural_test[str(uid)],
                                   betas[regime], 20)
        result = {
            "dataset": domain, "profiles": profile_report, "betas": beta_report,
            "CEARF-N": cearf.ranking_metrics(fused, data["test_queries"]),
            "CEARF": cearf.ranking_metrics(memory_test, data["test_queries"]),
            "PASGR": cearf.ranking_metrics(neural_test, data["test_queries"]),
            "seconds": time.time() - started,
        }
        results[domain] = result
        args.output.write_text(json.dumps(results, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
