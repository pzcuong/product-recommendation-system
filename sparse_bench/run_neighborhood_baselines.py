#!/usr/bin/env python3
"""Validation-tune V-SKNN/STAN and a first-order transition baseline.

V-SKNN and STAN are deterministic training-free algorithms.  We nevertheless
emit the paper's three matched seed IDs so tables and paired-analysis code can
use one schema; their per-seed values are intentionally identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

import cearf
import loaders
from neighborhood_baselines import NeighborhoodConfig, NeighborhoodIndex
from run_cearfn_evidence import metrics_from_ranks, ranks_at_20, targets_for
from validation_protocol import hold_out_validation_targets

HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Arts_Crafts_and_Sewing", "Diginetica_HID")
SEEDS = (42, 123, 456)
REPEAT_PROTOCOL_DOMAINS = frozenset({"Diginetica_HID", "Tmall"})


def matrix_from_predictions(predictions, keys, width=20):
    return np.asarray([predictions[k][:width] for k in keys], dtype=np.int32)


def evaluate(predictions, queries):
    keys = [str(k) for k in queries]
    matrix = matrix_from_predictions(predictions, keys)
    targets = targets_for(keys, queries)
    return metrics_from_ranks(ranks_at_20(matrix, targets)), ranks_at_20(matrix, targets)


def utility(metrics):
    return .5 * (metrics["recall@6"] + metrics["recall@20"])


def config_grid(method, exclude_seen):
    if method == "vsknn":
        return [NeighborhoodConfig(method=method, k=k, sample_size=sample,
                                   weighting=weighting, score_weighting="div",
                                   exclude_seen=exclude_seen)
                for k in (100, 500) for sample in (1000, 5000)
                for weighting in ("div", "quadratic")]
    # Session-order time decay is included as a validation-gated option.  None
    # cleanly disables it when ordinal recency is not informative.
    return [NeighborhoodConfig(method=method, k=k, sample_size=sample,
                               lambda_spw=spw, lambda_snh=snh,
                               lambda_inh=2.05, exclude_seen=exclude_seen)
            for k in (100, 500) for sample in (1000, 5000)
            for spw in (1.02, 2.0) for snh in (None, 5000.0)]


def tune(index, validation, method, exclude_seen):
    rows = []
    best = None
    for cfg in config_grid(method, exclude_seen):
        started = time.time()
        pred = index.predict(validation, cfg)
        metrics, _ = evaluate(pred, validation)
        row = {"config": cfg.__dict__, "metrics": metrics,
               "utility": utility(metrics), "seconds": time.time() - started}
        rows.append(row)
        candidate = (row["utility"], metrics["recall@20"], metrics["recall@6"],
                     -cfg.k, -cfg.sample_size)
        if best is None or candidate > best[0]:
            best = (candidate, cfg)
        print(f"[NEIGHBOR] {method} {cfg} utility={row['utility']:.6f}", flush=True)
    return best[1], rows


def transition_predict(index, queries, exclude_seen):
    output = {}
    for uid, query in queries.items():
        context = query.get("context", ())
        transition, _, popularity = index.component_rankings(context)
        blocked = set(context) if exclude_seen else set()
        rank = [x for x in transition if x not in blocked]
        chosen = set(rank)
        rank.extend(x for x in popularity if x not in blocked and x not in chosen)
        output[str(uid)] = rank[:20]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--output", type=Path,
                        default=HERE / "neighborhood_baseline_results.json")
    parser.add_argument("--artifact-dir", type=Path,
                        default=HERE / "neighborhood_baseline_artifacts")
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    for domain in args.domains:
        if domain in results and results[domain].get("complete"):
            print(f"[NEIGHBOR] {domain} already complete", flush=True)
            continue
        data = loaders.ALL_LOADERS[domain]()
        if len(data["valid_queries"]) > 5000:
            keys = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
            validation = {k: data["valid_queries"][k] for k in keys}
        else:
            validation = data["valid_queries"]
        test = data["test_queries"]
        tune_sessions = hold_out_validation_targets(
            data["train_sessions"], validation)
        exclude_seen = domain not in REPEAT_PROTOCOL_DOMAINS
        tune_neighbor_index = NeighborhoodIndex(tune_sessions, data["n_items"])
        final_neighbor_index = NeighborhoodIndex(
            data["train_sessions"], data["n_items"])
        cearf_index = cearf.CEARFIndex(
            data["train_sessions"], data["n_items"],
            cearf.CEARFConfig(exclude_seen=exclude_seen))
        block = results.get(domain, {"protocol": {
            "validation_queries": len(validation), "test_queries": len(test),
            "selection_uses_test_labels": False, "deterministic": True,
            "reported_seed_ids": args.seeds,
            "timestamp_adaptation": "loader session order; validation may disable STAN time decay",
            "exclude_seen": exclude_seen}, "methods": {}})
        for method in ("vsknn", "stan"):
            if method in block["methods"] and "test" in block["methods"][method]:
                print(f"[NEIGHBOR] {domain} {method} already complete", flush=True)
                continue
            chosen, grid = tune(
                tune_neighbor_index, validation, method, exclude_seen)
            started = time.time()
            test_pred = final_neighbor_index.predict(test, chosen)
            test_metrics, test_ranks = evaluate(test_pred, test)
            test_matrix = matrix_from_predictions(test_pred, [str(k) for k in test])
            artifact = args.artifact_dir / f"{domain.lower()}_{method}_ranks.npz"
            np.savez_compressed(artifact, top20=test_matrix.astype(np.int32),
                                ranks=test_ranks.astype(np.uint8),
                                keys=np.asarray(list(test), dtype=str))
            block["methods"][method] = {
                "selected_config": chosen.__dict__, "validation_grid": grid,
                "test": test_metrics,
                "per_seed": {str(seed): test_metrics for seed in args.seeds},
                "artifact": str(artifact), "test_seconds": time.time() - started}
            args.output.write_text(json.dumps({**results, domain: block}, indent=2))
        transition_pred = transition_predict(cearf_index, test, exclude_seen)
        transition_metrics, transition_ranks = evaluate(transition_pred, test)
        transition_matrix = matrix_from_predictions(transition_pred, [str(k) for k in test])
        artifact = args.artifact_dir / f"{domain.lower()}_transition_ranks.npz"
        np.savez_compressed(artifact, top20=transition_matrix.astype(np.int32),
                            ranks=transition_ranks.astype(np.uint8),
                            keys=np.asarray(list(test), dtype=str))
        block["methods"]["transition"] = {
            "test": transition_metrics,
            "per_seed": {str(seed): transition_metrics for seed in args.seeds},
            "artifact": str(artifact)}
        block["complete"] = True
        results[domain] = block
        args.output.write_text(json.dumps(results, indent=2))
    print(f"[NEIGHBOR] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
