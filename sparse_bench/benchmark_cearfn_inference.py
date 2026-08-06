#!/usr/bin/env python3
"""Benchmark warm CEARF-N inference, excluding training and index construction.

The benchmark uses the locked seed-42 PASGR cell and router family.  It times:
1) CEARF memory retrieval from an already-built index;
2) PASGR full-catalogue top-120 prediction from an already-trained model;
3) router-feature extraction from the CEARF index;
4) selected-router beta assignment plus RRF fusion.

Model training, semantic-teacher construction, index construction, profile
selection, router fitting, data loading, and warm-up are explicitly excluded.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import cearf
import loaders
import pasgr
from perquery_router import BucketedRouter, ContinuousRouter
from run_cearfn import tune_beta
from run_cearfn_evidence import build_memory_arrays, popularity_partition
from run_cearfn_v2 import (
    REPEAT_PROTOCOL_DOMAINS,
    build_features,
    build_features_from_memory_arrays,
    fuse_with_betas,
    load_pasgr_config,
    train_pasgr_v2,
)
from run_pasgr_full import semantic_matrix
from validation_protocol import hold_out_validation_targets


HERE = Path(__file__).resolve().parent
DEFAULT_DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")


def sync_device(model: pasgr.PASGRModel) -> None:
    device = next(model.parameters()).device
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed(start: float, model: pasgr.PASGRModel | None = None) -> float:
    if model is not None:
        sync_device(model)
    return time.perf_counter() - start


def load_or_train_model(
    checkpoint_path: Path,
    data: dict,
    sessions: dict,
    semantic: np.ndarray | None,
    seed: int,
    epochs: int,
    gated: dict,
) -> pasgr.PASGRModel:
    if checkpoint_path.exists():
        print(f"[INFER] loading {checkpoint_path.name}", flush=True)
        saved = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True)
        model_config = pasgr.PASGRConfig(**saved["config"])
        model = pasgr.PASGRModel(
            np.zeros((data["n_items"], model_config.dim), dtype=np.float32),
            model_config,
        )
        model.load_state_dict(saved["state_dict"])
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        return model.to(device).eval()

    print(f"[INFER] training {checkpoint_path.name} (excluded)", flush=True)
    model = train_pasgr_v2(
        data, sessions, semantic, seed, epochs, gated)
    torch.save({
        "config": asdict(model.config),
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
    }, checkpoint_path)
    return model


def fit_locked_router(
    selected_router: str,
    valid_queries: dict,
    valid_keys: list[str],
    valid_memory: np.ndarray,
    valid_neural: np.ndarray,
    valid_features: dict,
    short_context: int,
) -> dict | BucketedRouter | ContinuousRouter:
    memory_dict = {uid: list(valid_memory[row])
                   for row, uid in enumerate(valid_keys)}
    neural_dict = {uid: list(valid_neural[row])
                   for row, uid in enumerate(valid_keys)}
    fallback, _ = tune_beta(
        memory_dict, neural_dict, valid_queries, short_context)
    if selected_router == "regime":
        return fallback
    if selected_router == "bucketed":
        router = BucketedRouter(fallback_beta=fallback)
        router.fit(valid_queries, memory_dict, neural_dict,
                   valid_keys, valid_features)
        return router
    if selected_router == "continuous":
        router = ContinuousRouter(alpha=1.0)
        router.fit(valid_queries, memory_dict, neural_dict,
                   valid_keys, valid_features)
        return router
    raise ValueError(f"Unknown router family: {selected_router}")


def assign_betas(
    selected_router: str,
    fitted_router: dict | BucketedRouter | ContinuousRouter,
    keys: list[str],
    features: dict,
    short_context: int,
) -> dict[str, float]:
    if selected_router == "regime":
        return {
            uid: fitted_router[
                "short" if features[uid].length <= short_context else "long"
            ]
            for uid in keys
        }
    return {uid: fitted_router.beta_for(features[uid]) for uid in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--candidate-width", type=int, default=120)
    parser.add_argument("--config-file", type=Path,
                        default=HERE / "pasgr_config_per_domain.json")
    parser.add_argument("--canonical-results", type=Path,
                        default=HERE / "cearfn_v2_nested_results.json")
    parser.add_argument("--output", type=Path,
                        default=HERE / "cearfn_inference_benchmark.json")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=HERE / "inference_benchmark_checkpoints")
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    canonical = json.loads(args.canonical_results.read_text())
    result = json.loads(args.output.read_text()) if args.output.exists() else {
        "protocol": {
            "seed": args.seed,
            "warm_state": "trained PASGR model and built CEARF index",
            "pasgr_batch_size": 256,
            "warmup_queries_per_path": 512,
            "device_synchronization": (
                "Metal queue synchronized before and after PASGR timing"
            ),
            "included": [
                "CEARF memory retrieval to top-120",
                "PASGR full-catalogue prediction to top-120",
                "router-feature extraction reusing CEARF component ranks",
                "selected-router beta assignment and RRF fusion",
            ],
            "excluded": [
                "data loading", "semantic-teacher construction", "training",
                "index construction", "profile selection", "router fitting",
                "warm-up", "metric computation", "artifact writing",
            ],
            "repetitions": 1,
            "reason": (
                "Each timed pass covers 60k-151k queries; component times are "
                "reported separately and can be rerun from this script; no "
                "run-to-run dispersion is claimed."
            ),
            "hardware": (
                "Apple M2 Pro, 32 GB; CEARF single-process CPU; PASGR Metal"
            ),
        },
        "domains": {},
    }

    for domain in args.domains:
        if domain in result["domains"]:
            print(f"[INFER] {domain} already complete", flush=True)
            continue
        print(f"\n[INFER] === {domain} ===", flush=True)
        data = loaders.ALL_LOADERS[domain]()
        if len(data["valid_queries"]) > 5000:
            keys = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
            data["valid_queries"] = {key: data["valid_queries"][key] for key in keys}
        sessions = data["train_sessions"]
        tune_sessions = hold_out_validation_targets(
            sessions, data["valid_queries"])
        exclude_seen = domain not in REPEAT_PROTOCOL_DOMAINS
        config = cearf.CEARFConfig(exclude_seen=exclude_seen)

        # Preparation is intentionally outside all timers.
        tune_index = cearf.CEARFIndex(tune_sessions, data["n_items"], config)
        final_index = cearf.CEARFIndex(sessions, data["n_items"], config)
        profiles, _ = cearf.tune_profiles(tune_index, data["valid_queries"])
        freq = Counter(x for sequence in sessions.values() for x in sequence)
        head_items, _, _ = popularity_partition(freq, data["n_items"])
        head_set = set(head_items.tolist())
        valid_features = build_features(
            tune_index, data["valid_queries"], freq, head_set)
        semantic = semantic_matrix(domain, data)
        gated = load_pasgr_config(domain, args.config_file)
        validation_model = load_or_train_model(
            args.checkpoint_dir /
            f"{domain.lower()}_validation_seed{args.seed}.pt",
            data, tune_sessions, semantic, args.seed, args.epochs, gated)
        valid_keys, valid_neural = pasgr.predict_pasgr_array(
            validation_model, data["valid_queries"], data["n_items"],
            args.candidate_width, exclude_seen=exclude_seen)
        valid_memory_block = build_memory_arrays(
            tune_index, data["valid_queries"], profiles,
            args.candidate_width, f"{domain}-router-preparation")
        if [str(value) for value in valid_memory_block["keys"]] != valid_keys:
            raise RuntimeError(f"{domain}: validation query order mismatch")
        del validation_model
        gc.collect()

        model = load_or_train_model(
            args.checkpoint_dir / f"{domain.lower()}_final_seed{args.seed}.pt",
            data, sessions, semantic, args.seed, args.epochs, gated)

        test_keys = sorted(str(key) for key in data["test_queries"])
        warm_keys = test_keys[:min(512, len(test_keys))]
        warm_queries = {key: data["test_queries"][key] for key in warm_keys}

        # Warm both paths before timing.
        final_index.predict(warm_queries, profiles, topk=args.candidate_width)
        pasgr.predict_pasgr_array(
            model, warm_queries, data["n_items"], args.candidate_width,
            exclude_seen=exclude_seen)
        sync_device(model)

        print("[INFER] timing CEARF memory retrieval", flush=True)
        started = time.perf_counter()
        memory = build_memory_arrays(
            final_index, data["test_queries"], profiles,
            args.candidate_width, f"{domain}-inference-benchmark")
        memory_seconds = time.perf_counter() - started
        memory_keys = [str(value) for value in memory["keys"]]
        if memory_keys != test_keys:
            raise RuntimeError(f"{domain}: CEARF query order mismatch")

        print("[INFER] timing PASGR full-catalogue prediction", flush=True)
        sync_device(model)
        started = time.perf_counter()
        neural_keys, neural = pasgr.predict_pasgr_array(
            model, data["test_queries"], data["n_items"],
            args.candidate_width, exclude_seen=exclude_seen)
        neural_seconds = elapsed(started, model)
        if neural_keys != test_keys:
            raise RuntimeError(f"{domain}: PASGR query order mismatch")

        print("[INFER] timing router-feature extraction", flush=True)
        started = time.perf_counter()
        test_features = build_features_from_memory_arrays(
            data["test_queries"], test_keys, memory, freq, head_set)
        feature_seconds = time.perf_counter() - started

        run = next(
            item for item in canonical[domain]["runs"]
            if int(item["seed"]) == args.seed
        )
        selected_router = run["selected_router"]
        fitted_router = fit_locked_router(
            selected_router,
            {key: data["valid_queries"][key] for key in valid_keys},
            valid_keys,
            np.asarray(valid_memory_block["selected"], dtype=np.int32),
            valid_neural,
            valid_features,
            config.short_context,
        )
        print(f"[INFER] timing {selected_router} beta assignment", flush=True)
        started = time.perf_counter()
        test_betas = assign_betas(
            selected_router, fitted_router, test_keys, test_features,
            config.short_context)
        beta_seconds = time.perf_counter() - started

        # Warm fusion, then time the actual selected family over every query.
        prefix = warm_keys
        fuse_with_betas(
            memory["selected"][:len(prefix)], neural[:len(prefix)],
            prefix, {key: test_betas[key] for key in prefix})
        print("[INFER] timing RRF fusion", flush=True)
        started = time.perf_counter()
        fused = fuse_with_betas(
            memory["selected"], neural, test_keys, test_betas)
        fusion_seconds = time.perf_counter() - started
        if fused.shape != (len(test_keys), 20):
            raise RuntimeError(f"{domain}: invalid fused shape {fused.shape}")

        end_to_end = (
            memory_seconds + neural_seconds + feature_seconds
            + beta_seconds + fusion_seconds
        )
        n_queries = len(test_keys)
        result["domains"][domain] = {
            "n_queries": n_queries,
            "n_items": int(data["n_items"]),
            "selected_router": selected_router,
            "memory_seconds": memory_seconds,
            "neural_seconds": neural_seconds,
            "router_feature_seconds": feature_seconds,
            "router_beta_seconds": beta_seconds,
            "fusion_seconds": fusion_seconds,
            "end_to_end_seconds": end_to_end,
            "amortized_milliseconds_per_query": 1000.0 * end_to_end / n_queries,
            "queries_per_second": n_queries / end_to_end,
            "component_share": {
                "memory": memory_seconds / end_to_end,
                "neural": neural_seconds / end_to_end,
                "router_feature": feature_seconds / end_to_end,
                "router_beta": beta_seconds / end_to_end,
                "fusion": fusion_seconds / end_to_end,
            },
        }
        args.output.write_text(json.dumps(result, indent=2))
        print(
            f"[INFER] {domain}: {end_to_end:.2f}s, "
            f"{1000 * end_to_end / n_queries:.3f} amortized ms/query, "
            f"{n_queries / end_to_end:.1f} q/s",
            flush=True,
        )
        del model, memory, neural, fused
        gc.collect()

    print(f"\n[INFER] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
