#!/usr/bin/env python3
"""Efficiency profiling of CEARF-N at inference time on the matched protocol.

Reports: memory-index size, neural-residual parameter count, build cost
(offline, amortised over queries), and per-query latency for memory scoring,
neural scoring and fusion. All measurements share the test queries that the
baselines use, so numbers are directly comparable to the baseline table in
PAPER_ADMA.md.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import cearf
import loaders
import pasgr
from run_cearfn_evidence import query_fingerprint
from run_pasgr_full import semantic_matrix

HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products")
SEED = 42
EPOCHS = 4


def mps_memory() -> int:
    return int(torch.mps.current_allocated_memory()) if torch.backends.mps.is_available() else 0


def count_pasgr_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_memory_scoring(index: cearf.CEARFIndex, queries: dict, profiles: dict,
                           sample_size: int = 500) -> tuple[float, float, int]:
    keys = sorted(queries)
    samples = keys[: min(len(keys), sample_size)]
    started = time.time()
    for uid in samples:
        context = queries[uid].get("context", ())
        components = index.component_rankings(context)
        regime = "short" if len(context) <= index.config.short_context else "long"
        index.fuse_rankings(context, components, profiles[regime], 120)
    seconds = time.time() - started
    return seconds, 1000.0 * seconds / len(samples), len(samples)


def measure_neural_latency(model, queries: dict, n_items: int,
                           width: int = 120, sample_size: int = 500) -> tuple[float, int]:
    keys = sorted(queries)
    samples = keys[: min(len(keys), sample_size)]
    started = time.time()
    pasgr.predict_pasgr_array(model, {uid: queries[uid] for uid in samples},
                              n_items, width)
    seconds = time.time() - started
    return 1000.0 * seconds / len(samples), len(samples)


def main() -> None:
    out: dict = {"domains": {}}
    for domain in DOMAINS:
        print(f"\n=== {domain} ===", flush=True)
        data = loaders.ALL_LOADERS[domain]()
        n_items = data["n_items"]
        sessions = data["train_sessions"]
        test = data["test_queries"]

        # 1) Offline memory-index build.
        gc.collect()
        t0 = time.time()
        config = cearf.CEARFConfig()
        index = cearf.CEARFIndex(sessions, n_items, config)
        build_seconds = time.time() - t0
        transition_entries = sum(len(v) for v in index.transition.values())
        posting_entries = sum(len(v) for v in index.postings.values())
        print(f"  memory index build      = {build_seconds:.2f}s", flush=True)
        print(f"  transition entries      = {transition_entries}", flush=True)
        print(f"  session postings        = {posting_entries}", flush=True)

        # Reuse validation-selected profiles from the locked evidence run.
        with open(HERE / "cearfn_evidence_results.json") as f:
            evidence = json.load(f)
        ds_block = evidence[domain]
        selected_names = ds_block["selected_profile_names"]
        profiles = {regime: cearf.PROFILES[name]
                    for regime, name in selected_names.items()}

        # 2) Memory scoring + fusion latency per query.
        mem_s, mem_ms, n_mem = measure_memory_scoring(index, test, profiles)
        print(f"  memory score+fusion     = {mem_ms:.4f}ms/query "
              f"(n={n_mem}, total {mem_s:.2f}s)", flush=True)

        # 3) Neural residual parameters + inference latency.
        semantic = semantic_matrix(domain, data)
        pcfg = pasgr.PASGRConfig(
            dim=64, prototypes=min(96, max(8, n_items // 250)),
            epochs=EPOCHS, batch_size=512, hard_negatives=32,
            top_k=120, seed=SEED)
        gc.collect()
        train_peak0 = mps_memory()
        assets = pasgr.build_prototype_graph_embeddings(
            sessions, n_items, index.freq, semantic, pcfg)
        model = pasgr.train_pasgr(sessions, n_items, index.freq, semantic, pcfg,
                                  prepared_assets=assets)
        train_peak = mps_memory() - train_peak0
        n_params = count_pasgr_params(model)
        neural_ms, n_neural = measure_neural_latency(model, test, n_items, 120)
        print(f"  neural params           = {n_params}", flush=True)
        print(f"  neural inference        = {neural_ms:.4f}ms/query (n={n_neural})",
              flush=True)

        out["domains"][domain] = {
            "n_items": int(n_items),
            "test_queries": int(len(test)),
            "train_sessions": int(len(sessions)),
            "transition_memory_entries": int(transition_entries),
            "session_posting_entries": int(posting_entries),
            "offline_memory_index_build_seconds": float(build_seconds),
            "memory_score_plus_fusion_ms_per_query": float(mem_ms),
            "neural_residual_parameters": int(n_params),
            "neural_residual_ms_per_query": float(neural_ms),
            "neural_residual_train_peak_delta_bytes": int(max(train_peak, 0)),
            "end_to_end_ms_per_query": float(mem_ms + neural_ms),
            "short_context_threshold": int(config.short_context),
            "seed": SEED,
            "epochs": EPOCHS,
        }

        del model
        gc.collect()

    output_path = HERE / "cearfn_efficiency.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
