#!/usr/bin/env python3
"""CEARF-N v2 without metadata — fair comparison against ID-only baselines.

Trains PASGR with semantic=None (random init), which removes all metadata
signal. Runs the same protocol as run_cearfn_v2.py for the Amazon domains.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import cearf
import loaders
import pasgr
from perquery_router import extract_features
from run_cearfn import fuse, tune_beta
from run_cearfn_evidence import (
    load_or_build_memory, metrics_from_ranks, query_fingerprint,
    ranks_at_20, targets_for)
from validation_protocol import hold_out_validation_targets

HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")
SEEDS = (42, 123, 456)
EPOCHS = 4
CANDIDATE_WIDTH = 120
REPEAT_PROTOCOL_DOMAINS = frozenset({"Diginetica_HID", "Tmall"})


def load_pasgr_config(domain: str) -> dict:
    path = HERE / "pasgr_config_per_domain.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    sel = data.get(domain, {}).get("selected", {}).get("combination", {})
    return sel or {}


def train_pasgr_v2(data, sessions, seed, epochs, gated_config):
    freq = Counter(x for seq in sessions.values() for x in seq)
    config_data = dict(
        dim=64, prototypes=min(96, max(8, data["n_items"] // 250)),
        epochs=epochs, batch_size=512, hard_negatives=32,
        top_k=120, seed=seed)
    config_data.update(gated_config)
    config = pasgr.PASGRConfig(**config_data)
    # No metadata: semantic=None → random init
    assets = pasgr.build_prototype_graph_embeddings(
        sessions, data["n_items"], freq, None, config)
    return pasgr.train_pasgr(sessions, data["n_items"], freq, None,
                             config, prepared_assets=assets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--candidate-width", type=int, default=CANDIDATE_WIDTH)
    parser.add_argument("--output", type=Path,
        default=HERE / "cearfn_v2_nometa_nested_results.json")
    parser.add_argument("--artifact-dir", type=Path,
                        default=HERE / "cearfn_v2_nometa_artifacts")
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    for domain in args.domains:
        if domain in results:
            print(f"[V2-NOMETA] {domain} already complete", flush=True)
            continue
        domain_started = time.time()
        data = loaders.ALL_LOADERS[domain]()
        if len(data["valid_queries"]) > 5000:
            _vk = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
            data["valid_queries"] = {k: data["valid_queries"][k] for k in _vk}
        sessions = data["train_sessions"]
        tune_sessions = hold_out_validation_targets(
            sessions, data["valid_queries"])
        freq = Counter(x for seq in sessions.values() for x in seq)
        exclude_seen = domain not in REPEAT_PROTOCOL_DOMAINS
        config = cearf.CEARFConfig(exclude_seen=exclude_seen)
        tune_index = cearf.CEARFIndex(tune_sessions, data["n_items"], config)
        final_index = cearf.CEARFIndex(sessions, data["n_items"], config)
        profiles, _ = cearf.tune_profiles(tune_index, data["valid_queries"])
        valid_memory = load_or_build_memory(
            args.artifact_dir / f"{domain.lower()}_nested_valid_memory.npz",
            tune_index, data["valid_queries"], profiles, args.candidate_width,
            f"{domain}-nometa-nested-valid")
        test_memory = load_or_build_memory(
            args.artifact_dir / f"{domain.lower()}_nested_test_memory.npz",
            final_index, data["test_queries"], profiles, args.candidate_width,
            f"{domain}-nometa-test")

        valid_keys = [str(x) for x in valid_memory["keys"]]
        test_keys = [str(x) for x in test_memory["keys"]]
        test_targets = targets_for(test_keys, data["test_queries"])

        gated_config = load_pasgr_config(domain)
        domain_block = {"runs": [], "pasgr_config": gated_config}
        completed = {int(r["seed"]) for r in domain_block["runs"]}

        for seed in args.seeds:
            if seed in completed:
                continue
            print(f"\n[V2-NOMETA] {domain} seed={seed}", flush=True)
            seed_started = time.time()
            validation_model = train_pasgr_v2(
                data, tune_sessions, seed, args.epochs, gated_config)
            valid_memory_dict = {uid: list(valid_memory["selected"][i])
                                 for i, uid in enumerate(valid_keys)}
            _, neural_valid = pasgr.predict_pasgr_array(
                validation_model, data["valid_queries"], data["n_items"],
                args.candidate_width, exclude_seen=exclude_seen)
            del validation_model
            gc.collect()
            final_model = train_pasgr_v2(
                data, sessions, seed, args.epochs, gated_config)
            _, neural_test = pasgr.predict_pasgr_array(
                final_model, data["test_queries"], data["n_items"],
                args.candidate_width, exclude_seen=exclude_seen)
            valid_neural_dict = {uid: list(neural_valid[i])
                                 for i, uid in enumerate(valid_keys)}
            betas_regime, _ = tune_beta(valid_memory_dict, valid_neural_dict,
                                        data["valid_queries"], config.short_context)
            regime_per_uid = {
                uid: betas_regime["short" if len(data["test_queries"][uid]
                                                   .get("context", ())) <=
                                        config.short_context else "long"]
                for uid in test_keys}
            fused = np.empty((len(test_keys), 20), dtype=np.int32)
            for row, uid in enumerate(test_keys):
                fused[row] = fuse(test_memory["selected"][row],
                                  neural_test[row], regime_per_uid[uid])
            # Save rank artifact for paired bootstrap
            seed_artifact = args.artifact_dir / f"{domain.lower()}_nometa_seed{seed}_ranks.npz"
            np.savez_compressed(seed_artifact,
                                regime_rank=ranks_at_20(fused, test_targets).astype(np.uint8),
                                test_fingerprint=np.asarray(query_fingerprint(data["test_queries"])))
            domain_block["runs"].append({
                "seed": seed,
                "regime": metrics_from_ranks(ranks_at_20(fused, test_targets)),
                "rank_artifact": str(seed_artifact),
                "seconds": time.time() - seed_started,
            })
            r20 = domain_block["runs"][-1]["regime"]["recall@20"]
            print(f"[V2-NOMETA] DONE {domain} seed={seed} R@20={r20:.5f}", flush=True)
            del final_model, neural_valid, neural_test
            gc.collect()

        domain_block["seconds_total"] = time.time() - domain_started
        results[domain] = domain_block
        args.output.write_text(json.dumps(results, indent=2))

    print(f"\n[V2-NOMETA] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
