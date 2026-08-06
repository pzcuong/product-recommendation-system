#!/usr/bin/env python3
"""One-axis sensitivity sweep over the PASGR hyperparameters.

For each (prototype_temperature, hard_negatives, prototypes) we retrain PASGR
once (seed 42, 4 epochs) and report the fused-test Recall@20 with the default
validation-selected β. The sweep is one-axis-at-a-time so the total number of
runs stays bounded (3+3+3 = 9 cells per domain) and the resulting table is
readable as a sensitivity study rather than a search log.

Selection is validation-only; the test split is evaluated exactly once per
cell after β has been chosen on validation.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import cearf
import loaders
import pasgr
from run_cearfn_ablation import fuse_matrix, select_beta
from run_cearfn_evidence import (
    load_or_build_memory, metrics_from_ranks, ranks_at_20, targets_for)
from run_pasgr_full import semantic_matrix

HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID", "Tmall")
SEED = 42
EPOCHS = 4

TEMPS = (0.05, 0.10, 0.20)
HARD_NEGS = (16, 32, 64)
PROTOTYPES = (32, 64, 96)


def train_with(data, semantic, seed, epochs, *, prototype_temperature,
               hard_negatives, prototypes):
    sessions = data["train_sessions"]
    freq = Counter(x for sequence in sessions.values() for x in sequence)
    config = pasgr.PASGRConfig(
        dim=64, prototypes=min(prototypes, max(8, data["n_items"] // 250)),
        epochs=epochs, batch_size=512, hard_negatives=hard_negatives,
        top_k=120, seed=seed, prototype_temperature=prototype_temperature)
    assets = pasgr.build_prototype_graph_embeddings(
        sessions, data["n_items"], freq, semantic, config)
    return pasgr.train_pasgr(sessions, data["n_items"], freq, semantic,
                             config, prepared_assets=assets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--candidate-width", type=int, default=120)
    parser.add_argument("--output", type=Path,
                        default=HERE / "pasgr_hp_sensitivity.json")
    args = parser.parse_args()

    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    for domain in args.domains:
        if domain in results:
            print(f"[HP] {domain} already complete", flush=True)
            continue
        print(f"\n[HP] === {domain} ===", flush=True)
        data = loaders.ALL_LOADERS[domain]()
        # Match the evidence protocol's deterministic validation cap so the
        # locked Amazon memory cache can be reused (fingerprint is cap-sensitive).
        if len(data["valid_queries"]) > 5000:
            _vk = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
            data["valid_queries"] = {k: data["valid_queries"][k] for k in _vk}
        sessions = data["train_sessions"]
        freq = Counter(x for seq in sessions.values() for x in seq)
        config = cearf.CEARFConfig()
        index = cearf.CEARFIndex(sessions, data["n_items"], config)
        profiles, _ = cearf.tune_profiles(index, data["valid_queries"])
        # Reuse the locked evidence cache when present; build under hp_artifacts otherwise.
        evidence_dir = HERE / "cearfn_evidence_artifacts"
        valid_path = evidence_dir / f"{domain.lower()}_full_valid_memory.npz"
        test_path = evidence_dir / f"{domain.lower()}_full_test_memory.npz"
        if valid_path.exists() and test_path.exists():
            import numpy as _np
            valid_fp = query_fingerprint(data["valid_queries"])
            test_fp = query_fingerprint(data["test_queries"])
            with _np.load(valid_path) as sv:
                if str(sv["fingerprint"].item()) != valid_fp:
                    raise RuntimeError(f"{domain}: valid cache fingerprint mismatch")
                valid_memory = {k: sv[k] for k in sv.files}
            with _np.load(test_path) as st:
                if str(st["fingerprint"].item()) != test_fp:
                    raise RuntimeError(f"{domain}: test cache fingerprint mismatch")
                test_memory = {k: st[k] for k in st.files}
            print(f"[HP] {domain} reused evidence cache", flush=True)
        else:
            valid_memory = load_or_build_memory(
                HERE / f"hp_artifacts/{domain.lower()}_valid_memory.npz", index,
                data["valid_queries"], profiles, args.candidate_width, f"{domain}-valid")
            test_memory = load_or_build_memory(
                HERE / f"hp_artifacts/{domain.lower()}_test_memory.npz", index,
                data["test_queries"], profiles, args.candidate_width, f"{domain}-test")
        valid_keys = [str(x) for x in valid_memory["keys"]]
        test_keys = [str(x) for x in test_memory["keys"]]
        valid_targets = targets_for(valid_keys, data["valid_queries"])
        test_targets = targets_for(test_keys, data["test_queries"])
        semantic = semantic_matrix(domain, data)

        domain_result = {"prototype_temperature": {}, "hard_negatives": {},
                         "prototypes": {}}
        # Defaults: temperature 0.10, hard_negatives 32, prototypes 96.
        for temp in TEMPS:
            started = time.time()
            model = train_with(data, semantic, args.seed, args.epochs,
                               prototype_temperature=temp, hard_negatives=32,
                               prototypes=96)
            _, nv = pasgr.predict_pasgr_array(
                model, data["valid_queries"], data["n_items"], args.candidate_width)
            _, nt = pasgr.predict_pasgr_array(
                model, data["test_queries"], data["n_items"], args.candidate_width)
            beta, sel = select_beta(valid_memory["selected"], nv, valid_targets,
                                    "rrf20", args.candidate_width)
            mt = metrics_from_ranks(ranks_at_20(
                fuse_matrix(test_memory["selected"], nt, beta, "rrf20", args.candidate_width),
                test_targets))
            domain_result["prototype_temperature"][str(temp)] = {
                "validation": sel, "test": mt, "seconds": time.time() - started}
            print(f"[HP] {domain} temp={temp} R@20={mt['recall@20']:.5f}", flush=True)
            del model, nv, nt
            gc.collect()
            results[domain] = domain_result
            args.output.write_text(json.dumps(results, indent=2))

        for hn in HARD_NEGS:
            started = time.time()
            model = train_with(data, semantic, args.seed, args.epochs,
                               prototype_temperature=0.10, hard_negatives=hn,
                               prototypes=96)
            _, nv = pasgr.predict_pasgr_array(
                model, data["valid_queries"], data["n_items"], args.candidate_width)
            _, nt = pasgr.predict_pasgr_array(
                model, data["test_queries"], data["n_items"], args.candidate_width)
            beta, sel = select_beta(valid_memory["selected"], nv, valid_targets,
                                    "rrf20", args.candidate_width)
            mt = metrics_from_ranks(ranks_at_20(
                fuse_matrix(test_memory["selected"], nt, beta, "rrf20", args.candidate_width),
                test_targets))
            domain_result["hard_negatives"][str(hn)] = {
                "validation": sel, "test": mt, "seconds": time.time() - started}
            print(f"[HP] {domain} hard_neg={hn} R@20={mt['recall@20']:.5f}", flush=True)
            del model, nv, nt
            gc.collect()
            results[domain] = domain_result
            args.output.write_text(json.dumps(results, indent=2))

        for pr in PROTOTYPES:
            started = time.time()
            model = train_with(data, semantic, args.seed, args.epochs,
                               prototype_temperature=0.10, hard_negatives=32,
                               prototypes=pr)
            _, nv = pasgr.predict_pasgr_array(
                model, data["valid_queries"], data["n_items"], args.candidate_width)
            _, nt = pasgr.predict_pasgr_array(
                model, data["test_queries"], data["n_items"], args.candidate_width)
            beta, sel = select_beta(valid_memory["selected"], nv, valid_targets,
                                    "rrf20", args.candidate_width)
            mt = metrics_from_ranks(ranks_at_20(
                fuse_matrix(test_memory["selected"], nt, beta, "rrf20", args.candidate_width),
                test_targets))
            domain_result["prototypes"][str(pr)] = {
                "validation": sel, "test": mt, "seconds": time.time() - started}
            print(f"[HP] {domain} prototypes={pr} R@20={mt['recall@20']:.5f}", flush=True)
            del model, nv, nt
            gc.collect()
            results[domain] = domain_result
            args.output.write_text(json.dumps(results, indent=2))

        print(f"[HP] {domain} complete", flush=True)

    print(f"\n[HP] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
