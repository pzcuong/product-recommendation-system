#!/usr/bin/env python3
"""Mechanism, semantic, fusion-rule, and candidate-width ablations for CEARF-N."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import time

import numpy as np

import cearf
import loaders
import pasgr
from run_cearfn_evidence import (
    metrics_from_ranks, query_fingerprint, ranks_at_20, targets_for)
from run_pasgr_full import semantic_matrix


HERE = Path(__file__).resolve().parent
VARIANTS = {
    "full": {},
    "no_metadata": {"no_metadata": True},
    "no_graph": {"graph_weight": 0.0},
    "no_prototype_transport": {"prototype_transport": False},
    "no_contrastive_alignment": {"contrastive_weight": 0.0},
}
BETAS = tuple(round(step * .05, 2) for step in range(21))
RULES = ("rrf20", "rrf60", "borda", "inverse_rank")


def fuse_row(memory, neural, beta: float, rule: str, width: int = 120,
             topk: int = 20) -> list[int]:
    scores = defaultdict(float)
    # Treat negative-rank sentinel 0 as "out of width" so missing items never
    # contribute to either side of the fusion.
    for source_weight, ranking in ((1.0 - beta, memory[:width]),
                                   (beta, neural[:width])):
        if source_weight <= 0:
            continue
        for rank, item0 in enumerate(ranking, 1):
            item = int(item0)
            if item <= 0:
                continue
            if rule == "rrf20":
                value = 1.0 / (20.0 + rank)
            elif rule == "rrf60":
                value = 1.0 / (60.0 + rank)
            elif rule == "borda":
                value = (width - rank + 1.0) / width
            elif rule == "inverse_rank":
                # RRF with k=0: sharper weighting toward top ranks.
                value = 1.0 / rank
            else:
                raise KeyError(rule)
            scores[item] += source_weight * value
    return [item for item, _ in sorted(scores.items(),
                                        key=lambda pair: (-pair[1], pair[0]))[:topk]]


def fuse_matrix(memory: np.ndarray, neural: np.ndarray, beta: float,
                rule: str, width: int) -> np.ndarray:
    return np.asarray([fuse_row(m, n, beta, rule, width)
                       for m, n in zip(memory, neural)], dtype=np.int32)


def select_beta(memory: np.ndarray, neural: np.ndarray, targets: np.ndarray,
                rule: str, width: int) -> tuple[float, dict]:
    best = None
    for beta in BETAS:
        ranking = fuse_matrix(memory, neural, beta, rule, width)
        ranks = ranks_at_20(ranking, targets)
        metrics = metrics_from_ranks(ranks)
        utility = .5 * (metrics["recall@6"] + metrics["recall@20"])
        candidate = (utility, metrics["recall@20"], metrics["recall@6"],
                     -beta, beta, metrics)
        if best is None or candidate[:5] > best[:5]:
            best = candidate
    return best[4], {"beta": best[4], "utility": best[0], **best[5]}


def train_variant(data: dict, semantic: np.ndarray | None, seed: int,
                  epochs: int, changes: dict):
    sessions = data["train_sessions"]
    freq = Counter(x for sequence in sessions.values() for x in sequence)
    config_data = dict(
        dim=64, prototypes=min(96, max(8, data["n_items"] // 250)),
        epochs=epochs, batch_size=512, hard_negatives=32,
        top_k=120, seed=seed)
    no_metadata = bool(changes.get("no_metadata", False))
    config_data.update({key: value for key, value in changes.items()
                        if key != "no_metadata"})
    config = pasgr.PASGRConfig(**config_data)
    teacher = None if no_metadata else semantic
    assets = pasgr.build_prototype_graph_embeddings(
        sessions, data["n_items"], freq, teacher, config)
    return pasgr.train_pasgr(sessions, data["n_items"], freq, teacher,
                             config, prepared_assets=assets)


def load_memory(domain: str, split: str):
    path = HERE / "cearfn_evidence_artifacts" / f"{domain.lower()}_full_{split}_memory.npz"
    with np.load(path) as saved:
        return [str(x) for x in saved["keys"]], saved["selected"].copy(), \
            str(saved["fingerprint"].item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=["Video_Games", "Baby_Products"])
    parser.add_argument("--variants", nargs="*", choices=VARIANTS,
                        default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=HERE / "cearfn_ablation_results.json")
    parser.add_argument("--artifact-dir", type=Path,
                        default=HERE / "cearfn_ablation_artifacts")
    args = parser.parse_args()
    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    for domain in args.domains:
        data = loaders.ALL_LOADERS[domain]()
        valid_keys, memory_valid, valid_fp = load_memory(domain, "valid")
        test_keys, memory_test, test_fp = load_memory(domain, "test")
        validation = {uid: data["valid_queries"][uid] for uid in valid_keys}
        test = {uid: data["test_queries"][uid] for uid in test_keys}
        if query_fingerprint(validation) != valid_fp or query_fingerprint(test) != test_fp:
            raise RuntimeError(f"{domain}: evidence cache fingerprint mismatch")
        valid_targets = targets_for(valid_keys, validation)
        test_targets = targets_for(test_keys, test)
        semantic = semantic_matrix(domain, data)
        domain_result = results.setdefault(domain, {
            "protocol": {"seed": args.seed, "epochs": args.epochs,
                         "validation_queries": len(validation),
                         "test_queries": len(test), "selection_uses_test_labels": False},
            "variants": {}})
        for variant in args.variants:
            artifact = args.artifact_dir / f"{domain.lower()}_{variant}_seed{args.seed}_top120.npz"
            if variant in domain_result["variants"] and artifact.exists():
                print(f"[ABLATION] {domain} {variant} complete", flush=True)
                continue
            print(f"[ABLATION] START {domain} {variant}", flush=True)
            started = time.time()
            model = train_variant(data, semantic, args.seed, args.epochs,
                                  VARIANTS[variant])
            predicted_valid_keys, neural_valid = pasgr.predict_pasgr_array(
                model, validation, data["n_items"], 120)
            predicted_test_keys, neural_test = pasgr.predict_pasgr_array(
                model, test, data["n_items"], 120)
            if predicted_valid_keys != valid_keys or predicted_test_keys != test_keys:
                raise RuntimeError("prediction order mismatch")
            np.savez_compressed(artifact, valid=neural_valid, test=neural_test,
                                validation_fingerprint=np.asarray(valid_fp),
                                test_fingerprint=np.asarray(test_fp))
            rules = {}
            for rule in RULES:
                for width in (20, 50, 120):
                    beta, selection = select_beta(
                        memory_valid, neural_valid, valid_targets, rule, width)
                    fused = fuse_matrix(memory_test, neural_test, beta, rule, width)
                    ranks = ranks_at_20(fused, test_targets)
                    rules[f"{rule}_width{width}"] = {
                        "selection": selection,
                        "test": metrics_from_ranks(ranks),
                    }
            fixed = fuse_matrix(memory_test, neural_test, .5, "rrf20", 120)
            neural_ranks = ranks_at_20(neural_test, test_targets)
            # Memory-only (β=0) isolates the contribution of the neural residual.
            # It is the same regardless of fusion rule because only one source
            # contributes, so we compute it once.
            memory_ranks = ranks_at_20(memory_test, test_targets)
            domain_result["variants"][variant] = {
                "neural_only": metrics_from_ranks(neural_ranks),
                "memory_only": metrics_from_ranks(memory_ranks),
                "fusion_rules": rules,
                "fixed_beta_0.5_rrf20_width120": metrics_from_ranks(
                    ranks_at_20(fixed, test_targets)),
                "artifact": str(artifact), "seconds": time.time() - started,
            }
            results[domain] = domain_result
            args.output.write_text(json.dumps(results, indent=2))
            print(f"[ABLATION] DONE {domain} {variant} "
                  f"R20={rules['rrf20_width120']['test']['recall@20']:.6f}", flush=True)
            del model, neural_valid, neural_test
            gc.collect()
    args.output.write_text(json.dumps(results, indent=2))
    print(f"[ABLATION] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
