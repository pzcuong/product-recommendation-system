#!/usr/bin/env python3
"""Full, memory-bounded multi-domain benchmark for PASGR.

Unlike the legacy orchestrator, every seed trains a distinct model and metrics
are streamed over the complete test catalog. Results are persisted after every
seed so a long run remains auditable if interrupted.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import loaders
import pasgr


HERE = Path(__file__).resolve().parent
DEFAULT_DOMAINS = ["Rental_visit", "RetailRocket", "Video_Games", "Baby_Products"]
DEFAULT_SEEDS = [42, 43, 44, 123, 456]


def semantic_matrix(domain: str, data: dict, dim: int = 128) -> np.ndarray | None:
    n_items = data["n_items"]
    artifact = HERE / "artifacts" / f"{domain}_pasgr_semantic.npy"
    legacy = HERE / "artifacts" / f"{domain}_e5_small.npy"
    if legacy.exists():
        value = np.load(legacy).astype(np.float32)
        if value.shape[0] == n_items:
            return value
    if artifact.exists():
        value = np.load(artifact).astype(np.float32)
        if value.shape[0] == n_items:
            return value
    texts = data.get("item_texts", {})
    if texts:
        ids = sorted(i for i in texts if 0 < int(i) < n_items)
        corpus = [str(texts[i]) for i in ids]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                     max_features=40000, sublinear_tf=True,
                                     stop_words="english")
        sparse = vectorizer.fit_transform(corpus)
        out_dim = min(dim, max(2, min(sparse.shape) - 1))
        encoded = TruncatedSVD(out_dim, random_state=42).fit_transform(sparse)
        encoded /= np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-8)
        matrix = np.zeros((n_items, out_dim), dtype=np.float32)
        matrix[ids] = encoded.astype(np.float32)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.save(artifact, matrix)
        print(f"  [semantic] built TFIDF-SVD metadata matrix {matrix.shape}", flush=True)
        return matrix
    categories = data.get("item_categories", {})
    if categories:
        matrix = np.zeros((n_items, dim), dtype=np.float32)
        cache: dict[str, np.ndarray] = {}
        for item, category in categories.items():
            key = str(category)
            if key not in cache:
                seed = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
                vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
                cache[key] = vector / max(float(np.linalg.norm(vector)), 1e-8)
            if 0 < int(item) < n_items:
                matrix[int(item)] = cache[key]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.save(artifact, matrix)
        print(f"  [semantic] built category matrix {matrix.shape}", flush=True)
        return matrix
    return None


def aggregate(seed_results: list[dict]) -> dict:
    metrics = sorted(seed_results[0]["test"]) if seed_results else []
    output = {}
    for metric in metrics:
        if metric == "n":
            output[metric] = seed_results[0]["test"][metric]
            continue
        values = np.asarray([run["test"][metric] for run in seed_results], dtype=float)
        output[metric] = {
            "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci95": float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=DEFAULT_DOMAINS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=HERE / "pasgr_full_results.json")
    args = parser.parse_args()
    results: dict = {}
    for domain in args.domains:
        started = time.time()
        data = loaders.ALL_LOADERS[domain]()
        sessions = data["train_sessions"]
        freq = Counter(x for sequence in sessions.values() for x in sequence)
        semantic = semantic_matrix(domain, data)
        base_config = pasgr.PASGRConfig(
            dim=64, prototypes=min(96, max(8, data["n_items"] // 250)),
            epochs=args.epochs, batch_size=512, hard_negatives=32,
            top_k=20, seed=42)
        print(f"\n[PASGR FULL] {domain}: train={len(sessions)} "
              f"valid={len(data.get('valid_queries', {}))} "
              f"test={len(data['test_queries'])} items={data['n_items']}", flush=True)
        prepared = pasgr.build_prototype_graph_embeddings(
            sessions, data["n_items"], freq, semantic, base_config)
        domain_runs: list[dict] = []
        for seed in args.seeds:
            config = pasgr.PASGRConfig(**{**base_config.__dict__, "seed": seed})
            seed_start = time.time()
            model = pasgr.train_pasgr(
                sessions, data["n_items"], freq, semantic, config,
                prepared_assets=prepared)
            validation = (pasgr.evaluate_pasgr(
                model, data["valid_queries"], data["n_items"])
                if data.get("valid_queries") else None)
            test = pasgr.evaluate_pasgr(model, data["test_queries"], data["n_items"])
            run = {"seed": seed, "validation": validation, "test": test,
                   "seconds": time.time() - seed_start}
            domain_runs.append(run)
            results[domain] = {"runs": domain_runs,
                               "aggregate": aggregate(domain_runs),
                               "seconds": time.time() - started}
            args.output.write_text(json.dumps(results, indent=2))
            print(f"  [seed={seed}] test={test} seconds={run['seconds']:.1f}", flush=True)
            del model
            gc.collect()
        results[domain]["seconds"] = time.time() - started
        args.output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
