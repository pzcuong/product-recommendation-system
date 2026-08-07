#!/usr/bin/env python3
"""Minimal artifact verifier: read NPZ arrays, recompute metrics, check JSON.

Usage:
    cd sparse_bench/artifacts_paper
    python verify_artifacts.py

Exits 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RANKS = HERE / "per_query_ranks"
DOMAINS = ["Video_Games", "Baby_Products", "Diginetica"]
SEEDS = [42, 123, 456]
METHODS = [
    "uniform", "kmeans", "feature_bucketed", "equal_mixing",
    "query_cond", "narm_id", "narm_tfidf", "narm_minilm", "cearfn_minilm",
    "oof_global",
]
EXPECTED_ARRAYS = 90

def metrics_from_ranks(ranks: np.ndarray) -> dict:
    hits = ranks > 0
    r6 = float(np.mean(hits & (ranks <= 6)))
    r10 = float(np.mean(hits & (ranks <= 10)))
    r20 = float(np.mean(hits))
    gains = np.where(hits & (ranks <= 20),
                     1.0 / np.log2(ranks.clip(1).astype(np.float64) + 1.0), 0.0)
    ndcg = float(np.mean(gains))
    u = 0.5 * r6 + 0.5 * r20
    return {"recall@6": r6, "recall@10": r10, "recall@20": r20,
            "ndcg@20": ndcg, "utility": u}

def main() -> int:
    errors = []
    per_seed = json.loads((HERE / "per_seed_metrics.json").read_text())
    manifest = json.loads((HERE / "manifest.json").read_text())
    checked = 0
    for ds in DOMAINS:
        for m in METHODS:
            for s in SEEDS:
                fp = f"{ds}_{m}_seed{s}"
                path = RANKS / f"{fp}.npz"
                if not path.exists():
                    continue
                with np.load(path) as data:
                    ranks = data["ranks"]
                checked += 1
                if ranks.ndim != 1: errors.append(f"{fp}: not 1D")
                if ranks.dtype != np.uint8: errors.append(f"{fp}: dtype != uint8")
                if ranks.min() < 0 or ranks.max() > 20:
                    errors.append(f"{fp}: values out of [0,20]")
                computed = metrics_from_ranks(ranks)
                if not (computed["recall@6"] <= computed["recall@10"] + 1e-9
                        <= computed["recall@20"] + 1e-9):
                    errors.append(f"{fp}: monotonicity violated")
                if computed["ndcg@20"] > computed["recall@20"] + 1e-9:
                    errors.append(f"{fp}: nDCG > R@20")
                expected_u = 0.5 * computed["recall@6"] + 0.5 * computed["recall@20"]
                if abs(computed["utility"] - expected_u) > 1e-9:
                    errors.append(f"{fp}: utility formula mismatch")
                if ds in per_seed and m in per_seed[ds]:
                    json_r = per_seed[ds][m]["seeds"].get(str(s))
                    if json_r:
                        for metric in ("recall@6", "recall@10", "recall@20", "ndcg@20", "utility"):
                            if abs(computed[metric] - json_r[metric]) > 1e-4:
                                errors.append(f"{fp}: {metric} mismatch")
                if ds in manifest.get("domains", {}):
                    arr_info = manifest["domains"][ds].get("arrays", {}).get(fp)
                    if arr_info and "sha256" in arr_info:
                        actual = hashlib.sha256(ranks.tobytes()).hexdigest()
                        if actual != arr_info["sha256"]:
                            errors.append(f"{fp}: SHA-256 mismatch")
    if checked != EXPECTED_ARRAYS:
        errors.append(f"expected {EXPECTED_ARRAYS} arrays, found {checked}")
    if errors:
        print(f"FAIL: {len(errors)} errors")
        for e in errors[:20]: print(f"  {e}")
        return 1
    print(f"OK: {checked} arrays verified (monotonic, nDCG<=R@20, utility, JSON, SHA-256)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
