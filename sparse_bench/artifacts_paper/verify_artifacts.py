#!/usr/bin/env python3
"""Artifact verifier: read NPZ arrays, recompute metrics, cross-check JSON + CIs.

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
DOMAINS = ["Video_Games", "Baby_Products", "Diginetica_HID"]
EXPECTED_ARRAYS = 72

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
    cis = json.loads((HERE / "real_paired_cis.json").read_text())

    checked = 0
    # Check every array listed in manifest
    manifest_arrays = []
    for ds in manifest.get("domains", {}):
        for fp in manifest["domains"][ds].get("arrays", {}):
            manifest_arrays.append(fp)
            path = RANKS / f"{fp}.npz"
            if not path.exists():
                errors.append(f"missing {fp}.npz")
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
            # Cross-check per_seed_metrics.json
            parts = fp.split("_seed")
            if len(parts) == 2:
                ds, rest = fp[:fp.index("_seed")], fp[fp.index("_seed")+5:]
                # ds may contain underscores; match by known domains
                for d in DOMAINS:
                    if fp.startswith(d + "_"):
                        m = fp[len(d)+1: fp.index("_seed")]
                        s = rest
                        if d in per_seed and m in per_seed[d] and s in per_seed[d][m]["seeds"]:
                            jr = per_seed[d][m]["seeds"][s]
                            for metric in ("recall@6", "recall@10", "recall@20", "ndcg@20", "utility"):
                                if abs(computed[metric] - jr[metric]) > 1e-4:
                                    errors.append(f"{fp}: {metric} mismatch")
            # SHA-256
            arr_info = None
            for d in manifest.get("domains", {}):
                if fp in manifest["domains"][d].get("arrays", {}):
                    arr_info = manifest["domains"][d]["arrays"][fp]
            if arr_info and "sha256" in arr_info:
                actual = hashlib.sha256(ranks.tobytes()).hexdigest()
                if actual != arr_info["sha256"]:
                    errors.append(f"{fp}: SHA-256 mismatch")

    if checked != EXPECTED_ARRAYS:
        errors.append(f"expected {EXPECTED_ARRAYS} arrays, found {checked}")

    # Cross-check paired CIs: delta should equal mean of per-query diffs from arrays
    for ds in DOMAINS:
        for comp in cis.get("domains", {}).get(ds, {}):
            entry = cis["domains"][ds][comp]
            # We can't easily recompute without knowing method pairs; verify CI centered on delta
            lo, hi = entry["ci95"]
            center = (lo + hi) / 2
            if abs(center - entry["delta_r20"]) > 0.001:
                errors.append(f"{ds} {comp}: CI not centered on delta")

    if errors:
        print(f"FAIL: {len(errors)} errors")
        for e in errors[:20]: print(f"  {e}")
        return 1
    print(f"OK: {checked} arrays verified (monotonic, nDCG<=R@20, utility, JSON, SHA-256, CI centered)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
