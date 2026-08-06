#!/usr/bin/env python3
"""Query-clustered paired tests against deterministic external baselines."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import numpy as np
from paired_statistics import cluster_paired_recall

HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")
SEEDS = (42, 123, 456)
METHODS = ("vsknn", "stan", "transition")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cearfn-artifact-dir", type=Path,
                        default=HERE / "cearfn_v2_artifacts")
    parser.add_argument("--baseline-artifact-dir", type=Path,
                        default=HERE / "neighborhood_baseline_artifacts")
    parser.add_argument("--output", type=Path,
                        default=HERE / "external_baseline_paired.json")
    parser.add_argument("--domains", nargs="*", default=list(DOMAINS))
    args = parser.parse_args()
    out = {"unit": "test query", "matched_seeds": list(SEEDS), "domains": {}}
    for domain in args.domains:
        cearfn = []
        for seed in SEEDS:
            path = args.cearfn_artifact_dir / f"{domain.lower()}_v2_seed{seed}_ranks.npz"
            with np.load(path) as z:
                if "selected_rank" not in z.files:
                    raise RuntimeError(f"{path}: missing selected_rank; rerun v2")
                cearfn.append(z["selected_rank"].astype(np.int32))
        out["domains"][domain] = {}
        for method in METHODS:
            path = args.baseline_artifact_dir / \
                f"{domain.lower()}_{method}_ranks.npz"
            with np.load(path) as z:
                baseline = z["ranks"].astype(np.int32)
            out["domains"][domain][method] = cluster_paired_recall(
                cearfn, np.repeat(baseline[None, :], len(SEEDS), axis=0))
    path = args.output
    path.write_text(json.dumps(out, indent=2))
    print(path)


if __name__ == "__main__":
    main()
