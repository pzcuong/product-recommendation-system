#!/usr/bin/env python3
"""Summarize the locked v2 gated constituent ablation over matched seeds."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (42, 123, 456)
DOMAINS = ("Video_Games", "Baby_Products", "Arts_Crafts_and_Sewing", "Diginetica_HID")


def aggregate(runs, key):
    metrics = ("recall@6", "recall@10", "recall@20", "ndcg@20")
    out = {}
    for metric in metrics:
        values = np.asarray([run[key][metric] for run in runs], dtype=float)
        out[metric] = {"mean": float(values.mean()),
                       "std": float(values.std(ddof=1)),
                       "per_seed": {str(run["seed"]): float(run[key][metric])
                                    for run in runs}}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=HERE / "cearfn_v2_results.json")
    parser.add_argument("--output", type=Path,
                        default=HERE / "cearfn_v2_constituent_summary.json")
    parser.add_argument("--domains", nargs="*", default=list(DOMAINS))
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    output = {"matched_seeds": list(SEEDS), "domains": {}}
    for domain in args.domains:
        runs = sorted(source[domain]["runs"], key=lambda row: int(row["seed"]))
        found = tuple(int(row["seed"]) for row in runs)
        if found != SEEDS:
            raise RuntimeError(f"{domain}: expected seeds {SEEDS}, found {found}")
        selected_names = {row["selected_router"] for row in runs}
        output["domains"][domain] = {
            "pasgr_config": source[domain]["pasgr_config"],
            "selected_routers": sorted(selected_names),
            "memory_only": aggregate(runs, "memory_only"),
            "neural_only": aggregate(runs, "neural_only"),
            "selected_fusion": aggregate(runs, "selected"),
            "runs": runs,
        }
    path = args.output
    path.write_text(json.dumps(output, indent=2))
    print(path)


if __name__ == "__main__":
    main()
