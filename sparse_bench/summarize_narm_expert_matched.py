#!/usr/bin/env python3
"""Reselect and aggregate the NARM expert family with the core CEARF utility."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DOMAINS = {
    "Video_Games": "video_games",
    "Baby_Products": "baby_products",
    "Diginetica_HID": "diginetica",
}


def selection_key(metrics: dict, name: str) -> tuple:
    return (
        .5 * (metrics["recall@6"] + metrics["recall@20"]),
        metrics["recall@20"],
        metrics["ndcg@20"],
        name,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent /
                        "narm_expert_matched_r6r20.json")
    args = parser.parse_args()

    result = {
        "selection_rule": "0.5*recall@6 + 0.5*recall@20",
        "selection_uses_test_labels": False,
        "candidate_family": {
            "experts": ["cearf", "stan", "vsknn", "narm"],
            "members": 15,
            "definition": "all non-empty subsets, including four singletons",
            "empty_set": "inadmissible because the system must emit a ranking",
        },
        "domains": {},
    }
    for domain, slug in DOMAINS.items():
        runs = []
        for seed in (42, 123, 456):
            path = args.root / f"narm_expert_{slug}_seed{seed}.json"
            source = json.loads(path.read_text())
            selected = max(
                source["validation"],
                key=lambda name: selection_key(
                    source["validation"][name], name))
            runs.append({
                "seed": seed,
                "selected_candidate": selected,
                "validation": source["validation"][selected],
                "test": source["test"][selected],
                "source": str(path),
            })

        metrics = {}
        for metric in ("recall@6", "recall@10", "recall@20", "ndcg@20"):
            values = np.asarray(
                [run["test"][metric] for run in runs], dtype=float)
            metrics[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "per_seed": {
                    str(run["seed"]): float(run["test"][metric])
                    for run in runs
                },
            }
        result["domains"][domain] = {
            "runs": runs,
            "aggregate": metrics,
        }

    args.output.write_text(json.dumps(result, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
