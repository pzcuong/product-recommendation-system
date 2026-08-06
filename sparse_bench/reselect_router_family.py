#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")


def criterion_value(rule: str, metrics: dict) -> tuple[float, ...]:
    r6 = float(metrics["recall@6"])
    r10 = float(metrics["recall@10"])
    r20 = float(metrics["recall@20"])
    ndcg20 = float(metrics["ndcg@20"])
    if rule == "r6_r20":
        return (0.5 * (r6 + r20), r20, ndcg20)
    if rule == "r10_r20":
        return (0.5 * (r10 + r20), r20, ndcg20)
    if rule == "r20":
        return (r20, ndcg20)
    if rule == "r6_r10_r20":
        return (0.25 * r6 + 0.25 * r10 + 0.5 * r20, r20, ndcg20)
    raise ValueError(rule)


def metrics_from_ranks(ranks: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(ranks, dtype=np.int32)
    out = {"n": int(len(values))}
    for k in (6, 10, 20):
        hit = (values > 0) & (values <= k)
        gain = np.zeros(len(values), dtype=np.float64)
        gain[hit] = 1.0 / np.log2(values[hit].astype(np.float64) + 1.0)
        out[f"recall@{k}"] = float(hit.mean()) if len(values) else 0.0
        out[f"ndcg@{k}"] = float(gain.mean()) if len(values) else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=("r6_r20", "r10_r20", "r20", "r6_r10_r20"),
                        default="r10_r20")
    parser.add_argument("--results", type=Path, default=HERE / "cearfn_v2_nested_results.json")
    parser.add_argument("--artifact-dir", type=Path, default=HERE / "cearfn_v2_nested_artifacts")
    parser.add_argument("--output", type=Path, default=HERE / "router_family_reselected.json")
    args = parser.parse_args()

    source = json.loads(args.results.read_text())
    out = {"rule": args.rule, "domains": {}}
    for domain in DOMAINS:
        domain_rows = []
        for run in source[domain]["runs"]:
            variants = run["router_selection"]["variants"]
            selected = max(
                variants,
                key=lambda name: criterion_value(args.rule, variants[name]) + (name,)
            )
            seed = int(run["seed"])
            artifact = np.load(args.artifact_dir / f"{domain.lower()}_v2_seed{seed}_ranks.npz", allow_pickle=True)
            rank_field = {
                "regime": "regime_rank",
                "bucketed": "bucketed_rank",
                "continuous": "continuous_rank",
            }[selected]
            test_metrics = metrics_from_ranks(artifact[rank_field].astype(np.int32))
            domain_rows.append({
                "seed": seed,
                "old_selected_router": run["selected_router"],
                "new_selected_router": selected,
                "old_selected_test": run["selected"],
                "new_selected_test": test_metrics,
                "regime_test": run["regime"],
                "bucketed_test": run["bucketed"],
                "continuous_test": run["continuous"],
                "selection_validation_variants": variants,
            })

        def avg(metric_key: str, block: str) -> float:
            return float(np.mean([row[block][metric_key] for row in domain_rows]))

        out["domains"][domain] = {
            "runs": domain_rows,
            "mean_selected_before": {
                "recall@20": avg("recall@20", "old_selected_test"),
                "recall@10": avg("recall@10", "old_selected_test"),
                "ndcg@20": avg("ndcg@20", "old_selected_test"),
            },
            "mean_selected_after": {
                "recall@20": avg("recall@20", "new_selected_test"),
                "recall@10": avg("recall@10", "new_selected_test"),
                "ndcg@20": avg("ndcg@20", "new_selected_test"),
            },
        }

    args.output.write_text(json.dumps(out, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
