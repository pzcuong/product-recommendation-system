#!/usr/bin/env python3
"""Replace refit test ranks with validation-selected checkpoint ranks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import loaders
from paper_models import build_model
from run_cearfn_evidence import metrics_from_ranks, query_fingerprint, ranks_at_20, targets_for
from run_paper_baselines import aggregate, predict_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    results = json.loads(args.output.read_text())
    domain = "Diginetica_HID"
    data = loaders.ALL_LOADERS[domain]()
    block = results[domain]
    for name in args.models:
        model_block = block["models"][name]
        for run in model_block["runs"]:
            seed = int(run["seed"])
            checkpoint = Path(run["checkpoint"])
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = build_model(name, int(payload["n_items"]), int(payload["dim"]))
            model.load_state_dict(payload["state_dict"])
            device = "cpu" if name in {"SR-GNN", "SIGMA-compatible"} else (
                "mps" if torch.backends.mps.is_available() else "cpu")
            model = model.to(device).eval()
            keys, ranking, seconds, peak = predict_array(
                model, data["test_queries"], data["n_items"], topk=20,
                batch_size=128 if name == "SR-GNN" else 256,
                exclude_seen=False)
            ranks = ranks_at_20(ranking, targets_for(keys, data["test_queries"]))
            rank_path = args.artifact_dir / (
                f"diginetica_hid_full_{name.lower().replace('-', '_')}_"
                f"seed{seed}_ranks.npz")
            np.savez_compressed(
                rank_path, ranks=ranks,
                test_fingerprint=np.asarray(query_fingerprint(data["test_queries"])))
            run["test"] = metrics_from_ranks(ranks)
            run["inference_seconds"] = seconds
            run["latency_ms_per_query"] = 1000 * seconds / len(data["test_queries"])
            run["peak_tracked_device_bytes"] = max(
                int(run.get("peak_tracked_device_bytes", 0)), int(peak))
            run["training_scope"] = (
                "leakage-safe train; validation-selected checkpoint")
            print(f"{name} seed={seed} R20={run['test']['recall@20']:.6f}", flush=True)
        model_block["aggregate"] = aggregate(model_block["runs"])
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
