#!/usr/bin/env python3
"""Validation-gated Graph-NARM transfer run on the official Diginetica split."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import torch

import cearf
import hid_protocol
from graph_narm_mps import from_narm_checkpoint
from narm_mps import expand_sessions, predict_narm, train_narm


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--narm-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path,
                        default=HERE / "graph_narm_validation.json")
    args = parser.parse_args()
    started = time.time()
    data = hid_protocol.load_hid_diginetica()
    tune_sessions, validation = cearf.make_validation_split(
        data["train_sessions"], 0.10, 5000)
    model, source_history = from_narm_checkpoint(args.narm_checkpoint)
    valid_data = dict(data)
    valid_data["test_queries"] = validation
    before = hid_protocol.official_metrics(
        predict_narm(model, validation), valid_data)
    contexts, targets = expand_sessions(tune_sessions)
    config = replace(model.config, epochs=args.epochs,
                     learning_rate=args.learning_rate)
    checkpoint = args.output.with_suffix(".pt")
    print(f"[GRAPH-NARM] before={before} train={len(targets)} "
          f"edges={len(model.edge_src)}", flush=True)
    model, history = train_narm(
        contexts, targets, config, checkpoint=checkpoint,
        initial_model=model, initial_history=[])
    after = hid_protocol.official_metrics(
        predict_narm(model, validation), valid_data)
    result = {
        "before": before, "after": after,
        "source_history": source_history, "graph_history": history,
        "graph_gate": float(torch.sigmoid(model.graph_gate).detach().cpu()),
        "checkpoint": str(checkpoint), "seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
