#!/usr/bin/env python3
"""Transfer full-data NARM into MPS-native Graph-NARM and evaluate once."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import time
import torch

import hid_protocol
from graph_narm_mps import from_narm_checkpoint
from narm_mps import predict_narm, train_narm


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--narm-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path,
                        default=HERE / "graph_narm_hid.json")
    args = parser.parse_args()
    started = time.time()
    data = hid_protocol.load_hid_diginetica()
    train_x, train_y = pickle.load(open(hid_protocol.HID_DATA / "train.txt", "rb"))
    model, source_history = from_narm_checkpoint(args.narm_checkpoint)
    config = replace(model.config, epochs=args.epochs,
                     learning_rate=args.learning_rate)
    checkpoint = args.output.with_suffix(".pt")
    print(f"[GRAPH-NARM] train={len(train_y)} test={len(data['test_queries'])} "
          f"edges={len(model.edge_src)}", flush=True)
    model, graph_history = train_narm(
        train_x, train_y, config, checkpoint=checkpoint,
        initial_model=model, initial_history=[])
    metrics = hid_protocol.official_metrics(
        predict_narm(model, data["test_queries"]), data)
    result = {
        "protocol": "Code4HID/MGCOT byte-identical Diginetica artifacts",
        "model": "Graph-NARM-edge-index-MPS",
        "source_checkpoint": str(args.narm_checkpoint),
        "source_history": source_history, "graph_history": graph_history,
        "graph_gate": float(torch.sigmoid(model.graph_gate).detach().cpu()),
        "metrics": metrics, "checkpoint": str(checkpoint),
        "seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
