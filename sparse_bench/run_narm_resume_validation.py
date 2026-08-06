#!/usr/bin/env python3
"""Resume NARM only when held-out validation supports longer training."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import cearf
import hid_protocol
from narm_mps import expand_sessions, load_narm, predict_narm, train_narm


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--additional-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--output", type=Path,
                        default=HERE / "narm_resume_validation.json")
    args = parser.parse_args()
    started = time.time()
    data = hid_protocol.load_hid_diginetica()
    tune_sessions, validation = cearf.make_validation_split(
        data["train_sessions"], 0.10, 5000)
    model, history = load_narm(args.resume)
    valid_data = dict(data)
    valid_data["test_queries"] = validation
    before = hid_protocol.official_metrics(
        predict_narm(model, validation), valid_data)
    contexts, targets = expand_sessions(tune_sessions)
    config = replace(model.config, epochs=args.additional_epochs,
                     learning_rate=args.learning_rate)
    checkpoint = args.output.with_suffix(".pt")
    print(f"[NARM-RESUME] before={before} train={len(targets)}", flush=True)
    model, combined_history = train_narm(
        contexts, targets, config, checkpoint=checkpoint,
        initial_model=model, initial_history=history)
    after = hid_protocol.official_metrics(
        predict_narm(model, validation), valid_data)
    result = {
        "before": before, "after": after,
        "original_history": history,
        "combined_history": combined_history,
        "additional_epochs": args.additional_epochs,
        "learning_rate": args.learning_rate,
        "checkpoint": str(checkpoint),
        "seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
