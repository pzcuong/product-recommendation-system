#!/usr/bin/env python3
"""Train dense NARM on the exact Code4HID/MGCOT Diginetica split."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import time

import hid_protocol
from narm_mps import NARMConfig, load_narm, predict_narm, train_narm


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--train-cap", type=int)
    parser.add_argument("--test-cap", type=int)
    parser.add_argument("--resume", type=Path,
                        help="Resume model weights/history from this checkpoint")
    parser.add_argument("--learning-rate", type=float,
                        help="Override the learning rate (useful for resume)")
    parser.add_argument("--output", type=Path,
                        default=HERE / "narm_hid_results.json")
    args = parser.parse_args()
    started = time.time()
    data = hid_protocol.load_hid_diginetica()
    train_x, train_y = pickle.load(open(hid_protocol.HID_DATA / "train.txt", "rb"))
    queries = data["test_queries"]
    if args.test_cap:
        queries = dict(list(queries.items())[:args.test_cap])
    initial_model = None
    initial_history = None
    if args.resume:
        initial_model, initial_history = load_narm(args.resume)
        config = replace(initial_model.config, epochs=args.epochs,
                         batch_size=args.batch_size,
                         learning_rate=(args.learning_rate
                                        if args.learning_rate is not None
                                        else initial_model.config.learning_rate))
    else:
        config = NARMConfig(
            n_items=data["n_items"], dim=args.dim, batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=(args.learning_rate
                           if args.learning_rate is not None else 1e-3))
    checkpoint = args.output.with_suffix(".pt")
    print(f"[NARM-MPS] train={min(len(train_y), args.train_cap or len(train_y))} "
          f"test={len(queries)} device=MPS resume={args.resume}", flush=True)
    model, history = train_narm(train_x, train_y, config,
                                cap=args.train_cap, checkpoint=checkpoint,
                                initial_model=initial_model,
                                initial_history=initial_history)
    predictions = predict_narm(model, queries, topk=20)
    eval_data = dict(data)
    eval_data["test_queries"] = queries
    result = {
        "protocol": "Code4HID/MGCOT byte-identical Diginetica artifacts",
        "model": "NARM-bilinear-dense-MPS",
        "config": config.__dict__,
        "train_cap": args.train_cap,
        "resume": str(args.resume) if args.resume else None,
        "metrics": hid_protocol.official_metrics(predictions, eval_data),
        "history": history,
        "checkpoint": str(checkpoint),
        "seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
