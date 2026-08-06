"""ADMA external public-dataset check: RetailRocket + SIGMA-compatible model.

The runner deliberately records backend provenance.  It is not allowed to
label the pure-PyTorch port as official Mamba4Rec; a CUDA machine can add the
official Mamba4Rec run using ``mamba4rec_reference.build_official_mamba``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import grouped_eval, loaders, sigma_model
from .mamba4rec_reference import backend_status, sigma_provenance


def subset(data, max_train, max_test):
    rng = random.Random(42)
    train_keys = sorted(data["train_sessions"])
    if max_train and len(train_keys) > max_train:
        train_keys = sorted(rng.sample(train_keys, max_train))
        data["train_sessions"] = {k: data["train_sessions"][k] for k in train_keys}
    test_keys = sorted(data["test_queries"])
    if max_test and len(test_keys) > max_test:
        test_keys = sorted(random.Random(0).sample(test_keys, max_test))
        data["test_queries"] = {k: data["test_queries"][k] for k in test_keys}
    data["item_freq"] = Counter(x for seq in data["train_sessions"].values() for x in seq)
    return data


def predict(models, data, topk=20):
    device = next(models[0].parameters()).device
    ids = sorted(data["test_queries"]); predictions = {}
    for start in range(0, len(ids), 128):
        chunk = ids[start:start + 128]
        seqs = [[x for x in data["test_queries"][u]["context"] if 0 < x < data["n_items"]][-50:]
                for u in chunk]
        lengths = [len(s) for s in seqs]
        width = max(max(lengths, default=0), 1)
        inp = torch.zeros(len(chunk), width, dtype=torch.long, device=device)
        for i, seq in enumerate(seqs):
            if seq: inp[i, :len(seq)] = torch.tensor(seq, device=device)
        with torch.no_grad():
            scores = sum((m(inp, torch.tensor(lengths, device=device)) for m in models),
                         torch.zeros(len(chunk), data["n_items"], device=device))
        for uid, score, context in zip(chunk, scores.cpu().numpy(), seqs):
            score[0] = -np.inf
            score[list(set(context))] = -np.inf
            order = np.argsort(-score)[:topk]
            predictions[uid] = [int(i) for i in order if i != 0]
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-test", type=int, default=323)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--output", default="sparse_bench/scaling_study/artifacts/retailrocket_sigma_smoke.json")
    args = parser.parse_args()
    t0 = time.time()
    data = subset(loaders.load_retailrocket(), args.max_train, args.max_test)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    sessions = [s for s in data["train_sessions"].values() if len(s) >= 2]
    models = sigma_model.train_sigma(sessions, data["n_items"], epochs=args.epochs,
                                     seeds=seeds, embed_dim=64)
    predictions = predict(models, data)
    metrics = grouped_eval.evaluate_all_groups(predictions, data, k_values=[5, 10, 20])["overall"]
    result = {
        "dataset": "RetailRocket", "train_sessions": len(sessions),
        "test_queries": len(data["test_queries"]), "n_items": data["n_items"],
        "epochs": args.epochs, "seeds": list(seeds), "metrics": metrics,
        "backend": {"sigma_reference": sigma_provenance(),
                    "mamba4rec_status": backend_status().__dict__},
        "seconds": time.time() - t0,
        "claim_scope": "external public-dataset reference check; not official Mamba4Rec",
    }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

