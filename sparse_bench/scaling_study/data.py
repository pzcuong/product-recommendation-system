from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def load_diginetica():
    try:
        from sparse_bench import srgnn_preprocess
    except ImportError:
        import srgnn_preprocess
    return srgnn_preprocess.load_diginetica()


def checksum(data: dict) -> str:
    payload = json.dumps({"train": data["train_sessions"], "test": data["test_queries"]},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def split_train_validation(train: dict[str, list[int]], seed: int, fraction: float = .1):
    keys = sorted(k for k, v in train.items() if len(v) >= 2)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, round(len(keys) * fraction)) if len(keys) > 1 else 0
    val_keys = set(keys[:n_val])
    fit, validation = {}, {}
    for key in keys:
        seq = train[key]
        if key in val_keys:
            validation[key] = {"context": seq[:-1], "targets": [seq[-1]]}
        else:
            fit[key] = seq
    return fit, validation


def create_manifest(data: dict, scale: int, draw_seed: int, test_limit: int | None = 1500,
                    validation_fraction: float = .1) -> dict:
    keys = sorted(k for k, v in data["train_sessions"].items() if len(v) >= 2)
    if scale > len(keys):
        raise ValueError(f"scale {scale} exceeds {len(keys)} eligible sessions")
    selected = random.Random(draw_seed).sample(keys, scale)
    fit, validation = split_train_validation({k: data["train_sessions"][k] for k in selected},
                                             draw_seed + 1, validation_fraction)
    test_keys = sorted(data["test_queries"])
    if test_limit and len(test_keys) > test_limit:
        test_keys = random.Random(0).sample(test_keys, test_limit)
    return {"schema": 1, "dataset": "Diginetica", "dataset_checksum": checksum(data),
            "scale": scale, "draw_seed": draw_seed, "fit_session_ids": sorted(fit),
            "validation_session_ids": sorted(validation), "test_query_ids": sorted(test_keys),
            "validation_fraction": validation_fraction}


def materialize(data: dict, manifest: dict):
    fit = {k: data["train_sessions"][k] for k in manifest["fit_session_ids"]}
    val = {k: {"context": data["train_sessions"][k][:-1],
               "targets": [data["train_sessions"][k][-1]]}
           for k in manifest["validation_session_ids"]}
    test = {k: data["test_queries"][k] for k in manifest["test_query_ids"]}
    return fit, val, test


def coverage_stats(train: dict[str, list[int]], queries: dict[str, dict]) -> dict:
    freq = Counter(x for seq in train.values() for x in seq)
    targets = [q["targets"][0] for q in queries.values()]
    seen = sum(t in freq for t in targets)
    lengths = sorted(map(len, train.values()))
    return {"sessions": len(train), "interactions": sum(lengths),
            "unique_train_items": len(freq), "test_targets": len(targets),
            "test_targets_seen": seen, "target_coverage": seen / len(targets) if targets else 0,
            "singleton_items": sum(v == 1 for v in freq.values()),
            "singleton_item_rate": sum(v == 1 for v in freq.values()) / len(freq) if freq else 0,
            "mean_session_length": sum(lengths) / len(lengths) if lengths else 0,
            "median_session_length": lengths[len(lengths) // 2] if lengths else 0}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
