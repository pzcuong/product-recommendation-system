#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np

import cearf
import loaders
import pasgr
from perquery_router import BucketedRouter, ContinuousRouter, extract_features
from run_cearfn import fuse, tune_beta
from run_cearfn_evidence import popularity_partition, targets_for
from run_cearfn_v2 import build_features, load_pasgr_config, train_pasgr_v2
from run_pasgr_full import semantic_matrix
from validation_protocol import hold_out_validation_targets


HERE = Path(__file__).resolve().parent


def fuse_with_betas(memory: np.ndarray, neural: np.ndarray, keys: list[str], betas: dict[str, float]) -> np.ndarray:
    output = np.empty((len(keys), 20), dtype=np.int32)
    for row, uid in enumerate(keys):
        output[row] = fuse(memory[row], neural[row], betas[uid])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True,
                        choices=("Video_Games", "Baby_Products", "Diginetica_HID"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-file", type=Path, default=HERE / "pasgr_config_per_domain.json")
    parser.add_argument(
        "--semantic-matrix", type=Path,
        help="Optional precomputed teacher matrix; overrides TF-IDF/SVD.")
    parser.add_argument("--artifact-dir", type=Path, default=HERE / "cearfn_v2_artifacts")
    parser.add_argument("--work-artifact-dir", type=Path, required=True,
                        help="Directory containing *_nested_valid_memory.npz and *_nested_test_memory.npz")
    parser.add_argument("--nested-results", type=Path, default=HERE / "cearfn_v2_nested_results.json")
    parser.add_argument("--max-valid-queries", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = loaders.ALL_LOADERS[args.domain]()
    exclude_seen = args.domain not in {"Diginetica_HID", "Tmall"}
    if len(data["valid_queries"]) > args.max_valid_queries:
        vk = sorted(data["valid_queries"], key=cearf._stable_fraction)[:args.max_valid_queries]
        data["valid_queries"] = {k: data["valid_queries"][k] for k in vk}
    sessions = data["train_sessions"]
    tune_sessions = hold_out_validation_targets(sessions, data["valid_queries"])
    freq = Counter(x for seq in sessions.values() for x in seq)
    head_items, _, _ = popularity_partition(freq, data["n_items"])
    head_set = set(head_items.tolist())

    valid_memory = np.load(args.work_artifact_dir / f"{args.domain.lower()}_nested_valid_memory.npz", allow_pickle=True)
    test_memory = np.load(args.work_artifact_dir / f"{args.domain.lower()}_nested_test_memory.npz", allow_pickle=True)
    valid_keys = [str(x) for x in valid_memory["keys"]]
    test_keys = [str(x) for x in test_memory["keys"]]
    nested = json.loads(args.nested_results.read_text())
    match = None
    for run in nested[args.domain]["runs"]:
        if int(run["seed"]) == args.seed:
            match = run
            break
    if match is None:
        raise RuntimeError(f"Could not find seed {args.seed} for {args.domain} in {args.nested_results}")
    selected_router = str(match["selected_router"])

    tune_index = cearf.CEARFIndex(
        tune_sessions, data["n_items"],
        cearf.CEARFConfig(exclude_seen=exclude_seen))
    valid_features = build_features(tune_index, data["valid_queries"], freq, head_set)
    final_index = cearf.CEARFIndex(
        sessions, data["n_items"],
        cearf.CEARFConfig(exclude_seen=exclude_seen))
    test_features = build_features(final_index, data["test_queries"], freq, head_set)

    semantic = (np.load(args.semantic_matrix).astype(np.float32)
                if args.semantic_matrix else semantic_matrix(args.domain, data))
    if semantic is not None and semantic.shape[0] != data["n_items"]:
        raise ValueError(
            f"semantic rows ({semantic.shape[0]}) != n_items ({data['n_items']})")
    gated_config = load_pasgr_config(args.domain, args.config_file)
    validation_model = train_pasgr_v2(data, tune_sessions, semantic, args.seed, args.epochs, gated_config)
    _, neural_valid = pasgr.predict_pasgr_array(
        validation_model, data["valid_queries"], data["n_items"], 120,
        exclude_seen=exclude_seen)
    del validation_model
    gc.collect()
    final_model = train_pasgr_v2(data, sessions, semantic, args.seed, args.epochs, gated_config)
    _, neural_test = pasgr.predict_pasgr_array(
        final_model, data["test_queries"], data["n_items"], 120,
        exclude_seen=exclude_seen)
    del final_model
    gc.collect()

    valid_memory_dict = {uid: list(valid_memory["selected"][i]) for i, uid in enumerate(valid_keys)}
    valid_neural_dict = {uid: list(neural_valid[i]) for i, uid in enumerate(valid_keys)}

    config = cearf.CEARFConfig(exclude_seen=exclude_seen)
    betas_regime, _ = tune_beta(valid_memory_dict, valid_neural_dict, data["valid_queries"], config.short_context)
    regime_valid = {
        uid: betas_regime["short" if valid_features[uid].length <= config.short_context else "long"]
        for uid in valid_keys
    }
    fused_valid_regime = fuse_with_betas(valid_memory["selected"], neural_valid, valid_keys, regime_valid)

    bucketed = BucketedRouter(fallback_beta=betas_regime)
    bucketed.fit({uid: data["valid_queries"][uid] for uid in valid_keys},
                 valid_memory_dict, valid_neural_dict, valid_keys, valid_features)
    bucketed_valid = {uid: bucketed.beta_for(valid_features[uid]) for uid in valid_keys}
    fused_valid_bucketed = fuse_with_betas(valid_memory["selected"], neural_valid, valid_keys, bucketed_valid)

    continuous = ContinuousRouter(alpha=1.0)
    continuous.fit({uid: data["valid_queries"][uid] for uid in valid_keys},
                   valid_memory_dict, valid_neural_dict, valid_keys, valid_features)
    continuous_valid = {uid: continuous.beta_for(valid_features[uid]) for uid in valid_keys}
    fused_valid_continuous = fuse_with_betas(valid_memory["selected"], neural_valid, valid_keys, continuous_valid)
    regime_test = {
        uid: betas_regime["short" if test_features[uid].length <= config.short_context else "long"]
        for uid in test_keys
    }
    fused_test_regime = fuse_with_betas(test_memory["selected"], neural_test, test_keys, regime_test)
    bucketed_test = {uid: bucketed.beta_for(test_features[uid]) for uid in test_keys}
    fused_test_bucketed = fuse_with_betas(test_memory["selected"], neural_test, test_keys, bucketed_test)
    continuous_test = {uid: continuous.beta_for(test_features[uid]) for uid in test_keys}
    fused_test_continuous = fuse_with_betas(test_memory["selected"], neural_test, test_keys, continuous_test)

    valid_selected = {
        "regime": fused_valid_regime,
        "bucketed": fused_valid_bucketed,
        "continuous": fused_valid_continuous,
    }[selected_router]
    test_selected = {
        "regime": fused_test_regime,
        "bucketed": fused_test_bucketed,
        "continuous": fused_test_continuous,
    }[selected_router]

    np.savez_compressed(
        args.output,
        valid_keys=np.asarray(valid_keys, dtype=str),
        test_keys=np.asarray(test_keys, dtype=str),
        valid_selected_top20=valid_selected.astype(np.int32),
        selected_top20=test_selected.astype(np.int32),
        selected_router=np.asarray(selected_router),
    )
    print(args.output)


if __name__ == "__main__":
    main()
