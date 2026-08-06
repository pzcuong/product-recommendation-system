#!/usr/bin/env python3
"""Validation-select a rejection-complete RRF expert family including NARM.

The NARM checkpoint is the validation-selected checkpoint used by the main
baseline table.  This script performs inference only, persists top-20 lists,
and evaluates every non-empty subset of {CEARF-N, STAN, V-SKNN, NARM}.
The four singleton candidates are minimal admissible systems and therefore
allow validation to reject external ensembling itself.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import cearf
import loaders
from paper_models import build_model
from run_cearfn_evidence import targets_for
from run_global_fusion_selector import (
    DEFAULT_BASELINE_RESULTS,
    fuse_rrf,
    load_test_matrix,
    predict_matrix,
    summarize,
)
from run_paper_baselines import predict_array
from neighborhood_baselines import NeighborhoodConfig, NeighborhoodIndex
from validation_protocol import hold_out_validation_targets


HERE = Path(__file__).resolve().parent


def load_narm(checkpoint: Path) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model") != "NARM":
        raise RuntimeError(f"{checkpoint} is not a NARM checkpoint")
    model = build_model("NARM", int(payload["n_items"]), int(payload["dim"]))
    model.load_state_dict(payload["state_dict"])
    return model.cpu().eval()


def align(keys: list[str], source_keys: list[str], values: np.ndarray) -> np.ndarray:
    if set(keys) != set(source_keys):
        raise RuntimeError("Query-key coverage mismatch")
    row = {key: i for i, key in enumerate(source_keys)}
    return np.asarray([values[row[key]] for key in keys], dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True,
                        choices=("Video_Games", "Baby_Products", "Diginetica_HID"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--cearf-artifact", required=True, type=Path)
    parser.add_argument("--narm-checkpoint", required=True, type=Path)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--criterion",
                        choices=("r6_r20", "r10_r20", "r20", "r6_r10_r20"),
                        default="r10_r20")
    parser.add_argument("--narm-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = loaders.ALL_LOADERS[args.domain]()
    exclude_seen = args.domain not in {"Diginetica_HID", "Tmall"}
    valid_keys0 = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
    valid_queries = {key: data["valid_queries"][key] for key in valid_keys0}
    test_queries = data["test_queries"]

    c = np.load(args.cearf_artifact, allow_pickle=True)
    needed = {"valid_keys", "test_keys", "valid_selected_top20", "selected_top20"}
    missing = needed.difference(c.files)
    if missing:
        raise RuntimeError(f"{args.cearf_artifact}: missing {sorted(missing)}")
    valid_keys = [str(x) for x in c["valid_keys"]]
    test_keys = [str(x) for x in c["test_keys"]]

    baseline_path = args.baseline_results or DEFAULT_BASELINE_RESULTS[args.domain]
    baseline = json.loads(baseline_path.read_text())[args.domain]["methods"]
    vcfg = NeighborhoodConfig(**baseline["vsknn"]["selected_config"])
    scfg = NeighborhoodConfig(**baseline["stan"]["selected_config"])

    valid_index = NeighborhoodIndex(
        hold_out_validation_targets(data["train_sessions"], valid_queries),
        data["n_items"])
    vk, vv = predict_matrix(valid_index, valid_queries, vcfg)
    sk, sv = predict_matrix(valid_index, valid_queries, scfg)
    vv = align(valid_keys, vk, vv)
    sv = align(valid_keys, sk, sv)

    loaded_v = load_test_matrix(Path(baseline["vsknn"].get("artifact", "")))
    loaded_s = load_test_matrix(Path(baseline["stan"].get("artifact", "")))
    if loaded_v is None or loaded_s is None:
        test_index = NeighborhoodIndex(data["train_sessions"], data["n_items"])
        tvk, tv = predict_matrix(test_index, test_queries, vcfg)
        tsk, ts = predict_matrix(test_index, test_queries, scfg)
    else:
        (tvk, tv), (tsk, ts) = loaded_v, loaded_s
    tv = align(test_keys, tvk, tv)
    ts = align(test_keys, tsk, ts)

    narm = load_narm(args.narm_checkpoint)
    nvk, nv, _, _ = predict_array(
        narm, valid_queries, data["n_items"], topk=20, batch_size=256,
        exclude_seen=exclude_seen)
    ntk, nt, _, _ = predict_array(
        narm, test_queries, data["n_items"], topk=20, batch_size=256,
        exclude_seen=exclude_seen)
    nv = align(valid_keys, nvk, nv)
    nt = align(test_keys, ntk, nt)
    args.narm_artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.narm_artifact,
        valid_keys=np.asarray(valid_keys, dtype=str),
        test_keys=np.asarray(test_keys, dtype=str),
        valid_top20=nv,
        test_top20=nt,
        seed=np.asarray(args.seed),
    )

    # These four singleton candidates are intentionally retained in the
    # selection dictionary.  The loop below adds the remaining 11 subsets,
    # yielding all 2^4 - 1 = 15 non-empty members.
    candidate_valid = {
        "cearf": np.asarray(c["valid_selected_top20"], dtype=np.int32),
        "stan": sv,
        "vsknn": vv,
        "narm": nv,
    }
    candidate_test = {
        "cearf": np.asarray(c["selected_top20"], dtype=np.int32),
        "stan": ts,
        "vsknn": tv,
        "narm": nt,
    }
    experts = tuple(candidate_valid)
    for size in range(2, len(experts) + 1):
        for parts in itertools.combinations(experts, size):
            name = "rrf_" + "_".join(parts)
            candidate_valid[name] = np.asarray([
                fuse_rrf([candidate_valid[p][i] for p in parts])
                for i in range(len(valid_keys))
            ], dtype=np.int32)
            candidate_test[name] = np.asarray([
                fuse_rrf([candidate_test[p][i] for p in parts])
                for i in range(len(test_keys))
            ], dtype=np.int32)

    valid_targets = targets_for(valid_keys, valid_queries)
    test_targets = targets_for(test_keys, test_queries)
    validation, selected = summarize(candidate_valid, valid_targets, args.criterion)
    test, _ = summarize(candidate_test, test_targets, args.criterion)
    result = {
        "domain": args.domain,
        "seed": args.seed,
        "selection_rule": args.criterion,
        "selected_candidate": selected,
        "validation": validation,
        "test": test,
        "selected_test": test[selected],
        "narm_checkpoint": str(args.narm_checkpoint),
        "narm_rank_artifact": str(args.narm_artifact),
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output)
    print(json.dumps({"selected": selected, "test": test[selected]}, indent=2))


if __name__ == "__main__":
    main()
