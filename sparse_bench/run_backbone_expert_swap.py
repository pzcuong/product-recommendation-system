#!/usr/bin/env python3
"""Backbone expert-swap audit: SASRec / BERT4Rec inside the CEARF-N gate.

Generalizes run_dynamic_beta_expert_swap.py (NARM) to any paper-baseline
backbone.  Protocol per domain x seed:

  1. deterministic source-session OOF split (inner-fit / profile / gate)
  2. CEARFIndex on inner-fit; memory profiles tuned on the profile split
  3. backbone trained on inner-fit sessions -> gate top-120 rankings
  4. OOF-global beta + bounded 3-feature dynamic gate fit on
     (gate memory, gate expert ranks, gate targets) -- target-free
  5. backbone refit on full sessions -> test top-120; memory rebuilt on
     full sessions; fusion evaluated once under equal .5 / OOF-global /
     dynamic gate

Epoch budgets are fixed per backbone and disclosed (no per-seed selection):
SASRec 12, BERT4Rec 12 -- the suite maximum, identical across seeds.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch

import cearf
import loaders
from dynamic_beta import fuse_with_dynamic_beta
from paper_models import build_model, model_logits
from run_cearfn_evidence import (
    build_memory_arrays,
    popularity_partition,
    query_fingerprint,
    ranks_at_20,
    targets_for,
)
from run_paper_baselines import device_for, train_fixed_epochs
from run_dynamic_beta import (
    canonical_validation_sources,
    fit_dynamic,
    fit_global,
    make_training_oof_split,
)
from run_dynamic_beta_expert_swap import (
    align_rankings,
    canonical_profiles,
    context_feature_matrix,
    predict_array_target_free,
)
from validation_protocol import hold_out_validation_targets

PROTOCOL = "dynamic-beta-backbone-expert-swap-v1"
EPOCH_BUDGET = {"SASRec": 12, "BERT4Rec": 12}
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID",
           "RetailRocket_Large")
HERE = Path(__file__).resolve().parent


def build_and_cache_memory(path: Path, index, queries: dict, profiles: dict,
                           width: int, label: str) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path) as saved:
            if (str(saved["fingerprint"].item()) == query_fingerprint(queries)
                    and str(saved["profiles"].item())
                    == json.dumps(profiles, sort_keys=True)):
                keys = saved["keys"].astype(str)
                out = {name: saved[name].astype(np.int32)
                       for name in ("transition", "session", "popularity",
                                    "selected")}
                out["keys"] = keys
                return out
        path.unlink()
    arrays = build_memory_arrays(index, queries, profiles, width, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{k: v for k, v in arrays.items() if k != "keys"},
        keys=np.asarray(arrays["keys"]),
        fingerprint=np.asarray(query_fingerprint(queries)),
        profiles=np.asarray(json.dumps(profiles, sort_keys=True)),
    )
    return arrays


def train_backbone(name: str, sessions: dict, n_items: int, seed: int,
                   epochs: int, batch_size: int, role: str, ckpt: Path):
    fingerprint = None  # session fingerprint of the *fit* sessions
    import hashlib
    h = hashlib.sha256()
    for uid in sorted(sessions):
        h.update(f"{uid}:{','.join(map(str, sessions[uid]))};".encode())
    fingerprint = h.hexdigest()
    if ckpt.exists():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        if (payload.get("model") == name
                and int(payload.get("seed", -1)) == seed
                and int(payload.get("epoch", -1)) == epochs
                and int(payload.get("n_items", -1)) == n_items
                and payload.get("sessions_fingerprint") == fingerprint):
            model = build_model(name, n_items, 64)
            model.load_state_dict(payload["state_dict"])
            print(f"[SWAP:{name}] reuse {role} ckpt {ckpt.name}", flush=True)
            return model.eval(), payload.get("training_report", {})
    print(f"[SWAP:{name}] train {role} seed={seed} epochs={epochs}",
          flush=True)
    model, report = train_fixed_epochs(name, sessions, n_items, seed,
                                       epochs, batch_size)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": name, "seed": seed, "epoch": epochs, "n_items": n_items,
        "sessions_fingerprint": fingerprint,
        "state_dict": {k: v.detach().cpu()
                       for k, v in model.state_dict().items()},
        "training_report": report,
    }, ckpt)
    return model.cpu().eval(), report


def metrics_from_ranks(ranks: np.ndarray) -> dict:
    hit6 = (ranks > 0) & (ranks <= 6)
    hit20 = (ranks > 0) & (ranks <= 20)
    gains = np.where(hit20,
                     1.0 / np.log2(np.clip(ranks, 1, None).astype(
                         np.float64) + 1.0), 0.0)
    return {
        "recall@6": float(hit6.mean()),
        "recall@20": float(hit20.mean()),
        "ndcg@20": float(gains.mean()),
        "utility": float(.5 * hit6.mean() + .5 * hit20.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backbones", nargs="+",
                        choices=sorted(EPOCH_BUDGET))
    parser.add_argument("--domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--seeds", nargs="*", type=int,
                        default=[42, 123, 456])
    parser.add_argument("--candidate-width", type=int, default=120)
    parser.add_argument("--oof-fraction", type=float, default=0.10)
    parser.add_argument("--oof-cap", type=int, default=5_000)
    parser.add_argument("--profile-cap", type=int, default=1_000)
    parser.add_argument("--valid-cap", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--artifact-dir", type=Path,
                        default=HERE / "backbone_swap_artifacts")
    parser.add_argument("--output", type=Path,
                        default=HERE / "backbone_swap_results.json")
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    results = (json.loads(args.output.read_text())
               if args.output.exists() else {})

    for domain in args.domains:
        print(f"\n[SWAP] === {domain} ===", flush=True)
        data = loaders.ALL_LOADERS[domain]()
        if len(data["valid_queries"]) > args.valid_cap:
            valid_keys = sorted(data["valid_queries"],
                                key=cearf._stable_fraction)[:args.valid_cap]
            data["valid_queries"] = {uid: data["valid_queries"][uid]
                                     for uid in valid_keys}
        sessions = data["train_sessions"]
        tune_sessions = hold_out_validation_targets(
            sessions, data["valid_queries"])
        validation_sources = canonical_validation_sources(
            data["valid_queries"])
        inner_fit, profile_queries, gate_queries, split_report = (
            make_training_oof_split(tune_sessions, validation_sources,
                                    args.oof_fraction, args.oof_cap,
                                    args.profile_cap))
        exclude_seen = domain not in {"Diginetica_HID", "Tmall",
                                     "RetailRocket_Large"}
        config = cearf.CEARFConfig(exclude_seen=exclude_seen)
        inner_index = cearf.CEARFIndex(inner_fit, data["n_items"], config)
        profiles = canonical_profiles(
            cearf.tune_profiles(inner_index, profile_queries)[0])
        del inner_index
        gc.collect()

        final_index = cearf.CEARFIndex(sessions, data["n_items"], config)
        gate_memory = build_and_cache_memory(
            args.artifact_dir / domain / "gate_memory.npz",
            inner_index, gate_queries, profiles, args.candidate_width,
            f"{domain}-gate")
        test_memory = build_and_cache_memory(
            args.artifact_dir / domain / "test_memory.npz",
            final_index, data["test_queries"], profiles,
            args.candidate_width, f"{domain}-test")
        del inner_index, final_index
        gc.collect()

        gate_keys = [str(k) for k in gate_memory["keys"]]
        test_keys = [str(k) for k in test_memory["keys"]]
        gate_targets = targets_for(gate_keys, gate_queries)
        test_targets = targets_for(test_keys, data["test_queries"])

        inner_freq = {item: n for item, n in __import__(
            "collections").Counter(
            item for seq in inner_fit.values() for item in seq).items()}
        final_freq = {item: n for item, n in __import__(
            "collections").Counter(
            item for seq in sessions.values() for item in seq).items()}
        inner_head = set(popularity_partition(inner_freq,
                                              data["n_items"])[0].tolist())
        final_head = set(popularity_partition(final_freq,
                                              data["n_items"])[0].tolist())
        gate_features = context_feature_matrix(
            gate_queries, gate_keys, inner_freq, inner_head)
        test_features = context_feature_matrix(
            data["test_queries"], test_keys, final_freq, final_head)

        domain_key = f"{domain}::" + ",".join(args.backbones)
        if domain_key not in results:
            results[domain_key] = {
                "protocol": PROTOCOL, "profiles": profiles,
                "epoch_budgets": {b: EPOCH_BUDGET[b]
                                  for b in args.backbones},
                "backbones": {}}

        for backbone in args.backbones:
            epochs = EPOCH_BUDGET[backbone]
            ckpt_dir = args.artifact_dir / domain / "checkpoints"
            pred_dir = args.artifact_dir / domain / "predictions"
            bb_block = results[domain_key]["backbones"].setdefault(
                backbone, {})
            for seed in args.seeds:
                if str(seed) in bb_block:
                    print(f"[SWAP:{backbone}] {domain} seed={seed} done",
                          flush=True)
                    continue
                started = time.time()
                oof_ckpt = ckpt_dir / f"{backbone}_seed{seed}_oof.pt"
                full_ckpt = ckpt_dir / f"{backbone}_seed{seed}_full.pt"

                oof_model, _ = train_backbone(
                    backbone, inner_fit, data["n_items"], seed, epochs,
                    args.batch_size, "oof", oof_ckpt)
                dev = device_for(backbone)
                oof_model = oof_model.to(dev).eval()
                src_keys, oof_raw, _ = predict_array_target_free(
                    oof_model, gate_queries, data["n_items"],
                    args.candidate_width, args.batch_size, exclude_seen)
                oof_expert = align_rankings(gate_keys, src_keys, oof_raw)
                del oof_model
                gc.collect()

                global_model, global_report = fit_global(
                    gate_memory["selected"], oof_expert, gate_targets, seed)
                dynamic_model, dynamic_report = fit_dynamic(
                    gate_features, gate_memory["selected"], oof_expert,
                    gate_targets, seed, float(global_model.beta_))
                gate_betas = dynamic_model.predict(gate_features)

                full_model, _ = train_backbone(
                    backbone, sessions, data["n_items"], seed, epochs,
                    args.batch_size, "full", full_ckpt)
                full_model = full_model.to(dev).eval()
                _, test_expert_raw, _ = predict_array_target_free(
                    full_model, data["test_queries"], data["n_items"],
                    args.candidate_width, args.batch_size, exclude_seen)
                test_expert = align_rankings(test_keys, test_keys,
                                             test_expert_raw)
                del full_model
                gc.collect()

                betas = np.full(len(test_keys),
                                float(global_model.beta_), dtype=np.float32)
                fused_global = fuse_with_dynamic_beta(
                    test_memory["selected"], test_expert, betas,
                    topk=20, constant=20.0)
                fused_dyn = fuse_with_dynamic_beta(
                    test_memory["selected"], test_expert,
                    dynamic_model.predict(test_features),
                    topk=20, constant=20.0)

                def r20(ranking):
                    t = targets_for(test_keys, data["test_queries"])
                    return metrics_from_ranks(
                        ranks_at_20(ranking, t))["recall@20"]

                memory_metrics = metrics_from_ranks(ranks_at_20(
                    test_memory["selected"], test_targets))
                expert_metrics = metrics_from_ranks(ranks_at_20(
                    test_expert, test_targets))
                run = {
                    "seed": seed,
                    "memory": memory_metrics,
                    "expert_only": expert_metrics,
                    "fixed_half": {
                        "recall@20": r20(fuse_with_dynamic_beta(
                            test_memory["selected"], test_expert,
                            np.full(len(test_keys), .5, dtype=np.float32),
                            topk=20))},
                    "oof_global": {
                        "beta": float(global_model.beta_),
                        "recall@20": r20(fused_global)},
                    "dynamic": {
                        "beta_mean": float(
                            dynamic_model.predict(test_features).mean()),
                        "recall@20": r20(fused_dyn)},
                    "seconds": round(time.time() - started, 1),
                }
                bb_block[str(seed)] = run
                print(f"[SWAP:{backbone}] {domain} seed={seed} "
                      f"mem={memory_metrics['recall@20']:.5f} "
                      f"exp={expert_metrics['recall@20']:.5f} "
                      f"glob={run['oof_global']['recall@20']:.5f} "
                      f"dyn={run['dynamic']['recall@20']:.5f} "
                      f"({run['seconds']}s)", flush=True)
                args.output.write_text(json.dumps(results, indent=1))

    args.output.write_text(json.dumps(results, indent=1))
    print("saved", args.output)


if __name__ == "__main__":
    main()
