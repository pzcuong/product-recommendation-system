#!/usr/bin/env python3
"""Run baselines with semantic-initialized embeddings.

Tests whether baselines benefit from the same metadata signal that CEARF-N's
PASGR uses. If CEARF-N still wins, the fusion claim is stronger (metadata
alone doesn't explain the gain). If baselines win, the claim weakens.

Protocol: same as run_paper_baselines.py, but embeddings are initialised
from the semantic matrix S (projected to model dim via SVD).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cearf
import loaders
from run_pasgr_full import semantic_matrix
from paper_models import build_model, model_logits
from run_paper_baselines import (
    PrefixDataset, collate, predict_array, targets_for,
    ranks_at_20, metrics_from_ranks, device_for, mps_memory)
from run_cearfn_evidence import query_fingerprint
from validation_protocol import hold_out_validation_targets

HERE = Path(__file__).resolve().parent
MODELS = ("GRU4Rec", "NARM")
SEEDS = (42, 123, 456)
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")


def project_semantic(S: np.ndarray, n_items: int, dim: int) -> np.ndarray:
    """Project semantic matrix S (n_items × S_dim) to (n_items × dim)."""
    # Only project items that have non-zero rows
    mask = np.linalg.norm(S, axis=1) > 1e-8
    S_sub = S[mask]
    # SVD to dim
    out_dim = min(dim, S_sub.shape[0] - 1, S_sub.shape[1])
    U, sigma, Vt = np.linalg.svd(S_sub, full_matrices=False)
    projected = U[:, :out_dim] * sigma[:out_dim]
    # Normalise
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected /= np.maximum(norms, 1e-8)
    # Place back
    result = np.zeros((n_items, dim), dtype=np.float32)
    result[mask] = projected.astype(np.float32)
    return result


def train_with_semantic_init(name, sessions, validation, n_items, semantic,
                              seed, max_epochs, batch_size, patience,
                              checkpoint, exclude_seen=True):
    """Train baseline with semantic-initialized embeddings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = device_for(name)
    model = build_model(name, n_items, 64).to(dev)

    # Initialize item embeddings from semantic matrix
    if semantic is not None and np.linalg.norm(semantic) > 0:
        projected = (semantic if semantic.shape == (n_items, 64)
                     else project_semantic(semantic, n_items, 64))
        with torch.no_grad():
            model.item.weight.copy_(torch.from_numpy(projected).to(dev))
        print(f'  [INIT] Semantic init applied ({np.count_nonzero(projected)}/{n_items} items)',
              flush=True)
    else:
        print(f'  [INIT] No semantic matrix, using default init', flush=True)

    dataset = PrefixDataset(sessions, n_items)
    generator = torch.Generator().manual_seed(seed)
    effective_batch = min(batch_size, 128) if name == "SR-GNN" else batch_size
    loader = DataLoader(dataset, batch_size=effective_batch, shuffle=True,
                        collate_fn=collate, generator=generator, num_workers=0)
    # Per-architecture lr
    ARCH_LR = {"SASRec": 5e-4, "SIGMA-compatible": 5e-4}
    base_lr = ARCH_LR.get(name, 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
    best = None
    stale = 0
    history = []
    peak = mps_memory()
    train_started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        epoch_started = time.time()
        for step, (contexts, lengths, targets) in enumerate(loader, 1):
            if name != "SR-GNN":
                contexts, lengths = contexts.to(dev), lengths.to(dev)
            targets = targets.to(dev)
            logits = model_logits(model, contexts, lengths)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                count += len(targets)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            count += len(targets)
            peak = max(peak, mps_memory())
        valid_keys, valid_top, valid_seconds, valid_peak = predict_array(
            model, validation, n_items, topk=20,
            batch_size=128 if name == "SR-GNN" else 256,
            exclude_seen=exclude_seen)
        valid_ranks = ranks_at_20(valid_top, targets_for(valid_keys, validation))
        metrics = metrics_from_ranks(valid_ranks)
        utility = .5 * (metrics["recall@6"] + metrics["recall@20"])
        epoch_record = {
            "epoch": epoch,
            "loss": loss_sum / max(count, 1),
            "train_epoch_seconds": time.time() - epoch_started,
            "validation": metrics,
            "utility": utility,
        }
        history.append(epoch_record)
        print(f"  [{name}+sem] seed={seed} epoch={epoch}/{max_epochs} "
              f"valid_R20={metrics['recall@20']:.5f} util={utility:.5f}", flush=True)
        candidate = (utility, metrics["recall@20"], metrics["recall@6"],
                     -epoch, epoch, metrics)
        if best is None or candidate[:5] > best[:5]:
            best = candidate
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= patience:
            print(f"  [{name}+sem] Early stop at epoch {epoch}", flush=True)
            break
    train_seconds = time.time() - train_started
    model.load_state_dict(torch.load(checkpoint, map_location=dev,
                                      weights_only=True))
    training = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": train_seconds,
        "peak_tracked_device_bytes": peak,
        "history": history,
        "best_epoch": best[4] if best else max_epochs,
    }
    return model, training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-cap", type=int, default=5000)
    parser.add_argument("--teacher", choices=("tfidf", "minilm"),
                        default="tfidf")
    parser.add_argument(
        "--semantic-dir", type=Path,
        default=HERE / "semantic_teacher_artifacts",
        help="Directory containing <domain>_minilm.npy for MiniLM.")
    parser.add_argument("--output", type=Path,
                        default=HERE / "semantic_init_results.json")
    parser.add_argument("--artifact-dir", type=Path,
                        default=HERE / "semantic_init_artifacts")
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.output.read_text()) if args.output.exists() else {}

    for domain in args.domains:
        data = loaders.ALL_LOADERS[domain]()
        if len(data["valid_queries"]) > args.validation_cap:
            _vk = sorted(data["valid_queries"], key=cearf._stable_fraction)[
                :args.validation_cap]
            data["valid_queries"] = {k: data["valid_queries"][k] for k in _vk}
        sessions = data["train_sessions"]
        n_items = data["n_items"]
        validation = data["valid_queries"]
        test = data["test_queries"]
        if args.teacher == "minilm":
            semantic_path = args.semantic_dir / f"{domain.lower()}_minilm.npy"
            if not semantic_path.exists():
                raise FileNotFoundError(semantic_path)
            semantic = np.load(semantic_path).astype(np.float32)
        else:
            semantic_path = None
            semantic = semantic_matrix(domain, data)
        if semantic is not None and semantic.shape[0] != n_items:
            raise ValueError(
                f"semantic rows ({semantic.shape[0]}) != n_items ({n_items})")
        semantic_source_shape = list(semantic.shape) if semantic is not None else None
        if semantic is not None:
            semantic = project_semantic(semantic, n_items, 64)
        tune_sessions = hold_out_validation_targets(sessions, validation)

        exclude_seen = domain not in ("Diginetica_HID", "Tmall")
        eval_label = ("exact full catalog with seen-item masking"
                      if exclude_seen else
                      "exact full catalog, repeat consumption")
        domain_result = results.setdefault(domain, {
            "protocol": {
                "train_sessions": len(sessions),
                "validation_queries": len(validation),
                "test_queries": len(test),
                "n_items": n_items,
                "evaluation": eval_label,
                "exclude_seen": exclude_seen,
                "semantic_init": True,
                "teacher": args.teacher,
                "selection_uses_test_labels": False,
            },
            "semantic_shape": semantic_source_shape,
            "projected_semantic_shape": (
                list(semantic.shape) if semantic is not None else None),
            "models": {},
        })
        for name in args.models:
            model_result = domain_result["models"].setdefault(name, {"runs": []})
            completed = {int(r["seed"]) for r in model_result["runs"]}
            for seed in args.seeds:
                if seed in completed:
                    print(f"[SEM-INIT] {domain} {name} seed={seed} complete",
                          flush=True)
                    continue
                ckpt = args.artifact_dir / (
                    f"{domain.lower()}_{name.lower()}_{args.teacher}_seed{seed}.pt")
                rank_path = args.artifact_dir / (
                    f"{domain.lower()}_{name.lower()}_{args.teacher}_seed{seed}_ranks.npz")
                print(f"[SEM-INIT] START {domain} {name} seed={seed}", flush=True)
                model, training = train_with_semantic_init(
                    name, tune_sessions, validation, n_items, semantic, seed,
                    args.max_epochs, args.batch_size, args.patience, ckpt,
                    exclude_seen=exclude_seen)
                test_keys, ranking, inf_sec, inf_peak = predict_array(
                    model, test, n_items, topk=20,
                    batch_size=128 if name == "SR-GNN" else 256,
                    exclude_seen=exclude_seen)
                ranks = ranks_at_20(ranking, targets_for(test_keys, test))
                test_metrics = metrics_from_ranks(ranks)
                np.savez_compressed(rank_path, ranks=ranks,
                                    test_fingerprint=np.asarray(
                                        query_fingerprint(test)))
                run_record = {
                    "seed": seed,
                    "test": test_metrics,
                    "best_epoch": training["best_epoch"],
                    "training": {k: v for k, v in training.items()
                                 if k != "history"},
                    "rank_artifact": str(rank_path),
                    "teacher": args.teacher,
                    "semantic_matrix": (str(semantic_path)
                                        if semantic_path else "tfidf-svd-cache"),
                    "training_scope": "validation-target-held-out sessions",
                }
                model_result["runs"].append(run_record)
                domain_result["models"][name] = model_result
                args.output.write_text(json.dumps(results, indent=2))
                print(f"[SEM-INIT] DONE {domain} {name} seed={seed} "
                      f"R@20={test_metrics['recall@20']:.5f}", flush=True)
                del model
                gc.collect()

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\n[SEM-INIT] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
