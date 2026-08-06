#!/usr/bin/env python3
"""Leakage-audited strong-baseline suite for the CEARF-N paper."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from scipy.stats import t as student_t
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import cearf
import loaders
from paper_models import build_model, model_logits
from run_cearfn_evidence import (
    metrics_from_ranks, paired_recall_test, query_fingerprint, ranks_at_20,
    targets_for)
from validation_protocol import hold_out_validation_targets


HERE = Path(__file__).resolve().parent
MODELS = ("GRU4Rec", "SASRec", "NARM", "SR-GNN", "SIGMA-compatible")
SEEDS = (42, 123, 456)
# Datasets whose official protocol permits repeat consumption (target may
# already appear in the context). For those we must NOT mask seen items at
# scoring time, otherwise we silently cap recall at the no-repeat subset.
REPEAT_PROTOCOL_DOMAINS = frozenset({"Diginetica_HID", "Tmall"})


class PrefixDataset(Dataset):
    def __init__(self, sessions: dict, n_items: int, max_seq: int = 50):
        self.sequences = []
        self.examples = []
        for sequence0 in sessions.values():
            sequence = [int(x) for x in sequence0 if 0 < int(x) < n_items]
            if len(sequence) < 2:
                continue
            sid = len(self.sequences)
            self.sequences.append(sequence)
            for position in range(1, len(sequence)):
                self.examples.append((sid, position))
        self.max_seq = max_seq

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        sid, position = self.examples[index]
        sequence = self.sequences[sid]
        return sequence[max(0, position - self.max_seq):position], sequence[position]


def collate(batch):
    width = max(len(context) for context, _ in batch)
    contexts = torch.zeros(len(batch), width, dtype=torch.long)
    lengths = torch.empty(len(batch), dtype=torch.long)
    targets = torch.empty(len(batch), dtype=torch.long)
    for row, (context, target) in enumerate(batch):
        contexts[row, :len(context)] = torch.as_tensor(context)
        lengths[row] = len(context)
        targets[row] = target
    return contexts, lengths, targets


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Models whose forward pass produces NaNs on Apple Silicon MPS at the scale of
# the larger catalogs (Diginetica: 43k items). We pin them to CPU even when MPS
# is available so their numbers are valid; the cost is ~2x wallclock for these
# two baselines only.
MPS_UNSTABLE_MODELS = frozenset({"NARM", "SR-GNN", "SIGMA-compatible"})


def device_for(name: str) -> torch.device:
    if name in MPS_UNSTABLE_MODELS:
        return torch.device("cpu")
    return device()


def mps_memory() -> int:
    return int(torch.mps.current_allocated_memory()) if torch.backends.mps.is_available() else 0


def predict_array(model, queries: dict, n_items: int, topk: int = 20,
                  batch_size: int = 256, exclude_seen: bool = True) -> tuple[list[str], np.ndarray, float, int]:
    dev = next(model.parameters()).device
    keys = sorted(queries)
    output = np.empty((len(keys), topk), dtype=np.int32)
    peak = mps_memory()
    started = time.time()
    model.eval()
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start:start + batch_size]
        batch = [(queries[uid]["context"][-50:], queries[uid]["targets"][0])
                 for uid in batch_keys]
        contexts, lengths, _ = collate(batch)
        if model.__class__.__name__ != "SRGNN":
            contexts, lengths = contexts.to(dev), lengths.to(dev)
        with torch.no_grad():
            scores = model_logits(model, contexts, lengths)
            scores[:, 0] = -torch.inf
            if exclude_seen:
                for row, uid in enumerate(batch_keys):
                    seen = sorted(set(int(x) for x in queries[uid]["context"]
                                      if 0 < int(x) < n_items))
                    if seen:
                        scores[row, torch.as_tensor(seen, device=scores.device)] = -torch.inf
            ranking = torch.topk(scores, k=min(topk, n_items - 1), dim=1).indices
            output[start:start + len(batch_keys)] = ranking.cpu().numpy()
            peak = max(peak, mps_memory())
    return keys, output, time.time() - started, peak


def train_one(name: str, sessions: dict, validation: dict, n_items: int,
              seed: int, max_epochs: int, batch_size: int, patience: int,
              checkpoint: Path, exclude_seen: bool = True) -> tuple[torch.nn.Module, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = device_for(name)
    model = build_model(name, n_items, 64).to(dev)
    dataset = PrefixDataset(sessions, n_items)
    generator = torch.Generator().manual_seed(seed)
    effective_batch = min(batch_size, 128) if name == "SR-GNN" else batch_size
    loader = DataLoader(dataset, batch_size=effective_batch, shuffle=True,
                        collate_fn=collate, generator=generator, num_workers=0)
    # Per-architecture lr: SASRec and SIGMA destabilise at 1e-3 on large
    # vocabularies; GRU4Rec, NARM, and SR-GNN need 1e-3 to train at all.
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
            loss.backward()
            # Skip the step entirely when the loss diverged; this keeps the
            # run alive on large-vocab datasets (e.g. Diginetica 43k items)
            # where the initial cross-entropy can explode before warmup.
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                count += len(targets)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            count += len(targets)
            peak = max(peak, mps_memory())
            if step % 500 == 0:
                print(f"[BASELINE] {name} seed={seed} epoch={epoch}/{max_epochs} "
                      f"step={step}/{len(loader)} loss={loss_sum / count:.4f}", flush=True)
        valid_keys, valid_top, valid_seconds, valid_peak = predict_array(
            model, validation, n_items, topk=20,
            batch_size=128 if name == "SR-GNN" else 256,
            exclude_seen=exclude_seen)
        valid_ranks = ranks_at_20(valid_top, targets_for(valid_keys, validation))
        metrics = metrics_from_ranks(valid_ranks)
        utility = .5 * (metrics["recall@6"] + metrics["recall@20"])
        row = {"epoch": epoch, "loss": loss_sum / max(count, 1),
               "validation": metrics, "utility": utility,
               "train_epoch_seconds": time.time() - epoch_started,
               "validation_seconds": valid_seconds}
        history.append(row)
        print(f"[BASELINE] {name} seed={seed} epoch={epoch} "
              f"valid_R6={metrics['recall@6']:.6f} "
              f"valid_R20={metrics['recall@20']:.6f}", flush=True)
        candidate = (utility, metrics["recall@20"], metrics["recall@6"], -epoch)
        if best is None or candidate > best[0]:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best = (candidate, epoch, best_state, metrics)
            stale = 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": name, "seed": seed, "n_items": n_items,
                        "dim": 64, "epoch": epoch, "state_dict": best[2],
                        "validation": metrics}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
        peak = max(peak, valid_peak)
    model.load_state_dict(best[2])
    return model.eval(), {
        "best_epoch": best[1], "best_validation": best[3], "history": history,
        "train_seconds": time.time() - train_started,
        "peak_tracked_device_bytes": peak,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_examples": len(dataset),
        "batch_size": effective_batch,
    }


def train_fixed_epochs(name: str, sessions: dict, n_items: int, seed: int,
                       epochs: int, batch_size: int) -> tuple[torch.nn.Module, dict]:
    """Refit on full training data after validation has locked the epoch."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = device_for(name)
    model = build_model(name, n_items, 64).to(dev)
    dataset = PrefixDataset(sessions, n_items)
    generator = torch.Generator().manual_seed(seed)
    effective_batch = min(batch_size, 128) if name == "SR-GNN" else batch_size
    loader = DataLoader(dataset, batch_size=effective_batch, shuffle=True,
                        collate_fn=collate, generator=generator, num_workers=0)
    base_lr = {"SASRec": 5e-4, "SIGMA-compatible": 5e-4}.get(name, 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
    started = time.time(); peak = mps_memory()
    for epoch in range(1, epochs + 1):
        model.train()
        for contexts, lengths, targets in loader:
            if name != "SR-GNN":
                contexts, lengths = contexts.to(dev), lengths.to(dev)
            targets = targets.to(dev)
            logits = model_logits(model, contexts, lengths)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); peak = max(peak, mps_memory())
    return model.eval(), {"train_seconds": time.time() - started,
                          "peak_tracked_device_bytes": peak,
                          "train_examples": len(dataset),
                          "batch_size": effective_batch,
                          "parameters": sum(p.numel() for p in model.parameters())}


def aggregate(runs: list[dict]) -> dict:
    result = {}
    for metric in ("recall@6", "ndcg@6", "recall@10", "ndcg@10",
                   "recall@20", "ndcg@20"):
        values = np.asarray([run["test"][metric] for run in runs], dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[metric] = {
            "mean": float(values.mean()), "std": std,
            "ci95_half_width": float(student_t.ppf(.975, len(values) - 1)
                                      * std / math.sqrt(len(values)))
            if len(values) > 1 else 0.0,
        }
    for metric in ("parameters", "train_seconds", "inference_seconds",
                   "latency_ms_per_query", "peak_tracked_device_bytes"):
        result[metric] = float(np.mean([run[metric] for run in runs]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=["Video_Games", "Baby_Products"])
    parser.add_argument("--models", nargs="*", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-cap", type=int, default=5000)
    parser.add_argument("--train-cap", type=int, default=0)
    parser.add_argument("--test-cap", type=int, default=0)
    parser.add_argument(
        "--refit-full", action="store_true",
        help=("after validation selects the epoch, retrain from scratch on the "
              "full sessions; disabled by default because the primary protocol "
              "evaluates the leakage-safe validation-selected checkpoint"))
    parser.add_argument("--output", type=Path,
                        default=HERE / "paper_baseline_results.json")
    parser.add_argument("--artifact-dir", type=Path,
                        default=HERE / "paper_baseline_artifacts")
    args = parser.parse_args()
    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    for domain in args.domains:
        data = loaders.ALL_LOADERS[domain]()
        # Repeat-consumption datasets (HID Diginetica, Tmall) allow the target
        # to appear in the context; masking seen items would silently truncate
        # the recall ceiling on ~45% of their queries.
        exclude_seen = domain not in REPEAT_PROTOCOL_DOMAINS
        valid_keys = sorted(data["valid_queries"], key=cearf._stable_fraction)[
            :args.validation_cap]
        validation = {key: data["valid_queries"][key] for key in valid_keys}
        test = data["test_queries"]
        sessions = data["train_sessions"]
        tune_sessions = hold_out_validation_targets(sessions, validation)
        tag = "full"
        if args.train_cap:
            keys = sorted(sessions, key=cearf._stable_fraction)[:args.train_cap]
            sessions = {key: sessions[key] for key in keys}
            tag = f"smoke_tr{args.train_cap}"
        if args.test_cap:
            keys = sorted(test, key=cearf._stable_fraction)[:args.test_cap]
            test = {key: test[key] for key in keys}
            tag += f"_te{args.test_cap}"
        eval_label = ("exact full catalog with seen-item masking"
                      if exclude_seen else
                      "exact full catalog, repeat consumption (no seen-item masking)")
        domain_result = results.setdefault(domain, {
            "protocol": {
                "train_sessions": len(sessions), "validation_queries": len(validation),
                "test_queries": len(test), "n_items": data["n_items"],
                "validation_fingerprint_sha256": query_fingerprint(validation),
                "test_fingerprint_sha256": query_fingerprint(test),
                "selection_uses_test_labels": False,
                "evaluation": eval_label,
                "exclude_seen": exclude_seen,
            }, "models": {}})
        for name in args.models:
            model_result = domain_result["models"].setdefault(name, {"runs": []})
            complete = {int(run["seed"]) for run in model_result["runs"]
                        if Path(run["rank_artifact"]).exists()}
            for seed in args.seeds:
                if seed in complete:
                    print(f"[BASELINE] {domain} {name} seed={seed} complete", flush=True)
                    continue
                checkpoint = args.artifact_dir / f"{domain.lower()}_{tag}_{name.lower().replace('-', '_')}_seed{seed}.pt"
                rank_path = args.artifact_dir / f"{domain.lower()}_{tag}_{name.lower().replace('-', '_')}_seed{seed}_ranks.npz"
                print(f"[BASELINE] START {domain} {name} seed={seed}", flush=True)
                model, training = train_one(
                    name, tune_sessions, validation, data["n_items"], seed,
                    args.max_epochs, args.batch_size, args.patience, checkpoint,
                    exclude_seen=exclude_seen)
                if args.refit_full:
                    # Optional sensitivity analysis. The primary reported
                    # protocol evaluates the validation-selected checkpoint;
                    # it never restores held-out validation targets before
                    # test evaluation.
                    del model
                    gc.collect()
                    model, refit = train_fixed_epochs(
                        name, sessions, data["n_items"], seed,
                        training["best_epoch"], args.batch_size)
                    training["train_seconds"] += refit["train_seconds"]
                    training["peak_tracked_device_bytes"] = max(
                        training["peak_tracked_device_bytes"],
                        refit["peak_tracked_device_bytes"])
                    training["train_examples"] = refit["train_examples"]
                    training["parameters"] = refit["parameters"]
                test_keys, ranking, inference_seconds, inference_peak = predict_array(
                    model, test, data["n_items"], topk=20,
                    batch_size=128 if name == "SR-GNN" else 256,
                    exclude_seen=exclude_seen)
                targets = targets_for(test_keys, test)
                ranks = ranks_at_20(ranking, targets)
                np.savez_compressed(rank_path, ranks=ranks,
                                    test_fingerprint=np.asarray(query_fingerprint(test)))
                run = {
                    "seed": seed, "test": metrics_from_ranks(ranks),
                    "best_epoch": training["best_epoch"],
                    "best_validation": training["best_validation"],
                    "history": training["history"],
                    "parameters": training["parameters"],
                    "train_examples": training["train_examples"],
                    "batch_size": training["batch_size"],
                    "train_seconds": training["train_seconds"],
                    "inference_seconds": inference_seconds,
                    "latency_ms_per_query": 1000 * inference_seconds / max(len(test), 1),
                    "peak_tracked_device_bytes": max(
                        training["peak_tracked_device_bytes"], inference_peak),
                    "checkpoint": str(checkpoint), "rank_artifact": str(rank_path),
                    "training_scope": ("full-session refit at locked epoch"
                                       if args.refit_full else
                                       "leakage-safe train; validation-selected checkpoint"),
                    "provenance": ("pure-PyTorch architecture-compatible port; not official CUDA SIGMA"
                                   if name == "SIGMA-compatible" else
                                   "repository-local full-catalog reimplementation"),
                }
                cearfn_path = HERE / "cearfn_evidence_artifacts" / f"{domain.lower()}_full_seed{seed}_ranks.npz"
                if cearfn_path.exists() and len(test) == len(data["test_queries"]):
                    with np.load(cearfn_path) as saved:
                        cearfn_ranks = saved["cearfn_rank"]
                    run["paired_CEARF-N_vs_baseline_R20"] = paired_recall_test(
                        cearfn_ranks, ranks, 20, seed=seed)
                model_result["runs"].append(run)
                model_result["aggregate"] = aggregate(model_result["runs"])
                args.output.write_text(json.dumps(results, indent=2))
                print(f"[BASELINE] DONE {domain} {name} seed={seed} "
                      f"R20={run['test']['recall@20']:.6f}", flush=True)
                del model, ranking
                gc.collect()
            if model_result["runs"]:
                model_result["aggregate"] = aggregate(model_result["runs"])
            args.output.write_text(json.dumps(results, indent=2))
    print(f"[BASELINE] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
