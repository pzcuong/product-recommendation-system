from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import write_json
from .metrics import aggregate, per_query
from .models import build_model, parameter_count


class PrefixDataset(Dataset):
    def __init__(self, sessions, n_items, max_seq=50):
        self.examples = []
        for seq in sessions.values():
            seq = [x for x in seq if 0 < x < n_items]
            for i in range(1, len(seq)):
                self.examples.append((seq[max(0, i - max_seq):i], seq[i]))

    def __len__(self): return len(self.examples)
    def __getitem__(self, index): return self.examples[index]


def collate(batch):
    length = max(len(x[0]) for x in batch)
    return (torch.tensor([x[0] + [0] * (length - len(x[0])) for x in batch]),
            torch.tensor([len(x[0]) for x in batch]), torch.tensor([x[1] for x in batch]))


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_from(name="auto"):
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def rank(model, queries, n_items, device, max_seq=50, batch_size=256, topk=20):
    model.eval(); ids = sorted(queries); result = {}
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            chunk = ids[start:start + batch_size]
            seqs = [[x for x in queries[u]["context"] if 0 < x < n_items][-max_seq:] for u in chunk]
            lengths = [len(x) for x in seqs]
            width = max(max(lengths, default=0), 1)
            inp = torch.zeros(len(chunk), width, dtype=torch.long, device=device)
            for i, seq in enumerate(seqs):
                if seq: inp[i, :len(seq)] = torch.tensor(seq, device=device)
            scores = model(inp, torch.tensor(lengths, device=device)).cpu().numpy()
            for uid, score, context in zip(chunk, scores, seqs):
                score[0] = -np.inf
                score[list(set(context))] = -np.inf
                take = min(topk, max(1, n_items - 1))
                idx = np.argpartition(-score, take - 1)[:take]
                result[uid] = [int(x) for x in idx[np.argsort(-score[idx])] if x != 0]
    return result


def evaluate(model, queries, train, n_items, device, config):
    rankings = rank(model, queries, n_items, device, config.get("max_seq", 50),
                    config.get("eval_batch_size", 256), config.get("topk", 20))
    train_items = {x for seq in train.values() for x in seq}
    rows = per_query(rankings, queries, train_items)
    return aggregate(rows), rankings, rows


def git_revision():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return None


def train_run(variant, train, validation, test, n_items, config, seed, output_dir,
              draw_seed=None, scale=None, trial_id=None):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed); device = device_from(config.get("device", "auto"))
    model = build_model(variant, n_items, config).to(device)
    dataset = PrefixDataset(train, n_items, config.get("max_seq", 50))
    if not dataset: raise ValueError("training split has no next-item examples")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 256), shuffle=True,
                        collate_fn=collate, generator=generator, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-3),
                                  weight_decay=config.get("weight_decay", 1e-5))
    best_state, best_metric, best_epoch, stale = None, -1.0, 0, 0
    history, started = [], time.time()
    selection = config.get("selection_metric", "ndcg@20")
    for epoch in range(1, config.get("max_epochs", 30) + 1):
        model.train(); losses = []
        for seq, lengths, target in loader:
            seq, lengths, target = seq.to(device), lengths.to(device), target.to(device)
            loss = F.cross_entropy(model(seq, lengths), target)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0))
            optimizer.step(); losses.append(loss.item())
        val_metrics, _, _ = evaluate(model, validation, train, n_items, device, config)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **val_metrics})
        if val_metrics[selection] > best_metric + config.get("min_delta", 1e-5):
            best_metric, best_epoch, stale = val_metrics[selection], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else: stale += 1
        if stale >= config.get("patience", 5): break
    model.load_state_dict(best_state)
    test_metrics, rankings, rows = evaluate(model, test, train, n_items, device, config)
    artifact = {"schema": 1, "status": "complete", "variant": variant, "seed": seed,
                "draw_seed": draw_seed, "scale": scale, "trial_id": trial_id,
                "config": config, "parameters": parameter_count(model), "best_epoch": best_epoch,
                "best_validation_metric": best_metric, "selection_metric": selection,
                "test_metrics": test_metrics, "history": history,
                "training_seconds": time.time() - started, "device": str(device),
                "git_revision": git_revision(), "pid": os.getpid()}
    write_json(output_dir / "run.json", artifact)
    write_json(output_dir / "rankings.json", rankings)
    write_json(output_dir / "per_query.json", rows)
    torch.save({"state_dict": model.state_dict(), "variant": variant, "n_items": n_items,
                "config": config}, output_dir / "checkpoint.pt")
    return artifact


def load_config(path): return json.loads(Path(path).read_text())
