"""Dense MPS-compatible NARM backbone for official session recommendation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class NARMConfig:
    n_items: int = 43098
    dim: int = 100
    batch_size: int = 256
    epochs: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.25
    seed: int = 42


class PrefixExamples(Dataset):
    def __init__(self, contexts, targets, cap: int | None = None):
        count = min(len(contexts), cap) if cap else len(contexts)
        self.contexts = contexts[:count]
        self.targets = targets[:count]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.contexts[index], int(self.targets[index])


def collate_prefixes(batch):
    length = max(len(context) for context, _ in batch)
    contexts = torch.zeros(len(batch), length, dtype=torch.long)
    lengths = torch.empty(len(batch), dtype=torch.long)
    targets = torch.empty(len(batch), dtype=torch.long)
    for row, (context, target) in enumerate(batch):
        contexts[row, :len(context)] = torch.as_tensor(context)
        lengths[row] = len(context)
        targets[row] = target
    return contexts, lengths, targets


class NARM(nn.Module):
    def __init__(self, config: NARMConfig):
        super().__init__()
        dim = config.dim
        self.config = config
        self.item = nn.Embedding(config.n_items, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.local = nn.Linear(dim, dim, bias=False)
        self.global_state = nn.Linear(dim, dim, bias=False)
        self.attention = nn.Linear(dim, 1, bias=False)
        self.combine = nn.Linear(2 * dim, dim, bias=False)
        self.session_norm = nn.LayerNorm(dim)
        self.input_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / np.sqrt(self.config.dim)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def encode(self, contexts, lengths):
        embedded = self.input_dropout(self.item(contexts))
        outputs, _ = self.gru(embedded)
        rows = torch.arange(contexts.size(0), device=contexts.device)
        last = outputs[rows, lengths - 1]
        energy = self.attention(torch.sigmoid(
            self.local(outputs) + self.global_state(last)[:, None, :]
        )).squeeze(-1)
        positions = torch.arange(contexts.size(1), device=contexts.device)[None, :]
        mask = positions < lengths[:, None]
        energy = energy.masked_fill(~mask, -1e9)
        alpha = torch.softmax(energy, dim=1)
        local = torch.sum(alpha[..., None] * outputs, dim=1)
        session = self.session_norm(
            self.combine(torch.cat([local, last], dim=-1)))
        return F.normalize(self.output_dropout(session), dim=-1)

    def logits(self, contexts, lengths):
        session = self.encode(contexts, lengths)
        catalog = F.normalize(self.item.weight, dim=-1)
        scores = 20.0 * (session @ catalog.T)
        scores[:, 0] = -1e9
        return scores


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def expand_sessions(sessions):
    contexts = []
    targets = []
    for sequence in sessions.values():
        valid = [int(item) for item in sequence if int(item) > 0]
        for position in range(1, len(valid)):
            contexts.append(valid[:position])
            targets.append(valid[position])
    return contexts, targets


def load_narm(checkpoint, device=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = NARMConfig(**payload["config"])
    model = NARM(config)
    model.load_state_dict(payload["state_dict"])
    dev = torch.device(device) if device else default_device()
    return model.to(dev).eval(), payload.get("history", [])


def train_narm(contexts, targets, config=NARMConfig(), cap=None,
               device=None, checkpoint=None, initial_model=None,
               initial_history=None):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    dev = torch.device(device) if device else default_device()
    model = (initial_model if initial_model is not None
             else NARM(config)).to(dev)
    dataset = PrefixExamples(contexts, targets, cap)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        collate_fn=collate_prefixes, generator=generator,
                        num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = list(initial_history or [])
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        seen = 0
        for step, (batch_context, lengths, batch_targets) in enumerate(loader, 1):
            batch_context = batch_context.to(dev)
            lengths = lengths.to(dev)
            batch_targets = batch_targets.to(dev)
            loss = criterion(model.logits(batch_context, lengths), batch_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch_targets)
            seen += len(batch_targets)
            if step % 500 == 0:
                print(f"  [NARM-MPS] epoch={epoch + 1}/{config.epochs} "
                      f"step={step}/{len(loader)} loss={total / seen:.4f}",
                      flush=True)
        epoch_loss = total / max(seen, 1)
        history.append(epoch_loss)
        print(f"  [NARM-MPS] epoch={epoch + 1}/{config.epochs} "
              f"loss={epoch_loss:.4f}", flush=True)
        if checkpoint:
            torch.save({"state_dict": model.state_dict(),
                        "config": config.__dict__, "history": history}, checkpoint)
    return model.eval(), history


@torch.no_grad()
def predict_narm(model: NARM,
                 queries: Mapping[str, Mapping[str, Sequence[int]]],
                 topk: int = 20, batch_size: int = 512):
    device = next(model.parameters()).device
    keys = list(queries)
    predictions = {}
    model.eval()
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start:start + batch_size]
        batch = [(queries[uid]["context"], queries[uid]["targets"][0])
                 for uid in batch_keys]
        contexts, lengths, _ = collate_prefixes(batch)
        scores = model.logits(contexts.to(device), lengths.to(device))
        ranking = torch.topk(scores, min(topk, scores.shape[1] - 1),
                             dim=1).indices.cpu().numpy()
        for uid, items in zip(batch_keys, ranking):
            predictions[str(uid)] = [int(item) for item in items]
    return predictions
