"""
Selective-SSM session recommender — MPS-compatible, no CUDA Mamba kernels.

Inspired by Mamba4Rec (Liu et al., 2024) and SIGMA (Liu & Liu, AAAI 2025):
- Selective State Space Model (SSM) as the sequence backbone (linear-time
  recurrence, input-dependent gating) — a principled alternative to the causal
  transformer in CoDT's PGSA-Rec.
- Feature-Extract GRU + 1D conv to handle short sequences (SIGMA's key fix for
  Mamba's short-sequence weakness — exactly Rental's regime).
- Designed to DROP IN as the CoDT encoder, keeping the proven co-visitation
  fusion (session-adaptive boost cap) — the question is whether a selective-SSM
  backbone + fusion beats a transformer backbone + fusion.

Pure PyTorch (works on MPS): the selective recurrence is a plain Python scan,
which is fast enough for the small/short sequences in Rental. This avoids the
CUDA-only mamba_ssm/causal_conv1d dependencies.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

EMB_DIM = 128
SSM_STATE = 64
EPOCHS = 12
BATCH = 256
LR = 1e-3
MAX_SEQ = 50


# =============================================================================
# Selective SSM block (pure PyTorch, MPS-compatible)
# =============================================================================
class SelectiveSSM(nn.Module):
    """A minimal selective state-space block.

    Implements the Mamba-style input-dependent recurrence in pure PyTorch:
      h_t = A_t * h_{t-1} + B_t * x_t         (state update, A_t/B_t input-dependent)
      y_t = C_t * h_t                           (output projection)
      y_t = y_t * silu(gate)                    (swish gating, Mamba-style)
    where A_t = sigmoid(linear_A(x_t))  (selective: controls how much past to keep),
          B_t = softplus(linear_B(x_t)), C_t = linear_C(x_t).

    This is the input-dependent "selection" that lets the model decide per-token
    how much history to retain — the core Mamba mechanism, without CUDA kernels.
    """

    def __init__(self, d_model, d_state=SSM_STATE):
        super().__init__()
        self.d_state = d_state
        # input-dependent projections (the "selective" part)
        self.proj_A = nn.Linear(d_model, d_state)  # -> retention gate (per-state)
        self.proj_B = nn.Linear(d_model, d_state)  # -> write-in (per-state)
        self.proj_X = nn.Linear(d_model, d_state)  # project input INTO state space
        self.proj_C = nn.Linear(d_state, d_model)  # read-out: state -> model space
        self.proj_D = nn.Linear(d_model, d_model)  # residual / gate
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.15)

    def forward(self, x, padding_mask=None):
        """x: (B, L, D), padding_mask: (B, L) True at PAD positions."""
        B, L, D = x.shape
        S = self.d_state
        # input-dependent matrices
        A = torch.sigmoid(self.proj_A(x))   # (B,L,S): how much state to retain
        Bb = F.softplus(self.proj_B(x))     # (B,L,S): write-in strength
        Xs = self.proj_X(x)                 # (B,L,S): input projected to state space
        Cc = self.proj_C                    # applied later: (S)->(D)
        # sequential recurrence over the state space (S-dim, small -> fast)
        h = x.new_zeros(B, S)
        ys = []
        for t in range(L):
            h = A[:, t] * h + Bb[:, t] * Xs[:, t]   # (B,S) state update
            y_t = Cc(h)                              # (B,D) read-out
            ys.append(y_t)
        y = torch.stack(ys, dim=1)          # (B,L,D)
        if padding_mask is not None:
            y = y.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        # swish gate + residual (Mamba-style)
        y = y * F.silu(self.proj_D(x))
        return self.norm(self.drop(y) + x)


class FeatureExtractGRU(nn.Module):
    """SIGMA's short-sequence fix: 1D conv + GRU to capture local patterns that
    SSMs underfit on short sequences (Rental sessions are len 2-7)."""

    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        # conv over the sequence axis
        c = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (B,L,D)
        g, _ = self.gru(c)                                   # (B,L,D)
        out = self.norm(g + x)
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out


class SSMSessionModel(nn.Module):
    """Full Selective-SSM session encoder (drop-in replacement for PGSA-Rec).

    Pipeline: item-embed → Feature-GRU (short-seq) → Selective-SSM block(s) →
    last-position hidden → embedding-similarity head.
    """

    def __init__(self, n_items, embed_dim=EMB_DIM, n_blocks=2, d_state=SSM_STATE):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.item_embed = nn.Embedding(n_items, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.item_embed.weight)
        self.item_embed.weight.data[0].zero_()
        self.feat_gru = FeatureExtractGRU(embed_dim)
        self.blocks = nn.ModuleList([SelectiveSSM(embed_dim, d_state) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def encode(self, seq, lengths):
        B, L = seq.shape
        x = self.item_embed(seq)
        positions = torch.arange(L, device=seq.device).unsqueeze(0)
        pad_mask = positions >= lengths.to(seq.device).unsqueeze(1)
        x = self.feat_gru(x, pad_mask)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.norm(x)

    def last_hidden(self, seq, lengths):
        x = self.encode(seq, lengths)
        last_idx = (lengths.to(seq.device) - 1).clamp(min=0)
        return self.out_proj(x[torch.arange(x.size(0), device=x.device), last_idx])

    def forward(self, seq, lengths, targets, negatives):
        """Sampled-softmax training."""
        hidden = self.last_hidden(seq, lengths)
        emb = self.item_embed.weight
        pos = emb[targets]
        neg = emb[negatives]
        pos_score = (hidden * pos).sum(-1, keepdim=True)
        neg_score = torch.bmm(neg, hidden.unsqueeze(-1)).squeeze(-1)
        return torch.cat([pos_score, neg_score], dim=-1), hidden

    def score_all(self, seq, lengths):
        hidden = self.last_hidden(seq, lengths)
        return hidden @ self.item_embed.weight.t()

    def predict(self, seq, lengths):
        self.eval()
        with torch.no_grad():
            return self.score_all(seq, lengths)


# =============================================================================
# Dataset / training
# =============================================================================
class SeqDataset(Dataset):
    def __init__(self, sessions, n_items, max_seq=MAX_SEQ):
        self.ex = []
        for seq in sessions:
            seq = [x for x in seq if 1 <= x < n_items]
            for i in range(1, len(seq)):
                self.ex.append((seq[max(0, i - max_seq):i], seq[i]))

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(b):
    ml = max(len(c) for c, _ in b)
    return (torch.LongTensor([c + [0] * (ml - len(c)) for c, _ in b]),
            torch.LongTensor([len(c) for c, _ in b]),
            torch.LongTensor([t for _, t in b]))


def train_ssm(train_sessions, n_items, epochs=EPOCHS, seeds=(42, 123, 456, 789),
              embed_dim=EMB_DIM, n_blocks=2, n_neg=1024, batch=BATCH, lr=LR):
    models = []
    for seed in seeds:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        model = SSMSessionModel(n_items, embed_dim, n_blocks).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        ds = SeqDataset(train_sessions, n_items)
        loader = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate,
                            drop_last=len(ds) > batch)
        K = min(n_neg, n_items - 2)
        for ep in range(epochs):
            model.train()
            for inp, lens, tgt in loader:
                inp, lens, tgt = inp.to(DEVICE), lens.to(DEVICE), tgt.to(DEVICE)
                neg = torch.randint(1, n_items, (inp.size(0), K), device=DEVICE)
                logits, _ = model(inp, lens, tgt, neg)
                labels = torch.zeros(inp.size(0), dtype=torch.long, device=DEVICE)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
        models.append(model)
    return models


def predict_ssm(models, test_uids, test_queries, n_items, max_seq=MAX_SEQ, batch=128):
    preds = {}
    for m in models:
        m.eval()
    for bs in range(0, len(test_uids), batch):
        chunk = test_uids[bs:bs + batch]
        seqs, lens = [], []
        for uid in chunk:
            ctx = [x for x in test_queries[uid]["context"] if 1 <= x < n_items][-max_seq:]
            seqs.append(ctx); lens.append(len(ctx))
        ml = max(max(lens), 1)
        inp = torch.zeros(len(chunk), ml, dtype=torch.long, device=DEVICE)
        ln = torch.zeros(len(chunk), dtype=torch.long, device=DEVICE)
        for i, (s, l) in enumerate(zip(seqs, lens)):
            inp[i, :l] = torch.LongTensor(s); ln[i] = l
        with torch.no_grad():
            scores = torch.zeros(len(chunk), n_items, device=DEVICE)
            for m in models:
                scores = scores + m.score_all(inp, ln)
        scores = scores.cpu().numpy()
        for uid, sc, ctx in zip(chunk, scores, seqs):
            sc = sc.copy(); sc[0] = -1e9
            for c in set(ctx):
                sc[c] = -1e9
            preds[uid] = [int(x) for x in np.argsort(-sc) if int(x) != 0][:50]
    return preds
