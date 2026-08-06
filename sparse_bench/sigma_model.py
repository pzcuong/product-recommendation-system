"""
SIGMA (Selective Gated Mamba) — pure PyTorch implementation for MPS.

Faithful port of the official SIGMA architecture (AAAI 2025):
https://github.com/ziwliu8/SIGMA/blob/main/model/gated_mamba.py

Three components:
1. PF-Mamba (Partially Flipped Mamba): bidirectional via partial sequence flip
2. Dense Selective Gate: sigmoid + silu gating on each Mamba path
3. Feature Extract GRU: Conv1d + GRU for short-sequence patterns

Combining weights initialized to [mamba=0.8, flipped=0.1, gru=0.1] (from source).

Differences from official code:
- Uses pure-PyTorch SelectiveSSMBlock (no CUDA mamba_ssm dependency)
- Full-softmax head ( Rental vocab is small enough)
- Pure PyTorch → runs on MPS/CPU
"""

from __future__ import annotations
import math, random
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MAX_SEQ = 50


class SelectiveSSMCore(nn.Module):
    """The selective SSM scan (replaces mamba_ssm.Mamba). Pure PyTorch."""

    def __init__(self, d_model, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.d_inner = d_model * expand
        # input convolution: d_model → d_inner (causal)
        self.conv = nn.Conv1d(d_model, self.d_inner, kernel_size=d_conv,
                              padding=d_conv - 1, groups=1)
        # input-dependent projections
        self.proj_A = nn.Linear(self.d_inner, d_state)
        self.proj_B = nn.Linear(self.d_inner, d_state)
        self.proj_X = nn.Linear(self.d_inner, d_state)
        self.proj_C = nn.Linear(d_state, self.d_inner)
        self.proj_D = nn.Linear(self.d_inner, self.d_inner)
        self.in_proj = nn.Linear(d_model, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        """x: (B, L, D) → (B, L, D)"""
        B, L, D = x.shape
        # causal conv on raw input (d_model → d_inner)
        hc = self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)  # (B, L, d_inner)
        hc = F.silu(hc)
        # selective parameters
        A = torch.sigmoid(self.proj_A(hc))   # (B, L, S)
        Bb = F.softplus(self.proj_B(hc))     # (B, L, S)
        Xs = self.proj_X(hc)                 # (B, L, S)
        # sequential scan
        S = self.d_state
        state = x.new_zeros(B, S)
        ys = []
        for t in range(L):
            state = A[:, t] * state + Bb[:, t] * Xs[:, t]
            y_t = self.proj_C(state)  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        # swish gate
        y = y * F.silu(self.proj_D(hc))
        # output projection + residual
        y = self.out_proj(y)
        return self.norm(y + x)


class GMambaBlock(nn.Module):
    """SIGMA's GMambaBlock: PF-Mamba + Dense Selective Gate + Feature GRU."""

    def __init__(self, d_model, d_state=32, max_seq=MAX_SEQ):
        super().__init__()
        self.flip_len = 45  # from official code (MAX_ITEM_LIST_LENGTH=50)
        self.mamba = SelectiveSSMCore(d_model, d_state)
        # Dense Selective Gate
        self.selective_gate_sig = nn.Sequential(nn.Sigmoid(), nn.Linear(d_model, d_model))
        self.selective_gate_silu = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model))
        self.selective_gate_drop = nn.Sequential(nn.Dropout(0.2))
        # Feature Extract GRU
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.gru = nn.GRU(d_model, d_model, num_layers=1, bias=False, batch_first=True)
        # Combining weights [mamba, flipped, gru] — init from official code
        self.combining_weights = nn.Parameter(torch.tensor([0.1, 0.1, 0.8]))
        # projection
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x, padding_mask=None):
        B, L, D = x.shape
        flip = min(self.flip_len, L)

        # PF-Mamba: forward + partially-flipped
        flipped = x.clone()
        flipped[:, :flip] = x[:, :flip].flip(dims=[1])
        mamba_out = self.mamba(x, padding_mask)
        mamba_out_f = self.mamba(flipped, padding_mask)

        # Dense Selective Gate
        h1 = self.selective_gate_silu(x) + self.selective_gate_sig(x)
        h1 = self.selective_gate_drop(h1)
        mamba_out = mamba_out * h1 + mamba_out
        mamba_out_f = mamba_out_f * h1 + mamba_out_f  # reuse h1 (from official code)

        # Feature Extract GRU
        g = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        gru_out, _ = self.gru(g)

        # Weighted combination
        w = F.softmax(self.combining_weights, dim=0)
        combined = (w[2] * mamba_out + w[1] * mamba_out_f + w[0] * gru_out)
        return self.projection(combined)


class SIGMALayer(nn.Module):
    """One SIGMA layer: GMambaBlock + FFN + residuals."""

    def __init__(self, d_model, d_state=32, dropout=0.2):
        super().__init__()
        self.gmamba = GMambaBlock(d_model, d_state)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        out = self.gmamba(x, padding_mask)
        x = self.norm1(out + x)
        x = self.norm2(self.ffn(x) + x)
        return x


class SIGMA(nn.Module):
    """Full SIGMA model for session recommendation (pure PyTorch)."""

    def __init__(self, n_items, embed_dim=64, n_layers=1, d_state=32, dropout=0.2, max_seq=MAX_SEQ):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.item_embed = nn.Embedding(n_items, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.item_embed.weight)
        self.item_embed.weight.data[0].zero_()
        self.pe = nn.Embedding(max_seq, embed_dim)
        self.layers = nn.ModuleList([
            SIGMALayer(embed_dim, d_state, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_items)  # full softmax (Rental vocab is small)

    def forward(self, seq, lengths=None):
        B, L = seq.shape
        x = self.item_embed(seq) + self.pe(torch.arange(L, device=seq.device).unsqueeze(0))
        pad_mask = None
        if lengths is not None:
            pos = torch.arange(L, device=seq.device).unsqueeze(0)
            pad_mask = pos >= lengths.to(seq.device).unsqueeze(1)
        for layer in self.layers:
            x = layer(x, pad_mask)
        x = self.norm(x)
        if lengths is not None:
            last_idx = (lengths.to(seq.device) - 1).clamp(min=0)
            return self.head(x[torch.arange(B, device=x.device), last_idx])
        return self.head(x[:, -1])


# =============================================================================
# Training + evaluation (same interface as SSM/GRU4Rec)
# =============================================================================
class SeqDataset(Dataset):
    def __init__(self, sessions, n_items, max_seq=MAX_SEQ):
        self.ex = []
        for seq in sessions:
            seq = [x for x in seq if 1 <= x < n_items]
            for i in range(1, len(seq)):
                self.ex.append((seq[max(0, i - max_seq):i], seq[i]))
    def __len__(self): return len(self.ex)
    def __getitem__(self, i): return self.ex[i]


def collate(b):
    ml = max(len(c) for c, _ in b)
    return (torch.LongTensor([c + [0] * (ml - len(c)) for c, _ in b]),
            torch.LongTensor([len(c) for c, _ in b]),
            torch.LongTensor([t for _, t in b]))


def train_sigma(train_sessions, n_items, epochs=10, seeds=(42, 123, 456, 789),
                embed_dim=64, lr=1e-3, bs=256):
    models = []
    for seed in seeds:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        model = SIGMA(n_items, embed_dim=embed_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        ds = SeqDataset(train_sessions, n_items)
        loader = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate,
                            drop_last=len(ds) > bs)
        for ep in range(epochs):
            model.train()
            for inp, lens, tgt in loader:
                inp, lens, tgt = inp.to(DEVICE), lens.to(DEVICE), tgt.to(DEVICE)
                logits = model(inp, lens)
                loss = F.cross_entropy(logits, tgt)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sch.step()
        models.append(model)
    return models


def predict_sigma(models, test_uids, test_queries, n_items, max_seq=MAX_SEQ, batch=128):
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
                scores = scores + model_predict(m, inp, ln)
        scores = scores.cpu().numpy()
        for uid, sc, ctx in zip(chunk, scores, seqs):
            sc = sc.copy(); sc[0] = -1e9
            for c in set(ctx): sc[c] = -1e9
            preds[uid] = [int(x) for x in np.argsort(-sc) if int(x) != 0][:50]
    return preds


def model_predict(model, inp, ln):
    """Ensemble helper: predict and return scores."""
    with torch.no_grad():
        return model(inp, ln)
