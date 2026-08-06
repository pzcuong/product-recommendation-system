from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SelectiveSSMBlock(nn.Module):
    def __init__(self, dim: int, state: int, dropout: float, contractive: bool = False):
        super().__init__()
        self.contractive = contractive
        self.a = nn.Linear(dim, state)
        self.b = nn.Linear(dim, state)
        self.x = nn.Linear(dim, state)
        self.c = nn.Linear(state, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        if self.contractive:
            # Convex-state update: |h| remains bounded when initialized at zero.
            # Input-dependent decay preserves selectivity while preventing the
            # unbounded softplus-times-linear write used by the reference cell.
            keep = torch.exp(-F.softplus(self.a(x)))
            write = (1 - keep) * torch.sigmoid(self.b(x)) * torch.tanh(self.x(x))
        else:
            keep = torch.sigmoid(self.a(x))
            write = F.softplus(self.b(x)) * self.x(x)
        h = x.new_zeros(x.size(0), keep.size(-1))
        ys = []
        for t in range(x.size(1)):
            nxt = keep[:, t] * h + write[:, t]
            if padding_mask is not None:
                h = torch.where(padding_mask[:, t:t + 1], h, nxt)
            else:
                h = nxt
            ys.append(self.c(h))
        output_gate = torch.sigmoid(self.gate(x)) if self.contractive else F.silu(self.gate(x))
        y = torch.stack(ys, 1) * output_gate
        return self.norm(x + self.drop(y))


class FeatureGRU(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.gru(z)
        z = self.norm(x + self.drop(z))
        return z.masked_fill(padding_mask.unsqueeze(-1), 0) if padding_mask is not None else z


class SessionModel(nn.Module):
    """One output head and objective across all architecture variants."""

    VALID = {"gru4rec", "pure_ssm", "contractive_ssm", "fe_gru", "fe_gru_ssm", "sasrec"}

    def __init__(self, variant: str, n_items: int, dim: int = 64, layers: int = 1,
                 state: int = 64, dropout: float = 0.3, heads: int = 2,
                 max_seq: int = 50):
        super().__init__()
        if variant not in self.VALID:
            raise ValueError(f"unknown variant {variant!r}; expected {sorted(self.VALID)}")
        if variant == "sasrec" and dim % heads:
            raise ValueError("SASRec dimension must be divisible by heads")
        self.variant, self.max_seq = variant, max_seq
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.position = nn.Embedding(max_seq, dim) if variant == "sasrec" else None
        self.gru = nn.GRU(dim, dim, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0) if variant == "gru4rec" else None
        self.fe = FeatureGRU(dim, dropout) if variant in {"fe_gru", "fe_gru_ssm"} else None
        self.ssm = nn.ModuleList([
            SelectiveSSMBlock(dim, state, dropout, contractive=variant == "contractive_ssm")
            for _ in range(layers)
        ]) if variant in {"pure_ssm", "contractive_ssm", "fe_gru_ssm"} else None
        if variant == "sasrec":
            layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout,
                                               batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, layers)
        else:
            self.transformer = None
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, n_items)

    def forward(self, seq, lengths):
        b, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0)
        pad = pos >= lengths.unsqueeze(1)
        x = self.item_embedding(seq)
        if self.variant == "gru4rec":
            x, _ = self.gru(x)
        elif self.fe is not None:
            x = self.fe(x, pad)
        if self.ssm is not None:
            for block in self.ssm:
                x = block(x, pad)
        if self.transformer is not None:
            x = x + self.position(pos)
            causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=seq.device), 1)
            x = self.transformer(x, mask=causal, src_key_padding_mask=pad)
        idx = (lengths - 1).clamp(min=0)
        hidden = self.norm(x[torch.arange(b, device=seq.device), idx])
        return self.output(hidden)


def build_model(variant: str, n_items: int, config: dict) -> SessionModel:
    keys = {"dim", "layers", "state", "dropout", "heads", "max_seq"}
    return SessionModel(variant, n_items, **{k: v for k, v in config.items() if k in keys})


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
