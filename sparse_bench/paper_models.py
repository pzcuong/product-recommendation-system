"""Strong full-catalog baselines used by the CEARF-N paper protocol.

Every model consumes the same padded prefix tensor and returns logits over the
same complete item vocabulary. The implementations intentionally share only
data and evaluation code; no CEARF-N predictions or test labels enter training.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from narm_mps import NARM, NARMConfig
from sigma_model import SIGMA


class GRU4Rec(nn.Module):
    def __init__(self, n_items: int, dim: int = 64, dropout: float = .2):
        super().__init__()
        self.item = nn.Embedding(n_items, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, n_items)

    def logits(self, contexts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(self.dropout(self.item(contexts)))
        rows = torch.arange(len(contexts), device=contexts.device)
        state = hidden[rows, lengths - 1]
        return self.output(self.norm(self.dropout(state)))


class CausalSASRec(nn.Module):
    def __init__(self, n_items: int, dim: int = 64, heads: int = 2,
                 layers: int = 2, dropout: float = .2, max_seq: int = 50):
        super().__init__()
        self.item = nn.Embedding(n_items, dim, padding_idx=0)
        self.position = nn.Embedding(max_seq, dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(dim)

    def logits(self, contexts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        width = contexts.shape[1]
        positions = torch.arange(width, device=contexts.device)[None, :]
        hidden = self.item(contexts) * self.scale + self.position(positions)
        hidden = self.dropout(hidden)
        padding = positions >= lengths[:, None]
        causal = torch.triu(torch.ones(width, width, dtype=torch.bool,
                                      device=contexts.device), diagonal=1)
        hidden = self.encoder(hidden, mask=causal,
                              src_key_padding_mask=padding)
        rows = torch.arange(len(contexts), device=contexts.device)
        state = self.norm(hidden[rows, lengths - 1])
        return state @ self.item.weight.T


class SRGNN(nn.Module):
    """SR-GNN core with directed in/out propagation and attentive readout."""

    def __init__(self, n_items: int, dim: int = 64, steps: int = 1):
        super().__init__()
        self.item = nn.Embedding(n_items, dim, padding_idx=0)
        self.input_in = nn.Linear(dim, dim, bias=True)
        self.input_out = nn.Linear(dim, dim, bias=True)
        self.gru = nn.GRUCell(2 * dim, dim)
        self.linear_one = nn.Linear(dim, dim, bias=True)
        self.linear_two = nn.Linear(dim, dim, bias=True)
        self.linear_three = nn.Linear(dim, 1, bias=False)
        self.transform = nn.Linear(2 * dim, dim, bias=True)
        self.steps = steps
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.item.embedding_dim)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)
        with torch.no_grad():
            self.item.weight[0].zero_()

    @staticmethod
    def _graphs(contexts: torch.Tensor, lengths: torch.Tensor):
        contexts_np = contexts.numpy()
        lengths_np = lengths.numpy()
        batch = len(contexts_np)
        unique_rows = []
        aliases = []
        width = 1
        for row, length in zip(contexts_np, lengths_np):
            sequence = [int(x) for x in row[:int(length)] if int(x) > 0]
            unique = list(dict.fromkeys(sequence))
            lookup = {item: index for index, item in enumerate(unique)}
            unique_rows.append(unique)
            aliases.append([lookup[item] for item in sequence])
            width = max(width, len(unique))
        nodes = np.zeros((batch, width), dtype=np.int64)
        adjacency_in = np.zeros((batch, width, width), dtype=np.float32)
        adjacency_out = np.zeros((batch, width, width), dtype=np.float32)
        alias = np.zeros((batch, contexts.shape[1]), dtype=np.int64)
        sequence_mask = np.zeros((batch, contexts.shape[1]), dtype=np.float32)
        node_mask = np.zeros((batch, width), dtype=np.float32)
        for index, (unique, path) in enumerate(zip(unique_rows, aliases)):
            nodes[index, :len(unique)] = unique
            node_mask[index, :len(unique)] = 1.0
            alias[index, :len(path)] = path
            sequence_mask[index, :len(path)] = 1.0
            counts = np.zeros((len(unique), len(unique)), dtype=np.float32)
            for left, right in zip(path[:-1], path[1:]):
                counts[left, right] += 1.0
            in_degree = counts.sum(axis=0, keepdims=True)
            out_degree = counts.sum(axis=1, keepdims=True)
            adjacency_in[index, :len(unique), :len(unique)] = (
                counts / np.maximum(in_degree, 1.0))
            adjacency_out[index, :len(unique), :len(unique)] = (
                counts.T / np.maximum(out_degree.T, 1.0))
        return nodes, adjacency_in, adjacency_out, alias, sequence_mask, node_mask

    def logits(self, contexts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Graph construction stays on CPU and is deterministic; tensors are
        # transferred once after construction to avoid MPS scalar assignment.
        device = self.item.weight.device
        graph = self._graphs(contexts.cpu(), lengths.cpu())
        nodes, adjacency_in, adjacency_out, alias, sequence_mask, node_mask = [
            torch.from_numpy(value).to(device) for value in graph]
        hidden = self.item(nodes)
        for _ in range(self.steps):
            incoming = torch.bmm(adjacency_in, self.input_in(hidden))
            outgoing = torch.bmm(adjacency_out, self.input_out(hidden))
            inputs = torch.cat([incoming, outgoing], dim=-1)
            hidden = self.gru(inputs.reshape(-1, inputs.shape[-1]),
                              hidden.reshape(-1, hidden.shape[-1])).reshape_as(hidden)
            hidden = hidden * node_mask[..., None]
        sequence_hidden = torch.gather(
            hidden, 1, alias[..., None].expand(-1, -1, hidden.shape[-1]))
        rows = torch.arange(len(lengths), device=device)
        local = sequence_hidden[rows, lengths.to(device) - 1]
        alpha = self.linear_three(torch.sigmoid(
            self.linear_one(sequence_hidden) + self.linear_two(local)[:, None, :]
        )).squeeze(-1)
        alpha = alpha * sequence_mask
        global_state = torch.sum(alpha[..., None] * sequence_hidden, dim=1)
        state = self.transform(torch.cat([global_state, local], dim=-1))
        return state @ self.item.weight.T


def build_model(name: str, n_items: int, dim: int = 64) -> nn.Module:
    if name == "GRU4Rec":
        return GRU4Rec(n_items, dim)
    if name == "SASRec":
        return CausalSASRec(n_items, dim)
    if name == "NARM":
        return NARM(NARMConfig(n_items=n_items, dim=dim, dropout=.25))
    if name == "SR-GNN":
        return SRGNN(n_items, dim)
    if name == "SIGMA-compatible":
        return SIGMA(n_items, embed_dim=dim, n_layers=1, d_state=32)
    raise KeyError(name)


def model_logits(model: nn.Module, contexts: torch.Tensor,
                 lengths: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "logits"):
        return model.logits(contexts, lengths)
    return model(contexts, lengths)
