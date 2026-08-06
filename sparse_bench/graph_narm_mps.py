"""MPS-native global-graph NARM without ``torch.sparse.mm``.

The edge-index aggregation is algebraically equivalent to ``A @ H`` for a
COO adjacency, but uses ``index_add_``, which has forward/backward MPS support.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from narm_mps import NARM, NARMConfig, default_device


DEFAULT_ADJ = (Path(__file__).resolve().parents[1] / "reference_repos" /
               "MGCOT" / "diginetica" / "adj_global.npz")


def load_edges(path=DEFAULT_ADJ):
    adjacency = sp.load_npz(path).tocoo()
    return (torch.as_tensor(adjacency.col.astype(np.int64)),
            torch.as_tensor(adjacency.row.astype(np.int64)),
            torch.as_tensor(adjacency.data.astype(np.float32)))


class GraphNARM(NARM):
    def __init__(self, config: NARMConfig, edge_src, edge_dst, edge_weight):
        super().__init__(config)
        self.graph_linear = nn.Linear(config.dim, config.dim, bias=False)
        self.graph_gate = nn.Parameter(torch.tensor(-2.1972246))  # sigmoid=.1
        self.register_buffer("edge_src", edge_src.long())
        self.register_buffer("edge_dst", edge_dst.long())
        self.register_buffer("edge_weight", edge_weight.float())
        with torch.no_grad():
            nn.init.eye_(self.graph_linear.weight)

    def graph_catalog(self):
        propagated = self.graph_linear(self.item.weight)
        aggregated = torch.zeros_like(propagated)
        aggregated.index_add_(
            0, self.edge_dst,
            propagated[self.edge_src] * self.edge_weight[:, None])
        aggregated = F.normalize(aggregated, dim=-1)
        gate = torch.sigmoid(self.graph_gate)
        return self.item.weight + gate * aggregated

    def encode_with_catalog(self, contexts, lengths, catalog):
        embedded = self.input_dropout(catalog[contexts])
        outputs, _ = self.gru(embedded)
        rows = torch.arange(contexts.size(0), device=contexts.device)
        last = outputs[rows, lengths - 1]
        energy = self.attention(torch.sigmoid(
            self.local(outputs) + self.global_state(last)[:, None, :]
        )).squeeze(-1)
        positions = torch.arange(contexts.size(1), device=contexts.device)[None, :]
        mask = positions < lengths[:, None]
        alpha = torch.softmax(energy.masked_fill(~mask, -1e9), dim=1)
        local = torch.sum(alpha[..., None] * outputs, dim=1)
        session = self.session_norm(
            self.combine(torch.cat([local, last], dim=-1)))
        return F.normalize(self.output_dropout(session), dim=-1)

    def encode(self, contexts, lengths):
        return self.encode_with_catalog(contexts, lengths, self.graph_catalog())

    def logits(self, contexts, lengths):
        catalog = self.graph_catalog()
        session = self.encode_with_catalog(contexts, lengths, catalog)
        scores = 20.0 * (session @ F.normalize(catalog, dim=-1).T)
        scores[:, 0] = -1e9
        return scores


def from_narm_checkpoint(checkpoint, adjacency=DEFAULT_ADJ, device=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = NARMConfig(**payload["config"])
    model = GraphNARM(config, *load_edges(adjacency))
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    allowed = {"graph_gate", "graph_linear.weight", "edge_src", "edge_dst",
               "edge_weight"}
    if set(missing) - allowed or unexpected:
        raise RuntimeError(f"incompatible NARM checkpoint: {missing=} {unexpected=}")
    dev = torch.device(device) if device else default_device()
    return model.to(dev).eval(), payload.get("history", [])


def load_graph_narm(checkpoint, adjacency=DEFAULT_ADJ, device=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = NARMConfig(**payload["config"])
    model = GraphNARM(config, *load_edges(adjacency))
    # Edges are derived inputs, not learned state; tolerate checkpoints made
    # before buffers were excluded and always use the requested adjacency.
    state = {k: v for k, v in payload["state_dict"].items()
             if not k.startswith("edge_")}
    model.load_state_dict(state, strict=False)
    dev = torch.device(device) if device else default_device()
    return model.to(dev).eval(), payload.get("history", [])
