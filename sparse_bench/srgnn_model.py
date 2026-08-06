"""
SR-GNN-style Session-Graph GNN for session-based recommendation.

Faithful reimplementation of the core SR-GNN architecture (Wu et al., AAAI 2019):
each session is modeled as a directed transition graph; a Gated Graph Neural
Network (GGNN, GRU-updated) propagates over it; the session representation is an
attention-pooled combination of the last node's state (local intent) and all
nodes' states (global intent); items are scored by dot product with the learned
item embeddings (scales to any vocab).

Reference: https://github.com/CRIPAC-DIG/SR-GNN

This is the component CoDT lacks — the hypothesis is that graph-structured
propagation captures item-transition structure that a sequential transformer
(SASRec/PGSA) misses, which is exactly why SR-GNN beats them on session-rec
benchmarks. Here we test that hypothesis on Rental.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

EMB_DIM = 64
HIDDEN_DIM = 64
EPOCHS = 20
BATCH = 100
LR = 1e-3
N_NEG = 2048          # sampled-softmax negatives
MAX_NODES = 50        # cap session-graph size


# =============================================================================
# GGNN cell: GRU-based node-state update over the (batched) adjacency.
# =============================================================================
class GGNNCell(nn.Module):
    """Single gated-graph propagation step.

    Given node states H (B, N, D) and a normalized adjacency A (B, N, N),
    produce updated states by:
        a = A @ H          (aggregate neighbour states, (B,N,D))
        h' = GRU(a, H)     (gated update per node)
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, H, A):
        # H: (B, N, D), A: (B, N, N)
        a = torch.bmm(A, H)               # (B, N, D)
        B, N, D = a.shape
        a_flat = a.reshape(B * N, D)
        h_flat = H.reshape(B * N, D)
        out = self.gru(a_flat, h_flat)    # (B*N, D)
        return out.reshape(B, N, D)


# =============================================================================
# SR-GNN model
# =============================================================================
class SRGNN(nn.Module):
    def __init__(self, n_items, embed_dim=EMB_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # shared item embedding (input feature + output scoring)
        self.item_embed = nn.Embedding(n_items, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.item_embed.weight)
        self.item_embed.weight.data[0].zero_()
        # project input embedding into GGNN hidden space
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.ggnn = GGNNCell(hidden_dim)
        # session-attention: combines local (last node) + global (all nodes)
        self.attn_q = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # project session rep back into item-embed space for scoring
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    # ---- build the normalized in/out adjacency for a batch of sessions ----
    @staticmethod
    def build_adjacency(sessions: List[List[int]], device=DEVICE) -> torch.Tensor:
        """Returns A (B, Nmax, Nmax) with concatenated [A_in | A_out] stacked
        along the last dim... but bmm needs square. SR-GNN uses A = [A_in; A_out]
        vertically concatenated then multiplied — we approximate by summing
        A_in + A_out (a common simplification) for a single matrix.
        Each session's adjacency is row-normalized."""
        B = len(sessions)
        Nmax = max((len(set(s)) for s in sessions), default=1)
        Nmax = max(Nmax, 1)
        A = torch.zeros(B, Nmax, Nmax, device=device)
        for bi, sess in enumerate(sessions):
            uniq = list(dict.fromkeys(sess))  # preserve first-occurrence order
            idx = {v: i for i, v in enumerate(uniq)}
            n = len(uniq)
            # directed edges v_t -> v_{t+1}
            in_deg = [0.0] * n
            out_deg = [0.0] * n
            edges = []
            for a, b in zip(sess, sess[1:]):
                if a == b:
                    continue
                ia, ib = idx[a], idx[b]
                edges.append((ia, ib))
                out_deg[ia] += 1
                in_deg[ib] += 1
            for ia, ib in edges:
                if out_deg[ia] > 0:
                    A[bi, ia, ib] += 1.0 / out_deg[ia]   # out-normalized (A_out row)
                if in_deg[ib] > 0:
                    A[bi, ib, ia] += 1.0 / in_deg[ib]    # in-normalized  (A_in row)
        return A

    def session_rep(self, sessions: List[List[int]]) -> torch.Tensor:
        """Compute the SR-GNN session representation for a batch of sessions."""
        A = self.build_adjacency(sessions)
        B, Nmax, _ = A.shape
        # node input features = unique item embeddings per session
        H = torch.zeros(B, Nmax, self.hidden_dim, device=A.device)
        node_masks = torch.zeros(B, Nmax, device=A.device)
        for bi, sess in enumerate(sessions):
            uniq = list(dict.fromkeys(sess))
            for i, v in enumerate(uniq):
                H[bi, i] = self.input_proj(self.item_embed.weight[v])
                node_masks[bi, i] = 1.0
        # GGNN propagation (single layer; SR-GNN uses 1 step empirically OK)
        H = self.ggnn(H, A)
        # local intent = last-occurring item's hidden state
        local = torch.zeros(B, self.hidden_dim, device=A.device)
        for bi, sess in enumerate(sessions):
            if sess:
                uniq = list(dict.fromkeys(sess))
                local[bi] = H[bi, len(uniq) - 1]
        # global intent = attention-pool over nodes, query = local
        q = self.attn_q(H).squeeze(-1)            # (B, Nmax)
        q = q.masked_fill(node_masks == 0, -1e9)
        alpha = F.softmax(q, dim=1).unsqueeze(-1)  # (B, Nmax, 1)
        global_ = (alpha * H).sum(dim=1)          # (B, D)
        sg = local + global_
        return self.out_proj(sg)                  # (B, embed_dim)

    def forward(self, sessions, targets, negatives):
        """Sampled-softmax training. Returns logits over [pos, K neg]."""
        sg = self.session_rep(sessions)           # (B, E)
        emb = self.item_embed.weight              # (n_items, E)
        pos = emb[targets]                        # (B, E)
        neg = emb[negatives]                      # (B, K, E)
        pos_score = (sg * pos).sum(-1, keepdim=True)
        neg_score = torch.bmm(neg, sg.unsqueeze(-1)).squeeze(-1)
        return torch.cat([pos_score, neg_score], dim=-1)

    def score_all(self, sessions):
        sg = self.session_rep(sessions)
        return sg @ self.item_embed.weight.t()


# =============================================================================
# Dataset / training / inference
# =============================================================================
class SessDataset(Dataset):
    """Each session -> all next-item examples (prefix -> next)."""

    def __init__(self, sessions, n_items, max_nodes=MAX_NODES):
        self.ex = []
        for seq in sessions:
            seq = [x for x in seq if 1 <= x < n_items][:max_nodes]
            for i in range(1, len(seq)):
                self.ex.append((seq[:i], seq[i]))

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


class PopSampler:
    def __init__(self, n_items, item_freq):
        items = np.arange(1, n_items)
        f = np.array([item_freq.get(i, 0) for i in items], dtype=np.float64)
        # uniform sampling of negatives (SR-GNN default)
        self.p = np.full(items.shape, 1.0 / len(items))
        self.n_items = n_items

    def sample(self, n):
        return torch.from_numpy(np.random.choice(np.arange(1, self.n_items), size=n, p=self.p)).long()


def _collate(batch):
    # batch = list of (session_list, target_int); sessions are variable-length
    sess_list = [b[0] for b in batch]
    tgt = torch.LongTensor([b[1] for b in batch])
    return sess_list, tgt


def train_srgnn(train_sessions, n_items, item_freq, epochs=EPOCHS,
                embed_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, seeds=(42, 123)) -> List[SRGNN]:
    models = []
    sampler = PopSampler(n_items, item_freq)
    for seed in seeds:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        model = SRGNN(n_items, embed_dim, hidden_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        ds = SessDataset(train_sessions, n_items)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                            collate_fn=_collate, drop_last=len(ds) > BATCH)
        K = min(N_NEG, n_items - 2)
        for ep in range(epochs):
            model.train()
            tot = 0.0
            for sess_list, tgt in loader:
                targets = tgt.to(DEVICE)
                neg = sampler.sample(len(sess_list) * K).view(len(sess_list), K).to(DEVICE)
                logits = model(sess_list, targets, neg)
                labels = torch.zeros(len(sess_list), dtype=torch.long, device=DEVICE)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
            sch.step()
        models.append(model)
    return models


def predict_srgnn(models, test_uids, test_queries, n_items, batch=64):
    preds = {}
    for m in models:
        m.eval()
    for bs in range(0, len(test_uids), batch):
        chunk = test_uids[bs:bs + batch]
        sess_list = []
        for uid in chunk:
            ctx = [x for x in test_queries[uid]["context"] if 1 <= x < n_items]
            sess_list.append(ctx)
        with torch.no_grad():
            scores = torch.zeros(len(chunk), n_items, device=DEVICE)
            for m in models:
                scores = scores + torch.from_numpy(
                    predict_batch(m, sess_list, n_items)).to(DEVICE)
        scores = scores.cpu().numpy()
        for uid, sc, ctx in zip(chunk, scores, sess_list):
            sc = sc.copy(); sc[0] = -1e9
            for c in set(ctx):
                sc[c] = -1e9
            preds[uid] = [int(x) for x in np.argsort(-sc) if int(x) != 0][:50]
    return preds


def predict_batch(model, sess_list, n_items):
    """Score each session individually (graph is session-specific)."""
    out = np.zeros((len(sess_list), n_items), dtype=np.float32)
    for i, sess in enumerate(sess_list):
        if not sess:
            continue
        with torch.no_grad():
            sc = model.score_all([sess]).cpu().numpy()[0]
        out[i] = sc
    return out
