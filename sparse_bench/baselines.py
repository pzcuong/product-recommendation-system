"""
Baselines for the sparse short-session benchmark, ported to run on the unified
loader interface (same as codt_core): they consume train_sessions / test_queries
/ n_items and return predictions {uid: [item_ids]}.

Models: MostPop, ItemKNN (IUF-cosine + pop fallback), SKNN (session-KNN),
        GRU4Rec, SASRec  (neural, same config as multi_domain_benchmark.py).

These are apple-to-apple baselines trained on the SAME vocab/split as CoDT.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Defaults match multi_domain_benchmark.py for fairness.
EMB = 128
HEADS = 4
LAYERS = 2
DROPOUT = 0.4
MAXSEQ = 50
EPOCHS = 10
BS = 512
LR = 1e-3
WD = 1e-4
K_MAX = 20


# =============================================================================
# Neural models (same arch as multi_domain_benchmark.py)
# =============================================================================
class GRU4Rec(nn.Module):
    def __init__(self, n, emb=128, h=128, L=1, dp=0.4):
        super().__init__()
        self.ie = nn.Embedding(n, emb, padding_idx=0)
        self.gru = nn.GRU(emb, h, L, batch_first=True, dropout=dp)
        self.n = nn.LayerNorm(h)
        self.hd = nn.Linear(h, n)

    def forward(self, x, ln=None):
        z, _ = self.gru(self.ie(x))
        z = self.n(z)
        if ln is not None:
            li = (ln - 1).clamp(min=0)
            return self.hd(z[torch.arange(z.size(0), device=z.device), li])
        return self.hd(z[:, -1])


class SASRec(nn.Module):
    def __init__(self, n, emb=128, hds=4, L=2, dp=0.4, ml=50):
        super().__init__()
        self.ie = nn.Embedding(n, emb, padding_idx=0)
        self.pe = nn.Embedding(ml, emb)
        e = nn.TransformerEncoderLayer(emb, hds, emb * 4, dp, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(e, L)
        self.n = nn.LayerNorm(emb)
        self.hd = nn.Linear(emb, n)

    def forward(self, x, ln=None):
        B, L = x.shape
        z = self.ie(x) + self.pe(torch.arange(L, device=x.device).unsqueeze(0))
        m = None
        if ln is not None:
            p = torch.arange(L, device=x.device).unsqueeze(0)
            m = p >= ln.to(x.device).unsqueeze(1)
        z = self.tr(z, src_key_padding_mask=m)
        z = self.n(z)
        if ln is not None:
            li = (ln - 1).clamp(min=0)
            return self.hd(z[torch.arange(B, device=z.device), li])
        return self.hd(z[:, -1])


class SeqDataset(Dataset):
    def __init__(self, seqs, n_items, ml=50):
        self.ex = []
        for seq in seqs.values():
            seq = [x for x in seq if 1 <= x < n_items]
            for i in range(1, len(seq)):
                self.ex.append((seq[max(0, i - ml):i], seq[i]))

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(b):
    ml = max(len(c) for c, _ in b)
    return (torch.LongTensor([c + [0] * (ml - len(c)) for c, _ in b]),
            torch.LongTensor([len(c) for c, _ in b]),
            torch.LongTensor([t for _, t in b]))


def _train_nn(ModelClass, train_seqs, n_items, seed, epochs, emb, heads, layers):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = ModelClass(n_items, emb=emb, L=layers, dp=DROPOUT).to(DEVICE) \
        if ModelClass is GRU4Rec else \
        ModelClass(n_items, emb=emb, hds=heads, L=layers, dp=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = SeqDataset(train_seqs, n_items, MAXSEQ)
    if len(ds) == 0:
        return m
    loader = DataLoader(ds, batch_size=BS, shuffle=True, collate_fn=collate,
                        drop_last=len(ds) > BS)
    for _ in range(epochs):
        m.train()
        for inp, lens, tgt in loader:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            loss = F.cross_entropy(m(inp, lens), tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
    return m


def _predict_nn(model, test_uids, test_queries, n_items):
    model.eval()
    preds: Dict[str, List[int]] = {}
    BATCH = 256
    for bs in range(0, len(test_uids), BATCH):
        batch = test_uids[bs:bs + BATCH]
        seqs, lens = [], []
        for uid in batch:
            ctx = [x for x in test_queries[uid]["context"] if 1 <= x < n_items][-MAXSEQ:]
            seqs.append(ctx); lens.append(len(ctx))
        ml = max(max(lens), 1)
        inp = torch.zeros(len(batch), ml, dtype=torch.long, device=DEVICE)
        ln = torch.zeros(len(batch), dtype=torch.long, device=DEVICE)
        for i, (s, l) in enumerate(zip(seqs, lens)):
            inp[i, :l] = torch.LongTensor(s); ln[i] = l
        with torch.no_grad():
            logits = model(inp, ln).cpu().numpy()
        for uid, sc, ctx in zip(batch, logits, seqs):
            sc = sc.copy(); sc[0] = -1e9
            for c in set(ctx):
                sc[c] = -1e9
            preds[uid] = [int(x) for x in np.argsort(-sc) if int(x) != 0][:K_MAX]
    return preds


def run_gru4rec(train_sessions, test_queries, n_items, seed, epochs=10,
                emb=128, heads=4, layers=2) -> Dict[str, List[int]]:
    m = _train_nn(GRU4Rec, train_sessions, n_items, seed, epochs, emb, heads, layers)
    return _predict_nn(m, sorted(test_queries.keys()), test_queries, n_items)


def run_sasrec(train_sessions, test_queries, n_items, seed, epochs=10,
               emb=128, heads=4, layers=2) -> Dict[str, List[int]]:
    m = _train_nn(SASRec, train_sessions, n_items, seed, epochs, emb, heads, layers)
    return _predict_nn(m, sorted(test_queries.keys()), test_queries, n_items)


# =============================================================================
# Non-parametric baselines
# =============================================================================
def build_cooc(train_seqs, n_items):
    cooc = defaultdict(Counter); iuf = Counter(); pop = Counter()
    for seq in train_seqs.values():
        s = set(seq)
        for a in s:
            iuf[a] += 1; pop[a] += 1
        for a in s:
            for b in s:
                if a != b:
                    cooc[a][b] += 1
    return cooc, iuf, pop


def iknn_scores(ctx_p, cooc, iuf, n_items, pop_rank=None):
    sc = np.zeros(n_items)
    for it in ctx_p[-3:]:
        if it in cooc:
            for b, c in cooc[it].items():
                sc[b] += c / math.sqrt(max(iuf.get(it, 1), 1) * max(iuf.get(b, 1), 1))
    if pop_rank is not None:
        for rank, item in enumerate(pop_rank):
            if sc[item] == 0 and item != 0:
                sc[item] = 1e-9 - rank * 1e-12
    sc[0] = -1e9
    return sc


def build_sknn(train_seqs):
    return {u: set(s) for u, s in train_seqs.items()}


def sknn_scores(ctx_p, user_item_sets, n_items, k_sessions=100, pop_rank=None):
    cs = set(ctx_p[-5:]); sims = []
    for uid, us in user_item_sets.items():
        i = len(cs & us)
        if i > 0:
            sims.append((i / max(len(cs | us), 1), uid))
    sims.sort(reverse=True)
    sc = Counter()
    for _, uid in sims[:k_sessions]:
        for x in user_item_sets[uid]:
            if x not in cs:
                sc[x] += 1
    result = np.zeros(n_items)
    for x, c in sc.items():
        result[x] = c
    if pop_rank is not None:
        for rank, item in enumerate(pop_rank):
            if result[item] == 0 and item != 0 and item not in cs:
                result[item] = 1e-9 - rank * 1e-12
    result[0] = -1e9
    return result


def run_nonparametric(name: str, train_sessions, test_queries, n_items) -> Dict[str, List[int]]:
    cooc, iuf, pop = build_cooc(train_sessions, n_items)
    pop_rank = [x for x, _ in pop.most_common()]
    user_item_sets = build_sknn(train_sessions) if name == "SKNN" else None

    preds: Dict[str, List[int]] = {}
    if name == "MostPop":
        top = pop_rank[:K_MAX]
        for uid in test_queries:
            preds[uid] = [x for x in top if x != 0][:K_MAX]
        return preds

    for uid in tqdm(test_queries, desc=name, leave=False):
        ctx = test_queries[uid]["context"]
        seen = set(ctx)
        if name == "ItemKNN":
            sc = iknn_scores(ctx, cooc, iuf, n_items, pop_rank=pop_rank)
        elif name == "SKNN":
            sc = sknn_scores(ctx, user_item_sets, n_items, pop_rank=pop_rank)
        else:
            raise ValueError(name)
        for c in seen:
            sc[c] = -1e9
        preds[uid] = [int(x) for x in np.argsort(-sc) if int(x) != 0][:K_MAX]
    return preds
