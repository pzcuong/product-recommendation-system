"""
HACL-SBR — Hybrid Adaptive Contrastive Learning for Session-Based Recommendation.

Inspired by MACL ("Rethinking Contrastive Learning in Session-based
Recommendation", arXiv:2506.05044, PSU 2025), the current Q1-leading approach
for sparse / short-session SBR. HACL extends MACL with two novel components:

  Novelty 1 — Popularity-stratified hard-negative mining.
    MACL (and most CL4SRec-style methods) sample negatives uniformly, which
    biases the model toward popular items and leaves the long tail unlearnable
    (the exact tgt_tail=0 failure we diagnosed on Rental). HACL samples
    negatives from an inverse-popularity distribution so tail items appear as
    negatives more often, forcing the encoder to discriminate them.

  Novelty 2 — A third, co-visitation graph augmentation view.
    MACL uses item-text + sequence augmentation (2 views). HACL adds a graph
    view built from the global item co-visitation graph: augment a session by
    substituting an item with one of its co-visited neighbours. This injects
    global collaborative structure into the contrastive objective.

Pipeline:
  1. Item text encoder  : TF-IDF over name_en + main_category_en -> dense vec,
                          fused with a learnable ID embedding.
  2. Session encoder    : SASRec-style causal transformer (embedding-similarity
                          head, sampled-softmax training — scales to any vocab).
  3. Augmentation (3v)  : item-text view, sequence-crop view, co-visitation view.
  4. Adaptive CL loss   : MLP that re-weights each (anchor, candidate) pair
                          (MACL-style) instead of a fixed InfoNCE temperature.
  5. Joint objective    : L = L_rec (sampled-softmax) + lambda * L_CL.

This module is designed for the Rental visit-level protocol (small vocab,
runs on MPS) but is domain-agnostic and reusable on session benchmarks.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Defaults (Rental-tuned; adapt per dataset).
EMB_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.2
MAX_SEQ = 50
EPOCHS = 20
BATCH = 256
LR = 1e-3
N_NEG = 1024        # sampled-softmax negatives for the rec head
CL_N_NEG = 256      # negatives per anchor in the contrastive loss
LAMBDA_CL = 0.5     # weight of the CL term in the joint loss
AUG_RATE = 0.3      # fraction of items augmented per view


# =============================================================================
# 1. ITEM TEXT ENCODER (TF-IDF over name_en + main_category_en)
# =============================================================================
def build_tfidf(item_ids: List[int], slug2text: Dict[int, str]) -> np.ndarray:
    """Return a (n_items, D) dense TF-IDF matrix. Rows for items without text
    are left zero (the ID embedding carries them)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [slug2text.get(i, "") for i in range(max(item_ids) + 1)]
    vec = TfidfVectorizer(max_features=EMB_DIM, stop_words="english", token_pattern=r"(?u)\b\w\w+\b")
    try:
        M = vec.fit_transform(docs).toarray().astype(np.float32)
    except ValueError:
        # too few features; fall back to smaller dim, pad to EMB_DIM
        M = vec.fit_transform(docs).toarray().astype(np.float32)
    if M.shape[1] < EMB_DIM:
        M = np.pad(M, ((0, 0), (0, EMB_DIM - M.shape[1])))
    return M


# =============================================================================
# 2. SESSION ENCODER (SASRec-style, embedding-similarity head)
# =============================================================================
class SASRecSim(nn.Module):
    """Causal transformer with an embedding-similarity (dot-product) scoring
    head. Trained with sampled softmax -> scales to large vocabs."""

    def __init__(self, n_items, item_text_emb: Optional[np.ndarray], embed_dim=EMB_DIM,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT,
                 max_seq=MAX_SEQ, pad_idx=0):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        self.item_embed = nn.Embedding(n_items, embed_dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.item_embed.weight)
        self.item_embed.weight.data[pad_idx].zero_()
        # text feature projection: map frozen TF-IDF rows into the embed space
        if item_text_emb is not None:
            self.register_buffer("text_feat", torch.from_numpy(item_text_emb))
            self.text_gate = nn.Linear(embed_dim * 2, embed_dim)
        else:
            self.register_buffer("text_feat", None)
            self.text_gate = None
        self.pe = nn.Embedding(max_seq, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4,
                                               dropout, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # hidden -> item-embed space
        self.drop = nn.Dropout(dropout)

    def embed_items(self, items: torch.Tensor) -> torch.Tensor:
        """Fused ID + text embedding for a batch of item ids."""
        e = self.item_embed(items)
        if self.text_gate is not None and self.text_feat is not None:
            t = self.text_feat[items]
            e = self.text_gate(torch.cat([e, t], dim=-1))
        return e

    def encode(self, seq, lengths):
        B, L = seq.shape
        x = self.drop(self.embed_items(seq) + self.pe(torch.arange(L, device=seq.device).unsqueeze(0)))
        positions = torch.arange(L, device=seq.device).unsqueeze(0)
        pad_mask = positions >= lengths.to(seq.device).unsqueeze(1)
        x = self.tr(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        return x

    def last_hidden(self, seq, lengths):
        x = self.encode(seq, lengths)
        last_idx = (lengths.to(seq.device) - 1).clamp(min=0)
        return self.out_proj(x[torch.arange(x.size(0), device=x.device), last_idx])

    def score_all(self, hidden):
        return hidden @ self.item_embed.weight.t()

    def forward(self, seq, lengths, targets, negatives):
        """Sampled-softmax training step.
        Returns (logits_over_pos+neg (B,1+K), full hidden (B,D))."""
        hidden = self.last_hidden(seq, lengths)
        emb = self.item_embed.weight
        pos = emb[targets]
        neg = emb[negatives]
        pos_score = (hidden * pos).sum(-1, keepdim=True)
        neg_score = torch.bmm(neg, hidden.unsqueeze(-1)).squeeze(-1)
        return torch.cat([pos_score, neg_score], dim=-1), hidden


# =============================================================================
# 3. AUGMENTATION (3 views)
# =============================================================================
def aug_sequence_view(seq: List[int], rate=AUG_RATE) -> List[int]:
    """Crop + reorder (DuoRec-style)."""
    if len(seq) < 2:
        return list(seq)
    # random contiguous crop of 70-100% length
    keep = max(2, int(len(seq) * random.uniform(1 - rate, 1.0)))
    start = random.randint(0, len(seq) - keep)
    out = seq[start:start + keep]
    # small reorder: swap two adjacent items with prob rate
    if len(out) > 2 and random.random() < rate:
        i = random.randint(0, len(out) - 2)
        out = out[:i] + [out[i + 1], out[i]] + out[i + 2:]
    return out


def aug_item_view(seq: List[int], text_knn: Dict[int, List[int]], rate=AUG_RATE) -> List[int]:
    """Replace each item (prob rate) with a text-similar item (MACL-style)."""
    out = []
    for it in seq:
        if random.random() < rate and it in text_knn and text_knn[it]:
            out.append(random.choice(text_knn[it]))
        else:
            out.append(it)
    return out


def aug_graph_view(seq: List[int], covisit: Dict[int, List[int]], rate=AUG_RATE) -> List[int]:
    """Insert/substitute with co-visitation neighbours (HACL novelty 2)."""
    out = []
    for it in seq:
        if random.random() < rate and it in covisit and covisit[it]:
            out.append(random.choice(covisit[it]))
        else:
            out.append(it)
    return out


# =============================================================================
# 4. ADAPTIVE CONTRASTIVE LOSS + HARD-NEGATIVE SAMPLER
# =============================================================================
class AdaptiveCLLoss(nn.Module):
    """MACL-style adaptive InfoNCE: an MLP re-weights each (anchor, candidate)
    similarity before the softmax, instead of a fixed temperature."""

    def __init__(self, embed_dim=EMB_DIM):
        super().__init__()
        self.weighter = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, anchor, positive, negatives):
        """anchor (B,D), positive (B,D), negatives (B,K,D) -> scalar loss."""
        B, K, D = negatives.shape
        pos_pair = torch.cat([anchor, positive], dim=-1)
        pos_w = self.weighter(pos_pair)              # (B,1)
        pos_logit = ((anchor * positive).sum(-1, keepdim=True) * pos_w).squeeze(-1)
        neg_pair = torch.cat([anchor.unsqueeze(1).expand(-1, K, -1), negatives], dim=-1)
        neg_w = self.weighter(neg_pair).squeeze(-1)  # (B,K)
        neg_logit = (anchor.unsqueeze(1) * negatives).sum(-1) * neg_w
        logits = torch.cat([pos_logit.unsqueeze(1), neg_logit], dim=1)  # (B,1+K)
        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)


class HardNegativeSampler:
    """Popularity-stratified negative sampler (HACL novelty 1).

    Negatives are drawn from an inverse-popularity distribution so that rare
    (tail) items appear as negatives more often, forcing the encoder to learn
    discriminative representations for the long tail. Compare with uniform
    sampling, which over-weights popular negatives and leaves the tail
    unlearnable."""

    def __init__(self, n_items: int, item_freq: Counter, pad_idx: int = 0):
        self.n_items = n_items
        items = np.arange(1, n_items)  # skip PAD
        freq = np.array([item_freq.get(i, 0) for i in items], dtype=np.float64)
        # inverse-popularity sampling weights: w_i = 1 / (freq_i + 1)
        w = 1.0 / (freq + 1.0)
        self.hard_prob = w / w.sum()
        # uniform for the ablation
        self.uniform_prob = np.full(items.shape, 1.0 / len(items))

    def sample(self, n_samples: int, hard: bool = True) -> torch.Tensor:
        p = self.hard_prob if hard else self.uniform_prob
        return torch.from_numpy(np.random.choice(np.arange(1, self.n_items), size=n_samples, p=p)).long()


# =============================================================================
# 5. DATASET + TRAINING
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


def train_hacl(train_sessions: List[List[int]], n_items: int,
               item_text_emb: Optional[np.ndarray],
               text_knn: Dict[int, List[int]],
               covisit: Dict[int, List[int]],
               item_freq: Counter,
               epochs: int = EPOCHS, embed_dim: int = EMB_DIM,
               use_text_view: bool = True, use_graph_view: bool = True,
               hard_neg: bool = True, use_adaptive: bool = True,
               lambda_cl: float = LAMBDA_CL,
               seeds: Optional[List[int]] = None) -> List[SASRecSim]:
    """Train the HACL ensemble. Returns a list of trained models (one per seed)."""
    seeds = seeds or [42, 123]
    models = []
    sampler = HardNegativeSampler(n_items, item_freq)
    for seed in seeds:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        model = SASRecSim(n_items, item_text_emb, embed_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        cl_loss = AdaptiveCLLoss(embed_dim).to(DEVICE) if use_adaptive else None
        ds = SeqDataset(train_sessions, n_items)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate,
                            drop_last=len(ds) > BATCH)
        K = min(N_NEG, n_items - 2)
        Kcl = min(CL_N_NEG, n_items - 2)
        for ep in range(epochs):
            model.train()
            tot = 0.0
            for inp, lens, tgt in loader:
                inp, lens, tgt = inp.to(DEVICE), lens.to(DEVICE), tgt.to(DEVICE)
                # rec head: sampled softmax
                neg = sampler.sample(inp.size(0) * K, hard=hard_neg).view(inp.size(0), K).to(DEVICE)
                logits, hidden = model(inp, lens, tgt, neg)
                rec_loss = F.cross_entropy(logits, torch.zeros(inp.size(0), dtype=torch.long, device=DEVICE))

                # contrastive term: build augmented positive view, sample negatives
                if lambda_cl > 0:
                    # anchor = current session hidden (from model), positive = augmented session hidden
                    aug_seqs = []
                    for s in inp.cpu().tolist():
                        seq = [x for x in s if x != 0]
                        if use_graph_view and random.random() < 0.5:
                            seq = aug_graph_view(seq, covisit)
                        elif use_text_view and text_knn:
                            seq = aug_item_view(seq, text_knn)
                        else:
                            seq = aug_sequence_view(seq)
                        aug_seqs.append(seq)
                    ml = max(len(s) for s in aug_seqs)
                    a_inp = torch.LongTensor([s + [0] * (ml - len(s)) for s in aug_seqs]).to(DEVICE)
                    a_lens = torch.LongTensor([len(s) for s in aug_seqs]).to(DEVICE)
                    pos_hidden = model.last_hidden(a_inp, a_lens)
                    cl_neg = model.item_embed.weight[
                        sampler.sample(inp.size(0) * Kcl, hard=hard_neg).view(inp.size(0), Kcl).to(DEVICE)
                    ]
                    if use_adaptive and cl_loss is not None:
                        cl = cl_loss(hidden, pos_hidden, cl_neg)
                    else:
                        # plain InfoNCE fallback
                        pos = (hidden * pos_hidden).sum(-1, keepdim=True)
                        negs = torch.bmm(cl_neg, hidden.unsqueeze(-1)).squeeze(-1)
                        cl = F.cross_entropy(torch.cat([pos, negs], 1),
                                             torch.zeros(inp.size(0), dtype=torch.long, device=DEVICE))
                    loss = rec_loss + lambda_cl * cl
                else:
                    loss = rec_loss
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
            sch.step()
        models.append(model)
    return models


# =============================================================================
# 6. INFERENCE
# =============================================================================
def predict_hacl(models: List[SASRecSim], test_uids, test_queries, n_items,
                 max_seq=MAX_SEQ, batch=128,
                 covisit: Optional[dict] = None,
                 covisit_weight: float = 0.3) -> Dict[str, List[int]]:
    """Inference. Optionally fuse the neural score with a global co-visitation
    score (the mechanism that lets mid/tail items surface — without it the
    sampled-softmax head collapses onto popular items, as diagnosed)."""
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
                scores = scores + m.score_all(m.last_hidden(inp, ln))
        scores = scores.cpu().numpy()
        for uid, sc, ctx in zip(chunk, scores, seqs):
            sc = sc.copy(); sc[0] = -1e9
            for c in set(ctx):
                sc[c] = -1e9
            if covisit is not None and covisit_weight > 0 and ctx:
                # co-visitation score: sum of transition counts from last few ctx items
                cov = np.zeros(n_items)
                for prev in ctx[-5:]:
                    nbrs = covisit.get(prev)
                    if nbrs:
                        for rank, nb in enumerate(nbrs):
                            cov[nb] += 1.0 / (1.0 + rank)  # rank-discounted
                # normalize both to [0,1] then blend
                def _norm(x):
                    mx = x.max()
                    return x / mx if mx > 0 else x
                sc = (1 - covisit_weight) * _norm(sc) + covisit_weight * _norm(cov)
            preds[uid] = [int(x) for x in np.argsort(-sc) if int(x) != 0][:50]
    return preds
