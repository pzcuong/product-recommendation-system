"""
CoDT Core — Co-visitation Enhanced DualTwin (domain-agnostic).
==============================================================

Faithful, Rental-free port of "DualTwin-AgentCL V3.2 — Improved"
(archive/scripts/dualtwin_v32_improved.bak.py), the configuration that
achieved Recall@6 = 0.43 on the hidden Rental test set.

Components (all domain-agnostic — operate purely on int item-id sequences):
  1. PGSA-Rec   : Position-Gated Sequential Attention (causal transformer),
                  trained as next-item prediction, 4-seed ensemble averaged
                  at the logit level.
  2. M-CL       : Multimodal Contrastive item embeddings (InfoNCE on
                  co-occurring item pairs in a session) -> item embedding
                  space used for similarity boosts and MMR diversity.
  3. Co-vis / PMI: windowed (window=5) forward/backward co-occurrence graph
                  aggregated per training session, PMI-clamped (>=0) cache.
  4. Fusion     : full V3.2 additive boosts (PMI fwd/bwd, M-CL max/avg sim,
                  category, repurchase) with a *session-adaptive boost cap*
                  keyed on context length, plus multiplicative popularity
                  penalty on the PGSA base score.
  5. MMR        : Maximal Marginal Relevance diversity re-ranking.

Only inputs needed:
    - train_sessions : dict[str/uid -> List[int]]   (1-indexed, 0 = PAD)
    - test_queries   : dict[str/uid -> {"context": List[int], "targets": List[int]}]
    - n_items        : int (vocab size, 0 reserved for PAD)
    - item_categories: dict[int -> category_id] (optional, may be empty)
    - item_freq      : Counter[int] of training item frequencies

Reference: archive/scripts/dualtwin_v32_improved.bak.py
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

# =============================================================================
# DEVICE / GLOBAL CONFIG
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
NUM_WORKERS = 0

# --- Model defaults (match V3.2 .bak values) ---
PGSA_EMBED_DIM = 128
PGSA_NUM_HEADS = 4
PGSA_NUM_LAYERS = 2
PGSA_DROPOUT = 0.15
PGSA_MAX_SEQ = 50
PGSA_TOP_K = 200  # number of PGSA candidates kept per query before fusion

MCL_EMBED_DIM = 64
MCL_TEMP = 0.07

# --- Training defaults ---
PGSA_EPOCHS = 8
MCL_EPOCHS = 8
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
ENSEMBLE_SEEDS = [42, 123, 456, 789]  # 4-seed PGSA ensemble (intra-eval-seed)

# --- Fusion / re-ranking defaults (V3.2 .bak) ---
DIVERSITY_LAMBDA = 0.3      # MMR relevance/diversity weight (V3.2)
POPULARITY_PENALTY = 0.15   # multiplicative penalty on hot items (V3.2)


# =============================================================================
# MODELS  (byte-faithful to V3.2 .bak)
# =============================================================================
class TimeAwarePositionEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=50):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_scale = nn.Linear(1, embed_dim)
        self.register_buffer("sinusoidal", self._build_sinusoidal(max_seq_len, embed_dim))
        self.gate = nn.Linear(embed_dim * 2, embed_dim)

    @staticmethod
    def _build_sinusoidal(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, time_deltas=None):
        B, L, D = x.shape
        ordinal_pe = self.sinusoidal[:L].unsqueeze(0).expand(B, -1, -1)
        if time_deltas is not None:
            td = time_deltas.unsqueeze(-1)
            time_pe = self.time_scale(td)
            combined = torch.cat([ordinal_pe, time_pe], dim=-1)
            gate = torch.sigmoid(self.gate(combined))
            return gate * time_pe + (1 - gate) * ordinal_pe
        return x + ordinal_pe


class PositionGate(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate_net = nn.Sequential(nn.Linear(embed_dim + 1, embed_dim), nn.Sigmoid())

    def forward(self, attention_output, temporal_weights):
        gate_input = torch.cat([attention_output, temporal_weights], dim=-1)
        gate = self.gate_net(gate_input)
        return attention_output * gate


class PGSARecBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.15):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.position_gate = PositionGate(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout),
        )

    @staticmethod
    def _generate_causal_mask(seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, x, temporal_weights=None, padding_mask=None):
        B, L, D = x.shape
        causal_mask = self._generate_causal_mask(L, x.device)
        attn_out, _ = self.attention(x, x, x, attn_mask=causal_mask, key_padding_mask=padding_mask)
        attn_out = self.norm1(attn_out + x)
        if temporal_weights is None:
            temporal_weights = torch.ones(B, L, 1, device=x.device)
        gated_out = self.position_gate(attn_out, temporal_weights)
        return self.norm2(self.ffn(gated_out) + gated_out)


class PGSARecModel(nn.Module):
    """Position-Gated Sequential Attention recommender.

    head_type:
      "softmax"  : full Linear(embed->n_items) head. Works for small vocabs
                   (Rental ~1K) but is untrainable on large vocabs (50K+) due to
                   the huge softmax weight matrix and class imbalance.
      "sim"      : embedding-similarity head — score(item) = hidden · item_embed.
                   Scales to large vocabs (no Linear weight), trained with sampled
                   negatives. This is the SR-GNN/SASRec-style head and is REQUIRED
                   for SOTA on Diginetica/YooChoose/RetailRocket.
    """

    def __init__(self, n_items, embed_dim=128, num_heads=4, num_layers=2, dropout=0.15,
                 max_seq_len=50, pad_idx=0, head_type="softmax"):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.head_type = head_type
        self.item_embed = nn.Embedding(n_items, embed_dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.item_embed.weight)
        self.item_embed.weight.data[pad_idx].zero_()
        self.drop_emb = nn.Dropout(dropout)
        self.time_pos_embed = TimeAwarePositionEmbedding(embed_dim, max_seq_len)
        self.blocks = nn.ModuleList(
            [PGSARecBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        if head_type == "softmax":
            self.head = nn.Linear(embed_dim, n_items)
        elif head_type == "sim":
            # projection from hidden state into the item-embedding space, so the
            # dot product is meaningful (hidden is post-block, embed is input).
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.head = None  # scores computed on the fly
        else:
            raise ValueError(f"unknown head_type {head_type}")

    def encode(self, seq, lengths=None, time_deltas=None, dropout=True):
        """Run the transformer blocks; return per-position hidden states."""
        B, L = seq.shape
        x = self.drop_emb(self.item_embed(seq)) if dropout else self.item_embed(seq)
        x = self.time_pos_embed(x, time_deltas)
        if lengths is not None:
            lens_dev = lengths.to(seq.device)
            positions = torch.arange(L, device=seq.device).unsqueeze(0)
            padding_mask = positions >= lens_dev.unsqueeze(1)
            pos_from_end = (lens_dev.unsqueeze(1).float() - positions.float() - 1).clamp(min=0)
            temporal_weights = torch.exp(-0.1 * pos_from_end).unsqueeze(-1)
        else:
            padding_mask = None
            temporal_weights = None
        for block in self.blocks:
            x = block(x, temporal_weights, padding_mask)
        return x

    def _last_hidden(self, x, lengths, seq):
        B = x.size(0)
        if lengths is not None:
            lens_dev = lengths.to(seq.device)
            last_idx = (lens_dev - 1).clamp(min=0)
            return x[torch.arange(B, device=seq.device), last_idx]
        return x[:, -1]

    def _score_all(self, hidden):
        """Score every item given a (B, D) hidden state -> (B, n_items)."""
        if self.head_type == "softmax":
            return self.head(hidden)
        # sim head: project hidden, dot with all item embeddings.
        h = self.out_proj(hidden)                 # (B, D)
        emb = self.item_embed.weight              # (n_items, D)
        return h @ emb.t()                        # (B, n_items)

    def forward(self, seq, lengths=None, time_deltas=None):
        """Per-position logits for teacher-forced training (softmax head only).

        For the sim head, use forward_sim() with sampled negatives instead.
        """
        x = self.encode(seq, lengths, time_deltas, dropout=True)
        if self.head_type == "softmax":
            return self.head(x)
        # sim head returns per-position scores over the full vocab (for compat);
        # training should prefer forward_sim for efficiency.
        h = self.out_proj(x)
        return h @ self.item_embed.weight.t()

    def forward_sim(self, seq, lengths, targets, negatives, time_deltas=None):
        """Sampled-softmax forward for the sim head (large vocab training).

        targets    : (B,) positive item ids.
        negatives  : (B, K) sampled negative item ids.
        Returns logits over [positive, K negatives] of shape (B, K+1) and the
        full-vocab hidden state (B, D) (used for inference scoring).
        """
        x = self.encode(seq, lengths, time_deltas, dropout=True)
        h = self.out_proj(self._last_hidden(x, lengths, seq))   # (B, D)
        emb = self.item_embed.weight                            # (n_items, D)
        pos_emb = emb[targets]                                  # (B, D)
        neg_emb = emb[negatives]                                # (B, K, D)
        pos_score = (h * pos_emb).sum(-1, keepdim=True)         # (B, 1)
        neg_score = torch.bmm(neg_emb, h.unsqueeze(-1)).squeeze(-1)  # (B, K)
        logits = torch.cat([pos_score, neg_score], dim=-1)      # (B, K+1)
        return logits, h

    def predict(self, seq, lengths=None, time_deltas=None):
        """Return logits at the last non-pad position per sequence, over the FULL vocab."""
        self.eval()
        with torch.no_grad():
            x = self.encode(seq, lengths, time_deltas, dropout=False)
            h = self._last_hidden(x, lengths, seq)
            return self._score_all(h)


class MultimodalCLModel(nn.Module):
    def __init__(self, n_items, embed_dim=64):
        super().__init__()
        self.id_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.id_embedding.weight)
        self.id_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.semantic_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def get_id_embeddings(self, items):
        return F.normalize(self.id_projector(self.id_embedding(items)), dim=-1)

    def get_semantic_embeddings(self, items):
        return F.normalize(self.semantic_projector(self.id_embedding(items)), dim=-1)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature
        neg_sim = torch.mm(anchor, negatives.t()) / self.temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


# =============================================================================
# DATASET / COLLATE
# =============================================================================
class SeqDataset(Dataset):
    """Next-item windows from sessions (len>=3), mirroring V3.2 .bak."""

    def __init__(self, sessions, max_len=50):
        self.sessions = [s for s in sessions if len(s) >= 3]
        self.max_len = max_len

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        seq = self.sessions[idx]
        if len(seq) > self.max_len + 1:
            start = random.randint(0, len(seq) - self.max_len - 1)
            seq = seq[start:start + self.max_len + 1]
        return seq[:-1], seq[1:]


def collate_seq(batch):
    inputs, targets = zip(*batch)
    ml = max(len(s) for s in inputs)
    inp = torch.LongTensor([list(s) + [0] * (ml - len(s)) for s in inputs])
    tgt = torch.LongTensor([list(s) + [-1] * (ml - len(s)) for s in targets])
    lengths = torch.LongTensor([len(s) for s in inputs])
    return inp, lengths, tgt


# =============================================================================
# TRAINING
# =============================================================================
def _set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_siw_weights(item_freq: Counter, smoothing: float = 1.0) -> Dict[int, float]:
    """CSIL v2: build global self-information weights per item.

    w(target) = 1 / sqrt(P(target) + smoothing)

    where P(target) = freq(target) / sum(freq).

    - Popular items: high P → low weight (model already learns them well)
    - Rare items:    low P → high weight (need more gradient to learn)
    - smoothing avoids division by zero for unseen items

    Returns: {item_id: weight} for all items in item_freq.
    """
    total = sum(item_freq.values())
    weights = {}
    max_w = 0.0
    for item, freq in item_freq.items():
        p = freq / max(total, 1)
        w = 1.0 / math.sqrt(p + smoothing / max(total, 1))
        weights[item] = w
        max_w = max(max_w, w)
    # Normalize so max weight = 2.0 (conservative: never more than 2x normal)
    if max_w > 0:
        for item in weights:
            weights[item] = weights[item] / max_w * 2.0
    return weights


def build_conditional_weights(train_sessions: List[List[int]],
                              fwd_cooc: Dict[int, Counter],
                              bwd_cooc: Dict[int, Counter],
                              alpha: float = 0.5,
                              beta: float = 0.5) -> Dict[Tuple[int, ...], Dict[int, float]]:
    """Conditional self-information weights from co-visitation graph."""
    transition_counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
    for sess in train_sessions:
        for i in range(1, len(sess)):
            prefix = tuple(sess[max(0, i-5):i])
            target = sess[i]
            if prefix:
                transition_counts[prefix][target] += 1
    weight_map: Dict[Tuple[int, ...], Dict[int, float]] = {}
    for ctx, counts in transition_counts.items():
        total = sum(counts.values())
        weight_map[ctx] = {}
        for target, cnt in counts.items():
            p = cnt / total
            w = alpha + beta * (-math.log(max(p, 1e-10)))
            weight_map[ctx][target] = w
    return weight_map, 0.0


def get_csil_weight(context_items: List[int], target: int,
                    weight_map: Dict[Tuple[int, ...], Dict[int, float]],
                    alpha: float = 0.5, beta: float = 0.5,
                    default_w: float = 1.0) -> float:
    """Look up conditional weight. Falls back to default if not found."""
    if not context_items:
        return default_w
    for k in range(min(5, len(context_items)), 0, -1):
        prefix = tuple(context_items[-k:])
        if prefix in weight_map and target in weight_map[prefix]:
            return weight_map[prefix][target]
    return default_w


def train_pgsa(sessions: List[List[int]], n_items: int, seed: int,
               embed_dim=PGSA_EMBED_DIM, num_heads=PGSA_NUM_HEADS,
               num_layers=PGSA_NUM_LAYERS, dropout=PGSA_DROPOUT,
               max_seq=PGSA_MAX_SEQ, epochs=PGSA_EPOCHS,
               batch_size=BATCH_SIZE, lr=LEARNING_RATE,
               head_type="softmax", n_negatives=2048,
               contrastive: bool = False, lambda_cl: float = 0.2,
               csil: bool = False,
               csil_weight_map: Optional[Dict] = None,
               csil_alpha: float = 0.5, csil_beta: float = 0.5,
               siw: bool = False,
               siw_weights: Optional[Dict[int, float]] = None) -> PGSARecModel:
    """Train a single PGSA-Rec model (one ensemble member).

    head_type:
      "softmax" — full softmax (small vocabs only).
      "sim"     — sampled-softmax embedding-similarity head (large vocab).
    contrastive: if True, add CoSeRec-style contrastive auxiliary loss.
    csil: if True, use Conditional Self-Information Weighted Loss (conditional).
    siw: if True, use Global Self-Information Weighted Loss.
          w(target) = 1/sqrt(P(target)) — rare items get higher gradient.
          Simple, principled, no context dependency.
    """
    _set_seed(seed)
    ds = SeqDataset(sessions, max_seq)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_seq, num_workers=NUM_WORKERS, drop_last=True)
    model = PGSARecModel(n_items, embed_dim, num_heads, num_layers, dropout, max_seq,
                         head_type=head_type).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_sim = head_type == "sim"
    K = min(n_negatives, n_items - 2)
    # Precompute SIW per-position weights (global self-information based).
    use_siw = siw and siw_weights is not None
    if use_siw:
        # Precompute: for each (item, position) pair, the weight is SIW(target).
        # We compute this on CPU and store per-position for the batch.
        pass  # We'll compute inline in the loop for simplicity

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inp, lengths, tgt in loader:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)

            if not use_sim:
                logits = model(inp, lengths)
                B, L, V = logits.shape
                mask = tgt.reshape(-1) != -1
                logits_flat = logits.reshape(-1, V)[mask]
                tgt_flat = tgt.reshape(-1)[mask]

                if use_siw or (csil and csil_weight_map is not None):
                    # Per-position weighting (CSIL or SIW)
                    ce_loss = F.cross_entropy(logits_flat, tgt_flat, reduction='none')
                    # Compute per-position weights on CPU
                    inp_cpu = inp.cpu()
                    tgt_cpu = tgt.cpu()
                    mask_cpu = mask.cpu()
                    n_valid = int(mask_cpu.sum().item())
                    pos_weights = torch.ones(n_valid, dtype=torch.float32)
                    pos_idx = 0
                    for bi in range(B):
                        seq = inp_cpu[bi].tolist()
                        tgt_seq = tgt_cpu[bi].tolist()
                        length = int(lengths[bi].item())
                        for t in range(length):
                            target = tgt_seq[t]
                            if target == -1:
                                continue
                            if pos_idx >= n_valid:
                                break
                            if use_siw and target in siw_weights:
                                pos_weights[pos_idx] = siw_weights[target]
                            elif csil and csil_weight_map is not None:
                                ctx = [s for s in seq[max(0, t-4):t] if s != 0]
                                pos_weights[pos_idx] = get_csil_weight(
                                    ctx, target, csil_weight_map, csil_alpha, csil_beta)
                            pos_idx += 1
                    pos_weights = pos_weights / pos_weights.mean()
                    loss = (ce_loss * pos_weights.to(DEVICE)).mean()
                else:
                    loss = F.cross_entropy(logits_flat, tgt_flat)
            else:
                # sim head: train on the LAST position's target only (next-item
                # at the end of each session), with sampled negatives. SeqDataset
                # already returns shifted (seq[:-1] -> seq[1:]) so the last target
                # is the session's true next item.
                last_tgt = tgt[:, -1]
                valid = last_tgt != -1
                if valid.sum() == 0:
                    continue
                # lengths is a CPU tensor (from collate); index it with a CPU mask.
                valid_cpu = valid.cpu()
                valid_idx = valid_cpu.nonzero(as_tuple=True)[0]
                inp_v = inp[valid_idx.to(inp.device)]
                len_v = lengths[valid_idx].to(DEVICE)
                tgt_v = last_tgt[valid_idx.to(tgt.device)]
                B = inp_v.size(0)
                neg = torch.randint(1, n_items, (B, K), device=DEVICE)
                logits, _ = model.forward_sim(inp_v, len_v, tgt_v, neg)
                # label 0 = positive
                labels = torch.zeros(B, dtype=torch.long, device=DEVICE)
                loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            # Contrastive auxiliary loss (CoSeRec-style): two augmented views,
            # InfoNCE pulls a session's two views together and pushes apart from
            # other sessions in the batch.
            if contrastive and lambda_cl > 0:
                a_seqs = [_augment_seq(s.tolist(), max_seq) for s in inp.cpu()]
                b_seqs = [_augment_seq(s.tolist(), max_seq) for s in inp.cpu()]
                ml_a = max(len(s) for s in a_seqs); ml_b = max(len(s) for s in b_seqs)
                ml = max(ml_a, ml_b, 1)
                a_t = torch.zeros(len(a_seqs), ml, dtype=torch.long, device=DEVICE)
                a_l = torch.zeros(len(a_seqs), dtype=torch.long, device=DEVICE)
                b_t = torch.zeros(len(b_seqs), ml, dtype=torch.long, device=DEVICE)
                b_l = torch.zeros(len(b_seqs), dtype=torch.long, device=DEVICE)
                for i, (sa, sb) in enumerate(zip(a_seqs, b_seqs)):
                    a_t[i, :len(sa)] = torch.LongTensor(sa); a_l[i] = len(sa)
                    b_t[i, :len(sb)] = torch.LongTensor(sb); b_l[i] = len(sb)
                ha = F.normalize(_session_hidden(model, a_t, a_l), dim=-1)
                hb = F.normalize(_session_hidden(model, b_t, b_l), dim=-1)
                cl_logits = ha @ hb.t() / 0.1
                cl_labels = torch.arange(ha.size(0), device=DEVICE)
                loss = loss + lambda_cl * F.cross_entropy(cl_logits, cl_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
    return model


def train_mcl(sessions: List[List[int]], n_items: int, embed_dim=MCL_EMBED_DIM,
              epochs=MCL_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
              seed: int = 42) -> MultimodalCLModel:
    """Train M-CL: positive pair = any two distinct items co-occurring in a session."""
    _set_seed(seed)
    pairs: List[Tuple[int, int]] = []
    for sess in sessions:
        unique = list(set(sess))
        if len(unique) >= 2:
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    pairs.append((unique[i], unique[j]))
    random.shuffle(pairs)

    model = MultimodalCLModel(n_items, embed_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = InfoNCELoss(temperature=MCL_TEMP)
    all_items = list(range(n_items))
    for epoch in range(epochs):
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            if len(batch) < 2:
                continue
            anc = torch.LongTensor([p[0] for p in batch]).to(DEVICE)
            pos = torch.LongTensor([p[1] for p in batch]).to(DEVICE)
            neg = torch.LongTensor(random.choices(all_items, k=min(512, n_items))).to(DEVICE)
            loss = criterion(model.get_id_embeddings(anc),
                             model.get_semantic_embeddings(pos),
                             model.get_id_embeddings(neg))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


# =============================================================================
# CO-VISITATION / PMI  (windowed per training session)
# =============================================================================
def build_covisit_pmi(train_sessions: List[List[int]],
                      item_freq: Counter,
                      window: int = 5) -> Tuple[dict, dict, Counter, dict]:
    """Build forward/backward co-occurrence + PMI cache (clamped >=0).

    Mirrors V3.2 .bak logic but groups per-session (domain-agnostic) instead
    of per-client Rental history.

    Returns:
        fwd_cooc, bwd_cooc : dict[int, Counter]
        pair_freq           : Counter[(a,b)]
        pmi_cache           : dict[(a,b)] -> float (>=0)
    """
    fwd_cooc: Dict[int, Counter] = defaultdict(Counter)
    bwd_cooc: Dict[int, Counter] = defaultdict(Counter)
    pair_freq: Counter = Counter()
    for sess in train_sessions:
        # dedupe-adjacent-free list of items already seen in this session window
        prods: List[int] = []
        for pid in sess:
            for prev in prods[-window:]:
                if prev == pid:
                    continue
                fwd_cooc[prev][pid] += 1
                bwd_cooc[pid][prev] += 1
                pair_freq[(prev, pid)] += 1
            prods.append(pid)

    total_pairs = sum(pair_freq.values())
    total_items = sum(item_freq.values()) or 1
    pmi_cache: Dict[Tuple[int, int], float] = {}
    for (a, b), cnt in pair_freq.items():
        p_ab = cnt / total_pairs
        p_a = item_freq.get(a, 0) / total_items
        p_b = item_freq.get(b, 0) / total_items
        pmi = max(0.0, math.log(p_ab / (p_a * p_b + 1e-10) + 1e-10))
        pmi_cache[(a, b)] = pmi
    return fwd_cooc, bwd_cooc, pair_freq, pmi_cache


def augment_covisit_graph(fwd_cooc: Dict[int, Counter],
                          bwd_cooc: Dict[int, Counter],
                          pair_freq: Counter,
                          item_freq: Counter,
                          mcl_emb: Optional[np.ndarray],
                          tail_deg_threshold: int = 3,
                          top_k_neighbors: int = 10,
                          sim_floor: float = 0.25,
                          edge_weight: float = 0.5,
                          ) -> Tuple[Dict[int, Counter], Dict[int, Counter], Counter]:
    """GALORE-inspired graph augmentation for the co-visitation graph.

    Tail items (few co-visitation neighbours, degree < tail_deg_threshold) get
    PREDICTED edges added: for each tail item, find its top-K semantically
    similar items (via M-CL embeddings) and inject a co-visitation edge to each
    of their neighbours. This lets tail items inherit transition patterns from
    semantically-close popular items that the sparse co-visitation graph alone
    never observed.

    Only NEW edges are added (existing counts untouched), and only for tail
    items, so head-item rankings are unchanged (the CoDT 0.43 base is the floor).

    Returns new (fwd_cooc, bwd_cooc, pair_freq) with augmented edges merged in.
    The PMI cache should be rebuilt on the augmented pair_freq (the caller does
    this via build_covisit_pmi's PMI step; here we provide a helper below).
    """
    if mcl_emb is None:
        return fwd_cooc, bwd_cooc, pair_freq
    n = mcl_emb.shape[0]
    fwd = {k: Counter(v) for k, v in fwd_cooc.items()}
    bwd = {k: Counter(v) for k, v in bwd_cooc.items()}
    pairs = Counter(pair_freq)

    # Identify tail items by co-visitation degree (forward + backward).
    def _deg(it):
        return len(fwd.get(it, ())) + len(bwd.get(it, ()))
    tail_items = [it for it in range(1, n) if 0 < it < n and _deg(it) < tail_deg_threshold]
    if not tail_items:
        return fwd, bwd, pairs

    # Precompute semantic top-K for every item (matrix multiply once).
    sim = mcl_emb @ mcl_emb.T  # (n, n)
    np.fill_diagonal(sim, -1.0)
    added = 0
    for it in tail_items:
        # Semantic neighbours of this tail item.
        nb_idx = np.argpartition(-sim[it], min(top_k_neighbors, n - 2))[:top_k_neighbors]
        for nb in nb_idx:
            nb = int(nb)
            if sim[it, nb] < sim_floor:
                continue
            # Inherit the semantic neighbour's co-visitation partners.
            for partner, cnt in fwd.get(nb, Counter()).items():
                if partner == it or cnt <= 0:
                    continue
                w = edge_weight * (float(cnt) / max(1, sum(fwd[nb].values()))) * float(max(sim[it, nb], 0.0))
                if w <= 0:
                    continue
                fwd.setdefault(it, Counter())
                fwd[it][partner] = fwd[it].get(partner, 0.0) + w
                bwd.setdefault(partner, Counter())
                bwd[partner][it] = bwd[partner].get(it, 0.0) + w
                pairs[(it, partner)] = pairs.get((it, partner), 0.0) + w
                added += 1
    return fwd, bwd, pairs


def rebuild_pmi_from_pairs(pair_freq: Counter, item_freq: Counter) -> Dict[Tuple[int, int], float]:
    """Recompute the PMI cache (clamped >=0) from a (possibly augmented) pair_freq."""
    total_pairs = sum(pair_freq.values()) or 1
    total_items = sum(item_freq.values()) or 1
    pmi_cache: Dict[Tuple[int, int], float] = {}
    for (a, b), cnt in pair_freq.items():
        p_ab = cnt / total_pairs
        p_a = item_freq.get(a, 0) / total_items
        p_b = item_freq.get(b, 0) / total_items
        pmi = max(0.0, math.log(p_ab / (p_a * p_b + 1e-10) + 1e-10))
        pmi_cache[(a, b)] = pmi
    return pmi_cache


def build_category_cooc(train_sessions: List[List[int]],
                        item_categories: Dict[int, object],
                        window: int = 3) -> Dict[object, Counter]:
    """cat -> Counter[item] aggregated over the last `window` categories seen."""
    cat_to_prods: Dict[object, Counter] = defaultdict(Counter)
    for sess in train_sessions:
        cats: List[object] = []
        for pid in sess:
            c = item_categories.get(pid)
            if c is not None:
                cats.append(c)
            for c in cats[-window:]:
                cat_to_prods[c][pid] += 1
    return cat_to_prods


def build_popularity_penalty(item_freq: Counter, penalty=POPULARITY_PENALTY) -> Dict[int, float]:
    """Linear popularity penalty in [0, penalty] keyed by item id."""
    if not item_freq:
        return {}
    max_freq = max(item_freq.values())
    return {item: penalty * (freq / max_freq) for item, freq in item_freq.items()}


# =============================================================================
# PGSA ENSEMBLE PREDICTION (batched)
# =============================================================================
def pgsa_ensemble_predict(models: List[PGSARecModel],
                          test_uids: List[str],
                          test_queries: dict,
                          n_items: int,
                          max_seq: int = PGSA_MAX_SEQ,
                          top_k: int = PGSA_TOP_K,
                          batch_size: int = 64,
                          exclude_ctx: bool = True) -> Dict[str, List[Tuple[int, float]]]:
    """Average logits over the PGSA ensemble, return top-k candidates per user.

    Mirrors V3.2 .bak [5/7] PGSA PREDICTIONS (no allowed_set filtering — the
    candidate universe is the whole vocab, which is the correct behaviour for
    Amazon/RetailRocket where there is no Rental allowed_set split).
    """
    for m in models:
        m.eval()

    # Pre-build padded sequences per test user
    ordered: List[Tuple[str, List[int]]] = []
    for uid in test_uids:
        ctx = test_queries[uid]["context"]
        ctx = [x for x in ctx if 1 <= x < n_items][-max_seq:]
        ordered.append((uid, ctx))

    preds: Dict[str, List[Tuple[int, float]]] = {}
    for bs in tqdm(range(0, len(ordered), batch_size), desc="PGSA-ensemble", leave=False):
        batch = ordered[bs:bs + batch_size]
        max_len = max((len(c) for _, c in batch), default=1)
        max_len = max(max_len, 1)
        seq_t = torch.zeros(len(batch), max_len, dtype=torch.long, device=DEVICE)
        len_t = torch.zeros(len(batch), dtype=torch.long, device=DEVICE)
        for i, (_, ctx) in enumerate(batch):
            seq_t[i, :len(ctx)] = torch.LongTensor(ctx)
            len_t[i] = len(ctx)

        avg = torch.zeros(len(batch), n_items, device=DEVICE)
        with torch.no_grad():
            for m in models:
                avg = avg + m.predict(seq_t, len_t)
        avg = avg / len(models)
        avg = avg.cpu().numpy()

        for i, (uid, ctx) in enumerate(batch):
            scores = avg[i].copy()
            scores[0] = -np.inf  # drop PAD
            if exclude_ctx:
                for idx in set(ctx):
                    scores[idx] = -np.inf
            k = min(top_k, np.sum(np.isfinite(scores)))
            if k <= 0:
                preds[uid] = []
                continue
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            preds[uid] = [(int(idx), float(scores[idx])) for idx in top]
    return preds


# =============================================================================
# FUSION  (full V3.2 additive boosts + session-adaptive cap)
# =============================================================================
def build_popularity_tiers(item_freq: Counter,
                           head_pct: float = 0.2, tail_pct: float = 0.2
                           ) -> Tuple[set, set, set]:
    """Return (head, mid, tail) item-id sets from training frequency."""
    sorted_items = sorted(item_freq.items(), key=lambda x: -x[1])
    n = len(sorted_items)
    head = set(i for i, _ in sorted_items[:max(1, int(n * head_pct))])
    tail = set(i for i, _ in sorted_items[max(1, int(n * (1 - tail_pct))):])
    mid = set(i for i, _ in sorted_items) - head - tail
    return head, mid, tail


# Boost-cap multiplier per popularity tier (novelty: adapts to item rarity, not just context length).
_TIER_BOOST_MULT = {"head": 0.7, "mid": 1.0, "tail": 1.5}


def fuse_one_query(pgsa_items: List[Tuple[int, float]],
                   ctx_items: List[int],
                   ctx_cats: List[object],
                   pmi_cache: dict,
                   fwd_cooc: dict,
                   cat_to_prods: dict,
                   mcl_emb: Optional[np.ndarray],
                   popularity_penalty: Dict[int, float],
                   enable_fusion: bool = True,
                   pop_tiers: Optional[Tuple[set, set, set]] = None) -> List[Tuple[int, float]]:
    """Apply fusion to a single query's candidates.

    Novelty over V3.2: the session-adaptive boost cap is further modulated by
    the candidate's popularity tier — tail items get MORE fusion room, head
    items get LESS. This is the popularity-adaptive fusion that rescues tail
    items the neural backbone underfits.

    Args:
        pop_tiers: (head_set, mid_set, tail_set) from build_popularity_tiers.
                   If None, no tier modulation (V3.2 behaviour).
    Returns a list of (item, fused_score) for all PGSA candidates.
    """
    last_prods = list(ctx_items)
    last_cats = list(ctx_cats)
    last_prods_set = set(last_prods)
    n_hist = len(last_prods)
    n_lp = max(len(last_prods), 1)

    if enable_fusion:
        # Precompute per-context-item recency weights (V3.2 formulae).
        fwd_recency = [1.0 + 3.0 * (i / n_lp) for i in range(len(last_prods))]
        bwd_recency = [1.0 + 2.0 * (i / n_lp) for i in range(len(last_prods))]
        # Precompute fwd_cooc lookups per context item (None if absent).
        fwd_maps = [fwd_cooc.get(prev) for prev in last_prods]

    # M-CL: precompute the last-5 context embeddings once for vectorized sims.
    if mcl_emb is not None and last_prods:
        ctx_recent = [p for p in last_prods[-5:] if 0 < p < mcl_emb.shape[0]]
        if ctx_recent:
            ctx_emb = mcl_emb[ctx_recent]  # [k, dim]
        else:
            ctx_emb = None
    else:
        ctx_emb = None

    reranked: List[Tuple[int, float]] = []
    for item, pgsa_score in pgsa_items:
        base = float(pgsa_score)
        # Popularity penalty (multiplicative on base)
        if item in popularity_penalty:
            base *= (1.0 - popularity_penalty[item])

        if not enable_fusion:
            reranked.append((item, base))
            continue

        boost = 0.0

        # --- Forward PMI (recency-weighted) ---
        pmi_total = 0.0
        for i, prev in enumerate(last_prods):
            pmi = pmi_cache.get((prev, item))
            if pmi is not None:
                pmi_total += pmi * fwd_recency[i]
            else:
                fm = fwd_maps[i]
                if fm is not None and item in fm:
                    pmi_total += 0.05 * fm[item] * fwd_recency[i]
        boost += min(0.5, pmi_total * 0.08)

        # --- Backward PMI ---
        for i, prev in enumerate(last_prods):
            pmi = pmi_cache.get((item, prev))
            if pmi is not None:
                boost += min(0.3, pmi * bwd_recency[i] * 0.05)

        # --- Category boost ---
        for j, cat in enumerate(last_cats):
            cr = 1.0 + 1.5 * (j / max(len(last_cats), 1))
            cp = cat_to_prods.get(cat)
            if cp is not None:
                c = cp.get(item)
                if c:
                    boost += min(0.20, 0.015 * c) * cr

        # --- M-CL multi-scale similarity (vectorized) ---
        if ctx_emb is not None and 0 < item < mcl_emb.shape[0]:
            sims = ctx_emb @ mcl_emb[item]
            max_sim = float(sims.max())
            avg_sim = float(sims.mean())
            if max_sim > 0.2:
                boost += min(0.25, (max_sim - 0.2) * 0.3)
            if avg_sim > 0.15:
                boost += min(0.15, (avg_sim - 0.15) * 0.2)

        # --- DT-BGM repurchase boost ---
        if item in last_prods_set:
            positions = [i for i, p in enumerate(last_prods) if p == item]
            count = len(positions)
            recency_proxy = max(positions) / n_lp
            rp_boost = 0.15 * count * (1.0 + recency_proxy)
            boost += min(0.8, rp_boost)

        # --- Session-adaptive + popularity-adaptive boost cap (NOVELTY) ---
        # Context-length base cap (V3.2 session-adaptive).
        if n_hist >= 4:
            base_cap = 0.50
        elif n_hist >= 2:
            base_cap = 0.35
        elif n_hist >= 1:
            base_cap = 0.20
        else:
            base_cap = 0.10

        # Popularity-tier modulation (NOVELTY): tail items get MORE fusion
        # room, head items get LESS. This directly targets the popularity
        # collapse problem where neural models underfit tail items.
        if pop_tiers is not None:
            head_set, mid_set, tail_set = pop_tiers
            if item in tail_set:
                tier_mult = _TIER_BOOST_MULT["tail"]  # 1.5
            elif item in head_set:
                tier_mult = _TIER_BOOST_MULT["head"]  # 0.7
            else:
                tier_mult = _TIER_BOOST_MULT["mid"]   # 1.0
        else:
            tier_mult = 1.0

        boost_cap_pct = base_cap * tier_mult
        max_boost = max(abs(base) * boost_cap_pct, 0.1)
        boost = min(boost, max_boost)

        reranked.append((item, base + boost))
    return reranked


# =============================================================================
# MMR RE-RANKING
# =============================================================================
def mmr_rerank(item_scores: List[Tuple[int, float]],
               mcl_emb: Optional[np.ndarray],
               top_k: int = 20,
               lambda_div: float = DIVERSITY_LAMBDA) -> List[int]:
    """Greedy Maximal Marginal Relevance re-ranking -> ordered item ids.

    Vectorized over candidates: at each step the max-similarity of every
    remaining candidate to the selected set is computed as a single matmul.
    Numerically identical results to the per-pair dot-product loop.
    """
    if not item_scores:
        return []
    sorted_items = sorted(item_scores, key=lambda x: x[1], reverse=True)
    candidates = sorted_items[:top_k * 2]

    cand_items = np.array([c[0] for c in candidates], dtype=np.int64)
    cand_rel = np.array([c[1] for c in candidates], dtype=np.float64)

    # Embeddings for the candidate items (rows of mcl_emb). Items without an
    # embedding row (idx 0 or out of range) get a zero vector -> sim 0.
    if mcl_emb is not None:
        valid = (cand_items > 0) & (cand_items < mcl_emb.shape[0])
        cand_emb = np.zeros((len(cand_items), mcl_emb.shape[1]), dtype=np.float64)
        cand_emb[valid] = mcl_emb[cand_items[valid]]
    else:
        cand_emb = None

    active = np.ones(len(cand_items), dtype=bool)
    max_sim = np.zeros(len(cand_items), dtype=np.float64)  # vs selected set
    selected: List[int] = []

    while len(selected) < top_k and active.any():
        mmr_score = lambda_div * cand_rel - (1 - lambda_div) * max_sim
        mmr_score = np.where(active, mmr_score, -np.inf)
        pick = int(np.argmax(mmr_score))
        if not np.isfinite(mmr_score[pick]):
            break
        selected.append(int(cand_items[pick]))
        active[pick] = False
        # Update max_sim with the newly selected item.
        if cand_emb is not None:
            sim_to_new = cand_emb @ cand_emb[pick]
            np.maximum(max_sim, sim_to_new, out=max_sim)
    return selected


# =============================================================================
# FULL PIPELINE  (train-once + per-variant predict for efficiency)
# =============================================================================
def train_codt_assets(train_sessions_dict: dict,
                      n_items: int,
                      test_queries: dict,
                      item_categories: Optional[Dict[int, object]] = None,
                      item_freq: Optional[Counter] = None,
                      ensemble_seeds: Optional[List[int]] = None,
                      max_seq: int = PGSA_MAX_SEQ,
                      embed_dim: Optional[int] = None,
                      pgsa_epochs: Optional[int] = None,
                      mcl_epochs: Optional[int] = None,
                      head_type: Optional[str] = None,
                      n_negatives: int = 2048,
                      graph_aug: bool = False,
                      graph_aug_cfg: Optional[dict] = None,
                      contrastive: bool = False,
                      lambda_cl: float = 0.2,
                      csil: bool = False,
                      csil_alpha: float = 0.5,
                      csil_beta: float = 0.5,
                      siw: bool = False,
                      digital_twin: bool = False,
                      twin_config: Optional[dict] = None,
                      item_texts: Optional[Dict[int, str]] = None,
                      semantic_cache: Optional[str] = None) -> dict:
    """Train everything that is variant-independent ONCE per domain.

    head_type auto-selects: "softmax" for small vocabs (<2K items), "sim"
    (sampled-softmax embedding head) for large vocabs where full softmax is
    untrainable. Override with head_type=... to force.
    csil: Conditional Self-Information Loss — reweight the training loss by
          P(target|context) from the co-visitation graph. Tail items get higher
          gradient weight. Principled, not heuristic.
    """
    if ensemble_seeds is None:
        ensemble_seeds = list(ENSEMBLE_SEEDS)
    embed_dim = embed_dim or PGSA_EMBED_DIM
    pgsa_epochs = pgsa_epochs or PGSA_EPOCHS
    mcl_epochs = mcl_epochs or MCL_EPOCHS
    if head_type is None:
        head_type = "sim" if n_items > 2000 else "softmax"
    item_categories = item_categories or {}
    item_freq = item_freq or Counter()
    test_uids = sorted(test_queries.keys())

    sessions = [seq for seq in train_sessions_dict.values() if len(seq) >= 3]
    if not item_freq:
        for seq in sessions:
            item_freq.update(seq)

    fwd_cooc, bwd_cooc, pair_freq, pmi_cache = build_covisit_pmi(sessions, item_freq)
    cat_to_prods = build_category_cooc(sessions, item_categories) if item_categories else {}
    pop_penalty = build_popularity_penalty(item_freq)

    # CSIL: build conditional self-information weights from co-visitation graph.
    csil_weight_map = None
    if csil:
        csil_weight_map, csil_max_log_P = build_conditional_weights(
            sessions, fwd_cooc, bwd_cooc, alpha=csil_alpha, beta=csil_beta)
        print(f"  [CSIL] built {len(csil_weight_map)} context-prefix entries")

    # SIW: build global self-information weights.
    siw_weights = None
    if siw:
        siw_weights = build_siw_weights(item_freq)
        max_w = max(siw_weights.values()) if siw_weights else 0
        min_w = min(siw_weights.values()) if siw_weights else 0
        print(f"  [SIW] built per-item weights: min={min_w:.3f}, max={max_w:.3f} "
              f"(items: {len(siw_weights)})")

    print(f"  [CoDT assets] head={head_type}, embed={embed_dim}, "
          f"{len(ensemble_seeds)} seeds × {pgsa_epochs} ep, {len(sessions)} train sessions"
          f"{', contrastive=True' if contrastive else ''}"
          f"{', csil=True' if csil else ''}"
          f"{', siw=True' if siw else ''}")
    models = [train_pgsa(sessions, n_items, s, embed_dim=embed_dim, max_seq=max_seq,
                         epochs=pgsa_epochs, head_type=head_type, n_negatives=n_negatives,
                         contrastive=contrastive, lambda_cl=lambda_cl,
                         csil=csil, csil_weight_map=csil_weight_map,
                         csil_alpha=csil_alpha, csil_beta=csil_beta,
                         siw=siw, siw_weights=siw_weights)
              for s in ensemble_seeds]
    # Keep the behavior teacher in the same latent space as PGSA/Twin.  The
    # previous fixed 128-d M-CL silently disabled transfer on 64-d catalogs.
    mcl = train_mcl(sessions, n_items, embed_dim=embed_dim, epochs=mcl_epochs)
    mcl.eval()
    with torch.no_grad():
        mcl_emb = mcl.get_id_embeddings(torch.arange(n_items).to(DEVICE)).cpu().numpy()
    twin_teacher = mcl_emb
    semantic_embeddings = None
    if item_texts and semantic_cache:
        try:
            from .semantic_teacher import build_semantic_teacher
        except ImportError:
            from semantic_teacher import build_semantic_teacher
        from pathlib import Path
        twin_teacher = build_semantic_teacher(
            item_texts, n_items, mcl_emb, item_freq, Path(semantic_cache))
        # Keep the unprojected metadata geometry for direct candidate
        # generation.  Teacher fusion alone cannot retrieve unseen actions.
        semantic_embeddings = np.load(semantic_cache).astype(np.float32)
        print(f"  [SLM] fused E5 semantic teacher for {len(item_texts)} items")

    # Optional: GALORE-inspired co-visitation graph augmentation for tail items.
    # Done AFTER M-CL is trained (needs the embeddings). Only adds edges for
    # tail items, so head-item rankings are untouched (CoDT 0.43 is the floor).
    if graph_aug:
        fwd_cooc, bwd_cooc, pair_freq = augment_covisit_graph(
            fwd_cooc, bwd_cooc, pair_freq, item_freq, mcl_emb, **(graph_aug_cfg or {}))
        pmi_cache = rebuild_pmi_from_pairs(pair_freq, item_freq)
        extra = sum(len(v) for v in fwd_cooc.values())
        print(f"  [graph-aug] augmented co-visitation graph ({extra} total edges)")

    pgsa_preds = pgsa_ensemble_predict(models, test_uids, test_queries, n_items,
                                       max_seq=max_seq, exclude_ctx=True)
    # Factual anchor for the digital twin. Session-neighbour retrieval is very
    # strong in the sparse/short regime and, crucially, expands the intervention
    # support beyond candidates proposed by the parametric world model.
    retrieval_preds = None
    popularity_preds = None
    semantic_preds = None
    if digital_twin:
        try:
            from . import baselines as _baselines
        except ImportError:
            import baselines as _baselines
        retrieval_preds = _baselines.run_nonparametric(
            "SKNN", train_sessions_dict, test_queries, n_items)
        popularity_preds = _baselines.run_nonparametric(
            "MostPop", train_sessions_dict, test_queries, n_items)
        if semantic_embeddings is not None:
            try:
                from .semantic_teacher import semantic_retrieve
            except ImportError:
                from semantic_teacher import semantic_retrieve
            semantic_preds = semantic_retrieve(test_queries, semantic_embeddings,
                                               top_k=PGSA_TOP_K)
    # Compute popularity tiers ONCE (used by pop-adaptive fusion if enabled).
    pop_tiers = build_popularity_tiers(item_freq)

    twin = None
    if digital_twin:
        try:
            from .digital_twin import TwinConfig, train_digital_twin
        except ImportError:  # support legacy scripts that put sparse_bench on sys.path
            from digital_twin import TwinConfig, train_digital_twin
        twin = train_digital_twin(
            sessions, n_items, TwinConfig(**(twin_config or {})),
            teacher_embeddings=twin_teacher)

    return {
        "n_items": n_items,
        "item_categories": item_categories,
        "item_freq": item_freq,
        "test_queries": test_queries,
        "test_uids": test_uids,
        "pmi_cache": pmi_cache,
        "fwd_cooc": fwd_cooc,
        "cat_to_prods": cat_to_prods,
        "pop_penalty": pop_penalty,
        "pop_tiers": pop_tiers,
        "mcl_emb": mcl_emb,
        "pgsa_preds": pgsa_preds,
        "retrieval_preds": retrieval_preds,
        "popularity_preds": popularity_preds,
        "semantic_preds": semantic_preds,
        "semantic_embeddings": semantic_embeddings,
        "digital_twin": twin,
    }


def predict_codt(assets: dict, variant: str = "full",
                 ltar_config: Optional[dict] = None) -> Dict[str, List[int]]:
    """Score one variant from shared assets. No retraining.

    variant:
      "pgsa"        : PGSA ensemble + MMR (no fusion)            == DT-PGSA
      "full"        : full V3.2 fusion + MMR                      == DT-FullFusion
      "full_nommr"  : full fusion, no MMR (plain argmax)          == ablation
      "full+ltar"   : full fusion + LTAR tail rescue + MMR        == proposed
    """
    n_items = assets["n_items"]
    item_categories = assets["item_categories"]
    item_freq = assets["item_freq"]
    test_queries = assets["test_queries"]
    test_uids = assets["test_uids"]
    pmi_cache = assets["pmi_cache"]
    fwd_cooc = assets["fwd_cooc"]
    cat_to_prods = assets["cat_to_prods"]
    pop_penalty = assets["pop_penalty"]
    mcl_emb = assets["mcl_emb"]
    pgsa_preds = assets["pgsa_preds"]
    retrieval_preds = assets.get("retrieval_preds")
    popularity_preds = assets.get("popularity_preds")
    semantic_preds = assets.get("semantic_preds")
    pop_tiers = assets.get("pop_tiers", None)

    use_ltar = variant == "full+ltar"
    use_twin = variant in ("dualtwin", "dualtwin_nommr")
    twin = assets.get("digital_twin")
    if use_twin and twin is None:
        raise ValueError("dualtwin variant requires train_codt_assets(..., digital_twin=True)")
    if use_ltar:
        from ltar import compute_tiers, ltar_rerank
        head, mid, tail = compute_tiers(item_freq)
        ltar_kw = ltar_config or {}

    enable_fusion = variant in ("full", "full_nommr", "full+ltar", "dualtwin", "dualtwin_nommr")
    use_mmr = variant in ("pgsa", "full", "full+ltar", "dualtwin")

    predictions: Dict[str, List[int]] = {}
    for uid in tqdm(test_uids, desc=f"{variant}", leave=False):
        ctx = [x for x in test_queries[uid]["context"] if 1 <= x < n_items]
        last_prods = ctx[-5:]
        last_cats: List[object] = []
        for x in ctx:
            c = item_categories.get(x)
            if c is not None:
                last_cats.append(c)
        last_cats = last_cats[-3:]

        fused = fuse_one_query(pgsa_preds.get(uid, []), last_prods, last_cats,
                               pmi_cache, fwd_cooc, cat_to_prods, mcl_emb, pop_penalty,
                               enable_fusion=enable_fusion,
                               pop_tiers=pop_tiers)
        if use_twin:
            # Reciprocal-rank factual anchoring. The world model may change the
            # order, but it cannot silently discard high-support graph actions.
            graph_items = (retrieval_preds or {}).get(uid, [])[:PGSA_TOP_K]
            popular_items = (popularity_preds or {}).get(uid, [])[:PGSA_TOP_K]
            semantic_items = [item for item, _ in
                              (semantic_preds or {}).get(uid, [])[:PGSA_TOP_K]]
            neural_rank = {item: rank for rank, (item, _) in enumerate(fused, 1)}
            graph_rank = {item: rank for rank, item in enumerate(graph_items, 1)}
            pop_rank = {item: rank for rank, item in enumerate(popular_items, 1)}
            semantic_rank = {item: rank for rank, item in enumerate(semantic_items, 1)}
            union = set(neural_rank) | set(graph_rank) | set(pop_rank) | set(semantic_rank)
            anchored = []
            for item in union:
                nr = neural_rank.get(item)
                gr = graph_rank.get(item)
                pr = pop_rank.get(item)
                sr = semantic_rank.get(item)
                # Graph dominates in short contexts; neural evidence increases
                # smoothly as more observations synchronize the user twin.
                graph_weight = 0.80 * np.exp(-max(len(ctx) - 1, 0) / 8.0) + 0.10
                score = (graph_weight / (8.0 + gr) if gr is not None else 0.0)
                score += ((1.0 - graph_weight) / (8.0 + nr) if nr is not None else 0.0)
                # A catalog-size-aware factual world captures exposure priors
                # when session-neighbour support becomes extremely sparse.
                pop_weight = 0.55 if n_items > 10000 else 0.0
                score = (1.0 - pop_weight) * score
                score += (pop_weight / (8.0 + pr) if pr is not None else 0.0)
                # Metadata retrieval expands the counterfactual action support;
                # the twin, rather than a hard-coded semantic rank, decides
                # whether those actions survive the rollout.
                semantic_weight = 0.25 if semantic_items else 0.0
                score *= (1.0 - semantic_weight)
                score += (semantic_weight / (8.0 + sr) if sr is not None else 0.0)
                anchored.append((int(item), float(score)))
            fused = sorted(anchored, key=lambda x: x[1], reverse=True)
            fused = twin.rerank(str(uid), ctx, fused)
        if use_ltar:
            fused = ltar_rerank(fused, mcl_emb, last_prods, item_freq, n_items,
                                head, tail, **ltar_kw)
        if use_mmr:
            top = mmr_rerank(fused, mcl_emb, top_k=PGSA_TOP_K, lambda_div=DIVERSITY_LAMBDA)
        else:
            top = [it for it, _ in sorted(fused, key=lambda x: x[1], reverse=True)[:PGSA_TOP_K]]
        predictions[uid] = top
    return predictions


def predict_twin_residual(assets: dict, factual_predictions: Dict[str, List[int]],
                          use_mmr: bool = False) -> Dict[str, List[int]]:
    """Apply the learned counterfactual twin to a calibrated factual world.

    Candidate generation/fusion is deliberately external: it can be fitted on
    validation labels, frozen, and then supplied here.  The twin therefore
    learns only a residual intervention policy instead of first destroying a
    strong factual rank with hard-coded modality weights.
    """
    twin = assets.get("digital_twin")
    if twin is None:
        raise ValueError("digital_twin asset is required")
    queries = assets["test_queries"]
    embeddings = assets["mcl_emb"]
    output: Dict[str, List[int]] = {}
    for uid in tqdm(queries, desc="dualtwin_residual", leave=False):
        context = [x for x in queries[uid]["context"]
                   if 1 <= x < assets["n_items"]]
        ranking = factual_predictions.get(uid, [])
        candidates = [(int(item), 1.0 / (40.0 + rank))
                      for rank, item in enumerate(ranking, 1)]
        reranked = twin.rerank(str(uid), context, candidates)
        if use_mmr:
            output[uid] = mmr_rerank(reranked, embeddings, top_k=PGSA_TOP_K,
                                     lambda_div=DIVERSITY_LAMBDA)
        else:
            output[uid] = [item for item, _ in reranked[:PGSA_TOP_K]]
    return output


def run_codt(train_sessions_dict: dict,
             test_queries: dict,
             n_items: int,
             item_categories: Optional[Dict[int, object]] = None,
             item_freq: Optional[Counter] = None,
             variant: str = "full",
             ensemble_seeds: Optional[List[int]] = None,
             max_seq: int = PGSA_MAX_SEQ,
             embed_dim: Optional[int] = None,
             pgsa_epochs: Optional[int] = None,
             mcl_epochs: Optional[int] = None,
             ltar_config: Optional[dict] = None,
             assets: Optional[dict] = None,
             digital_twin: bool = False,
             twin_config: Optional[dict] = None) -> Dict[str, List[int]]:
    """Convenience: train assets (if not given) + predict one variant."""
    if assets is None:
        assets = train_codt_assets(train_sessions_dict, n_items, test_queries,
                                   item_categories=item_categories, item_freq=item_freq,
                                   ensemble_seeds=ensemble_seeds, max_seq=max_seq,
                                   embed_dim=embed_dim, pgsa_epochs=pgsa_epochs,
                                   mcl_epochs=mcl_epochs,
                                   digital_twin=digital_twin or variant.startswith("dualtwin"),
                                   twin_config=twin_config)
    return predict_codt(assets, variant=variant, ltar_config=ltar_config)
