#!/usr/bin/env python3
"""
Kaggle GPU Notebook: SSM vs Transformer baselines on Diginetica + RetailRocket
=================================================================================

Run this on Kaggle with GPU (P100/T4 x2). 
1. First cell: clone repo + install deps
2. Second cell: data download + preprocessing
3. Third cell: train GRU4Rec, SASRec, SSM baselines
4. Fourth cell: evaluate and compare

Runtime estimate: ~2-3 hours on T4 GPU (4 seeds × 2 models × 2 datasets)
"""

# =============================================================================
# CELL 0: Setup
# =============================================================================
import os, sys, time, math, random, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Clone repo
if not Path("product-recommendation-system").exists():
    os.system("git clone https://github.com/YOUR_USERNAME/product-recommendation-system.git 2>/dev/null || echo 'clone failed, using local files'")
sys.path.insert(0, "product-recommendation-system/sparse_bench")

# Copy local modules if running from local
import importlib
for mod_name in ["codt_core", "ssm_model", "srgnn_preprocess", "loaders", "grouped_eval", "baselines"]:
    try:
        importlib.import_module(mod_name)
    except ImportError:
        print(f"Import {mod_name} failed — copy required")

# =============================================================================
# CELL 1: Data preprocessing — Diginetica
# =============================================================================
def prepare_diginetica():
    """Download + preprocess Diginetica. Returns (train_seqs, test_queries, n_items)."""
    data_dir = Path("datasets/diginetica")
    if not data_dir.exists():
        os.makedirs(data_dir, exist_ok=True)
        os.system("kaggle datasets download -d profalbusdumbledore/diginetica-dataset -p datasets/diginetica")
        os.system("unzip -o datasets/diginetica/diginetica-dataset.zip -d datasets/diginetica")

    df = pd.read_csv(data_dir / "train-item-views.csv", sep=";", dtype=str)
    df = df.dropna(subset=["sessionId", "itemId", "eventdate"])

    # Group by native sessionId
    sess_to_items = defaultdict(list)
    sess_to_date = {}
    for sid, iid, ed in zip(df["sessionId"], df["itemId"], df["eventdate"]):
        sess_to_items[sid].append(iid)
        sess_to_date[sid] = ed

    sessions = [(sess_to_date[sid], items) for sid, items in sess_to_items.items()
                if len(items) >= 2]

    # Item support filter (≥5)
    item_cnt = Counter()
    for _, items in sessions:
        item_cnt.update(items)
    valid = {it for it, c in item_cnt.items() if c >= 5}
    sessions = [(d, [it for it in items if it in valid]) for d, items in sessions
                if len([it for it in items if it in valid]) >= 2]

    # Temporal split (last 7 days = test)
    sessions.sort(key=lambda x: x[0])
    max_ord = max(sessions, key=lambda x: x[0])[0]
    import datetime as _dt
    def _ord(d):
        try: return _dt.date.fromisoformat(str(d)).toordinal()
        except: return 0
    cutoff_ord = max_ord - 7
    train_raw = [items for d, items in sessions if _ord(d) <= cutoff_ord]
    test_raw = [items for d, items in sessions if _ord(d) > cutoff_ord]

    # Build vocab
    all_items = sorted({it for sess in train_raw for it in sess})
    item2id = {it: i+1 for i, it in enumerate(all_items)}
    n_items = len(all_items) + 1

    train_ids = [[item2id[it] for it in s] for s in train_raw if len(s) >= 2]
    test_ids = [[item2id[it] for it in s] for s in test_raw if len(s) >= 2]

    train_dict = {str(i): seq for i, seq in enumerate(train_ids)}
    test_dict = {}
    item_freq = Counter()
    for seq in train_ids:
        item_freq.update(seq)
    for i, seq in enumerate(test_ids):
        if len(seq) < 2: continue
        tgt = seq[-1]
        if tgt in item_freq:
            test_dict[str(i)] = {"context": seq[:-1], "targets": [tgt]}

    print(f"Diginetica: train={len(train_dict)}, test={len(test_dict)}, items={n_items}")
    return train_dict, test_dict, n_items, item_freq


def prepare_retailrocket():
    """Load + sessionize RetailRocket."""
    import sys; sys.path.insert(0, "product-recommendation-system/sparse_bench")
    import srgnn_preprocess as sp
    return sp.load_retailrocket_srgnn(test_days=1)


# =============================================================================
# CELL 2: Model definitions (copied from sparse_bench for independence)
# =============================================================================
class GRU4Rec(nn.Module):
    def __init__(self, n_items, emb=128, h=128, L=1, dp=0.3):
        super().__init__()
        self.ie = nn.Embedding(n_items, emb, padding_idx=0)
        self.gru = nn.GRU(emb, h, L, batch_first=True, dropout=dp)
        self.n = nn.LayerNorm(h)
        self.hd = nn.Linear(h, n_items)

    def forward(self, x, ln=None):
        z, _ = self.gru(self.ie(x))
        z = self.n(z)
        if ln is not None:
            li = (ln - 1).clamp(min=0)
            return self.hd(z[torch.arange(z.size(0), device=z.device), li])
        return self.hd(z[:, -1])


class SASRec(nn.Module):
    def __init__(self, n_items, emb=128, hds=4, L=2, dp=0.3, ml=50):
        super().__init__()
        self.ie = nn.Embedding(n_items, emb, padding_idx=0)
        self.pe = nn.Embedding(ml, emb)
        e = nn.TransformerEncoderLayer(emb, hds, emb*4, dp, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(e, L)
        self.n = nn.LayerNorm(emb)
        self.hd = nn.Linear(emb, n_items)

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


class SelectiveSSMBlock(nn.Module):
    """Selective state-space block (pure PyTorch, no CUDA Mamba)."""
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_state = d_state
        self.proj_A = nn.Linear(d_model, d_state)
        self.proj_B = nn.Linear(d_model, d_state)
        self.proj_X = nn.Linear(d_model, d_state)
        self.proj_C = nn.Linear(d_state, d_model)
        self.proj_D = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.15)

    def forward(self, x, padding_mask=None):
        B, L, D = x.shape
        S = self.d_state
        A = torch.sigmoid(self.proj_A(x))
        Bb = F.softplus(self.proj_B(x))
        Xs = self.proj_X(x)
        Cc = self.proj_C
        h = x.new_zeros(B, S)
        ys = []
        for t in range(L):
            h = A[:, t] * h + Bb[:, t] * Xs[:, t]
            y_t = Cc(h)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        if padding_mask is not None:
            y = y.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        y = y * F.silu(self.proj_D(x))
        return self.norm(self.drop(y) + x)


class SSMSessionModel(nn.Module):
    """SSM session encoder."""
    def __init__(self, n_items, embed_dim=128):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.item_embed = nn.Embedding(n_items, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.item_embed.weight)
        self.item_embed.weight.data[0].zero_()
        # Feature-extract GRU (SIGMA-inspired)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.gru_norm = nn.LayerNorm(embed_dim)
        # SSM blocks
        self.blocks = nn.ModuleList([SelectiveSSMBlock(embed_dim) for _ in range(2)])
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def encode(self, seq, lengths):
        B, L = seq.shape
        x = self.item_embed(seq)
        positions = torch.arange(L, device=seq.device).unsqueeze(0)
        pad_mask = positions >= lengths.to(seq.device).unsqueeze(1)
        # Feature GRU
        c = self.conv(x.transpose(1, 2)).transpose(1, 2)
        g, _ = self.gru(c)
        x = self.gru_norm(g + x)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        # SSM blocks
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.norm(x)

    def last_hidden(self, seq, lengths):
        x = self.encode(seq, lengths)
        last_idx = (lengths.to(seq.device) - 1).clamp(min=0)
        return self.out_proj(x[torch.arange(x.size(0), device=x.device), last_idx])

    def forward(self, seq, lengths):
        hidden = self.last_hidden(seq, lengths)
        return hidden @ self.item_embed.weight.t()


def train_model(ModelClass, train_dict, n_items, epochs, lr, bs, device):
    """Generic training loop for any model with full-softmax head."""
    sessions = list(train_dict.values())
    ds = []
    for seq in sessions:
        seq = [x for x in seq if 1 <= x < n_items]
        for i in range(1, len(seq)):
            ds.append((seq[max(0, i-50):i], seq[i]))
    loader = DataLoader(ds, batch_size=bs, shuffle=True,
                        collate_fn=lambda b: (
                            torch.LongTensor([c+[0]*(max(len(c) for c,_ in b)-len(c)) for c,_ in b]),
                            torch.LongTensor([len(c) for c,_ in b]),
                            torch.LongTensor([t for _,t in b])
                        ))
    model = ModelClass(n_items).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        for inp, lens, tgt in loader:
            inp, lens, tgt = inp.to(device), lens.to(device), tgt.to(device)
            logits = model(inp, lens)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
    return model


def evaluate(model, test_queries, n_items, batch=256, device=DEVICE):
    """R@6, R@10, R@20 for model on test_queries."""
    model.eval()
    uids = sorted(test_queries.keys())
    hits6 = hits10 = hits20 = 0
    for bs in range(0, len(uids), batch):
        chunk = uids[bs:bs+batch]
        seqs = [[x for x in test_queries[uid]["context"] if 1 <= x < n_items][-50:] for uid in chunk]
        ml = max(max(len(s) for s in seqs), 1)
        inp = torch.zeros(len(chunk), ml, dtype=torch.long, device=device)
        ln = torch.zeros(len(chunk), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            inp[i, :len(s)] = torch.LongTensor(s)
            ln[i] = len(s)
        with torch.no_grad():
            logits = model(inp, ln)
        logits[logits == 0] = -1e9
        for i, uid in enumerate(chunk):
            sc = logits[i].cpu().numpy().copy()
            for c in set(seqs[i]): sc[c] = -1e9
            sc[0] = -1e9
            top = np.argsort(-sc)
            tgt = test_queries[uid]["targets"][0]
            if tgt in top[:6]: hits6 += 1
            if tgt in top[:10]: hits10 += 1
            if tgt in top[:20]: hits20 += 1
    n = len(uids)
    return {"R@6": hits6/n, "R@10": hits10/n, "R@20": hits20/n}


# =============================================================================
# CELL 3: Main experiment
# =============================================================================
def main():
    print("=" * 70)
    print("SSM vs Transformer baselines — Full-scale DIGINETICA + RETAILROCKET")
    print("=" * 70)

    results_file = Path("results.json")
    all_results = {}

    for dataset_name, (train_dict, test_dict, n_items, item_freq) in [
        ("Diginetica", prepare_diginetica()),
        ("RetailRocket", None)  # placeholder
    ]:
        print(f"\n{'='*70}\n{dataset_name}\n{'='*70}")

        # Subsample test if too large
        import random as _r
        if len(test_dict) > 5000:
            keep = set(_r.Random(0).sample(sorted(test_dict.keys()), 5000))
            test_dict = {k: v for k, v in test_dict.items() if k in keep}

        n = n_items
        print(f"  items={n}, train={len(train_dict)}, test={len(test_dict)}")

        # Baseline: MostPop
        pop_items = [x for x, _ in item_freq.most_common(100)]
        hits = sum(1 for q in test_dict.values() if q["targets"][0] in pop_items[:20])
        print(f"  MostPop R@20={hits/len(test_dict):.4f}")

        # Run models with 4 seeds
        for ModelClass, name, ep, lr, bs in [
            (GRU4Rec, "GRU4Rec", 10, 1e-3, 512),
            (SASRec, "SASRec", 10, 1e-3, 512),
            (SSMSessionModel, "SSM", 10, 1e-3, 256),  # SSM needs smaller batch
        ]:
            seeds = [42, 123, 456, 789]
            r_all = []
            t0 = time.time()
            for s in seeds:
                print(f"  [{name}] seed {s}...", end=" ", flush=True)
                torch.manual_seed(s); random.seed(s); np.random.seed(s)
                model = train_model(ModelClass, train_dict, n, ep, lr, bs, DEVICE)
                r = evaluate(model, test_dict, n, device=DEVICE)
                r_all.append(r)
                print(f"R@20={r['R@20']:.4f}")
            mean = {k: np.mean([r[k] for r in r_all]) for k in ["R@6","R@10","R@20"]}
            print(f"  {name}: R@6={mean['R@6']:.4f} R@10={mean['R@10']:.4f} R@20={mean['R@20']:.4f}")
            all_results[f"{dataset_name}_{name}"] = mean
            print(f"    time: {time.time()-t0:.0f}s")

        print()

    # Save
    Path("results.json").write_text(json.dumps(all_results, indent=2))
    print("\n" + "=" * 70)
    print("ALL DONE. Results saved to results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
