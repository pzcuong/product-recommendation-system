"""CL-GRU4Rec+RP — Academic Evaluation (Synerise RecSys 2025)
============================================================

Per-user 80/20 split + K=10 evaluation + Extended metrics
Same model architecture as unified v4.
Reuses v4 cached models where possible.

Usage:
  python cl_gru4rec_rp_academic.py
"""
import os, pickle, time, random, gc, math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ============================================================================
# CONFIG
# ============================================================================
K = 10  # Standard academic cutoff

# GRU config (same architecture as v4)
GRU_EMBED_DIM = 128
GRU_HIDDEN_DIM = 200
GRU_DROPOUT = 0.15
GRU_MAX_SEQ = 50
GRU_BATCH = 256
GRU_EPOCHS = 25
GRU_LR = 0.001
GRU_SEEDS = [42, 123, 456]

# CL config
CL_EMBED_DIM = 64
CL_EPOCHS = 25
CL_LR = 0.003
CL_TEMP = 0.07
CL_NEG = 256
CL_BATCH = 1024

MIN_FREQ = 100   # Minimum item frequency
MIN_USER_EVENTS = 5  # Minimum user events


# ============================================================================
# METRICS (Academic Standard)
# ============================================================================
def recall_at_k(rec, gt, k):
    if not gt: return 0.0
    return len(set(rec[:k]) & set(gt)) / min(len(set(gt)), k)

def ndcg_at_k(rec, gt, k):
    gs = set(gt)
    dcg = sum(1.0/np.log2(i+2) for i, x in enumerate(rec[:k]) if x in gs)
    idcg = sum(1.0/np.log2(i+2) for i in range(min(len(gs), k)))
    return dcg/idcg if idcg > 0 else 0.0

def hit_at_k(rec, gt, k):
    return 1.0 if len(set(rec[:k]) & set(gt)) > 0 else 0.0

def evaluate(preds, gt, uids, k=10):
    rs = [recall_at_k(preds.get(u,[]), gt[u], k) for u in uids]
    ns = [ndcg_at_k(preds.get(u,[]), gt[u], k) for u in uids]
    hs = [hit_at_k(preds.get(u,[]), gt[u], k) for u in uids]
    return np.mean(rs), np.mean(ns), np.mean(hs)

def compute_novelty(preds, item_popularity, total_interactions, k=10):
    """Novelty = 1 - popularity^100 (competition formula)"""
    novelties = []
    for uid, recs in preds.items():
        for item in recs[:k]:
            pop = item_popularity.get(item, 0) / max(total_interactions, 1)
            novelties.append(1.0 - pop**100)
    return np.mean(novelties) if novelties else 0.0

def compute_diversity(preds, k=10):
    """Diversity = Entropy of prediction distribution"""
    all_items = []
    for uid, recs in preds.items():
        all_items.extend(recs[:k])
    if not all_items: return 0.0
    counts = Counter(all_items)
    total = sum(counts.values())
    probs = [c/total for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return entropy / max_entropy  # Normalized entropy [0, 1]

def compute_coverage(preds, all_items_set, k=10):
    """Catalog coverage: fraction of items that appear in recommendations"""
    rec_items = set()
    for uid, recs in preds.items():
        rec_items.update(recs[:k])
    return len(rec_items) / max(len(all_items_set), 1)


# ============================================================================
# MODEL: GRU4Rec (same as unified v4)
# ============================================================================
class GRU4RecModel(nn.Module):
    def __init__(self, n_items, embed_dim=128, hidden_dim=200, n_layers=1, 
                 dropout=0.15, pad_idx=0):
        super().__init__()
        self.n_items = n_items
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(n_items, embed_dim, padding_idx=pad_idx)
        nn.init.uniform_(self.embed.weight, -0.05, 0.05)
        self.embed.weight.data[pad_idx].zero_()
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.output = nn.Linear(hidden_dim, n_items)
    
    def forward(self, seq, lengths=None):
        x = self.drop(self.embed(seq))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            output, _ = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.gru(x)
        return self.output(self.drop(output))
    
    def predict(self, seq, lengths=None):
        self.eval()
        x = self.embed(seq)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(x)
        return self.output(hidden[-1])


class SeqDataset(Dataset):
    def __init__(self, sessions, max_len=50):
        self.sessions = [s for s in sessions if len(s) >= 3]
        self.max_len = max_len
    def __len__(self): return len(self.sessions)
    def __getitem__(self, idx):
        seq = self.sessions[idx]
        if len(seq) > self.max_len + 1:
            start = random.randint(0, len(seq) - self.max_len - 1)
            seq = seq[start:start + self.max_len + 1]
        return seq[:-1], seq[1:]

def collate_fn(batch):
    inputs, targets = zip(*batch)
    ml = max(len(s) for s in inputs)
    inp = torch.LongTensor([list(s) + [0]*(ml-len(s)) for s in inputs])
    tgt = torch.LongTensor([list(s) + [-1]*(ml-len(s)) for s in targets])
    lengths = torch.LongTensor([len(s) for s in inputs])
    return inp, lengths, tgt

def train_gru(sessions, n_items, seed, cache_path, config):
    if os.path.exists(cache_path):
        print(f"    Loading {cache_path}")
        model = GRU4RecModel(n_items, config['embed'], config['hidden'], 
                             dropout=config['dropout'])
        model.load_state_dict(torch.load(cache_path, map_location=DEVICE, weights_only=True))
        return model.to(DEVICE)
    
    print(f"    Training (seed={seed})...")
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    ds = SeqDataset(sessions, config['max_seq'])
    loader = DataLoader(ds, batch_size=config['batch'], shuffle=True,
                       collate_fn=collate_fn, num_workers=0, drop_last=True)
    model = GRU4RecModel(n_items, config['embed'], config['hidden'],
                         dropout=config['dropout']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss, nb = 0, 0
        pbar = tqdm(loader, desc=f"    Epoch {epoch+1}/{config['epochs']}", leave=False)
        for inp, lengths, tgt in pbar:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            logits = model(inp, lengths)
            B, L, V = logits.shape
            mask = tgt.reshape(-1) != -1
            loss = F.cross_entropy(logits.reshape(-1, V)[mask], tgt.reshape(-1)[mask])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); nb += 1
            pbar.set_postfix(loss=f"{total_loss/nb:.4f}")
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1}/{config['epochs']}: loss={total_loss/max(nb,1):.4f}")
    
    torch.save(model.state_dict(), cache_path)
    print(f"    Cached: {cache_path}")
    return model


# ============================================================================
# MODEL: Contrastive Item Similarity (same as unified v4)
# ============================================================================
class ContrastiveItemModel(nn.Module):
    def __init__(self, n_items, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2), nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim),
        )
    def forward(self, items):
        return F.normalize(self.projector(self.embedding(items)), dim=-1)

def train_cl(sessions, n_items, cache_path):
    if os.path.exists(cache_path):
        print(f"    Loading {cache_path}")
        model = ContrastiveItemModel(n_items, CL_EMBED_DIM)
        model.load_state_dict(torch.load(cache_path, map_location=DEVICE, weights_only=True))
        return model.to(DEVICE)
    
    print("    Building CL pairs...")
    pairs = []
    for sess in sessions:
        unique = list(set(sess))
        if len(unique) < 2: continue
        if len(unique) > 20:
            for _ in range(40):
                i, j = random.sample(range(len(unique)), 2)
                pairs.append((unique[i], unique[j]))
        else:
            for i in range(len(unique)):
                for j in range(i+1, len(unique)):
                    pairs.append((unique[i], unique[j]))
    random.shuffle(pairs)
    print(f"    {len(pairs):,} pairs from {len(sessions):,} sessions")
    
    model = ContrastiveItemModel(n_items, CL_EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CL_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CL_EPOCHS)
    all_items = list(range(n_items))
    
    for epoch in range(CL_EPOCHS):
        random.shuffle(pairs)
        total_loss, nb = 0, 0
        for i in range(0, len(pairs), CL_BATCH):
            batch = pairs[i:i+CL_BATCH]
            if len(batch) < 4: continue
            anc = torch.LongTensor([p[0] for p in batch]).to(DEVICE)
            pos = torch.LongTensor([p[1] for p in batch]).to(DEVICE)
            neg = torch.LongTensor(random.choices(all_items, k=min(CL_NEG, n_items))).to(DEVICE)
            za, zp, zn = model(anc), model(pos), model(neg)
            logits = torch.cat([(za*zp).sum(-1,keepdim=True)/CL_TEMP,
                               torch.mm(za, zn.t())/CL_TEMP], dim=1)
            loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=DEVICE))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); nb += 1
        scheduler.step()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1}/{CL_EPOCHS}: loss={total_loss/max(nb,1):.4f}")
    
    torch.save(model.state_dict(), cache_path)
    print(f"    Cached: {cache_path}")
    return model


# ============================================================================
# DATA LOADING: Per-user 80/20 split (reuse synerise_final.pkl)
# ============================================================================
def load_synerise_data():
    """Load Synerise data with per-user 80/20 split from existing cache."""
    CACHE = "synerise_final.pkl"
    if os.path.exists(CACHE):
        print(f"  Loading {CACHE}...")
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        # Add item_pop and total_interactions if missing
        if "item_pop" not in d:
            item_pop = Counter()
            for items in d["train_items"].values():
                item_pop.update(items)
            d["item_pop"] = dict(item_pop)
            d["total_interactions"] = sum(item_pop.values())
        return d
    else:
        raise FileNotFoundError(f"{CACHE} not found. Run cl_gru4rec_rp_unified.py --dataset synerise first.")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("CL-GRU4Rec+RP — Academic Evaluation (K=10, Per-User 80/20 Split)")
    print("=" * 80)
    
    # ---- Load Data ----
    print("\n1. LOADING DATA (Per-User 80/20 Split)")
    d = load_synerise_data()
    train_items = d["train_items"]; train_events = d["train_events"]
    test_gt = d["test_gt"]; cooccur = d["cooccur"]
    cat_pop = d.get("cat_pop", {}); s2c = d.get("s2c", {}); freq = d["freq"]
    item_pop = d.get("item_pop", {}); total_interactions = d.get("total_interactions", 1)
    
    test_uids = sorted(test_gt.keys())
    all_freq_items = sorted(freq)
    print(f"  {len(test_uids):,} test users, {len(freq):,} frequent items")
    
    # Build vocabulary
    item_to_idx = {"<PAD>": 0}
    for i, item in enumerate(all_freq_items):
        item_to_idx[item] = i + 1
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    n_items = len(item_to_idx)
    print(f"  Vocabulary: {n_items} items")
    
    # Build sessions
    gru_sessions, cl_sessions = [], []
    for uid, items in train_items.items():
        indices = [item_to_idx[s] for s in items if s in item_to_idx]
        if len(indices) >= 3: gru_sessions.append(indices)
        if len(set(indices)) >= 2: cl_sessions.append(indices)
    print(f"  GRU sessions: {len(gru_sessions):,} | CL sessions: {len(cl_sessions):,}")
    
    # ---- Baselines ----
    print("\n2. BASELINES")
    pop = Counter()
    for v in train_items.values(): pop.update(v)
    pop_top = [p for p, _ in pop.most_common(K)]
    r_pop, n_pop, h_pop = evaluate({u: pop_top for u in test_uids}, test_gt, test_uids, K)
    print(f"  Popularity:    R@{K}={r_pop:.4f} | NDCG@{K}={n_pop:.4f} | HR@{K}={h_pop:.4f}")
    
    preds_rp = {}
    for uid in test_uids:
        preds_rp[uid] = [p for p, _ in Counter(train_items[uid]).most_common(K)]
    r_rp, n_rp, h_rp = evaluate(preds_rp, test_gt, test_uids, K)
    print(f"  RePurchase:    R@{K}={r_rp:.4f} | NDCG@{K}={n_rp:.4f} | HR@{K}={h_rp:.4f}")
    
    # ---- Train GRU ----
    print("\n3. GRU4Rec TRAINING")
    gru_config = {'embed': GRU_EMBED_DIM, 'hidden': GRU_HIDDEN_DIM, 'dropout': GRU_DROPOUT,
                  'max_seq': GRU_MAX_SEQ, 'batch': GRU_BATCH, 'lr': GRU_LR, 'epochs': GRU_EPOCHS}
    gru_models = []
    for i, seed in enumerate(GRU_SEEDS):
        cache = f"peruser_gru_seed{seed}.pkl"
        print(f"  Model {i+1}/{len(GRU_SEEDS)}:")
        m = train_gru(gru_sessions, n_items, seed, cache, gru_config)
        gru_models.append(m)
    print(f"  Ensemble: {len(gru_models)} models")
    
    # ---- Train CL ----
    print("\n4. CONTRASTIVE LEARNING")
    cl_model = train_cl(cl_sessions, n_items, "peruser_cl.pkl")
    cl_model.eval()
    with torch.no_grad():
        chunk = 2000; embs = []
        for i in range(0, n_items, chunk):
            idx = torch.arange(i, min(i+chunk, n_items)).to(DEVICE)
            embs.append(cl_model(idx).cpu().numpy())
        cl_emb = np.vstack(embs)
    print(f"  CL embeddings: {cl_emb.shape}")
    
    # ---- GRU-only baseline ----
    print("\n5. GRU-ONLY BASELINE")
    preds_gru = {}
    for uid in tqdm(test_uids, desc="  GRU predict"):
        hist = train_items[uid]
        indices = [item_to_idx[s] for s in hist if s in item_to_idx]
        if not indices:
            preds_gru[uid] = pop_top[:K]; continue
        seq = torch.LongTensor([indices[-GRU_MAX_SEQ:]]).to(DEVICE)
        length = torch.LongTensor([min(len(indices), GRU_MAX_SEQ)])
        avg = np.zeros(n_items)
        for m in gru_models:
            m.eval()
            with torch.no_grad(): avg += m.predict(seq, length).squeeze(0).cpu().numpy()
        avg /= len(gru_models)
        avg[0] = -np.inf
        for idx in indices: avg[idx] = -np.inf
        top = np.argsort(avg)[-K:][::-1]
        preds_gru[uid] = [idx_to_item[i] for i in top if i in idx_to_item and idx_to_item[i] != "<PAD>"]
    r_gru, n_gru, h_gru = evaluate(preds_gru, test_gt, test_uids, K)
    print(f"  GRU-only:      R@{K}={r_gru:.4f} | NDCG@{K}={n_gru:.4f} | HR@{K}={h_gru:.4f}")
    
    # ---- CL-GRU4Rec+RP: Two-Stage Fusion ----
    print(f"\n6. ★ CL-GRU4Rec+RP (Two-Stage Fusion, K={K})")
    preds_best = {}
    for uid in tqdm(test_uids, desc="  CL-GRU+RP"):
        hist = train_items[uid]; evts = train_events[uid]; hist_set = set(hist)
        
        # Stage 1: Re-Purchase (buy-boosted + recency)
        rp_sc = Counter()
        for i, (item, evt) in enumerate(zip(hist, evts)):
            recency = 1.0 + (i / len(hist))
            w = 5.0 if evt == "buy" else 2.0
            rp_sc[item] += w * recency
        rp_top = [p for p, _ in rp_sc.most_common(K)]
        
        # Stage 2: Discovery (CL + GRU + CoOccur) for remaining slots
        if len(rp_top) < K:
            disc = Counter()
            # Co-occurrence
            for item in hist:
                if item in cooccur:
                    for pid, cnt in cooccur[item].items():
                        if pid not in hist_set: disc[pid] += cnt
            # CL similarity
            user_cl = [item_to_idx[s] for s in hist if s in item_to_idx]
            if user_cl:
                ue = cl_emb[user_cl[-10:]].mean(0)
                ue /= (np.linalg.norm(ue) + 1e-8)
                sims = cl_emb @ ue
                for ci in np.argsort(sims)[-30:][::-1]:
                    item = idx_to_item.get(ci, "")
                    if item and item != "<PAD>" and item not in hist_set and sims[ci] > 0.2:
                        disc[item] += (sims[ci] - 0.2) * 5.0
            # GRU sequential
            if user_cl:
                seq = torch.LongTensor([user_cl[-GRU_MAX_SEQ:]]).to(DEVICE)
                length = torch.LongTensor([min(len(user_cl), GRU_MAX_SEQ)])
                avg = np.zeros(n_items)
                for m in gru_models:
                    m.eval()
                    with torch.no_grad(): avg += m.predict(seq, length).squeeze(0).cpu().numpy()
                avg /= len(gru_models)
                for gi in np.argsort(avg)[-20:][::-1]:
                    item = idx_to_item.get(gi, "")
                    if item and item != "<PAD>" and item not in hist_set:
                        disc[item] += max(0, float(avg[gi])) * 0.5
            
            rp_set = set(rp_top)
            for p, _ in disc.most_common(K):
                if len(rp_top) >= K: break
                if p not in rp_set: rp_top.append(p); rp_set.add(p)
        
        preds_best[uid] = rp_top[:K]
    
    r_best, n_best, h_best = evaluate(preds_best, test_gt, test_uids, K)
    print(f"  ★ CL-GRU+RP:   R@{K}={r_best:.4f} | NDCG@{K}={n_best:.4f} | HR@{K}={h_best:.4f}")
    
    # ---- Extended Metrics ----
    print(f"\n7. EXTENDED METRICS (Novelty, Diversity, Coverage)")
    
    all_methods = {
        "Popularity": {u: pop_top for u in test_uids},
        "RePurchase": preds_rp,
        "GRU4Rec": preds_gru,
        "CL-GRU4Rec+RP": preds_best,
    }
    
    for name, preds in all_methods.items():
        nov = compute_novelty(preds, item_pop, total_interactions, K)
        div = compute_diversity(preds, K)
        cov = compute_coverage(preds, freq, K)
        r, n, h = evaluate(preds, test_gt, test_uids, K)
        # Competition composite score: 0.8*NDCG + 0.1*Novelty + 0.1*Diversity 
        # (adapted: using NDCG as proxy for AUROC)
        composite = 0.8 * n + 0.1 * nov + 0.1 * div
        print(f"  {name:20s}: Nov={nov:.4f} | Div={div:.4f} | Cov={cov:.4f} | Composite={composite:.4f}")
    
    # ---- Comparison Table ----
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"COMPARISON TABLE (Synerise RecSys 2025 — Per-User 80/20 Split, K={K})")
    print("=" * 80)
    results = [
        ("Popularity",      r_pop,  n_pop,  h_pop),
        ("RePurchase only",  r_rp,   n_rp,   h_rp),
        ("GRU4Rec only",     r_gru,  n_gru,  h_gru),
        ("★ CL-GRU4Rec+RP", r_best, n_best, h_best),
    ]
    best_r = max(x[1] for x in results)
    print(f"\n  {'Method':<20} | {'R@'+str(K):>9} | {'NDCG@'+str(K):>9} | {'HR@'+str(K):>9}")
    print(f"  {'-'*20}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for nm, r, n, h in results:
        mk = " ◀ BEST" if r == best_r else ""
        print(f"  {nm:<20} | {r:>9.4f} | {n:>9.4f} | {h:>9.4f}{mk}")
    
    print(f"\n  Split: Per-user 80/20 (leave-last-20%)")
    print(f"  ⏱️  {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Method: CL-GRU4Rec+RP (Unified PyTorch)")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    main()
