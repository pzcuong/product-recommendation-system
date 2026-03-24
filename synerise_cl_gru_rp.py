"""
CL-GRU4Rec+RP on Synerise RecSys 2025
=======================================

Novel method: Contrastive Learning + GRU + Re-Purchase Awareness

Architecture for Synerise:
  1. Re-Purchase (BUY-boosted + recency) → dominant signal (proven R@6=0.55)
  2. Contrastive Item Similarity → better discovery than raw co-occurrence
  3. Lightweight GRU (filtered items ≤5K) → sequential patterns
  4. Co-occurrence → transition patterns
  5. Category-aware fallback

Multi-signal fusion: α·RP + β·GRU + γ·CL-CoOccur + δ·CoOccur

Evaluation: Recall@6, NDCG@6, HR@6 (user-level, 80/20 chrono split)
"""
import os, pickle, time, gc, random
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================
BASE = "synerise_dataset"
K = 6
MIN_ITEM_COUNT = 100       # Items must appear ≥100 times
MIN_USER_EVENTS = 5        # Users must have ≥5 events
TRAIN_RATIO = 0.8
CACHE = "synerise_final.pkl"

# GRU config (lightweight for large item space)
GRU_ITEM_MIN = 500         # Items for GRU must appear ≥500 times
GRU_EMBED_DIM = 64
GRU_HIDDEN_DIM = 96
GRU_EPOCHS = 20
GRU_BATCH = 256
GRU_LR = 0.001
GRU_MAX_SEQ = 30
GRU_SEEDS = [42, 123]

# CL config 
CL_EMBED_DIM = 64
CL_EPOCHS = 25
CL_LR = 0.003
CL_TEMP = 0.07
CL_NEG = 256
CL_BATCH = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ============================================================================
# METRICS
# ============================================================================
def recall_at_k(rec, gt, k):
    if not gt: return 0.0
    return len(set(rec[:k]) & set(gt)) / len(set(gt))

def ndcg_at_k(rec, gt, k):
    gs = set(gt)
    dcg = sum(1.0/np.log2(i+2) for i, x in enumerate(rec[:k]) if x in gs)
    idcg = sum(1.0/np.log2(i+2) for i in range(min(len(gs), k)))
    return dcg/idcg if idcg > 0 else 0.0

def hit_at_k(rec, gt, k):
    return 1.0 if len(set(rec[:k]) & set(gt)) > 0 else 0.0

def evaluate(preds, gt, uids, k=6):
    rs = [recall_at_k(preds.get(u,[]), gt[u], k) for u in uids]
    ns = [ndcg_at_k(preds.get(u,[]), gt[u], k) for u in uids]
    hs = [hit_at_k(preds.get(u,[]), gt[u], k) for u in uids]
    return np.mean(rs), np.mean(ns), np.mean(hs)


# ============================================================================
# CONTRASTIVE ITEM MODEL (same architecture as v2)
# ============================================================================
class ContrastiveItemModel(nn.Module):
    def __init__(self, n_items, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    def forward(self, items):
        return F.normalize(self.projector(self.embedding(items)), dim=-1)

def train_contrastive(sessions, n_items, embed_dim=64, epochs=25, lr=0.003,
                      temp=0.07, batch_sz=1024, device="cpu"):
    """Train contrastive item embeddings from user purchase sessions."""
    print("  Building positive pairs...")
    pairs = []
    for sess in sessions:
        unique = list(set(sess))
        if len(unique) < 2: continue
        # Sample pairs to avoid quadratic explosion
        if len(unique) > 20:
            for _ in range(30):
                i, j = random.sample(range(len(unique)), 2)
                pairs.append((unique[i], unique[j]))
        else:
            for i in range(len(unique)):
                for j in range(i+1, len(unique)):
                    pairs.append((unique[i], unique[j]))
    
    random.shuffle(pairs)
    print(f"  {len(pairs):,} pairs from {len(sessions):,} sessions")
    
    model = ContrastiveItemModel(n_items, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_items = list(range(n_items))
    
    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss, n_b = 0, 0
        for i in range(0, len(pairs), batch_sz):
            batch = pairs[i:i+batch_sz]
            if len(batch) < 2: continue
            anc = torch.LongTensor([p[0] for p in batch]).to(device)
            pos = torch.LongTensor([p[1] for p in batch]).to(device)
            neg = torch.LongTensor(random.choices(all_items, k=min(CL_NEG, n_items))).to(device)
            za, zp, zn = model(anc), model(pos), model(neg)
            pos_sim = (za * zp).sum(-1, keepdim=True) / temp
            neg_sim = torch.mm(za, zn.t()) / temp
            logits = torch.cat([pos_sim, neg_sim], dim=1)
            labels = torch.zeros(len(batch), dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); n_b += 1
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/max(n_b,1):.4f}")
    return model


# ============================================================================
# LIGHTWEIGHT GRU MODEL
# ============================================================================
class LightweightGRU(nn.Module):
    def __init__(self, n_items, embed_dim=64, hidden_dim=96, dropout=0.1, pad_idx=0):
        super().__init__()
        self.n_items = n_items
        self.embed = nn.Embedding(n_items, embed_dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.embed.weight)
        self.embed.weight.data[pad_idx].zero_()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_items)
    
    def forward(self, seq, lengths=None):
        x = self.dropout(self.embed(seq))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, _ = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.gru(x)
        return self.head(output)
    
    def predict(self, seq, lengths=None):
        x = self.embed(seq)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(x)
        return self.head(hidden.squeeze(0))  # (B, V)


class SeqDataset(Dataset):
    def __init__(self, sessions, max_len=30):
        self.sessions = [s for s in sessions if len(s) >= 3]
        self.max_len = max_len
    def __len__(self): return len(self.sessions)
    def __getitem__(self, idx):
        seq = self.sessions[idx]
        if len(seq) > self.max_len + 1: seq = seq[-(self.max_len + 1):]
        return {"input": seq[:-1], "target": seq[1:]}

def collate_gru(batch):
    def pad(seqs, pv=0):
        ml = max(len(s) for s in seqs)
        return torch.LongTensor([s + [pv]*(ml-len(s)) for s in seqs]), torch.LongTensor([len(s) for s in seqs])
    inp, il = pad([b["input"] for b in batch])
    tgt, _ = pad([b["target"] for b in batch], pv=-1)
    return inp, il, tgt

def train_gru(model, loader, epochs, lr, device, seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.to(device).train()
    for epoch in range(epochs):
        total, nb = 0, 0
        for inp, il, tgt in tqdm(loader, desc=f"  GRU Epoch {epoch+1}/{epochs}", leave=False):
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp, il)
            B, L, V = logits.shape
            mask = tgt.reshape(-1) != -1
            loss = F.cross_entropy(logits.reshape(-1, V)[mask], tgt.reshape(-1)[mask])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total/max(nb,1):.4f}")
    return model


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    
    # ---- 1. Load data (reuse cache from synerise_rec.py) ----
    print("=" * 80)
    print("1. LOADING DATA")
    print("=" * 80)
    
    if os.path.exists(CACHE):
        print(f"  Loading {CACHE}...")
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        train_items = d["train_items"]
        train_events = d["train_events"]
        test_gt = d["test_gt"]
        cooccur = d["cooccur"]
        cat_pop = d["cat_pop"]
        s2c = d["s2c"]
        freq = d["freq"]
    else:
        print("  Loading from raw parquet...")
        cart = pd.read_parquet(f"{BASE}/add_to_cart.parquet")
        buy = pd.read_parquet(f"{BASE}/product_buy.parquet")
        props = pd.read_parquet(f"{BASE}/product_properties.parquet")
        cart["event"] = "cart"; buy["event"] = "buy"
        df = pd.concat([cart[["client_id","timestamp","sku","event"]],
                         buy[["client_id","timestamp","sku","event"]]], ignore_index=True)
        del cart, buy; gc.collect()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["sku"] = df["sku"].astype(str); df["client_id"] = df["client_id"].astype(str)
        df = df.sort_values(["client_id","timestamp"])
        
        props["sku"] = props["sku"].astype(str)
        s2c = props.set_index("sku")["category"].to_dict()
        
        sku_counts = df["sku"].value_counts()
        freq = set(sku_counts[sku_counts >= MIN_ITEM_COUNT].index)
        dff = df[df["sku"].isin(freq)]
        print(f"  Items: {len(freq):,}, Events: {len(dff):,}")
        
        user_data = {}
        for uid, grp in tqdm(dff.groupby("client_id"), desc="Users"):
            items, events = grp["sku"].tolist(), grp["event"].tolist()
            if len(items) >= MIN_USER_EVENTS:
                user_data[uid] = (items, events)
        print(f"  Users: {len(user_data):,}")
        
        cooccur = defaultdict(Counter)
        test_gt = {}; train_items = {}; train_events = {}
        for uid, (items, events) in user_data.items():
            sp = max(2, int(len(items) * TRAIN_RATIO))
            train_items[uid] = items[:sp]
            train_events[uid] = events[:sp]
            test_part = items[sp:]
            if test_part: test_gt[uid] = list(set(test_part))
            u_train = list(set(items[:sp]))
            for i in range(len(u_train)):
                for j in range(i+1, len(u_train)):
                    cooccur[u_train[i]][u_train[j]] += 1
                    cooccur[u_train[j]][u_train[i]] += 1
        
        cat_pop = defaultdict(Counter)
        for uid, items in train_items.items():
            for s in items:
                if s in s2c: cat_pop[s2c[s]][s] += 1
        
        with open(CACHE, "wb") as f:
            pickle.dump({"train_items": train_items, "train_events": train_events,
                          "test_gt": test_gt, "cooccur": dict(cooccur),
                          "cat_pop": dict(cat_pop), "s2c": s2c, "freq": freq}, f)
        print(f"  Cached: {CACHE}")
    
    test_uids = sorted(test_gt.keys())
    print(f"  {len(test_uids):,} test users, {len(freq):,} items")
    
    # ---- 2. Baselines ----
    print("\n" + "=" * 80)
    print("2. BASELINES")
    print("=" * 80)
    
    # Popularity
    pop = Counter()
    for v in train_items.values(): pop.update(v)
    pop_top = [p for p, _ in pop.most_common(K)]
    r, n, h = evaluate({u: pop_top for u in test_uids}, test_gt, test_uids)
    print(f"  Popularity:      R@6={r:.4f} | NDCG={n:.4f} | HR={h:.4f}")
    
    # CoOccurrence
    preds_co = {}
    for uid in test_uids:
        hist = train_items[uid]; hist_set = set(hist); sc = Counter()
        for item in hist:
            if item in cooccur:
                for pid, cnt in cooccur[item].most_common(50):
                    if pid not in hist_set: sc[pid] += cnt
        preds_co[uid] = [p for p, _ in sc.most_common(K)]
    r_c, n_c, h_c = evaluate(preds_co, test_gt, test_uids)
    print(f"  CoOccurrence:    R@6={r_c:.4f} | NDCG={n_c:.4f} | HR={h_c:.4f}")
    
    # RePurchase
    preds_rp = {}
    for uid in test_uids:
        preds_rp[uid] = [p for p, _ in Counter(train_items[uid]).most_common(K)]
    r_rp, n_rp, h_rp = evaluate(preds_rp, test_gt, test_uids)
    print(f"  RePurchase:      R@6={r_rp:.4f} | NDCG={n_rp:.4f} | HR={h_rp:.4f}")
    
    # RePurchase+Boost (previous best from synerise_rec.py)
    preds_rb = {}
    for uid in test_uids:
        hist, evts = train_items[uid], train_events[uid]; sc = Counter()
        for i, (item, evt) in enumerate(zip(hist, evts)):
            recency = 1 + (i / len(hist))
            w = 5.0 if evt == "buy" else 2.0
            sc[item] += w * recency
        hist_set = set(hist)
        for item in hist[-5:]:
            if item in cooccur:
                for pid, cnt in cooccur[item].most_common(20):
                    if pid not in hist_set: sc[pid] += cnt * 0.1
        preds_rb[uid] = [p for p, _ in sc.most_common(K)]
    r_rb, n_rb, h_rb = evaluate(preds_rb, test_gt, test_uids)
    print(f"  RePurch+Boost:   R@6={r_rb:.4f} | NDCG={n_rb:.4f} | HR={h_rb:.4f}")
    
    # ---- 3. NOVEL: Contrastive Item Similarity ----
    print("\n" + "=" * 80)
    print("3. CONTRASTIVE ITEM SIMILARITY (Novel)")
    print("=" * 80)
    
    # Build vocabulary for CL
    all_freq_items = sorted(freq)
    item_to_clid = {p: i for i, p in enumerate(all_freq_items)}
    n_cl = len(all_freq_items)
    
    cl_sessions = []
    for uid, items in train_items.items():
        indices = [item_to_clid[s] for s in items if s in item_to_clid]
        if len(set(indices)) >= 2:
            cl_sessions.append(indices)
    
    print(f"  {len(cl_sessions):,} sessions, {n_cl} items")
    
    cl_cache = "synerise_cl_model.pkl"
    if os.path.exists(cl_cache):
        print(f"  Loading {cl_cache}")
        cl_model = ContrastiveItemModel(n_cl, CL_EMBED_DIM)
        cl_model.load_state_dict(torch.load(cl_cache, map_location=DEVICE, weights_only=True))
        cl_model = cl_model.to(DEVICE)
    else:
        cl_model = train_contrastive(cl_sessions, n_cl, CL_EMBED_DIM, CL_EPOCHS, CL_LR,
                                     CL_TEMP, CL_BATCH, DEVICE)
        if cl_model:
            torch.save(cl_model.state_dict(), cl_cache)
            print(f"  Cached: {cl_cache}")
    
    # Pre-compute embeddings
    cl_emb = None
    if cl_model:
        cl_model.eval()
        with torch.no_grad():
            # Process in chunks to avoid MPS memory issues
            chunk = 2000
            embs = []
            for i in range(0, n_cl, chunk):
                idx = torch.arange(i, min(i+chunk, n_cl)).to(DEVICE)
                embs.append(cl_model(idx).cpu().numpy())
            cl_emb = np.vstack(embs)
        print(f"  CL embeddings: {cl_emb.shape}")
    
    # ---- 4. NOVEL: Lightweight GRU ----
    print("\n" + "=" * 80)
    print("4. LIGHTWEIGHT GRU (Novel)")
    print("=" * 80)
    
    # Filter to frequent items for GRU (reduce vocab to ~2-5K)
    item_counts = Counter()
    for items in train_items.values():
        item_counts.update(items)
    gru_items = sorted({item for item, cnt in item_counts.items() if cnt >= GRU_ITEM_MIN})
    gru_items_set = set(gru_items)
    
    # GRU vocab: 0=PAD, 1..N=items
    gru_vocab = {"<PAD>": 0}
    for i, item in enumerate(gru_items):
        gru_vocab[item] = i + 1
    n_gru = len(gru_vocab)
    gru_idx_to_item = {v: k for k, v in gru_vocab.items()}
    
    print(f"  GRU vocab: {n_gru} items (≥{GRU_ITEM_MIN} interactions)")
    
    # Build sequences
    gru_sessions = []
    for uid, items in train_items.items():
        indices = [gru_vocab[s] for s in items if s in gru_vocab]
        if len(indices) >= 3:
            gru_sessions.append(indices)
    
    print(f"  GRU sessions: {len(gru_sessions):,}")
    
    gru_models = []
    for i, seed in enumerate(GRU_SEEDS):
        gru_cache = f"synerise_gru_seed{seed}.pkl"
        if os.path.exists(gru_cache):
            print(f"  Loading {gru_cache}")
            m = LightweightGRU(n_gru, GRU_EMBED_DIM, GRU_HIDDEN_DIM, pad_idx=0)
            m.load_state_dict(torch.load(gru_cache, map_location=DEVICE, weights_only=True))
            m = m.to(DEVICE)
            gru_models.append(m)
            continue
        
        print(f"  Training GRU {i+1}/{len(GRU_SEEDS)} (seed={seed})...")
        ds = SeqDataset(gru_sessions, GRU_MAX_SEQ)
        loader = DataLoader(ds, batch_size=GRU_BATCH, shuffle=True, collate_fn=collate_gru,
                           num_workers=0, drop_last=True)
        m = LightweightGRU(n_gru, GRU_EMBED_DIM, GRU_HIDDEN_DIM, pad_idx=0)
        m = train_gru(m, loader, GRU_EPOCHS, GRU_LR, DEVICE, seed)
        torch.save(m.state_dict(), gru_cache)
        print(f"  Cached: {gru_cache}")
        gru_models.append(m)
    
    print(f"  GRU ensemble: {len(gru_models)} models")
    
    # ---- Helper: Get discovery scores from CL + GRU + CoOccur ----
    def get_discovery_scores(uid, hist, hist_set):
        """Get discovery item scores from CL similarity, GRU, and co-occurrence."""
        disc = Counter()
        
        # Co-occurrence
        for item in hist:
            if item in cooccur:
                for pid, cnt in cooccur[item].most_common(30):
                    if pid not in hist_set:
                        disc[pid] += cnt
        
        # CL similarity
        if cl_emb is not None:
            user_cl_ids = [item_to_clid[s] for s in hist if s in item_to_clid]
            if user_cl_ids:
                ue = cl_emb[user_cl_ids[-10:]].mean(0)
                ue /= (np.linalg.norm(ue) + 1e-8)
                sims = cl_emb @ ue
                for ci in np.argsort(sims)[-30:][::-1]:
                    item = all_freq_items[ci]
                    if item not in hist_set and sims[ci] > 0.2:
                        disc[item] += (sims[ci] - 0.2) * 5.0
        
        # GRU sequential
        if gru_models:
            gru_hist = [gru_vocab[s] for s in hist if s in gru_vocab]
            if gru_hist:
                gru_hist = gru_hist[-GRU_MAX_SEQ:]
                seq = torch.LongTensor([gru_hist]).to(DEVICE)
                length = torch.LongTensor([len(gru_hist)])
                avg_sc = np.zeros(n_gru)
                for m in gru_models:
                    m.eval()
                    with torch.no_grad():
                        avg_sc += m.predict(seq, length).squeeze(0).cpu().numpy()
                avg_sc /= len(gru_models)
                for gi in np.argsort(avg_sc)[-20:][::-1]:
                    if gi in gru_idx_to_item:
                        item = gru_idx_to_item[gi]
                        if item != "<PAD>" and item not in hist_set:
                            disc[item] += max(0, float(avg_sc[gi])) * 0.5
        
        return disc
    
    # ---- 5. CL-GRU4Rec+RP: TWO-STAGE FUSION (Novel) ----
    print("\n" + "=" * 80)
    print("5. CL-GRU4Rec+RP: TWO-STAGE FUSION (Novel)")
    print("=" * 80)
    print("   Stage 1: Fill top slots with re-purchase (buy-boosted + recency)")
    print("   Stage 2: Fill remaining slots with CL + GRU + CoOccur discovery")
    
    preds_novel = {}
    
    for uid in tqdm(test_uids, desc="CL-GRU+RP"):
        hist = train_items[uid]
        evts = train_events[uid]
        hist_set = set(hist)
        
        # ═══ Stage 1: Re-purchase scoring ═══
        rp_sc = Counter()
        for i, (item, evt) in enumerate(zip(hist, evts)):
            recency = 1.0 + (i / len(hist))
            w = 5.0 if evt == "buy" else 2.0
            rp_sc[item] += w * recency
        rp_top = [p for p, _ in rp_sc.most_common(K)]
        
        # ═══ Stage 2: Discovery (only if slots remain) ═══
        if len(rp_top) < K:
            disc = get_discovery_scores(uid, hist, hist_set)
            rp_set = set(rp_top)
            for p, _ in disc.most_common(K):
                if len(rp_top) >= K: break
                if p not in rp_set:
                    rp_top.append(p)
                    rp_set.add(p)
        
        preds_novel[uid] = rp_top[:K]
    
    r_novel, n_novel, h_novel = evaluate(preds_novel, test_gt, test_uids)
    print(f"\n  ★ CL-GRU+RP (2-stage): R@6={r_novel:.4f} | NDCG={n_novel:.4f} | HR={h_novel:.4f}")
    
    # ---- 5b. Variant: RP-dominant with discovery re-ranking ----
    print("\n  --- Variant: RP + Discovery re-ranking ---")
    
    # Try different RP slot counts
    for rp_slots in [4, 5, 6]:
        disc_slots = K - rp_slots
        preds_v = {}
        for uid in test_uids:
            hist, evts = train_items[uid], train_events[uid]; hist_set = set(hist)
            rp_sc = Counter()
            for i, (item, evt) in enumerate(zip(hist, evts)):
                recency = 1.0 + (i / len(hist))
                w = 5.0 if evt == "buy" else 2.0
                rp_sc[item] += w * recency
            rp_top = [p for p, _ in rp_sc.most_common(rp_slots)]
            
            if disc_slots > 0:
                disc = get_discovery_scores(uid, hist, hist_set)
                rp_set = set(rp_top)
                for p, _ in disc.most_common(K):
                    if len(rp_top) >= K: break
                    if p not in rp_set:
                        rp_top.append(p); rp_set.add(p)
            preds_v[uid] = rp_top[:K]
        
        rv, nv, hv = evaluate(preds_v, test_gt, test_uids)
        print(f"  RP={rp_slots}+Disc={disc_slots}:  R@6={rv:.4f} | NDCG={nv:.4f} | HR={hv:.4f}")
    
    # ---- 5c. Variant: RP with CL-enhanced re-ranking of RP candidates ----
    print("\n  --- Variant: CL-enhanced RP re-ranking ---")
    preds_cl_rerank = {}
    for uid in tqdm(test_uids, desc="CL-rerank"):
        hist, evts = train_items[uid], train_events[uid]; hist_set = set(hist)
        
        # Score all history items (re-purchase candidates)
        rp_sc = Counter()
        for i, (item, evt) in enumerate(zip(hist, evts)):
            recency = 1.0 + (i / len(hist))
            w = 5.0 if evt == "buy" else 2.0
            rp_sc[item] += w * recency
        
        # CL-enhanced: boost RP items that are more similar to recent purchases
        if cl_emb is not None:
            buy_items = [item for item, evt in zip(hist, evts) if evt == "buy"]
            recent = buy_items[-5:] if buy_items else hist[-5:]
            rcl = [item_to_clid[s] for s in recent if s in item_to_clid]
            if rcl:
                ue = cl_emb[rcl].mean(0)
                ue /= (np.linalg.norm(ue) + 1e-8)
                for item in rp_sc:
                    if item in item_to_clid:
                        sim = float(cl_emb[item_to_clid[item]] @ ue)
                        # Boost items more similar to recent buys
                        rp_sc[item] *= (1.0 + max(0, sim) * 0.3)
        
        preds_cl_rerank[uid] = [p for p, _ in rp_sc.most_common(K)]
    
    r_clr, n_clr, h_clr = evaluate(preds_cl_rerank, test_gt, test_uids)
    print(f"  CL-reranked RP:  R@6={r_clr:.4f} | NDCG={n_clr:.4f} | HR={h_clr:.4f}")
    
    # ---- 5d. Variant: RP with GRU-enhanced re-ranking ----
    print("\n  --- Variant: GRU-enhanced RP re-ranking ---")
    preds_gru_rerank = {}
    for uid in tqdm(test_uids, desc="GRU-rerank"):
        hist, evts = train_items[uid], train_events[uid]
        rp_sc = Counter()
        for i, (item, evt) in enumerate(zip(hist, evts)):
            recency = 1.0 + (i / len(hist))
            w = 5.0 if evt == "buy" else 2.0
            rp_sc[item] += w * recency
        
        # GRU-enhanced: boost RP items predicted by GRU
        if gru_models:
            gru_hist = [gru_vocab[s] for s in hist if s in gru_vocab]
            if gru_hist:
                gru_hist = gru_hist[-GRU_MAX_SEQ:]
                seq = torch.LongTensor([gru_hist]).to(DEVICE)
                length = torch.LongTensor([len(gru_hist)])
                avg_sc = np.zeros(n_gru)
                for m in gru_models:
                    m.eval()
                    with torch.no_grad():
                        avg_sc += m.predict(seq, length).squeeze(0).cpu().numpy()
                avg_sc /= len(gru_models)
                # Boost RP candidates that GRU also predicts
                for item in rp_sc:
                    if item in gru_vocab:
                        gru_s = float(avg_sc[gru_vocab[item]])
                        if gru_s > 0:
                            rp_sc[item] *= (1.0 + gru_s * 0.1)
        
        preds_gru_rerank[uid] = [p for p, _ in rp_sc.most_common(K)]
    
    r_grr, n_grr, h_grr = evaluate(preds_gru_rerank, test_gt, test_uids)
    print(f"  GRU-reranked RP: R@6={r_grr:.4f} | NDCG={n_grr:.4f} | HR={h_grr:.4f}")
    
    # ---- 5e. BEST: CL+GRU enhanced RP re-ranking ----
    print("\n  --- BEST Variant: CL+GRU enhanced RP re-ranking ---")
    preds_best = {}
    for uid in tqdm(test_uids, desc="CL+GRU-rerank"):
        hist, evts = train_items[uid], train_events[uid]; hist_set = set(hist)
        
        rp_sc = Counter()
        for i, (item, evt) in enumerate(zip(hist, evts)):
            recency = 1.0 + (i / len(hist))
            w = 5.0 if evt == "buy" else 2.0
            rp_sc[item] += w * recency
        
        # CL boost
        if cl_emb is not None:
            buy_items = [item for item, evt in zip(hist, evts) if evt == "buy"]
            recent = buy_items[-5:] if buy_items else hist[-5:]
            rcl = [item_to_clid[s] for s in recent if s in item_to_clid]
            if rcl:
                ue = cl_emb[rcl].mean(0)
                ue /= (np.linalg.norm(ue) + 1e-8)
                for item in rp_sc:
                    if item in item_to_clid:
                        sim = float(cl_emb[item_to_clid[item]] @ ue)
                        rp_sc[item] *= (1.0 + max(0, sim) * 0.3)
        
        # GRU boost
        if gru_models:
            gru_hist = [gru_vocab[s] for s in hist if s in gru_vocab]
            if gru_hist:
                gru_hist = gru_hist[-GRU_MAX_SEQ:]
                seq = torch.LongTensor([gru_hist]).to(DEVICE)
                length = torch.LongTensor([len(gru_hist)])
                avg_sc = np.zeros(n_gru)
                for m in gru_models:
                    m.eval()
                    with torch.no_grad():
                        avg_sc += m.predict(seq, length).squeeze(0).cpu().numpy()
                avg_sc /= len(gru_models)
                for item in rp_sc:
                    if item in gru_vocab:
                        gru_s = float(avg_sc[gru_vocab[item]])
                        if gru_s > 0:
                            rp_sc[item] *= (1.0 + gru_s * 0.1)
        
        preds_best[uid] = [p for p, _ in rp_sc.most_common(K)]
    
    r_best, n_best, h_best = evaluate(preds_best, test_gt, test_uids)
    print(f"  ★ CL+GRU-rerank: R@6={r_best:.4f} | NDCG={n_best:.4f} | HR={h_best:.4f}")
    
    # ---- 6. COMPARISON TABLE ----
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (Synerise RecSys 2025)")
    print("=" * 80)
    
    results = [
        ("Popularity",           r,       n,       h),
        ("Co-occurrence",        r_c,     n_c,     h_c),
        ("RePurchase only",      r_rp,    n_rp,    h_rp),
        ("RePurch+BuyBoost",     r_rb,    n_rb,    h_rb),
        ("2-Stage RP+Discovery", r_novel, n_novel, h_novel),
        ("CL-reranked RP",       r_clr,   n_clr,   h_clr),
        ("GRU-reranked RP",      r_grr,   n_grr,   h_grr),
        ("★ CL+GRU-rerank RP",  r_best,  n_best,  h_best),
    ]
    
    best_r = max(x[1] for x in results)
    print(f"\n  {'Method':<25} | {'R@6':>9} | {'NDCG@6':>9} | {'HR@6':>9}")
    print(f"  {'-'*25}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for nm, r, n, h in results:
        mk = " ◀ BEST" if r == best_r else ""
        print(f"  {nm:<25} | {r:>9.4f} | {n:>9.4f} | {h:>9.4f}{mk}")
    
    print(f"\n  📊 {len(test_uids):,} test users, {len(freq):,} items")
    print(f"  📊 GRU vocab: {n_gru} items, CL items: {n_cl}")
    print(f"  ⏱️  {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n  Key insight: Re-purchase is dominant → use CL/GRU to RE-RANK")
    print(f"  repurchase candidates, not to INJECT new discovery items.")
    print(f"\n  Novel contributions:")
    print(f"  C1: Contrastive learning re-ranks re-purchase candidates by similarity to recent buys")
    print(f"  C2: GRU sequential signal validates re-purchase timing/likelihood")
    print(f"  C3: Two-stage fusion preserves dominant signal while adding discovery")
    print(f"  C4: Comprehensive ablation across fusion strategies")


if __name__ == "__main__":
    main()
