"""
CL-GRU4Rec+RP v3 — Unified High-Performance Model
====================================================

Key improvements over v2:
  1. GRU4Rec with BPR loss + in-batch negative sampling (matches cornac's approach)
  2. TOP1 loss option (original GRU4Rec paper)
  3. Same architecture, same code, both datasets
  4. Adaptive Two-Stage Fusion (auto-detects dominant signal)

Components:
  C1: GRU4Rec (BPR) — sequential next-item prediction
  C2: Contrastive Item Similarity — learned item embeddings for discovery
  C3: Re-Purchase Awareness — buy-boosted recency scoring
  C4: Adaptive Fusion — dataset-adaptive signal combination

Usage:
  python cl_gru4rec_rp_v3.py --dataset rental
  python cl_gru4rec_rp_v3.py --dataset synerise
"""
import argparse, ast, os, pickle, time, random, gc
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

K = 6

# GRU config
GRU_EMBED = 128
GRU_HIDDEN = 256
GRU_LAYERS = 1
GRU_DROP = 0.2
GRU_MAX_SEQ = 50
GRU_BATCH = 128
GRU_EPOCHS = 35
GRU_LR = 0.002
GRU_N_NEG = 128       # negatives per positive for BPR
GRU_SEEDS = [42, 123, 456]
GRU_LOSS = "bpr"       # "bpr" or "top1" or "ce"

# CL config
CL_EMBED = 64
CL_EPOCHS = 30
CL_LR = 0.002
CL_TEMP = 0.07
CL_NEG = 256
CL_BATCH = 1024

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
# COMPONENT 1: GRU4Rec with BPR Loss
# ============================================================================
class GRU4Rec(nn.Module):
    """GRU4Rec with BPR/TOP1 loss — matches original paper's approach."""
    def __init__(self, n_items, embed_dim=128, hidden_dim=256, n_layers=1, 
                 dropout=0.2, pad_idx=0):
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
        # Tie output weights with embeddings (if dimensions match)
        # self.output.weight = self.embed.weight  # Only if embed_dim == hidden_dim
    
    def forward(self, seq, lengths=None):
        """Full sequence forward for CE training."""
        x = self.drop(self.embed(seq))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            output, _ = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.gru(x)
        return self.output(self.drop(output))  # (B, L, V)
    
    def forward_hidden(self, seq, lengths=None):
        """Get hidden states for BPR training."""
        x = self.drop(self.embed(seq))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            output, _ = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.gru(x)
        return output  # (B, L, H)
    
    def score_items(self, hidden_states, item_ids):
        """Score specific items given hidden states.
        hidden_states: (B, H) or (B, L, H)
        item_ids: (N,)
        Returns: (B, N) or (B, L, N)
        """
        item_emb = self.output.weight[item_ids]  # (N, H) via output layer
        if hidden_states.dim() == 2:
            return torch.mm(hidden_states, item_emb.t()) + self.output.bias[item_ids]
        else:
            return torch.einsum('blh,nh->bln', hidden_states, item_emb) + self.output.bias[item_ids]
    
    def predict(self, seq, lengths=None):
        """Get scores for last hidden state."""
        self.eval()
        x = self.embed(seq)  # no dropout
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(x)
        # hidden: (n_layers, B, H) → take last layer
        return self.output(hidden[-1])  # (B, V)


class BPRSeqDataset(Dataset):
    """Dataset for BPR training — returns input/target pairs."""
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

def collate_bpr(batch):
    inputs, targets = zip(*batch)
    ml = max(len(s) for s in inputs)
    inp = torch.LongTensor([list(s) + [0]*(ml-len(s)) for s in inputs])
    tgt = torch.LongTensor([list(s) + [-1]*(ml-len(s)) for s in targets])
    lengths = torch.LongTensor([len(s) for s in inputs])
    return inp, lengths, tgt


def bpr_loss(pos_scores, neg_scores):
    """BPR loss: -log sigmoid(pos - neg)"""
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

def top1_loss(pos_scores, neg_scores):
    """TOP1 loss from GRU4Rec paper"""
    diff = neg_scores - pos_scores
    return (torch.sigmoid(diff) + torch.sigmoid(neg_scores**2)).mean()


def train_gru_bpr(sessions, n_items, seed, cache_path):
    """Train GRU4Rec with BPR loss and in-batch negative sampling."""
    if os.path.exists(cache_path):
        print(f"    Loading {cache_path}")
        model = GRU4Rec(n_items, GRU_EMBED, GRU_HIDDEN, GRU_LAYERS, GRU_DROP)
        model.load_state_dict(torch.load(cache_path, map_location=DEVICE, weights_only=True))
        return model.to(DEVICE)
    
    print(f"    Training GRU4Rec-BPR (seed={seed})...")
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    
    ds = BPRSeqDataset(sessions, GRU_MAX_SEQ)
    loader = DataLoader(ds, batch_size=GRU_BATCH, shuffle=True,
                       collate_fn=collate_bpr, num_workers=0, drop_last=True)
    
    model = GRU4Rec(n_items, GRU_EMBED, GRU_HIDDEN, GRU_LAYERS, GRU_DROP).to(DEVICE)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=GRU_LR)  # Adagrad like original GRU4Rec
    
    all_items = torch.arange(1, n_items).to(DEVICE)  # exclude PAD
    
    for epoch in range(GRU_EPOCHS):
        model.train()
        total_loss, nb = 0, 0
        pbar = tqdm(loader, desc=f"    Epoch {epoch+1}/{GRU_EPOCHS}", leave=False)
        for inp, lengths, tgt in pbar:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            
            if GRU_LOSS == "ce":
                # Standard CE (fallback)
                logits = model(inp, lengths)
                B, L, V = logits.shape
                mask = tgt.reshape(-1) != -1
                loss = F.cross_entropy(logits.reshape(-1, V)[mask], tgt.reshape(-1)[mask])
            else:
                # BPR / TOP1 with negative sampling
                hidden = model.forward_hidden(inp, lengths)  # (B, L, H)
                B, L, H = hidden.shape
                
                # Flatten valid positions
                mask = tgt != -1  # (B, L)
                valid_hidden = hidden[mask]  # (N_valid, H)
                valid_targets = tgt[mask]     # (N_valid,)
                
                if valid_hidden.shape[0] == 0: continue
                
                # Positive scores: for each valid position, score its target item
                pos_emb = model.output.weight[valid_targets]  # (N_valid, H)
                pos_scores = (valid_hidden * pos_emb).sum(-1) + model.output.bias[valid_targets]  # (N_valid,)
                
                # Sample negatives
                neg_idx = all_items[torch.randint(0, len(all_items), (GRU_N_NEG,))]
                neg_emb = model.output.weight[neg_idx]  # (N_neg, H)
                neg_bias = model.output.bias[neg_idx]   # (N_neg,)
                neg_scores = torch.mm(valid_hidden, neg_emb.t()) + neg_bias  # (N_valid, N_neg)
                
                if GRU_LOSS == "bpr":
                    # BPR: average over negatives
                    loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8).mean()
                else:  # top1
                    loss = top1_loss(pos_scores.unsqueeze(1), neg_scores)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item(); nb += 1
            pbar.set_postfix(loss=f"{total_loss/nb:.4f}")
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1}/{GRU_EPOCHS}: loss={total_loss/max(nb,1):.4f}")
    
    torch.save(model.state_dict(), cache_path)
    print(f"    Saved: {cache_path}")
    return model


# ============================================================================
# COMPONENT 2: Contrastive Item Similarity 
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
    def get_embeddings(self, items):
        """Raw embeddings without projector (for diversity)."""
        return F.normalize(self.embedding(items), dim=-1)


def train_cl(sessions, n_items, cache_path):
    if os.path.exists(cache_path):
        print(f"    Loading {cache_path}")
        model = ContrastiveItemModel(n_items, CL_EMBED)
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
    
    model = ContrastiveItemModel(n_items, CL_EMBED).to(DEVICE)
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
    print(f"    Saved: {cache_path}")
    return model


# ============================================================================
# COMPONENT 3: Adaptive Two-Stage Fusion
# ============================================================================
def adaptive_fusion(gru_scores, rp_scores, cl_scores, cooc_scores,
                    n_rp_items, k=6):
    """
    Adaptive Two-Stage Fusion:
    - If user has strong RP signal (≥K unique items), RP fills most slots
    - If user has weak RP (< K items), GRU+CL+CoOccur fills discovery slots
    - CL and CoOccur always re-rank within each stage
    """
    final = Counter()
    
    # Normalize each signal to [0, 1] range
    def norm(scores):
        if not scores: return {}
        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        if mx == mn: return {k: 0.5 for k in scores}
        return {k: (v - mn) / (mx - mn) for k, v in scores.items()}
    
    rp_n = norm(rp_scores)
    gru_n = norm(gru_scores)
    cl_n = norm(cl_scores)
    co_n = norm(cooc_scores)
    
    all_items = set(rp_n) | set(gru_n) | set(cl_n) | set(co_n)
    
    # Adaptive weights based on RP coverage
    rp_coverage = min(n_rp_items / k, 1.0)  # 0..1
    
    # High RP coverage → RP dominant; Low → GRU dominant
    w_rp  = 0.8 * rp_coverage
    w_gru = 0.6 * (1 - rp_coverage * 0.5)
    w_cl  = 0.15
    w_co  = 0.1
    
    for item in all_items:
        final[item] = (w_rp * rp_n.get(item, 0) +
                       w_gru * gru_n.get(item, 0) +
                       w_cl * cl_n.get(item, 0) +
                       w_co * co_n.get(item, 0))
    
    return [p for p, _ in final.most_common(k)]


# ============================================================================
# DATA LOADING (same as v2)
# ============================================================================
def load_rental_data():
    ROOT = "data"
    def get_old_to_new():
        m = pd.read_csv(f"{ROOT}/old_site_new_site_products.csv", dtype=str)
        return m.set_index("old_site_id")["new_site_id"].to_dict()
    def get_slug_to_ids():
        o2n = get_old_to_new()
        df_old = pd.read_csv(f"{ROOT}/old_site_products.csv", usecols=["id","slug"], dtype=str)
        df_new = pd.read_csv(f"{ROOT}/new_site_products.csv", usecols=["id","slug"], dtype=str)
        df_old["id"] = df_old["id"].map(o2n); df_old = df_old.dropna(subset=["id"])
        return pd.concat([df_new, df_old]).drop_duplicates(["id","slug"]).set_index("slug")["id"].to_dict()
    
    stoi = get_slug_to_ids()
    df = pd.concat([
        pd.read_csv(f"{ROOT}/metrika_hits.csv", usecols=['date_time','slug','page_type','project_id','is_page_view','watch_id'], dtype=str),
        pd.read_csv(f"{ROOT}/metrika_hits_test.csv", usecols=['date_time','slug','page_type','project_id','is_page_view','watch_id'], dtype=str),
    ], ignore_index=True)
    df["date_time"] = pd.to_datetime(df["date_time"], format="ISO8601")
    df = df[df["is_page_view"].eq("1")]
    for pt, val in [("SEARCH","search"),("CART","cart"),("CHECKOUT","checkout"),("ORDER","order"),("UNAVAILABLE_PRODUCT","unavailable")]:
        df.loc[df["page_type"].eq(pt), "slug"] = val
    df = df.dropna(subset=["slug"])
    df["product_id"] = df["slug"].map(stoi)
    missing = df["product_id"].isnull()
    nm = {s: str(500000000+i) for i,s in enumerate(df.loc[missing,"slug"].unique())}
    stoi.update(nm)
    df["product_id"] = df["slug"].map(stoi)
    
    visits = pd.concat([
        pd.read_csv(f"{ROOT}/metrika_visits.csv", usecols=['client_id','visit_id','watch_ids'], dtype=str),
        pd.read_csv(f"{ROOT}/metrika_visits_test.csv", usecols=['client_id','visit_id','watch_ids'], dtype=str),
    ], ignore_index=True)
    visits["watch_ids"] = visits["watch_ids"].apply(ast.literal_eval)
    visits = visits.explode("watch_ids").rename(columns={"watch_ids": "watch_id"})
    
    merged = pd.merge(df, visits, on="watch_id", how="left")
    merged = pd.concat([
        merged[merged["page_type"].ne("PRODUCT")],
        merged[merged["page_type"].eq("PRODUCT")].drop_duplicates(["visit_id","product_id"], keep="first")
    ])
    merged = merged[["client_id","visit_id","product_id","is_page_view","page_type","date_time","slug","project_id"]].dropna()
    merged['date_time'] = merged['date_time'].astype('int64') // 10**9
    merged = merged.sort_values(['visit_id','date_time'])
    
    s = (merged.groupby("visit_id").head(1)
         .assign(product_id=lambda d: np.where(d["project_id"]=="1","000000000","000000001"),
                 page_type=lambda d: np.where(d["project_id"]=="1","START_OLD","START_NEW"),
                 is_page_view="1"))
    s["date_time"] = s["date_time"] - 1
    merged = pd.concat([merged, s], ignore_index=True).sort_values(['visit_id','date_time'])
    
    test_vids = pd.read_csv(f"{ROOT}/metrika_visits_test.csv", usecols=['visit_id'], dtype=str)["visit_id"].unique()
    allowed = set(pd.read_csv(f"{ROOT}/new_site_products.csv", usecols=["id"], dtype=str)["id"].unique())
    return merged, test_vids, allowed


def load_synerise_data():
    CACHE = "synerise_final.pkl"
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f: return pickle.load(f)
    
    BASE = "synerise_dataset"
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
    freq = set(sku_counts[sku_counts >= 100].index)
    dff = df[df["sku"].isin(freq)]
    user_data = {}
    for uid, grp in tqdm(dff.groupby("client_id"), desc="Users"):
        items, events = grp["sku"].tolist(), grp["event"].tolist()
        if len(items) >= 5: user_data[uid] = (items, events)
    cooccur = defaultdict(Counter)
    test_gt = {}; train_items = {}; train_events = {}
    for uid, (items, events) in user_data.items():
        sp = max(2, int(len(items) * 0.8))
        train_items[uid] = items[:sp]; train_events[uid] = events[:sp]
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
    d = {"train_items": train_items, "train_events": train_events,
         "test_gt": test_gt, "cooccur": dict(cooccur),
         "cat_pop": dict(cat_pop), "s2c": s2c, "freq": freq}
    with open(CACHE, "wb") as f: pickle.dump(d, f)
    return d


# ============================================================================
# RENTAL PIPELINE
# ============================================================================
def run_rental():
    t0 = time.time()
    print("=" * 80)
    print("CL-GRU4Rec+RP v3 — RENTAL PRODUCT")
    print("=" * 80)
    
    print("\n[1] LOADING DATA")
    df_data, test_vids, allowed_set = load_rental_data()
    print(f"  {len(df_data):,} rows | {df_data['visit_id'].nunique():,} sessions")
    
    # Vocabulary
    all_prods = sorted(df_data["product_id"].unique())
    i2idx = {"<PAD>": 0}
    for i, p in enumerate(all_prods): i2idx[p] = i + 1
    idx2i = {v: k for k, v in i2idx.items()}
    n_items = len(i2idx)
    print(f"  Vocab: {n_items} items")
    
    # Sessions
    visit_groups = df_data.groupby("visit_id")
    gru_sessions = []
    for vid, grp in visit_groups:
        grp = grp.head(20).sort_values("date_time")
        indices = [i2idx[pid] for pid in grp["product_id"] if pid in i2idx]
        if len(indices) >= 3: gru_sessions.append(indices)
    print(f"  GRU sessions: {len(gru_sessions):,}")
    
    # CL sessions (allowed products only)
    cl_items = sorted(allowed_set)
    cl2idx = {p: i for i, p in enumerate(cl_items)}
    n_cl = len(cl_items)
    cl_sessions = []
    for vid, grp in visit_groups:
        prods = [cl2idx[pid] for pid in grp["product_id"] if pid in cl2idx]
        if len(set(prods)) >= 2: cl_sessions.append(prods)
    print(f"  CL sessions: {len(cl_sessions):,}, {n_cl} items")
    
    # Co-occurrence (from lru.py proven logic)
    print("\n[2] CO-OCCURRENCE")
    fwd = defaultdict(Counter); bwd = defaultdict(Counter)
    cat2p = defaultdict(Counter)
    test_set = set(test_vids)
    for cid, grp in tqdm(df_data.groupby("client_id"), desc="  Building"):
        grp = grp.sort_values("date_time")
        prods, cats = [], []
        for _, r in grp.iterrows():
            pt = r.get("page_type",""); pid = str(r.get("product_id","")); slug = str(r.get("slug",""))
            if pt == "CATEGORY" and slug != "nan":
                cats.append(slug)
            elif pt == "PRODUCT" and pid in allowed_set:
                for c in cats[-3:]: cat2p[c][pid] += 1
                for prev in prods[-5:]:
                    fwd[prev][pid] += 1; bwd[pid][prev] += 1
                prods.append(pid)
    print(f"  Fwd: {len(fwd)} | Bwd: {len(bwd)} | Cat: {len(cat2p)}")
    
    # Visit context
    vctx = {}
    for vid, grp in df_data[df_data["visit_id"].isin(test_set)].groupby("visit_id"):
        grp = grp.sort_values("date_time")
        prods, cats = [], []
        for _, r in grp.iterrows():
            if r["page_type"] == "PRODUCT" and str(r["product_id"]) in allowed_set:
                prods.append(str(r["product_id"]))
            elif r["page_type"] == "CATEGORY" and pd.notna(r["slug"]):
                cats.append(str(r["slug"]))
        vctx[vid] = {"products": prods[-5:], "categories": cats[-3:]}
    
    # ---- Train GRU ----
    print("\n[3] GRU4Rec-BPR TRAINING")
    gru_models = []
    for i, seed in enumerate(GRU_SEEDS):
        cache = f"v3_rental_gru_{GRU_LOSS}_seed{seed}.pkl"
        print(f"  Model {i+1}/{len(GRU_SEEDS)}:")
        m = train_gru_bpr(gru_sessions, n_items, seed, cache)
        gru_models.append(m)
    
    # ---- Train CL ----
    print("\n[4] CONTRASTIVE LEARNING")
    cl_model = train_cl(cl_sessions, n_cl, f"v3_rental_cl.pkl")
    cl_model.eval()
    with torch.no_grad():
        cl_emb = cl_model(torch.arange(n_cl).to(DEVICE)).cpu().numpy()
    print(f"  CL embeddings: {cl_emb.shape}")
    
    # ---- Predict ----
    print("\n[5] PREDICTIONS + FUSION")
    test_data = (df_data[df_data["visit_id"].isin(test_set)]
                 .groupby("visit_id", sort=False)
                 .agg(historical_items=("product_id", lambda x: x.tolist()))
                 .reset_index())
    
    # Pre-compute allowed mask
    allowed_mask = np.zeros(n_items, dtype=bool)
    for item in allowed_set:
        if item in i2idx: allowed_mask[i2idx[item]] = True
    
    predictions = []
    for vid_row in tqdm(test_data.itertuples(), total=len(test_data), desc="  Predicting"):
        vid = vid_row.visit_id
        hist = vid_row.historical_items[-GRU_MAX_SEQ:]
        ctx = vctx.get(vid, {"products": [], "categories": []})
        last_prods = ctx["products"]
        hist_idx = [i2idx[p] for p in hist if p in i2idx]
        
        # GRU scores
        gru_scores = {}
        if hist_idx:
            seq = torch.LongTensor([hist_idx]).to(DEVICE)
            length = torch.LongTensor([len(hist_idx)])
            avg = np.zeros(n_items)
            for m in gru_models:
                m.eval()
                with torch.no_grad(): avg += m.predict(seq, length).squeeze(0).cpu().numpy()
            avg /= len(gru_models)
            avg[0] = -np.inf  # mask PAD
            for idx in hist_idx: avg[idx] = -np.inf  # mask history
            avg[~allowed_mask] = -np.inf  # mask non-allowed
            
            top_k = np.argsort(avg)[-50:][::-1]
            for idx in top_k:
                item = idx2i.get(idx, "")
                if item in allowed_set:
                    gru_scores[item] = float(avg[idx])
        
        # CoOccurrence scores
        cooc_scores = {}
        hist_set = set(last_prods)
        for prev in last_prods:
            if prev in fwd:
                for pid, cnt in fwd[prev].most_common(20):
                    if pid not in hist_set:
                        cooc_scores[pid] = cooc_scores.get(pid, 0) + cnt
            if prev in bwd:
                for pid, cnt in bwd[prev].most_common(20):
                    if pid not in hist_set:
                        cooc_scores[pid] = cooc_scores.get(pid, 0) + cnt * 0.7
        
        # CL similarity scores
        cl_scores = {}
        if last_prods:
            recent_cl = [cl2idx[p] for p in last_prods if p in cl2idx]
            if recent_cl:
                user_emb = cl_emb[recent_cl].mean(0)
                user_emb /= (np.linalg.norm(user_emb) + 1e-8)
                sims = cl_emb @ user_emb
                for ci in np.argsort(sims)[-30:][::-1]:
                    item = cl_items[ci]
                    if item not in hist_set and sims[ci] > 0.2:
                        cl_scores[item] = float(sims[ci])
        
        # RP scores (session revisit)
        rp_scores = {}
        for p in last_prods:
            rp_scores[p] = rp_scores.get(p, 0) + 1
        
        # Fusion (GRU-dominant for rental)
        all_candidates = set(gru_scores) | set(cooc_scores) | set(cl_scores)
        final = Counter()
        for item in all_candidates:
            gs = gru_scores.get(item, 0)
            cs = cooc_scores.get(item, 0) 
            cls = cl_scores.get(item, 0)
            rps = rp_scores.get(item, 0)
            # Normalize co-occurrence
            cn = min(cs / 10.0, 1.0) if cs > 0 else 0
            # GRU is primary, others boost
            final[item] = gs + cn * 0.3 * max(abs(gs), 1.0) + cls * 0.2 * max(abs(gs), 1.0) + rps * 0.1 * max(abs(gs), 1.0)
        
        top = [p for p, _ in final.most_common(K)]
        
        # Category fallback
        if len(top) < K:
            seen = set(top)
            for cat in ctx["categories"]:
                if cat in cat2p:
                    for pid, _ in cat2p[cat].most_common(K):
                        if pid not in seen: top.append(pid); seen.add(pid)
                        if len(top) >= K: break
                if len(top) >= K: break
        
        predictions.append({"visit_id": vid, "product_ids": " ".join(top[:K])})
    
    # Build submission
    df_sub = pd.DataFrame(predictions)
    all_recs = []
    for _, r in df_sub.iterrows():
        if r["product_ids"]: all_recs.extend(r["product_ids"].split())
    popular = [p for p, _ in Counter(all_recs).most_common(K)]
    
    def pad(s, n, fb):
        items = str(s).split() if pd.notna(s) and s else []
        seen = set(); res = []
        for x in items:
            if x not in seen: res.append(x); seen.add(x)
        for x in fb:
            if len(res) >= n: break
            if x not in seen: res.append(x); seen.add(x)
        return " ".join(res[:n])
    
    df_sub["product_ids"] = df_sub["product_ids"].apply(lambda x: pad(x, K, popular))
    test_df = pd.read_csv("data/metrika_visits_test.csv", usecols=["visit_id"], dtype=str)
    df_sub["visit_id"] = df_sub["visit_id"].astype(str)
    df_sub = df_sub.set_index("visit_id").reindex(test_df["visit_id"]).reset_index()
    df_sub["product_ids"] = df_sub["product_ids"].fillna(" ".join(popular))
    df_sub.to_csv("submission.csv", index=False)
    
    elapsed = time.time() - t0
    n_uniq = len(set(" ".join(df_sub["product_ids"]).split()))
    print(f"\n  ✅ submission.csv ({len(df_sub)} rows, {n_uniq} unique products)")
    print(f"  ⏱️  {elapsed:.0f}s ({elapsed/60:.1f} min)")


# ============================================================================
# SYNERISE PIPELINE  
# ============================================================================
def run_synerise():
    t0 = time.time()
    print("=" * 80)
    print("CL-GRU4Rec+RP v3 — SYNERISE RecSys 2025")
    print("=" * 80)
    
    print("\n[1] LOADING DATA")
    d = load_synerise_data()
    train_items = d["train_items"]; train_events = d["train_events"]
    test_gt = d["test_gt"]; cooccur = d["cooccur"]
    cat_pop = d["cat_pop"]; s2c = d["s2c"]; freq = d["freq"]
    test_uids = sorted(test_gt.keys())
    print(f"  {len(test_uids):,} test users, {len(freq):,} items")
    
    # Vocabulary (same for GRU and CL)
    all_items = sorted(freq)
    i2idx = {"<PAD>": 0}
    for i, item in enumerate(all_items): i2idx[item] = i + 1
    idx2i = {v: k for k, v in i2idx.items()}
    n_items = len(i2idx)
    print(f"  Vocab: {n_items} items")
    
    # Sessions
    gru_sessions, cl_sessions = [], []
    for uid, items in train_items.items():
        indices = [i2idx[s] for s in items if s in i2idx]
        if len(indices) >= 3: gru_sessions.append(indices)
        if len(set(indices)) >= 2: cl_sessions.append(indices)
    print(f"  GRU sessions: {len(gru_sessions):,} | CL sessions: {len(cl_sessions):,}")
    
    # Baselines
    print("\n[2] BASELINES")
    pop = Counter()
    for v in train_items.values(): pop.update(v)
    pop_top = [p for p, _ in pop.most_common(K)]
    r_pop, n_pop, h_pop = evaluate({u: pop_top for u in test_uids}, test_gt, test_uids)
    print(f"  Popularity:    R@6={r_pop:.4f} | NDCG={n_pop:.4f} | HR={h_pop:.4f}")
    preds_rp = {}
    for uid in test_uids:
        preds_rp[uid] = [p for p, _ in Counter(train_items[uid]).most_common(K)]
    r_rp, n_rp, h_rp = evaluate(preds_rp, test_gt, test_uids)
    print(f"  RePurchase:    R@6={r_rp:.4f} | NDCG={n_rp:.4f} | HR={h_rp:.4f}")
    
    # ---- Train GRU ----
    print("\n[3] GRU4Rec-BPR TRAINING")
    gru_models = []
    for i, seed in enumerate(GRU_SEEDS):
        cache = f"v3_synerise_gru_{GRU_LOSS}_seed{seed}.pkl"
        print(f"  Model {i+1}/{len(GRU_SEEDS)}:")
        m = train_gru_bpr(gru_sessions, n_items, seed, cache)
        gru_models.append(m)
    
    # ---- Train CL ----
    print("\n[4] CONTRASTIVE LEARNING")
    cl_model = train_cl(cl_sessions, n_items, f"v3_synerise_cl.pkl")
    cl_model.eval()
    with torch.no_grad():
        chunk = 2000; embs = []
        for i in range(0, n_items, chunk):
            idx = torch.arange(i, min(i+chunk, n_items)).to(DEVICE)
            embs.append(cl_model(idx).cpu().numpy())
        cl_emb = np.vstack(embs)
    print(f"  CL embeddings: {cl_emb.shape}")
    
    # ---- GRU-only baseline ----
    print("\n[5] GRU-ONLY BASELINE")
    preds_gru = {}
    for uid in tqdm(test_uids, desc="  GRU predict"):
        hist = train_items[uid]
        indices = [i2idx[s] for s in hist if s in i2idx]
        if not indices: preds_gru[uid] = pop_top[:K]; continue
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
        preds_gru[uid] = [idx2i[i] for i in top if i in idx2i and idx2i[i] != "<PAD>"]
    r_gru, n_gru, h_gru = evaluate(preds_gru, test_gt, test_uids)
    print(f"  GRU-only:      R@6={r_gru:.4f} | NDCG={n_gru:.4f} | HR={h_gru:.4f}")
    
    # ---- CL-GRU4Rec+RP: Two-Stage Fusion ----
    print("\n[6] ★ CL-GRU4Rec+RP (Adaptive Two-Stage Fusion)")
    preds_best = {}
    for uid in tqdm(test_uids, desc="  CL-GRU+RP"):
        hist = train_items[uid]; evts = train_events[uid]; hist_set = set(hist)
        
        # Stage 1: RP scoring (dominant for repeat-purchase data)
        rp_sc = Counter()
        for i, (item, evt) in enumerate(zip(hist, evts)):
            recency = 1.0 + (i / len(hist))
            w = 5.0 if evt == "buy" else 2.0
            rp_sc[item] += w * recency
        rp_top = [p for p, _ in rp_sc.most_common(K)]
        
        # Stage 2: Discovery for remaining slots
        if len(rp_top) < K:
            disc = Counter()
            # CoOccurrence
            for item in hist:
                if item in cooccur:
                    for pid, cnt in cooccur[item].most_common(30):
                        if pid not in hist_set: disc[pid] += cnt
            # CL similarity
            user_cl = [i2idx[s] for s in hist if s in i2idx]
            if user_cl:
                ue = cl_emb[user_cl[-10:]].mean(0)
                ue /= (np.linalg.norm(ue) + 1e-8)
                sims = cl_emb @ ue
                for ci in np.argsort(sims)[-30:][::-1]:
                    item = idx2i.get(ci, "")
                    if item and item != "<PAD>" and item not in hist_set and sims[ci] > 0.2:
                        disc[item] += (sims[ci] - 0.2) * 5.0
            # GRU
            if user_cl:
                seq = torch.LongTensor([user_cl[-GRU_MAX_SEQ:]]).to(DEVICE)
                length = torch.LongTensor([min(len(user_cl), GRU_MAX_SEQ)])
                avg = np.zeros(n_items)
                for m in gru_models:
                    m.eval()
                    with torch.no_grad(): avg += m.predict(seq, length).squeeze(0).cpu().numpy()
                avg /= len(gru_models)
                for gi in np.argsort(avg)[-20:][::-1]:
                    item = idx2i.get(gi, "")
                    if item and item != "<PAD>" and item not in hist_set:
                        disc[item] += max(0, float(avg[gi])) * 0.5
            
            rp_set = set(rp_top)
            for p, _ in disc.most_common(K):
                if len(rp_top) >= K: break
                if p not in rp_set: rp_top.append(p); rp_set.add(p)
        
        preds_best[uid] = rp_top[:K]
    
    r_best, n_best, h_best = evaluate(preds_best, test_gt, test_uids)
    print(f"\n  ★ CL-GRU+RP:   R@6={r_best:.4f} | NDCG={n_best:.4f} | HR={h_best:.4f}")
    
    # Comparison table
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (Synerise)")
    print("=" * 80)
    results = [
        ("Popularity",      r_pop,  n_pop,  h_pop),
        ("RePurchase",       r_rp,   n_rp,   h_rp),
        ("GRU4Rec-BPR",     r_gru,  n_gru,  h_gru),
        ("★ CL-GRU4Rec+RP", r_best, n_best, h_best),
    ]
    best_r = max(x[1] for x in results)
    print(f"\n  {'Method':<20} | {'R@6':>9} | {'NDCG@6':>9} | {'HR@6':>9}")
    print(f"  {'-'*20}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for nm, r, n, h in results:
        mk = " ◀ BEST" if r == best_r else ""
        print(f"  {nm:<20} | {r:>9.4f} | {n:>9.4f} | {h:>9.4f}{mk}")
    print(f"\n  ⏱️  {elapsed:.0f}s ({elapsed/60:.1f} min)")


# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rental", "synerise"], required=True)
    parser.add_argument("--loss", choices=["bpr", "top1", "ce"], default="bpr",
                       help="GRU loss function (default: bpr)")
    args = parser.parse_args()
    GRU_LOSS = args.loss
    
    print(f"Device: {DEVICE} | GRU Loss: {GRU_LOSS}")
    if args.dataset == "rental":
        run_rental()
    else:
        run_synerise()
