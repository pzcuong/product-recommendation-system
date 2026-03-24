"""
CL-GRU4Rec+RP (Unified) — One model, two datasets
====================================================

Proposed method for paper:
  1. GRU4Rec: Clean PyTorch GRU for sequential patterns (no multi-task!)
  2. CL: Contrastive Learning for item similarity (separate training)
  3. RP: Re-Purchase signal at inference
  4. Two-Stage Fusion: RP fills first → CL+GRU+CoOccur fills discovery slots

Key fix over v1: SEPARATE training (GRU alone, CL alone), combine at INFERENCE.
v1 failed because multi-task training confused the GRU encoder.

Usage:
  python cl_gru4rec_rp_unified.py --dataset rental
  python cl_gru4rec_rp_unified.py --dataset synerise
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

# ============================================================================
# CONFIG (shared across datasets)
# ============================================================================
K = 6

# GRU config
GRU_EMBED_DIM = 128
GRU_HIDDEN_DIM = 192
GRU_DROPOUT = 0.15
GRU_MAX_SEQ = 50
GRU_BATCH = 256
GRU_EPOCHS = 30
GRU_LR = 0.001
GRU_SEEDS = [42, 123, 456]
GRU_TOP_K = 100  # top-K from GRU

# CL config
CL_EMBED_DIM = 64
CL_EPOCHS = 25
CL_LR = 0.003
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
# COMPONENT 1: GRU4Rec (Clean PyTorch)
# ============================================================================
class GRU4RecModel(nn.Module):
    """Clean GRU4Rec: next-item prediction only, no multi-task."""
    def __init__(self, n_items, embed_dim=128, hidden_dim=192, dropout=0.15, pad_idx=0):
        super().__init__()
        self.n_items = n_items
        self.embed = nn.Embedding(n_items, embed_dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.embed.weight)
        self.embed.weight.data[pad_idx].zero_()
        self.drop_emb = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.drop_out = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_items)
    
    def forward(self, seq, lengths=None):
        x = self.drop_emb(self.embed(seq))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            output, _ = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.gru(x)
        return self.head(self.drop_out(output))  # (B, L, V)
    
    def predict(self, seq, lengths=None):
        """Get scores for last position in sequence."""
        self.eval()
        x = self.embed(seq)  # no dropout at inference
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(x)
        return self.head(hidden.squeeze(0))  # (B, V)


class SeqDataset(Dataset):
    def __init__(self, sessions, max_len=50):
        self.sessions = [s for s in sessions if len(s) >= 3]
        self.max_len = max_len
    def __len__(self): return len(self.sessions)
    def __getitem__(self, idx):
        seq = self.sessions[idx]
        if len(seq) > self.max_len + 1:
            # Random crop for data augmentation
            start = random.randint(0, len(seq) - self.max_len - 1)
            seq = seq[start:start + self.max_len + 1]
        return seq[:-1], seq[1:]

def collate_seq(batch):
    inputs, targets = zip(*batch)
    ml = max(len(s) for s in inputs)
    inp = torch.LongTensor([list(s) + [0]*(ml-len(s)) for s in inputs])
    tgt = torch.LongTensor([list(s) + [-1]*(ml-len(s)) for s in targets])
    lengths = torch.LongTensor([len(s) for s in inputs])
    return inp, lengths, tgt

def train_gru(sessions, n_items, seed, cache_path, config):
    """Train a single GRU4Rec model."""
    if os.path.exists(cache_path):
        print(f"    Loading {cache_path}")
        model = GRU4RecModel(n_items, config['embed'], config['hidden'], config['dropout'])
        model.load_state_dict(torch.load(cache_path, map_location=DEVICE, weights_only=True))
        return model.to(DEVICE)
    
    print(f"    Training (seed={seed})...")
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    
    ds = SeqDataset(sessions, config['max_seq'])
    loader = DataLoader(ds, batch_size=config['batch'], shuffle=True,
                       collate_fn=collate_seq, num_workers=0, drop_last=True)
    
    model = GRU4RecModel(n_items, config['embed'], config['hidden'], config['dropout']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss, nb = 0, 0
        pbar = tqdm(loader, desc=f"    Epoch {epoch+1}/{config['epochs']}", leave=False)
        for inp, lengths, tgt in pbar:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            logits = model(inp, lengths)  # (B, L, V)
            B, L, V = logits.shape
            mask = tgt.reshape(-1) != -1
            loss = F.cross_entropy(logits.reshape(-1, V)[mask], tgt.reshape(-1)[mask],
                                  label_smoothing=0.1)  # Label smoothing helps
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); nb += 1
            pbar.set_postfix(loss=f"{total_loss/nb:.4f}")
        scheduler.step()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1}/{config['epochs']}: loss={total_loss/max(nb,1):.4f}")
    
    torch.save(model.state_dict(), cache_path)
    print(f"    Cached: {cache_path}")
    return model


# ============================================================================
# COMPONENT 2: Contrastive Learning (Item Similarity)
# ============================================================================
class ContrastiveItemModel(nn.Module):
    def __init__(self, n_items, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    def forward(self, items):
        return F.normalize(self.projector(self.embedding(items)), dim=-1)

def train_cl(sessions, n_items, cache_path):
    """Train contrastive item embeddings."""
    if os.path.exists(cache_path):
        print(f"    Loading {cache_path}")
        model = ContrastiveItemModel(n_items, CL_EMBED_DIM)
        model.load_state_dict(torch.load(cache_path, map_location=DEVICE, weights_only=True))
        return model.to(DEVICE)
    
    print("    Building positive pairs...")
    pairs = []
    for sess in sessions:
        unique = list(set(sess))
        if len(unique) < 2: continue
        if len(unique) > 20:
            for _ in range(30):
                i, j = random.sample(range(len(unique)), 2)
                pairs.append((unique[i], unique[j]))
        else:
            for i in range(len(unique)):
                for j in range(i+1, len(unique)):
                    pairs.append((unique[i], unique[j]))
    random.shuffle(pairs)
    print(f"    {len(pairs):,} pairs from {len(sessions):,} sessions")
    
    model = ContrastiveItemModel(n_items, CL_EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CL_LR)
    all_items = list(range(n_items))
    
    for epoch in range(CL_EPOCHS):
        random.shuffle(pairs)
        total_loss, nb = 0, 0
        for i in range(0, len(pairs), CL_BATCH):
            batch = pairs[i:i+CL_BATCH]
            if len(batch) < 2: continue
            anc = torch.LongTensor([p[0] for p in batch]).to(DEVICE)
            pos = torch.LongTensor([p[1] for p in batch]).to(DEVICE)
            neg = torch.LongTensor(random.choices(all_items, k=min(CL_NEG, n_items))).to(DEVICE)
            za, zp, zn = model(anc), model(pos), model(neg)
            logits = torch.cat([(za*zp).sum(-1,keepdim=True)/CL_TEMP,
                               torch.mm(za, zn.t())/CL_TEMP], dim=1)
            loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=DEVICE))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); nb += 1
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1}/{CL_EPOCHS}: loss={total_loss/max(nb,1):.4f}")
    
    torch.save(model.state_dict(), cache_path)
    print(f"    Cached: {cache_path}")
    return model


# ============================================================================
# DATA LOADING: RENTAL PRODUCT
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
    
    # Add start token
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
    BASE = "synerise_dataset"
    CACHE = "synerise_final.pkl"
    
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        return d
    
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
# MAIN: RENTAL PRODUCT
# ============================================================================
def run_rental():
    t0 = time.time()
    print("=" * 80)
    print("CL-GRU4Rec+RP (Unified) — RENTAL PRODUCT")
    print("=" * 80)
    
    print("\n1. LOADING DATA")
    df_data, test_vids, allowed_set = load_rental_data()
    print(f"   {len(df_data):,} rows | {df_data['visit_id'].nunique():,} sessions")
    
    # Build vocabulary
    all_items = sorted(df_data["product_id"].unique())
    item_to_idx = {"<PAD>": 0}
    for i, item in enumerate(all_items):
        item_to_idx[item] = i + 1
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    n_items = len(item_to_idx)
    print(f"   Vocabulary: {n_items} items")
    
    # Build sessions for GRU training
    gru_sessions = []
    visit_groups = df_data.groupby("visit_id")
    for vid, grp in visit_groups:
        grp = grp.head(20).sort_values("date_time")
        indices = [item_to_idx[pid] for pid in grp["product_id"] if pid in item_to_idx]
        if len(indices) >= 3:
            gru_sessions.append(indices)
    print(f"   GRU sessions: {len(gru_sessions):,}")
    
    # Build CL sessions (only allowed products)
    cl_item_list = sorted(allowed_set)
    cl_item_to_idx = {p: i for i, p in enumerate(cl_item_list)}
    n_cl = len(cl_item_list)
    cl_sessions = []
    for vid, grp in visit_groups:
        prods = [cl_item_to_idx[pid] for pid in grp["product_id"]
                 if pid in cl_item_to_idx]
        if len(set(prods)) >= 2:
            cl_sessions.append(prods)
    print(f"   CL sessions: {len(cl_sessions):,}, {n_cl} items")
    
    # Build co-occurrence
    print("\n2. CO-OCCURRENCE")
    fwd_cooc = defaultdict(Counter)
    bwd_cooc = defaultdict(Counter)
    cat_to_prods = defaultdict(Counter)
    test_set = set(test_vids)
    
    for cid, grp in tqdm(df_data.groupby("client_id"), desc="   CoOccur"):
        grp = grp.sort_values("date_time")
        prods, cats = [], []
        for _, r in grp.iterrows():
            pt, pid, slug = r.get("page_type",""), str(r.get("product_id","")), str(r.get("slug",""))
            if pt == "CATEGORY" and slug != "nan":
                cats.append(slug)
            elif pt == "PRODUCT" and pid in allowed_set:
                for c in cats[-3:]: cat_to_prods[c][pid] += 1
                for prev in prods[-5:]:
                    fwd_cooc[prev][pid] += 1; bwd_cooc[pid][prev] += 1
                prods.append(pid)
    print(f"   Fwd: {len(fwd_cooc)} | Bwd: {len(bwd_cooc)} | Cat: {len(cat_to_prods)}")
    
    # Visit context for test set
    visit_ctx = {}
    for vid, grp in df_data[df_data["visit_id"].isin(test_set)].groupby("visit_id"):
        grp = grp.sort_values("date_time")
        prods, cats = [], []
        for _, r in grp.iterrows():
            if r["page_type"] == "PRODUCT" and str(r["product_id"]) in allowed_set:
                prods.append(str(r["product_id"]))
            elif r["page_type"] == "CATEGORY" and pd.notna(r["slug"]):
                cats.append(str(r["slug"]))
        visit_ctx[vid] = {"products": prods[-5:], "categories": cats[-3:]}
    
    # ---- Train GRU4Rec ensemble ----
    print("\n3. GRU4Rec TRAINING")
    gru_config = {'embed': GRU_EMBED_DIM, 'hidden': GRU_HIDDEN_DIM, 'dropout': GRU_DROPOUT,
                  'max_seq': GRU_MAX_SEQ, 'batch': GRU_BATCH, 'lr': GRU_LR, 'epochs': GRU_EPOCHS}
    gru_models = []
    for i, seed in enumerate(GRU_SEEDS):
        cache = f"unified_rental_gru_seed{seed}.pkl"
        print(f"  Model {i+1}/{len(GRU_SEEDS)}:")
        m = train_gru(gru_sessions, n_items, seed, cache, gru_config)
        gru_models.append(m)
    print(f"  Ensemble: {len(gru_models)} models")
    
    # ---- Train CL ----
    print("\n4. CONTRASTIVE LEARNING")
    cl_model = train_cl(cl_sessions, n_cl, "unified_rental_cl.pkl")
    cl_model.eval()
    with torch.no_grad():
        cl_emb = cl_model(torch.arange(n_cl).to(DEVICE)).cpu().numpy()
    print(f"  CL embeddings: {cl_emb.shape}")
    
    # ---- Generate GRU predictions ----
    print("\n5. GRU PREDICTIONS")
    test_data = (df_data[df_data["visit_id"].isin(test_set)]
                 .groupby("visit_id", sort=False)
                 .agg(historical_items=("product_id", lambda x: x.tolist()))
                 .reset_index())
    
    gru_preds = {}  # vid -> [(item, score), ...]
    for vid_row in tqdm(test_data.itertuples(), total=len(test_data), desc="   GRU predict"):
        vid = vid_row.visit_id
        hist = vid_row.historical_items[-GRU_MAX_SEQ:]
        indices = [item_to_idx[p] for p in hist if p in item_to_idx]
        if not indices:
            gru_preds[vid] = []
            continue
        
        seq = torch.LongTensor([indices]).to(DEVICE)
        length = torch.LongTensor([len(indices)])
        
        avg_scores = np.zeros(n_items)
        for m in gru_models:
            m.eval()
            with torch.no_grad():
                scores = m.predict(seq, length).squeeze(0).cpu().numpy()
            avg_scores += scores
        avg_scores /= len(gru_models)
        
        # Mask pad and history
        avg_scores[0] = -np.inf
        for idx in indices: avg_scores[idx] = -np.inf
        # Mask non-allowed items
        for idx in range(n_items):
            item = idx_to_item.get(idx, "")
            if item not in allowed_set:
                avg_scores[idx] = -np.inf
        
        top_k = np.argsort(avg_scores)[-GRU_TOP_K:][::-1]
        gru_preds[vid] = [(idx_to_item[idx], float(avg_scores[idx])) for idx in top_k
                         if idx in idx_to_item and idx_to_item[idx] in allowed_set]
    
    # ---- Two-Stage Fusion ----
    print("\n6. TWO-STAGE FUSION")
    predictions = []
    for vid in tqdm(test_vids, desc="   Fusion"):
        ctx = visit_ctx.get(vid, {"products": [], "categories": []})
        last_prods = ctx["products"]
        hist_set = set(last_prods)
        
        # Stage 1: GRU top predictions (primary for rental)
        gru_items = gru_preds.get(vid, [])
        recs = []
        seen = set()
        
        # GRU candidates with boost signals
        for item, gru_score in gru_items[:30]:
            boost = 0.0
            # Co-occurrence boost
            for prev in last_prods:
                if prev in fwd_cooc and item in fwd_cooc[prev]:
                    boost += min(0.15, 0.012 * fwd_cooc[prev][item])
                if prev in bwd_cooc and item in bwd_cooc[prev]:
                    boost += min(0.08, 0.008 * bwd_cooc[prev][item])
            # CL boost
            if item in cl_item_to_idx and last_prods:
                item_emb = cl_emb[cl_item_to_idx[item]]
                for prev in last_prods[-3:]:
                    if prev in cl_item_to_idx:
                        sim = float(np.dot(item_emb, cl_emb[cl_item_to_idx[prev]]))
                        if sim > 0.3:
                            boost += min(0.12, (sim - 0.3) * 0.2)
            # RP boost (if item was visited before in session)
            if item in hist_set:
                boost += 0.05
            
            max_boost = max(abs(gru_score) * 0.25, 0.05)
            final_score = gru_score + min(boost, max_boost)
            recs.append((item, final_score))
        
        recs.sort(key=lambda x: x[1], reverse=True)
        top = []
        for item, _ in recs:
            if item not in seen:
                top.append(item); seen.add(item)
            if len(top) >= K: break
        
        # Stage 2: Fill remaining with category + popularity
        if len(top) < K:
            for cat in ctx["categories"]:
                if cat in cat_to_prods:
                    for pid, _ in cat_to_prods[cat].most_common(K):
                        if pid not in seen:
                            top.append(pid); seen.add(pid)
                        if len(top) >= K: break
                if len(top) >= K: break
        
        predictions.append({"visit_id": vid, "product_ids": " ".join(top[:K]) if top else ""})
    
    # Popular fallback
    df_subm = pd.DataFrame(predictions)
    all_recs = []
    for _, r in df_subm.iterrows():
        if r["product_ids"]: all_recs.extend(r["product_ids"].split())
    popular = [p for p, _ in Counter(all_recs).most_common(K)]
    pop_str = " ".join(popular)
    
    def pad(s, n, fb):
        items = str(s).split() if pd.notna(s) and s else []
        seen = set(); res = []
        for x in items:
            if x not in seen: res.append(x); seen.add(x)
        for x in fb:
            if len(res) >= n: break
            if x not in seen: res.append(x); seen.add(x)
        return " ".join(res[:n])
    
    df_subm["product_ids"] = df_subm["product_ids"].apply(lambda x: pad(x, K, popular))
    
    test_df = pd.read_csv("data/metrika_visits_test.csv", usecols=["visit_id"], dtype=str)
    df_subm["visit_id"] = df_subm["visit_id"].astype(str)
    df_subm = df_subm.set_index("visit_id").reindex(test_df["visit_id"]).reset_index()
    df_subm["product_ids"] = df_subm["product_ids"].fillna(pop_str)
    df_subm.to_csv("submission.csv", index=False)
    
    elapsed = time.time() - t0
    n_unique = len(set(" ".join(df_subm["product_ids"]).split()))
    print(f"\n  Saved: submission.csv ({len(df_subm)} rows, {n_unique} unique products)")
    print(f"  ⏱️  {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Method: CL-GRU4Rec+RP (Unified PyTorch)")


# ============================================================================
# MAIN: SYNERISE
# ============================================================================
def run_synerise():
    t0 = time.time()
    print("=" * 80)
    print("CL-GRU4Rec+RP (Unified) — SYNERISE RecSys 2025")
    print("=" * 80)
    
    print("\n1. LOADING DATA")
    d = load_synerise_data()
    train_items = d["train_items"]; train_events = d["train_events"]
    test_gt = d["test_gt"]; cooccur = d["cooccur"]
    cat_pop = d["cat_pop"]; s2c = d["s2c"]; freq = d["freq"]
    test_uids = sorted(test_gt.keys())
    print(f"  {len(test_uids):,} test users, {len(freq):,} items")
    
    # Build vocabulary
    all_items = sorted(freq)
    item_to_idx = {"<PAD>": 0}
    for i, item in enumerate(all_items):
        item_to_idx[item] = i + 1
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    n_items = len(item_to_idx)
    print(f"  GRU vocabulary: {n_items} items")
    
    # Build sessions for GRU
    gru_sessions = []
    for uid, items in train_items.items():
        indices = [item_to_idx[s] for s in items if s in item_to_idx]
        if len(indices) >= 3:
            gru_sessions.append(indices)
    print(f"  GRU sessions: {len(gru_sessions):,}")
    
    # Build CL sessions
    cl_sessions = []
    for uid, items in train_items.items():
        indices = [item_to_idx[s] for s in items if s in item_to_idx]
        if len(set(indices)) >= 2:
            cl_sessions.append(indices)
    print(f"  CL sessions: {len(cl_sessions):,}")
    
    # Baselines
    print("\n2. BASELINES")
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
    print("\n3. GRU4Rec TRAINING")
    gru_config = {'embed': GRU_EMBED_DIM, 'hidden': GRU_HIDDEN_DIM, 'dropout': GRU_DROPOUT,
                  'max_seq': GRU_MAX_SEQ, 'batch': GRU_BATCH, 'lr': GRU_LR, 'epochs': GRU_EPOCHS}
    gru_models = []
    for i, seed in enumerate(GRU_SEEDS):
        cache = f"unified_synerise_gru_seed{seed}.pkl"
        print(f"  Model {i+1}/{len(GRU_SEEDS)}:")
        m = train_gru(gru_sessions, n_items, seed, cache, gru_config)
        gru_models.append(m)
    print(f"  Ensemble: {len(gru_models)} models")
    
    # ---- Train CL ----
    print("\n4. CONTRASTIVE LEARNING")
    cl_model = train_cl(cl_sessions, n_items, "unified_synerise_cl.pkl")
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
    r_gru, n_gru, h_gru = evaluate(preds_gru, test_gt, test_uids)
    print(f"  GRU-only:      R@6={r_gru:.4f} | NDCG={n_gru:.4f} | HR={h_gru:.4f}")
    
    # ---- CL-GRU4Rec+RP: Two-Stage Fusion ----
    print("\n6. ★ CL-GRU4Rec+RP (Two-Stage Fusion)")
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
                    for pid, cnt in cooccur[item].most_common(30):
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
    
    r_best, n_best, h_best = evaluate(preds_best, test_gt, test_uids)
    print(f"  ★ CL-GRU+RP:   R@6={r_best:.4f} | NDCG={n_best:.4f} | HR={h_best:.4f}")
    
    # ---- Comparison ----
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (Synerise RecSys 2025)")
    print("=" * 80)
    results = [
        ("Popularity",      r_pop,  n_pop,  h_pop),
        ("RePurchase only",  r_rp,   n_rp,   h_rp),
        ("GRU4Rec only",     r_gru,  n_gru,  h_gru),
        ("★ CL-GRU4Rec+RP", r_best, n_best, h_best),
    ]
    best_r = max(x[1] for x in results)
    print(f"\n  {'Method':<20} | {'R@6':>9} | {'NDCG@6':>9} | {'HR@6':>9}")
    print(f"  {'-'*20}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for nm, r, n, h in results:
        mk = " ◀ BEST" if r == best_r else ""
        print(f"  {nm:<20} | {r:>9.4f} | {n:>9.4f} | {h:>9.4f}{mk}")
    print(f"\n  ⏱️  {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Method: CL-GRU4Rec+RP (Unified PyTorch)")


# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rental", "synerise"], required=True)
    args = parser.parse_args()
    
    print(f"Device: {DEVICE}")
    if args.dataset == "rental":
        run_rental()
    else:
        run_synerise()
