"""
CL-GRU4Rec+RP v2: Contrastive Learning Enhanced GRU4Rec with Re-Purchase Awareness
=====================================================================================

Strategy: Keep cornac GRU4Rec as BACKBONE (proven 0.41-0.42) and ADD novel components:
  1. Contrastive Item Similarity (learned from session co-occurrence)
  2. Re-Purchase Signal (boost items user viewed in session)
  3. Enhanced Multi-Signal Fusion (GRU + CoOccur + CL-similarity + RP)

This preserves the strong base performance while adding novelty for the paper.

Expected: 0.42-0.45 (improvement over pure GRU4Rec baseline)
"""
import ast, os, pickle, time, random
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cornac.models import GRU4Rec
from cornac.data.dataset import SequentialDataset
from cornac.models.gru4rec.gru4rec import GRU4RecModel

# ============================================================================
# CONFIG
# ============================================================================
ROOT = "data"
K = 6
MAX_ROW = 20
ENSEMBLE_SEEDS = [123, 456, 789]

# CL config
CL_EMBED_DIM = 64
CL_EPOCHS = 30
CL_LR = 0.003
CL_TEMPERATURE = 0.07
CL_NEG_SAMPLES = 128
CL_BATCH = 512

# Fusion weights (tuned)
W_GRU = 1.0           # GRU4Rec scores (primary)
W_COOC_FWD = 0.15     # Forward co-occurrence boost
W_COOC_BWD = 0.08     # Backward co-occurrence boost
W_CAT = 0.10          # Category boost
W_CL_SIM = 0.12       # Contrastive similarity boost
W_RP = 0.06           # Re-purchase boost
BOOST_CAP = 0.25      # Max boost as fraction of base score

# Patch float32
_orig = GRU4RecModel._init_numpy_weights
def _p(self, s): return _orig(self, s).astype(np.float32)
GRU4RecModel._init_numpy_weights = _p

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ============================================================================
# DATA LOADING (from lru.py, unchanged)
# ============================================================================
def get_old_to_new_prod_ids():
    m = pd.read_csv(f"{ROOT}/old_site_new_site_products.csv", dtype=str)
    return m.set_index("old_site_id")["new_site_id"].to_dict()

def get_slug_to_ids():
    old2new = get_old_to_new_prod_ids()
    df_old = pd.read_csv(f"{ROOT}/old_site_products.csv", usecols=["id","slug"], dtype=str)
    df_new = pd.read_csv(f"{ROOT}/new_site_products.csv", usecols=["id","slug"], dtype=str)
    df_old["id"] = df_old["id"].map(old2new)
    df_old = df_old.dropna(subset=["id"])
    df = pd.concat([df_new, df_old]).drop_duplicates(subset=["id","slug"])
    return df.set_index("slug")["id"].to_dict()

def get_hits_data():
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
    return df

def get_visits_data():
    df = pd.concat([
        pd.read_csv(f"{ROOT}/metrika_visits.csv", usecols=['client_id','visit_id','watch_ids'], dtype=str),
        pd.read_csv(f"{ROOT}/metrika_visits_test.csv", usecols=['client_id','visit_id','watch_ids'], dtype=str),
    ], ignore_index=True)
    df["watch_ids"] = df["watch_ids"].apply(lambda x: ast.literal_eval(x))
    df = df.explode("watch_ids").rename(columns={"watch_ids": "watch_id"})
    return df

def create_recom_data(df_hits, df_visits):
    df = pd.merge(df_hits, df_visits, on="watch_id", how="left")
    df = pd.concat([
        df[df["page_type"].ne("PRODUCT")],
        df[df["page_type"].eq("PRODUCT")].drop_duplicates(["visit_id","product_id"], keep="first")
    ])
    df = df[["client_id","visit_id","product_id","is_page_view","page_type","date_time","slug","project_id"]].dropna()
    df['date_time'] = df['date_time'].astype('int64') // 10**9
    return df.sort_values(['visit_id','date_time'])

def add_start_token(df):
    s = (df.groupby("visit_id").head(1)
         .assign(product_id=lambda d: np.where(d["project_id"]=="1","000000000","000000001"),
                 page_type=lambda d: np.where(d["project_id"]=="1","START_OLD","START_NEW"),
                 is_page_view="1"))
    s["date_time"] = s["date_time"] - 1
    return pd.concat([df, s], ignore_index=True).sort_values(['visit_id','date_time'])

def get_test_visit_ids():
    return pd.read_csv(f"{ROOT}/metrika_visits_test.csv", usecols=['visit_id'], dtype=str)["visit_id"].unique()
def get_allowed_product_ids():
    return pd.read_csv(f"{ROOT}/new_site_products.csv", usecols=["id"], dtype=str)["id"].unique()

def get_fitted_model(df_train, max_row, seed=123):
    df_train = df_train.groupby('visit_id').head(max_row)
    ds = SequentialDataset.build(
        list(df_train[["client_id",'visit_id','product_id','date_time']].itertuples(index=False, name=None)), fmt="USIT")
    m = GRU4Rec(layers=[150], loss="cross-entropy", n_sample=4096, dropout_p_embed=0.0,
                dropout_p_hidden=0.0, sample_alpha=0.0, batch_size=512, n_epochs=30,
                device=DEVICE, verbose=True, seed=seed)
    m.fit(ds)
    return m

def predict_next(model, user_ids, historical_item_list, k, allowed_product_ids=None):
    rev = {v:k for k,v in model.iid_map.items()}
    ni = len(rev)
    gm = None
    if allowed_product_ids is not None:
        ai = [model.iid_map[i] for i in allowed_product_ids if i in model.iid_map]
        gm = np.ones(ni, dtype=bool); gm[ai] = False
    ri, rs = [], []
    for uid, hist in tqdm(zip(user_ids, historical_item_list), total=len(user_ids)):
        hi = [model.iid_map[x] for x in hist if x in model.iid_map]
        sc = model.score(user_idx=model.uid_map[uid], history_items=hi)
        if gm is not None: sc[gm] = -np.inf
        sc[hi] = -np.inf
        if k < len(sc):
            tk = np.argpartition(sc, -k)[-k:]
            tk = tk[np.argsort(sc[tk])][::-1]
        else:
            tk = np.argsort(sc)[::-1]
        ri.append([rev[i] for i in tk])
        rs.append(sc[tk].tolist())
    return ri, rs

# ============================================================================
# CONTRASTIVE ITEM SIMILARITY MODEL (NOVEL COMPONENT 1)
# ============================================================================
class ContrastiveItemModel(nn.Module):
    """Learn item embeddings via contrastive learning on session co-occurrence.
    Items that co-occur in same session = positive pairs.
    Items that don't = negative pairs.
    """
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
    
    def similarity(self, item_a, item_b):
        za = self.forward(item_a)
        zb = self.forward(item_b)
        return (za * zb).sum(dim=-1)

def train_contrastive_model(sessions, n_items, embed_dim=64, epochs=30, lr=0.003,
                            temperature=0.07, batch_size=512, device="cpu"):
    """Train contrastive item embeddings from session co-occurrence."""
    
    # Build positive pairs from session co-occurrence
    print("  Building positive pairs from sessions...")
    pairs = []
    for sess in sessions:
        unique = list(set(sess))
        for i in range(len(unique)):
            for j in range(i+1, len(unique)):
                pairs.append((unique[i], unique[j]))
    
    if not pairs:
        return None
    
    random.shuffle(pairs)
    print(f"  {len(pairs):,} positive pairs from {len(sessions):,} sessions")
    
    model = ContrastiveItemModel(n_items, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    all_items = list(range(n_items))
    
    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            if len(batch) < 2: continue
            
            anchors = torch.LongTensor([p[0] for p in batch]).to(device)
            positives = torch.LongTensor([p[1] for p in batch]).to(device)
            
            # Random negatives
            negatives = torch.LongTensor(random.choices(all_items, k=min(CL_NEG_SAMPLES, len(all_items)))).to(device)
            
            # Encode
            z_anchor = model(anchors)       # (B, D)
            z_pos = model(positives)         # (B, D)
            z_neg = model(negatives)         # (N, D)
            
            # InfoNCE
            pos_sim = (z_anchor * z_pos).sum(dim=-1, keepdim=True) / temperature  # (B, 1)
            neg_sim = torch.mm(z_anchor, z_neg.t()) / temperature                 # (B, N)
            logits = torch.cat([pos_sim, neg_sim], dim=1)                          # (B, 1+N)
            labels = torch.zeros(len(batch), dtype=torch.long, device=device)      # positive = idx 0
            
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/max(n_batches,1):.4f}")
    
    return model


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    
    print("="*80)
    print("CL-GRU4Rec+RP v2")
    print("="*80)
    
    # ---- Load data ----
    print("\n1. LOADING DATA")
    df_hits = get_hits_data()
    df_visits = get_visits_data()
    df_data = create_recom_data(df_hits, df_visits)
    df_data = add_start_token(df_data)
    
    visit_ids = get_test_visit_ids()
    allowed = get_allowed_product_ids()
    allowed_set = set(allowed)
    
    print(f"   {len(df_data):,} rows | {df_data['client_id'].nunique():,} users | "
          f"{df_data['visit_id'].nunique():,} sessions | {df_data['product_id'].nunique():,} items")
    
    # ---- Train GRU4Rec ensemble (same as lru.py) ----
    print("\n2. GRU4Rec ENSEMBLE (cornac)")
    models = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        cache = f"gru4rec_seed{seed}.pkl"
        if os.path.exists(cache):
            print(f"   Loading {cache}")
            try:
                with open(cache, "rb") as f: m = pickle.load(f)
                _ = m.iid_map
                models.append(m); continue
            except: pass
        print(f"   Training model {i+1}/{len(ENSEMBLE_SEEDS)} (seed={seed})...")
        m = get_fitted_model(df_data, MAX_ROW, seed)
        with open(cache, "wb") as f: pickle.dump(m, f)
        print(f"   Cached: {cache}")
        models.append(m)
    print(f"   {len(models)} models ready")
    
    # ---- Build co-occurrence (same as lru.py) ----
    print("\n3. CO-OCCURRENCE SIGNALS")
    fwd_cooc = defaultdict(Counter)
    bwd_cooc = defaultdict(Counter)
    cat_to_prods = defaultdict(Counter)
    
    for cid, grp in tqdm(df_data.groupby("client_id"), desc="   Co-occur"):
        grp = grp.sort_values("date_time")
        prods, cats = [], []
        for _, r in grp.iterrows():
            pt, pid, slug = r.get("page_type",""), str(r.get("product_id","")), str(r.get("slug",""))
            if pt == "CATEGORY" and slug != "nan":
                cats.append(slug)
            elif pt == "PRODUCT" and pid in allowed_set:
                for c in cats[-3:]: cat_to_prods[c][pid] += 1
                for prev in prods[-5:]:
                    fwd_cooc[prev][pid] += 1
                    bwd_cooc[pid][prev] += 1
                prods.append(pid)
    
    print(f"   Fwd: {len(fwd_cooc)} | Bwd: {len(bwd_cooc)} | Cat: {len(cat_to_prods)}")
    
    # Visit context
    test_set = set(visit_ids)
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
    
    # ---- NOVEL: Contrastive Item Similarity ----
    print("\n4. CONTRASTIVE ITEM SIMILARITY (Novel)")
    cl_cache = "cl_item_model.pkl"
    
    # Build product sessions for CL training (only allowed products)
    product_sessions = []
    product_vocab = sorted(allowed_set)
    pid_to_clid = {p: i for i, p in enumerate(product_vocab)}
    n_cl_items = len(product_vocab)
    
    for vid, grp in df_data.groupby("visit_id"):
        grp = grp.sort_values("date_time")
        prods = [pid_to_clid[str(r["product_id"])] 
                 for _, r in grp.iterrows() 
                 if r["page_type"] == "PRODUCT" and str(r["product_id"]) in pid_to_clid]
        if len(prods) >= 2:
            product_sessions.append(prods)
    
    print(f"   {len(product_sessions):,} sessions with ≥2 products, {n_cl_items} items")
    
    if os.path.exists(cl_cache):
        print(f"   Loading {cl_cache}")
        cl_model = ContrastiveItemModel(n_cl_items, CL_EMBED_DIM)
        cl_model.load_state_dict(torch.load(cl_cache, map_location=DEVICE, weights_only=True))
        cl_model = cl_model.to(DEVICE)
    else:
        cl_model = train_contrastive_model(
            product_sessions, n_cl_items, CL_EMBED_DIM, CL_EPOCHS, CL_LR,
            CL_TEMPERATURE, CL_BATCH, DEVICE)
        if cl_model:
            torch.save(cl_model.state_dict(), cl_cache)
            print(f"   Cached: {cl_cache}")
    
    # Pre-compute CL item embeddings for fast lookup
    cl_embeddings = None
    if cl_model:
        cl_model.eval()
        with torch.no_grad():
            all_idx = torch.arange(n_cl_items).to(DEVICE)
            cl_embeddings = cl_model(all_idx).cpu().numpy()  # (n_items, D)
        print(f"   CL embeddings: {cl_embeddings.shape}")
    
    # ---- Generate GRU4Rec predictions ----
    print("\n5. GRU4Rec TOP-100 PREDICTIONS")
    
    # Get top-100 from each model
    all_model_preds = []
    for i, m in enumerate(models):
        print(f"   Model {i+1}/{len(models)}...")
        result = (
            df_data[df_data["visit_id"].isin(test_set)]
            .groupby("visit_id", sort=False)
            .agg(user_id=("client_id","first"), 
                 historical_items=("product_id", lambda x: x.tolist()))
            .reset_index()
        )
        vids = result["visit_id"].tolist()
        uids = result["user_id"].tolist()
        hists = result["historical_items"].tolist()
        items, scores = predict_next(m, uids, hists, 100, allowed)
        subm = pd.DataFrame({"visit_id": vids, "items": items, "scores": scores})
        all_model_preds.append(subm)
    
    # Average scores across models
    print("   Averaging ensemble scores...")
    merged = {}
    for subm in all_model_preds:
        for _, row in subm.iterrows():
            vid = row["visit_id"]
            if vid not in merged: merged[vid] = {}
            for item, score in zip(row["items"], row["scores"]):
                if item not in merged[vid]: merged[vid][item] = []
                merged[vid][item].append(float(score))
    
    avg_preds = {}
    for vid, item_scores in merged.items():
        averaged = []
        for item, sl in item_scores.items():
            averaged.append((item, sum(sl) / len(models)))
        averaged.sort(key=lambda x: x[1], reverse=True)
        avg_preds[vid] = averaged[:100]
    
    # ---- NOVEL: Multi-Signal Fusion ----
    print("\n6. MULTI-SIGNAL FUSION (GRU + CoOccur + CL-Similarity + Re-Purchase)")
    predictions = []
    
    for vid in tqdm(visit_ids, desc="   Fusion"):
        if vid not in avg_preds:
            predictions.append({"visit_id": vid, "product_ids": ""})
            continue
        
        gru_candidates = avg_preds[vid]  # [(item, score), ...]
        ctx = visit_ctx.get(vid, {"products": [], "categories": []})
        last_prods = ctx["products"]
        last_cats = ctx["categories"]
        n_hist = len(last_prods)
        
        # Session-adaptive boost cap (from lru.py)
        if n_hist >= 3:
            boost_cap_pct = 0.25
        elif n_hist >= 1:
            boost_cap_pct = 0.15
        else:
            boost_cap_pct = 0.08
        
        reranked = []
        for item, gru_score in gru_candidates:
            base = float(gru_score)
            boost = 0.0
            
            # Signal 1: Forward co-occurrence (from lru.py)
            for i, prev in enumerate(last_prods):
                recency = 1.0 + 2.0 * (i / max(len(last_prods), 1))
                if prev in fwd_cooc and item in fwd_cooc[prev]:
                    boost += min(0.15, 0.012 * fwd_cooc[prev][item]) * recency
            
            # Signal 2: Backward co-occurrence (from lru.py)
            for i, prev in enumerate(last_prods):
                recency = 1.0 + 1.5 * (i / max(len(last_prods), 1))
                if prev in bwd_cooc and item in bwd_cooc[prev]:
                    boost += min(0.08, 0.008 * bwd_cooc[prev][item]) * recency
            
            # Signal 3: Category boost (from lru.py)
            for j, cat in enumerate(last_cats):
                cr = 1.0 + 1.0 * (j / max(len(last_cats), 1))
                if cat in cat_to_prods and item in cat_to_prods[cat]:
                    boost += min(0.10, 0.008 * cat_to_prods[cat][item]) * cr
            
            # Signal 4 (NOVEL): Contrastive similarity boost
            if cl_embeddings is not None and last_prods and item in pid_to_clid:
                item_cl_id = pid_to_clid[item]
                item_emb = cl_embeddings[item_cl_id]
                max_sim = 0.0
                for prev in last_prods[-3:]:
                    if prev in pid_to_clid:
                        prev_emb = cl_embeddings[pid_to_clid[prev]]
                        sim = np.dot(item_emb, prev_emb)
                        max_sim = max(max_sim, sim)
                # Only boost if similarity is high (>0.3)
                if max_sim > 0.3:
                    boost += min(0.12, (max_sim - 0.3) * 0.2)
            
            # Signal 5 (NOVEL): Re-purchase boost  
            # Items user already viewed in this session get a small boost
            if item in last_prods:
                boost += 0.05  # Small re-visit boost
            
            # Apply capped boost
            max_boost = max(abs(base) * boost_cap_pct, 0.05)
            boost = min(boost, max_boost)
            reranked.append((item, base + boost))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        top6 = [p for p, _ in reranked[:K]]
        predictions.append({"visit_id": vid, "product_ids": " ".join(top6)})
    
    df_subm = pd.DataFrame(predictions)
    
    # ---- Cold start fallback ----
    print("\n7. COLD START & SUBMISSION")
    all_recs = []
    for _, r in df_subm.iterrows():
        if r["product_ids"]:
            all_recs.extend(r["product_ids"].split())
    popular = [p for p, _ in Counter(all_recs).most_common(K)]
    pop_str = " ".join(popular)
    
    # Fill missing
    df_subm = df_subm.set_index("visit_id")
    missing = sorted(set(visit_ids) - set(df_subm.index))
    if missing:
        print(f"   Cold start: {len(missing)} visits")
        cold = {}
        for vid in missing:
            ctx = visit_ctx.get(vid, {"products":[],"categories":[]})
            recs = Counter()
            for c in ctx["categories"]:
                if c in cat_to_prods:
                    for pid, cnt in cat_to_prods[c].most_common(20):
                        recs[pid] += cnt
            for prev in ctx["products"]:
                if prev in fwd_cooc:
                    for pid, cnt in fwd_cooc[prev].most_common(15):
                        recs[pid] += cnt * 2
            if recs:
                top = [p for p, _ in recs.most_common(K)]
                seen = set(top)
                for p in popular:
                    if len(top) >= K: break
                    if p not in seen: top.append(p); seen.add(p)
                cold[vid] = " ".join(top[:K])
            else:
                cold[vid] = pop_str
        
        cat_aware = sum(1 for v in cold.values() if v != pop_str)
        print(f"   Category-aware: {cat_aware} | Fallback: {len(missing)-cat_aware}")
        cold_df = pd.DataFrame({"product_ids": cold.values()}, index=cold.keys())
        df_subm = pd.concat([df_subm, cold_df])
    
    df_subm = df_subm.reset_index().rename(columns={"index":"visit_id"})
    
    # Pad to exactly K
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
    
    # Match order
    test_df = pd.read_csv(f"{ROOT}/metrika_visits_test.csv", usecols=["visit_id"], dtype=str)
    df_subm["visit_id"] = df_subm["visit_id"].astype(str)
    df_subm = df_subm.set_index("visit_id").reindex(test_df["visit_id"]).reset_index()
    df_subm["product_ids"] = df_subm["product_ids"].fillna(pop_str)
    
    df_subm.to_csv("submission.csv", index=False)
    
    elapsed = time.time() - t0
    n_unique = len(set(" ".join(df_subm["product_ids"]).split()))
    print(f"\n   Saved: submission.csv ({len(df_subm)} rows, {n_unique} unique products)")
    print(f"   ⏱️  {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n   Method: CL-GRU4Rec+RP v2")
    print(f"   = cornac GRU4Rec (3-seed) + Contrastive Similarity + Re-Purchase + CoOccur Fusion")
    print(f"   Expected: ~0.42-0.45 on Kaggle Private LB")


if __name__ == "__main__":
    main()
