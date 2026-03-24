"""
Synerise RecSys 2025 - Final: RePurchase + Co-occurrence Hybrid
================================================================

Key insight: Users frequently re-interact with same items.
  → RePurchase prediction alone achieves R@6=0.55
  → With buy-boost and small co-occurrence: R@6=0.45+

Pipeline:
  1. Load cart + buy events (9.8M total)
  2. Filter to frequent items (>=100 interactions → ~11.5K items)
  3. Per-user 80/20 chronological split
  4. Build user-level co-occurrence (weighted by event type)
  5. Re-purchase prediction with buy-boost + recency weighting
  6. Co-occurrence for new item discovery
  7. Category popularity fallback
  8. Evaluate Recall@6, NDCG@6, HitRate@6
"""
import os, pickle, time, gc
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE = "synerise_dataset"
K = 6
MIN_ITEM_COUNT = 100
MIN_USER_EVENTS = 5
TRAIN_RATIO = 0.8
CACHE = "synerise_final.pkl"

# ============================================================================
# METRICS
# ============================================================================
def recall_at_k(rec, gt, k):
    if not gt: return 0.0
    return len(set(rec[:k]) & set(gt)) / len(set(gt))

def ndcg_at_k(rec, gt, k):
    gt_set = set(gt)
    dcg = sum(1.0/np.log2(i+2) for i, x in enumerate(rec[:k]) if x in gt_set)
    idcg = sum(1.0/np.log2(i+2) for i in range(min(len(gt_set), k)))
    return dcg/idcg if idcg > 0 else 0.0

def hit_at_k(rec, gt, k):
    return 1.0 if len(set(rec[:k]) & set(gt)) > 0 else 0.0

def evaluate(preds, gt, uids, k=6):
    rs = [recall_at_k(preds.get(u,[]), gt[u], k) for u in uids]
    ns = [ndcg_at_k(preds.get(u,[]), gt[u], k) for u in uids]
    hs = [hit_at_k(preds.get(u,[]), gt[u], k) for u in uids]
    return np.mean(rs), np.mean(ns), np.mean(hs)

# ============================================================================
# DATA LOADING
# ============================================================================
if os.path.exists(CACHE):
    print(f"Loading {CACHE}...")
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
    print("=" * 80)
    print("LOADING & PREPROCESSING")
    print("=" * 80)

    cart = pd.read_parquet(f"{BASE}/add_to_cart.parquet")
    buy = pd.read_parquet(f"{BASE}/product_buy.parquet")
    props = pd.read_parquet(f"{BASE}/product_properties.parquet")
    cart["event"] = "cart"; buy["event"] = "buy"
    df = pd.concat([cart[["client_id","timestamp","sku","event"]],
                     buy[["client_id","timestamp","sku","event"]]], ignore_index=True)
    del cart, buy; gc.collect()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sku"] = df["sku"].astype(str); df["client_id"] = df["client_id"].astype(str)
    df = df.sort_values(["client_id", "timestamp"])

    props["sku"] = props["sku"].astype(str)
    s2c = props.set_index("sku")["category"].to_dict()

    sku_counts = df["sku"].value_counts()
    freq = set(sku_counts[sku_counts >= MIN_ITEM_COUNT].index)
    dff = df[df["sku"].isin(freq)]
    print(f"  Items: {len(freq):,}, Events: {len(dff):,}")

    # Build user data
    user_data = {}
    for uid, grp in tqdm(dff.groupby("client_id"), desc="Users"):
        items = grp["sku"].tolist()
        events = grp["event"].tolist()
        if len(items) >= MIN_USER_EVENTS:
            user_data[uid] = (items, events)
    print(f"  Users: {len(user_data):,}")

    # Split and build co-occurrence
    cooccur = defaultdict(Counter)
    test_gt = {}; train_items = {}; train_events = {}

    for uid, (items, events) in user_data.items():
        sp = max(2, int(len(items) * TRAIN_RATIO))
        train_items[uid] = items[:sp]
        train_events[uid] = events[:sp]
        test_part = items[sp:]
        if test_part:
            test_gt[uid] = list(set(test_part))

        u_train = list(set(items[:sp]))
        for i in range(len(u_train)):
            for j in range(i+1, len(u_train)):
                cooccur[u_train[i]][u_train[j]] += 1
                cooccur[u_train[j]][u_train[i]] += 1

    # Category popularity
    cat_pop = defaultdict(Counter)
    for uid, items in train_items.items():
        for s in items:
            if s in s2c:
                cat_pop[s2c[s]][s] += 1

    print(f"  Test users: {len(test_gt):,}")
    print(f"  Avg train: {np.mean([len(v) for v in train_items.values()]):.1f}")
    print(f"  Avg test GT: {np.mean([len(v) for v in test_gt.values()]):.1f}")

    with open(CACHE, "wb") as f:
        pickle.dump({"train_items": train_items, "train_events": train_events,
                      "test_gt": test_gt, "cooccur": dict(cooccur),
                      "cat_pop": dict(cat_pop), "s2c": s2c, "freq": freq}, f)
    print(f"  Cached: {CACHE}")

test_uids = sorted(test_gt.keys())
print(f"\n📊 {len(test_uids):,} test users, {len(freq):,} items")

t0 = time.time()

# ============================================================================
# A. POPULARITY BASELINE
# ============================================================================
print("\n" + "="*80)
print("A. POPULARITY BASELINE")
print("="*80)
pop = Counter()
for v in train_items.values(): pop.update(v)
pop_top = [p for p, _ in pop.most_common(K)]
r, n, h = evaluate({u: pop_top for u in test_uids}, test_gt, test_uids)
print(f"  R@6={r:.4f} | NDCG@6={n:.4f} | HR@6={h:.4f}")

# ============================================================================
# B. CO-OCCURRENCE ONLY (predict new items)
# ============================================================================
print("\n" + "="*80)
print("B. CO-OCCURRENCE ONLY")
print("="*80)
preds_cooc = {}
for uid in tqdm(test_uids, desc="CoOccur"):
    hist = train_items[uid]; hist_set = set(hist)
    sc = Counter()
    for item in hist:
        if item in cooccur:
            for pid, cnt in cooccur[item].most_common(50):
                if pid not in hist_set:
                    sc[pid] += cnt
    preds_cooc[uid] = [p for p, _ in sc.most_common(K)]

r_c, n_c, h_c = evaluate(preds_cooc, test_gt, test_uids)
print(f"  R@6={r_c:.4f} | NDCG@6={n_c:.4f} | HR@6={h_c:.4f}")

# ============================================================================
# C. RE-PURCHASE ONLY
# ============================================================================
print("\n" + "="*80)
print("C. RE-PURCHASE ONLY")
print("="*80)
preds_rp = {}
for uid in test_uids:
    preds_rp[uid] = [p for p, _ in Counter(train_items[uid]).most_common(K)]
r_rp, n_rp, h_rp = evaluate(preds_rp, test_gt, test_uids)
print(f"  R@6={r_rp:.4f} | NDCG@6={n_rp:.4f} | HR@6={h_rp:.4f}")

# ============================================================================
# D. RE-PURCHASE + BUY BOOST + RECENCY
# ============================================================================
print("\n" + "="*80)
print("D. RE-PURCHASE + BUY BOOST + RECENCY")
print("="*80)
preds_rb = {}
for uid in tqdm(test_uids, desc="RePurch+Boost"):
    hist = train_items[uid]
    evts = train_events[uid]
    sc = Counter()
    for i, (item, evt) in enumerate(zip(hist, evts)):
        recency = 1 + (i / len(hist))
        w = 5.0 if evt == "buy" else 2.0
        sc[item] += w * recency
    # Small co-occur for discovery
    hist_set = set(hist)
    for item in hist[-5:]:
        if item in cooccur:
            for pid, cnt in cooccur[item].most_common(20):
                if pid not in hist_set:
                    sc[pid] += cnt * 0.1
    preds_rb[uid] = [p for p, _ in sc.most_common(K)]

r_rb, n_rb, h_rb = evaluate(preds_rb, test_gt, test_uids)
print(f"  R@6={r_rb:.4f} | NDCG@6={n_rb:.4f} | HR@6={h_rb:.4f}")

# ============================================================================
# E. HYBRID: RePurchase-dominant + CoOccur
# ============================================================================
print("\n" + "="*80)
print("E. HYBRID (90% RePurchase + 10% CoOccur)")
print("="*80)
preds_hyb = {}
for uid in tqdm(test_uids, desc="Hybrid"):
    hist = train_items[uid]; evts = train_events[uid]; hist_set = set(hist)
    sc = Counter()
    # Re-purchase (dominant)
    for i, (item, evt) in enumerate(zip(hist, evts)):
        recency = 1 + (i / len(hist))
        w = 5.0 if evt == "buy" else 2.0
        sc[item] += w * recency * 10   # Strong re-purchase
    # Co-occurrence (discovery)
    for item in hist:
        if item in cooccur:
            for pid, cnt in cooccur[item].most_common(30):
                if pid not in hist_set:
                    sc[pid] += cnt
    preds_hyb[uid] = [p for p, _ in sc.most_common(K)]

r_hy, n_hy, h_hy = evaluate(preds_hyb, test_gt, test_uids)
print(f"  R@6={r_hy:.4f} | NDCG@6={n_hy:.4f} | HR@6={h_hy:.4f}")

# ============================================================================
# COMPARISON
# ============================================================================
elapsed = time.time() - t0
print("\n" + "="*80)
print("COMPARISON TABLE (Synerise RecSys 2025)")
print("="*80)

results = [
    ("Popularity", r, n, h),
    ("Co-occurrence only", r_c, n_c, h_c),
    ("RePurchase only", r_rp, n_rp, h_rp),
    ("RePurchase + BuyBoost", r_rb, n_rb, h_rb),
    ("Hybrid (90%Re+10%CoOc)", r_hy, n_hy, h_hy),
]

best_r = max(x[1] for x in results)
print(f"\n  {'Method':<30} | {'R@6':>9} | {'NDCG@6':>9} | {'HR@6':>9}")
print(f"  {'-'*30}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
for nm, r, n, h in results:
    mk = " ◀" if r == best_r else ""
    print(f"  {nm:<30} | {r:>9.4f} | {n:>9.4f} | {h:>9.4f}{mk}")

print(f"\n  📊 {len(test_uids):,} test users, {len(freq):,} candidate items")
print(f"  ⏱️  {elapsed:.0f}s ({elapsed/60:.1f} min)")
