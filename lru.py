# %% [code] {"jupyter":{"outputs_hidden":false}}
# !pip install -qq cornac==2.3.5

# %% [markdown]
"""
# Product Recommendation System - GRU4Rec + Prediction-Level Fusion
#
# Pipeline:
#   1. Load & prepare data
#   2. Train GRU4Rec (150 layers, 30 epochs) - cached after first run
#   3. Build lightweight co-occurrence signals
#   4. Prediction-level fusion: re-rank GRU4Rec top-50 using co-occurrence
#   5. Handle cold start, save submission
#
# Why prediction-level fusion instead of score-mixing?
#   The old ensemble (v16/v42/v45 score-mixing) had 3 critical bugs:
#     Bug 1: v16 base bias (5.2) >> GRU4Rec base bias (1.5) -> GRU crushed
#     Bug 2: Multiplicative consensus (2.53x) amplified behavioral items
#     Bug 3: v16/v42/v45 used identical data -> fake consensus
#   Result: 0.296 instead of 0.42
#
#   Fix: GRU4Rec stays primary. Behavioral signals ONLY re-rank within
#   GRU4Rec's own top-50 candidates (cannot hurt, can only help).
#
# Expected: ~0.41-0.42 (Kaggle Private)
# Runtime:  ~25 min first run, ~3 min cached reruns
"""

# %% [code] {"jupyter":{"outputs_hidden":false}}
import ast
import re
import os
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from cornac.models import GRU4Rec
from cornac.data.dataset import SequentialDataset
from cornac.models.gru4rec.gru4rec import GRU4RecModel

root = "data"

# %% [code] {"jupyter":{"outputs_hidden":false}}
_original_init = GRU4RecModel._init_numpy_weights

def _patched_init_numpy_weights(self, shape):
    weights = _original_init(self, shape)
    return weights.astype(np.float32)

GRU4RecModel._init_numpy_weights = _patched_init_numpy_weights

# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using CUDA")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using MPS (Apple Silicon)")
else:
    DEVICE = "cpu"
    print("Using CPU")

# %% [code] {"jupyter":{"outputs_hidden":false}}
def get_old_to_new_prod_ids():
    old_to_new_prod_id = pd.read_csv(f"{root}/old_site_new_site_products.csv", dtype=str)
    old_to_new_prod_id = old_to_new_prod_id.set_index("old_site_id")["new_site_id"].to_dict()
    return old_to_new_prod_id

# %% [code] {"jupyter":{"outputs_hidden":false}}
def get_slug_to_ids():
    old_to_new_prod_id = get_old_to_new_prod_ids()
    df_prod_old = pd.read_csv(f"{root}/old_site_products.csv", usecols=["id", "slug"], dtype=str)
    df_prod_new = pd.read_csv(f"{root}/new_site_products.csv", usecols=["id", "slug"], dtype=str)
    df_prod_old["id"] = df_prod_old["id"].map(old_to_new_prod_id)
    df_prod_old = df_prod_old.dropna(subset=["id"])
    df_prod = pd.concat([df_prod_new, df_prod_old], axis=0)
    df_prod = df_prod.drop_duplicates(subset=["id", "slug"])
    slug_to_id = df_prod.set_index("slug")["id"].to_dict()
    return slug_to_id

# %% [code] {"jupyter":{"outputs_hidden":false}}
def predict_next(model, user_ids, historical_item_list, k, allowed_product_ids=None):
    rev_iid_map = {v: k for k, v in model.iid_map.items()}
    num_items = len(rev_iid_map)
    global_mask = None
    if allowed_product_ids is not None:
        allowed_internal_ids = [
            model.iid_map[iid]
            for iid in allowed_product_ids
            if iid in model.iid_map
        ]
        global_mask = np.ones(num_items, dtype=bool)
        global_mask[allowed_internal_ids] = False

    recom_items, recom_scores = [], []
    for user_id, history_items in tqdm(zip(user_ids, historical_item_list), total=len(user_ids)):
        history_internal_idxs = [model.iid_map[x] for x in history_items if x in model.iid_map]
        scores = model.score(
            user_idx=model.uid_map[user_id],
            history_items=history_internal_idxs,
        )
        if global_mask is not None:
            scores[global_mask] = -np.inf
        scores[history_internal_idxs] = -np.inf

        if k < len(scores):
            unsorted_topk_idx = np.argpartition(scores, -k)[-k:]
            topk_idx_sorted = unsorted_topk_idx[np.argsort(scores[unsorted_topk_idx])][::-1]
        else:
            topk_idx_sorted = np.argsort(scores)[::-1]

        topk_vals = scores[topk_idx_sorted]
        topk_iid = [rev_iid_map[idx] for idx in topk_idx_sorted]
        recom_items.append(topk_iid)
        recom_scores.append(topk_vals.tolist())

    return recom_items, recom_scores

# %% [code] {"jupyter":{"outputs_hidden":false}}
def get_hits_data():
    slug_to_ids = get_slug_to_ids()
    df_hits = pd.concat([
        pd.read_csv(f"{root}/metrika_hits.csv",
                     usecols=['date_time', 'slug', 'page_type', "project_id", "is_page_view", "watch_id"],
                     dtype=str),
        pd.read_csv(f"{root}/metrika_hits_test.csv",
                     usecols=['date_time', 'slug', 'page_type', "project_id", "is_page_view", "watch_id"],
                     dtype=str),
    ], ignore_index=True, axis=0)

    df_hits["date_time"] = pd.to_datetime(df_hits["date_time"], format="ISO8601")
    df_hits = df_hits[df_hits["is_page_view"].eq("1")]

    for pt, val in [("SEARCH", "search"), ("CART", "cart"), ("CHECKOUT", "checkout"),
                    ("ORDER", "order"), ("UNAVAILABLE_PRODUCT", "unavailable")]:
        df_hits.loc[df_hits["page_type"].eq(pt), "slug"] = val

    df_hits = df_hits.dropna(subset=["slug"])
    df_hits["product_id"] = df_hits["slug"].map(slug_to_ids)
    missing_product_id = df_hits["product_id"].isnull()
    non_identified_slugs = df_hits.loc[missing_product_id, "slug"].unique()
    new_mapper = {slug: str(500000000 + i) for i, slug in enumerate(non_identified_slugs)}
    slug_to_ids.update(new_mapper)
    df_hits["product_id"] = df_hits["slug"].map(slug_to_ids)
    return df_hits

# %% [code] {"jupyter":{"outputs_hidden":false}}
def get_visits_data():
    df_visits = pd.concat([
        pd.read_csv(f"{root}/metrika_visits.csv",
                     usecols=['client_id', 'visit_id', 'watch_ids'], dtype=str),
        pd.read_csv(f"{root}/metrika_visits_test.csv",
                     usecols=['client_id', 'visit_id', 'watch_ids'], dtype=str),
    ], ignore_index=True, axis=0)
    df_visits["watch_ids"] = df_visits["watch_ids"].apply(lambda x: ast.literal_eval(x))
    df_visits = df_visits.explode("watch_ids")
    df_visits = df_visits.rename(columns={"watch_ids": "watch_id"})
    return df_visits

# %% [code] {"jupyter":{"outputs_hidden":false}}
def create_recom_data(df_hits, df_visits):
    df_merged = pd.merge(df_hits, df_visits, on="watch_id", how="left")
    df_merged = pd.concat([
        df_merged[df_merged["page_type"].ne("PRODUCT")],
        df_merged[df_merged["page_type"].eq("PRODUCT")].drop_duplicates(["visit_id", "product_id"], keep="first")
    ])
    df_data = df_merged[["client_id", "visit_id", "product_id", "is_page_view", "page_type", "date_time", "slug", "project_id"]].dropna()
    df_data['date_time'] = df_data['date_time'].astype('int64') // 10 ** 9
    df_data = df_data.sort_values(['visit_id', 'date_time'])
    return df_data

# %% [code] {"jupyter":{"outputs_hidden":false}}
def add_start_token(df):
    df_start = (
        df
        .groupby("visit_id").head(1)
        .assign(
            product_id=lambda d: np.where(d["project_id"] == "1", "000000000", "000000001"),
            page_type=lambda d: np.where(d["project_id"] == "1", "START_OLD", "START_NEW"),
            is_page_view="1",
        )
    )
    df_start["date_time"] = df_start["date_time"] - 1
    df = pd.concat([df, df_start], axis=0, ignore_index=True)
    df = df.sort_values(['visit_id', 'date_time'])
    return df

# %% [code] {"jupyter":{"outputs_hidden":false}}
def get_data_splits(df_data, split_threshold):
    df_data = df_data.copy()
    df_data = df_data.sort_values(["visit_id", "date_time"])
    df_data['pos'] = df_data.groupby('visit_id', sort=False).cumcount() + 1
    df_data['total_items'] = df_data.groupby('visit_id')['product_id'].transform('count')
    df_data['is_train'] = df_data['pos'] <= (df_data['total_items'] * split_threshold).round()
    df_data.loc[df_data["project_id"].eq("1"), "is_train"] = True
    df_train = df_data[df_data['is_train']].copy()
    df_valid = df_data[~df_data['is_train']].copy()
    df_train = df_train.drop(columns=['pos', 'total_items', 'is_train'])
    df_valid = df_valid.drop(columns=['pos', 'total_items', 'is_train'])
    visit_ids = get_test_visit_ids()
    product_ids = get_allowed_product_ids()
    df_test = df_valid[
        df_valid["project_id"].eq("0") &
        df_valid["visit_id"].isin(visit_ids) &
        df_valid["product_id"].isin(product_ids)
    ]
    df_valid = df_valid[
        df_valid["project_id"].eq("0") &
        (~df_valid["visit_id"].isin(visit_ids)) &
        df_valid["product_id"].isin(product_ids)
    ]
    return df_train, df_valid, df_test

# %% [code] {"jupyter":{"outputs_hidden":false}}
def get_test_visit_ids():
    return pd.read_csv(f"{root}/metrika_visits_test.csv", usecols=['visit_id'], dtype=str)["visit_id"].unique()

def get_allowed_product_ids():
    return pd.read_csv(f"{root}/new_site_products.csv", usecols=["id"], dtype=str)["id"].unique()

# %% [code] {"jupyter":{"outputs_hidden":false}}
def get_fitted_model(df_train, max_row_per_session, seed=123):
    df_train = df_train.groupby('visit_id').head(max_row_per_session)
    train_data = SequentialDataset.build(
        list(df_train[["client_id", 'visit_id', 'product_id', 'date_time'
        ]].itertuples(index=False, name=None)), fmt="USIT")
    model = GRU4Rec(
        layers=[150],
        loss="cross-entropy",
        n_sample=4096,
        dropout_p_embed=0.0,
        dropout_p_hidden=0.0,
        sample_alpha=0.0,
        batch_size=512,
        n_epochs=30,
        device=DEVICE,
        verbose=True,
        seed=seed,
    )
    model.fit(train_data)
    return model

# %% [code] {"jupyter":{"outputs_hidden":false}}
def create_submission(fitted_model, hist_data, visit_ids, k, allowed_product_ids):
    result = (
        hist_data[hist_data["visit_id"].isin(visit_ids)]
        .groupby("visit_id", sort=False)
        .agg(
            user_id=("client_id", "first"),
            historical_items=("product_id", lambda x: x.tolist())
        )
        .reset_index()
    )
    visit_ids = result["visit_id"].tolist()
    user_ids = result["user_id"].tolist()
    historical_item_list = result["historical_items"].tolist()
    recoms, scores = predict_next(fitted_model, user_ids, historical_item_list, k, allowed_product_ids)
    df_subm = pd.DataFrame({"visit_id": visit_ids, "product_ids": recoms, "scores": scores})
    df_subm["product_ids"] = df_subm["product_ids"].apply(lambda x: " ".join(x))
    return df_subm

# %% [code] {"jupyter":{"outputs_hidden":false}}
def create_gt_submission(df_test):
    return (
        df_test.groupby("visit_id", sort=False)["product_id"]
        .agg(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )

def evaluate_submission(df_subm, df_gt, n_list):
    def recall_at_k(rec, gt, k):
        if len(gt) == 0:
            return 0.0
        return len(set(rec[:k]) & set(gt)) / len(set(gt))
    df_gt = df_gt.copy()
    df_subm = df_subm.copy()
    df_subm["rec_list"] = df_subm["product_ids"].str.split()
    df_gt["gt_list"] = df_gt["product_id"].str.split()
    df = df_subm.merge(df_gt, on="visit_id", how="inner")
    for n in n_list:
        df["recall"] = df.apply(lambda x: recall_at_k(x["rec_list"], x["gt_list"], k=n), axis=1)
        print(f"  recall@{n}: {df['recall'].mean():.4f}")

# ============================================================================
# ========================  EXECUTION STARTS HERE  ===========================
# ============================================================================

# %% [code] {"jupyter":{"outputs_hidden":false}}
max_row_per_session = 20
n_candidates = 6
split_ratio = 0.50
MODEL_CACHE = "gru4rec_final.pkl"
SKIP_VALIDATION = True  # True = skip validation (~22 min faster)
ENSEMBLE_SEEDS = [123, 456, 789]  # Multi-seed ensemble for score averaging

visit_ids = get_test_visit_ids()
allowed_product_ids = get_allowed_product_ids()

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df_hits = get_hits_data()
df_visits = get_visits_data()
df_data = create_recom_data(df_hits, df_visits)
df_data = add_start_token(df_data)

print(f"Total: {len(df_data):,} rows | "
      f"{df_data['client_id'].nunique():,} users | "
      f"{df_data['visit_id'].nunique():,} sessions | "
      f"{df_data['product_id'].nunique():,} items\n")

# %% [code] {"jupyter":{"outputs_hidden":false}}
if not SKIP_VALIDATION:
    df_train, df_valid, df_test = get_data_splits(df_data, split_ratio)
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print(f"Train: {len(df_train):,} | Valid: {len(df_valid):,} | Test: {len(df_test):,}")
    print("\nTraining validation model...")
    val_model = get_fitted_model(df_train, max_row_per_session)

    subm_valid = create_submission(val_model, df_train,
                                   df_valid["visit_id"].unique(),
                                   n_candidates, allowed_product_ids)
    subm_gt_valid = create_gt_submission(df_valid)
    print("\n--- VALIDATION RECALL ---")
    evaluate_submission(subm_valid, subm_gt_valid, [n_candidates])

    subm_test = create_submission(val_model, df_train,
                                  df_test["visit_id"].unique(),
                                  n_candidates, allowed_product_ids)
    subm_gt_test = create_gt_submission(df_test)
    print("\n--- TEST RECALL ---")
    evaluate_submission(subm_test, subm_gt_test, [n_candidates])
    del val_model
    print()
else:
    print("SKIP_VALIDATION=True - skipping validation\n")

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("=" * 80)
print("FINAL MODELS (Multi-Seed Ensemble)")
print("=" * 80)

models = []
for i, seed in enumerate(ENSEMBLE_SEEDS):
    cache_path = f"gru4rec_seed{seed}.pkl"
    if os.path.exists(cache_path):
        print(f"Loading cached model (seed={seed}): {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                m = pickle.load(f)
            _ = m.iid_map
            print(f"  Model {i+1}/{len(ENSEMBLE_SEEDS)} loaded from cache")
            models.append(m)
            continue
        except Exception as e:
            print(f"  Cache load failed ({e}), retraining...")

    print(f"Training model {i+1}/{len(ENSEMBLE_SEEDS)} (seed={seed})...")
    m = get_fitted_model(df_data, max_row_per_session, seed=seed)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(m, f)
        sz = os.path.getsize(cache_path) / 1024 / 1024
        print(f"  Cached: {cache_path} ({sz:.1f} MB)")
    except Exception as e:
        print(f"  Could not cache: {e}")
    models.append(m)

print(f"\nEnsemble: {len(models)} models ready")

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("\n" + "=" * 80)
print("BUILDING CO-OCCURRENCE SIGNALS")
print("=" * 80)

allowed_set = set(allowed_product_ids)
forward_cooccur = defaultdict(Counter)
backward_cooccur = defaultdict(Counter)
cat_to_products = defaultdict(Counter)

for client_id, group in tqdm(df_data.groupby("client_id"), desc="Co-occurrence"):
    group = group.sort_values("date_time")
    products = []
    cats = []
    for _, row in group.iterrows():
        pt = row.get("page_type", "")
        pid = str(row.get("product_id", ""))
        slug = str(row.get("slug", ""))
        if pt == "CATEGORY" and slug != "nan":
            cats.append(slug)
        elif pt == "PRODUCT" and pid in allowed_set:
            for cat in cats[-3:]:
                cat_to_products[cat][pid] += 1
            for prev in products[-5:]:
                forward_cooccur[prev][pid] += 1
                backward_cooccur[pid][prev] += 1
            products.append(pid)

print(f"Forward co-occurrence:  {len(forward_cooccur)} items")
print(f"Backward co-occurrence: {len(backward_cooccur)} items")
print(f"Category mappings:      {len(cat_to_products)}")

# Pre-extract test visit context
test_ids_set = set(visit_ids)
visit_context = {}
test_subset = df_data[df_data["visit_id"].isin(test_ids_set)]
for vid, group in test_subset.groupby("visit_id"):
    group = group.sort_values("date_time")
    prods = []
    cats = []
    for _, vrow in group.iterrows():
        if vrow["page_type"] == "PRODUCT" and str(vrow["product_id"]) in allowed_set:
            prods.append(str(vrow["product_id"]))
        elif vrow["page_type"] == "CATEGORY" and pd.notna(vrow["slug"]):
            cats.append(str(vrow["slug"]))
    visit_context[vid] = {"products": prods[-5:], "categories": cats[-3:]}

print(f"Visit contexts:      {len(visit_context)} test visits")

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

# Step 1: Get top-100 candidates from each model and average scores
print("Getting ensemble top-100 candidates (averaging across seeds)...")

# Get predictions from each model
all_model_preds = []
for i, m in enumerate(models):
    print(f"  Model {i+1}/{len(models)}...")
    subm = create_submission(
        fitted_model=m,
        hist_data=df_data,
        visit_ids=visit_ids,
        k=100,
        allowed_product_ids=allowed_product_ids,
    )
    all_model_preds.append(subm)

# Merge: for each visit, average scores across models
print("Averaging scores across models...")
merged_preds = {}
for subm in all_model_preds:
    for _, row in subm.iterrows():
        vid = row["visit_id"]
        items = row["product_ids"].split()
        scores = row["scores"]
        if vid not in merged_preds:
            merged_preds[vid] = {}
        for item, score in zip(items, scores):
            if item not in merged_preds[vid]:
                merged_preds[vid][item] = []
            merged_preds[vid][item].append(float(score))

# Build averaged submission
avg_rows = []
for vid, item_scores in merged_preds.items():
    # Average scores (items not in a model get 0)
    averaged = []
    for item, scores_list in item_scores.items():
        avg_score = sum(scores_list) / len(models)  # Divide by total models, not appearances
        averaged.append((item, avg_score))
    averaged.sort(key=lambda x: x[1], reverse=True)
    top_items = [p for p, _ in averaged[:100]]
    top_scores = [s for _, s in averaged[:100]]
    avg_rows.append({
        "visit_id": vid,
        "product_ids": " ".join(top_items),
        "scores": top_scores,
    })

subm_topk = pd.DataFrame(avg_rows)
print(f"Ensemble coverage: {len(subm_topk)} / {len(visit_ids)} visits")

# Step 2: Prediction-level fusion
# Key: Can INJECT co-occurrence items that GRU4Rec missed (not just re-rank)
print("\nPrediction-level fusion (re-rank + inject)...")
predictions = []
for _, row in tqdm(subm_topk.iterrows(), total=len(subm_topk), desc="Fusion"):
    vid = row["visit_id"]
    gru_items = row["product_ids"].split()
    gru_scores = row["scores"]

    ctx = visit_context.get(vid, {"products": [], "categories": []})
    last_prods = ctx["products"]
    last_cats = ctx["categories"]
    n_history = len(last_prods)

    # Session-adaptive boost cap
    if n_history >= 3:
        boost_cap_pct = 0.25
    elif n_history >= 1:
        boost_cap_pct = 0.15
    else:
        boost_cap_pct = 0.08

    # Re-rank GRU4Rec candidates
    reranked = []
    for item, score in zip(gru_items, gru_scores):
        base = float(score)
        boost = 0.0

        for i, prev in enumerate(last_prods):
            recency = 1.0 + 2.0 * (i / max(len(last_prods), 1))
            if prev in forward_cooccur and item in forward_cooccur[prev]:
                boost += min(0.15, 0.012 * forward_cooccur[prev][item]) * recency

        for i, prev in enumerate(last_prods):
            recency = 1.0 + 1.5 * (i / max(len(last_prods), 1))
            if prev in backward_cooccur and item in backward_cooccur[prev]:
                boost += min(0.08, 0.008 * backward_cooccur[prev][item]) * recency

        for j, cat in enumerate(last_cats):
            cat_recency = 1.0 + 1.0 * (j / max(len(last_cats), 1))
            if cat in cat_to_products and item in cat_to_products[cat]:
                boost += min(0.10, 0.008 * cat_to_products[cat][item]) * cat_recency

        max_boost = max(abs(base) * boost_cap_pct, 0.05)
        boost = min(boost, max_boost)
        reranked.append((item, base + boost))

    reranked.sort(key=lambda x: x[1], reverse=True)
    top6 = [p for p, _ in reranked[:n_candidates]]

    predictions.append({"visit_id": vid, "product_ids": " ".join(top6)})

subm_infer = pd.DataFrame(predictions)
print(f"Predictions generated: {len(subm_infer)} visits")

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("\n" + "=" * 80)
print("FINALIZING SUBMISSION")
print("=" * 80)

# Popular items fallback (global)
all_recs = []
for _, row in subm_infer.iterrows():
    all_recs.extend(str(row["product_ids"]).split())
popular_items = [p for p, _ in Counter(all_recs).most_common(n_candidates)]
popular_str = " ".join(popular_items)

# Fill missing visits with category-aware cold start
subm_infer = subm_infer.set_index("visit_id")
test_visit_ids = get_test_visit_ids()
missing = sorted(set(test_visit_ids) - set(subm_infer.index))
if missing:
    print(f"Cold start: {len(missing)} visits")
    cold_rows = {}
    for vid in missing:
        ctx = visit_context.get(vid, {"products": [], "categories": []})
        cats = ctx["categories"]
        prods = ctx["products"]

        # Try category-specific recommendations
        cat_recs = Counter()
        for cat in cats:
            if cat in cat_to_products:
                for pid, cnt in cat_to_products[cat].most_common(20):
                    cat_recs[pid] += cnt
        # Try co-occurrence from viewed products
        for prev in prods:
            if prev in forward_cooccur:
                for pid, cnt in forward_cooccur[prev].most_common(15):
                    cat_recs[pid] += cnt * 2  # stronger signal

        if cat_recs:
            top = [p for p, _ in cat_recs.most_common(n_candidates)]
            # Pad with global popular if needed
            seen = set(top)
            for p in popular_items:
                if len(top) >= n_candidates:
                    break
                if p not in seen:
                    top.append(p)
                    seen.add(p)
            cold_rows[vid] = " ".join(top[:n_candidates])
        else:
            cold_rows[vid] = popular_str

    cat_aware = sum(1 for v in cold_rows.values() if v != popular_str)
    print(f"  Category-aware: {cat_aware} | Global fallback: {len(missing) - cat_aware}")
    missing_df = pd.DataFrame({"product_ids": cold_rows.values()}, index=cold_rows.keys())
    subm_infer = pd.concat([subm_infer, missing_df])
else:
    print("100% coverage")

subm_infer = subm_infer.reset_index()
subm_infer = subm_infer.rename(columns={"index": "visit_id"})

# Pad rows to exactly n_candidates
def pad_row(pid_str, n, fallback):
    items = str(pid_str).split() if pd.notna(pid_str) else []
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    for item in fallback:
        if len(result) >= n:
            break
        if item not in seen:
            result.append(item)
            seen.add(item)
    return " ".join(result[:n])

subm_infer["product_ids"] = subm_infer["product_ids"].apply(
    lambda x: pad_row(x, n_candidates, popular_items)
)

# CRITICAL: match test file visit_id order exactly
test_df = pd.read_csv(f"{root}/metrika_visits_test.csv", usecols=["visit_id"], dtype=str)
subm_infer["visit_id"] = subm_infer["visit_id"].astype(str)
subm_infer = subm_infer.set_index("visit_id").reindex(test_df["visit_id"]).reset_index()

# Fill any NaN rows
nan_count = subm_infer["product_ids"].isna().sum()
if nan_count > 0:
    print(f"Filling {nan_count} NaN rows with popular items")
    subm_infer["product_ids"] = subm_infer["product_ids"].fillna(popular_str)

# Save
subm_infer.to_csv("submission.csv", index=False)

# Summary
n_unique = len(set(" ".join(subm_infer["product_ids"]).split()))
print(f"\nSaved {len(subm_infer)} predictions -> submission.csv")
print(f"Unique products: {n_unique}")
print(f"Cold start: {len(missing)} visits ({100 * len(missing) / len(test_visit_ids):.1f}%)")
print(f"\nGRU4Rec (150 layers, 30 epochs) + Prediction-Level Fusion")
print(f"Expected: ~0.41-0.42 (Kaggle Private)")
print(f"\nTips:")
print(f"  SKIP_VALIDATION=True -> skip validation (~22 min saved)")
print(f"  Delete {MODEL_CACHE} to retrain model")
print(f"  Cached model -> reruns take ~3 min instead of ~25 min")

subm_infer.head(10)
