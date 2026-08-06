"""
Generate submission_ssm.csv for Kaggle: Selective-SSM encoder + CoDT fusion.

Replaces the PGSA-Rec transformer in the .bak pipeline with the SSM encoder,
keeps M-CL + co-visitation PMI fusion + session-adaptive boost cap + MMR,
and writes submission_ssm.csv in the exact Kaggle format (visit_id, product_ids).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict

import codt_core
import ssm_model as SSM

REPO = Path(__file__).resolve().parent.parent
RENTAL_DIR = REPO / "rental_data"

K = 6
PGSA_TOP_K = 200
DIVERSITY_LAMBDA = 0.3
EMB_DIM = 128
SSM_EPOCHS = 8
SEEDS = [42, 123, 456, 789]


def load_rental_data():
    """Reuse the .bak loader (loads metrika data, maps slugs->product_ids)."""
    fname = str(REPO / "archive" / "scripts" / "dualtwin_v32_improved.bak.py")
    src = open(fname).read().replace(
        'ROOT = os.path.join(SCRIPT_DIR, "rental_data")',
        'ROOT = os.path.abspath("rental_data")')
    g = {"__file__": os.path.abspath(fname), "__name__": "__main__"}
    exec(compile(src.split("if __name__")[0], fname, "exec"), g)
    return g["load_rental_data"](), g


def main():
    t0 = time.time()
    (df_data, test_vids, allowed_set), bak = load_rental_data()
    print(f"loaded: {len(df_data)} rows, {len(test_vids)} test visits, {len(allowed_set)} allowed items")

    # Vocabulary (product_id strings, 1-indexed, 0=PAD) — same as .bak
    all_items = sorted(df_data["product_id"].unique())
    item_to_idx = {"<PAD>": 0}
    for i, item in enumerate(all_items):
        item_to_idx[item] = i + 1
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    n_items = len(item_to_idx)

    # Build training sessions + co-visitation + M-CL (reuse .bak logic)
    visit_groups = df_data.groupby("visit_id")
    sessions = []
    for vid, grp in visit_groups:
        grp = grp.head(20).sort_values("date_time")
        indices = [item_to_idx[pid] for pid in grp["product_id"] if pid in item_to_idx]
        if len(indices) >= 3:
            sessions.append(indices)
    print(f"train sessions: {len(sessions)}")

    fwd_cooc, bwd_cooc, pair_freq, pmi_cache = codt_core.build_covisit_pmi(sessions, Counter())
    item_freq = Counter()
    for s in sessions:
        item_freq.update(s)
    cat_to_prods = {}
    pop_penalty = codt_core.build_popularity_penalty(item_freq)

    cl_item_list = sorted(allowed_set)
    cl_item_to_idx = {p: i for i, p in enumerate(cl_item_list)}
    n_cl = len(cl_item_list)
    cl_sessions = []
    for vid, grp in visit_groups:
        prods = [cl_item_to_idx[pid] for pid in grp["product_id"] if pid in cl_item_to_idx]
        if len(set(prods)) >= 2:
            cl_sessions.append(prods)
    mcl = bak["train_mcl"](cl_sessions, n_cl, str(REPO / "v3_2_rental_mcl.pkl"))
    if mcl is None:
        # cached: load it explicitly (train_mcl returns None when cache exists)
        mcl = bak["MultimodalCLModel"](n_cl, bak["MCL_EMBED_DIM"]).to(SSM.DEVICE)
        sd = torch.load(str(REPO / "v3_2_rental_mcl.pkl"), map_location=SSM.DEVICE, weights_only=True)
        mcl.load_state_dict(bak["_strip_compile_prefix"](sd))
    mcl.eval()
    with torch.no_grad():
        mcl_emb = mcl.get_id_embeddings(torch.arange(n_cl).to(SSM.DEVICE)).cpu().numpy()

    # Train the SSM ensemble (replaces PGSA) — subsample sessions for SSM speed
    import random as _r
    if len(sessions) > 20000:
        _r.seed(42)
        sessions = _r.sample(sessions, 20000)
        print(f"  subsampled to {len(sessions)} sessions for SSM speed")
    print(f"training SSM ensemble ({len(SEEDS)} seeds × {SSM_EPOCHS} ep)...")
    ssm_models = SSM.train_ssm(sessions, n_items, epochs=SSM_EPOCHS, seeds=tuple(SEEDS),
                                embed_dim=EMB_DIM, n_blocks=2)
    for m in ssm_models:
        m.eval()

    # Per-visit SSM predictions (top-200 candidates with raw scores)
    test_set = set(test_vids)
    test_data = df_data[df_data["visit_id"].isin(test_set)].groupby("visit_id", sort=False).agg(
        historical_items=("product_id", lambda x: x.tolist())).reset_index()

    visit_ctx = {}
    for vid, grp in df_data[df_data["visit_id"].isin(test_set)].groupby("visit_id"):
        grp = grp.sort_values("date_time")
        prods = []
        for _, r in grp.iterrows():
            if r["page_type"] == "PRODUCT" and str(r["product_id"]) in allowed_set:
                prods.append(str(r["product_id"]))
        visit_ctx[vid] = prods

    # SSM scores per visit (integer index space — keep int for fusion, convert at end)
    ssm_scores: dict = {}
    test_uids_ordered = []
    for vid_row in test_data.itertuples():
        vid = vid_row.visit_id
        hist = vid_row.historical_items[-SSM.MAX_SEQ:]
        indices = [item_to_idx[p] for p in hist if p in item_to_idx]
        if indices:
            test_uids_ordered.append((vid, indices))

    BATCH = 64
    for bs in range(0, len(test_uids_ordered), BATCH):
        batch = test_uids_ordered[bs:bs + BATCH]
        ml = max(len(idx) for _, idx in batch)
        seq_t = torch.zeros(len(batch), ml, dtype=torch.long, device=SSM.DEVICE)
        len_t = torch.zeros(len(batch), dtype=torch.long, device=SSM.DEVICE)
        for i, (vid, idx) in enumerate(batch):
            seq_t[i, :len(idx)] = torch.LongTensor(idx)
            len_t[i] = len(idx)
        with torch.no_grad():
            avg = torch.zeros(len(batch), n_items, device=SSM.DEVICE)
            for m in ssm_models:
                avg = avg + m.score_all(seq_t, len_t)
        avg = (avg / len(ssm_models)).cpu().numpy()
        for i, (vid, idx) in enumerate(batch):
            sc = avg[i].copy()
            sc[0] = -np.inf
            for j in idx:
                sc[j] = -np.inf
            for j in range(n_items):
                it = idx_to_item.get(j, "")
                if it not in allowed_set:
                    sc[j] = -np.inf
            top = np.argpartition(-sc, min(PGSA_TOP_K, int(np.sum(np.isfinite(sc))) - 1))[:PGSA_TOP_K]
            top = top[np.argsort(-sc[top])]
            ssm_scores[vid] = [(int(j), float(sc[j])) for j in top if sc[j] > -1e8]

    # Fusion + MMR per visit (all in integer index space)
    pop_tiers = codt_core.build_popularity_tiers(item_freq)
    print("fusion + MMR...")
    predictions = []
    for vid in test_vids:
        ctx = visit_ctx.get(vid, [])
        last_prods_int = [item_to_idx[p] for p in ctx[-5:] if p in item_to_idx]
        fused = codt_core.fuse_one_query(
            ssm_scores.get(vid, []), last_prods_int, [],
            pmi_cache, fwd_cooc, cat_to_prods, mcl_emb, pop_penalty, enable_fusion=True,
            pop_tiers=pop_tiers)
        top_ints = codt_core.mmr_rerank(fused, mcl_emb, top_k=K * 3, lambda_div=DIVERSITY_LAMBDA)[:K]
        # Convert integer indices back to product_id strings
        top_strs = [idx_to_item.get(p, str(p)) for p in top_ints]
        predictions.append({"visit_id": vid, "product_ids": " ".join(top_strs) if top_strs else ""})

    df_subm = pd.DataFrame(predictions)
    # pad/dedupe to K with global popular items
    all_recs = []
    for _, r in df_subm.iterrows():
        if r["product_ids"]:
            all_recs.extend(r["product_ids"].split())
    popular = [p for p, _ in Counter(all_recs).most_common(K)]

    def pad(s, n, fb):
        items = str(s).split() if pd.notna(s) and s else []
        seen, res = set(), []
        for x in items:
            if x not in seen:
                res.append(x); seen.add(x)
        for x in fb:
            if len(res) >= n:
                break
            if x not in seen:
                res.append(x); seen.add(x)
        while len(res) < n:
            res.append("463480210")
        return " ".join(res[:n])

    df_subm["product_ids"] = df_subm["product_ids"].apply(lambda x: pad(x, K, popular))
    test_df = pd.read_csv(RENTAL_DIR / "metrika_visits_test.csv", usecols=["visit_id"], dtype=str)
    df_subm["visit_id"] = df_subm["visit_id"].astype(str)
    df_subm = df_subm.set_index("visit_id").reindex(test_df["visit_id"]).reset_index()
    df_subm["product_ids"] = df_subm["product_ids"].fillna(" ".join(popular))
    df_subm.to_csv("submission_ssm.csv", index=False)

    n_unique = len(set(" ".join(df_subm["product_ids"]).split()))
    print(f"\nSUBMISSION SAVED: submission_ssm.csv")
    print(f"  Rows: {len(df_subm)} | Unique items: {n_unique}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print(f"  Method: Selective-SSM + CoDT fusion (SSM replaces PGSA-Rec transformer)")


if __name__ == "__main__":
    main()
