"""Gate-check: Selective-SSM (+ CoDT fusion) vs CoDT 0.43 on Rental."""

from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ssm_model as SSM
import codt_core, loaders, grouped_eval
import numpy as np

K_EVAL = [6, 10, 20]


def main():
    data = loaders.load_rental_visit()
    n_items = data["n_items"]
    sessions = [s for s in data["train_sessions"].values() if len(s) >= 2]
    test_uids = sorted(data["test_queries"].keys())
    print(f"Rental: {len(sessions)} sessions, {n_items} items, {len(test_uids)} test")

    # 1. SSM-only baseline (no fusion) — does the SSM backbone itself beat PGSA?
    print("\n[SSM-only]")
    t0 = time.time()
    models = SSM.train_ssm(sessions, n_items, epochs=8, seeds=(42, 123, 456, 789))
    print(f"  trained in {time.time()-t0:.0f}s")
    preds = SSM.predict_ssm(models, test_uids, data["test_queries"], n_items)
    gm = grouped_eval.evaluate_all_groups(preds, data, k_values=K_EVAL)
    o = gm["overall"]
    print(f"  SSM-only: R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f} "
          f"tgt_mid R@6={gm['tgt_mid']['recall@6']:.4f}")

    # 2. SSM + CoDT co-visitation fusion (the real test: does SSM backbone + fusion beat transformer + fusion?)
    print("\n[SSM + co-visitation fusion]")
    # Build the co-visitation graph and M-CL embeddings from the same sessions,
    # then fuse the SSM predictions with co-visitation boosts (same V3.2 fusion logic).
    fwd_cooc, bwd_cooc, pair_freq, pmi_cache = codt_core.build_covisit_pmi(sessions, data["item_freq"])
    cat_to_prods = codt_core.build_category_cooc(sessions, data["item_categories"]) if data["item_categories"] else {}
    pop_penalty = codt_core.build_popularity_penalty(data["item_freq"])
    # M-CL for similarity + MMR
    mcl = codt_core.train_mcl(sessions, n_items, epochs=8)
    mcl.eval()
    import torch
    with torch.no_grad():
        mcl_emb = mcl.get_id_embeddings(torch.arange(n_items).to(SSM.DEVICE)).cpu().numpy()

    # Re-run inference but keep raw scores, then fuse
    ssm_scores = {}
    for bs in range(0, len(test_uids), 128):
        chunk = test_uids[bs:bs + 128]
        seqs, lens = [], []
        for uid in chunk:
            ctx = [x for x in data["test_queries"][uid]["context"] if 1 <= x < n_items][-SSM.MAX_SEQ:]
            seqs.append(ctx); lens.append(len(ctx))
        ml = max(max(lens), 1)
        inp = torch.zeros(len(chunk), ml, dtype=torch.long, device=SSM.DEVICE)
        ln = torch.zeros(len(chunk), dtype=torch.long, device=SSM.DEVICE)
        for i, (s, l) in enumerate(zip(seqs, lens)):
            inp[i, :l] = torch.LongTensor(s); ln[i] = l
        with torch.no_grad():
            scores = torch.zeros(len(chunk), n_items, device=SSM.DEVICE)
            for m in models:
                scores = scores + m.score_all(inp, ln)
        scores = scores.cpu().numpy()
        for uid, sc, ctx in zip(chunk, scores, seqs):
            sc = sc.copy(); sc[0] = -1e9
            for c in set(ctx): sc[c] = -1e9
            # keep top-200 candidates like CoDT's PGSA_TOP_K
            top = np.argsort(-sc)[:200]
            ssm_scores[uid] = [(int(i), float(sc[i])) for i in top]

    fused_preds = {}
    for uid in test_uids:
        ctx = [x for x in data["test_queries"][uid]["context"] if 1 <= x < n_items]
        last_prods = ctx[-5:]
        last_cats = [data["item_categories"].get(x) for x in ctx if data["item_categories"].get(x) is not None][-3:]
        fused = codt_core.fuse_one_query(ssm_scores.get(uid, []), last_prods, last_cats,
                                         pmi_cache, fwd_cooc, cat_to_prods, mcl_emb, pop_penalty,
                                         enable_fusion=True)
        top = codt_core.mmr_rerank(fused, mcl_emb, top_k=200, lambda_div=codt_core.DIVERSITY_LAMBDA)
        fused_preds[uid] = top

    gm = grouped_eval.evaluate_all_groups(fused_preds, data, k_values=K_EVAL)
    o = gm["overall"]
    print(f"  SSM+fusion: R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f} "
          f"tgt_mid R@6={gm['tgt_mid']['recall@6']:.4f}")
    print("\nREF CoDT (transformer+fusion): R@6=0.4131 R@10=0.5174 R@20=0.6641")


if __name__ == "__main__":
    main()
