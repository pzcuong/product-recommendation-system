"""Quick gate-check: SSM + pop-adaptive fusion vs normal fusion on Rental."""

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

    # Train SSM + M-CL (4 seeds)
    t0 = time.time()
    print("[1] training SSM...")
    ssm_models = SSM.train_ssm(sessions, n_items, epochs=8, seeds=(42, 123, 456, 789))
    for m in ssm_models: m.eval()
    import torch
    print("[2] training M-CL...")
    mcl = codt_core.train_mcl(sessions, n_items, epochs=8)
    mcl.eval()
    with torch.no_grad():
        mcl_emb = mcl.get_id_embeddings(torch.arange(n_items).to(SSM.DEVICE)).cpu().numpy()
    print(f"  models trained in {time.time()-t0:.0f}s")

    # Build co-visitation + tiers
    fwd_cooc, bwd_cooc, pair_freq, pmi_cache = codt_core.build_covisit_pmi(sessions, data["item_freq"])
    cat_to_prods = {}
    pop_penalty = codt_core.build_popularity_penalty(data["item_freq"])
    pop_tiers = codt_core.build_popularity_tiers(data["item_freq"])

    # Compute SSM scores per test visit
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
            for m in ssm_models:
                scores = scores + m.score_all(inp, ln)
        scores = scores.cpu().numpy()
        for uid, sc, ctx in zip(chunk, scores, seqs):
            sc = sc.copy(); sc[0] = -1e9
            for c in set(ctx): sc[c] = -1e9
            top = np.argsort(-sc)[:200]
            ssm_scores[uid] = [(int(i), float(sc[i])) for i in top]

    # Test both fusion variants
    for label, use_pop_tiers in [("normal_fusion", None), ("pop_adaptive_fusion", pop_tiers)]:
        fused = {}
        for uid in test_uids:
            ctx = [x for x in data["test_queries"][uid]["context"] if 1 <= x < n_items]
            last_prods = ctx[-5:]
            last_cats = []
            fused_s = codt_core.fuse_one_query(ssm_scores.get(uid, []), last_prods, last_cats,
                                               pmi_cache, fwd_cooc, cat_to_prods, mcl_emb,
                                               pop_penalty, enable_fusion=True, pop_tiers=use_pop_tiers)
            top = codt_core.mmr_rerank(fused_s, mcl_emb, top_k=200, lambda_div=codt_core.DIVERSITY_LAMBDA)
            fused[uid] = top
        gm = grouped_eval.evaluate_all_groups(fused, data, k_values=K_EVAL)
        o = gm["overall"]
        print(f"  {label:24} R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f} "
              f"tgt_mid R@6={gm['tgt_mid']['recall@6']:.4f} tgt_head R@20={gm['tgt_head']['recall@20']:.4f}")
    print("  REF CoDT: R@6=0.4131 R@10=0.5174 R@20=0.6641 | Kaggle Private=0.43455")


if __name__ == "__main__":
    main()
