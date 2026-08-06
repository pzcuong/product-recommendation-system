"""
Cross-domain SSM experiment: Does SSM backbone work on Diginetica + RetailRocket?
(Replaces CoDT PGSA backbone with Selective-SSM from ssg_model.py)
"""
import sys, time, random as _r
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ssm_model as SSM
import codt_core, grouped_eval, baselines as B
import srgnn_preprocess as sp
import numpy as np
import torch

K_EVAL = [6, 10, 20]


def subsample(data, max_train=15000, max_test=1500):
    keys = list(data["train_sessions"].keys())
    if len(keys) > max_train:
        keep = set(_r.Random(42).sample(keys, max_train))
        data["train_sessions"] = {k: v for k, v in data["train_sessions"].items() if k in keep}
        data["item_freq"] = Counter()
        for s in data["train_sessions"].values():
            data["item_freq"].update(s)
    keys2 = list(data["test_queries"].keys())
    if len(keys2) > max_test:
        keep2 = set(_r.Random(0).sample(keys2, max_test))
        data["test_queries"] = {k: v for k, v in data["test_queries"].items() if k in keep2}
    return data


def run_domain(name, data):
    data = subsample(data)
    n_items = data["n_items"]
    sessions = [s for s in data["train_sessions"].values() if len(s) >= 2]
    print(f"\n{'='*60}")
    print(f"{name}: {len(sessions)} sessions, {n_items} items, {len(data['test_queries'])} test")
    print(f"{'='*60}")

    # 1. Baselines
    print("  [baselines]")
    for bname in ["MostPop", "ItemKNN", "SKNN"]:
        try:
            preds = B.run_nonparametric(bname, data["train_sessions"], data["test_queries"], n_items)
            gm = grouped_eval.evaluate_all_groups(preds, data, k_values=K_EVAL)
            o = gm["overall"]
            print(f"  {bname:8} R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f}")
        except Exception as e:
            print(f"  {bname:8} FAILED: {e}")

    # 2. Train SSM backbone
    print("  [SSM backbone] training...")
    t0 = time.time()
    embed = 64 if n_items > 10000 else 128
    ssm_models = SSM.train_ssm(sessions, n_items, epochs=6, seeds=(42, 123, 456, 789),
                                embed_dim=embed, n_blocks=2)
    for m in ssm_models:
        m.eval()
    print(f"  SSM trained in {time.time()-t0:.0f}s")

    # 3. Train M-CL for fusion
    mcl = codt_core.train_mcl(sessions, n_items, epochs=6)
    mcl.eval()
    with torch.no_grad():
        mcl_emb = mcl.get_id_embeddings(torch.arange(n_items).to(SSM.DEVICE)).cpu().numpy()

    # 4. Build co-visitation stats
    fwd_cooc, bwd_cooc, pair_freq, pmi_cache = codt_core.build_covisit_pmi(sessions, data["item_freq"])
    pop_penalty = codt_core.build_popularity_penalty(data["item_freq"])
    cat_to_prods = {}

    # 5. Compute SSM scores for all test users
    ssm_scores = {}
    test_uids = sorted(data["test_queries"].keys())
    for bs in range(0, len(test_uids), 128):
        chunk = test_uids[bs:bs + 128]
        seqs, lens = [], []
        for uid in chunk:
            ctx = [x for x in data["test_queries"][uid]["context"] if 1 <= x < n_items][-SSM.MAX_SEQ:]
            seqs.append(ctx)
            lens.append(len(ctx))
        if not seqs:
            continue
        ml = max(max(lens), 1)
        inp = torch.zeros(len(chunk), ml, dtype=torch.long, device=SSM.DEVICE)
        ln = torch.zeros(len(chunk), dtype=torch.long, device=SSM.DEVICE)
        for i, (s, l) in enumerate(zip(seqs, lens)):
            inp[i, :l] = torch.LongTensor(s)
            ln[i] = l
        with torch.no_grad():
            scores = torch.zeros(len(chunk), n_items, device=SSM.DEVICE)
            for m in ssm_models:
                scores = scores + m.score_all(inp, ln)
        scores = scores.cpu().numpy()
        for uid, sc, ctx in zip(chunk, scores, seqs):
            sc = sc.copy()
            sc[0] = -np.inf
            for c in set(ctx):
                sc[c] = -np.inf
            top = np.argpartition(-sc, min(200, int(np.sum(np.isfinite(sc))) - 1))[:200]
            top = top[np.argsort(-sc[top])]
            ssm_scores[uid] = [(int(i), float(sc[i])) for i in top]

    # 6. SSM-only results
    preds_pgsa = {}
    for uid in test_uids:
        sc = ssm_scores.get(uid, [])
        ctx = [x for x in data["test_queries"][uid]["context"] if 1 <= x < n_items]
        sorted_items = [(it, s) for it, s in sc]
        preds_pgsa[uid] = [it for it, _ in sorted_items][:50]
    gm = grouped_eval.evaluate_all_groups(preds_pgsa, data, k_values=K_EVAL)
    o = gm["overall"]
    print(f"  SSM-only  R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f}")

    # 7. SSM + fusion
    pop_tiers = codt_core.build_popularity_tiers(data["item_freq"])
    preds_full = {}
    for uid in test_uids:
        ctx = [x for x in data["test_queries"][uid]["context"] if 1 <= x < n_items]
        last_prods = [item2idx for item2idx in ctx[-5:]]  # already integers
        fused = codt_core.fuse_one_query(
            ssm_scores.get(uid, []), last_prods, [],
            pmi_cache, fwd_cooc, cat_to_prods, mcl_emb, pop_penalty,
            enable_fusion=True, pop_tiers=pop_tiers)
        top = codt_core.mmr_rerank(fused, mcl_emb, top_k=200, lambda_div=codt_core.DIVERSITY_LAMBDA)
        preds_full[uid] = top
    gm2 = grouped_eval.evaluate_all_groups(preds_full, data, k_values=K_EVAL)
    o2 = gm2["overall"]
    delta = o2['recall@20'] - o['recall@20']
    print(f"  SSM+fusion R@6={o2['recall@6']:.4f} R@10={o2['recall@10']:.4f} R@20={o2['recall@20']:.4f} ΔR@20={delta:+.4f}")

    # 8. Best baseline
    print(f"  Best baseline: SKNN (see above)")
    return {"ssm_only": o, "ssm_fusion": o2, "delta": delta}


def main():
    print("=" * 60)
    print("SSM CROSS-DOMAIN: Does SSM backbone help on other domains?")
    print("=" * 60)

    # Diginetica
    d = sp.load_diginetica()
    r_digi = run_domain("Diginetica", d)

    # RetailRocket
    d2 = sp.load_retailrocket_srgnn()
    r_rr = run_domain("RetailRocket", d2)

    print(f"\n{'='*60}")
    print("SUMMARY: SSM backbone cross-domain results")
    print(f"{'='*60}")
    print(f"{'Domain':15} {'SSM-only':>8} {'SSM+fusion':>10} {'ΔR@20':>8}")
    for dom, res in [("Diginetica", r_digi), ("RetailRocket", r_rr)]:
        print(f"  {dom:15} {res['ssm_only']['recall@20']:.4f} {res['ssm_fusion']['recall@20']:.4f} {res['delta']:+.4f}")
    print("  Rental (from earlier): SSM-only R@20=0.695 vs CoDT 0.664")


if __name__ == "__main__":
    main()
