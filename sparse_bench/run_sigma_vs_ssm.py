"""
Compare SIGMA vs SSM vs GRU4Rec on Rental ultra-short sessions.

Key analysis: per-session-length breakdown (ctx=1, ctx=2, ctx=3+) — which
architecture wins when there's almost no sequential information?
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch, numpy as np
import sigma_model, ssm_model
import baselines as B
import loaders, grouped_eval
from collections import Counter

K_EVAL = [6, 10, 20]


def predict_with_model(models, test_uids, test_queries, n_items, predict_fn, max_seq=50):
    """Generic ensemble predict for any model."""
    for m in models:
        m.eval()
    preds = {}
    for uid in test_uids:
        ctx = [x for x in test_queries[uid]["context"] if 1 <= x < n_items][-max_seq:]
        if not ctx:
            preds[uid] = list(range(1, min(51, n_items)))
            continue
        inp = torch.LongTensor([ctx]).to(models[0].item_embed.weight.device if hasattr(models[0],'item_embed') else DEVICE)
        ln = torch.LongTensor([len(ctx)]).to(inp.device)
        scores = torch.zeros(1, n_items).to(inp.device)
        with torch.no_grad():
            for m in models:
                sc = m(inp, ln) if hasattr(m, 'forward') else m.score_all(inp, ln)
                scores += sc
        sc = scores[0].cpu().numpy()
        sc[0] = -1e9
        for c in set(ctx): sc[c] = -1e9
        preds[uid] = [int(x) for x in np.argsort(-sc) if int(x) != 0][:50]
    return preds


def eval_by_length(name, preds, data):
    """Evaluate and break down by context length."""
    test = data["test_queries"]
    # Group by context length
    groups = {1: [], 2: [], "3+": []}
    for uid in test:
        ctx_len = len([x for x in test[uid]["context"] if 1 <= x < data["n_items"]])
        if ctx_len <= 1:
            groups[1].append(uid)
        elif ctx_len == 2:
            groups[2].append(uid)
        else:
            groups["3+"].append(uid)
    
    results = {}
    for g, uids in groups.items():
        if not uids:
            results[g] = {"n": 0, "R@6": 0, "R@10": 0, "R@20": 0}
            continue
        sub_data = {
            "n_items": data["n_items"],
            "train_sessions": data.get("train_sessions", {}),
            "test_queries": {u: test[u] for u in uids},
            "item_freq": data["item_freq"],
            "item_categories": {},
            "reference_groups": None,
        }
        sub_preds = {u: preds.get(u, []) for u in uids}
        gm = grouped_eval.evaluate_all_groups(sub_preds, sub_data, k_values=K_EVAL)
        o = gm["overall"]
        results[g] = {"n": len(uids), "R@6": o["recall@6"], "R@10": o["recall@10"], "R@20": o["recall@20"]}
    
    # Overall
    gm = grouped_eval.evaluate_all_groups(preds, data, k_values=K_EVAL)
    o = gm["overall"]
    results["all"] = {"n": len(test), "R@6": o["recall@6"], "R@10": o["recall@10"], "R@20": o["recall@20"]}
    return results


def main():
    data = loaders.load_rental_visit()
    n_items = data["n_items"]
    sessions = [s for s in data["train_sessions"].values() if len(s) >= 2]
    test_uids = sorted(data["test_queries"].keys())
    print(f"Rental: {len(sessions)} sessions, {n_items} items, {len(test_uids)} test")
    
    # Session length distribution
    lens = [len([x for x in data["test_queries"][u]["context"] if 1<=x<n_items]) for u in test_uids]
    print(f"Test ctx length: 1-item={sum(1 for l in lens if l<=1)}, 2-item={sum(1 for l in lens if l==2)}, 3+={sum(1 for l in lens if l>=3)}")
    
    all_results = {}
    
    # 1. SIGMA
    print(f"\n[SIGMA] training (4 seeds, 10 ep)...")
    t0 = time.time()
    sigma_models = sigma_model.train_sigma(sessions, n_items, epochs=10, seeds=(42,123,456,789))
    sigma_preds = predict_with_model(sigma_models, test_uids, data["test_queries"], n_items, sigma_model.predict_sigma)
    all_results["SIGMA"] = eval_by_length("SIGMA", sigma_preds, data)
    print(f"  SIGMA trained in {time.time()-t0:.0f}s")
    
    # 2. SSM (our model)
    print(f"\n[SSM] training (4 seeds, 8 ep)...")
    t0 = time.time()
    ssm_models = ssm_model.train_ssm(sessions, n_items, epochs=8, seeds=(42,123,456,789), embed_dim=128)
    ssm_preds = ssm_model.predict_ssm(ssm_models, test_uids, data["test_queries"], n_items)
    all_results["SSM"] = eval_by_length("SSM", ssm_preds, data)
    print(f"  SSM trained in {time.time()-t0:.0f}s")
    
    # 3. GRU4Rec (baseline)
    print(f"\n[GRU4Rec] training (4 seeds, 10 ep)...")
    t0 = time.time()
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    gru_models = []
    for s in [42,123,456,789]:
        torch.manual_seed(s)
        m = B.train_nn_baseline(B.GRU4Rec, data["train_sessions"], n_items, s, epochs=10, emb=128, heads=4, layers=2)
        gru_models.append(m)
    gru_preds = {}
    for uid in test_uids:
        ctx = [x for x in data["test_queries"][uid]["context"] if 1<=x<n_items][-50:]
        if not ctx: continue
        inp = torch.LongTensor([ctx]).to(DEVICE)
        ln = torch.LongTensor([len(ctx)]).to(DEVICE)
        scores = torch.zeros(1, n_items).to(DEVICE)
        with torch.no_grad():
            for m in gru_models:
                scores += m(inp, ln)
        sc = scores[0].cpu().numpy(); sc[0]=-1e9
        for c in set(ctx): sc[c]=-1e9
        gru_preds[uid] = [int(x) for x in np.argsort(-sc) if int(x)!=0][:50]
    all_results["GRU4Rec"] = eval_by_length("GRU4Rec", gru_preds, data)
    print(f"  GRU4Rec trained in {time.time()-t0:.0f}s")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("SIGMA vs SSM vs GRU4Rec — per session length")
    print(f"{'='*80}")
    print(f"{'Model':12} {'ctx=1':>20} {'ctx=2':>20} {'ctx=3+':>20} {'ALL':>20}")
    print(f"{'':12} {'N':>4} {'R@6':>7} {'R@20':>7} {'N':>4} {'R@6':>7} {'R@20':>7} {'N':>4} {'R@6':>7} {'R@20':>7} {'N':>4} {'R@6':>7} {'R@20':>7}")
    print("-" * 92)
    for model_name in ["GRU4Rec", "SIGMA", "SSM"]:
        r = all_results[model_name]
        def fmt(g):
            d = r[g]
            return f"{d['n']:>4} {d['R@6']:.4f} {d['R@20']:.4f}"
        print(f"{model_name:12} {fmt(1)}  {fmt(2)}  {fmt('3+')}  {fmt('all')}")


if __name__ == "__main__":
    main()
