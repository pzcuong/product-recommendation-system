"""Paired bootstrap: SSM vs CoDT on same pipeline (load_rental_visit)."""
import sys, time
sys.path.insert(0, "sparse_bench")
import numpy as np, torch, json, scipy.stats as stats
import loaders, codt_core, ssm_model, grouped_eval

data = loaders.load_rental_visit()
n = data["n_items"]
sessions = [s for s in data["train_sessions"].values() if len(s) >= 2]
uids = sorted(data["test_queries"].keys())
print(f"Rental: {len(sessions)} sessions, {n} items, {len(uids)} test")

def get_predictions(model_fn, name):
    """Train model, get per-query hit vectors at k=[6,10,20]."""
    t0 = time.time()
    models = model_fn()
    for m in models: m.eval()
    print(f"  {name} trained in {time.time()-t0:.0f}s")
    hits = {k: {} for k in [6, 10, 20]}
    for uid in uids:
        ctx = [x for x in data["test_queries"][uid]["context"] if 1 <= x < n][-50:]
        target = data["test_queries"][uid]["targets"][0] if isinstance(data["test_queries"][uid]["targets"], list) else data["test_queries"][uid]["targets"]
        inp = torch.LongTensor([ctx]).to("cpu")
        ln = torch.LongTensor([len(ctx)]).to("cpu")
        # Move to correct device
        DEV = next(m.parameters()).device if hasattr(models[0], "parameters") else torch.device("cpu")
        inp, ln = inp.to(DEV), ln.to(DEV)
        with torch.no_grad():
            logits = []
            for m in models:
                logits.append(m.score_all(inp, ln) if hasattr(m, "score_all") else m(inp, ln))
            avg = sum(logits) / len(logits)
        sc = avg[0].cpu().numpy() if avg.dim() > 1 else avg.cpu().numpy()
        sc[0] = -1e9
        for c in set(ctx): sc[c] = -1e9
        top6 = set(np.argsort(-sc)[:6])
        top10 = set(np.argsort(-sc)[:10])
        top20 = set(np.argsort(-sc)[:20])
        hits[6][uid] = 1 if target in top6 else 0
        hits[10][uid] = 1 if target in top10 else 0
        hits[20][uid] = 1 if target in top20 else 0
    return hits

# CoDT (PGSA + fusion)
print("\n[CoDT]")
assets = codt_core.train_codt_assets(data["train_sessions"], n, data["test_queries"],
    item_categories=data["item_categories"], item_freq=data["item_freq"],
    ensemble_seeds=[42, 123, 456, 789], embed_dim=128, pgsa_epochs=8, mcl_epochs=8)
codt_preds = codt_core.predict_codt(assets, variant="full")
codt_hits = {k: {} for k in [6, 10, 20]}
for uid in uids:
    ctx = [x for x in data["test_queries"][uid]["context"] if 1 <= x < n][-50:]
    target = data["test_queries"][uid]["targets"][0]
    top = codt_preds.get(uid, [])
    codt_hits[6][uid] = 1 if target in top[:6] else 0
    codt_hits[10][uid] = 1 if target in top[:10] else 0
    codt_hits[20][uid] = 1 if target in top[:20] else 0

# SSM
print("\n[SSM]")
s_hits = get_predictions(lambda: ssm_model.train_ssm(sessions, n, epochs=8, seeds=(42,123,456,789), embed_dim=128), "SSM")

# Paired bootstrap
print(f"\n{'='*60}\nPAIRED TEST SSM vs CoDT (N={len(uids)})\n{'='*60}")
results = {}
for k in [6, 10, 20]:
    a = np.array([s_hits[k][u] for u in uids])
    b = np.array([codt_hits[k][u] for u in uids])
    delta = a.mean() - b.mean()
    rng = np.random.RandomState(42)
    deltas = np.array([a[rng.randint(0, len(a), len(a))].mean() - b[rng.randint(0, len(a), len(a))].mean() for _ in range(10000)])
    ci = np.percentile(deltas, [2.5, 97.5])
    d_pos = ((a == 1) & (b == 0)).sum()
    d_neg = ((a == 0) & (b == 1)).sum()
    mc = (abs(d_pos - d_neg) - 1)**2 / max(d_pos + d_neg, 1)
    p = 1 - stats.chi2.cdf(mc, 1)
    sig = "SIGNIFICANT" if ci[0] > 0 and p < 0.05 else "NOT significant"
    print(f"R@{k:2}: SSM={a.mean():.4f} CoDT={b.mean():.4f} Δ={delta:+.4f} 95%CI=[{ci[0]:.4f},{ci[1]:.4f}] SSM>CoDT={d_pos} CoDT>SSM={d_neg} p={p:.4f} {sig}")
    results[f"R@{k}"] = {"ssm": float(a.mean()), "codt": float(b.mean()), "delta": float(delta), "ci_low": float(ci[0]), "ci_high": float(ci[1]), "p": float(p), "sig": sig}

Path("sparse_bench/paired_test.json").write_text(json.dumps(results, indent=2))
