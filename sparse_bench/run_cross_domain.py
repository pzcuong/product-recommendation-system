"""
Multi-domain experiment: Does CoDT's co-visitation fusion generalize?

Runs on Diginetica (sparse, large vocab) and RetailRocket (sparse, large vocab).
Tests: DT-PGSA (no fusion) vs DT-Full (with fusion) vs baselines (ItemKNN, SKNN).
Core question: does fusion rescue tail recall across domains?

All datasets subsampled for MPS speed. Full-scale needs GPU.
"""
import sys, time, random as _r
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codt_core, loaders, grouped_eval, baselines as B
import srgnn_preprocess as sp

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

    # Baselines (fast, no training)
    print("  [baselines]")
    for bname in ["MostPop", "ItemKNN", "SKNN"]:
        try:
            preds = B.run_nonparametric(bname, data["train_sessions"], data["test_queries"], n_items)
            gm = grouped_eval.evaluate_all_groups(preds, data, k_values=K_EVAL)
            o = gm["overall"]
            print(f"  {bname:8} R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f} "
                  f"tgt_mid R@6={gm['tgt_mid']['recall@6']:.4f}")
        except Exception as e:
            print(f"  {bname:8} FAILED: {e}")

    # DT-PGSA (no fusion — transformer backbone only)
    print("  [DT-PGSA] training...")
    t0 = time.time()
    assets = codt_core.train_codt_assets(
        data["train_sessions"], n_items, data["test_queries"],
        item_categories=data["item_categories"], item_freq=data["item_freq"],
        ensemble_seeds=[42, 123], embed_dim=64 if n_items > 10000 else 128,
        pgsa_epochs=6, mcl_epochs=6)
    preds = codt_core.predict_codt(assets, variant="pgsa")
    gm = grouped_eval.evaluate_all_groups(preds, data, k_values=K_EVAL)
    o = gm["overall"]
    print(f"  DT-PGSA   R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f} "
          f"tgt_mid R@6={gm['tgt_mid']['recall@6']:.4f}")

    # DT-Full (with fusion)
    print("  [DT-Full] scoring from shared assets...")
    preds2 = codt_core.predict_codt(assets, variant="full")
    gm2 = grouped_eval.evaluate_all_groups(preds2, data, k_values=K_EVAL)
    o2 = gm2["overall"]
    delta = o2['recall@20'] - o['recall@20']
    print(f"  DT-Full   R@6={o2['recall@6']:.4f} R@10={o2['recall@10']:.4f} R@20={o2['recall@20']:.4f} "
          f"tgt_mid R@6={gm2['tgt_mid']['recall@6']:.4f} ΔR@20={delta:+.4f}")
    print(f"  time: {time.time()-t0:.0f}s")
    return {"base": o, "pgsa": o, "full": o2, "delta": delta}


def main():
    results = {}

    # Diginetica
    d2 = sp.load_diginetica()
    results["Diginetica"] = run_domain("Diginetica", d2)

    # RetailRocket
    d3 = sp.load_retailrocket_srgnn()
    results["RetailRocket"] = run_domain("RetailRocket", d3)

    # Summary
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN SUMMARY: Does fusion rescue tail recall?")
    print(f"{'='*60}")
    print(f"{'Domain':15} {'DT-PGSA':>8} {'DT-Full':>8} {'ΔR@20':>8} {'tail_rescued':>12}")
    for dom, res in results.items():
        p = res["pgsa"]["recall@20"]
        f = res["full"]["recall@20"]
        d = res["delta"]
        tail = "YES" if d > 0 else ("NO" if d == 0 else "HURT")
        print(f"  {dom:15} {p:.4f}   {f:.4f}   {d:+.4f}   {tail}")


if __name__ == "__main__":
    main()
