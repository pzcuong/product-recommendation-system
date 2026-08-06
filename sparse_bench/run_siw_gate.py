"""Gate-check: SIW (Global Self-Information Weighted Loss) on Rental + Diginetica.

Tests whether SIW improves tail recall on Rental (dense) and Diginetica (sparse).
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import codt_core, loaders, grouped_eval, srgnn_preprocess as sp

K_EVAL = [6, 10, 20]


def run_single(name, data, k_eval=K_EVAL):
    """Run SIW sweep + baseline on one dataset."""
    n_items = data["n_items"]
    sessions = [s for s in data["train_sessions"].values() if len(s) >= 2]
    print(f"\n{'='*60}\n{name}: {len(sessions)} sessions, {n_items} items\n{'='*60}")

    # Baseline (no SIW)
    t0 = time.time()
    assets_base = codt_core.train_codt_assets(
        data["train_sessions"], n_items, data["test_queries"],
        item_categories=data["item_categories"], item_freq=data["item_freq"],
        ensemble_seeds=[42, 123, 456, 789], embed_dim=128, pgsa_epochs=8, mcl_epochs=8)
    preds = codt_core.predict_codt(assets_base, variant="full")
    gm = grouped_eval.evaluate_all_groups(preds, data, k_values=k_eval)
    o = gm["overall"]
    print(f"  Base:        R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f} "
          f"tgt_mid R@6={gm['tgt_mid']['recall@6']:.4f}")
    base_r20 = o['recall@20']

    # SIW sweep
    for siw in [True]:
        t0 = time.time()
        assets_siw = codt_core.train_codt_assets(
            data["train_sessions"], n_items, data["test_queries"],
            item_categories=data["item_categories"], item_freq=data["item_freq"],
            ensemble_seeds=[42, 123, 456, 789], embed_dim=128, pgsa_epochs=8, mcl_epochs=8,
            siw=True)
        preds = codt_core.predict_codt(assets_siw, variant="full")
        gm = grouped_eval.evaluate_all_groups(preds, data, k_values=k_eval)
        o = gm["overall"]
        delta = o['recall@20'] - base_r20
        print(f"  SIW:         R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f} "
              f"tgt_mid R@6={gm['tgt_mid']['recall@6']:.4f} ΔR@20={delta:+.4f}")
        print(f"  time: {time.time()-t0:.0f}s")


def main():
    # Rental
    d = loaders.load_rental_visit()
    run_single("Rental", d)

    # Diginetica (sparse regime — where SIW should help)
    d2 = sp.load_diginetica()
    # Subsample train for speed
    import random as _r
    keys = list(d2["train_sessions"].keys())
    if len(keys) > 20000:
        keep = set(_r.Random(42).sample(keys, 20000))
        d2["train_sessions"] = {k: v for k, v in d2["train_sessions"].items() if k in keep}
        d2["item_freq"] = Counter()
        for s in d2["train_sessions"].values():
            d2["item_freq"].update(s)
    keys2 = list(d2["test_queries"].keys())
    if len(keys2) > 2000:
        keep2 = set(_r.Random(0).sample(keys2, 2000))
        d2["test_queries"] = {k: v for k, v in d2["test_queries"].items() if k in keep2}
    run_single("Diginetica", d2)


if __name__ == "__main__":
    main()
