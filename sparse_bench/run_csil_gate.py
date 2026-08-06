"""Gate-check: CSIL (Conditional Self-Information Loss) on Rental.

Tests whether CSIL-trained PGSA + CoDT fusion beats the standard CoDT 0.43.
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import codt_core, loaders, grouped_eval

K_EVAL = [6, 10, 20]


def main():
    data = loaders.load_rental_visit()
    n_items = data["n_items"]
    sessions = [s for s in data["train_sessions"].values() if len(s) >= 2]
    print(f"Rental: {len(sessions)} sessions, {n_items} items")

    # Build co-visitation once
    fwd_cooc, bwd_cooc, pair_freq, pmi_cache = codt_core.build_covisit_pmi(sessions, data["item_freq"])

    # Test alpha/beta sweep for CSIL
    for alpha, beta in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3), (0.5, 1.0)]:
        print(f"\n[CSIL α={alpha} β={beta}]")
        t0 = time.time()
        assets = codt_core.train_codt_assets(
            data["train_sessions"], n_items, data["test_queries"],
            item_categories=data["item_categories"], item_freq=data["item_freq"],
            ensemble_seeds=[42, 123, 456, 789], embed_dim=128, pgsa_epochs=8, mcl_epochs=8,
            csil=True, csil_alpha=alpha, csil_beta=beta)
        preds = codt_core.predict_codt(assets, variant="full")
        gm = grouped_eval.evaluate_all_groups(preds, data, k_values=K_EVAL)
        o = gm["overall"]
        print(f"  CSIL(α={alpha},β={beta}): R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f} "
              f"tgt_mid R@6={gm['tgt_mid']['recall@6']:.4f}")
        print(f"  time: {time.time()-t0:.0f}s")

    # Baseline for comparison
    print(f"\n[CoDT baseline (no CSIL)]")
    t0 = time.time()
    assets = codt_core.train_codt_assets(
        data["train_sessions"], n_items, data["test_queries"],
        item_categories=data["item_categories"], item_freq=data["item_freq"],
        ensemble_seeds=[42, 123, 456, 789], embed_dim=128, pgsa_epochs=8, mcl_epochs=8)
    preds = codt_core.predict_codt(assets, variant="full")
    gm = grouped_eval.evaluate_all_groups(preds, data, k_values=K_EVAL)
    o = gm["overall"]
    print(f"  CoDT: R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f} "
          f"tgt_mid R@6={gm['tgt_mid']['recall@6']:.4f}")
    print(f"  Kaggle Private=0.43455 | time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
