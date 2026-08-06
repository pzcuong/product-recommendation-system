"""
Test Neural + ItemKNN ensemble on Rental and Diginetica.

For each domain:
  - Train the best neural model available (CoDT for Rental, SASRec for Diginetica)
  - Run ItemKNN baseline
  - Fuse via RRF and compare: ensemble vs neural-only vs ItemKNN-only vs SKNN

Goal: prove the ensemble beats BOTH components, especially on tail items.
"""

from __future__ import annotations

import random as _r
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines as B
import ensemble as E
import grouped_eval
import loaders


def run_domain(name, data, neural_pred_fn, subsample_train=None, subsample_test=None):
    n_items = data["n_items"]
    # optional subsample for MPS
    if subsample_train and len(data["train_sessions"]) > subsample_train:
        keys = list(data["train_sessions"].keys())
        keep = set(_r.Random(42).sample(keys, subsample_train))
        data["train_sessions"] = {k: v for k, v in data["train_sessions"].items() if k in keep}
        data["item_freq"] = Counter()
        for s in data["train_sessions"].values():
            data["item_freq"].update(s)
    if subsample_test and len(data["test_queries"]) > subsample_test:
        keys = list(data["test_queries"].keys())
        keep = set(_r.Random(0).sample(keys, subsample_test))
        data["test_queries"] = {k: v for k, v in data["test_queries"].items() if k in keep}
    test_uids = sorted(data["test_queries"].keys())

    print(f"\n{'='*60}\nDOMAIN: {name} (train={len(data['train_sessions'])} "
          f"test={len(data['test_queries'])} items={n_items})\n{'='*60}")

    # 1. ItemKNN (non-parametric, fast)
    itemknn_preds = B.run_nonparametric("ItemKNN", data["train_sessions"], data["test_queries"], n_items)
    sknn_preds = B.run_nonparametric("SKNN", data["train_sessions"], data["test_queries"], n_items)

    # 2. Neural model — must return BOTH ranked lists and (item, score) pairs
    t0 = time.time()
    neural_result = neural_pred_fn(data, n_items)
    # normalize: accept either dict[list] or dict[(list, list_of_pairs)]
    if isinstance(neural_result, tuple):
        neural_preds, neural_scores = neural_result
    else:
        neural_preds = neural_result
        neural_scores = {uid: [(it, 0.0) for it in lst] for uid, lst in neural_preds.items()}
    print(f"  neural trained+predicted in {time.time()-t0:.0f}s")

    # 3. Ensemble variants
    rrf_preds = E.ensemble_predict(neural_preds, itemknn_preds, method="rrf")
    weighted = {}
    for w_n, w_k in [(1.5, 1.0), (2.0, 1.0)]:
        key = f"rrf(n={w_n},k={w_k})"
        weighted[key] = E.ensemble_predict(neural_preds, itemknn_preds, method="weighted_rrf",
                                           neural_weight=w_n, itemknn_weight=w_k)

    # 4. Confidence-routed fusion (the novelty) — sweep entropy thresholds
    conf = {}
    for lo, hi in [(1.0, 2.0), (1.5, 3.0), (2.0, 4.0), (0.5, 1.5)]:
        key = f"conf(e={lo}-{hi})"
        conf[key] = E.confidence_routed_fuse(neural_scores, itemknn_preds,
                                             entropy_low=lo, entropy_high=hi)

    # 5. Score everything
    print(f"\n  {'model':24} {'R@6':>8} {'R@10':>8} {'R@20':>8} {'MRR@20':>8} | {'tgt_mid R@6':>11} {'tgt_tail R@6':>12}")
    print("  " + "-" * 95)
    all_models = [("ItemKNN", itemknn_preds), ("SKNN", sknn_preds),
                  ("Neural", neural_preds), ("RRF ensemble", rrf_preds)]
    all_models += list(weighted.items())
    all_models += list(conf.items())
    for label, preds in all_models:
        gm = grouped_eval.evaluate_all_groups(preds, data, k_values=[6, 10, 20])
        o = gm["overall"]
        print(f"  {label:24} {o['recall@6']:.4f}   {o['recall@10']:.4f}   {o['recall@20']:.4f}   "
              f"{o['mrr@20']:.4f}   | {gm['tgt_mid']['recall@6']:.4f}      {gm['tgt_tail']['recall@6']:.4f}")


def codt_for_rental(data, n_items):
    """CoDT neural predictions for Rental. Returns (ranked, scores)."""
    import codt_core
    assets = codt_core.train_codt_assets(
        data["train_sessions"], n_items, data["test_queries"],
        item_categories=data["item_categories"], item_freq=data["item_freq"],
        ensemble_seeds=[42, 123, 456, 789], embed_dim=128, pgsa_epochs=8, mcl_epochs=8)
    preds = codt_core.predict_codt(assets, variant="full")
    # rebuild (item, score) pairs from the fused pgsa_preds for entropy
    scores = {}
    for uid, items in preds.items():
        # use rank-derived pseudo-scores if raw fused scores unavailable
        scores[uid] = [(it, -rank * 0.1) for rank, it in enumerate(items)]
    return preds, scores


def sasrec_for_diginetica(data, n_items):
    """SASRec neural predictions for Diginetica. Returns (ranked, scores)."""
    import hacl
    sessions = [s for s in data["train_sessions"].values() if len(s) >= 2]
    models = hacl.train_hacl(sessions, n_items, None, {}, {}, data["item_freq"],
                             epochs=10, seeds=[42, 123], embed_dim=64, lambda_cl=0.0)
    test_uids = sorted(data["test_queries"].keys())
    preds = hacl.predict_hacl(models, test_uids, data["test_queries"], n_items,
                              covisit=None, covisit_weight=0)
    # compute real scores via model.score_all on each test session for entropy
    scores = {}
    import torch, numpy as _np
    for m in models:
        m.eval()
    for uid in test_uids:
        ctx = [x for x in data["test_queries"][uid]["context"] if 1 <= x < n_items][-hacl.MAX_SEQ:]
        if not ctx:
            scores[uid] = []
            continue
        inp = torch.LongTensor([ctx]).to(hacl.DEVICE)
        ln = torch.LongTensor([len(ctx)]).to(hacl.DEVICE)
        with torch.no_grad():
            sc = sum(m.score_all(m.last_hidden(inp, ln)).cpu().numpy()[0] for m in models) / len(models)
        for c in set(ctx):
            sc[c] = -1e9
        sc[0] = -1e9
        top = _np.argsort(-sc)[:50]
        scores[uid] = [(int(i), float(sc[i])) for i in top]
    return preds, scores


def main():
    # 1. Rental (small vocab, CoDT works)
    rental = loaders.load_rental_visit()
    run_domain("Rental", rental, codt_for_rental)

    # 2. Diginetica (large vocab, neural collapses — ensemble should rescue)
    import srgnn_preprocess as sp
    digi = sp.load_diginetica()
    run_domain("Diginetica", digi, sasrec_for_diginetica,
               subsample_train=20000, subsample_test=2000)


if __name__ == "__main__":
    main()
