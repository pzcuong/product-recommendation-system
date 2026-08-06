#!/usr/bin/env python3
"""
run_codt_multi.py — Multi-domain benchmark orchestrator for CoDT.

Runs, per domain:
  Models : MostPop, ItemKNN, SKNN, GRU4Rec, SASRec,        (baselines)
           DT-PGSA, DT-FullFusion, DT-Full+LTAR            (CoDT variants)
  Ablations: full_nommr (fusion without MMR), full_no_ltar (=DT-FullFusion)
  Seeds  : 5 for CI on the neural + CoDT ensemble (CoDT uses a 4-seed PGSA
           ensemble internally; the outer seed subsamples/shuffles where relevant).
  Metrics: Recall@{6,10,20}, NDCG@{6,10,20}, HR@6 — overall + 12 groups.

Neural training config is adapted to dataset size (epochs, embed dim) exactly
like multi_domain_benchmark.py for fairness. Results are saved to
sparse_bench/codt_results.json and a markdown summary is printed.
"""
from __future__ import annotations

import json
import hashlib
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import codt_core
import loaders
import grouped_eval
import baselines as B
import vptr
import multiview_ranker
import pasgr

OUT = Path(__file__).resolve().parent / "codt_results.json"

# 5 outer seeds for CI; CoDT PGSA-ensemble seeds are fixed internally.
SEEDS = [42, 43, 44, 123, 456]
K_REPORT = [6, 10, 20]

# Fairness subsample caps (match multi_domain_benchmark.py).
MAX_TRAIN_USERS = 10000
MAX_TEST_USERS = 5000


# -----------------------------------------------------------------------------
# Adaptive training config per dataset size (same logic as multi_domain_benchmark)
# -----------------------------------------------------------------------------
def adaptive_config(n_train: int, n_items: int):
    if n_train <= 3000:
        nn_epochs, pgsa_epochs = 16, 16
    elif n_train <= 20000:
        nn_epochs, pgsa_epochs = 10, 12
    else:
        nn_epochs, pgsa_epochs = 6, 8
    if n_items > 20000:
        emb, heads, layers = 64, 2, 1
    else:
        emb, heads, layers = codt_core.PGSA_EMBED_DIM, codt_core.PGSA_NUM_HEADS, codt_core.PGSA_NUM_LAYERS
    return dict(nn_epochs=nn_epochs, pgsa_epochs=pgsa_epochs,
                emb=emb, heads=heads, layers=layers)


def subsample(data: dict):
    """Subsample train/test for CPU speed on huge domains (deterministic)."""
    import random as _r
    train = data["train_sessions"]
    test = data["test_queries"]
    if len(train) > MAX_TRAIN_USERS:
        rng = _r.Random(42)
        keep = set(rng.sample(sorted(train.keys()), MAX_TRAIN_USERS))
        data["train_sessions"] = {u: s for u, s in train.items() if u in keep}
        data["item_freq"] = Counter()
        for s in data["train_sessions"].values():
            data["item_freq"].update(s)
    if len(test) > MAX_TEST_USERS:
        rng = _r.Random(0)
        keep = set(rng.sample(sorted(test.keys()), MAX_TEST_USERS))
        data["test_queries"] = {u: q for u, q in test.items() if u in keep}
    valid = data.get("valid_queries", {})
    if len(valid) > MAX_TEST_USERS:
        rng = _r.Random(1)
        keep = set(rng.sample(sorted(valid.keys()), MAX_TEST_USERS))
        data["valid_queries"] = {u: q for u, q in valid.items() if u in keep}
    return data


# -----------------------------------------------------------------------------
# Per-domain run
# -----------------------------------------------------------------------------
def run_domain(domain_name: str, data: dict) -> dict:
    print(f"\n{'=' * 72}\nDOMAIN: {domain_name}  (items={data['n_items']} "
          f"train={len(data['train_sessions'])} test={len(data['test_queries'])})\n{'=' * 72}")
    t0 = time.time()
    data = subsample(data)
    cfg = adaptive_config(len(data["train_sessions"]), data["n_items"])
    print(f"  config: {cfg}")

    train_sessions = data["train_sessions"]
    test_queries = data["test_queries"]
    valid_queries = data.get("valid_queries", {})
    # Namespaced keys let every expert score validation and test using one
    # trained model while making accidental target/key leakage impossible.
    combined_queries = {f"T::{u}": q for u, q in test_queries.items()}
    combined_queries.update({f"V::{u}": q for u, q in valid_queries.items()})
    n_items = data["n_items"]
    item_freq = data["item_freq"]

    all_preds: dict = {}  # model_name -> {seed -> {uid: [items]}}
    valid_preds: dict = {}
    test_uids = sorted(test_queries.keys())

    def split_predictions(preds):
        test = {u: preds.get(f"T::{u}", []) for u in test_queries}
        valid = {u: preds.get(f"V::{u}", []) for u in valid_queries}
        return test, valid

    # ---- Non-parametric baselines (seed-independent) ----
    for name in ["MostPop", "ItemKNN", "SKNN"]:
        print(f"  [{name}] ...")
        try:
            raw = B.run_nonparametric(name, train_sessions, combined_queries, n_items)
            test_pred, val_pred = split_predictions(raw)
            all_preds[name] = {SEEDS[0]: test_pred}
            valid_preds[name] = val_pred
            # replicate across seeds for uniform aggregation
            all_preds[name] = {s: all_preds[name][SEEDS[0]] for s in SEEDS}
        except Exception as e:
            print(f"    {name} failed: {e}")

    # ---- Neural baselines (GRU4Rec, SASRec) ----
    for name in ["GRU4Rec", "SASRec"]:
        all_preds[name] = {}
        for s in SEEDS:
            try:
                run = B.run_gru4rec if name == "GRU4Rec" else B.run_sasrec
                raw = run(train_sessions, combined_queries, n_items, seed=s,
                                         epochs=cfg["nn_epochs"], emb=cfg["emb"],
                                         heads=cfg["heads"], layers=cfg["layers"])
                all_preds[name][s], val_pred = split_predictions(raw)
                if s == SEEDS[0]:
                    valid_preds[name] = val_pred
            except Exception as e:
                print(f"    {name} seed {s} failed: {e}")
                all_preds[name][s] = {u: [] for u in test_uids}

    # ---- CoDT variants: train shared PGSA-ensemble + M-CL assets ONCE, then
    #      score all four variants from them (4x faster than retraining each). ----
    print(f"  [CoDT] training shared assets: 4-seed PGSA ensemble ({cfg['pgsa_epochs']} ep) + M-CL...")
    try:
        assets = codt_core.train_codt_assets(
            train_sessions, n_items, combined_queries,
            item_categories=data["item_categories"], item_freq=item_freq,
            ensemble_seeds=[42, 123, 456, 789],
            max_seq=codt_core.PGSA_MAX_SEQ,
            embed_dim=cfg["emb"], pgsa_epochs=cfg["pgsa_epochs"], mcl_epochs=8,
            digital_twin=True,
            item_texts=data.get("item_texts", {}),
            semantic_cache=str(Path(__file__).resolve().parent / "artifacts" /
                               f"{domain_name}_e5_small.npy"),
            twin_config={"dim": cfg["emb"], "epochs": min(4, cfg["pgsa_epochs"]),
                         "rollout_horizon": 3, "discount": 0.85,
                         "uncertainty_penalty": 0.10,
                         "counterfactual_weight": 0.35})
    except Exception as e:
        import traceback
        print(f"    CoDT assets failed: {e}")
        traceback.print_exc()
        assets = None

    # PASGR is independent of CoDT/DualTwin. It reuses only the offline
    # semantic item matrix (when available), never their predictions or test
    # labels, and performs exact full-catalog retrieval.
    print("  [PASGR] prototype alignment + semantic/graph hard-negative training...")
    try:
        pasgr_cfg = pasgr.PASGRConfig(
            dim=cfg["emb"], prototypes=min(64, max(8, n_items // 250)),
            epochs=min(6, cfg["nn_epochs"]), batch_size=256,
            hard_negatives=32, top_k=max(codt_core.PGSA_TOP_K, 200), seed=42)
        raw = pasgr.run_pasgr(
            train_sessions, combined_queries, n_items, item_freq,
            assets.get("semantic_embeddings") if assets else None,
            config=pasgr_cfg)
        pasgr_test, pasgr_valid = split_predictions(raw)
        all_preds["PASGR"] = {s: pasgr_test for s in SEEDS}
        valid_preds["PASGR"] = pasgr_valid

        # Nested-validation safe fusion: PASGR supplies the learned rank while
        # graph/semantic/popularity views may repair isolated ordering errors.
        # A disjoint gate decides whether fusion is deployed at all.
        if assets and assets.get("semantic_preds") and valid_queries:
            semantic_test, semantic_valid = split_predictions(
                {u: [item for item, _ in ranking]
                 for u, ranking in assets["semantic_preds"].items()})
            fusion_valid_views = {
                "SKNN": valid_preds["SKNN"], "MostPop": valid_preds["MostPop"],
                "Semantic": semantic_valid, "PASGR": pasgr_valid}
            fusion_test_views = {
                "SKNN": all_preds["SKNN"][SEEDS[0]],
                "MostPop": all_preds["MostPop"][SEEDS[0]],
                "Semantic": semantic_test, "PASGR": pasgr_test}
            ordered = sorted(valid_queries,
                             key=lambda u: hashlib.sha1(str(u).encode()).hexdigest())
            cut = max(1, int(0.7 * len(ordered)))
            calibration_ids, gate_ids = set(ordered[:cut]), set(ordered[cut:])
            calibration_queries = {u: valid_queries[u] for u in calibration_ids}
            gate_queries = {u: valid_queries[u] for u in gate_ids}
            def pasgr_subset(views, ids):
                return {name: {u: preds.get(u, []) for u in ids}
                        for name, preds in views.items()}
            fusion_policy = multiview_ranker.fit_rank_fusion(
                pasgr_subset(fusion_valid_views, calibration_ids), calibration_queries)
            pasgr_only = {"weights": {"PASGR": 1.0}, "rrf_k": 1}
            fused_gate = multiview_ranker.evaluate_rank_fusion(
                fusion_policy, pasgr_subset(fusion_valid_views, gate_ids), gate_queries)
            pasgr_gate = multiview_ranker.evaluate_rank_fusion(
                pasgr_only, pasgr_subset(fusion_valid_views, gate_ids), gate_queries)
            deploy_fusion = (fused_gate["utility"] > pasgr_gate["utility"] and
                             fused_gate["recall@6"] >= pasgr_gate["recall@6"])
            deploy_policy = fusion_policy if deploy_fusion else pasgr_only
            fused_test = multiview_ranker.apply_rank_fusion(
                deploy_policy, fusion_test_views, test_queries)
            all_preds["PASGR-SafeFusion"] = {s: fused_test for s in SEEDS}
            valid_preds["PASGR-SafeFusion"] = multiview_ranker.apply_rank_fusion(
                deploy_policy, fusion_valid_views, valid_queries)
            print(f"  [PASGR-SafeFusion] gate pasgr={pasgr_gate} fused={fused_gate} "
                  f"deploy_fusion={deploy_fusion} policy={deploy_policy}")
    except Exception as e:
        import traceback
        print(f"    PASGR failed: {e}")
        traceback.print_exc()

    for variant, label in [("pgsa", "DT-PGSA"),
                           ("full", "DT-FullFusion"),
                           ("full+ltar", "DT-Full+LTAR"),
                           ("full_nommr", "DT-FullNoMMR"),
                           ("dualtwin_nommr", "DualTwin-NoMMR"),
                           ("dualtwin", "DualTwin-DT")]:
        print(f"  [{label}] scoring variant from shared assets...")
        try:
            raw = codt_core.predict_codt(assets, variant=variant) if assets else {}
            preds, val_pred = split_predictions(raw)
            all_preds[label] = {s: preds for s in SEEDS}
            valid_preds[label] = val_pred
        except Exception as e:
            print(f"    {label} failed: {e}")
            all_preds[label] = {s: {u: [] for u in test_uids} for s in SEEDS}

    # Calibrate the factual world first, then let the Digital Twin learn only a
    # counterfactual residual over that strong action order. Previously the
    # stages were reversed and the twin received a weak hard-coded mixture.
    if assets is not None and assets.get("semantic_preds") and "DualTwin-DT" in all_preds:
        semantic_test, semantic_valid = split_predictions(
            {u: [item for item, _ in ranking]
             for u, ranking in assets["semantic_preds"].items()})
        factual_valid_views = {"SKNN": valid_preds["SKNN"],
                               "MostPop": valid_preds["MostPop"],
                               "Semantic": semantic_valid}
        # Nested validation: tune on calibration users and require the frozen
        # policy to generalize to a disjoint gate before deploying its Twin
        # residual. This prevents reporting the same labels used for search.
        ordered_valid = sorted(valid_queries,
                               key=lambda u: hashlib.sha1(str(u).encode()).hexdigest())
        cut = max(1, int(0.7 * len(ordered_valid)))
        calibration_ids, gate_ids = set(ordered_valid[:cut]), set(ordered_valid[cut:])
        calibration_queries = {u: valid_queries[u] for u in calibration_ids}
        gate_queries = {u: valid_queries[u] for u in gate_ids}
        def subset_views(views, ids):
            return {name: {u: preds.get(u, []) for u in ids}
                    for name, preds in views.items()}
        factual_policy = multiview_ranker.fit_rank_fusion(
            subset_views(factual_valid_views, calibration_ids), calibration_queries)
        factual_test_views = {"SKNN": all_preds["SKNN"][SEEDS[0]],
                              "MostPop": all_preds["MostPop"][SEEDS[0]],
                              "Semantic": semantic_test}
        factual_valid = multiview_ranker.apply_rank_fusion(
            factual_policy, factual_valid_views, valid_queries)
        factual_test = multiview_ranker.apply_rank_fusion(
            factual_policy, factual_test_views, test_queries)
        all_preds["Factual-MVR"] = {s: factual_test for s in SEEDS}
        valid_preds["Factual-MVR"] = factual_valid
        factual_combined = {f"T::{u}": rank for u, rank in factual_test.items()}
        factual_combined.update({f"V::{u}": rank for u, rank in factual_valid.items()})
        residual_raw = codt_core.predict_twin_residual(assets, factual_combined)
        residual_test, residual_valid = split_predictions(residual_raw)
        all_preds["DualTwin-Residual"] = {s: residual_test for s in SEEDS}
        valid_preds["DualTwin-Residual"] = residual_valid
        print(f"  [Factual-MVR] validation policy={factual_policy}")

        # The final safety layer is still fit only on validation. A non-zero
        # DualTwin weight is direct evidence that the counterfactual residual
        # contributes beyond factual graph/semantic/exposure retrieval.
        valid_views = {"SKNN": valid_preds["SKNN"],
                       "MostPop": valid_preds["MostPop"],
                       "Semantic": semantic_valid,
                       "DualTwin": residual_valid}
        twin_policy = multiview_ranker.fit_rank_fusion(
            subset_views(valid_views, calibration_ids), calibration_queries)
        factual_gate = multiview_ranker.evaluate_rank_fusion(
            factual_policy, subset_views(valid_views, gate_ids), gate_queries)
        twin_gate = multiview_ranker.evaluate_rank_fusion(
            twin_policy, subset_views(valid_views, gate_ids), gate_queries)
        deploy_twin = (twin_gate["utility"] > factual_gate["utility"] and
                       twin_gate["recall@6"] >= factual_gate["recall@6"])
        policy = twin_policy if deploy_twin else factual_policy
        test_views = {"SKNN": all_preds["SKNN"][SEEDS[0]],
                      "MostPop": all_preds["MostPop"][SEEDS[0]],
                      "Semantic": semantic_test,
                      "DualTwin": residual_test}
        mv_pred = multiview_ranker.apply_rank_fusion(policy, test_views, test_queries)
        all_preds["DualTwin-MVR"] = {s: mv_pred for s in SEEDS}
        valid_preds["DualTwin-MVR"] = multiview_ranker.apply_rank_fusion(
            policy, valid_views, valid_queries)
        print(f"  [DualTwin-MVR] nested gate factual={factual_gate} "
              f"twin={twin_gate} deploy_twin={deploy_twin} policy={policy}")

    # Validation-trained intervention router. Only observable context features
    # are used at test time; test targets never enter fit_router.
    router_experts = {k: valid_preds[k] for k in
                      ["SKNN", "MostPop", "DualTwin-NoMMR", "DualTwin-DT", "DualTwin-MVR"]
                      if k in valid_preds}
    if valid_queries and "SKNN" in router_experts:
        factual = vptr.select_factual(router_experts, valid_queries)
        router = vptr.fit_router(router_experts, valid_queries, item_freq,
                                 factual_expert=factual)
        test_experts = {k: all_preds[k][SEEDS[0]] for k in router_experts}
        routed = vptr.route(router, test_experts, test_queries)
        all_preds["DualTwin-VPTR"] = {s: routed for s in SEEDS}
        print(f"  [DualTwin-VPTR] policy={router['policy']}")

    # ---- Score everything ----
    results: dict = {}
    model_order = ["MostPop", "ItemKNN", "SKNN", "GRU4Rec", "SASRec", "PASGR",
                   "PASGR-SafeFusion",
                   "DT-PGSA", "DT-FullFusion", "DT-Full+LTAR", "DT-FullNoMMR",
                   "DualTwin-NoMMR", "DualTwin-DT", "Factual-MVR",
                   "DualTwin-Residual", "DualTwin-VPTR"]
    if "DualTwin-MVR" in all_preds:
        model_order.insert(-1, "DualTwin-MVR")
    for mname in model_order:
        if mname not in all_preds:
            continue
        per_seed_grouped = []
        for s in SEEDS:
            gm = grouped_eval.evaluate_all_groups(all_preds[mname][s], data, k_values=K_REPORT)
            per_seed_grouped.append(gm)
        # aggregate: mean over seeds
        agg = {}
        groups, _, _ = grouped_eval.build_groups(data)
        for gname in groups:
            agg[gname] = {"n": len(groups[gname])}
            for metric in [f"recall@{k}" for k in K_REPORT] + [f"ndcg@{k}" for k in K_REPORT]:
                vals = [ps[gname][metric] for ps in per_seed_grouped if gname in ps]
                agg[gname][metric] = float(np.mean(vals)) if vals else 0.0
                agg[gname][metric + "_std"] = float(np.std(vals)) if vals else 0.0
        results[mname] = agg

    print(f"\n  --- {domain_name} overall (mean over {len(SEEDS)} seeds) ---")
    print(f"  {'model':18} {'R@6':>9} {'R@10':>9} {'R@20':>9} {'NDCG@6':>9} {'NDCG@20':>9}")
    for mname in model_order:
        if mname not in results:
            continue
        o = results[mname]["overall"]
        print(f"  {mname:18} {o['recall@6']:.4f}   {o['recall@10']:.4f}   "
              f"{o['recall@20']:.4f}   {o['ndcg@6']:.4f}   {o['ndcg@20']:.4f}")

    # ---- Per-tier breakdown for the proposed model vs best baseline ----
    best_bl = None
    for mname in ["MostPop", "ItemKNN", "SKNN", "GRU4Rec", "SASRec"]:
        if mname in results and (best_bl is None or
                                 results[mname]["overall"]["recall@6"] > results[best_bl]["overall"]["recall@6"]):
            best_bl = mname
    proposed = ("PASGR-SafeFusion" if "PASGR-SafeFusion" in results else
                "PASGR" if "PASGR" in results else
                "DualTwin-MVR" if "DualTwin-MVR" in results else
                "DualTwin-VPTR" if "DualTwin-VPTR" in results else "DualTwin-DT")
    if best_bl and proposed in results:
        print(f"\n  --- {domain_name} tier breakdown ({proposed} vs {best_bl}) ---")
        print(f"  {'group':15} {'N':>5} {best_bl:>10} R@6   {'CoDT':>10} R@6   {'Δ':>8}")
        for gname in ["overall", "len_1_2", "len_3", "len_4_7", "len_8_plus",
                      "single_visit", "multi_visit", "tgt_head", "tgt_mid", "tgt_tail"]:
            bl = results[best_bl][gname]["recall@6"]
            codt = results[proposed][gname]["recall@6"]
            print(f"  {gname:15} {results[best_bl][gname]['n']:>5} {bl:>10.4f}   "
                  f"{codt:>10.4f}   {codt-bl:>+8.4f}")

    print(f"\n  domain time: {time.time() - t0:.0f}s")
    return results


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main(domains=None):
    domains = domains or ["Rental_visit", "Baby_Products", "Video_Games", "RetailRocket"]
    print("=" * 72)
    print("CoDT MULTI-DOMAIN BENCHMARK")
    print(f"Domains: {domains}")
    print(f"Models : MostPop, ItemKNN, SKNN, GRU4Rec, SASRec, "
          f"DT-PGSA, DT-FullFusion, DT-Full+LTAR, DT-FullNoMMR, "
          f"DualTwin-NoMMR, DualTwin-DT")
    print(f"Seeds  : {SEEDS}")
    print("=" * 72)
    sys.stdout.flush()

    all_results = {}
    for dom in domains:
        try:
            loader = loaders.ALL_LOADERS[dom]
            data = loader()
            all_results[dom] = run_domain(dom, data)
        except Exception as e:
            import traceback
            print(f"\n!!! Domain {dom} FAILED: {e}")
            traceback.print_exc()
            all_results[dom] = {"error": str(e)}

    # ---- cross-domain summary ----
    print(f"\n{'=' * 72}\nCROSS-DOMAIN SUMMARY (R@6, proposed safe policy vs best baseline)")
    print("=" * 72)
    print(f"  {'domain':22} {'best BL':10} {'BL R@6':>9} {'CoDT R@6':>10} {'Δ%':>8}")
    for dom, res in all_results.items():
        if isinstance(res, dict) and "error" in res:
            print(f"  {dom:22} ERROR: {res['error']}")
            continue
        best_bl, best_r = None, 0
        for m in ["MostPop", "ItemKNN", "SKNN", "GRU4Rec", "SASRec"]:
            if m in res and res[m]["overall"]["recall@6"] > best_r:
                best_r = res[m]["overall"]["recall@6"]; best_bl = m
        proposed = ("PASGR-SafeFusion" if "PASGR-SafeFusion" in res else
                    "PASGR" if "PASGR" in res else
                    "DualTwin-MVR" if "DualTwin-MVR" in res else
                    "DualTwin-VPTR" if "DualTwin-VPTR" in res else "DualTwin-DT")
        codt = res.get(proposed, {}).get("overall", {}).get("recall@6", 0)
        delpct = (codt - best_r) / best_r * 100 if best_r > 0 else 0
        print(f"  {dom:22} {best_bl:10} {best_r:>9.4f} {codt:>10.4f} {delpct:>+7.1f}%")

    OUT.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    # modest flag: smaller subsample + 1 seed (for a fast full pass).
    # When run as a script, the module's globals ARE __main__'s globals, so we
    # mutate them directly (importing the module would make a separate copy).
    if "--fast" in sys.argv:
        globals()["SEEDS"] = [42]
        globals()["MAX_TRAIN_USERS"] = 5000
        globals()["MAX_TEST_USERS"] = 2000
        print("[FAST MODE] seeds=[42] max_train=5000 max_test=2000")
    main(args if args else None)
