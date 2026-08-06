#!/usr/bin/env python3
"""Pilot: dynamic β CEARF-N on Amazon domains.

Key novelty shift: β is LEARNED from training prefixes, not swept on
validation. Validation only evaluates the complete model.

Pipeline:
1. Build memory index + PASGR (reuse cached artifacts where possible)
2. Extract per-query features from memory/neural rankings
3. Fit dynamic β gate on TRAINING prefixes (hold-out last item as target)
4. Predict β for test queries
5. Evaluate fused ranking

This is a proof-of-concept pilot; full multi-seed runs come later.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import cearf
import loaders
import pasgr
from dynamic_beta import (
    TrainOnlyDynamicBeta, feature_matrix, fuse_with_dynamic_beta)
from perquery_router import extract_features
from run_cearfn import fuse, tune_beta
from run_cearfn_evidence import (
    load_or_build_memory, metrics_from_ranks, query_fingerprint,
    ranks_at_20, targets_for, popularity_partition)
from run_pasgr_full import semantic_matrix

HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products")
SEEDS = (42, 123, 456)
EPOCHS = 4
CANDIDATE_WIDTH = 120


def build_calibration_data(data, index, profiles, neural_model, n_items,
                           item_freq, head_items, max_cal=5000):
    """Build features, memory, neural, targets from training prefixes.

    Uses trained PASGR to get neural rankings for each prefix.
    Subsamples to max_cal for speed.
    """
    sessions = data["train_sessions"]
    # Subsample for speed
    all_uids = list(sessions.keys())
    rng = np.random.default_rng(42)
    sample_uids = rng.choice(all_uids, size=min(max_cal, len(all_uids)),
                             replace=False)

    calibration_features = []
    calibration_memory = []
    calibration_neural = []
    calibration_targets = []

    for uid in sample_uids:
        seq = sessions[uid]
        if len(seq) < 3:
            continue
        prefix = seq[:-1]
        target = seq[-1]

        # Memory ranking
        comps = index.component_rankings(prefix)
        regime = "short" if len(prefix) <= index.config.short_context else "long"
        mem_rank = list(index.fuse_rankings(prefix, comps, profiles[regime], 120))

        # Neural ranking from PASGR
        prefix_query = {str(uid): {"context": list(prefix), "targets": [target]}}
        _, neural_rank = pasgr.predict_pasgr_array(
            neural_model, prefix_query, n_items, 120)
        neu_rank = list(neural_rank[0])

        # Features
        last = prefix[-1] if prefix else 0
        feat = np.zeros(14, dtype=np.float32)
        feat[0] = np.log1p(len(prefix))
        feat[1] = np.log1p(item_freq.get(last, 0))
        feat[2] = float(last not in head_items)
        mem5 = set(int(x) for x in mem_rank[:5] if int(x) > 0)
        neu5 = set(int(x) for x in neu_rank[:5] if int(x) > 0)
        mem20 = set(int(x) for x in mem_rank[:20] if int(x) > 0)
        neu20 = set(int(x) for x in neu_rank[:20] if int(x) > 0)
        union5 = mem5 | neu5
        union20 = mem20 | neu20
        feat[3] = len(mem5 & neu5) / max(len(union5), 1)
        feat[4] = len(mem20 & neu20) / max(len(union20), 1)
        feat[5] = float(mem_rank[0] == neu_rank[0]) if mem_rank and neu_rank else 0.0

        calibration_features.append(feat)
        calibration_memory.append(np.array(mem_rank[:120], dtype=np.int32))
        calibration_neural.append(np.array(neu_rank[:120], dtype=np.int32))
        calibration_targets.append(target)

    return (np.array(calibration_features, dtype=np.float32),
            np.array(calibration_memory, dtype=np.int32),
            np.array(calibration_neural, dtype=np.int32),
            np.array(calibration_targets, dtype=np.int32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--candidate-width", type=int, default=CANDIDATE_WIDTH)
    parser.add_argument("--output", type=Path,
                        default=HERE / "dynamic_beta_results.json")
    args = parser.parse_args()

    results = {}
    for domain in args.domains:
        if domain in results:
            print(f"[DYN-BETA] {domain} already complete", flush=True)
            continue

        print(f"\n[DYN-BETA] === {domain} ===", flush=True)
        data = loaders.ALL_LOADERS[domain]()
        if len(data["valid_queries"]) > 5000:
            _vk = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
            data["valid_queries"] = {k: data["valid_queries"][k] for k in _vk}

        sessions = data["train_sessions"]
        n_items = data["n_items"]
        freq = Counter(x for seq in sessions.values() for x in seq)
        config = cearf.CEARFConfig()
        index = cearf.CEARFIndex(sessions, n_items, config)
        profiles, _ = cearf.tune_profiles(index, data["valid_queries"])

        # Load cached memory arrays
        evidence_dir = HERE / "cearfn_evidence_artifacts"
        v2_dir = HERE / "cearfn_v2_artifacts"
        for d in [v2_dir, evidence_dir]:
            vp = d / f"{domain.lower()}_full_valid_memory.npz"
            tp = d / f"{domain.lower()}_full_test_memory.npz"
            if vp.exists() and tp.exists():
                valid_path, test_path = vp, tp
                break
        else:
            raise FileNotFoundError(f"No memory cache for {domain}")

        with np.load(valid_path) as sv:
            valid_memory = {k: sv[k] for k in sv.files}
        with np.load(test_path) as st:
            test_memory = {k: st[k] for k in st.files}

        valid_keys = [str(x) for x in valid_memory["keys"]]
        test_keys = [str(x) for x in test_memory["keys"]]
        test_targets = targets_for(test_keys, data["test_queries"])

        # Build calibration data from training prefixes
        head_items, _, _ = popularity_partition(freq, n_items)
        head_set = set(head_items.tolist())

        # Train PASGR first (needed for calibration neural rankings)
        print("[DYN-BETA] Training PASGR...", flush=True)
        semantic = semantic_matrix(domain, data)
        pcfg = pasgr.PASGRConfig(dim=64, prototypes=min(96, max(8, n_items // 250)),
                                 epochs=args.epochs, batch_size=512, hard_negatives=32,
                                 top_k=120, seed=42)
        assets = pasgr.build_prototype_graph_embeddings(sessions, n_items, freq, semantic, pcfg)
        model = pasgr.train_pasgr(sessions, n_items, freq, semantic, pcfg, prepared_assets=assets)

        # Build calibration data with neural model
        print("[DYN-BETA] Building calibration data from training prefixes...",
              flush=True)
        cal_feat, cal_mem, cal_neu, cal_tgt = build_calibration_data(
            data, index, profiles, model, n_items, freq, head_set)

        # Fit dynamic β gate
        print(f"[DYN-BETA] Fitting gate on {len(cal_feat)} calibration queries...",
              flush=True)
        gate = TrainOnlyDynamicBeta(epochs=160, hidden=16, seed=42)
        fit_report = gate.fit(cal_feat, cal_mem, cal_neu, cal_tgt)
        print(f"[DYN-BETA] Fit report: {json.dumps({k: v for k, v in fit_report.items() if k != 'feature_names'}, indent=2)}",
              flush=True)

        # Extract features for test queries using the same trained model
        _, neural_test = pasgr.predict_pasgr_array(
            model, data["test_queries"], n_items, args.candidate_width)

        # Build features for test
        test_feat = np.zeros((len(test_keys), 14), dtype=np.float32)
        for i, uid in enumerate(test_keys):
            ctx = data["test_queries"][uid].get("context", ())
            last = ctx[-1] if ctx else 0
            test_feat[i, 0] = np.log1p(len(ctx))
            test_feat[i, 1] = np.log1p(freq.get(last, 0))
            test_feat[i, 2] = float(last not in head_set)

        # Predict β for test
        test_betas = gate.predict(test_feat)
        print(f"[DYN-BETA] Test β stats: mean={test_betas.mean():.3f} std={test_betas.std():.3f} "
              f"min={test_betas.min():.3f} max={test_betas.max():.3f}", flush=True)

        # Fuse with dynamic β
        fused_dynamic = fuse_with_dynamic_beta(
            test_memory["selected"], neural_test, test_betas)

        # Compute neural predictions for validation (for regime baseline)
        _, neural_valid = pasgr.predict_pasgr_array(
            model, data["valid_queries"], n_items, args.candidate_width)

        # Also compute regime baseline for comparison
        betas_regime, _ = tune_beta(
            {uid: list(valid_memory["selected"][i]) for i, uid in enumerate(valid_keys)},
            {uid: list(neural_valid[i]) for i, uid in enumerate(valid_keys)},
            data["valid_queries"], config.short_context)
        regime_per_uid = {uid: betas_regime["long"] for uid in test_keys}
        fused_regime = np.zeros((len(test_keys), 20), dtype=np.int32)
        for row, uid in enumerate(test_keys):
            fused_regime[row] = fuse(test_memory["selected"][row],
                                     neural_test[row], regime_per_uid[uid])

        # Evaluate
        dyn_metrics = metrics_from_ranks(ranks_at_20(fused_dynamic, test_targets))
        regime_metrics = metrics_from_ranks(ranks_at_20(fused_regime, test_targets))
        mem_metrics = metrics_from_ranks(ranks_at_20(test_memory["selected"], test_targets))

        print(f"[DYN-BETA] {domain} results:")
        print(f"  Dynamic β:   R@20={dyn_metrics['recall@20']:.5f}")
        print(f"  Regime β:    R@20={regime_metrics['recall@20']:.5f}")
        print(f"  Memory-only: R@20={mem_metrics['recall@20']:.5f}")

        results[domain] = {
            "fit_report": fit_report,
            "dynamic_beta": dyn_metrics,
            "regime_beta": regime_metrics,
            "memory_only": mem_metrics,
            "beta_distribution": {
                "mean": float(test_betas.mean()),
                "std": float(test_betas.std()),
                "min": float(test_betas.min()),
                "max": float(test_betas.max()),
            },
        }
        args.output.write_text(json.dumps(results, indent=2))
        print(f"[DYN-BETA] Saved {args.output}", flush=True)
        del model, neural_test
        gc.collect()

    print(f"\n[DYN-BETA] All done.", flush=True)


if __name__ == "__main__":
    main()
