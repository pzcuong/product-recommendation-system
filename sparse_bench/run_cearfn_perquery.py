#!/usr/bin/env python3
"""End-to-end per-query adaptive β runner.

Reproduces the CEARF-N evidence protocol but swaps the coarse regime β for
three router variants and reports their test-time Recall@20 side by side:

  * ``regime``     — baseline, one β per (short, long) regime.
  * ``bucketed``   — one β per (length × popularity × agreement) bucket.
  * ``continuous`` — ridge regression from query features to β.

Selection uses ONLY validation queries. Test labels are touched exactly once,
to compute the final ranking metrics after the β has been chosen.
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
from perquery_router import (
    BucketedRouter, ContinuousRouter, extract_features, feature_vector,
    QueryFeatures)
from run_cearfn import BETAS, fuse, tune_beta
from run_cearfn_evidence import (
    load_or_build_memory, metrics_from_ranks, popularity_partition,
    query_fingerprint, ranks_at_20, targets_for)
from run_pasgr_full import semantic_matrix

HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID", "Tmall")
SEED = 42
EPOCHS = 4


def build_features_for(index: cearf.CEARFIndex, queries: dict, freq: Counter,
                       head_items: set[int]) -> dict[str, QueryFeatures]:
    out: dict[str, QueryFeatures] = {}
    for uid, q in queries.items():
        context = q.get("context", ())
        comps = index.component_rankings(context)
        out[str(uid)] = extract_features(context, comps, freq, head_items)
    return out


def fuse_with_betas(memory, neural, keys, betas_per_uid: dict[str, float]):
    output = np.empty((len(keys), 20), dtype=np.int32)
    for row, uid in enumerate(keys):
        output[row] = fuse(memory[row], neural[row], betas_per_uid[uid])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--candidate-width", type=int, default=120)
    parser.add_argument("--output", type=Path,
                        default=HERE / "cearfn_perquery_results.json")
    args = parser.parse_args()

    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    for domain in args.domains:
        if domain in results:
            print(f"[PQ] {domain} already complete", flush=True)
            continue
        print(f"\n[PQ] === {domain} ===", flush=True)
        started = time.time()
        data = loaders.ALL_LOADERS[domain]()
        # Match the evidence protocol's deterministic validation cap so the
        # locked Amazon memory cache can be reused (fingerprint is cap-sensitive).
        if len(data["valid_queries"]) > 5000:
            _vk = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
            data["valid_queries"] = {k: data["valid_queries"][k] for k in _vk}
        sessions = data["train_sessions"]
        freq = Counter(x for seq in sessions.values() for x in seq)
        config = cearf.CEARFConfig()
        index = cearf.CEARFIndex(sessions, data["n_items"], config)
        profiles, _ = cearf.tune_profiles(index, data["valid_queries"])

        artifact_dir = HERE / "perquery_artifacts"
        # Reuse the locked evidence cache for Amazon domains to avoid a costly
        # rebuild; the component arrays are identical across runners because
        # the underlying CEARFIndex + profiles are deterministic.
        evidence_dir = HERE / "cearfn_evidence_artifacts"
        valid_path = evidence_dir / f"{domain.lower()}_full_valid_memory.npz"
        test_path = evidence_dir / f"{domain.lower()}_full_test_memory.npz"
        valid_fp = query_fingerprint(data["valid_queries"])
        test_fp = query_fingerprint(data["test_queries"])
        if valid_path.exists() and test_path.exists():
            import numpy as _np
            with _np.load(valid_path) as sv:
                if str(sv["fingerprint"].item()) != valid_fp:
                    raise RuntimeError(f"{domain}: valid cache fingerprint mismatch")
                valid_memory = {k: sv[k] for k in sv.files}
            with _np.load(test_path) as st:
                if str(st["fingerprint"].item()) != test_fp:
                    raise RuntimeError(f"{domain}: test cache fingerprint mismatch")
                test_memory = {k: st[k] for k in st.files}
            print(f"[PQ] {domain} reused evidence cache", flush=True)
        else:
            valid_memory = load_or_build_memory(
                artifact_dir / f"{domain.lower()}_valid_memory.npz", index,
                data["valid_queries"], profiles, args.candidate_width, f"{domain}-valid")
            test_memory = load_or_build_memory(
                artifact_dir / f"{domain.lower()}_test_memory.npz", index,
                data["test_queries"], profiles, args.candidate_width, f"{domain}-test")
        valid_keys = [str(x) for x in valid_memory["keys"]]
        test_keys = [str(x) for x in test_memory["keys"]]

        head_items, tail_items, _ = popularity_partition(freq, data["n_items"])
        valid_features = build_features_for(
            index, data["valid_queries"], freq, set(head_items.tolist()))
        test_features = build_features_for(
            index, data["test_queries"], freq, set(head_items.tolist()))

        semantic = semantic_matrix(domain, data)
        pcfg = pasgr.PASGRConfig(
            dim=64, prototypes=min(96, max(8, data["n_items"] // 250)),
            epochs=args.epochs, batch_size=512, hard_negatives=32,
            top_k=120, seed=args.seed)
        assets = pasgr.build_prototype_graph_embeddings(
            sessions, data["n_items"], freq, semantic, pcfg)
        model = pasgr.train_pasgr(sessions, data["n_items"], freq, semantic,
                                  pcfg, prepared_assets=assets)
        _, neural_valid = pasgr.predict_pasgr_array(
            model, data["valid_queries"], data["n_items"], args.candidate_width)
        _, neural_test = pasgr.predict_pasgr_array(
            model, data["test_queries"], data["n_items"], args.candidate_width)
        del model
        gc.collect()

        # Build uid-indexed views for the routers (the evidence cache is row
        # ordered; we keep that order in memory_dict/neural_dict).
        valid_memory_dict = {uid: list(valid_memory["selected"][i])
                             for i, uid in enumerate(valid_keys)}
        valid_neural_dict = {uid: list(neural_valid[i]) for i, uid in enumerate(valid_keys)}
        test_targets = targets_for(test_keys, data["test_queries"])

        # 1) Regime baseline β.
        betas_regime, _ = tune_beta(valid_memory_dict, valid_neural_dict,
                                    data["valid_queries"], config.short_context)
        regime_per_uid = {
            uid: betas_regime["short" if f.length <= config.short_context else "long"]
            for uid, f in test_features.items()
        }
        fused_regime = fuse_with_betas(
            test_memory["selected"], neural_test, test_keys, regime_per_uid)

        # 2) Bucketed router.
        # Use the regime β as fallback so empty buckets inherit the safe choice.
        bucketed = BucketedRouter(fallback_beta=betas_regime)
        bucketed.fit({uid: data["valid_queries"][uid] for uid in valid_keys},
                     valid_memory_dict, valid_neural_dict,
                     valid_keys, valid_features)
        bucketed_per_uid = {uid: bucketed.beta_for(test_features[uid]) for uid in test_keys}
        fused_bucketed = fuse_with_betas(
            test_memory["selected"], neural_test, test_keys, bucketed_per_uid)

        # 3) Continuous router.
        continuous = ContinuousRouter(alpha=1.0)
        fit_report = continuous.fit(
            {uid: data["valid_queries"][uid] for uid in valid_keys},
            valid_memory_dict, valid_neural_dict,
            valid_keys, valid_features)
        continuous_per_uid = {uid: continuous.beta_for(test_features[uid]) for uid in test_keys}
        fused_continuous = fuse_with_betas(
            test_memory["selected"], neural_test, test_keys, continuous_per_uid)

        results[domain] = {
            "protocol": {
                "seed": args.seed, "epochs": args.epochs,
                "validation_queries": len(valid_keys),
                "test_queries": len(test_keys),
                "selection_uses_test_labels": False,
            },
            "regime": {
                "betas": betas_regime,
                "test": metrics_from_ranks(ranks_at_20(fused_regime, test_targets)),
            },
            "bucketed": {
                "report": {f"{k.length_bucket}_{k.pop_bucket}_{k.agreement_bucket}": v
                           for k, v in bucketed.report_per_bucket.items()},
                "test": metrics_from_ranks(ranks_at_20(fused_bucketed, test_targets)),
            },
            "continuous": {
                "fit": fit_report,
                "test": metrics_from_ranks(ranks_at_20(fused_continuous, test_targets)),
            },
            "seconds": time.time() - started,
        }
        args.output.write_text(json.dumps(results, indent=2))
        print(f"[PQ] {domain} regime R@20="
              f"{results[domain]['regime']['test']['recall@20']:.5f} | "
              f"bucketed={results[domain]['bucketed']['test']['recall@20']:.5f} | "
              f"continuous={results[domain]['continuous']['test']['recall@20']:.5f}",
              flush=True)

    print(f"\n[PQ] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
