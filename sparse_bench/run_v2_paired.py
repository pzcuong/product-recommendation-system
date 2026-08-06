#!/usr/bin/env python3
"""Paired bootstrap: CEARF-N v2 vs baselines on Amazon (using v2 predictions).

Uses cearfn_v2_artifacts/ for CEARF-N v2 regime ranks, and
paper_baseline_artifacts/ for baseline ranks. Both use the same test set
(fingerprint-matched).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from paired_statistics import cluster_paired_recall

HERE = Path(__file__).resolve().parent
SEEDS = (42, 123, 456)
K = 20
REPS = 20000
BASELINES = ("narm", "sr_gnn", "gru4rec", "sasrec", "sigma_compatible")
DISPLAY = {"narm": "NARM", "sr_gnn": "SR-GNN", "gru4rec": "GRU4Rec",
           "sasrec": "SASRec", "sigma_compatible": "SIGMA"}
DOMAINS = {
    "Video_Games": "video_games",
    "Baby_Products": "baby_products",
}


def paired_test(challenger: np.ndarray, baseline: np.ndarray,
                k: int = K, reps: int = REPS, seed: int = 20260721) -> dict:
    hit_a = (challenger > 0) & (challenger <= k)
    hit_b = (baseline > 0) & (baseline <= k)
    pos = int(np.sum(hit_a & ~hit_b))
    neg = int(np.sum(~hit_a & hit_b))
    disc = pos + neg
    p = float(binomtest(pos, disc, 0.5).pvalue) if disc else 1.0
    rng = np.random.default_rng(seed)
    probs = np.asarray([pos, neg, int(len(hit_a) - pos - neg)], dtype=float) / len(hit_a)
    draws = rng.multinomial(len(hit_a), probs, size=reps)
    delta = (draws[:, 0] - draws[:, 1]) / len(hit_a)
    return {
        "challenger_recall": float(hit_a.mean()),
        "baseline_recall": float(hit_b.mean()),
        "difference": float(hit_a.mean() - hit_b.mean()),
        "ci95": [float(x) for x in np.quantile(delta, [.025, .975])],
        "pos": pos, "neg": neg,
        "mcnemar_p": p,
        "n": int(len(hit_a)),
    }


def main():
    results = {}
    for ds_key, ds_label in DOMAINS.items():
        results[ds_key] = {"baselines": {}}
        print(f"\n=== {ds_key} (v2 regime) ===")
        for seed in SEEDS:
            v2_path = HERE / "cearfn_v2_artifacts" / f"{ds_label}_v2_seed{seed}_ranks.npz"
            if not v2_path.exists():
                print(f"  MISSING v2 seed={seed}")
                continue
            v2 = np.load(v2_path)
            rank_key = "selected_rank" if "selected_rank" in v2.files else "regime_rank"
            cearfn = v2[rank_key].astype(np.int32)

            for bl in BASELINES:
                bl_path = HERE / "paper_baseline_artifacts" / f"{ds_label}_full_{bl}_seed{seed}_ranks.npz"
                if not bl_path.exists():
                    continue
                bl_data = np.load(bl_path)
                bl_ranks = bl_data["ranks"].astype(np.int32)

                r = paired_test(cearfn, bl_ranks)
                if bl not in results[ds_key]["baselines"]:
                    results[ds_key]["baselines"][bl] = {"per_seed": {}}
                results[ds_key]["baselines"][bl]["per_seed"][seed] = r

        # Aggregate
        for bl in list(results[ds_key]["baselines"].keys()):
            per_seed = results[ds_key]["baselines"][bl]["per_seed"]
            total_pos = sum(r["pos"] for r in per_seed.values())
            total_neg = sum(r["neg"] for r in per_seed.values())
            disc = total_pos + total_neg
            agg_p = float(binomtest(total_pos, disc, 0.5).pvalue) if disc else 1.0
            diffs = [r["difference"] for r in per_seed.values()]
            results[ds_key]["baselines"][bl]["aggregate"] = {
                "mean_difference": float(np.mean(diffs)),
                "sign_test_p": agg_p,
                "total_pos": total_pos,
                "total_neg": total_neg,
            }
            c_stack = []
            b_stack = []
            for seed in SEEDS:
                with np.load(HERE / "cearfn_v2_artifacts" /
                             f"{ds_label}_v2_seed{seed}_ranks.npz") as z:
                    key = "selected_rank" if "selected_rank" in z.files else "regime_rank"
                    c_stack.append(z[key].astype(np.int32))
                with np.load(HERE / "paper_baseline_artifacts" /
                             f"{ds_label}_full_{bl}_seed{seed}_ranks.npz") as z:
                    b_stack.append(z["ranks"].astype(np.int32))
            results[ds_key]["baselines"][bl]["cluster_aggregate"] = \
                cluster_paired_recall(c_stack, b_stack, k=K, reps=REPS)

    # Print
    for ds_key in results:
        print(f"\n=== {ds_key} (v2 regime vs baselines) ===")
        for bl in BASELINES:
            if bl not in results[ds_key]["baselines"]:
                continue
            agg = results[ds_key]["baselines"][bl]["aggregate"]
            ps = results[ds_key]["baselines"][bl]["per_seed"]
            # Mean recalls
            c_mean = np.mean([ps[s]["challenger_recall"] for s in ps])
            b_mean = np.mean([ps[s]["baseline_recall"] for s in ps])
            # CI from individual seeds
            ci_lo = np.mean([ps[s]["ci95"][0] for s in ps])
            ci_hi = np.mean([ps[s]["ci95"][1] for s in ps])
            print(f"  vs {DISPLAY[bl]:12s}: Δ={agg['mean_difference']:+.5f} "
                  f"CI95=[{ci_lo:+.5f},{ci_hi:+.5f}] p={agg['sign_test_p']:.2e} "
                  f"(CEARF-N={c_mean:.5f}, {DISPLAY[bl]}={b_mean:.5f})")

    path = HERE / "v2_paired_amazon.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
