#!/usr/bin/env python3
"""One-time test read for the CASM pilot (run AFTER pilot_guarded_analysis.py
has frozen all selections — read-once rule, DESIGN_CASM.md §4).

Reads stored test rank vectors for the three memory variants and the three
selection-rule picks, computes full metrics + train-frequency strata, and a
paired McNemar of each variant vs the locked baseline.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import cearf
import loaders
from guarded_selection import mcnemar_one_sided_p
from run_cearfn_evidence import metrics_from_ranks, targets_for, ranks_at_20

HERE = Path(__file__).resolve().parent
ART = HERE / "cearfn_v2_pilot_artifacts"
RRF_C = 20.0

# Frozen selection-rule picks (from pilot_guarded_audit.json, validation only)
AUDIT = json.loads((HERE / "pilot_guarded_audit.json").read_text())

# candidate name -> (npz variant, test rank key) mapping
CANDIDATE_TEST_KEY = {
    "v2_pasgr_regime": ("off", "regime_rank"),
    "v2_pasgr_bucketed": ("off", "bucketed_rank"),
    "v2_pasgr_continuous": ("off", "continuous_rank"),
    "raw_semantic_regime": ("raw_semantic", "regime_rank"),
    "casm_regime": ("casm", "regime_rank"),
    "casm_bucketed": ("casm", "bucketed_rank"),
}


def fuse_beta_top20(memory_row, neural_row, beta, topk=20):
    score = defaultdict(float)
    if beta < 1.0:
        for rank, item in enumerate(memory_row, 1):
            if item > 0:
                score[int(item)] += (1.0 - beta) / (RRF_C + rank)
    if beta > 0.0:
        for rank, item in enumerate(neural_row, 1):
            if item > 0:
                score[int(item)] += beta / (RRF_C + rank)
    ranking = [i for i, _ in sorted(score.items(),
                                    key=lambda x: (-x[1], x[0]))[:topk]]
    out = np.zeros(topk, dtype=np.int32)
    out[:len(ranking)] = ranking
    return out


def strata_metrics(ranks, tgt_label, tgt_freq):
    out = {"overall": metrics_from_ranks(ranks)}
    for s in ("head", "torso", "tail"):
        out[s] = metrics_from_ranks(ranks, tgt_label == s)
    out["coldstart"] = metrics_from_ranks(ranks, tgt_freq == 0)
    return out


def main():
    results = {}
    for domain in ("Video_Games", "Baby_Products"):
        tag = domain.lower()
        data = loaders.ALL_LOADERS[domain]()
        npz = {
            "off": np.load(ART / f"{tag}_v2_seed42_ranks.npz"),
            "raw_semantic": np.load(ART / f"{tag}_v2_raw_semantic_seed42_ranks.npz"),
            "casm": np.load(ART / f"{tag}_v2_casm_seed42_ranks.npz"),
        }
        test_keys = [str(x) for x in npz["off"]["test_keys"]]
        for v in ("raw_semantic", "casm"):
            assert [str(x) for x in npz[v]["test_keys"]] == test_keys
        targets = targets_for(test_keys, data["test_queries"])
        freq_arr = np.load(HERE / f"pilot_strata_{tag}_freq.npy")
        lab_arr = np.load(HERE / f"pilot_strata_{tag}_labels.npy")
        tgt_freq = freq_arr[targets]
        tgt_label = lab_arr[targets]

        dom = {"n_test_queries": len(test_keys),
               "strata_sizes": {s: int((tgt_label == s).sum())
                                for s in ("head", "torso", "tail")},
               "coldstart_n": int((tgt_freq == 0).sum())}

        # --- three memory variants: runner-selected router rank vectors ---
        variants = {}
        for vname, key in (("off", "off"), ("raw-semantic", "raw_semantic"),
                           ("casm", "casm")):
            z = npz[key]
            r = z["selected_rank"]
            variants[vname] = {
                "selected_router": str(z["selected_router"]),
                **strata_metrics(r, tgt_label, tgt_freq),
            }
        # paired vs off (two-sided direction shown via both one-sided ps)
        hit_off = (npz["off"]["selected_rank"] > 0) & (npz["off"]["selected_rank"] <= 20)
        paired = {}
        for vname, key in (("raw-semantic", "raw_semantic"), ("casm", "casm")):
            h = (npz[key]["selected_rank"] > 0) & (npz[key]["selected_rank"] <= 20)
            n01 = int(np.sum(hit_off & ~h))  # off hits, variant misses
            n10 = int(np.sum(h & ~hit_off))
            paired[vname] = {"variant_only_hits": n10, "off_only_hits": n01,
                             "p_variant_worse": mcnemar_one_sided_p(n01, n10),
                             "p_off_worse": mcnemar_one_sided_p(n10, n01)}
        dom["variants"] = variants
        dom["paired_vs_off"] = paired

        # --- selection-rule decision table (frozen picks -> test, once) ---
        sel = AUDIT[domain]["candidate_family_selection"]
        beta_const = AUDIT[domain]["extra"]["constant_beta"]
        decision = {}
        for rule in ("argmax", "one_se", "guarded"):
            pick = sel[rule]["selected"]
            if pick in CANDIDATE_TEST_KEY:
                variant, key = CANDIDATE_TEST_KEY[pick]
                r = npz[variant][key]
                approx = False
            elif pick == "v2_pasgr_constant_beta":
                z = npz["off"]
                mem20 = np.asarray(z["memory_top20"])
                neu20 = np.asarray(z["neural_top20"])
                fused = np.stack([
                    fuse_beta_top20(mem20[i], neu20[i], beta_const)
                    for i in range(len(test_keys))])
                r = ranks_at_20(fused, targets)
                approx = True
            else:
                raise ValueError(f"no test mapping for pick {pick!r}")
            decision[rule] = {
                "selected": pick,
                "test_approximated_from_top20": approx,
                **strata_metrics(r, tgt_label, tgt_freq),
            }
        dom["decision_table"] = decision
        results[domain] = dom
        print(f"[TEST-READ] {domain} done", flush=True)

    out = HERE / "pilot_test_metrics.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"[TEST-READ] saved {out}", flush=True)


if __name__ == "__main__":
    main()
