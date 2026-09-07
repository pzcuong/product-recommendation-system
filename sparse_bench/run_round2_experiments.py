"""Round-2 experiments for the CEARF-N journal submission.

Implements the three reviewer-priority experiments after the filesystem
incident that destroyed the un-tracked artifact trees:

  E1  selection-procedure comparison  (endpoint vs global-beta vs router
      family vs forced-inclusion, with tuning budgets and regret)
  E2  matched-information comparison   (ID-only vs same-teacher conditions,
      plus neural-expert swap generality)
  E3  cost-quality audit               (R@20 / MRR@20 recomputed from the
      surviving frozen arrays, tuning budgets, recorded serving latency)

Provenance discipline
---------------------
RECOMPUTED  numbers are derived in this script from
            artifacts_paper/per_query_ranks (72 frozen per-query rank
            arrays; integrity-checked against per_seed_metrics.json).
RECORDED    numbers are transcribed verbatim from the locked auto-generated
            LaTeX tables in paper/generated_dynamic_beta_*.tex (produced by
            the original summarizers before the incident) and from the
            paper's own Table 5.  Each constant carries its source.
Nothing else is used; nothing is estimated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE
OUT = HERE / "experiments_round2"
OUT.mkdir(exist_ok=True)

DOMAINS = ["Video_Games", "Baby_Products", "Diginetica_HID"]
PRETTY = {"Video_Games": "Video", "Baby_Products": "Baby",
          "Diginetica_HID": "Diginetica"}

# ---------------------------------------------------------------------------
# RECORDED (paper/generated_dynamic_beta_main_table.tex — summarize_dynamic_beta.py)
# Test R@20, matched-seed means; DeltaU = dynamic minus OOF-global with
# paired query-level 95% bootstrap CI.
# ---------------------------------------------------------------------------
MAIN = {
    "Video_Games":    {"memory": .11901, "neural": .12571, "equal05": .14610,
                       "oof_global": .14719, "oof_short_long": .14726,
                       "dynamic": .14735, "dU_ci": (.00009, .00035),
                       "dU": .00022},
    "Baby_Products":  {"memory": .03879, "neural": .04873, "equal05": .05555,
                       "oof_global": .05671, "oof_short_long": .05655,
                       "dynamic": .05669, "dU_ci": (-.00003, .00006),
                       "dU": .00002},
    "Diginetica_HID": {"memory": .49428, "neural": .44852, "equal05": .51430,
                       "oof_global": .51702, "oof_short_long": .51692,
                       "dynamic": .51701, "dU_ci": (-.00015, .00018),
                       "dU": .00002},
}

# RECORDED utility U = .5 R@6 + .5 R@20 (generated_dynamic_beta_full_metrics.tex)
UTIL = {
    "Video_Games":    {"equal05": .11285, "oof_global": .11259,
                       "oof_short_long": .11278, "dynamic": .11281},
    "Baby_Products":  {"equal05": .04298, "oof_global": .04341,
                       "oof_short_long": .04346, "dynamic": .04343},
    "Diginetica_HID": {"equal05": .41148, "oof_global": .41330,
                       "oof_short_long": .41309, "dynamic": .41332},
}

# RECORDED rescue/damage (generated_dynamic_beta_macros.tex)
RESCUE = {
    "Video_Games":    {"rescues": 4451, "damage": 1766, "net": 2685},
    "Baby_Products":  {"rescues": 3889, "damage": 1191, "net": 2698},
    "Diginetica_HID": {"rescues": 3076, "damage": 1693, "net": 1383},
}

# RECORDED R@20 of allocation-control family (allocation_controls.tex) —
# used only for the P6 family spread.
FAMILY = {
    "Video_Games": [.14719, .14730, .14728, .14729, .14735, .14725, .14744,
                    .14720, .14728, .14731, .14733, .14728, .14734, .14734,
                    .14735],
    "Baby_Products": [.05671, .05661, .05643, .05672, .05669, .05671, .05670,
                      .05670, .05666, .05664, .05667, .05668, .05671, .05669,
                      .05669],
    "Diginetica_HID": [.51702, .51707, .51695, .51704, .51701, .51685,
                       .51702, .51692, .51706, .51709, .51710, .51697,
                       .51698, .51701, .51701],
}

# RECORDED expert swap (generated_dynamic_beta_expert_swap.tex), 3 seeds
SWAP = {
    "Video_Games":    {"narm_only": .13763, "oof_global": .14819,
                       "dynamic": .14854},
    "Baby_Products":  {"narm_only": .03880, "oof_global": .04994,
                       "dynamic": .05001},
    "Diginetica_HID": {"narm_only": .53178, "oof_global": .53910,
                       "dynamic": .53939},
}

# RECORDED matched-teacher table (paper/main_short.tex, tab:metadata)
MATCHED = {
    "Video_Games":   {"narm_id": .13705, "cearfn_nometa": .13208,
                      "narm_sem": .15387, "cearfn_full": .147},
    "Baby_Products": {"narm_id": .02990, "cearfn_nometa": .05055,
                      "narm_sem": .06628, "cearfn_full": .056},
}

# RECORDED serving latency (generated_dynamic_beta_runtime_macros.tex,
# benchmark_dynamic_beta_inference.py, Apple M2 Pro, warm)
RUNTIME = {
    "Video_Games":    {"gate_us": 1.171, "post_us": 85.89,
                       "expert_ms": 3.710, "total_ms": 3.796},
    "Baby_Products":  {"gate_us": 1.184, "post_us": 87.57,
                       "expert_ms": 4.031, "total_ms": 4.119},
    "Diginetica_HID": {"gate_us": .774, "post_us": 80.60,
                       "expert_ms": 2.428, "total_ms": 2.509},
}

# RECORDED tuning budgets (generated_dynamic_beta_baseline_audit.tex)
BUDGET_NEIGHBOR = {"V-SKNN": 8, "STAN": 16}
BUDGET_NEURAL = {"GRU4Rec": "epoch only (<=12)", "NARM": "epoch only (<=12)",
                 "SR-GNN": "epoch only (<=5)", "SIGMA-compat": "epoch only (<=10)"}
GATE_FAMILY_CELLS = 15  # allocation_controls.tex policy rows


def load_arrays():
    """RECOMPUTED source: frozen per-query rank arrays."""
    ranks = {}
    for f in sorted((ROOT / "artifacts_paper" / "per_query_ranks").glob("*.npz")):
        with np.load(f) as z:
            r = z["ranks"] if "ranks" in z.files else z[z.files[0]]
        dom, model, seed = f.stem.rsplit("_", 2)
        ranks.setdefault(dom, {}).setdefault(model, {})[seed] = np.asarray(r)
    return ranks


def mrr20(r):
    hit = (r > 0) & (r <= 20)
    return float(np.mean(np.where(hit, 1.0 / np.clip(r, 1, None), 0.0)))


def main() -> None:
    ranks = load_arrays()
    mrr_file = json.loads((OUT / "mrr_from_arrays.json").read_text())

    # =====================================================================
    # E1 — selection-procedure comparison
    # =====================================================================
    e1 = {"provenance": {
        "candidate_test_values": "RECORDED generated_dynamic_beta_main_table.tex",
        "win_frequencies": "RECOMPUTED from frozen per-query arrays",
        "deltaU_ci": "RECORDED generated_dynamic_beta_main_table.tex"},
        "domains": {}}

    for dom in DOMAINS:
        m = MAIN[dom]
        best_endpoint = max(m["memory"], m["neural"])
        best_endpoint_name = ("neural" if m["neural"] >= m["memory"]
                              else "memory")
        oracle = max(FAMILY[dom])
        procedures = {
            "P1 endpoint-only (best of memory/neural)":
                {"r20": best_endpoint, "budget": 2,
                 "note": f"selects {best_endpoint_name}"},
            "P2 global-beta fusion (training-learned)":
                {"r20": m["oof_global"], "budget": 0,
                 "note": "no validation cells; OOF scalar"},
            "P3 regime router (short/long)":
                {"r20": m["oof_short_long"], "budget": 2,
                 "note": "two OOF scalars"},
            "P4 oracle over gate family (upper bound)":
                {"r20": oracle, "budget": GATE_FAMILY_CELLS,
                 "note": f"test-side argmax; spread {max(FAMILY[dom])-min(FAMILY[dom]):.5f}"},
            "P5 forced neural inclusion (fusion-only best)":
                {"r20": max(m["equal05"], m["oof_global"],
                            m["oof_short_long"], m["dynamic"]),
                 "budget": 4,
                 "note": "endpoints forbidden"},
            "P6 query-conditioned gate (primary)":
                {"r20": m["dynamic"], "budget": GATE_FAMILY_CELLS,
                 "note": "3-feature bounded gate"},
        }
        for name, p in procedures.items():
            p["regret_vs_oracle"] = round(oracle - p["r20"], 5)

        # RECOMPUTED per-seed win frequency: fusion vs best endpoint
        freq = None
        if all(k in ranks.get(dom, {}) for k in
               ("cearfn", "memory", "neural")):
            wins, total = 0, 0
            for seed in sorted(ranks[dom]["cearfn"]):
                f = (ranks[dom]["cearfn"][seed] <= 20) & \
                    (ranks[dom]["cearfn"][seed] > 0)
                mem = (ranks[dom]["memory"][seed] > 0) & \
                      (ranks[dom]["memory"][seed] <= 20)
                neu = (ranks[dom]["neural"][seed] > 0) & \
                      (ranks[dom]["neural"][seed] <= 20)
                wins += int(f.mean() > max(mem.mean(), neu.mean()))
                total += 1
            freq = {"fusion_beats_best_endpoint_seeds": wins, "seeds": total}
        e1["domains"][dom] = {"procedures": procedures,
                              "fusion_win_frequency": freq,
                              "dU_dynamic_minus_global": m["dU"],
                              "dU_ci95": list(m["dU_ci"])}

    (OUT / "e1_selection_procedures.json").write_text(json.dumps(e1, indent=1))

    # =====================================================================
    # E2 — matched-information comparison
    # =====================================================================
    cis = json.loads((ROOT / "artifacts_paper" /
                      "real_paired_cis.json").read_text())["domains"]
    e2 = {"provenance": {
        "conditions": "RECORDED paper/main_short.tex tab:metadata",
        "swap": "RECORDED generated_dynamic_beta_expert_swap.tex (3 seeds)",
        "cis": "RECORDED artifacts_paper/real_paired_cis.json"},
        "domains": {}}
    for dom in DOMAINS:
        blk = {"matched_teacher": MATCHED.get(dom),
               "expert_swap": SWAP[dom],
               "swap_fusion_gain_over_narm_only":
                   round(SWAP[dom]["dynamic"] - SWAP[dom]["narm_only"], 5),
               "pasgr_fusion_gain_over_neural_only":
                   round(MAIN[dom]["dynamic"] - MAIN[dom]["neural"], 5),
               "rescue_damage": RESCUE[dom]}
        if dom in MATCHED:
            blk["metadata_gain_narm"] = round(
                MATCHED[dom]["narm_sem"] - MATCHED[dom]["narm_id"], 5)
            blk["metadata_gain_cearfn"] = round(
                MATCHED[dom]["cearfn_full"] - MATCHED[dom]["cearfn_nometa"],
                5)
        dom_cis = cis.get(dom, {})
        blk["paired_cis_vs_baselines"] = {
            k: {"delta": v["delta_r20"], "ci": v["ci95"], "p": v["p"]}
            for k, v in dom_cis.items() if k.startswith("vs_")}
        e2["domains"][dom] = blk
    (OUT / "e2_matched_teacher.json").write_text(json.dumps(e2, indent=1))

    # =====================================================================
    # E3 — cost-quality audit
    # =====================================================================
    e3 = {"provenance": {
        "quality": "RECOMPUTED from frozen arrays (R@20 verified 0 mismatches)",
        "mrr": "RECOMPUTED from frozen arrays",
        "latency": "RECORDED generated_dynamic_beta_runtime_macros.tex",
        "budgets": "RECORDED generated_dynamic_beta_baseline_audit.tex"},
        "domains": {}}
    method_key = {"Video_Games": "cearfn", "Baby_Products": "cearfn",
                  "Diginetica_HID": "continuous"}
    budgets = dict(BUDGET_NEIGHBOR)
    budgets.update(BUDGET_NEURAL)
    budgets["CEARF-N gate family"] = GATE_FAMILY_CELLS
    for dom in DOMAINS:
        per_method = {}
        for model, seeds in ranks[dom].items():
            rs = [float((((r > 0) & (r <= 20)).mean()))
                  for r in seeds.values()]
            ms = [mrr20(r) for r in seeds.values()]
            label = ("CEARF-N" if model == method_key[dom] else model)
            per_method[label] = {
                "r20_mean": round(float(np.mean(rs)), 5),
                "mrr20_mean": round(float(np.mean(ms)), 5),
                "tuning_budget": budgets.get(
                    {"sigma_compatible": "SIGMA-compat",
                     "sr_gnn": "SR-GNN"}.get(model, model), "n/a")}
        e3["domains"][dom] = {
            "methods": per_method,
            "cearfn_serving_latency": RUNTIME[dom],
            "latency_note": ("gate+fusion overhead is "
                             f"{RUNTIME[dom]['post_us']/1000:.3f} ms/query "
                             f"({RUNTIME[dom]['post_us']/RUNTIME[dom]['total_ms']/1000*100:.1f}% "
                             "of total); dominant cost is expert retrieval")}
    (OUT / "e3_cost_quality.json").write_text(json.dumps(e3, indent=1))

    for name in ("e1_selection_procedures", "e2_matched_teacher",
                 "e3_cost_quality"):
        print("wrote", OUT / f"{name}.json")


if __name__ == "__main__":
    main()
