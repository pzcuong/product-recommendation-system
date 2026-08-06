#!/usr/bin/env python3
"""Pilot analysis (DESIGN_CASM.md §4): assemble per-query validation hits for
the frozen candidate family (§5), run guarded selection, and produce the
validation-side audit record. Test metrics are computed in a separate script
AFTER all selections are frozen (read-once rule).

All inputs are validation-only artifacts from the seed-42 pilot runs.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import cearf
import loaders
from guarded_selection import (Candidate, candidate_from_ranks,
                               compare_selection_rules, guarded_selection)
from run_cearfn_evidence import ranks_at_20, targets_for

HERE = Path(__file__).resolve().parent
ART = HERE / "cearfn_v2_pilot_artifacts"
RRF_C = 20.0
CONSENSUS = 0.12
SHORT_CONTEXT = 2  # cearf.CEARFConfig default


def fuse_rows(component_rows: list[np.ndarray], weights: list[float],
              pop_row: np.ndarray, topk: int = 20) -> np.ndarray:
    """Replicate CEARFIndex.fuse_rankings on stored (blocked) component rows.

    Rows are int32 padded with 0; item ids are > 0. Pad with popularity."""
    fused: dict[int, float] = defaultdict(float)
    votes: Counter = Counter()
    for weight, row in zip(weights, component_rows):
        if weight <= 0:
            continue
        for rank, item in enumerate(row, 1):
            if item <= 0:
                break
            fused[int(item)] += weight / (RRF_C + rank)
            votes[int(item)] += 1
    for item, count in votes.items():
        if count >= 2:
            fused[item] *= 1.0 + CONSENSUS * (count - 1)
    ranking = [item for item, _ in
               sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:topk]]
    if len(ranking) < topk:
        seen = set(ranking)
        for item in pop_row:
            if item > 0 and int(item) not in seen:
                ranking.append(int(item))
                if len(ranking) == topk:
                    break
    out = np.zeros(topk, dtype=np.int32)
    out[:len(ranking)] = ranking
    return out


def fuse_beta_top20(memory_row: np.ndarray, neural_row: np.ndarray,
                    beta: float, topk: int = 20) -> np.ndarray:
    """run_cearfn.fuse on top-20 truncated lists (approximation, disclosed)."""
    score: dict[int, float] = defaultdict(float)
    if beta < 1.0:
        for rank, item in enumerate(memory_row, 1):
            if item > 0:
                score[int(item)] += (1.0 - beta) / (RRF_C + rank)
    if beta > 0.0:
        for rank, item in enumerate(neural_row, 1):
            if item > 0:
                score[int(item)] += beta / (RRF_C + rank)
    ranking = [item for item, _ in
               sorted(score.items(), key=lambda x: (-x[1], x[0]))[:topk]]
    out = np.zeros(topk, dtype=np.int32)
    out[:len(ranking)] = ranking
    return out


def profile_predictions(components: dict[str, np.ndarray], names: tuple,
                        profile: tuple, pop: np.ndarray) -> np.ndarray:
    rows = len(components[names[0]])
    out = np.zeros((rows, 20), dtype=np.int32)
    comp_rows = [components[n] for n in names]
    for r in range(rows):
        out[r] = fuse_rows([c[r] for c in comp_rows], list(profile), pop[r])
    return out


def regime_masks(contexts: list, short_context: int = SHORT_CONTEXT):
    lengths = np.asarray([len(c) for c in contexts])
    return lengths <= short_context


def analyse_domain(domain: str) -> dict:
    tag = domain.lower()
    data = loaders.ALL_LOADERS[domain]()
    if len(data["valid_queries"]) > 5000:
        _vk = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
        data["valid_queries"] = {k: data["valid_queries"][k] for k in _vk}

    ranks_off = np.load(ART / f"{tag}_v2_seed42_ranks.npz")
    ranks_raw = np.load(ART / f"{tag}_v2_raw_semantic_seed42_ranks.npz")
    ranks_casm = np.load(ART / f"{tag}_v2_casm_seed42_ranks.npz")
    valid_keys = [str(x) for x in ranks_off["valid_keys"]]
    assert valid_keys == [str(x) for x in ranks_raw["valid_keys"]]
    assert valid_keys == [str(x) for x in ranks_casm["valid_keys"]]
    targets = targets_for(valid_keys, data["valid_queries"])
    contexts = [data["valid_queries"][k].get("context", ()) for k in valid_keys]
    short_mask = regime_masks(contexts)

    mem_off = np.load(ART / f"{tag}_nested_valid_memory.npz")
    mem_raw = np.load(ART / f"{tag}_raw_semantic_nested_valid_memory.npz")
    mem_casm = np.load(ART / f"{tag}_casm_nested_valid_memory.npz")
    mkeys = [str(x) for x in mem_off["keys"]]
    assert mkeys == valid_keys, "memory npz keys mismatch"
    assert [str(x) for x in mem_raw["keys"]] == valid_keys
    assert [str(x) for x in mem_casm["keys"]] == valid_keys

    comp = {name: mem_off[name] for name in
            ("transition", "session", "popularity")}
    pop = comp["popularity"]

    candidates: list[Candidate] = []
    extra: dict = {}

    # 1 popularity only / 2 transition only
    for pos, (name, row_arr, weights) in enumerate([
            ("popularity_only", None, (0.0, 0.0, 1.0)),
            ("transition_only", None, (1.0, 0.0, 0.0))]):
        preds = profile_predictions(comp, ("transition", "session",
                                           "popularity"), weights, pop)
        r = ranks_at_20(preds, targets)
        candidates.append(candidate_from_ranks(
            name, r, active_components=1, router=None, total_weights=1,
            position=pos))

    # 3 best v1 3-slot profile family (argmax within, per regime)
    best_by_regime = {}
    for regime, mask in (("short", short_mask), ("long", ~short_mask)):
        best = None
        for name, profile in cearf.PROFILES.items():
            preds = profile_predictions(comp, ("transition", "session",
                                               "popularity"), profile, pop)
            r = ranks_at_20(preds, targets)
            hit6 = ((r > 0) & (r <= 6))[mask].mean()
            hit20 = ((r > 0) & (r <= 20))[mask].mean()
            score = 0.5 * hit6 + 0.5 * hit20
            n_active = sum(1 for w in profile if w > 0)
            key = (score, hit20, hit6, -n_active, name)
            if best is None or key > best[0]:
                best = (key, name, profile, r)
        best_by_regime[regime] = best
    r_v1 = np.where(short_mask, best_by_regime["short"][3],
                    best_by_regime["long"][3])
    active_v1 = len({i for reg in best_by_regime.values()
                     for i, w in enumerate(reg[2]) if w > 0})
    total_w_v1 = sum(sum(1 for w in reg[2] if w > 0)
                     for reg in best_by_regime.values())
    candidates.append(candidate_from_ranks(
        "v1_profiles_argmax", r_v1, active_components=active_v1,
        router=None, total_weights=total_w_v1, position=2))
    extra["v1_profiles"] = {reg: best_by_regime[reg][1]
                            for reg in best_by_regime}

    # Complexity of the locked v2 memory profiles, computed from the stored
    # tuned profiles (identical convention to the variant candidates below).
    prof_off = json.loads(str(mem_off["profiles"].item()))
    active_off = len({i for reg in prof_off.values()
                      for i, w in enumerate(reg) if w > 0})
    total_w_off = sum(sum(1 for w in reg if w > 0)
                      for reg in prof_off.values())

    # 4 v2 + PASGR constant beta (top-20 fusion approximation, disclosed)
    mem20 = np.asarray(ranks_off["valid_memory_top20"])
    neu20 = np.asarray(ranks_off["valid_neural_top20"])
    best_beta = None
    for beta in [round(b * 0.05, 2) for b in range(21)]:
        preds = np.stack([fuse_beta_top20(mem20[i], neu20[i], beta)
                          for i in range(len(valid_keys))])
        r = ranks_at_20(preds, targets)
        hit6 = ((r > 0) & (r <= 6)).mean()
        hit20 = ((r > 0) & (r <= 20)).mean()
        key = (0.5 * hit6 + 0.5 * hit20, hit20, hit6, -beta)
        if best_beta is None or key > best_beta[0]:
            best_beta = (key, beta, r)
    candidates.append(candidate_from_ranks(
        "v2_pasgr_constant_beta", best_beta[2],
        active_components=active_off + 1, router="constant",
        total_weights=total_w_off, position=3))
    extra["constant_beta"] = best_beta[1]

    # 5-7 off-variant routers (exact stored rank vectors)
    for pos, (name, key, router) in enumerate([
            ("v2_pasgr_regime", "valid_regime_rank", "regime"),
            ("v2_pasgr_bucketed", "valid_bucketed_rank", "bucketed"),
            ("v2_pasgr_continuous", "valid_continuous_rank", "continuous")],
            start=4):
        candidates.append(candidate_from_ranks(
            name, ranks_off[key], active_components=active_off + 1,
            router=router, total_weights=total_w_off, position=pos))

    # 8 raw-semantic + regime; 9 casm + regime; 10 casm + bucketed
    prof_raw = json.loads(str(mem_raw["profiles"].item()))
    prof_casm = json.loads(str(mem_casm["profiles"].item()))

    def slots_of(profiles: dict) -> tuple[int, int]:
        active = len({i for reg in profiles.values()
                      for i, w in enumerate(reg) if w > 0})
        total = sum(sum(1 for w in reg if w > 0) for reg in profiles.values())
        return active, total

    a_raw, t_raw = slots_of(prof_raw)
    a_casm, t_casm = slots_of(prof_casm)
    candidates.append(candidate_from_ranks(
        "raw_semantic_regime", ranks_raw["valid_regime_rank"],
        active_components=a_raw + 1, router="regime",
        total_weights=t_raw, position=7))
    candidates.append(candidate_from_ranks(
        "casm_regime", ranks_casm["valid_regime_rank"],
        active_components=a_casm + 1, router="regime",
        total_weights=t_casm, position=8))
    candidates.append(candidate_from_ranks(
        "casm_bucketed", ranks_casm["valid_bucketed_rank"],
        active_components=a_casm + 1, router="bucketed",
        total_weights=t_casm, position=9))
    extra["tuned_profiles"] = {"raw_semantic": prof_raw, "casm": prof_casm}

    selection = compare_selection_rules(candidates, q=0.10, method="mcnemar")

    # Retrospective legacy gates -------------------------------------------
    # (a) router gate: guarded selection restricted to {regime,bucketed,
    #     continuous} reproduces/simplifies the runner's argmax router pick.
    router_family = [c for c in candidates
                     if c.name in ("v2_pasgr_regime", "v2_pasgr_bucketed",
                                   "v2_pasgr_continuous")]
    router_audit = compare_selection_rules(router_family, q=0.10,
                                           method="mcnemar")
    # (b) memory-profile gate: memory-only candidates incl. semantic/casm
    #     profiles — reproduces the tune_profiles_v3 argmax decision.
    mem_variants = {
        "v1_best": r_v1,
    }
    comp6_raw = {n: mem_raw[n] for n in
                 ("transition", "session", "popularity", "semantic_raw",
                  "repeat", "casm")}
    comp6_casm = {n: mem_casm[n] for n in
                  ("transition", "session", "popularity", "semantic_raw",
                   "repeat", "casm")}
    names6 = ("transition", "session", "popularity", "semantic_raw",
              "repeat", "casm")
    from cearf_v3_ext import PROFILES_V3
    for pname in ("semantic_only", "balanced_semantic", "semantic_tail"):
        preds = profile_predictions(comp6_raw, names6, PROFILES_V3[pname], pop)
        mem_variants[f"raw:{pname}"] = ranks_at_20(preds, targets)
    for pname in ("casm_only", "balanced_casm", "casm_tail"):
        preds = profile_predictions(comp6_casm, names6, PROFILES_V3[pname], pop)
        mem_variants[f"casm:{pname}"] = ranks_at_20(preds, targets)
    mem_family = []
    for pos, (name, r) in enumerate(mem_variants.items()):
        if name == "v1_best":
            act, tw = active_v1, total_w_v1
        else:
            prof = PROFILES_V3[name.split(":", 1)[1]]
            act, tw = sum(1 for w in prof if w > 0), sum(
                1 for w in prof if w > 0) * 2
        mem_family.append(candidate_from_ranks(
            name, r, active_components=act, router=None,
            total_weights=tw, position=pos))
    memory_audit = compare_selection_rules(mem_family, q=0.10,
                                           method="mcnemar")

    # Component-level control comparison (validation): casm vs semantic_raw
    comp_ranks = {
        "semantic_raw_component": ranks_at_20(mem_raw["semantic_raw"][:, :20],
                                              targets),
        "casm_component": ranks_at_20(mem_casm["casm"][:, :20], targets),
        "transition_component": ranks_at_20(mem_off["transition"][:, :20],
                                            targets),
    }
    freq_arr = np.load(HERE / f"pilot_strata_{tag}_freq.npy")
    lab_arr = np.load(HERE / f"pilot_strata_{tag}_labels.npy")
    tgt_freq = freq_arr[targets]
    tgt_label = lab_arr[targets]
    strata = {}
    for name, r in comp_ranks.items():
        hit20 = (r > 0) & (r <= 20)
        strata[name] = {"overall_hit20": float(hit20.mean())}
        for s in ("head", "torso", "tail"):
            m = tgt_label == s
            strata[name][s] = {"n": int(m.sum()),
                               "hit20": float(hit20[m].mean()) if m.any() else None}
        m = tgt_freq == 0
        strata[name]["coldstart"] = {"n": int(m.sum()),
                                     "hit20": float(hit20[m].mean()) if m.any() else None}
    # paired casm vs raw on validation (component level), overall + strata
    hits_c = (comp_ranks["casm_component"] > 0) & (comp_ranks["casm_component"] <= 20)
    hits_r = (comp_ranks["semantic_raw_component"] > 0) & (comp_ranks["semantic_raw_component"] <= 20)
    from guarded_selection import mcnemar_one_sided_p
    paired = {}
    for sname, m in [("overall", np.ones(len(targets), bool)),
                     ("head", tgt_label == "head"),
                     ("torso", tgt_label == "torso"),
                     ("tail", tgt_label == "tail"),
                     ("coldstart", tgt_freq == 0)]:
        n01 = int(np.sum(hits_r[m] & ~hits_c[m]))  # raw hits, casm misses
        n10 = int(np.sum(hits_c[m] & ~hits_r[m]))  # casm hits, raw misses
        paired[sname] = {"n": int(m.sum()), "casm_only_hits": n10,
                         "raw_only_hits": n01,
                         "p_casm_worse": mcnemar_one_sided_p(n01, n10),
                         "p_raw_worse": mcnemar_one_sided_p(n10, n01)}

    # equality check: did variant runs collapse to the off baseline?
    collapse = {
        "raw_regime_equals_off": bool(np.array_equal(
            ranks_raw["valid_regime_rank"], ranks_off["valid_regime_rank"])),
        "casm_regime_equals_off": bool(np.array_equal(
            ranks_casm["valid_regime_rank"], ranks_off["valid_regime_rank"])),
        "casm_test_selected_equals_off": bool(np.array_equal(
            ranks_casm["selected_rank"], ranks_off["selected_rank"])),
    }

    return {
        "domain": domain,
        "n_valid_queries": len(valid_keys),
        "candidate_family_selection": selection,
        "router_gate_retrospective": router_audit,
        "memory_gate_retrospective": memory_audit,
        "component_validation_strata": strata,
        "component_paired_casm_vs_raw": paired,
        "collapse_check": collapse,
        "extra": extra,
    }


def main():
    out = {}
    for domain in ("Video_Games", "Baby_Products"):
        print(f"[GUARDED] {domain}", flush=True)
        out[domain] = analyse_domain(domain)
    path = HERE / "pilot_guarded_audit.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[GUARDED] saved {path}", flush=True)


if __name__ == "__main__":
    main()
