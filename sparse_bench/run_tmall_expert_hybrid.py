#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import cearf
import loaders
from neighborhood_baselines import NeighborhoodConfig, NeighborhoodIndex
from run_cearfn_evidence import metrics_from_ranks, ranks_at_20, targets_for
from validation_protocol import hold_out_validation_targets


HERE = Path(__file__).resolve().parent


def fuse_rrf(rankings: list[np.ndarray], k: float = 20.0, topk: int = 20) -> np.ndarray:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking[:topk], 1):
            item = int(item)
            if item > 0:
                scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out = np.zeros(topk, dtype=np.int32)
    for i, (item, _) in enumerate(ranked[:topk]):
        out[i] = item
    return out


def predict_matrix(index: NeighborhoodIndex, queries: dict, cfg: NeighborhoodConfig) -> tuple[list[str], np.ndarray]:
    preds = index.predict(queries, cfg)
    keys = [str(k) for k in queries]
    matrix = np.asarray([preds[k][:20] for k in keys], dtype=np.int32)
    return keys, matrix


def group_name(context: list[int] | tuple[int, ...]) -> str:
    n = len(context)
    if n <= 2:
        return "short"
    if n <= 7:
        return "mid"
    return "long"


def fit_group_policy(group_rows: dict[str, list[int]],
                     candidate_rankings: dict[str, np.ndarray],
                     targets: np.ndarray) -> tuple[dict[str, str], dict[str, dict]]:
    policy: dict[str, str] = {}
    report: dict[str, dict] = {}
    for group, rows in group_rows.items():
        chosen = None
        for name, ranking in candidate_rankings.items():
            rr = ranks_at_20(ranking[rows], targets[rows])
            metrics = metrics_from_ranks(rr)
            utility = 0.5 * (metrics["recall@6"] + metrics["recall@20"])
            candidate = (utility, metrics["recall@20"], metrics["ndcg@20"], name)
            if chosen is None or candidate > chosen[0]:
                chosen = (candidate, name, metrics)
        assert chosen is not None
        policy[group] = chosen[1]
        report[group] = {"selected": chosen[1], **chosen[2]}
    return policy, report


def apply_policy(policy: dict[str, str],
                 query_keys: list[str],
                 queries: dict,
                 candidates: dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros_like(next(iter(candidates.values())))
    for row, key in enumerate(query_keys):
        out[row] = candidates[policy[group_name(queries[key]["context"]) ]][row]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="Tmall")
    parser.add_argument("--cearf-artifact", type=Path,
                        default=HERE / "cearfn_v2_tmall_validation_artifacts" / "tmall_v2_seed42_ranks.npz")
    parser.add_argument("--baseline-results", type=Path,
                        default=HERE / "neighborhood_baseline_tmall.json")
    parser.add_argument("--output", type=Path, default=HERE / "tmall_expert_hybrid.json")
    args = parser.parse_args()

    c = np.load(args.cearf_artifact, allow_pickle=True)
    if "valid_keys" not in c.files or "valid_memory_top20" not in c.files:
        raise RuntimeError("CEARF artifact missing validation arrays. Rerun run_cearfn_v2.py after the validation-artifact patch.")

    data = loaders.load_tmall()
    validation_subset_keys = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
    valid_queries = {k: data["valid_queries"][k] for k in validation_subset_keys}
    test_queries = data["test_queries"]
    exclude_seen = False

    baseline_cfgs = json.loads(args.baseline_results.read_text())["Tmall"]["methods"]
    vsknn_cfg = NeighborhoodConfig(**baseline_cfgs["vsknn"]["selected_config"])
    stan_cfg = NeighborhoodConfig(**baseline_cfgs["stan"]["selected_config"])

    tune_sessions = hold_out_validation_targets(data["train_sessions"], valid_queries)
    valid_index = NeighborhoodIndex(tune_sessions, data["n_items"])
    test_index = NeighborhoodIndex(data["train_sessions"], data["n_items"])

    v_valid_keys, vsknn_valid = predict_matrix(valid_index, valid_queries, vsknn_cfg)
    s_valid_keys, stan_valid = predict_matrix(valid_index, valid_queries, stan_cfg)
    v_test_keys, vsknn_test = predict_matrix(test_index, test_queries, vsknn_cfg)
    s_test_keys, stan_test = predict_matrix(test_index, test_queries, stan_cfg)
    c_valid_keys = [str(x) for x in c["valid_keys"]]
    if set(c_valid_keys) != set(validation_subset_keys):
        raise RuntimeError("CEARF validation key coverage mismatch.")
    vrow_valid = {k: i for i, k in enumerate(v_valid_keys)}
    srow_valid = {k: i for i, k in enumerate(s_valid_keys)}
    vsknn_valid = np.asarray([vsknn_valid[vrow_valid[k]] for k in c_valid_keys], dtype=np.int32)
    stan_valid = np.asarray([stan_valid[srow_valid[k]] for k in c_valid_keys], dtype=np.int32)
    test_keys = sorted(test_queries.keys())
    if set(v_test_keys) != set(test_keys) or set(s_test_keys) != set(test_keys):
        raise RuntimeError("Test baseline key coverage mismatch.")
    vrow = {k: i for i, k in enumerate(v_test_keys)}
    srow = {k: i for i, k in enumerate(s_test_keys)}
    vsknn_test = np.asarray([vsknn_test[vrow[k]] for k in test_keys], dtype=np.int32)
    stan_test = np.asarray([stan_test[srow[k]] for k in test_keys], dtype=np.int32)

    c_test_keys = [str(x) for x in c["test_keys"]]
    if c_test_keys != test_keys:
        raise RuntimeError("CEARF test key order mismatch.")

    valid_targets = targets_for(c_valid_keys, valid_queries)
    test_targets = targets_for(test_keys, test_queries)
    cearf_valid = np.asarray(c["selected_top20"][:0], dtype=np.int32)  # placeholder for dtype
    cearf_valid = np.asarray(c["valid_selected_top20"] if "valid_selected_top20" in c.files else c["valid_memory_top20"], dtype=np.int32)
    if "valid_selected_top20" not in c.files:
        raise RuntimeError("CEARF artifact missing valid_selected_top20. Rerun after saving fused validation rankings.")
    cearf_test = np.asarray(c["selected_top20"], dtype=np.int32)

    candidate_valid = {
        "cearf": cearf_valid,
        "stan": stan_valid,
        "vsknn": vsknn_valid,
    }
    candidate_test = {
        "cearf": cearf_test,
        "stan": stan_test,
        "vsknn": vsknn_test,
    }
    pairings = [
        ("rrf_cearf_stan", ("cearf", "stan")),
        ("rrf_cearf_vsknn", ("cearf", "vsknn")),
        ("rrf_stan_vsknn", ("stan", "vsknn")),
        ("rrf_all3", ("cearf", "stan", "vsknn")),
    ]
    for name, parts in pairings:
        candidate_valid[name] = np.asarray(
            [fuse_rrf([candidate_valid[p][i] for p in parts]) for i in range(len(c_valid_keys))],
            dtype=np.int32)
        candidate_test[name] = np.asarray(
            [fuse_rrf([candidate_test[p][i] for p in parts]) for i in range(len(test_keys))],
            dtype=np.int32)

    group_rows: dict[str, list[int]] = defaultdict(list)
    for row, key in enumerate(c_valid_keys):
        group_rows[group_name(valid_queries[key]["context"])].append(row)
    policy, validation_groups = fit_group_policy(group_rows, candidate_valid, valid_targets)
    hybrid_test = apply_policy(policy, test_keys, test_queries, candidate_test)
    hybrid_valid = apply_policy(policy, c_valid_keys, valid_queries, candidate_valid)

    metrics = {
        name: metrics_from_ranks(ranks_at_20(ranking, test_targets))
        for name, ranking in candidate_test.items()
    }
    metrics["hybrid"] = metrics_from_ranks(ranks_at_20(hybrid_test, test_targets))
    validation_metrics = {
        name: metrics_from_ranks(ranks_at_20(ranking, valid_targets))
        for name, ranking in candidate_valid.items()
    }
    validation_metrics["hybrid"] = metrics_from_ranks(ranks_at_20(hybrid_valid, valid_targets))

    output = {
        "domain": args.domain,
        "policy": policy,
        "validation_group_selection": validation_groups,
        "validation": validation_metrics,
        "test": metrics,
        "note": "Per-group hybrid policy is fit on validation only, then applied once to test."
    }
    args.output.write_text(json.dumps(output, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
