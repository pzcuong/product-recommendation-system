#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cearf
import loaders
from neighborhood_baselines import NeighborhoodConfig, NeighborhoodIndex
from run_cearfn_evidence import metrics_from_ranks, ranks_at_20, targets_for
from validation_protocol import hold_out_validation_targets


HERE = Path(__file__).resolve().parent


def utility(metrics: dict[str, float | int]) -> float:
    return 0.5 * (float(metrics["recall@6"]) + float(metrics["recall@20"]))


def criterion_value(name: str, metrics: dict[str, float | int]) -> tuple[float, ...]:
    r6 = float(metrics["recall@6"])
    r10 = float(metrics["recall@10"])
    r20 = float(metrics["recall@20"])
    ndcg20 = float(metrics["ndcg@20"])
    if name == "r6_r20":
        return (0.5 * (r6 + r20), r20, ndcg20)
    if name == "r10_r20":
        return (0.5 * (r10 + r20), r20, ndcg20)
    if name == "r20":
        return (r20, ndcg20)
    if name == "r6_r10_r20":
        return (0.25 * r6 + 0.25 * r10 + 0.5 * r20, r20, ndcg20)
    raise ValueError(f"Unknown criterion: {name}")


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


def summarize(rankings: dict[str, np.ndarray], targets: np.ndarray,
              criterion: str) -> tuple[dict[str, dict], str]:
    summary: dict[str, dict] = {}
    best = None
    for name, ranking in rankings.items():
        metrics = metrics_from_ranks(ranks_at_20(ranking, targets))
        summary[name] = metrics
        candidate = criterion_value(criterion, metrics) + (name,)
        if best is None or candidate > best[0]:
            best = (candidate, name)
    assert best is not None
    return summary, best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="Tmall")
    parser.add_argument("--cearf-artifact", type=Path,
                        default=HERE / "cearfn_v2_tmall_validation_artifacts" / "tmall_v2_seed42_ranks.npz")
    parser.add_argument("--baseline-results", type=Path,
                        default=HERE / "neighborhood_baseline_tmall.json")
    parser.add_argument("--output", type=Path,
                        default=HERE / "tmall_global_selector.json")
    parser.add_argument("--criterion",
                        choices=("r6_r20", "r10_r20", "r20", "r6_r10_r20"),
                        default="r6_r20")
    args = parser.parse_args()

    c = np.load(args.cearf_artifact, allow_pickle=True)
    needed = {"valid_keys", "test_keys", "valid_selected_top20", "selected_top20"}
    missing = needed.difference(c.files)
    if missing:
        raise RuntimeError(f"Missing CEARF arrays: {sorted(missing)}")

    data = loaders.load_tmall()
    validation_subset_keys = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
    valid_queries = {k: data["valid_queries"][k] for k in validation_subset_keys}
    test_queries = data["test_queries"]

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
        raise RuntimeError("Validation key coverage mismatch.")
    vrow_valid = {k: i for i, k in enumerate(v_valid_keys)}
    srow_valid = {k: i for i, k in enumerate(s_valid_keys)}
    vsknn_valid = np.asarray([vsknn_valid[vrow_valid[k]] for k in c_valid_keys], dtype=np.int32)
    stan_valid = np.asarray([stan_valid[srow_valid[k]] for k in c_valid_keys], dtype=np.int32)

    test_keys = sorted(test_queries.keys())
    c_test_keys = [str(x) for x in c["test_keys"]]
    if c_test_keys != test_keys:
        raise RuntimeError("CEARF test key order mismatch.")
    if set(v_test_keys) != set(test_keys) or set(s_test_keys) != set(test_keys):
        raise RuntimeError("Test baseline key coverage mismatch.")
    vrow_test = {k: i for i, k in enumerate(v_test_keys)}
    srow_test = {k: i for i, k in enumerate(s_test_keys)}
    vsknn_test = np.asarray([vsknn_test[vrow_test[k]] for k in test_keys], dtype=np.int32)
    stan_test = np.asarray([stan_test[srow_test[k]] for k in test_keys], dtype=np.int32)

    candidate_valid = {
        "cearf": np.asarray(c["valid_selected_top20"], dtype=np.int32),
        "stan": stan_valid,
        "vsknn": vsknn_valid,
    }
    candidate_test = {
        "cearf": np.asarray(c["selected_top20"], dtype=np.int32),
        "stan": stan_test,
        "vsknn": vsknn_test,
    }
    pairings = {
        "rrf_cearf_stan": ("cearf", "stan"),
        "rrf_cearf_vsknn": ("cearf", "vsknn"),
        "rrf_stan_vsknn": ("stan", "vsknn"),
        "rrf_all3": ("cearf", "stan", "vsknn"),
    }
    for name, parts in pairings.items():
        candidate_valid[name] = np.asarray(
            [fuse_rrf([candidate_valid[p][i] for p in parts]) for i in range(len(c_valid_keys))],
            dtype=np.int32)
        candidate_test[name] = np.asarray(
            [fuse_rrf([candidate_test[p][i] for p in parts]) for i in range(len(test_keys))],
            dtype=np.int32)

    valid_targets = targets_for(c_valid_keys, valid_queries)
    test_targets = targets_for(test_keys, test_queries)
    validation_summary, selected = summarize(candidate_valid, valid_targets, args.criterion)
    test_summary, _ = summarize(candidate_test, test_targets, args.criterion)

    output = {
        "domain": args.domain,
        "selection_rule": args.criterion,
        "selected_candidate": selected,
        "validation": validation_summary,
        "test": test_summary,
        "selected_test": test_summary[selected],
    }
    args.output.write_text(json.dumps(output, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
