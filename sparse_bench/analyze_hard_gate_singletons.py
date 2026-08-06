#!/usr/bin/env python3
"""Audit the 15-member external expert family against its best singleton.

The external family contains every non-empty subset of
{CEARF-N, STAN, V-SKNN, NARM}.  Existing rank artifacts are reused; no model
is retrained and no test label is used for selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import loaders
from neighborhood_baselines import NeighborhoodConfig, NeighborhoodIndex
from paired_statistics import cluster_paired_recall
from run_cearfn_evidence import ranks_at_20, targets_for
from run_global_fusion_selector import (
    fuse_rrf,
    load_test_matrix,
    predict_matrix,
)


HERE = Path(__file__).resolve().parent
SEEDS = (42, 123, 456)
EXPERTS = ("cearf", "stan", "vsknn", "narm")
DOMAINS = {
    "Video_Games": {
        "slug": "video_games",
        "seed42_selector": (
            HERE / "cearfn_v2_video_games_validation_artifacts"
            / "video_games_v2_seed42_ranks.npz"
        ),
    },
    "Baby_Products": {
        "slug": "baby_products",
        "seed42_selector": HERE / "baby_selector_seed42.npz",
    },
    "Diginetica_HID": {
        "slug": "diginetica",
        "seed42_selector": HERE / "diginetica_selector_seed42.npz",
    },
}


def selection_key(metrics: dict, name: str) -> tuple:
    return (
        .5 * (metrics["recall@6"] + metrics["recall@20"]),
        metrics["recall@20"],
        metrics["ndcg@20"],
        name,
    )


def utility(metrics: dict) -> float:
    return .5 * (metrics["recall@6"] + metrics["recall@20"])


def align(keys: list[str], source_keys: list[str], values: np.ndarray) -> np.ndarray:
    if set(keys) != set(source_keys):
        raise RuntimeError("Query-key coverage mismatch")
    row = {key: i for i, key in enumerate(source_keys)}
    return np.asarray([values[row[key]] for key in keys], dtype=np.int32)


def selector_path(domain: str, seed: int) -> Path:
    spec = DOMAINS[domain]
    if seed == 42:
        return spec["seed42_selector"]
    return (
        HERE / "narm_expert_artifacts" / "selectors"
        / f"{spec['slug']}_selector_seed{seed}.npz"
    )


def artifact_name(path: Path) -> str:
    """Store repository-relative provenance without leaking a local username."""
    return str(path.resolve().relative_to(HERE.resolve()))


def candidate_parts(name: str) -> tuple[str, ...]:
    if name in EXPERTS:
        return (name,)
    if not name.startswith("rrf_"):
        raise ValueError(f"Unknown candidate: {name}")
    parts = tuple(name.removeprefix("rrf_").split("_"))
    if not parts or any(part not in EXPERTS for part in parts):
        raise ValueError(f"Invalid candidate: {name}")
    return parts


def candidate_rank(name: str, rankings: dict[str, np.ndarray]) -> np.ndarray:
    parts = candidate_parts(name)
    if len(parts) == 1:
        return rankings[parts[0]]
    return np.asarray(
        [fuse_rrf([rankings[part][row] for part in parts])
         for row in range(len(next(iter(rankings.values()))))],
        dtype=np.int32,
    )


def main() -> None:
    baseline_results = json.loads(
        (HERE / "neighborhood_baseline_results.json").read_text()
    )
    matched = json.loads(
        (HERE / "narm_expert_matched_r6r20.json").read_text()
    )
    output = {
        "selection_objective": "0.5*Recall@6 + 0.5*Recall@20",
        "family": {
            "experts": list(EXPERTS),
            "members": 15,
            "definition": "all non-empty subsets; singleton rejects ensembling",
            "empty_set": "inadmissible because a recommender must emit a ranking",
        },
        "canonical_lineage": {
            "core_results": "cearfn_v2_nested_results.json",
            "hard_gate_selection": "narm_expert_matched_r6r20.json",
            "per_seed_candidates": "narm_expert_{domain}_seed{seed}.json",
            "legacy_diginetica_0.49028": (
                "exploratory continuous-router result; excluded from submission"
            ),
        },
        "domains": {},
    }

    for domain, spec in DOMAINS.items():
        data = loaders.ALL_LOADERS[domain]()
        selected_runs = {
            int(run["seed"]): run
            for run in matched["domains"][domain]["runs"]
        }
        baseline = baseline_results[domain]["methods"]
        loaded_stan = load_test_matrix(Path(baseline["stan"]["artifact"]))
        loaded_vsknn = load_test_matrix(Path(baseline["vsknn"]["artifact"]))
        if loaded_stan is None or loaded_vsknn is None:
            test_index = NeighborhoodIndex(
                data["train_sessions"], data["n_items"]
            )
            if loaded_stan is None:
                loaded_stan = predict_matrix(
                    test_index,
                    data["test_queries"],
                    NeighborhoodConfig(**baseline["stan"]["selected_config"]),
                )
            if loaded_vsknn is None:
                loaded_vsknn = predict_matrix(
                    test_index,
                    data["test_queries"],
                    NeighborhoodConfig(**baseline["vsknn"]["selected_config"]),
                )

        selected_positions = []
        singleton_positions = []
        runs = []
        for seed in SEEDS:
            source_path = HERE / f"narm_expert_{spec['slug']}_seed{seed}.json"
            source = json.loads(source_path.read_text())
            if len(source["validation"]) != 15:
                raise RuntimeError(
                    f"{source_path} has {len(source['validation'])} candidates, not 15"
                )
            selected = selected_runs[seed]["selected_candidate"]
            best_singleton = max(
                EXPERTS,
                key=lambda name: selection_key(source["validation"][name], name),
            )

            with np.load(selector_path(domain, seed), allow_pickle=True) as c:
                test_keys = [str(value) for value in c["test_keys"]]
                cearf_rank = np.asarray(c["selected_top20"], dtype=np.int32)
            with np.load(
                HERE / "narm_expert_artifacts"
                / f"{spec['slug']}_narm_seed{seed}_top20.npz",
                allow_pickle=True,
            ) as n:
                narm_rank = align(
                    test_keys,
                    [str(value) for value in n["test_keys"]],
                    np.asarray(n["test_top20"], dtype=np.int32),
                )
            stan_rank = align(test_keys, loaded_stan[0], loaded_stan[1])
            vsknn_rank = align(test_keys, loaded_vsknn[0], loaded_vsknn[1])
            rankings = {
                "cearf": cearf_rank,
                "stan": stan_rank,
                "vsknn": vsknn_rank,
                "narm": narm_rank,
            }

            selected_rank = candidate_rank(selected, rankings)
            singleton_rank = candidate_rank(best_singleton, rankings)
            targets = targets_for(test_keys, data["test_queries"])
            selected_position = ranks_at_20(selected_rank, targets)
            singleton_position = ranks_at_20(singleton_rank, targets)
            selected_positions.append(selected_position)
            singleton_positions.append(singleton_position)

            admission_margin = (
                utility(source["validation"][selected])
                - utility(source["validation"][best_singleton])
            )
            test_margin = (
                utility(source["test"][selected])
                - utility(source["test"][best_singleton])
            )
            runs.append({
                "seed": seed,
                "selected": selected,
                "best_validation_singleton": best_singleton,
                "validation_admission_margin": admission_margin,
                "test_retention_margin": test_margin,
                "reversal": bool(admission_margin > 0 and test_margin < 0),
                "selected_test_recall@20": float(
                    np.mean((selected_position > 0) & (selected_position <= 20))
                ),
                "singleton_test_recall@20": float(
                    np.mean((singleton_position > 0) & (singleton_position <= 20))
                ),
                "candidate_source": artifact_name(source_path),
                "selector_artifact": artifact_name(selector_path(domain, seed)),
            })

        paired = cluster_paired_recall(
            np.stack(selected_positions),
            np.stack(singleton_positions),
            k=20,
        )
        output["domains"][domain] = {
            "runs": runs,
            "mean_validation_admission_margin": float(np.mean([
                run["validation_admission_margin"] for run in runs
            ])),
            "mean_test_retention_margin": float(np.mean([
                run["test_retention_margin"] for run in runs
            ])),
            "reversals": int(sum(run["reversal"] for run in runs)),
            "paired_selected_vs_validation_best_singleton": paired,
        }

    output_path = HERE / "hard_gate_singleton_audit.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(output_path)
    for domain, block in output["domains"].items():
        paired = block["paired_selected_vs_validation_best_singleton"]
        print(
            domain,
            f"selected={paired['challenger_recall_mean']:.5f}",
            f"singleton={paired['baseline_recall_mean']:.5f}",
            f"delta={paired['difference']:+.5f}",
            f"CI95={paired['cluster_bootstrap_ci95']}",
            f"p={paired['cluster_sign_flip_p']:.6g}",
            f"reversals={block['reversals']}",
        )


if __name__ == "__main__":
    main()
