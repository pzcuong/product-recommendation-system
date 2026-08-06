#!/usr/bin/env python3
"""Locked shortcut diagnostics for CEARF memory evidence.

The audit measures component target coverage and a no-reselection intervention
that removes transition evidence from each already selected memory profile.
"""
from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path

import numpy as np

import loaders


HERE = Path(__file__).resolve().parent
DOMAINS = {
    "Video Games": {
        "slug": "video_games",
        "loader": lambda: loaders.load_amazon("Video_Games"),
    },
    "Baby Products": {
        "slug": "baby_products",
        "loader": lambda: loaders.load_amazon("amazon_baby"),
    },
    "Diginetica": {
        "slug": "diginetica_hid",
        "loader": loaders.load_diginetica_hid,
    },
}
RRF_K = 20.0
CONSENSUS_BONUS = 0.12
SHORT_CONTEXT = 2


def query_fingerprint(queries: dict) -> str:
    digest = hashlib.sha256()
    for uid in sorted(queries):
        query = queries[uid]
        digest.update(str(uid).encode())
        digest.update(b"|")
        digest.update(" ".join(map(str, query.get("context", ()))).encode())
        digest.update(b"->")
        digest.update(" ".join(map(str, query.get("targets", ()))).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def hit_in_topk(array: np.ndarray, targets: np.ndarray, k: int = 20) -> np.ndarray:
    return (array[:, :k] == targets[:, None]).any(axis=1)


def fuse_row(rankings: tuple[np.ndarray, np.ndarray, np.ndarray],
             weights: tuple[float, float, float], topk: int = 20) -> list[int]:
    scores: dict[int, float] = {}
    votes: Counter[int] = Counter()
    for weight, ranking in zip(weights, rankings):
        if weight <= 0:
            continue
        for rank, value in enumerate(ranking, 1):
            item = int(value)
            if item <= 0:
                continue
            scores[item] = scores.get(item, 0.0) + weight / (RRF_K + rank)
            votes[item] += 1
    for item, count in votes.items():
        if count >= 2:
            scores[item] *= 1.0 + CONSENSUS_BONUS * (count - 1)
    if not scores:
        return []
    return [
        item for item, _ in heapq.nlargest(
            topk, scores.items(), key=lambda pair: (pair[1], pair[0]))
    ]


def main() -> None:
    result = {
        "protocol": {
            "selection_uses_test_labels": False,
            "component_coverage": "target appears in component top-20",
            "transition_removal": (
                "set the transition weight to zero in each locked short/long "
                "memory profile; keep all other weights and do not reselect"
            ),
            "scope": (
                "memory-path intervention only; it does not refit the neural "
                "model or the final memory-neural router"
            ),
        },
        "domains": {},
    }
    for domain, spec in DOMAINS.items():
        data = spec["loader"]()
        keys = sorted(data["test_queries"])
        targets = np.asarray(
            [int(data["test_queries"][uid]["targets"][0]) for uid in keys],
            dtype=np.int32,
        )
        lengths = np.asarray(
            [len(data["test_queries"][uid].get("context", ())) for uid in keys],
            dtype=np.int32,
        )
        path = (HERE / "cearfn_v2_nested_artifacts" /
                f"{spec['slug']}_nested_test_memory.npz")
        with np.load(path, allow_pickle=True) as saved:
            artifact_keys = [str(value) for value in saved["keys"]]
            artifact_fp = str(saved["fingerprint"].item())
            if artifact_keys != keys or artifact_fp != query_fingerprint(
                    data["test_queries"]):
                raise ValueError(f"{path}: query alignment mismatch")
            arrays = {
                name: np.asarray(saved[name], dtype=np.int32)
                for name in ("transition", "session", "popularity", "selected")
            }
            profiles = json.loads(str(saved["profiles"].item()))

        hits = {
            name: hit_in_topk(arrays[name], targets)
            for name in ("transition", "session", "popularity", "selected")
        }
        any_component = hits["transition"] | hits["session"] | hits["popularity"]
        regimes = np.where(lengths <= SHORT_CONTEXT, "short", "long")
        active_transition = np.asarray([
            float(profiles[str(regime)][0]) > 0 for regime in regimes
        ], dtype=bool)
        # Removing an inactive source is an identity operation; only reconstruct
        # rows whose locked profile actually uses transition evidence.
        no_transition_hit = hits["selected"].copy()
        reconstructed_hit = hits["selected"].copy()
        for row in np.flatnonzero(active_transition):
            weights = tuple(
                float(value) for value in profiles[str(regimes[row])])
            rankings = (
                arrays["transition"][row],
                arrays["session"][row],
                arrays["popularity"][row],
            )
            baseline = fuse_row(rankings, weights)
            reconstructed_hit[row] = int(targets[row]) in baseline
            no_t_weights = (0.0, weights[1], weights[2])
            if not any(no_t_weights):
                no_t_weights = (0.0, 0.0, 1.0)
            no_transition = fuse_row(rankings, no_t_weights)
            no_transition_hit[row] = int(targets[row]) in no_transition

        if not np.array_equal(reconstructed_hit, hits["selected"]):
            mismatch = float(np.mean(reconstructed_hit != hits["selected"]))
            raise AssertionError(
                f"{domain}: reconstructed memory mismatch share={mismatch}")

        def rate(mask: np.ndarray) -> float:
            return float(mask.mean())

        domain_result = {
            "n_queries": len(keys),
            "profiles": profiles,
            "component_recall@20": {
                "transition": rate(hits["transition"]),
                "similar_session": rate(hits["session"]),
                "popularity": rate(hits["popularity"]),
                "selected_memory": rate(hits["selected"]),
                "union_of_components": rate(any_component),
            },
            "component_outcome_regions": {
                "transition_only": rate(
                    hits["transition"] & ~hits["session"] & ~hits["popularity"]),
                "session_only": rate(
                    hits["session"] & ~hits["transition"] & ~hits["popularity"]),
                "popularity_only": rate(
                    hits["popularity"] & ~hits["transition"] & ~hits["session"]),
                "none": rate(~any_component),
            },
            "transition_removal": {
                "queries_with_transition_weight_active": int(active_transition.sum()),
                "active_share": rate(active_transition),
                "locked_memory_recall@20": rate(hits["selected"]),
                "no_transition_memory_recall@20": rate(no_transition_hit),
                "delta_no_transition_minus_locked": (
                    rate(no_transition_hit) - rate(hits["selected"])
                ),
                "rescued_by_removal": rate(
                    ~hits["selected"] & no_transition_hit),
                "damaged_by_removal": rate(
                    hits["selected"] & ~no_transition_hit),
            },
            "transition_removal_by_context": {},
        }
        for label, mask in {
            "short": lengths <= SHORT_CONTEXT,
            "long": lengths > SHORT_CONTEXT,
        }.items():
            if not mask.any():
                domain_result["transition_removal_by_context"][label] = {
                    "n_queries": 0,
                    "transition_weight_active_share": None,
                    "locked_memory_recall@20": None,
                    "no_transition_memory_recall@20": None,
                    "delta_no_transition_minus_locked": None,
                }
                continue
            domain_result["transition_removal_by_context"][label] = {
                "n_queries": int(mask.sum()),
                "transition_weight_active_share": float(
                    active_transition[mask].mean()),
                "locked_memory_recall@20": float(
                    hits["selected"][mask].mean()),
                "no_transition_memory_recall@20": float(
                    no_transition_hit[mask].mean()),
                "delta_no_transition_minus_locked": float(
                    no_transition_hit[mask].mean()
                    - hits["selected"][mask].mean()),
            }
        result["domains"][domain] = domain_result

    output = HERE / "shortcut_diagnostics_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
