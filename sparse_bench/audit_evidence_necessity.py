#!/usr/bin/env python3
"""Decompose CEARF-N test gains into memory rescues and fusion damage.

This is a read-only audit over the locked v2 rank artifacts.  It never
reselects a model on test.  Query strata are derived from training frequency
and test context only; test targets are used solely for evaluation.
"""
from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

import cearf
import loaders
from run_cearfn_evidence import popularity_partition
from run_stratified_evaluation import context_bins, popularity_strata


HERE = Path(__file__).resolve().parent
SEEDS = (42, 123, 456)
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


def _hit(rank: np.ndarray, cutoff: int = 20) -> np.ndarray:
    rank = np.asarray(rank)
    return (rank > 0) & (rank <= cutoff)


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def decompose(
        memory_rank: np.ndarray,
        neural_rank: np.ndarray,
        fused_rank: np.ndarray,
        mask: np.ndarray,
        union_memory_top120: np.ndarray | None = None,
        selected_memory_top120: np.ndarray | None = None,
) -> dict:
    memory = _hit(memory_rank)[mask]
    neural = _hit(neural_rank)[mask]
    fused = _hit(fused_rank)[mask]

    rescue = ~memory & fused
    damage = memory & ~fused
    memory_miss = ~memory
    neural_only_region = ~memory & neural
    categories = {
        "both_hit": memory & neural,
        "memory_only": memory & ~neural,
        "neural_only": ~memory & neural,
        "neither": ~memory & ~neural,
    }

    result = {
        "n_queries": int(len(memory)),
        "memory_recall@20": _safe_mean(memory),
        "neural_recall@20": _safe_mean(neural),
        "fused_recall@20": _safe_mean(fused),
        "fusion_minus_memory": _safe_mean(fused) - _safe_mean(memory),
        "rescue_rate": _safe_mean(rescue),
        "damage_rate": _safe_mean(damage),
        "net_rescue_rate": _safe_mean(rescue) - _safe_mean(damage),
        "rescue_given_memory_miss": (
            float(rescue.sum() / memory_miss.sum()) if memory_miss.any() else 0.0
        ),
        "neural_hit_given_memory_miss": (
            float((memory_miss & neural).sum() / memory_miss.sum())
            if memory_miss.any() else 0.0
        ),
        "fusion_capture_given_neural_only_region": (
            float((neural_only_region & fused).sum() / neural_only_region.sum())
            if neural_only_region.any() else 0.0
        ),
        "outcome_regions": {},
    }
    if union_memory_top120 is not None and selected_memory_top120 is not None:
        union120 = np.asarray(union_memory_top120, dtype=bool)[mask]
        selected120 = np.asarray(selected_memory_top120, dtype=bool)[mask]
        rescue_mechanisms = {
            "outside_all_memory_top120": rescue & ~union120,
            "only_in_unselected_memory_component_top120": (
                rescue & union120 & ~selected120
            ),
            "promoted_from_selected_memory_rank21_120": (
                rescue & selected120
            ),
        }
        partition = np.zeros(len(rescue), dtype=bool)
        result["rescue_mechanisms"] = {}
        for name, region in rescue_mechanisms.items():
            partition |= region
            count = int(region.sum())
            result["rescue_mechanisms"][name] = {
                "n_queries": count,
                "query_rate": _safe_mean(region),
                "share_of_rescues": (
                    float(count / rescue.sum()) if rescue.any() else 0.0
                ),
            }
        if not np.array_equal(partition, rescue):
            raise AssertionError("rescue mechanism partition failed")
    for name, region in categories.items():
        result["outcome_regions"][name] = {
            "n_queries": int(region.sum()),
            "query_share": _safe_mean(region),
            "fused_recall@20": (
                float(fused[region].mean()) if region.any() else 0.0
            ),
        }

    if not np.isclose(result["fusion_minus_memory"],
                      result["net_rescue_rate"], atol=1e-12):
        raise AssertionError("rescue-damage identity failed")
    return result


def aggregate_seed_blocks(blocks: dict[str, dict]) -> dict:
    scalar_keys = [
        "memory_recall@20", "neural_recall@20", "fused_recall@20",
        "fusion_minus_memory", "rescue_rate", "damage_rate",
        "net_rescue_rate", "rescue_given_memory_miss",
        "neural_hit_given_memory_miss",
        "fusion_capture_given_neural_only_region",
    ]
    summary = {"n_queries": next(iter(blocks.values()))["n_queries"]}
    for key in scalar_keys:
        values = np.asarray([blocks[str(seed)][key] for seed in SEEDS])
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    summary["outcome_regions"] = {}
    for region in ("both_hit", "memory_only", "neural_only", "neither"):
        summary["outcome_regions"][region] = {}
        for key in ("n_queries", "query_share", "fused_recall@20"):
            values = np.asarray(
                [blocks[str(seed)]["outcome_regions"][region][key]
                 for seed in SEEDS],
                dtype=np.float64,
            )
            summary["outcome_regions"][region][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
    if "rescue_mechanisms" in next(iter(blocks.values())):
        summary["rescue_mechanisms"] = {}
        mechanisms = next(iter(blocks.values()))["rescue_mechanisms"]
        for mechanism in mechanisms:
            summary["rescue_mechanisms"][mechanism] = {}
            for key in ("n_queries", "query_rate", "share_of_rescues"):
                values = np.asarray(
                    [blocks[str(seed)]["rescue_mechanisms"][mechanism][key]
                     for seed in SEEDS],
                    dtype=np.float64,
                )
                summary["rescue_mechanisms"][mechanism][key] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
            # Equal query counts make this the pooled share across all three
            # seed-level prediction vectors; retain seed dispersion as `std`.
            summary["rescue_mechanisms"][mechanism][
                "share_of_rescues"
            ]["mean"] = (
                summary["rescue_mechanisms"][mechanism]["query_rate"]["mean"]
                / summary["rescue_rate"]["mean"]
                if summary["rescue_rate"]["mean"] else 0.0
            )
    return {"per_seed": blocks, "summary": summary}


def memory_agreement(memory_arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Maximum top-5 vote count across transition/session/popularity lists."""
    components = [
        np.asarray(memory_arrays[name], dtype=np.int32)
        for name in ("transition", "session", "popularity")
    ]
    agreement = np.zeros(len(components[0]), dtype=np.int8)
    for row in range(len(agreement)):
        votes: dict[int, int] = {}
        for ranking in components:
            for item in ranking[row, :5]:
                item = int(item)
                if item > 0:
                    votes[item] = votes.get(item, 0) + 1
        agreement[row] = max(votes.values()) if votes else 0
    return agreement


def transition_branching(
        data: dict, keys: list[str], exclude_seen: bool
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Effective outgoing branch count of the final context item.

    The effective count is exp(Shannon entropy) over the weighted outgoing
    transition distribution retained by the locked CEARF index.  Lower values
    indicate a clearer/local transition; higher values indicate branching.
    """
    index = cearf.CEARFIndex(
        data["train_sessions"], data["n_items"],
        cearf.CEARFConfig(exclude_seen=exclude_seen),
    )
    def metrics(queries: dict, query_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
        effective = np.full(len(query_keys), np.nan, dtype=np.float64)
        top1_share = np.full(len(query_keys), np.nan, dtype=np.float64)
        for row, uid in enumerate(query_keys):
            context = queries[uid].get("context", ())
            outgoing = (
                index.transition.get(int(context[-1]), {}) if context else {}
            )
            if not outgoing:
                continue
            weights = np.asarray(list(outgoing.values()), dtype=np.float64)
            probabilities = weights / weights.sum()
            entropy = -float(np.sum(probabilities * np.log(probabilities)))
            effective[row] = float(np.exp(entropy))
            top1_share[row] = float(probabilities.max())
        return effective, top1_share

    valid_keys = sorted(data["valid_queries"])
    valid_effective, valid_top1_share = metrics(
        data["valid_queries"], valid_keys)
    effective, top1_share = metrics(data["test_queries"], keys)
    valid_supported = np.isfinite(valid_effective)
    if not valid_supported.any():
        raise ValueError("validation set has no supported outgoing transitions")
    q1, q2 = np.quantile(
        valid_effective[valid_supported], [1 / 3, 2 / 3], method="nearest")
    labels = np.full(len(keys), "unsupported", dtype=object)
    supported = np.isfinite(effective)
    labels[supported & (effective <= q1)] = "low"
    labels[supported & (effective > q1) & (effective <= q2)] = "mid"
    labels[supported & (effective > q2)] = "high"
    c1, c2 = np.quantile(
        valid_top1_share[np.isfinite(valid_top1_share)],
        [1 / 3, 2 / 3],
        method="nearest",
    )
    certainty_labels = np.full(len(keys), "unsupported", dtype=object)
    certainty_labels[supported & (top1_share <= c1)] = "uncertain"
    certainty_labels[
        supported & (top1_share > c1) & (top1_share <= c2)
    ] = "mid"
    certainty_labels[supported & (top1_share > c2)] = "clear"
    metadata = {
        "definition": (
            "exp(Shannon entropy) over the final context item's weighted "
            "outgoing transitions in the locked CEARF index"
        ),
        "threshold_source": (
            "tercile boundaries frozen on supported validation contexts and "
            "then applied unchanged to test"
        ),
        "clear_max_effective_branches": float(q1),
        "mid_max_effective_branches": float(q2),
        "mean_effective_branches_supported_test": float(
            np.nanmean(effective)),
        "mean_top1_transition_share_supported_test": float(
            np.nanmean(top1_share)),
        "unsupported_test_share": float((~supported).mean()),
        "uncertain_max_top1_share": float(c1),
        "mid_max_top1_share": float(c2),
    }
    return labels, certainty_labels, metadata


def main() -> None:
    output = {
        "protocol": {
            "decision_status": "locked test audit; no test-time reselection",
            "cutoff": 20,
            "rescue": "memory misses target and selected fusion hits target",
            "damage": "memory hits target and selected fusion misses target",
            "identity": "fusion-memory = rescue-damage",
            "query_order": "sorted test-query identifiers, matching v2 construction",
            "popularity": "training-frequency item-rank 20/60/20 split",
            "context_length": "per-domain test-context terciles",
            "router_context_length": "locked router buckets: <=2, 3--7, >7",
            "router_last_item_popularity": (
                "head contains the most frequent training items accounting "
                "for 80% of interactions; all remaining items are tail"
            ),
            "memory_agreement": (
                "low=max top-5 vote count <=1; high=max vote count >=2 "
                "across transition/session/popularity"
            ),
            "transition_branching": (
                "per-domain terciles of effective outgoing branches for the "
                "final context item, frozen on validation and applied to test; "
                "zero-outgoing contexts are reported as unsupported"
            ),
            "rescue_mechanisms": (
                "every memory-miss@20 rescue is partitioned into target absent "
                "from the union of all three memory top-120 lists, target found "
                "only by an unselected component, or promotion from selected "
                "memory rank 21--120"
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
            [len(data["test_queries"][uid]["context"]) for uid in keys],
            dtype=np.int32,
        )
        pop_map, pop_meta = popularity_strata(
            Counter(data["item_freq"]), data["n_items"])
        pop_labels = np.asarray(
            [pop_map.get(int(target), "tail") for target in targets],
            dtype=object,
        )
        length_labels_list, length_meta = context_bins(lengths)
        length_labels = np.asarray(length_labels_list, dtype=object)
        router_length_labels = np.full(len(keys), "mid", dtype=object)
        router_length_labels[lengths <= 2] = "short"
        router_length_labels[lengths > 7] = "long"
        router_head, _, router_head_size = popularity_partition(
            Counter(data["item_freq"]), data["n_items"])
        router_head_set = set(int(item) for item in router_head)
        last_items = np.asarray(
            [int(data["test_queries"][uid]["context"][-1])
             if data["test_queries"][uid].get("context") else 0
             for uid in keys],
            dtype=np.int32,
        )
        last_item_pop_labels = np.asarray(
            ["head" if int(item) in router_head_set else "tail"
             for item in last_items],
            dtype=object,
        )
        branch_labels, certainty_labels, branch_meta = transition_branching(
            data, keys, exclude_seen=(domain != "Diginetica"))
        memory_path = (HERE / "cearfn_v2_nested_artifacts" /
                       f"{spec['slug']}_nested_test_memory.npz")
        with np.load(memory_path, allow_pickle=True) as saved:
            memory_keys = [str(value) for value in saved["keys"]]
            memory_fingerprint = str(saved["fingerprint"].item())
            if memory_keys != keys or memory_fingerprint != query_fingerprint(
                    data["test_queries"]):
                raise ValueError(f"{memory_path}: query alignment mismatch")
            memory_arrays = {
                name: np.asarray(saved[name], dtype=np.int32).copy()
                for name in ("transition", "session", "popularity", "selected")
            }
            agreement = memory_agreement(memory_arrays)
        component_presence = {
            name: np.any(array == targets[:, None], axis=1)
            for name, array in memory_arrays.items()
        }
        union_memory_top120 = (
            component_presence["transition"]
            | component_presence["session"]
            | component_presence["popularity"]
        )
        selected_memory_top120 = component_presence["selected"]

        ranks_by_seed = {}
        expected_fingerprint = query_fingerprint(data["test_queries"])
        for seed in SEEDS:
            path = (HERE / "cearfn_v2_nested_artifacts" /
                    f"{spec['slug']}_v2_seed{seed}_ranks.npz")
            with np.load(path, allow_pickle=True) as saved:
                artifact_fingerprint = str(saved["test_fingerprint"].item())
                if artifact_fingerprint != expected_fingerprint:
                    raise ValueError(f"{path}: query fingerprint mismatch")
                ranks_by_seed[seed] = {
                    key: np.asarray(saved[key], dtype=np.int32)
                    for key in ("memory_rank", "neural_rank", "selected_rank")
                }
                if any(len(v) != len(keys) for v in ranks_by_seed[seed].values()):
                    raise ValueError(f"{path}: rank/query length mismatch")

        domain_block = {
            "metadata": {
                "n_queries": len(keys),
                "query_fingerprint_sha256": expected_fingerprint,
                "popularity_split": pop_meta,
                "context_cutoffs": length_meta,
                "router_length_cutoffs": {
                    "short": "<=2",
                    "mid": "3--7",
                    "long": ">7",
                },
                "router_last_item_head_size": router_head_size,
                "transition_branching": branch_meta,
            },
            "overall": {},
            "popularity_strata": {},
            "context_length_strata": {},
            "router_context_length_strata": {},
            "router_last_item_popularity_strata": {},
            "memory_agreement_strata": {},
            "transition_branching_strata": {},
            "transition_certainty_strata": {},
        }
        masks = {
            "overall": np.ones(len(keys), dtype=bool),
            **{f"popularity::{name}": pop_labels == name
               for name in ("head", "torso", "tail")},
            **{f"context::{name}": length_labels == name
               for name in ("short", "medium", "long")},
            **{f"router_context::{name}": router_length_labels == name
               for name in ("short", "mid", "long")},
            **{f"router_last_pop::{name}": last_item_pop_labels == name
               for name in ("head", "tail")},
            "agreement::low": agreement <= 1,
            "agreement::high": agreement >= 2,
            **{f"branching::{name}": branch_labels == name
               for name in ("unsupported", "low", "mid", "high")},
            **{f"certainty::{name}": certainty_labels == name
               for name in ("unsupported", "uncertain", "mid", "clear")},
        }
        for label, mask in masks.items():
            seed_blocks = {}
            for seed in SEEDS:
                ranks = ranks_by_seed[seed]
                seed_blocks[str(seed)] = decompose(
                    ranks["memory_rank"], ranks["neural_rank"],
                    ranks["selected_rank"], mask,
                    union_memory_top120=union_memory_top120,
                    selected_memory_top120=selected_memory_top120)
            aggregated = aggregate_seed_blocks(seed_blocks)
            if label == "overall":
                domain_block["overall"] = aggregated
            elif label.startswith("popularity::"):
                domain_block["popularity_strata"][label.split("::", 1)[1]] = aggregated
            elif label.startswith("context::"):
                domain_block["context_length_strata"][label.split("::", 1)[1]] = aggregated
            elif label.startswith("router_context::"):
                domain_block["router_context_length_strata"][
                    label.split("::", 1)[1]
                ] = aggregated
            elif label.startswith("router_last_pop::"):
                domain_block["router_last_item_popularity_strata"][
                    label.split("::", 1)[1]
                ] = aggregated
            elif label.startswith("agreement::"):
                domain_block["memory_agreement_strata"][label.split("::", 1)[1]] = aggregated
            elif label.startswith("branching::"):
                domain_block["transition_branching_strata"][label.split("::", 1)[1]] = aggregated
            else:
                domain_block["transition_certainty_strata"][label.split("::", 1)[1]] = aggregated
        output["domains"][domain] = domain_block

    path = HERE / "evidence_necessity_audit.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
