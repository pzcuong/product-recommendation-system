#!/usr/bin/env python3
"""Post-hoc fusion-operator control for training-calibrated CEARF-N.

This script changes exactly one object: the operator that combines the two
already-frozen experts. It compares each operator under two allocation laws:

* the already-frozen primary query-wise ``beta_q``; and
* a parameter-free equal allocation ``beta=.5``.

For each allocation it contrasts weighted reciprocal-rank fusion (the primary
CEARF-N operator) with per-query min--max normalized CombSUM over the same
top-120 candidate union. The equal-allocation pair prevents a score operator
from being judged only under an allocator trained on reciprocal-rank evidence.

The control never fits or selects a parameter.  Before targets are read, it:

1. reconstructs CEARF's final native RRF score from persisted component ranks;
2. recovers PASGR cosine retrieval scores from a frozen checkpoint;
3. asserts that the recovered full-catalogue top-120 item IDs exactly equal
   the persisted prediction cache;
4. freezes the two top-20 rankings and writes an input/operator manifest.

Only then are target labels loaded to compute Recall, nDCG and the locked
utility ``0.5 * R@6 + 0.5 * R@20``.

Important scope: normalized CombSUM is evaluated on the union of the experts'
full-catalogue top-120 outputs (at most 240 candidates).  Items absent from one
expert receive zero from that expert.  This is not a full-catalogue score
normalization because scores outside each persisted top-120 are unavailable.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import cearf
import loaders
import pasgr
from dynamic_beta import fuse_with_dynamic_beta
from run_cearfn_evidence import (
    metrics_from_ranks,
    query_fingerprint,
    ranks_at_20,
    targets_for,
)
from run_cearfn_v2 import REPEAT_PROTOCOL_DOMAINS


HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")
SEEDS = (42, 123, 456)
PROTOCOL = "fusion-operator-control-v2-dynamic-and-equal-allocation"
RRF_CONSTANT = 20.0


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _positive_items(row: np.ndarray) -> list[int]:
    return [int(item) for item in row if int(item) > 0]


def reconstruct_cearf_final_scores(
    memory_arrays: Mapping[str, np.ndarray],
    queries: Mapping[str, Mapping[str, Sequence[int]]],
    keys: Sequence[str],
    profiles: Mapping[str, Sequence[float]],
    config: cearf.CEARFConfig,
) -> tuple[np.ndarray, dict]:
    """Reconstruct CEARF's final expert score aligned to ``selected``.

    CEARF first converts its transition, neighbour-session and popularity
    components to ranks.  Its final expert score is therefore the weighted
    component RRF score with the consensus multiplier from
    :meth:`cearf.CEARFIndex.fuse_rankings`.  Popularity backfill items that
    were not scored by an admitted component have native score zero.
    """
    required = {"transition", "session", "popularity", "selected"}
    missing = required - set(memory_arrays)
    if missing:
        raise ValueError(f"memory artifact misses fields: {sorted(missing)}")
    selected = np.asarray(memory_arrays["selected"], dtype=np.int32)
    if len(selected) != len(keys):
        raise ValueError("memory rows and query keys differ")
    scores = np.zeros(selected.shape, dtype=np.float32)
    fallback_items = 0
    rows_with_fallback = 0

    for row, uid0 in enumerate(keys):
        uid = str(uid0)
        if uid not in queries:
            raise KeyError(f"query key not found: {uid}")
        regime = (
            "short"
            if len(queries[uid].get("context", ())) <= config.short_context
            else "long"
        )
        if regime not in profiles:
            raise ValueError(f"missing CEARF profile for {regime}")
        profile = tuple(float(value) for value in profiles[regime])
        if len(profile) != 3:
            raise ValueError("CEARF profile must contain three weights")

        fused: dict[int, float] = {}
        votes: dict[int, int] = {}
        for name, weight in zip(
            ("transition", "session", "popularity"), profile
        ):
            if weight <= 0.0:
                continue
            for rank, item0 in enumerate(memory_arrays[name][row], 1):
                item = int(item0)
                if item <= 0:
                    continue
                fused[item] = (
                    fused.get(item, 0.0)
                    + weight / (config.rrf_constant + rank)
                )
                votes[item] = votes.get(item, 0) + 1
        for item, count in votes.items():
            if count >= 2:
                fused[item] *= (
                    1.0 + config.consensus_bonus * (count - 1)
                )

        # CEARF._rank uses heapq.nlargest on (score, item), hence larger item
        # IDs break exact score ties.  Verify the persisted positive-score
        # prefix before using reconstructed values.
        native_order = [
            item
            for item, _ in sorted(
                fused.items(),
                key=lambda pair: (pair[1], pair[0]),
                reverse=True,
            )
        ][: selected.shape[1]]
        persisted = _positive_items(selected[row])
        persisted_scored = [
            item for item in persisted if fused.get(item, 0.0) > 0.0
        ]
        if persisted_scored != native_order[: len(persisted_scored)]:
            raise AssertionError(
                f"CEARF score reconstruction mismatch at row {row} ({uid})"
            )
        row_fallback = 0
        for column, item in enumerate(persisted):
            value = float(fused.get(item, 0.0))
            scores[row, column] = value
            if value <= 0.0:
                row_fallback += 1
        fallback_items += row_fallback
        rows_with_fallback += int(row_fallback > 0)

    return scores, {
        "rows": int(len(keys)),
        "width": int(selected.shape[1]),
        "fallback_items_with_native_score_zero": int(fallback_items),
        "rows_with_fallback": int(rows_with_fallback),
        "verified_native_order": True,
        "score_definition": (
            "weighted component RRF with CEARF consensus multiplier; "
            "unscored popularity backfill=0"
        ),
    }


def minmax_normalize_rows(
    items: np.ndarray,
    scores: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Normalize each expert independently; missing/padded items remain zero."""
    items = np.asarray(items)
    scores = np.asarray(scores, dtype=np.float32)
    if items.shape != scores.shape:
        raise ValueError("items and scores must have identical shapes")
    output = np.zeros(scores.shape, dtype=np.float32)
    for row in range(len(scores)):
        present = (items[row] > 0) & np.isfinite(scores[row])
        if not present.any():
            continue
        values = scores[row, present]
        low = float(values.min())
        high = float(values.max())
        if high - low <= epsilon:
            # A constant score vector contains no within-expert preference.
            # Returning zero is the conservative, deterministic convention.
            continue
        output[row, present] = (values - low) / (high - low)
    return output


def normalized_combsum(
    memory_items: np.ndarray,
    memory_scores: np.ndarray,
    neural_items: np.ndarray,
    neural_scores: np.ndarray,
    betas: np.ndarray,
    topk: int = 20,
) -> np.ndarray:
    """Fuse independently normalized scores over the top-120 candidate union."""
    if not (
        memory_items.shape == memory_scores.shape
        and neural_items.shape == neural_scores.shape
    ):
        raise ValueError("each expert's items/scores must have equal shapes")
    if not (
        len(memory_items) == len(neural_items) == len(betas)
    ):
        raise ValueError("fusion inputs must share row count")
    memory_normalized = minmax_normalize_rows(
        memory_items, memory_scores)
    neural_normalized = minmax_normalize_rows(
        neural_items, neural_scores)
    return fuse_normalized_combsum(
        memory_items,
        memory_normalized,
        neural_items,
        neural_normalized,
        betas,
        topk,
    )


def fuse_normalized_combsum(
    memory_items: np.ndarray,
    memory_normalized: np.ndarray,
    neural_items: np.ndarray,
    neural_normalized: np.ndarray,
    betas: np.ndarray,
    topk: int = 20,
) -> np.ndarray:
    """Fuse pre-normalized expert scores under a supplied allocation."""
    if not (
        memory_items.shape == memory_normalized.shape
        and neural_items.shape == neural_normalized.shape
    ):
        raise ValueError("normalized score arrays must align with item arrays")
    if not (
        len(memory_items) == len(neural_items) == len(betas)
    ):
        raise ValueError("fusion inputs must share row count")
    output = np.zeros((len(betas), topk), dtype=np.int32)

    for row, beta0 in enumerate(betas):
        beta = float(beta0)
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta outside [0,1] at row {row}: {beta}")
        combined: dict[int, float] = {}
        for item0, value0 in zip(
            memory_items[row], memory_normalized[row]
        ):
            item = int(item0)
            if item > 0:
                combined[item] = (
                    combined.get(item, 0.0)
                    + (1.0 - beta) * float(value0)
                )
        for item0, value0 in zip(
            neural_items[row], neural_normalized[row]
        ):
            item = int(item0)
            if item > 0:
                combined[item] = (
                    combined.get(item, 0.0)
                    + beta * float(value0)
                )
        ranking = [
            item
            for item, _ in sorted(
                combined.items(), key=lambda pair: (-pair[1], pair[0])
            )[:topk]
        ]
        output[row, : len(ranking)] = ranking
    return output


def _model_from_checkpoint(
    checkpoint: Path,
    n_items: int,
    device: str,
) -> pasgr.PASGRModel:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = pasgr.PASGRConfig(**saved["config"])
    model = pasgr.PASGRModel(
        np.zeros((n_items, config.dim), dtype=np.float32),
        config,
    )
    model.load_state_dict(saved["state_dict"])
    return model.to(torch.device(device)).eval()


def auto_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def recover_pasgr_topk_scores(
    model: torch.nn.Module,
    queries: Mapping[str, Mapping[str, Sequence[int]]],
    keys: Sequence[str],
    persisted_rankings: np.ndarray,
    n_items: int,
    exclude_seen: bool,
    batch_size: int = 256,
) -> np.ndarray:
    """Recover cosine scores and verify every persisted top-k item ID."""
    if len(keys) != len(persisted_rankings):
        raise ValueError("PASGR query keys and ranking rows differ")
    device = next(model.parameters()).device
    max_seq = int(model.config.max_seq)
    width = int(persisted_rankings.shape[1])
    catalog = F.normalize(model.item.weight, dim=-1)
    recovered = np.empty(persisted_rankings.shape, dtype=np.float32)

    for start in range(0, len(keys), batch_size):
        batch_keys = [str(uid) for uid in keys[start:start + batch_size]]
        sequences = [
            [
                int(item)
                for item in queries[uid].get("context", ())
                if 0 < int(item) < n_items
            ][-max_seq:]
            for uid in batch_keys
        ]
        max_len = max([len(sequence) for sequence in sequences] + [1])
        contexts = torch.zeros(
            len(sequences), max_len, dtype=torch.long, device=device)
        lengths = torch.ones(
            len(sequences), dtype=torch.long, device=device)
        for row, sequence in enumerate(sequences):
            if sequence:
                contexts[row, : len(sequence)] = torch.as_tensor(
                    sequence, dtype=torch.long, device=device)
                lengths[row] = len(sequence)
        query = model.encode(contexts, lengths)
        full_scores = query @ catalog.T
        full_scores[:, 0] = -torch.inf
        if exclude_seen:
            for row, sequence in enumerate(sequences):
                if sequence:
                    full_scores[
                        row,
                        torch.as_tensor(
                            list(set(sequence)),
                            dtype=torch.long,
                            device=device,
                        ),
                    ] = -torch.inf
        values, indices = torch.topk(full_scores, k=width, dim=-1)
        actual = indices.cpu().numpy().astype(np.int32, copy=False)
        expected = np.asarray(
            persisted_rankings[start:start + len(batch_keys)],
            dtype=np.int32,
        )
        if not np.array_equal(actual, expected):
            difference = np.argwhere(actual != expected)[0]
            local_row, column = map(int, difference)
            global_row = start + local_row
            raise AssertionError(
                "frozen checkpoint does not reproduce persisted PASGR "
                f"top-{width}: query={keys[global_row]!s}, "
                f"rank={column + 1}, expected={expected[local_row, column]}, "
                f"actual={actual[local_row, column]}"
            )
        recovered[start:start + len(batch_keys)] = (
            values.cpu().numpy().astype(np.float32, copy=False)
        )
    return recovered


def _score_cache_metadata(
    checkpoint_sha256: str,
    queries: Mapping[str, Mapping[str, Sequence[int]]],
    rankings: np.ndarray,
    candidate_width: int,
    exclude_seen: bool,
    compute_device: str,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "checkpoint_sha256": checkpoint_sha256,
        "query_fingerprint": query_fingerprint(queries),
        "ranking_sha256": array_sha256(rankings),
        "candidate_width": int(candidate_width),
        "exclude_seen": bool(exclude_seen),
        "compute_device": str(compute_device),
        "score_definition": (
            "PASGR normalized-query dot normalized-item embedding "
            "(cosine retrieval score; no softmax)"
        ),
        "full_catalogue_topk_ids_verified": True,
    }


def load_score_cache(
    path: Path,
    expected: Mapping[str, object],
    keys: Sequence[str],
    rankings: np.ndarray,
) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        required = {"keys", "items", "scores", *expected.keys()}
        if not required.issubset(saved.files):
            return None
        for name, value in expected.items():
            if str(saved[name].item()) != str(value):
                return None
        if not np.array_equal(
            np.asarray(saved["keys"], dtype=str),
            np.asarray(keys, dtype=str),
        ):
            return None
        if not np.array_equal(saved["items"], rankings):
            return None
        scores = saved["scores"]
        if scores.shape != rankings.shape:
            return None
        return scores.astype(np.float32)


def save_score_cache(
    path: Path,
    metadata: Mapping[str, object],
    keys: Sequence[str],
    rankings: np.ndarray,
    scores: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "keys": np.asarray(keys, dtype=str),
        "items": np.asarray(rankings, dtype=np.int32),
        "scores": np.asarray(scores, dtype=np.float32),
    }
    payload.update({
        name: np.asarray(value)
        for name, value in metadata.items()
    })
    np.savez_compressed(path, **payload)


def resolve_checkpoint(
    artifact_dir: Path,
    legacy_artifact_dir: Path,
    inference_checkpoint_dir: Path,
    domain: str,
    seed: int,
    split: str,
) -> Path:
    tag = "valid" if split == "validation" else "test"
    candidates = [
        artifact_dir / domain.lower() / "checkpoints"
        / f"seed{seed}_{tag}.pt",
        legacy_artifact_dir / domain.lower() / "checkpoints"
        / f"seed{seed}_{tag}.pt",
    ]
    if seed == 42:
        legacy_tag = "validation" if split == "validation" else "final"
        candidates.append(
            inference_checkpoint_dir
            / f"{domain.lower()}_{legacy_tag}_seed{seed}.pt"
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no frozen PASGR checkpoint for {domain} seed={seed} split={split}; "
        f"checked: {[str(path) for path in candidates]}"
    )


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def _load_npz_fields(
    path: Path,
    fields: Sequence[str],
) -> dict[str, np.ndarray]:
    """Materialize only declared fields from an otherwise mixed artifact."""
    with np.load(path, allow_pickle=False) as saved:
        missing = set(fields) - set(saved.files)
        if missing:
            raise ValueError(
                f"{path} misses fields: {sorted(missing)}")
        return {name: saved[name] for name in fields}


def _target_free_queries(
    queries: Mapping[str, Mapping[str, Sequence[int]]],
    keys: Sequence[str],
) -> dict[str, dict[str, list[int]]]:
    return {
        str(uid): {
            "context": [
                int(item)
                for item in queries[str(uid)].get("context", ())
            ]
        }
        for uid in keys
    }


def evaluate_top20(
    ranking: np.ndarray,
    targets: np.ndarray,
) -> tuple[dict, np.ndarray]:
    ranks = ranks_at_20(ranking, targets)
    metrics = metrics_from_ranks(ranks)
    metrics["utility"] = float(
        0.5 * metrics["recall@6"] + 0.5 * metrics["recall@20"]
    )
    return metrics, ranks


def _mean_std(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": [float(value) for value in array],
    }


def aggregate_domain(runs: Sequence[dict]) -> dict:
    metrics = (
        "recall@6", "ndcg@6",
        "recall@10", "ndcg@10",
        "recall@20", "ndcg@20",
        "utility",
    )
    output = {}
    for method in (
        "weighted_rrf",
        "normalized_combsum",
        "fixed_05_weighted_rrf",
        "fixed_05_normalized_combsum",
    ):
        output[method] = {
            metric: _mean_std([
                float(run["test"]["metrics"][method][metric])
                for run in runs
            ])
            for metric in metrics
        }
    output["normalized_combsum_minus_weighted_rrf"] = {
        metric: _mean_std([
            float(
                run["test"]["metrics"]["normalized_combsum"][metric]
                - run["test"]["metrics"]["weighted_rrf"][metric]
            )
            for run in runs
        ])
        for metric in metrics
    }
    return output


def _parse_profiles(memory_arrays: Mapping[str, np.ndarray]) -> dict:
    if "profiles" not in memory_arrays:
        raise ValueError("memory artifact does not contain frozen profiles")
    return json.loads(str(memory_arrays["profiles"].item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument(
        "--primary-results", type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_results.json")
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_artifacts")
    parser.add_argument(
        "--legacy-artifact-dir", type=Path,
        default=HERE / "dynamic_beta_artifacts")
    parser.add_argument(
        "--inference-checkpoint-dir", type=Path,
        default=HERE / "inference_benchmark_checkpoints")
    parser.add_argument(
        "--control-artifact-dir", type=Path,
        default=HERE / "dynamic_beta_fusion_operator_artifacts")
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "dynamic_beta_fusion_operator_control.json")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"),
        default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    primary = json.loads(args.primary_results.read_text())
    requested_seeds = set(args.seeds)
    compute_device = auto_device(args.device)
    args.control_artifact_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "protocol": PROTOCOL,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
        "score_recovery_device": compute_device,
        "primary_results": str(args.primary_results),
        "primary_results_sha256": file_sha256(args.primary_results),
        "operator_control": {
            "allocations": {
                "dynamic": (
                    "already-frozen primary per-query dynamic beta; "
                    "no refit, search or selection"
                ),
                "fixed_05": (
                    "parameter-free beta=.5 for every query"
                ),
            },
            "primary": (
                "(1-beta_q)/(20+rank_memory) + "
                "beta_q/(20+rank_neural)"
            ),
            "control": (
                "(1-beta_q)*minmax(CEARF_final_score) + "
                "beta_q*minmax(PASGR_cosine_score)"
            ),
            "normalization": (
                "independent per query and expert over its persisted top-120; "
                "zero-span expert maps to all zeros"
            ),
            "candidate_union": (
                "union of the two full-catalogue top-120 expert outputs; "
                "an item absent from one expert receives zero from that expert"
            ),
            "tie_break": "ascending item ID after descending fused score",
            "target_labels_used_to_form_rankings": False,
            "parameters_fit_or_selected": False,
        },
        "limitations": [
            (
                "CombSUM normalization is over each expert's top-120, not all "
                "catalogue items; scores outside top-120 were not persisted."
            ),
            (
                "CEARF is itself a rank-fused memory expert. Its final native "
                "score is reconstructed from component ranks and the frozen "
                "profile; this control does not expose pre-ranking component "
                "scores."
            ),
            (
                "Neither allocation is optimized for CombSUM. The dynamic "
                "pair isolates a frozen-allocator perturbation; the beta=.5 "
                "pair is allocation-neutral but still does not give CombSUM "
                "an operator-specific learned allocator."
            ),
        ],
        "domains": {},
    }

    for domain in args.domains:
        if domain not in primary:
            raise ValueError(f"{domain} missing from primary results")
        data = loaders.ALL_LOADERS[domain]()
        n_items = int(data["n_items"])
        exclude_seen = domain not in REPEAT_PROTOCOL_DOMAINS
        config = cearf.CEARFConfig(exclude_seen=exclude_seen)
        completed = {
            int(run["seed"]): run for run in primary[domain].get("runs", [])
        }
        missing_seeds = requested_seeds - set(completed)
        if missing_seeds:
            raise ValueError(
                f"{domain} is incomplete; missing seeds "
                f"{sorted(missing_seeds)}"
            )
        domain_runs = []
        domain_dir = args.artifact_dir / domain.lower()
        control_domain_dir = (
            args.control_artifact_dir / domain.lower())
        control_domain_dir.mkdir(parents=True, exist_ok=True)

        memory_by_split = {}
        for split, tag in (("validation", "valid"), ("test", "test")):
            memory_path = domain_dir / f"{tag}_memory.npz"
            memory_arrays = _load_npz_arrays(memory_path)
            profiles = _parse_profiles(memory_arrays)
            queries = (
                data["valid_queries"]
                if split == "validation"
                else data["test_queries"]
            )
            keys = [str(value) for value in memory_arrays["keys"]]
            target_free = _target_free_queries(queries, keys)
            memory_scores, memory_report = reconstruct_cearf_final_scores(
                memory_arrays,
                target_free,
                keys,
                profiles,
                config,
            )
            memory_by_split[split] = {
                "path": memory_path,
                "path_sha256": file_sha256(memory_path),
                "arrays": memory_arrays,
                "keys": keys,
                "target_free_queries": target_free,
                "scores": memory_scores,
                "report": memory_report,
            }

        for seed in sorted(requested_seeds):
            rank_path = (
                domain_dir / f"seed{seed}_dynamic_beta_ranks.npz")
            # The primary artifact also contains target-derived rank vectors.
            # Do not materialize those fields while forming control rankings.
            rank_arrays = _load_npz_fields(
                rank_path,
                (
                    "valid_keys",
                    "valid_dynamic_beta",
                    "valid_dynamic_top20",
                    "valid_fixed_05_top20",
                    "test_keys",
                    "test_dynamic_beta",
                    "test_dynamic_top20",
                    "test_fixed_05_top20",
                ),
            )
            split_rankings = {}
            split_inputs = {}

            for split, tag in (
                ("validation", "valid"),
                ("test", "test"),
            ):
                queries = (
                    data["valid_queries"]
                    if split == "validation"
                    else data["test_queries"]
                )
                memory_block = memory_by_split[split]
                memory_path = memory_block["path"]
                memory_arrays = memory_block["arrays"]
                keys = memory_block["keys"]
                neural_path = (
                    domain_dir / "predictions"
                    / f"seed{seed}_{tag}_top120.npz"
                )
                neural_arrays = _load_npz_arrays(neural_path)
                neural_keys = [
                    str(value) for value in neural_arrays["keys"]]
                rank_keys = [
                    str(value)
                    for value in rank_arrays[f"{tag}_keys"]
                ]
                if not (keys == neural_keys == rank_keys):
                    raise ValueError(
                        f"{domain} seed={seed} {split}: key mismatch")
                beta = np.asarray(
                    rank_arrays[f"{tag}_dynamic_beta"],
                    dtype=np.float32,
                )
                if len(beta) != len(keys):
                    raise ValueError("beta and query rows differ")
                target_free = memory_block["target_free_queries"]
                memory_scores = memory_block["scores"]
                memory_report = memory_block["report"]
                checkpoint = resolve_checkpoint(
                    args.artifact_dir,
                    args.legacy_artifact_dir,
                    args.inference_checkpoint_dir,
                    domain,
                    seed,
                    split,
                )
                checkpoint_hash = file_sha256(checkpoint)
                neural_rankings = np.asarray(
                    neural_arrays["rankings"], dtype=np.int32)
                cache_metadata = _score_cache_metadata(
                    checkpoint_hash,
                    target_free,
                    neural_rankings,
                    neural_rankings.shape[1],
                    exclude_seen,
                    compute_device,
                )
                score_cache = (
                    control_domain_dir
                    / f"seed{seed}_{tag}_pasgr_scores_top120.npz"
                )
                neural_scores = load_score_cache(
                    score_cache,
                    cache_metadata,
                    keys,
                    neural_rankings,
                )
                if neural_scores is None:
                    model = _model_from_checkpoint(
                        checkpoint, n_items, compute_device)
                    neural_scores = recover_pasgr_topk_scores(
                        model,
                        target_free,
                        keys,
                        neural_rankings,
                        n_items,
                        exclude_seen,
                        batch_size=args.batch_size,
                    )
                    save_score_cache(
                        score_cache,
                        cache_metadata,
                        keys,
                        neural_rankings,
                        neural_scores,
                    )
                    del model
                    gc.collect()

                weighted_rrf = fuse_with_dynamic_beta(
                    memory_arrays["selected"],
                    neural_rankings,
                    beta,
                    constant=RRF_CONSTANT,
                )
                persisted_primary = np.asarray(
                    rank_arrays[f"{tag}_dynamic_top20"],
                    dtype=np.int32,
                )
                if not np.array_equal(weighted_rrf, persisted_primary):
                    raise AssertionError(
                        f"{domain} seed={seed} {split}: recomputed primary "
                        "weighted RRF differs from persisted top-20"
                    )
                memory_items = np.asarray(
                    memory_arrays["selected"], dtype=np.int32)
                memory_normalized = minmax_normalize_rows(
                    memory_items, memory_scores)
                neural_normalized = minmax_normalize_rows(
                    neural_rankings, neural_scores)
                combsum = fuse_normalized_combsum(
                    memory_items,
                    memory_normalized,
                    neural_rankings,
                    neural_normalized,
                    beta,
                    topk=20,
                )
                fixed_05_beta = np.full(
                    len(keys), 0.5, dtype=np.float32)
                fixed_05_weighted_rrf = fuse_with_dynamic_beta(
                    memory_items,
                    neural_rankings,
                    fixed_05_beta,
                    constant=RRF_CONSTANT,
                )
                persisted_fixed_05 = np.asarray(
                    rank_arrays[f"{tag}_fixed_05_top20"],
                    dtype=np.int32,
                )
                if not np.array_equal(
                    fixed_05_weighted_rrf, persisted_fixed_05
                ):
                    raise AssertionError(
                        f"{domain} seed={seed} {split}: recomputed "
                        "beta=.5 weighted RRF differs from persisted top-20"
                    )
                fixed_05_combsum = fuse_normalized_combsum(
                    memory_items,
                    memory_normalized,
                    neural_rankings,
                    neural_normalized,
                    fixed_05_beta,
                    topk=20,
                )
                split_rankings[split] = {
                    "keys": np.asarray(keys, dtype=str),
                    "weighted_rrf": weighted_rrf,
                    "normalized_combsum": combsum,
                    "fixed_05_weighted_rrf": fixed_05_weighted_rrf,
                    "fixed_05_normalized_combsum": fixed_05_combsum,
                }
                split_inputs[split] = {
                    "query_count": int(len(keys)),
                    "memory_artifact": str(memory_path),
                    "memory_artifact_sha256": (
                        memory_block["path_sha256"]),
                    "neural_rank_artifact": str(neural_path),
                    "neural_rank_artifact_sha256": file_sha256(neural_path),
                    "primary_rank_artifact": str(rank_path),
                    "primary_rank_artifact_sha256": file_sha256(rank_path),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "score_recovery_device": compute_device,
                    "score_cache": str(score_cache),
                    "score_cache_sha256": file_sha256(score_cache),
                    "query_keys_sha256": array_sha256(
                        np.asarray(keys, dtype=str)),
                    "beta_sha256": array_sha256(beta),
                    "weighted_rrf_top20_sha256": array_sha256(
                        weighted_rrf),
                    "normalized_combsum_top20_sha256": array_sha256(
                        combsum),
                    "fixed_05_weighted_rrf_top20_sha256": array_sha256(
                        fixed_05_weighted_rrf),
                    "fixed_05_normalized_combsum_top20_sha256": array_sha256(
                        fixed_05_combsum),
                    "pasgr_full_catalogue_top120_exact_match": True,
                    "primary_rrf_exact_match": True,
                    "fixed_05_rrf_exact_match": True,
                    "cearf_score_reconstruction": memory_report,
                }

            manifest = {
                "protocol": PROTOCOL,
                "domain": domain,
                "seed": int(seed),
                "frozen_before_target_evaluation": True,
                "target_labels_used_to_form_rankings": False,
                "parameters_fit_or_selected": False,
                "rrf_constant": RRF_CONSTANT,
                "cearf_config": asdict(config),
                "operator_control": output["operator_control"],
                "limitations": output["limitations"],
                "inputs": split_inputs,
            }
            manifest_path = (
                control_domain_dir / f"seed{seed}_frozen_manifest.json")
            manifest_path.write_text(json.dumps(manifest, indent=2))

            rank_payload = {}
            evaluated = {}
            for split, tag in (
                ("validation", "valid"),
                ("test", "test"),
            ):
                rankings = split_rankings[split]
                queries = (
                    data["valid_queries"]
                    if split == "validation"
                    else data["test_queries"]
                )
                keys = [str(value) for value in rankings["keys"]]
                # This is the first direct target-array access in the control.
                targets = targets_for(keys, queries)
                evaluated[split] = {"metrics": {}}
                rank_payload[f"{tag}_keys"] = rankings["keys"]
                for method in (
                    "weighted_rrf",
                    "normalized_combsum",
                    "fixed_05_weighted_rrf",
                    "fixed_05_normalized_combsum",
                ):
                    metric, ranks = evaluate_top20(
                        rankings[method], targets)
                    evaluated[split]["metrics"][method] = metric
                    rank_payload[f"{tag}_{method}_top20"] = (
                        rankings[method].astype(np.int32))
                    rank_payload[f"{tag}_{method}_rank"] = (
                        ranks.astype(np.uint8))

            control_rank_path = (
                control_domain_dir
                / f"seed{seed}_fusion_operator_ranks.npz"
            )
            np.savez_compressed(control_rank_path, **rank_payload)
            run = {
                "seed": int(seed),
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "rank_artifact": str(control_rank_path),
                "rank_artifact_sha256": file_sha256(control_rank_path),
                **evaluated,
            }
            domain_runs.append(run)
            print(
                f"[FUSION-CONTROL] {domain} seed={seed}: "
                f"RRF R@20="
                f"{run['test']['metrics']['weighted_rrf']['recall@20']:.6f} "
                f"CombSUM R@20="
                f"{run['test']['metrics']['normalized_combsum']['recall@20']:.6f}",
                flush=True,
            )

        output["domains"][domain] = {
            "runs": domain_runs,
            "aggregate": aggregate_domain(domain_runs),
        }
        args.output.write_text(json.dumps(output, indent=2))

    args.output.write_text(json.dumps(output, indent=2))
    print(f"[FUSION-CONTROL] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
