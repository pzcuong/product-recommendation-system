#!/usr/bin/env python3
"""NARM expert-swap audit for training-only dynamic-beta CEARF-N.

This audit changes exactly one *system-level* expert choice: the PASGR neural
ranker used by the primary experiment is replaced by a fresh NARM refit.  The
CEARF memory profile, deterministic source-session OOF
split, continuous global beta objective, bounded context-only dynamic gate,
candidate width, RRF constant, and test protocol remain unchanged.

Protocol
--------
1. Reconstruct the primary deterministic OOF source-session split.
2. Train NARM from scratch on the inner-fit sessions for the epoch budget that
   was locked by the pre-existing NARM baseline run.
3. Fit the scalar and dynamic beta policies only from OOF training ranks.
4. Refit NARM on the complete training sessions for that locked budget and
   fingerprint the fit.
5. Write a frozen manifest before NARM test inference and test evaluation.
6. Predict target-free test ranks, then evaluate memory, NARM, global fusion,
   and dynamic fusion.

No validation or test target is passed to either beta fitter.  The upstream
NARM epoch budget was selected before this audit and is disclosed rather than
reselected.  On domains where PASGR uses metadata while NARM is ID-only, the
PASGR/NARM contrast is a system-sensitivity check, not metadata-matched
architecture attribution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch

import cearf
import loaders
from dynamic_beta import FEATURE_GROUPS, FEATURE_NAMES, fuse_with_dynamic_beta
from paper_models import build_model, model_logits
from run_cearfn_evidence import (
    popularity_partition,
    query_fingerprint,
    ranks_at_20,
    targets_for,
)
from run_dynamic_beta import (
    beta_summary,
    canonical_validation_sources,
    evaluate_ranking,
    fit_dynamic,
    fit_global,
    make_training_oof_split,
    session_fingerprint,
)
from run_paper_baselines import collate, train_fixed_epochs
from validation_protocol import hold_out_validation_targets


HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")
SEEDS = (42, 123, 456)
EXPERT_SWAP_PROTOCOL = "dynamic-beta-narm-expert-swap-v2-full-refit"
PRIMARY_COLUMNS = FEATURE_GROUPS["context"]
PRIMARY_FEATURE_NAMES = tuple(FEATURE_NAMES[i] for i in PRIMARY_COLUMNS)


def canonical_profiles(
    profiles: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    """Normalize profile weights for stable JSON round-trip comparison."""
    return {
        str(regime): [float(weight) for weight in weights]
        for regime, weights in profiles.items()
    }


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    """Return a stable content identity for a checkpoint or artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def align_rankings(
        expected_keys: Sequence[str],
        source_keys: Sequence[str],
        rankings: np.ndarray,
) -> np.ndarray:
    """Align a rank matrix to CEARF's persisted query order."""
    expected = [str(key) for key in expected_keys]
    source = [str(key) for key in source_keys]
    if len(source) != len(set(source)):
        raise ValueError("neural prediction keys contain duplicates")
    if set(expected) != set(source):
        missing = sorted(set(expected) - set(source))[:3]
        extra = sorted(set(source) - set(expected))[:3]
        raise ValueError(
            f"query coverage mismatch; missing={missing}, extra={extra}")
    if len(rankings) != len(source):
        raise ValueError("rank matrix row count does not match source keys")
    row_for = {key: row for row, key in enumerate(source)}
    return np.asarray(
        [rankings[row_for[key]] for key in expected], dtype=np.int32)


def context_feature_matrix(
        queries: Mapping[str, Mapping[str, Sequence[int]]],
        keys: Sequence[str],
        item_frequency: Mapping[int, int],
        head_items: set[int],
) -> np.ndarray:
    """Compute the exact three target-free features of the primary gate."""
    output = np.zeros((len(keys), len(PRIMARY_COLUMNS)), dtype=np.float32)
    for row, uid0 in enumerate(keys):
        uid = str(uid0)
        context = [
            int(item)
            for item in queries[uid].get("context", ())
            if int(item) > 0
        ]
        last = context[-1] if context else 0
        output[row] = (
            math.log1p(len(context)),
            math.log1p(item_frequency.get(last, 0)),
            float(bool(last) and last not in head_items),
        )
    return output


def inspect_narm_checkpoint(
        path: Path,
        *,
        expected_seed: int | None = None,
        expected_n_items: int | None = None,
) -> dict:
    """Validate checkpoint identity and return non-tensor protocol metadata."""
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model") != "NARM":
        raise RuntimeError(f"{path} is not a paper-baseline NARM checkpoint")
    if expected_seed is not None and int(payload.get("seed", -1)) != expected_seed:
        raise RuntimeError(
            f"{path}: expected seed {expected_seed}, got {payload.get('seed')}")
    if (
        expected_n_items is not None
        and int(payload.get("n_items", -1)) != expected_n_items
    ):
        raise RuntimeError(
            f"{path}: expected {expected_n_items} items, "
            f"got {payload.get('n_items')}")
    epoch = int(payload.get("epoch", 0))
    if epoch <= 0:
        raise RuntimeError(f"{path}: missing positive locked epoch budget")
    return {
        "model": "NARM",
        "protocol": payload.get("protocol"),
        "seed": int(payload["seed"]),
        "n_items": int(payload["n_items"]),
        "dim": int(payload["dim"]),
        "epoch": epoch,
        "batch_size_requested": payload.get("batch_size_requested"),
        "sessions_fingerprint": payload.get("sessions_fingerprint"),
        "validation": payload.get("validation"),
    }


def load_narm_model(path: Path) -> tuple[torch.nn.Module, dict]:
    """Load a paper-baseline-format NARM checkpoint on CPU."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model") != "NARM":
        raise RuntimeError(f"{path} is not a NARM checkpoint")
    model = build_model(
        "NARM", int(payload["n_items"]), int(payload.get("dim", 64)))
    model.load_state_dict(payload["state_dict"])
    return model.cpu().eval(), payload


def load_or_train_narm_refit(
        checkpoint: Path,
        sessions: dict,
        n_items: int,
        seed: int,
        epochs: int,
        batch_size: int,
        *,
        protocol: str = "dynamic-beta-narm-expert-swap-oof-v1",
        role: str = "OOF",
) -> tuple[torch.nn.Module, dict]:
    """Load an exact NARM refit or train it from a fresh initialization."""
    fingerprint = session_fingerprint(sessions)
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        compatible = (
            payload.get("model") == "NARM"
            and payload.get("protocol") == protocol
            and int(payload.get("seed", -1)) == seed
            and int(payload.get("n_items", -1)) == n_items
            and int(payload.get("epoch", -1)) == epochs
            and int(payload.get("batch_size_requested", -1)) == batch_size
            and str(payload.get("sessions_fingerprint", "")) == fingerprint
        )
        if compatible:
            model = build_model(
                "NARM", n_items, int(payload.get("dim", 64)))
            model.load_state_dict(payload["state_dict"])
            print(
                f"[NARM-SWAP] loading {role} model {checkpoint}",
                flush=True,
            )
            return model.cpu().eval(), dict(payload.get("training_report", {}))
        print(
            f"[NARM-SWAP] ignoring incompatible {role} checkpoint "
            f"{checkpoint}",
            flush=True,
        )

    print(
        f"[NARM-SWAP] training {role} NARM seed={seed}, epochs={epochs}",
        flush=True,
    )
    model, report = train_fixed_epochs(
        "NARM", sessions, n_items, seed, epochs, batch_size)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": "NARM",
            "protocol": protocol,
            "seed": seed,
            "n_items": n_items,
            "dim": 64,
            "epoch": epochs,
            "batch_size_requested": batch_size,
            "sessions_fingerprint": fingerprint,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "training_report": report,
        },
        checkpoint,
    )
    return model.cpu().eval(), report


def load_prediction_cache(
        path: Path,
        queries: dict,
        checkpoint_sha256: str,
        width: int,
        exclude_seen: bool,
) -> tuple[list[str], np.ndarray] | None:
    """Load predictions only when every protocol identity field matches."""
    if not path.exists():
        return None
    with np.load(path) as saved:
        required = {
            "keys",
            "rankings",
            "query_fingerprint",
            "checkpoint_sha256",
            "candidate_width",
            "exclude_seen",
        }
        if not required.issubset(saved.files):
            return None
        if (
            str(saved["query_fingerprint"].item())
            != query_fingerprint(queries)
            or str(saved["checkpoint_sha256"].item()) != checkpoint_sha256
            or int(saved["candidate_width"].item()) != width
            or bool(saved["exclude_seen"].item()) != bool(exclude_seen)
        ):
            return None
        rankings = saved["rankings"].astype(np.int32)
        if rankings.shape != (len(queries), width):
            return None
        return [str(key) for key in saved["keys"]], rankings


def predict_and_cache_narm(
        cache: Path,
        checkpoint: Path,
        queries: dict,
        n_items: int,
        width: int,
        batch_size: int,
        exclude_seen: bool,
        model: torch.nn.Module | None = None,
) -> tuple[list[str], np.ndarray]:
    """Predict exact top-``width`` NARM ranks with an identity-checked cache."""
    checkpoint_digest = file_sha256(checkpoint)
    cached = load_prediction_cache(
        cache, queries, checkpoint_digest, width, exclude_seen)
    if cached is not None:
        print(f"[NARM-SWAP] loading predictions {cache}", flush=True)
        return cached
    if model is None:
        model, _ = load_narm_model(checkpoint)
    keys, rankings, seconds = predict_array_target_free(
        model, queries, n_items, width, batch_size, exclude_seen)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        keys=np.asarray(keys, dtype=str),
        rankings=rankings.astype(np.int32),
        query_fingerprint=np.asarray(query_fingerprint(queries)),
        checkpoint_sha256=np.asarray(checkpoint_digest),
        candidate_width=np.asarray(width, dtype=np.int32),
        exclude_seen=np.asarray(bool(exclude_seen)),
        inference_seconds=np.asarray(seconds, dtype=np.float64),
    )
    return keys, rankings


def predict_array_target_free(
        model: torch.nn.Module,
        queries: Mapping[str, Mapping[str, Sequence[int]]],
        n_items: int,
        topk: int,
        batch_size: int,
        exclude_seen: bool,
) -> tuple[list[str], np.ndarray, float]:
    """Predict NARM ranks without reading or collating query targets."""
    device = next(model.parameters()).device
    keys = sorted(str(uid) for uid in queries)
    width = min(topk, n_items - 1)
    output = np.empty((len(keys), width), dtype=np.int32)
    started = time.time()
    model.eval()
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start:start + batch_size]
        # ``collate`` structurally expects a target, but this constant is not
        # read from the query and its returned target tensor is discarded.
        batch = [
            (
                [
                    int(item)
                    for item in queries[uid].get("context", ())
                    if 0 < int(item) < n_items
                ][-50:],
                0,
            )
            for uid in batch_keys
        ]
        contexts, lengths, _ = collate(batch)
        contexts = contexts.to(device)
        lengths = lengths.to(device)
        with torch.no_grad():
            scores = model_logits(model, contexts, lengths)
            scores[:, 0] = -torch.inf
            if exclude_seen:
                for row, uid in enumerate(batch_keys):
                    seen = sorted({
                        int(item)
                        for item in queries[uid].get("context", ())
                        if 0 < int(item) < n_items
                    })
                    if seen:
                        scores[
                            row,
                            torch.as_tensor(
                                seen, device=scores.device),
                        ] = -torch.inf
            ranking = torch.topk(scores, k=width, dim=1).indices
            output[start:start + len(batch_keys)] = (
                ranking.cpu().numpy())
    return keys, output, time.time() - started


def resolve_epoch_budget_checkpoint(
        domain: str,
        seed: int,
        amazon_root: Path,
        diginetica_root: Path,
) -> Path:
    """Resolve the pre-existing checkpoint used only to lock epoch count."""
    root = diginetica_root if domain == "Diginetica_HID" else amazon_root
    return root / f"{domain.lower()}_full_narm_seed{seed}.pt"


def cache_compatibility(
        path: Path,
        queries: dict,
        profiles: dict,
) -> bool:
    """Read-only compatibility check used by ``--dry-run``."""
    if not path.exists():
        return False
    with np.load(path) as saved:
        return bool(
            {"fingerprint", "profiles"}.issubset(saved.files)
            and str(saved["fingerprint"].item()) == query_fingerprint(queries)
            and str(saved["profiles"].item())
            == json.dumps(profiles, sort_keys=True)
        )


def load_primary_memory_cache(
        path: Path,
        queries: dict,
        profiles: dict,
        width: int,
) -> dict[str, np.ndarray]:
    """Strictly load a provenance-audited primary memory cache."""
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as saved:
        required = {"keys", "selected", "fingerprint", "profiles"}
        if not required.issubset(saved.files):
            raise RuntimeError(f"{path}: incomplete primary memory cache")
        if (
            str(saved["fingerprint"].item()) != query_fingerprint(queries)
            or str(saved["profiles"].item())
            != json.dumps(profiles, sort_keys=True)
        ):
            raise RuntimeError(f"{path}: primary memory identity differs")
        keys = saved["keys"].astype(str, copy=True)
        selected = saved["selected"].astype(np.int32, copy=True)
    if selected.shape != (len(keys), width) or len(keys) != len(queries):
        raise RuntimeError(
            f"{path}: expected {(len(queries), width)}, got "
            f"{selected.shape}"
        )
    return {"keys": keys, "selected": selected}


def pasgr_reference(
        result_path: Path,
        domain: str,
        seed: int,
) -> dict | None:
    """Return the matched primary PASGR run when it has completed."""
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text())
    for run in payload.get(domain, {}).get("runs", ()):
        if int(run.get("seed", -1)) == seed:
            return {
                "protocol": payload[domain].get("protocol"),
                "rank_artifact": run.get("rank_artifact"),
                "training": run.get("training"),
                "test_metrics": run.get("test", {}).get("metrics"),
            }
    return None


def metric_deltas(left: dict, right: dict) -> dict:
    """Return selected metric differences ``left - right``."""
    names = (
        "recall@6",
        "ndcg@6",
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "utility",
        "net_rescues_vs_memory",
    )
    return {
        name: float(left[name] - right[name])
        for name in names
        if name in left and name in right
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--candidate-width", type=int, default=120)
    parser.add_argument("--oof-fraction", type=float, default=0.10)
    parser.add_argument("--oof-cap", type=int, default=5_000)
    parser.add_argument("--profile-cap", type=int, default=1_000)
    parser.add_argument("--valid-cap", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate split/cache/checkpoint identities without training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "dynamic_beta_expert_swap_results.json",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=HERE / "dynamic_beta_expert_swap_artifacts",
    )
    parser.add_argument(
        "--dynamic-artifact-dir",
        type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_artifacts",
    )
    parser.add_argument(
        "--primary-results",
        type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_results.json",
    )
    parser.add_argument(
        "--amazon-checkpoint-dir",
        type=Path,
        default=HERE / "paper_baseline_artifacts",
    )
    parser.add_argument(
        "--diginetica-checkpoint-dir",
        type=Path,
        default=HERE / "paper_baseline_digi_nested_artifacts",
    )
    args = parser.parse_args()

    unknown = sorted(set(args.domains) - set(DOMAINS))
    if unknown:
        parser.error(f"unsupported domains: {unknown}")
    if args.candidate_width < 20:
        parser.error("--candidate-width must be at least 20")
    if args.oof_cap <= args.profile_cap:
        parser.error("--oof-cap must exceed --profile-cap")

    if not args.dry_run:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
    results = (
        json.loads(args.output.read_text())
        if args.output.exists() and not args.dry_run
        else {}
    )
    dry_report: dict[str, dict] = {}

    for domain in args.domains:
        domain_started = time.time()
        print(f"\n[NARM-SWAP] === {domain} ===", flush=True)
        data = loaders.ALL_LOADERS[domain]()
        validation_candidate_pool_queries = len(data["valid_queries"])
        validation_candidate_pool_sources = canonical_validation_sources(
            data["valid_queries"])
        if len(data["valid_queries"]) > args.valid_cap:
            valid_keys = sorted(
                data["valid_queries"], key=cearf._stable_fraction
            )[:args.valid_cap]
            data["valid_queries"] = {
                uid: data["valid_queries"][uid] for uid in valid_keys
            }
        sessions = data["train_sessions"]
        tune_sessions = hold_out_validation_targets(
            sessions, data["valid_queries"])
        validation_sources = canonical_validation_sources(
            data["valid_queries"])
        (
            inner_fit_sessions,
            profile_queries,
            gate_queries,
            split_report,
        ) = make_training_oof_split(
            tune_sessions,
            validation_sources,
            args.oof_fraction,
            args.oof_cap,
            args.profile_cap,
        )
        held_oof_sources = {
            str(uid).split("::", 1)[1]
            for uid in (*profile_queries, *gate_queries)
        }
        split_report.update({
            "validation_definition": (
                "stable-hash declared subset selected before allocation fit"
            ),
            "validation_candidate_pool_queries": (
                validation_candidate_pool_queries
            ),
            "validation_candidate_pool_sources": len(
                validation_candidate_pool_sources),
            "declared_validation_queries": len(data["valid_queries"]),
            "declared_validation_sources": len(validation_sources),
            "validation_candidate_pool_source_overlap": len(
                held_oof_sources & validation_candidate_pool_sources),
            "unselected_validation_candidates_remain_training_events": True,
        })
        exclude_seen = domain not in {"Diginetica_HID", "Tmall"}
        config = cearf.CEARFConfig(exclude_seen=exclude_seen)
        inner_index = cearf.CEARFIndex(
            inner_fit_sessions, data["n_items"], config)
        profiles_raw, profile_report = cearf.tune_profiles(
            inner_index, profile_queries)
        profiles = canonical_profiles(profiles_raw)

        primary_domain_dir = args.dynamic_artifact_dir / domain.lower()
        swap_domain_dir = args.artifact_dir / domain.lower()
        gate_memory_path = primary_domain_dir / "gate_oof_memory.npz"
        test_memory_path = primary_domain_dir / "test_memory.npz"

        if args.dry_run:
            seed_report = {}
            for seed in args.seeds:
                epoch_source_checkpoint = resolve_epoch_budget_checkpoint(
                    domain,
                    seed,
                    args.amazon_checkpoint_dir,
                    args.diginetica_checkpoint_dir,
                )
                epoch_source_metadata = inspect_narm_checkpoint(
                    epoch_source_checkpoint,
                    expected_seed=seed,
                    expected_n_items=data["n_items"],
                )
                full_refit_checkpoint = (
                    swap_domain_dir
                    / "checkpoints"
                    / f"seed{seed}_narm_full_locked.pt"
                )
                oof_checkpoint = (
                    swap_domain_dir
                    / "checkpoints"
                    / f"seed{seed}_narm_oof.pt"
                )
                seed_report[str(seed)] = {
                    "epoch_budget_source_checkpoint": str(
                        epoch_source_checkpoint),
                    "epoch_budget_source_metadata": epoch_source_metadata,
                    "full_refit_checkpoint_exists": (
                        full_refit_checkpoint.exists()),
                    "oof_checkpoint_exists": oof_checkpoint.exists(),
                    "oof_prediction_cache_exists": (
                        swap_domain_dir
                        / "predictions"
                        / f"seed{seed}_narm_gate_oof_top"
                        f"{args.candidate_width}.npz"
                    ).exists(),
                    "test_prediction_cache_exists": (
                        swap_domain_dir
                        / "predictions"
                        / f"seed{seed}_narm_test_top"
                        f"{args.candidate_width}.npz"
                    ).exists(),
                }
            dry_report[domain] = {
                "n_items": int(data["n_items"]),
                "train_sessions": len(sessions),
                "inner_fit_sessions": len(inner_fit_sessions),
                "split": split_report,
                "profiles": profiles,
                "profile_report": profile_report,
                "gate_memory_cache_compatible": cache_compatibility(
                    gate_memory_path, gate_queries, profiles),
                "test_memory_cache_compatible": cache_compatibility(
                    test_memory_path, data["test_queries"], profiles),
                "seeds": seed_report,
            }
            continue

        gate_memory = load_primary_memory_cache(
            gate_memory_path,
            gate_queries,
            profiles,
            args.candidate_width,
        )
        test_memory = load_primary_memory_cache(
            test_memory_path,
            data["test_queries"],
            profiles,
            args.candidate_width,
        )
        gate_keys = [str(key) for key in gate_memory["keys"]]
        test_keys = [str(key) for key in test_memory["keys"]]
        gate_targets = targets_for(gate_keys, gate_queries)

        inner_frequency = Counter(
            item
            for sequence in inner_fit_sessions.values()
            for item in sequence
        )
        final_frequency = Counter(
            item for sequence in sessions.values() for item in sequence)
        inner_head = set(
            popularity_partition(inner_frequency, data["n_items"])[0].tolist())
        final_head = set(
            popularity_partition(final_frequency, data["n_items"])[0].tolist())
        gate_features = context_feature_matrix(
            gate_queries,
            gate_keys,
            inner_frequency,
            inner_head,
        )
        test_features = context_feature_matrix(
            data["test_queries"],
            test_keys,
            final_frequency,
            final_head,
        )

        identity = {
            "candidate_width": args.candidate_width,
            "oof_fraction": args.oof_fraction,
            "oof_cap": args.oof_cap,
            "profile_cap": args.profile_cap,
            "gate_query_fingerprint": split_report[
                "gate_query_fingerprint"],
            "profile_query_fingerprint": split_report[
                "profile_query_fingerprint"],
        }
        if domain in results:
            domain_block = results[domain]
            if (
                domain_block.get("protocol")
                != EXPERT_SWAP_PROTOCOL
                or domain_block.get("identity") != identity
                or domain_block.get("profiles") != profiles
            ):
                raise RuntimeError(
                    f"{args.output}: existing {domain} block belongs to a "
                    "different protocol identity; use a new output path")
        else:
            domain_block = {
                "domain": domain,
                "protocol": EXPERT_SWAP_PROTOCOL,
                "comparison_scope": (
                    "system-level neural-expert sensitivity; not "
                    "metadata-matched architecture attribution"
                ),
                "identity": identity,
                "profiles": profiles,
                "profile_report": profile_report,
                "split": split_report,
                "runs": [],
            }
        completed = {
            int(run["seed"]): index
            for index, run in enumerate(domain_block["runs"])
        }
        for seed in args.seeds:
            if seed in completed and not args.force:
                print(
                    f"[NARM-SWAP] {domain} seed={seed} already complete",
                    flush=True,
                )
                continue
            seed_started = time.time()
            epoch_source_checkpoint = resolve_epoch_budget_checkpoint(
                domain,
                seed,
                args.amazon_checkpoint_dir,
                args.diginetica_checkpoint_dir,
            )
            epoch_source_metadata = inspect_narm_checkpoint(
                epoch_source_checkpoint,
                expected_seed=seed,
                expected_n_items=data["n_items"],
            )
            locked_epochs = int(epoch_source_metadata["epoch"])
            checkpoint_dir = swap_domain_dir / "checkpoints"
            prediction_dir = swap_domain_dir / "predictions"
            oof_checkpoint = (
                checkpoint_dir / f"seed{seed}_narm_oof.pt")
            full_refit_checkpoint = (
                checkpoint_dir / f"seed{seed}_narm_full_locked.pt")
            oof_model, oof_training = load_or_train_narm_refit(
                oof_checkpoint,
                inner_fit_sessions,
                data["n_items"],
                seed,
                locked_epochs,
                args.batch_size,
            )
            oof_source_keys, oof_narm_raw = predict_and_cache_narm(
                prediction_dir
                / f"seed{seed}_narm_gate_oof_top"
                f"{args.candidate_width}.npz",
                oof_checkpoint,
                gate_queries,
                data["n_items"],
                args.candidate_width,
                args.batch_size,
                exclude_seen,
                model=oof_model,
            )
            oof_narm = align_rankings(
                gate_keys, oof_source_keys, oof_narm_raw)
            del oof_model, oof_narm_raw
            gc.collect()

            global_model, global_report = fit_global(
                gate_memory["selected"],
                oof_narm,
                gate_targets,
                seed,
            )
            dynamic_model, dynamic_report = fit_dynamic(
                gate_features,
                gate_memory["selected"],
                oof_narm,
                gate_targets,
                seed,
                float(global_model.beta_),
                hidden=0,
                max_residual=0.10,
            )
            dynamic_report["feature_names"] = list(PRIMARY_FEATURE_NAMES)

            full_model, full_training = load_or_train_narm_refit(
                full_refit_checkpoint,
                sessions,
                data["n_items"],
                seed,
                locked_epochs,
                args.batch_size,
                protocol=(
                    "dynamic-beta-narm-expert-swap-full-training-refit-v1"
                ),
                role="full-training",
            )
            full_refit_metadata = inspect_narm_checkpoint(
                full_refit_checkpoint,
                expected_seed=seed,
                expected_n_items=data["n_items"],
            )
            expected_full_fingerprint = session_fingerprint(sessions)
            if (
                full_refit_metadata.get("sessions_fingerprint")
                != expected_full_fingerprint
            ):
                raise RuntimeError(
                    f"{full_refit_checkpoint}: full-training session "
                    "fingerprint differs"
                )
            epoch_source_digest = file_sha256(epoch_source_checkpoint)
            full_refit_digest = file_sha256(full_refit_checkpoint)
            manifest = {
                "protocol": EXPERT_SWAP_PROTOCOL,
                "domain": domain,
                "seed": seed,
                "created_before_narm_test_prediction_and_test_evaluation": True,
                "changed_factor": "neural expert: PASGR -> ID-only NARM",
                "held_constant": [
                    "deterministic OOF source-session split",
                    "CEARF memory profiles",
                    "candidate width",
                    "continuous pairwise beta objective",
                    "bounded linear context gate",
                    "RRF k=20",
                    "complete-training refit scope for test experts",
                    "official test protocol",
                ],
                "comparison_scope": (
                    "system-level sensitivity; PASGR/NARM is not a "
                    "metadata-matched architecture contrast where PASGR uses "
                    "side information"
                ),
                "validation_labels_used_for_beta": False,
                "test_labels_used_for_beta": False,
                "beta_grid_or_search": False,
                "beta_training_source": (
                    "leave-last-out queries from OOF training sessions only"
                ),
                "narm_oof_initialization": "fresh random initialization",
                "narm_epoch_budget": locked_epochs,
                "narm_epoch_budget_status": (
                    "locked by the pre-existing baseline before this audit; "
                    "not selected by the expert-swap experiment"
                ),
                "epoch_budget_source_checkpoint": str(
                    epoch_source_checkpoint),
                "epoch_budget_source_checkpoint_sha256": (
                    epoch_source_digest),
                "epoch_budget_source_checkpoint_metadata": (
                    epoch_source_metadata),
                "full_narm_checkpoint": str(full_refit_checkpoint),
                "full_narm_checkpoint_sha256": full_refit_digest,
                "full_narm_checkpoint_metadata": full_refit_metadata,
                "full_narm_training": full_training,
                "oof_narm_checkpoint": str(oof_checkpoint),
                "oof_narm_checkpoint_sha256": file_sha256(oof_checkpoint),
                "oof_narm_training": oof_training,
                "candidate_width": args.candidate_width,
                "rrf_constant": 20.0,
                "dynamic_beta_equation": (
                    "beta_q = beta_OOF + delta_eff * tanh(w^T z_q + b); "
                    "delta_eff = min(0.10, beta_OOF, 1-beta_OOF)"
                ),
                "gate_architecture": "bounded linear residual",
                "gate_feature_names": list(PRIMARY_FEATURE_NAMES),
                "split": split_report,
                "memory_profiles": profiles,
                "memory_profile_training": profile_report,
                "global_beta_training": global_report,
                "dynamic_beta_training": dynamic_report,
            }
            manifest_path = (
                swap_domain_dir / f"seed{seed}_frozen_manifest.json")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))

            test_source_keys, test_narm_raw = predict_and_cache_narm(
                prediction_dir
                / f"seed{seed}_narm_test_top"
                f"{args.candidate_width}.npz",
                full_refit_checkpoint,
                data["test_queries"],
                data["n_items"],
                args.candidate_width,
                args.batch_size,
                exclude_seen,
                model=full_model,
            )
            test_narm = align_rankings(
                test_keys, test_source_keys, test_narm_raw)
            del full_model, test_narm_raw
            gc.collect()

            # This is the first test-target materialization, after the frozen
            # manifest and target-free NARM test prediction.
            test_targets = targets_for(test_keys, data["test_queries"])
            memory_ranks = ranks_at_20(
                test_memory["selected"], test_targets)
            global_beta = global_model.predict(len(test_keys))
            dynamic_beta = dynamic_model.predict(test_features)
            rankings = {
                "memory_only": test_memory["selected"][:, :20],
                "neural_only": test_narm[:, :20],
                "fixed_05": fuse_with_dynamic_beta(
                    test_memory["selected"],
                    test_narm,
                    np.full(len(test_keys), 0.5, dtype=np.float32),
                    constant=20.0,
                ),
                "oof_global": fuse_with_dynamic_beta(
                    test_memory["selected"],
                    test_narm,
                    global_beta,
                    constant=20.0,
                ),
                "dynamic": fuse_with_dynamic_beta(
                    test_memory["selected"],
                    test_narm,
                    dynamic_beta,
                    constant=20.0,
                ),
            }
            metrics = {}
            rank_vectors = {}
            for name, ranking in rankings.items():
                metrics[name], rank_vectors[name] = evaluate_ranking(
                    ranking, test_targets, memory_ranks)
            metrics["oof_global"]["beta"] = beta_summary(global_beta)
            metrics["dynamic"]["beta"] = beta_summary(dynamic_beta)

            pasgr = pasgr_reference(
                args.primary_results, domain, seed)
            pasgr_comparison = None
            if pasgr is not None:
                pasgr_metrics = pasgr.get("test_metrics", {})
                pasgr_comparison = {
                    "scope": (
                        "system-level expert sensitivity; metadata is not "
                        "matched"
                    ),
                    "narm_dynamic_minus_pasgr_dynamic": metric_deltas(
                        metrics["dynamic"],
                        pasgr_metrics.get("dynamic", {}),
                    ),
                    "narm_global_minus_pasgr_global": metric_deltas(
                        metrics["oof_global"],
                        pasgr_metrics.get("oof_global", {}),
                    ),
                    "pasgr_reference": pasgr,
                }

            rank_artifact = (
                swap_domain_dir
                / f"seed{seed}_narm_expert_swap_ranks.npz"
            )
            rank_payload = {
                "test_keys": np.asarray(test_keys, dtype=str),
                "test_query_fingerprint": np.asarray(
                    query_fingerprint(data["test_queries"])),
                "test_memory_top20": rankings["memory_only"].astype(
                    np.int32),
                "test_narm_top20": rankings["neural_only"].astype(np.int32),
                "test_fixed_05_top20": rankings["fixed_05"].astype(np.int32),
                "test_oof_global_top20": rankings["oof_global"].astype(
                    np.int32),
                "test_dynamic_top20": rankings["dynamic"].astype(np.int32),
                "test_oof_global_beta": global_beta.astype(np.float32),
                "test_dynamic_beta": dynamic_beta.astype(np.float32),
                "test_context_features": test_features.astype(np.float32),
                "oof_keys": np.asarray(gate_keys, dtype=str),
                "oof_query_fingerprint": np.asarray(
                    query_fingerprint(gate_queries)),
                "oof_memory_top120": gate_memory["selected"].astype(
                    np.int32),
                "oof_narm_top120": oof_narm.astype(np.int32),
                "oof_targets": gate_targets.astype(np.int32),
                "manifest_sha256": np.asarray(file_sha256(manifest_path)),
            }
            for name, ranks in rank_vectors.items():
                rank_payload[f"test_{name}_rank"] = ranks.astype(np.uint8)
            np.savez_compressed(rank_artifact, **rank_payload)

            run = {
                "seed": seed,
                "neural_expert": "NARM (ID-only)",
                "locked_narm_epochs": locked_epochs,
                "manifest": str(manifest_path),
                "rank_artifact": str(rank_artifact),
                "training": {
                    "oof_narm": oof_training,
                    "global": global_report,
                    "dynamic": dynamic_report,
                },
                "test": {
                    "metrics": metrics,
                    "dynamic_minus_global": metric_deltas(
                        metrics["dynamic"], metrics["oof_global"]),
                },
                "pasgr_comparison": pasgr_comparison,
                "seconds": time.time() - seed_started,
            }
            if seed in completed:
                domain_block["runs"][completed[seed]] = run
            else:
                domain_block["runs"].append(run)
            results[domain] = domain_block
            args.output.write_text(json.dumps(results, indent=2))
            print(
                f"[NARM-SWAP] DONE {domain} seed={seed}: "
                f"NARM={metrics['neural_only']['recall@20']:.5f} "
                f"global={metrics['oof_global']['recall@20']:.5f} "
                f"dynamic={metrics['dynamic']['recall@20']:.5f}",
                flush=True,
            )

        domain_block["seconds_total_latest_invocation"] = (
            time.time() - domain_started)
        results[domain] = domain_block
        args.output.write_text(json.dumps(results, indent=2))

    if args.dry_run:
        print(json.dumps(dry_report, indent=2))
    else:
        print(f"[NARM-SWAP] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
