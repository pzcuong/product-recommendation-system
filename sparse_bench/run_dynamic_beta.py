#!/usr/bin/env python3
"""Run CEARF-N with training-calibrated query-conditioned dynamic beta.

Protocol
--------
1. Hold out source sessions from the training split by a deterministic hash.
2. Fit CEARF memory and PASGR without those sessions.
3. Use one disjoint subset to lock the CEARF memory profile and the remainder
   to learn continuous beta policies from OOF ranks.
4. Freeze the policies before producing official validation/test predictions.

No beta grid is evaluated. No validation target is used to fit, select or
early-stop a beta policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch

import cearf
import loaders
import pasgr
from dynamic_beta import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    TrainOnlyDynamicBeta,
    TrainOnlyGlobalBeta,
    feature_matrix,
    fuse_with_dynamic_beta,
)
from run_cearfn_evidence import (
    load_or_build_memory,
    metrics_from_ranks,
    popularity_partition,
    query_fingerprint,
    ranks_at_20,
    targets_for,
)
from run_cearfn_v2 import (
    REPEAT_PROTOCOL_DOMAINS,
    load_pasgr_config,
    train_pasgr_v2,
)
from run_pasgr_full import semantic_matrix
from validation_protocol import hold_out_validation_targets


HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")
SEEDS = (42, 123, 456)
RRF_CONSTANTS = (10.0, 20.0, 60.0)
PRIMARY_PROTOCOL = "dynamic-beta-train-only-v2-declared-validation-5k"
LEGACY_PROTOCOL_ALIASES = {
    "dynamic-beta-train-only-v2-no-validation-source-overlap",
    PRIMARY_PROTOCOL,
}


def canonical_validation_sources(
        queries: Mapping[str, Mapping[str, Sequence[int]]]) -> set[str]:
    """Map validation query IDs to their source-session IDs.

    Diginetica appends ``_v`` to a source-session identifier, whereas the
    Amazon loaders use the source identifier directly.
    """
    return {
        str(uid)[:-2] if str(uid).endswith("_v") else str(uid)
        for uid in queries
    }


def session_fingerprint(
        sessions: Mapping[str, Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for uid in sorted(sessions):
        digest.update(str(uid).encode())
        digest.update(b":")
        digest.update(" ".join(map(str, sessions[uid])).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def array_fingerprint(array: np.ndarray | None) -> str:
    if array is None:
        return "none"
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def make_training_oof_split(
        sessions: Mapping[str, Sequence[int]],
        declared_validation_sources: set[str],
        fraction: float,
        cap: int,
        profile_cap: int,
) -> tuple[dict, dict, dict, dict]:
    """Return inner-fit sessions, profile queries, gate queries and metadata."""
    eligible = [
        str(uid) for uid, seq in sessions.items()
        if (
            len(seq) >= 3
            and str(uid) not in declared_validation_sources
        )
    ]
    ordered = sorted(
        eligible,
        key=lambda uid: cearf._stable_fraction(f"dynamic-beta::{uid}"),
    )
    wanted = min(cap, max(2, int(len(ordered) * fraction)))
    held = ordered[:wanted]
    n_profile = min(profile_cap, max(1, len(held) // 4))
    profile_sources = held[:n_profile]
    gate_sources = held[n_profile:]
    if not gate_sources:
        raise ValueError("OOF split leaves no gate-calibration queries")
    held_set = set(held)
    inner_fit = {
        str(uid): [int(item) for item in seq]
        for uid, seq in sessions.items()
        if str(uid) not in held_set
    }

    def queries_for(source_ids: Sequence[str], label: str) -> dict:
        return {
            f"{label}::{uid}": {
                "context": [int(item) for item in sessions[uid][:-1]],
                "targets": [int(sessions[uid][-1])],
            }
            for uid in source_ids
        }

    profile_queries = queries_for(profile_sources, "profile-oof")
    gate_queries = queries_for(gate_sources, "gate-oof")
    metadata = {
        "split_method": "stable BLAKE2 ordering over source-session IDs",
        "fraction": fraction,
        "cap": cap,
        "eligible_source_sessions": len(eligible),
        "excluded_declared_validation_sources": len(
            declared_validation_sources),
        "held_source_sessions": len(held),
        "profile_source_sessions": len(profile_sources),
        "gate_source_sessions": len(gate_sources),
        "profile_gate_source_overlap": 0,
        "declared_validation_source_overlap": int(
            len(set(held) & set(declared_validation_sources))),
        "profile_query_fingerprint": query_fingerprint(profile_queries),
        "gate_query_fingerprint": query_fingerprint(gate_queries),
    }
    return inner_fit, profile_queries, gate_queries, metadata


def _load_memory_file_if_compatible(
        path: Path,
        queries: dict,
        profiles: dict,
) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    fingerprint = query_fingerprint(queries)
    profile_json = json.dumps(profiles, sort_keys=True)
    with np.load(path) as saved:
        if (
            "fingerprint" not in saved.files
            or "profiles" not in saved.files
            or str(saved["fingerprint"].item()) != fingerprint
            or str(saved["profiles"].item()) != profile_json
        ):
            return None
        return {
            key: saved[key]
            for key in saved.files
            if key not in {"fingerprint", "profiles"}
        }


def load_or_build_memory_with_fallback(
        destination: Path,
        fallbacks: Sequence[Path],
        index,
        queries: dict,
        profiles: dict,
        width: int,
        label: str,
) -> dict[str, np.ndarray]:
    cached = _load_memory_file_if_compatible(
        destination, queries, profiles)
    if cached is not None:
        print(f"[DYBETA] loading {destination}", flush=True)
        return cached
    for fallback in fallbacks:
        cached = _load_memory_file_if_compatible(
            fallback, queries, profiles)
        if cached is not None:
            print(f"[DYBETA] reusing {fallback}", flush=True)
            return cached
    return load_or_build_memory(
        destination, index, queries, profiles, width, label)


def _model_from_checkpoint(
        checkpoint: Path,
        data: dict,
        expected_sessions_fingerprint: str | None = None,
) -> pasgr.PASGRModel | None:
    if not checkpoint.exists():
        return None
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    stored_fingerprint = saved.get("sessions_fingerprint")
    if (
        expected_sessions_fingerprint is not None
        and stored_fingerprint is not None
        and str(stored_fingerprint) != expected_sessions_fingerprint
    ):
        return None
    config = pasgr.PASGRConfig(**saved["config"])
    model = pasgr.PASGRModel(
        np.zeros((data["n_items"], config.dim), dtype=np.float32),
        config,
    )
    model.load_state_dict(saved["state_dict"])
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[DYBETA] loading model {checkpoint}", flush=True)
    return model.to(device).eval()


def load_or_train_model(
        checkpoint: Path,
        fallback_checkpoint: Path | None,
        data: dict,
        sessions: dict,
        semantic: np.ndarray | None,
        seed: int,
        epochs: int,
        gated_config: dict,
) -> pasgr.PASGRModel:
    fingerprint = session_fingerprint(sessions)
    model = _model_from_checkpoint(
        checkpoint, data, fingerprint)
    if model is None and fallback_checkpoint is not None:
        model = _model_from_checkpoint(
            fallback_checkpoint, data, None)
    if model is None:
        print(f"[DYBETA] training {checkpoint.name}", flush=True)
        model = train_pasgr_v2(
            data, sessions, semantic, seed, epochs, gated_config)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        torch.save({
            "config": asdict(model.config),
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "sessions_fingerprint": fingerprint,
            "protocol": PRIMARY_PROTOCOL,
        }, checkpoint)
    return model


def load_prediction_cache(
        path: Path,
        queries: dict,
) -> tuple[list[str], np.ndarray] | None:
    if not path.exists():
        return None
    fingerprint = query_fingerprint(queries)
    with np.load(path) as saved:
        if str(saved["fingerprint"].item()) != fingerprint:
            return None
        return (
            [str(value) for value in saved["keys"]],
            saved["rankings"].astype(np.int32),
        )


def load_or_predict(
        cache_path: Path,
        checkpoint_path: Path,
        fallback_checkpoint: Path | None,
        data: dict,
        sessions: dict,
        queries: dict,
        semantic: np.ndarray | None,
        seed: int,
        epochs: int,
        gated_config: dict,
        width: int,
        exclude_seen: bool,
) -> tuple[list[str], np.ndarray]:
    cached = load_prediction_cache(cache_path, queries)
    if cached is not None:
        print(f"[DYBETA] loading predictions {cache_path}", flush=True)
        return cached
    model = load_or_train_model(
        checkpoint_path,
        fallback_checkpoint,
        data,
        sessions,
        semantic,
        seed,
        epochs,
        gated_config,
    )
    keys, rankings = pasgr.predict_pasgr_array(
        model, queries, data["n_items"], width,
        exclude_seen=exclude_seen)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        keys=np.asarray(keys, dtype=str),
        rankings=rankings.astype(np.int32),
        fingerprint=np.asarray(query_fingerprint(queries)),
    )
    del model
    gc.collect()
    return keys, rankings


def fit_global(
        memory: np.ndarray,
        neural: np.ndarray,
        targets: np.ndarray,
        seed: int,
        constant: float = 20.0,
        initial_beta: float = 0.35,
) -> tuple[TrainOnlyGlobalBeta, dict]:
    model = TrainOnlyGlobalBeta(
        seed=seed,
        rrf_constant=constant,
        initial_beta=initial_beta,
    )
    return model, model.fit(memory, neural, targets)


def fit_dynamic(
        features: np.ndarray,
        memory: np.ndarray,
        neural: np.ndarray,
        targets: np.ndarray,
        seed: int,
        initial_beta: float,
        constant: float = 20.0,
        hidden: int = 0,
        max_residual: float = 0.10,
) -> tuple[TrainOnlyDynamicBeta, dict]:
    model = TrainOnlyDynamicBeta(
        seed=seed,
        initial_beta=initial_beta,
        rrf_constant=constant,
        hidden=hidden,
        max_residual=max_residual,
    )
    return model, model.fit(features, memory, neural, targets)


def fit_short_long(
        queries: dict,
        keys: Sequence[str],
        memory: np.ndarray,
        neural: np.ndarray,
        targets: np.ndarray,
        seed: int,
        fallback_beta: float,
        short_context: int,
) -> tuple[dict[str, TrainOnlyGlobalBeta | None], dict]:
    lengths = np.asarray([
        len(queries[str(uid)].get("context", ())) for uid in keys
    ])
    models: dict[str, TrainOnlyGlobalBeta | None] = {}
    report = {}
    for regime, mask in (
        ("short", lengths <= short_context),
        ("long", lengths > short_context),
    ):
        if not mask.any():
            models[regime] = None
            report[regime] = {
                "beta": fallback_beta,
                "n": 0,
                "fallback": True,
            }
            continue
        model, block = fit_global(
            memory[mask], neural[mask], targets[mask], seed,
            initial_beta=fallback_beta)
        models[regime] = model
        report[regime] = {**block, "n": int(mask.sum()), "fallback": False}
    return models, report


def short_long_betas(
        models: dict[str, TrainOnlyGlobalBeta | None],
        queries: dict,
        keys: Sequence[str],
        fallback_beta: float,
        short_context: int,
) -> np.ndarray:
    values = []
    for uid0 in keys:
        uid = str(uid0)
        regime = (
            "short"
            if len(queries[uid].get("context", ())) <= short_context
            else "long"
        )
        model = models[regime]
        values.append(
            fallback_beta if model is None else float(model.beta_))
    return np.asarray(values, dtype=np.float32)


def evaluate_ranking(
        ranking: np.ndarray,
        targets: np.ndarray,
        memory_ranks: np.ndarray,
) -> tuple[dict, np.ndarray]:
    ranks = ranks_at_20(ranking, targets)
    metrics = metrics_from_ranks(ranks)
    memory_hit = memory_ranks > 0
    method_hit = ranks > 0
    rescue = int(np.sum(~memory_hit & method_hit))
    damage = int(np.sum(memory_hit & ~method_hit))
    metrics.update({
        "utility": float(
            .5 * metrics["recall@6"] + .5 * metrics["recall@20"]),
        "rescues_vs_memory": rescue,
        "damage_vs_memory": damage,
        "net_rescues_vs_memory": rescue - damage,
        "rescue_rate": float(rescue / len(ranks)),
        "damage_rate": float(damage / len(ranks)),
    })
    return metrics, ranks


def beta_summary(beta: np.ndarray) -> dict:
    quantiles = np.quantile(beta, [.1, .25, .5, .75, .9])
    return {
        "mean": float(beta.mean()),
        "std": float(beta.std()),
        "q10": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q90": float(quantiles[4]),
        "min": float(beta.min()),
        "max": float(beta.max()),
    }


def realized_beta_deciles(
        beta: np.ndarray,
        memory_ranks: np.ndarray,
        neural_ranks: np.ndarray,
) -> list[dict]:
    edges = np.quantile(beta, np.linspace(0.0, 1.0, 11))

    def gain(ranks: np.ndarray) -> np.ndarray:
        output = np.zeros(len(ranks), dtype=np.float64)
        hit = ranks > 0
        output[hit] = 1.0 / np.log2(ranks[hit].astype(float) + 1.0)
        return output

    advantage = gain(neural_ranks) - gain(memory_ranks)
    blocks = []
    for decile in range(10):
        if decile == 9:
            mask = (beta >= edges[decile]) & (beta <= edges[decile + 1])
        else:
            mask = (beta >= edges[decile]) & (beta < edges[decile + 1])
        blocks.append({
            "decile": decile + 1,
            "n": int(mask.sum()),
            "beta_mean": float(beta[mask].mean()) if mask.any() else 0.0,
            "realized_neural_minus_memory_ndcg20": (
                float(advantage[mask].mean()) if mask.any() else 0.0
            ),
        })
    return blocks


def protocol_manifest(
        domain: str,
        seed: int,
        split_report: dict,
        gated_config: dict,
        profiles: dict,
        global_report: dict,
        dynamic_report: dict,
        args,
) -> dict:
    return {
        "protocol": PRIMARY_PROTOCOL,
        "domain": domain,
        "seed": seed,
        "frozen_before_official_validation_or_test_evaluation_under_protocol": (
            True
        ),
        "beta_is_continuous": True,
        "beta_grid_or_search": False,
        "beta_training_source": "OOF leave-last-out queries from training only",
        "validation_labels_used_for_beta": False,
        "test_labels_used_for_beta": False,
        "base_expert_config_status": (
            "frozen before the dynamic-beta experiment"
        ),
        "candidate_width": args.candidate_width,
        "epochs": args.epochs,
        "rrf_constant_primary": 20.0,
        "dynamic_beta_equation": (
            "beta_q = beta_OOF + delta_eff * tanh(w^T z_q + b); "
            "delta_eff = min(0.10, beta_OOF, 1-beta_OOF)"
        ),
        "primary_gate": {
            "architecture": "linear bounded residual",
            "features": [
                FEATURE_NAMES[column]
                for column in FEATURE_GROUPS["context"]
            ],
            "max_residual": 0.10,
        },
        "feature_names": list(FEATURE_NAMES),
        "split": split_report,
        "pasgr_config": gated_config,
        "memory_profiles": profiles,
        "global_beta_training": global_report,
        "dynamic_beta_training": dynamic_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--candidate-width", type=int, default=120)
    parser.add_argument("--oof-fraction", type=float, default=0.10)
    parser.add_argument("--oof-cap", type=int, default=20_000)
    parser.add_argument("--profile-cap", type=int, default=5_000)
    parser.add_argument("--valid-cap", type=int, default=5_000)
    parser.add_argument("--pilot", action="store_true",
                        help="Use one seed and at most 5,000 OOF queries.")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_results.json")
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_artifacts")
    parser.add_argument(
        "--legacy-memory-dir", type=Path,
        default=HERE / "cearfn_v2_nested_artifacts")
    parser.add_argument(
        "--legacy-checkpoint-dir", type=Path,
        default=HERE / "inference_benchmark_checkpoints")
    parser.add_argument(
        "--config-file", type=Path,
        default=HERE / "pasgr_config_per_domain.json")
    parser.add_argument("--semantic-dir", type=Path)
    args = parser.parse_args()
    if args.pilot:
        args.seeds = [args.seeds[0]]
        args.oof_cap = min(args.oof_cap, 5_000)
        args.profile_cap = min(args.profile_cap, 1_000)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.output.read_text()) if args.output.exists() else {}

    for domain in args.domains:
        started = time.time()
        print(f"\n[DYBETA] === {domain} ===", flush=True)
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
        exclude_seen = domain not in REPEAT_PROTOCOL_DOMAINS
        config = cearf.CEARFConfig(exclude_seen=exclude_seen)
        inner_index = cearf.CEARFIndex(
            inner_fit_sessions, data["n_items"], config)
        profiles, profile_report = cearf.tune_profiles(
            inner_index, profile_queries)
        tune_index = cearf.CEARFIndex(
            tune_sessions, data["n_items"], config)
        final_index = cearf.CEARFIndex(
            sessions, data["n_items"], config)

        domain_dir = args.artifact_dir / domain.lower()
        domain_dir.mkdir(parents=True, exist_ok=True)
        gate_memory = load_or_build_memory_with_fallback(
            domain_dir / "gate_oof_memory.npz",
            (),
            inner_index,
            gate_queries,
            profiles,
            args.candidate_width,
            f"{domain}-gate-oof",
        )
        valid_memory = load_or_build_memory_with_fallback(
            domain_dir / "valid_memory.npz",
            (
                args.legacy_memory_dir
                / f"{domain.lower()}_nested_valid_memory.npz",
            ),
            tune_index,
            data["valid_queries"],
            profiles,
            args.candidate_width,
            f"{domain}-valid",
        )
        test_memory = load_or_build_memory_with_fallback(
            domain_dir / "test_memory.npz",
            (
                args.legacy_memory_dir
                / f"{domain.lower()}_nested_test_memory.npz",
            ),
            final_index,
            data["test_queries"],
            profiles,
            args.candidate_width,
            f"{domain}-test",
        )
        gate_keys = [str(value) for value in gate_memory["keys"]]
        valid_keys = [str(value) for value in valid_memory["keys"]]
        test_keys = [str(value) for value in test_memory["keys"]]
        gate_targets = targets_for(gate_keys, gate_queries)
        valid_targets = targets_for(valid_keys, data["valid_queries"])
        test_targets = targets_for(test_keys, data["test_queries"])

        if args.semantic_dir:
            semantic_path = (
                args.semantic_dir / f"{domain.lower()}_minilm.npy")
            semantic = np.load(semantic_path).astype(np.float32)
        else:
            semantic_path = None
            semantic = semantic_matrix(domain, data)
        gated_config = load_pasgr_config(domain, args.config_file)
        freq_inner = Counter(
            item for sequence in inner_fit_sessions.values()
            for item in sequence)
        freq_tune = Counter(
            item for sequence in tune_sessions.values()
            for item in sequence)
        freq_final = Counter(
            item for sequence in sessions.values()
            for item in sequence)
        head_inner = set(popularity_partition(
            freq_inner, data["n_items"])[0].tolist())
        head_tune = set(popularity_partition(
            freq_tune, data["n_items"])[0].tolist())
        head_final = set(popularity_partition(
            freq_final, data["n_items"])[0].tolist())

        profiles_identity = json.loads(json.dumps(profiles))
        identity = {
            "protocol": PRIMARY_PROTOCOL,
            "candidate_width": args.candidate_width,
            "epochs": args.epochs,
            "oof_fraction": args.oof_fraction,
            "oof_cap": args.oof_cap,
            "profile_cap": args.profile_cap,
            "valid_cap": args.valid_cap,
            "exclude_seen": exclude_seen,
            "inner_fit_sessions_fingerprint": session_fingerprint(
                inner_fit_sessions),
            "tune_sessions_fingerprint": session_fingerprint(tune_sessions),
            "final_sessions_fingerprint": session_fingerprint(sessions),
            "profile_query_fingerprint": split_report[
                "profile_query_fingerprint"],
            "gate_query_fingerprint": split_report[
                "gate_query_fingerprint"],
            "declared_validation_query_fingerprint": query_fingerprint(
                data["valid_queries"]),
            "test_query_fingerprint": query_fingerprint(data["test_queries"]),
            "semantic_teacher_fingerprint": array_fingerprint(semantic),
            "pasgr_config": gated_config,
            "memory_profiles": profiles_identity,
        }
        if domain in results:
            domain_block = results[domain]
            if str(domain_block.get("protocol")) not in LEGACY_PROTOCOL_ALIASES:
                raise RuntimeError(
                    f"{args.output}: {domain} belongs to another protocol")
            existing_identity = domain_block.get("identity")
            if (
                    existing_identity is not None
                    and json.loads(json.dumps(existing_identity)) != identity):
                raise RuntimeError(
                    f"{args.output}: {domain} protocol identity changed; "
                    "use a new output/artifact path")
            recorded_split = domain_block.get("split", {})
            for key in (
                    "profile_query_fingerprint",
                    "gate_query_fingerprint"):
                if recorded_split.get(key) != split_report[key]:
                    raise RuntimeError(
                        f"{args.output}: {domain} {key} changed; "
                        "refusing to mix resumed runs")
            if domain_block.get("profiles") != profiles_identity:
                raise RuntimeError(
                    f"{args.output}: {domain} memory profiles changed")
            domain_block["protocol"] = PRIMARY_PROTOCOL
            domain_block["identity"] = identity
            domain_block["split"] = split_report
        else:
            domain_block = {
                "domain": domain,
                "protocol": PRIMARY_PROTOCOL,
                "identity": identity,
                "profile_report": profile_report,
                "profiles": profiles_identity,
                "split": split_report,
                "runs": [],
            }
        completed = {int(run["seed"]) for run in domain_block["runs"]}
        for seed in args.seeds:
            if seed in completed:
                print(f"[DYBETA] {domain} seed={seed} already complete",
                      flush=True)
                continue
            seed_started = time.time()
            print(f"[DYBETA] {domain} seed={seed}: OOF experts", flush=True)
            checkpoint_dir = domain_dir / "checkpoints"
            prediction_dir = domain_dir / "predictions"
            gate_neural_keys, gate_neural = load_or_predict(
                prediction_dir / f"seed{seed}_gate_oof_top120.npz",
                checkpoint_dir / f"seed{seed}_gate_oof.pt",
                None,
                data,
                inner_fit_sessions,
                gate_queries,
                semantic,
                seed,
                args.epochs,
                gated_config,
                args.candidate_width,
                exclude_seen,
            )
            if gate_neural_keys != gate_keys:
                raise ValueError("gate memory/neural query order mismatch")
            gate_features = feature_matrix(
                gate_queries,
                gate_keys,
                gate_memory,
                gate_neural,
                freq_inner,
                head_inner,
                inner_index,
            )
            global_model, global_report = fit_global(
                gate_memory["selected"],
                gate_neural,
                gate_targets,
                seed,
            )
            primary_columns = FEATURE_GROUPS["context"]
            dynamic_model, dynamic_report = fit_dynamic(
                gate_features[:, primary_columns],
                gate_memory["selected"],
                gate_neural,
                gate_targets,
                seed,
                float(global_model.beta_),
                hidden=0,
                max_residual=.10,
            )
            dynamic_report["feature_names"] = [
                FEATURE_NAMES[column] for column in primary_columns]
            short_long_models, short_long_report = fit_short_long(
                gate_queries,
                gate_keys,
                gate_memory["selected"],
                gate_neural,
                gate_targets,
                seed,
                float(global_model.beta_),
                config.short_context,
            )

            ablation_models = {}
            ablation_reports = {}
            all_columns = tuple(range(len(FEATURE_NAMES)))
            feature_variants = {
                "context_mlp": (
                    FEATURE_GROUPS["context"], 16, .10),
                "full_linear": (
                    all_columns, 0, .10),
                "full_mlp": (
                    all_columns, 16, .10),
                "without_cross_expert": (tuple(
                    column for column in all_columns
                    if column not in FEATURE_GROUPS["cross_expert"]), 16, .10),
                "without_memory_certainty": (tuple(
                    column for column in all_columns
                    if column not in FEATURE_GROUPS["memory_certainty"]), 16, .10),
            }
            for name, (columns, hidden, max_residual) in (
                    feature_variants.items()):
                model, report = fit_dynamic(
                    gate_features[:, columns],
                    gate_memory["selected"],
                    gate_neural,
                    gate_targets,
                    seed,
                    float(global_model.beta_),
                    hidden=hidden,
                    max_residual=max_residual,
                )
                report["feature_names"] = [
                    FEATURE_NAMES[column] for column in columns]
                ablation_models[name] = (model, columns)
                ablation_reports[name] = report

            manifest = protocol_manifest(
                domain,
                seed,
                split_report,
                gated_config,
                profiles,
                global_report,
                dynamic_report,
                args,
            )
            manifest["profile_training"] = profile_report
            manifest["short_long_training"] = short_long_report
            manifest["feature_ablation_training"] = ablation_reports
            manifest_path = (
                domain_dir / f"seed{seed}_frozen_manifest.json")
            manifest_path.write_text(json.dumps(manifest, indent=2))

            print(
                f"[DYBETA] {domain} seed={seed}: official validation",
                flush=True)
            legacy_validation_checkpoint = (
                args.legacy_checkpoint_dir
                / f"{domain.lower()}_validation_seed{seed}.pt"
            )
            valid_neural_keys, valid_neural = load_or_predict(
                prediction_dir / f"seed{seed}_valid_top120.npz",
                checkpoint_dir / f"seed{seed}_valid.pt",
                legacy_validation_checkpoint,
                data,
                tune_sessions,
                data["valid_queries"],
                semantic,
                seed,
                args.epochs,
                gated_config,
                args.candidate_width,
                exclude_seen,
            )
            if valid_neural_keys != valid_keys:
                raise ValueError("validation memory/neural order mismatch")
            valid_features = feature_matrix(
                data["valid_queries"],
                valid_keys,
                valid_memory,
                valid_neural,
                freq_tune,
                head_tune,
                tune_index,
            )

            print(f"[DYBETA] {domain} seed={seed}: official test",
                  flush=True)
            legacy_final_checkpoint = (
                args.legacy_checkpoint_dir
                / f"{domain.lower()}_final_seed{seed}.pt"
            )
            test_neural_keys, test_neural = load_or_predict(
                prediction_dir / f"seed{seed}_test_top120.npz",
                checkpoint_dir / f"seed{seed}_test.pt",
                legacy_final_checkpoint,
                data,
                sessions,
                data["test_queries"],
                semantic,
                seed,
                args.epochs,
                gated_config,
                args.candidate_width,
                exclude_seen,
            )
            if test_neural_keys != test_keys:
                raise ValueError("test memory/neural order mismatch")
            test_features = feature_matrix(
                data["test_queries"],
                test_keys,
                test_memory,
                test_neural,
                freq_final,
                head_final,
                final_index,
            )

            split_payloads = {}
            rank_payload = {
                "valid_keys": np.asarray(valid_keys, dtype=str),
                "test_keys": np.asarray(test_keys, dtype=str),
            }
            for split_name, queries, keys, memory_arrays, neural, features, targets in (
                (
                    "validation",
                    data["valid_queries"],
                    valid_keys,
                    valid_memory,
                    valid_neural,
                    valid_features,
                    valid_targets,
                ),
                (
                    "test",
                    data["test_queries"],
                    test_keys,
                    test_memory,
                    test_neural,
                    test_features,
                    test_targets,
                ),
            ):
                memory_ranks = ranks_at_20(
                    memory_arrays["selected"], targets)
                neural_ranks = ranks_at_20(neural, targets)
                global_beta = global_model.predict(len(keys))
                short_long_beta = short_long_betas(
                    short_long_models,
                    queries,
                    keys,
                    float(global_model.beta_),
                    config.short_context,
                )
                dynamic_beta = dynamic_model.predict(
                    features[:, primary_columns])
                beta_variants = {
                    "fixed_05": np.full(
                        len(keys), .5, dtype=np.float32),
                    "oof_global": global_beta,
                    "oof_short_long": short_long_beta,
                    "dynamic": dynamic_beta,
                }
                for name, (model, columns) in ablation_models.items():
                    beta_variants[f"dynamic_{name}"] = model.predict(
                        features[:, columns])

                rankings = {
                    "memory_only": memory_arrays["selected"][:, :20],
                    "neural_only": neural[:, :20],
                }
                for name, beta in beta_variants.items():
                    rankings[name] = fuse_with_dynamic_beta(
                        memory_arrays["selected"],
                        neural,
                        beta,
                        constant=20.0,
                    )

                metrics = {}
                ranks_by_method = {}
                for name, ranking in rankings.items():
                    metrics[name], ranks_by_method[name] = evaluate_ranking(
                        ranking, targets, memory_ranks)
                metrics["dynamic"]["beta"] = beta_summary(dynamic_beta)
                metrics["oof_global"]["beta"] = beta_summary(global_beta)
                metrics["oof_short_long"]["beta"] = beta_summary(
                    short_long_beta)
                metrics["dynamic"]["beta_deciles"] = realized_beta_deciles(
                    dynamic_beta, memory_ranks, neural_ranks)

                sensitivity = {}
                if not args.skip_sensitivity:
                    for constant in RRF_CONSTANTS:
                        if constant == 20.0:
                            sensitivity["rrf_k20"] = metrics["dynamic"]
                            continue
                        global_k, _ = fit_global(
                            gate_memory["selected"],
                            gate_neural,
                            gate_targets,
                            seed,
                            constant=constant,
                            initial_beta=float(global_model.beta_),
                        )
                        dynamic_k, _ = fit_dynamic(
                            gate_features[:, primary_columns],
                            gate_memory["selected"],
                            gate_neural,
                            gate_targets,
                            seed,
                            float(global_k.beta_),
                            constant=constant,
                            hidden=0,
                            max_residual=.10,
                        )
                        beta_k = dynamic_k.predict(
                            features[:, primary_columns])
                        ranking_k = fuse_with_dynamic_beta(
                            memory_arrays["selected"],
                            neural,
                            beta_k,
                            constant=constant,
                        )
                        sensitivity[f"rrf_k{int(constant)}"], _ = (
                            evaluate_ranking(
                                ranking_k, targets, memory_ranks))
                        sensitivity[f"rrf_k{int(constant)}"]["beta"] = (
                            beta_summary(beta_k))
                split_payloads[split_name] = {
                    "metrics": metrics,
                    "fusion_sensitivity": sensitivity,
                }
                prefix = "valid_" if split_name == "validation" else "test_"
                for name, ranking in rankings.items():
                    rank_payload[f"{prefix}{name}_top20"] = (
                        ranking[:, :20].astype(np.int32))
                    rank_payload[f"{prefix}{name}_rank"] = (
                        ranks_by_method[name].astype(np.uint8))
                for name, beta in beta_variants.items():
                    rank_payload[f"{prefix}{name}_beta"] = (
                        beta.astype(np.float32))
                rank_payload[f"{prefix}features"] = features.astype(
                    np.float32)

            rank_artifact = (
                domain_dir / f"seed{seed}_dynamic_beta_ranks.npz")
            np.savez_compressed(rank_artifact, **rank_payload)
            gate_state = {
                **dynamic_model.state_dict_numpy(),
                "global_beta": np.asarray(global_model.beta_),
                "feature_names": np.asarray([
                    FEATURE_NAMES[column] for column in primary_columns
                ]),
            }
            np.savez_compressed(
                domain_dir / f"seed{seed}_dynamic_beta_gate.npz",
                **gate_state,
            )
            run = {
                "seed": seed,
                "manifest": str(manifest_path),
                "rank_artifact": str(rank_artifact),
                "training": {
                    "global": global_report,
                    "short_long": short_long_report,
                    "dynamic": dynamic_report,
                    "feature_ablations": ablation_reports,
                },
                **split_payloads,
                "seconds": time.time() - seed_started,
            }
            domain_block["runs"].append(run)
            results[domain] = domain_block
            args.output.write_text(json.dumps(results, indent=2))
            test_metrics = run["test"]["metrics"]
            print(
                f"[DYBETA] DONE {domain} seed={seed}: "
                f"memory={test_metrics['memory_only']['recall@20']:.5f} "
                f"global={test_metrics['oof_global']['recall@20']:.5f} "
                f"dynamic={test_metrics['dynamic']['recall@20']:.5f}",
                flush=True,
            )
            del gate_neural, valid_neural, test_neural
            gc.collect()

        domain_block["seconds_total_latest_invocation"] = (
            time.time() - started)
        results[domain] = domain_block
        args.output.write_text(json.dumps(results, indent=2))

    print(f"[DYBETA] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
