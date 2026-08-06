#!/usr/bin/env python3
"""Strict post-run provenance audit for CEARF-N dynamic-beta v2.

This program independently reconstructs the declared validation subset and
the two-part training-only OOF split, then verifies the persisted expert ranks
against fresh inference:

* CEARF transition/session/popularity/selected top-120 rows are rebuilt from
  the appropriate inner-fit, validation-fit, and final-fit indices.
* PASGR checkpoints are reloaded and exact full-catalog top-120 IDs are
  regenerated for gate OOF, declared validation, and test.

Ranking functions receive a target-free view containing query IDs and
contexts only. Targets are read solely to fingerprint the declared dataset
objects and to reproduce the training-only CEARF profile-selection audit; they
are never supplied to CEARF or PASGR rank construction.

The full audit is intentionally expensive. Unit tests exercise the split,
identity, target-isolation, CEARF, and PASGR checks on tiny fixtures.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import cearf
import loaders
import pasgr


HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")
SEEDS = (42, 123, 456)
MEMORY_COMPONENTS = ("transition", "session", "popularity", "selected")
REPEAT_PROTOCOL_DOMAINS = frozenset({"Diginetica_HID", "Tmall"})
AUDIT_SCHEMA = "dynamic-beta-v2-provenance-audit-v1"


class ProvenanceAuditError(RuntimeError):
    """Raised when a persisted artifact cannot be reproduced exactly."""


def stable_fraction(key: str) -> float:
    """Independent reconstruction of CEARF's stable BLAKE2 ordering."""
    digest = hashlib.blake2b(str(key).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64 - 1)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def session_fingerprint(
        sessions: Mapping[str, Sequence[int]]) -> str:
    """Match the session fingerprint stored by the dynamic-beta runner."""
    digest = hashlib.sha256()
    for uid in sorted(sessions):
        digest.update(str(uid).encode())
        digest.update(b":")
        digest.update(" ".join(map(str, sessions[uid])).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def labeled_query_fingerprint(queries: Mapping[str, Mapping[str, Any]]) -> str:
    """Match the cache fingerprint, including labels as data identity only."""
    digest = hashlib.sha256()
    for uid in sorted(queries):
        query = queries[uid]
        digest.update(str(uid).encode())
        digest.update(b"|")
        digest.update(
            " ".join(map(str, query.get("context", ()))).encode())
        digest.update(b"->")
        digest.update(
            " ".join(map(str, query.get("targets", ()))).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def context_query_fingerprint(
        queries: Mapping[str, Mapping[str, Any]]) -> str:
    """Fingerprint exactly the target-free information ranking may consume."""
    digest = hashlib.sha256()
    for uid in sorted(queries):
        digest.update(str(uid).encode())
        digest.update(b"|")
        digest.update(
            " ".join(map(str, queries[uid].get("context", ()))).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def ordered_key_fingerprint(keys: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(str(key).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def ranking_view(
        queries: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, list[int]]]:
    """Return a deep target-free query view for all expert inference."""
    return {
        str(uid): {
            "context": [
                int(item) for item in query.get("context", ())
            ],
        }
        for uid, query in queries.items()
    }


def query_identity(
        queries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    keys = sorted(str(uid) for uid in queries)
    return {
        "queries": len(keys),
        "ordered_keys_sha256": ordered_key_fingerprint(keys),
        "labeled_query_sha256": labeled_query_fingerprint(queries),
        "ranking_input_context_only_sha256": (
            context_query_fingerprint(queries)
        ),
        "ranking_input_fields": ["query_id", "context"],
        "target_labels_supplied_to_ranker": False,
    }


def session_identity(
        sessions: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    return {
        "sessions": len(sessions),
        "events": int(sum(len(sequence) for sequence in sessions.values())),
        "sha256": session_fingerprint(sessions),
    }


def canonical_validation_sources(
        queries: Mapping[str, Mapping[str, Any]]) -> set[str]:
    return {
        str(uid)[:-2] if str(uid).endswith("_v") else str(uid)
        for uid in queries
    }


def reconstruct_declared_validation(
        candidate_queries: Mapping[str, Mapping[str, Any]],
        cap: int,
) -> dict[str, dict[str, list[int]]]:
    """Reconstruct the runner's stable-hash declared validation subset."""
    if cap <= 0:
        raise ValueError("validation cap must be positive")
    if len(candidate_queries) > cap:
        keys = sorted(
            candidate_queries,
            key=lambda uid: stable_fraction(str(uid)),
        )[:cap]
    else:
        keys = list(candidate_queries)
    return {
        str(uid): {
            "context": [
                int(item)
                for item in candidate_queries[uid].get("context", ())
            ],
            "targets": [
                int(item)
                for item in candidate_queries[uid].get("targets", ())
            ],
        }
        for uid in keys
    }


def reconstruct_tune_sessions(
        sessions: Mapping[str, Sequence[int]],
        validation: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[int]]:
    """Independently reconstruct leakage-safe validation-fit sessions."""
    output = {
        str(uid): [int(item) for item in sequence]
        for uid, sequence in sessions.items()
    }
    for query_id, query in validation.items():
        key = str(query_id)
        if not key.endswith("_v"):
            continue
        source = key[:-2]
        if source not in output:
            continue
        context = [int(item) for item in query.get("context", ())]
        targets = [int(item) for item in query.get("targets", ())]
        if len(targets) == 1 and output[source] == context + targets:
            output[source] = context
    return output


def reconstruct_training_oof_split(
        sessions: Mapping[str, Sequence[int]],
        declared_validation_sources: set[str],
        fraction: float,
        cap: int,
        profile_cap: int,
) -> tuple[
    dict[str, list[int]],
    dict[str, dict[str, list[int]]],
    dict[str, dict[str, list[int]]],
    dict[str, Any],
]:
    """Independently reconstruct the v2 inner-fit/profile/gate split."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("OOF fraction must be in (0, 1]")
    if cap < 2 or profile_cap < 1:
        raise ValueError("OOF cap must be >=2 and profile cap >=1")
    eligible = [
        str(uid)
        for uid, sequence in sessions.items()
        if (
            len(sequence) >= 3
            and str(uid) not in declared_validation_sources
        )
    ]
    ordered = sorted(
        eligible,
        key=lambda uid: stable_fraction(f"dynamic-beta::{uid}"),
    )
    wanted = min(cap, max(2, int(len(ordered) * fraction)))
    held = ordered[:wanted]
    n_profile = min(profile_cap, max(1, len(held) // 4))
    profile_sources = held[:n_profile]
    gate_sources = held[n_profile:]
    if not gate_sources:
        raise ProvenanceAuditError(
            "reconstructed OOF split contains no gate queries")
    held_set = set(held)
    inner_fit = {
        str(uid): [int(item) for item in sequence]
        for uid, sequence in sessions.items()
        if str(uid) not in held_set
    }

    def make_queries(
            source_ids: Sequence[str], prefix: str
    ) -> dict[str, dict[str, list[int]]]:
        return {
            f"{prefix}::{uid}": {
                "context": [int(item) for item in sessions[uid][:-1]],
                "targets": [int(sessions[uid][-1])],
            }
            for uid in source_ids
        }

    profile_queries = make_queries(profile_sources, "profile-oof")
    gate_queries = make_queries(gate_sources, "gate-oof")
    report = {
        "split_method": "stable BLAKE2 ordering over source-session IDs",
        "fraction": fraction,
        "cap": cap,
        "eligible_source_sessions": len(eligible),
        "excluded_declared_validation_sources": len(
            declared_validation_sources),
        "held_source_sessions": len(held),
        "profile_source_sessions": len(profile_sources),
        "gate_source_sessions": len(gate_sources),
        "profile_gate_source_overlap": len(
            set(profile_sources) & set(gate_sources)),
        "declared_validation_source_overlap": len(
            set(held) & declared_validation_sources),
        "profile_query_fingerprint": labeled_query_fingerprint(
            profile_queries),
        "gate_query_fingerprint": labeled_query_fingerprint(gate_queries),
    }
    return inner_fit, profile_queries, gate_queries, report


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceAuditError(message)


def _require_file(path: Path) -> None:
    _require(path.is_file(), f"required artifact is missing: {path}")


def _artifact_identity(path: Path) -> dict[str, Any]:
    _require_file(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def _first_array_difference(
        actual: np.ndarray, expected: np.ndarray) -> dict[str, Any] | None:
    if actual.shape != expected.shape:
        return {
            "shape_actual": list(actual.shape),
            "shape_expected": list(expected.shape),
        }
    unequal = np.argwhere(actual != expected)
    if not unequal.size:
        return None
    index = tuple(int(value) for value in unequal[0])
    return {
        "index": list(index),
        "actual": int(actual[index]),
        "expected": int(expected[index]),
        "total_unequal": int(np.count_nonzero(actual != expected)),
    }


def _expected_memory_row(
        index: cearf.CEARFIndex,
        context: Sequence[int],
        profiles: Mapping[str, Sequence[float]],
        width: int,
) -> dict[str, np.ndarray]:
    """Build expert rows from context only; no target parameter exists."""
    context_only = [int(item) for item in context]
    components = index.component_rankings(context_only)
    output = {
        name: np.zeros(width, dtype=np.int32)
        for name in MEMORY_COMPONENTS
    }
    for name, ranking in zip(MEMORY_COMPONENTS[:3], components):
        take = min(width, len(ranking))
        output[name][:take] = np.asarray(ranking[:take], dtype=np.int32)
    regime = (
        "short"
        if len(context_only) <= index.config.short_context
        else "long"
    )
    selected = index.fuse_rankings(
        context_only,
        components,
        tuple(float(value) for value in profiles[regime]),
        width,
    )
    output["selected"][:len(selected)] = np.asarray(
        selected, dtype=np.int32)
    return output


def replay_profile_lock(
        index: cearf.CEARFIndex,
        profile_queries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, tuple[float, float, float]], dict[str, Any]]:
    """Replay profile selection while keeping labels outside rank creation."""
    target_free = ranking_view(profile_queries)
    targets = {
        str(uid): {
            int(item) for item in query.get("targets", ())
        }
        for uid, query in profile_queries.items()
    }
    selected: dict[str, tuple[float, float, float]] = {}
    report: dict[str, Any] = {}
    for regime in ("short", "long"):
        keys = [
            uid for uid, query in target_free.items()
            if (
                len(query["context"]) <= index.config.short_context
            ) == (regime == "short")
        ]
        if not keys:
            name = "short_safe" if regime == "short" else "session_transition"
            selected[regime] = cearf.PROFILES[name]
            report[regime] = {
                "profile": name, "n": 0, "score": None}
            continue
        components = {
            uid: index.component_rankings(target_free[uid]["context"])
            for uid in keys
        }
        best = None
        for name, profile in cearf.PROFILES.items():
            hits6 = 0
            hits20 = 0
            for uid in keys:
                ranking = index.fuse_rankings(
                    target_free[uid]["context"],
                    components[uid],
                    profile,
                    20,
                )
                hits6 += bool(targets[uid].intersection(ranking[:6]))
                hits20 += bool(targets[uid].intersection(ranking[:20]))
            recall6 = hits6 / len(keys)
            recall20 = hits20 / len(keys)
            score = 0.5 * recall6 + 0.5 * recall20
            candidate = (
                score, recall20, recall6, name, profile)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            raise ProvenanceAuditError(
                f"profile replay produced no candidate for {regime}")
        selected[regime] = best[-1]
        report[regime] = {
            "profile": best[-2],
            "n": len(keys),
            "score": best[0],
            "recall@6": best[2],
            "recall@20": best[1],
        }
    return selected, report


def audit_memory_cache(
        path: Path,
        index: cearf.CEARFIndex,
        queries: Mapping[str, Mapping[str, Any]],
        profiles: Mapping[str, Sequence[float]],
        width: int,
        progress_label: str | None = None,
) -> dict[str, Any]:
    """Verify every CEARF cache cell against fresh context-only inference."""
    identity = _artifact_identity(path)
    expected_keys = sorted(str(uid) for uid in queries)
    expected_fingerprint = labeled_query_fingerprint(queries)
    expected_profiles = json.dumps(profiles, sort_keys=True)
    required = {
        *MEMORY_COMPONENTS, "keys", "fingerprint", "profiles"}
    with np.load(path, allow_pickle=False) as saved:
        _require(
            required.issubset(saved.files),
            f"{path}: missing arrays {sorted(required - set(saved.files))}",
        )
        # ``NpzFile.__getitem__`` decompresses an array on every access.
        # Materialize each compressed field once before the row-wise exact
        # comparison; indexing ``saved[name]`` inside the loop would inflate
        # the complete array hundreds of thousands of times.
        actual_keys = [str(value) for value in saved["keys"]]
        stored_fingerprint = str(saved["fingerprint"].item())
        stored_profiles = str(saved["profiles"].item())
        stored_components = {
            name: saved[name] for name in MEMORY_COMPONENTS
        }
        _require(
            actual_keys == expected_keys,
            f"{path}: query key order differs from sorted reconstruction",
        )
        _require(
            stored_fingerprint == expected_fingerprint,
            f"{path}: labeled query fingerprint mismatch",
        )
        _require(
            stored_profiles == expected_profiles,
            f"{path}: selected profile JSON mismatch",
        )
        for name in MEMORY_COMPONENTS:
            component = stored_components[name]
            _require(
                component.shape == (len(expected_keys), width),
                f"{path}:{name}: expected shape "
                f"{(len(expected_keys), width)}, got {component.shape}",
            )
            _require(
                np.issubdtype(component.dtype, np.integer),
                f"{path}:{name}: expected integer dtype, got "
                f"{component.dtype}",
            )
        target_free = ranking_view(queries)
        for row, uid in enumerate(expected_keys):
            expected = _expected_memory_row(
                index,
                target_free[uid]["context"],
                profiles,
                width,
            )
            for name in MEMORY_COMPONENTS:
                actual = stored_components[name][row]
                difference = _first_array_difference(actual, expected[name])
                _require(
                    difference is None,
                    f"{path}:{name}: exact mismatch for row={row}, "
                    f"query={uid}: {difference}",
                )
            if (
                progress_label
                and (row + 1) % 10_000 == 0
            ):
                print(
                    f"[PROVENANCE] {progress_label} CEARF "
                    f"{row + 1}/{len(expected_keys)}",
                    flush=True,
                )
        dtypes = {
            name: str(stored_components[name].dtype)
            for name in MEMORY_COMPONENTS
        }
    return {
        **identity,
        "query_identity": query_identity(queries),
        "rows": len(expected_keys),
        "width": width,
        "arrays_exact": {
            name: True for name in MEMORY_COMPONENTS
        },
        "dtypes": dtypes,
        "ranking_used_target_labels": False,
    }


def _device_from_name(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        _require(torch.cuda.is_available(), "CUDA requested but unavailable")
    if name == "mps":
        _require(
            torch.backends.mps.is_available(),
            "MPS requested but unavailable",
        )
    return torch.device(name)


def load_frozen_pasgr(
        checkpoint_path: Path,
        expected_sessions: Mapping[str, Sequence[int]],
        expected_seed: int,
        manifest: Mapping[str, Any],
        device: torch.device,
) -> tuple[pasgr.PASGRModel, dict[str, Any]]:
    identity = _artifact_identity(checkpoint_path)
    saved = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True)
    _require(
        isinstance(saved, dict),
        f"{checkpoint_path}: checkpoint payload is not a mapping",
    )
    _require(
        {"config", "state_dict", "sessions_fingerprint"}.issubset(saved),
        f"{checkpoint_path}: missing checkpoint provenance fields",
    )
    expected_fingerprint = session_fingerprint(expected_sessions)
    _require(
        str(saved["sessions_fingerprint"]) == expected_fingerprint,
        f"{checkpoint_path}: training-session fingerprint mismatch; "
        f"stored={saved['sessions_fingerprint']} "
        f"reconstructed={expected_fingerprint}",
    )
    config = pasgr.PASGRConfig(**saved["config"])
    _require(
        int(config.seed) == int(expected_seed),
        f"{checkpoint_path}: config seed={config.seed}, "
        f"requested seed={expected_seed}",
    )
    _require(
        int(config.top_k) == int(manifest["candidate_width"]),
        f"{checkpoint_path}: checkpoint top_k={config.top_k}, "
        f"manifest width={manifest['candidate_width']}",
    )
    for key, expected in manifest.get("pasgr_config", {}).items():
        actual = getattr(config, key)
        _require(
            actual == expected,
            f"{checkpoint_path}: PASGR config {key}={actual!r}, "
            f"manifest={expected!r}",
        )
    n_items = int(saved["state_dict"]["item.weight"].shape[0])
    model = pasgr.PASGRModel(
        np.zeros((n_items, config.dim), dtype=np.float32),
        config,
    )
    model.load_state_dict(saved["state_dict"], strict=True)
    model = model.to(device).eval()
    identity.update({
        "stored_protocol": saved.get("protocol"),
        "training_sessions_sha256": expected_fingerprint,
        "config": asdict(config),
        "config_sha256": sha256_json(asdict(config)),
        "state_dict_tensor_count": len(saved["state_dict"]),
    })
    return model, identity


@torch.no_grad()
def regenerate_pasgr_topk_and_compare(
        model: pasgr.PASGRModel,
        queries: Mapping[str, Mapping[str, Any]],
        n_items: int,
        cache_path: Path,
        width: int,
        exclude_seen: bool,
        batch_size: int,
        progress_label: str | None = None,
) -> dict[str, Any]:
    """Independently regenerate exact full-catalog IDs from contexts only."""
    identity = _artifact_identity(cache_path)
    target_free = ranking_view(queries)
    expected_keys = sorted(target_free)
    expected_fingerprint = labeled_query_fingerprint(queries)
    with np.load(cache_path, allow_pickle=False) as saved:
        _require(
            {"keys", "rankings", "fingerprint"}.issubset(saved.files),
            f"{cache_path}: prediction cache schema is incomplete",
        )
        # Materialize the compressed rankings exactly once. Slicing through
        # the NpzFile in every inference batch would re-inflate the full array.
        cached_keys = [str(value) for value in saved["keys"]]
        cached_fingerprint = str(saved["fingerprint"].item())
        cached_rankings = saved["rankings"].astype(
            np.int32, copy=False)
        _require(
            cached_keys == expected_keys,
            f"{cache_path}: query key order mismatch",
        )
        _require(
            cached_fingerprint == expected_fingerprint,
            f"{cache_path}: labeled query fingerprint mismatch",
        )
        expected_width = min(width, n_items - 1)
        _require(
            cached_rankings.shape == (
                len(expected_keys), expected_width),
            f"{cache_path}: ranking shape mismatch; "
            f"expected={(len(expected_keys), expected_width)}, "
            f"actual={cached_rankings.shape}",
        )
        _require(
            np.issubdtype(cached_rankings.dtype, np.integer),
            f"{cache_path}: ranking dtype must be integer",
        )

        device = next(model.parameters()).device
        catalog = F.normalize(model.item.weight, dim=-1)
        max_seq = int(model.config.max_seq)
        for start in range(0, len(expected_keys), batch_size):
            batch_keys = expected_keys[start:start + batch_size]
            sequences = [
                [
                    int(item)
                    for item in target_free[uid]["context"]
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
                    contexts[row, :len(sequence)] = torch.as_tensor(
                        sequence, device=device)
                    lengths[row] = len(sequence)
            query = model.encode(contexts, lengths)
            scores = query @ catalog.T
            scores[:, 0] = -torch.inf
            if exclude_seen:
                for row, sequence in enumerate(sequences):
                    if sequence:
                        scores[
                            row,
                            torch.as_tensor(
                                list(set(sequence)), device=device),
                        ] = -torch.inf
            regenerated = torch.topk(
                scores, k=expected_width, dim=-1
            ).indices.cpu().numpy().astype(np.int32, copy=False)
            cached = cached_rankings[
                start:start + len(batch_keys)
            ]
            difference = _first_array_difference(cached, regenerated)
            if difference is not None:
                local_row = int(difference.get("index", [0])[0])
                difference["global_row"] = start + local_row
                difference["query"] = batch_keys[local_row]
            _require(
                difference is None,
                f"{cache_path}: exact PASGR top-{expected_width} "
                f"mismatch: {difference}",
            )
            if progress_label and (
                start + len(batch_keys)
            ) % 10_000 < batch_size:
                print(
                    f"[PROVENANCE] {progress_label} PASGR "
                    f"{start + len(batch_keys)}/{len(expected_keys)}",
                    flush=True,
                )
    return {
        **identity,
        "query_identity": query_identity(queries),
        "rows": len(expected_keys),
        "width": expected_width,
        "full_catalog_scoring": True,
        "topk_ids_exact": True,
        "ranking_used_target_labels": False,
        "exclude_seen": bool(exclude_seen),
        "inference_device": str(next(model.parameters()).device),
    }


def _normalize_profiles(
        profiles: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    return {
        str(regime): [float(value) for value in values]
        for regime, values in profiles.items()
    }


def _manifest_split_value(
        split: Mapping[str, Any], current: str, legacy: str | None = None
) -> Any:
    if current in split:
        return split[current]
    if legacy and legacy in split:
        return split[legacy]
    raise ProvenanceAuditError(
        f"manifest split is missing {current!r}"
        + (f" (legacy alias {legacy!r})" if legacy else "")
    )


def verify_manifest(
        manifest_path: Path,
        domain: str,
        seed: int,
        split_report: Mapping[str, Any],
        profiles: Mapping[str, Sequence[float]],
        width: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _artifact_identity(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    _require(
        manifest.get("domain") == domain,
        f"{manifest_path}: domain mismatch",
    )
    _require(
        int(manifest.get("seed", -1)) == int(seed),
        f"{manifest_path}: seed mismatch",
    )
    _require(
        int(manifest.get("candidate_width", -1)) == int(width),
        f"{manifest_path}: candidate width mismatch",
    )
    _require(
        manifest.get("validation_labels_used_for_beta") is False
        and manifest.get("test_labels_used_for_beta") is False,
        f"{manifest_path}: beta provenance flags are not leakage-safe",
    )
    _require(
        str(manifest.get("protocol", "")).startswith(
            "dynamic-beta-train-only-v2"),
        f"{manifest_path}: not a dynamic-beta train-only v2 manifest",
    )
    frozen_before_evaluation = manifest.get(
        "frozen_before_declared_validation_or_test_evaluation",
        manifest.get(
            "frozen_before_official_validation_or_test_evaluation_under_protocol",
            manifest.get("created_before_official_validation_prediction"),
        ),
    )
    _require(
        frozen_before_evaluation is True,
        f"{manifest_path}: missing true pre-evaluation freeze declaration",
    )
    manifest_split = manifest.get("split", {})
    direct_fields = (
        "fraction",
        "cap",
        "eligible_source_sessions",
        "held_source_sessions",
        "profile_source_sessions",
        "gate_source_sessions",
        "profile_gate_source_overlap",
        "profile_query_fingerprint",
        "gate_query_fingerprint",
    )
    for key in direct_fields:
        _require(
            manifest_split.get(key) == split_report[key],
            f"{manifest_path}: split {key} mismatch; "
            f"stored={manifest_split.get(key)!r}, "
            f"reconstructed={split_report[key]!r}",
        )
    excluded = _manifest_split_value(
        manifest_split,
        "excluded_declared_validation_sources",
        "excluded_official_validation_sources",
    )
    overlap = _manifest_split_value(
        manifest_split,
        "declared_validation_source_overlap",
        "official_validation_source_overlap",
    )
    _require(
        excluded == split_report["excluded_declared_validation_sources"],
        f"{manifest_path}: excluded validation-source count mismatch",
    )
    _require(
        overlap == split_report["declared_validation_source_overlap"] == 0,
        f"{manifest_path}: OOF/validation source overlap is nonzero",
    )
    optional_split_fields = (
        "validation_candidate_pool_queries",
        "validation_candidate_pool_sources",
        "declared_validation_queries",
        "declared_validation_sources",
        "validation_candidate_pool_source_overlap",
    )
    for key in optional_split_fields:
        if key in manifest_split:
            _require(
                manifest_split[key] == split_report[key],
                f"{manifest_path}: optional split {key} mismatch",
            )
    stored_profiles = _normalize_profiles(
        manifest.get("memory_profiles", {}))
    _require(
        stored_profiles == _normalize_profiles(profiles),
        f"{manifest_path}: memory profile mismatch",
    )
    identity["content_json_sha256"] = sha256_json(manifest)
    identity["protocol"] = manifest.get("protocol")
    return manifest, identity


def _verify_result_entry(
        results: Mapping[str, Any],
        domain: str,
        seed: int,
        manifest_path: Path,
        rank_artifact_path: Path,
) -> dict[str, Any]:
    _require(domain in results, f"results JSON has no domain {domain}")
    matches = [
        run for run in results[domain].get("runs", [])
        if int(run.get("seed", -1)) == int(seed)
    ]
    _require(
        len(matches) == 1,
        f"results JSON expected one {domain} seed={seed} run, "
        f"found {len(matches)}",
    )
    declared_manifest = matches[0].get("manifest")
    declared_rank_artifact = matches[0].get("rank_artifact")
    _require(
        isinstance(declared_manifest, str) and bool(declared_manifest),
        f"results JSON {domain} seed={seed} has no manifest path",
    )
    _require(
        isinstance(declared_rank_artifact, str) and bool(declared_rank_artifact),
        f"results JSON {domain} seed={seed} has no rank-artifact path",
    )
    resolved_manifest = Path(declared_manifest)
    if not resolved_manifest.is_absolute():
        resolved_manifest = HERE / resolved_manifest
    resolved_rank_artifact = Path(declared_rank_artifact)
    if not resolved_rank_artifact.is_absolute():
        resolved_rank_artifact = HERE / resolved_rank_artifact
    _require(
        resolved_manifest.resolve() == manifest_path.resolve(),
        f"results JSON {domain} seed={seed} points to a different manifest",
    )
    _require(
        resolved_rank_artifact.resolve() == rank_artifact_path.resolve(),
        f"results JSON {domain} seed={seed} points to a different rank artifact",
    )
    return {
        "present": True,
        "declared_manifest": declared_manifest,
        "declared_rank_artifact": declared_rank_artifact,
        "resolved_manifest": str(resolved_manifest.resolve()),
        "resolved_rank_artifact": str(resolved_rank_artifact.resolve()),
    }


def _audit_rank_artifact_keys(
        path: Path,
        validation: Mapping[str, Mapping[str, Any]],
        test: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    identity = _artifact_identity(path)
    with np.load(path, allow_pickle=False) as saved:
        _require(
            {"valid_keys", "test_keys"}.issubset(saved.files),
            f"{path}: dynamic rank artifact lacks query keys",
        )
        _require(
            [str(value) for value in saved["valid_keys"]]
            == sorted(validation),
            f"{path}: dynamic rank validation keys mismatch",
        )
        _require(
            [str(value) for value in saved["test_keys"]]
            == sorted(test),
            f"{path}: dynamic rank test keys mismatch",
        )
        identity["arrays"] = sorted(saved.files)
    identity["query_keys_exact"] = True
    return identity


def audit_domain(
        domain: str,
        seeds: Sequence[int],
        artifact_root: Path,
        results: Mapping[str, Any],
        valid_cap: int,
        oof_fraction: float,
        oof_cap: int,
        profile_cap: int,
        width: int,
        device: torch.device,
        batch_size: int,
) -> dict[str, Any]:
    print(f"[PROVENANCE] reconstructing {domain}", flush=True)
    data = loaders.ALL_LOADERS[domain]()
    candidate_validation = data["valid_queries"]
    declared_validation = reconstruct_declared_validation(
        candidate_validation, valid_cap)
    train_sessions = {
        str(uid): [int(item) for item in sequence]
        for uid, sequence in data["train_sessions"].items()
    }
    tune_sessions = reconstruct_tune_sessions(
        train_sessions, declared_validation)
    validation_sources = canonical_validation_sources(
        declared_validation)
    (
        inner_fit_sessions,
        profile_queries,
        gate_queries,
        split_report,
    ) = reconstruct_training_oof_split(
        tune_sessions,
        validation_sources,
        oof_fraction,
        oof_cap,
        profile_cap,
    )
    held_sources = {
        str(uid).split("::", 1)[1]
        for uid in (*profile_queries, *gate_queries)
    }
    candidate_sources = canonical_validation_sources(
        candidate_validation)
    split_report.update({
        "validation_candidate_pool_queries": len(candidate_validation),
        "validation_candidate_pool_sources": len(candidate_sources),
        "declared_validation_queries": len(declared_validation),
        "declared_validation_sources": len(validation_sources),
        "validation_candidate_pool_source_overlap": len(
            held_sources & candidate_sources),
    })

    domain_dir = artifact_root / domain.lower()
    first_manifest_path = (
        domain_dir / f"seed{int(seeds[0])}_frozen_manifest.json")
    _require_file(first_manifest_path)
    first_manifest = json.loads(first_manifest_path.read_text())
    declared_profiles = _normalize_profiles(
        first_manifest.get("memory_profiles", {}))

    exclude_seen = domain not in REPEAT_PROTOCOL_DOMAINS
    cearf_config = cearf.CEARFConfig(exclude_seen=exclude_seen)
    inner_index = cearf.CEARFIndex(
        inner_fit_sessions, data["n_items"], cearf_config)
    validation_index = cearf.CEARFIndex(
        tune_sessions, data["n_items"], cearf_config)
    test_index = cearf.CEARFIndex(
        train_sessions, data["n_items"], cearf_config)

    retuned_profiles, retuned_report = replay_profile_lock(
        inner_index, profile_queries)
    normalized_retuned = _normalize_profiles(retuned_profiles)
    _require(
        normalized_retuned == declared_profiles,
        f"{domain}: independently selected CEARF profiles differ; "
        f"stored={declared_profiles}, reconstructed={normalized_retuned}",
    )

    query_sets = {
        "gate_oof": gate_queries,
        "declared_validation": declared_validation,
        "test": data["test_queries"],
    }
    session_sets = {
        "gate_oof_inner_fit": inner_fit_sessions,
        "declared_validation_fit": tune_sessions,
        "test_final_fit": train_sessions,
    }
    memory_specs = {
        "gate_oof": (
            domain_dir / "gate_oof_memory.npz",
            inner_index,
            gate_queries,
        ),
        "declared_validation": (
            domain_dir / "valid_memory.npz",
            validation_index,
            declared_validation,
        ),
        "test": (
            domain_dir / "test_memory.npz",
            test_index,
            data["test_queries"],
        ),
    }
    memory_audits = {}
    for split, (path, index, queries) in memory_specs.items():
        memory_audits[split] = audit_memory_cache(
            path,
            index,
            queries,
            declared_profiles,
            width,
            progress_label=f"{domain}/{split}",
        )

    manifest_audits = {}
    seed_audits = {}
    pasgr_specs = {
        "gate_oof": (
            inner_fit_sessions,
            gate_queries,
            "gate_oof",
        ),
        "declared_validation": (
            tune_sessions,
            declared_validation,
            "valid",
        ),
        "test": (
            train_sessions,
            data["test_queries"],
            "test",
        ),
    }
    for seed in seeds:
        seed = int(seed)
        print(f"[PROVENANCE] {domain} seed={seed}", flush=True)
        manifest_path = domain_dir / f"seed{seed}_frozen_manifest.json"
        manifest, manifest_identity = verify_manifest(
            manifest_path,
            domain,
            seed,
            split_report,
            declared_profiles,
            width,
        )
        manifest_audits[str(seed)] = manifest_identity
        dynamic_gate = (
            domain_dir / f"seed{seed}_dynamic_beta_gate.npz")
        dynamic_ranks = (
            domain_dir / f"seed{seed}_dynamic_beta_ranks.npz")
        result_entry = _verify_result_entry(
            results,
            domain,
            seed,
            manifest_path,
            dynamic_ranks,
        )
        split_audits = {}
        for split, (
            fit_sessions,
            queries,
            artifact_stem,
        ) in pasgr_specs.items():
            checkpoint_path = (
                domain_dir / "checkpoints"
                / f"seed{seed}_{artifact_stem}.pt"
            )
            cache_path = (
                domain_dir / "predictions"
                / f"seed{seed}_{artifact_stem}_top120.npz"
            )
            model, checkpoint_identity = load_frozen_pasgr(
                checkpoint_path,
                fit_sessions,
                seed,
                manifest,
                device,
            )
            _require(
                model.item.num_embeddings == int(data["n_items"]),
                f"{checkpoint_path}: checkpoint catalog size "
                f"{model.item.num_embeddings} != loader {data['n_items']}",
            )
            cache_audit = regenerate_pasgr_topk_and_compare(
                model,
                queries,
                data["n_items"],
                cache_path,
                width,
                exclude_seen,
                batch_size,
                progress_label=f"{domain}/seed{seed}/{split}",
            )
            split_audits[split] = {
                "checkpoint": checkpoint_identity,
                "prediction_cache": cache_audit,
            }
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        seed_audits[str(seed)] = {
            "results_entry": result_entry,
            "pasgr": split_audits,
            "dynamic_beta_gate_artifact": _artifact_identity(
                dynamic_gate),
            "dynamic_beta_rank_artifact": _audit_rank_artifact_keys(
                dynamic_ranks,
                declared_validation,
                data["test_queries"],
            ),
        }

    return {
        "status": "pass",
        "catalog_items_including_padding": int(data["n_items"]),
        "ranking_contract": {
            "inputs": ["query_id", "context"],
            "target_labels_supplied_to_cearf": False,
            "target_labels_supplied_to_pasgr": False,
            "full_catalog_pasgr_scoring": True,
            "exact_id_equality_required": True,
            "profile_targets_used_only_to_replay_training_only_profile_lock": (
                True
            ),
        },
        "split_reconstruction": split_report,
        "identities": {
            "sessions": {
                name: session_identity(sessions)
                for name, sessions in session_sets.items()
            },
            "queries": {
                "validation_candidate_pool": query_identity(
                    candidate_validation),
                "profile_oof": query_identity(profile_queries),
                **{
                    name: query_identity(queries)
                    for name, queries in query_sets.items()
                },
            },
            "cearf_config": {
                "value": asdict(cearf_config),
                "sha256": sha256_json(asdict(cearf_config)),
            },
            "memory_profiles": {
                "value": declared_profiles,
                "sha256": sha256_json(declared_profiles),
            },
        },
        "profile_lock_reconstruction": {
            "selected_profiles_exact": True,
            "selected_profiles": normalized_retuned,
            "report": retuned_report,
        },
        "memory_artifacts": memory_audits,
        "manifest_artifacts": manifest_audits,
        "seeds": seed_audits,
    }


def audit_all(args: argparse.Namespace) -> dict[str, Any]:
    _require_file(args.results)
    results = json.loads(args.results.read_text())
    device = _device_from_name(args.device)
    report = {
        "audit_schema": AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "scope": {
            "domains": list(args.domains),
            "seeds": [int(seed) for seed in args.seeds],
            "valid_cap": int(args.valid_cap),
            "oof_fraction": float(args.oof_fraction),
            "oof_cap": int(args.oof_cap),
            "profile_cap": int(args.profile_cap),
            "candidate_width": int(args.candidate_width),
            "batch_size": int(args.batch_size),
            "inference_device": str(device),
        },
        "guarantees": {
            "ranking_input_is_target_free": True,
            "cearf_every_persisted_top120_cell_checked": True,
            "pasgr_full_catalog_top120_exact_equality_required": True,
            "requested_seed_split_cartesian_product_required": True,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_mps_available": bool(torch.backends.mps.is_available()),
            "torch_cuda_available": bool(torch.cuda.is_available()),
        },
        "results_artifact": _artifact_identity(args.results),
        "domains": {},
    }
    for domain in args.domains:
        _require(
            domain in loaders.ALL_LOADERS,
            f"unknown loader domain: {domain}",
        )
        report["domains"][domain] = audit_domain(
            domain,
            args.seeds,
            args.artifact_dir,
            results,
            args.valid_cap,
            args.oof_fraction,
            args.oof_cap,
            args.profile_cap,
            args.candidate_width,
            device,
            args.batch_size,
        )
    report["status"] = "pass"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly reproduce dynamic-beta v2 split and expert caches. "
            "This is a full, expensive post-run audit."
        )
    )
    parser.add_argument(
        "domains", nargs="*", default=list(DOMAINS),
        choices=sorted(loaders.ALL_LOADERS),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--valid-cap", type=int, default=5_000)
    parser.add_argument("--oof-fraction", type=float, default=0.10)
    parser.add_argument("--oof-cap", type=int, default=5_000)
    parser.add_argument("--profile-cap", type=int, default=1_000)
    parser.add_argument("--candidate-width", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help=(
            "Use the same backend that created prediction caches when exact "
            "top-k ties may be backend-sensitive."
        ),
    )
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_artifacts",
    )
    parser.add_argument(
        "--results", type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_results.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_provenance_audit.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_all(args)
    except Exception as error:
        report = {
            "audit_schema": AUDIT_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "ranking_input_is_target_free_by_construction": True,
        }
        write_json_atomic(args.output, report)
        print(
            f"[PROVENANCE] FAIL: {error}; report={args.output}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    write_json_atomic(args.output, report)
    print(f"[PROVENANCE] PASS: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
