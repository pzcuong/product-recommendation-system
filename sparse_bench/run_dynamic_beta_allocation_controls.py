#!/usr/bin/env python3
"""Post-run allocation controls for training-calibrated CEARF-N.

This script is intentionally downstream of ``run_dynamic_beta.py``.  It never
trains or predicts an expert.  Instead, it reconstructs the exact OOF
calibration queries and features, reads the already frozen 120-wide memory and
neural rank lists, fits allocation-only controls on OOF training targets, and
evaluates the frozen policies on the saved full test rank lists.

The controls are:

* two OOF scalar policies split by head/tail;
* four OOF scalar policies split by short/long x head/tail;
* bounded linear gates with Delta in {0.05, 0.10, 0.20};
* a deterministic permutation of the primary test betas that preserves their
  marginal distribution but breaks the learned query-to-beta assignment;
* length-only, frequency-only, tail-only and three drop-one gates;
* the primary gate without admission cost or without the prior penalty.

No validation or test label is passed to any fitting routine.  A per-seed
manifest containing the fitted policy states and all input identities is
written before test targets are materialized or evaluated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import cearf
import loaders
from dynamic_beta import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    TrainOnlyDynamicBeta,
    TrainOnlyGlobalBeta,
    feature_matrix,
    fuse_with_dynamic_beta,
    rank_evidence_training_arrays,
)
from run_cearfn_evidence import (
    metrics_from_ranks,
    popularity_partition,
    query_fingerprint,
    ranks_at_20,
    targets_for,
)
from run_cearfn_v2 import REPEAT_PROTOCOL_DOMAINS
from run_dynamic_beta import (
    LEGACY_PROTOCOL_ALIASES,
    array_fingerprint,
    canonical_validation_sources,
    make_training_oof_split,
)
from summarize_dynamic_beta import _discrete_bootstrap, _per_query_value
from validation_protocol import hold_out_validation_targets


HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")
SEEDS = (42, 123, 456)
CONTROL_PROTOCOL = "dynamic-beta-allocation-controls-v2-assignment-shuffle"
PRIMARY_COLUMNS = tuple(FEATURE_GROUPS["context"])
METRICS = (
    "recall@6",
    "ndcg@6",
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "utility",
)


@dataclass(frozen=True)
class DynamicControlSpec:
    columns: tuple[int, ...]
    max_residual: float = 0.10
    admission_cost: float = 1e-3
    prior_penalty: float = 1e-3


DYNAMIC_CONTROL_SPECS = {
    "dynamic_delta_005": DynamicControlSpec(
        PRIMARY_COLUMNS, max_residual=0.05),
    "dynamic_delta_010": DynamicControlSpec(
        PRIMARY_COLUMNS, max_residual=0.10),
    "dynamic_delta_020": DynamicControlSpec(
        PRIMARY_COLUMNS, max_residual=0.20),
    "feature_length_only": DynamicControlSpec((0,)),
    "feature_frequency_only": DynamicControlSpec((1,)),
    "feature_tail_only": DynamicControlSpec((2,)),
    "feature_drop_length": DynamicControlSpec((1, 2)),
    "feature_drop_frequency": DynamicControlSpec((0, 2)),
    "feature_drop_tail": DynamicControlSpec((0, 1)),
    "regularization_no_admission_cost": DynamicControlSpec(
        PRIMARY_COLUMNS, admission_cost=0.0),
    "regularization_no_prior_penalty": DynamicControlSpec(
        PRIMARY_COLUMNS, prior_penalty=0.0),
}

CONTROL_ORDER = (
    "oof_global",
    "bucket_head_tail",
    "bucket_short_long_head_tail",
    "dynamic_delta_005",
    "dynamic_delta_010",
    "dynamic_beta_permuted",
    "dynamic_delta_020",
    "feature_length_only",
    "feature_frequency_only",
    "feature_tail_only",
    "feature_drop_length",
    "feature_drop_frequency",
    "feature_drop_tail",
    "regularization_no_admission_cost",
    "regularization_no_prior_penalty",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source_path(value: str | Path) -> Path:
    """Resolve paths emitted by a run launched either here or from repo root."""
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    local = HERE / path
    if local.exists():
        return local
    return path


def _mapping_array_fingerprint(
        payload: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        value = np.ascontiguousarray(payload[name])
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_npz(path: Path, required: Sequence[str]) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as saved:
        missing = [name for name in required if name not in saved.files]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        # Load only requested arrays.  The primary rank artifact contains many
        # large method matrices, and materializing unrelated arrays can add
        # hundreds of megabytes without strengthening an identity check.
        return {name: saved[name] for name in required}


def _string_keys(values: np.ndarray) -> list[str]:
    return [str(value) for value in values]


def primary_context_features(
        queries: Mapping[str, Mapping[str, Sequence[int]]],
        keys: Sequence[str],
        item_freq: Mapping[int, int],
        head_items: set[int],
) -> np.ndarray:
    """Reconstruct the three target-free primary features without an index."""
    output = np.zeros((len(keys), len(PRIMARY_COLUMNS)), dtype=np.float32)
    for row, uid0 in enumerate(keys):
        uid = str(uid0)
        context = [
            int(item) for item in queries[uid].get("context", ())
            if int(item) > 0
        ]
        last = context[-1] if context else 0
        output[row] = np.asarray([
            math.log1p(len(context)),
            math.log1p(item_freq.get(last, 0)),
            float(bool(last) and last not in head_items),
        ], dtype=np.float32)
    return output


def bucket_assignments(
        primary_features: np.ndarray,
        short_context: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return head/tail and short/long x head/tail labels."""
    values = np.asarray(primary_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("bucket features must have exactly three columns")
    is_short = values[:, 0] <= np.float32(
        math.log1p(short_context) + 1e-7)
    is_tail = values[:, 2] >= 0.5
    head_tail = np.where(is_tail, "tail", "head")
    crossed = np.asarray([
        f"{'short' if short else 'long'}_"
        f"{'tail' if tail else 'head'}"
        for short, tail in zip(is_short, is_tail)
    ])
    return head_tail, crossed


def _fit_scalar_bucket(
        memory: np.ndarray,
        neural: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        fallback_beta: float,
        seed: int,
        epochs: int,
) -> tuple[TrainOnlyGlobalBeta | None, dict]:
    n = int(mask.sum())
    if n == 0:
        return None, {
            "n": 0,
            "n_actionable_queries": 0,
            "beta": float(fallback_beta),
            "fallback_to_oof_global": True,
            "fallback_reason": "empty bucket",
        }
    _, _, _, actionable = rank_evidence_training_arrays(
        memory[mask], neural[mask], targets[mask])
    if not actionable.any():
        return None, {
            "n": n,
            "n_actionable_queries": 0,
            "beta": float(fallback_beta),
            "fallback_to_oof_global": True,
            "fallback_reason": "no actionable OOF target",
        }
    model = TrainOnlyGlobalBeta(
        seed=seed,
        epochs=epochs,
        initial_beta=float(fallback_beta),
    )
    report = model.fit(memory[mask], neural[mask], targets[mask])
    return model, {
        **report,
        "n": n,
        "fallback_to_oof_global": False,
    }


def fit_bucket_policy(
        labels: np.ndarray,
        declared_labels: Sequence[str],
        memory: np.ndarray,
        neural: np.ndarray,
        targets: np.ndarray,
        fallback_beta: float,
        seed: int,
        epochs: int = 80,
) -> tuple[dict[str, TrainOnlyGlobalBeta | None], dict]:
    """Fit independent OOF scalars, with the OOF global beta as fallback."""
    labels = np.asarray(labels, dtype=str)
    if len(labels) != len(targets):
        raise ValueError("bucket labels and OOF targets differ in length")
    models: dict[str, TrainOnlyGlobalBeta | None] = {}
    report = {}
    for label in declared_labels:
        model, block = _fit_scalar_bucket(
            memory,
            neural,
            targets,
            labels == label,
            fallback_beta,
            seed,
            epochs,
        )
        models[label] = model
        report[label] = block
    unexpected = sorted(set(labels) - set(declared_labels))
    if unexpected:
        raise ValueError(f"undeclared OOF bucket labels: {unexpected}")
    return models, report


def predict_bucket_policy(
        models: Mapping[str, TrainOnlyGlobalBeta | None],
        labels: np.ndarray,
        fallback_beta: float,
) -> np.ndarray:
    output = np.full(len(labels), fallback_beta, dtype=np.float32)
    for label, model in models.items():
        if model is not None:
            output[np.asarray(labels) == label] = float(model.beta_)
    unknown = sorted(set(np.asarray(labels, dtype=str)) - set(models))
    if unknown:
        raise ValueError(f"unknown prediction bucket labels: {unknown}")
    return output


def fit_dynamic_control(
        spec: DynamicControlSpec,
        features: np.ndarray,
        memory: np.ndarray,
        neural: np.ndarray,
        targets: np.ndarray,
        initial_beta: float,
        seed: int,
        epochs: int = 50,
) -> tuple[TrainOnlyDynamicBeta, dict]:
    model = TrainOnlyDynamicBeta(
        seed=seed,
        hidden=0,
        epochs=epochs,
        initial_beta=float(initial_beta),
        max_residual=spec.max_residual,
        admission_cost=spec.admission_cost,
        prior_penalty=spec.prior_penalty,
    )
    report = model.fit(
        features[:, spec.columns], memory, neural, targets)
    report.update({
        "feature_names": [FEATURE_NAMES[column] for column in spec.columns],
        "columns": list(spec.columns),
        "admission_cost": spec.admission_cost,
        "prior_penalty": spec.prior_penalty,
    })
    state = model.state_dict_numpy()
    weights = np.asarray(
        state["model::network.weight"], dtype=np.float64).reshape(-1)
    bias = float(np.asarray(
        state["model::network.bias"], dtype=np.float64).reshape(-1)[0])
    report["standardized_coefficients"] = {
        FEATURE_NAMES[column]: float(weights[offset])
        for offset, column in enumerate(spec.columns)
    }
    report["bias"] = bias
    return model, report


def deterministic_beta_permutation(
        beta: np.ndarray,
        keys: Sequence[str],
        seed: int,
) -> np.ndarray:
    """Break query assignment without changing the beta marginal distribution.

    The permutation seed is derived only from the experiment seed and ordered
    query identifiers. It is therefore fixed before test targets are read.
    """
    values = np.asarray(beta, dtype=np.float32)
    if len(values) != len(keys):
        raise ValueError("beta and query keys differ in length")
    if len(values) <= 1:
        return values.copy()
    digest = hashlib.sha256()
    digest.update(f"dynamic-beta-permutation-v1::{seed}\n".encode())
    for key in keys:
        digest.update(str(key).encode())
        digest.update(b"\n")
    generator_seed = int.from_bytes(digest.digest()[:8], "little")
    permutation = np.random.default_rng(generator_seed).permutation(
        len(values))
    if np.array_equal(permutation, np.arange(len(values))):
        permutation = np.roll(permutation, 1)
    output = values[permutation]
    if not np.array_equal(np.sort(output), np.sort(values)):
        raise RuntimeError("beta permutation changed the marginal values")
    return output


def _evaluate_ranking(
        ranking: np.ndarray,
        targets: np.ndarray,
        memory_ranks: np.ndarray,
) -> tuple[dict, np.ndarray]:
    ranks = ranks_at_20(ranking, targets)
    metrics = metrics_from_ranks(ranks)
    metrics["utility"] = float(
        .5 * metrics["recall@6"] + .5 * metrics["recall@20"])
    memory_hit = memory_ranks > 0
    method_hit = ranks > 0
    metrics.update({
        "rescues_vs_memory": int(np.sum(~memory_hit & method_hit)),
        "damage_vs_memory": int(np.sum(memory_hit & ~method_hit)),
        "net_rescues_vs_memory": int(
            np.sum(~memory_hit & method_hit)
            - np.sum(memory_hit & ~method_hit)
        ),
    })
    return metrics, ranks


def _beta_summary(beta: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(beta)),
        "std": float(np.std(beta)),
        "min": float(np.min(beta)),
        "max": float(np.max(beta)),
    }


def assert_saved_rank_identity(
        memory: dict[str, np.ndarray],
        neural: dict[str, np.ndarray],
        primary_rank_artifact: dict[str, np.ndarray],
) -> list[str]:
    """Verify exact keys and expert top-20 rankings across frozen artifacts."""
    memory_keys = _string_keys(memory["keys"])
    neural_keys = _string_keys(neural["keys"])
    primary_keys = _string_keys(primary_rank_artifact["test_keys"])
    if memory_keys != neural_keys or memory_keys != primary_keys:
        raise ValueError("saved test query order mismatch")
    if not np.array_equal(
            memory["selected"][:, :20],
            primary_rank_artifact["test_memory_only_top20"]):
        raise ValueError("saved test memory ranks changed")
    if not np.array_equal(
            neural["rankings"][:, :20],
            primary_rank_artifact["test_neural_only_top20"]):
        raise ValueError("saved test neural ranks changed")
    return memory_keys


def evaluate_after_manifest(
        manifest_path: Path,
        queries: Mapping[str, Mapping[str, Sequence[int]]],
        keys: Sequence[str],
        memory: np.ndarray,
        neural: np.ndarray,
        betas: Mapping[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    """Evaluate controls only after a frozen manifest is present on disk."""
    if not manifest_path.exists():
        raise RuntimeError(
            "refusing test-target evaluation before frozen manifest")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("frozen_before_test_target_evaluation", False):
        raise RuntimeError("manifest does not attest pre-test freezing")

    actual_fingerprint = query_fingerprint(dict(queries))
    expected_fingerprint = manifest["inputs"]["test_query_fingerprint"]
    if actual_fingerprint != expected_fingerprint:
        raise ValueError("test query identity changed after policy freezing")
    targets = targets_for(list(keys), dict(queries))
    memory_ranks = ranks_at_20(memory, targets)
    metrics = {}
    rank_payload = {}
    for name, beta in betas.items():
        if len(beta) != len(keys):
            raise ValueError(f"{name}: beta/test row count mismatch")
        ranking = fuse_with_dynamic_beta(memory, neural, beta)
        block, ranks = _evaluate_ranking(
            ranking, targets, memory_ranks)
        block["beta"] = _beta_summary(beta)
        metrics[name] = block
        rank_payload[f"test_{name}_rank"] = ranks.astype(np.uint8)
        rank_payload[f"test_{name}_beta"] = np.asarray(
            beta, dtype=np.float32)
    return metrics, rank_payload


def _declared_validation_subset(data: dict, size: int) -> None:
    if len(data["valid_queries"]) <= size:
        return
    keys = sorted(
        data["valid_queries"], key=cearf._stable_fraction)[:size]
    data["valid_queries"] = {
        uid: data["valid_queries"][uid] for uid in keys
    }


def _split_parameter(
        split: Mapping[str, object],
        current: str,
        legacy: str,
        default: int,
) -> int:
    return int(split.get(current, split.get(legacy, default)))


def reconstruct_oof(
        domain: str,
        domain_block: dict,
        source_manifest: dict,
        artifact_dir: Path,
) -> dict:
    """Reconstruct and verify exact OOF queries, ranks and features."""
    data = loaders.ALL_LOADERS[domain]()
    split_expected = source_manifest["split"]
    valid_size = _split_parameter(
        split_expected,
        "declared_validation_queries",
        "excluded_official_validation_sources",
        5_000,
    )
    _declared_validation_subset(data, valid_size)
    sessions = data["train_sessions"]
    tune_sessions = hold_out_validation_targets(
        sessions, data["valid_queries"])
    validation_sources = canonical_validation_sources(
        data["valid_queries"])
    inner_fit, profile_queries, gate_queries, split_actual = (
        make_training_oof_split(
            tune_sessions,
            validation_sources,
            float(split_expected["fraction"]),
            int(split_expected["cap"]),
            int(split_expected["profile_source_sessions"]),
        )
    )
    for key in ("profile_query_fingerprint", "gate_query_fingerprint"):
        if split_actual[key] != split_expected[key]:
            raise ValueError(f"{domain}: reconstructed {key} differs")
    if split_actual["declared_validation_source_overlap"] != 0:
        raise ValueError(f"{domain}: OOF/validation source overlap")

    memory_path = artifact_dir / domain.lower() / "gate_oof_memory.npz"
    memory = _load_npz(
        memory_path,
        ("keys", "fingerprint", "profiles", "selected",
         "transition", "session", "popularity"),
    )
    keys = _string_keys(memory["keys"])
    # ``load_or_build_memory`` canonicalizes cache rows by sorted query ID,
    # whereas the OOF split dictionary preserves stable-hash selection order.
    # Compare against the cache contract, not dictionary insertion order.
    if keys != sorted(str(uid) for uid in gate_queries):
        raise ValueError(f"{domain}: reconstructed OOF order differs")
    expected_gate_fingerprint = query_fingerprint(gate_queries)
    if str(memory["fingerprint"].item()) != expected_gate_fingerprint:
        raise ValueError(f"{domain}: OOF memory query fingerprint differs")
    profiles_json = json.dumps(domain_block["profiles"], sort_keys=True)
    if str(memory["profiles"].item()) != profiles_json:
        raise ValueError(f"{domain}: OOF memory profile identity differs")

    config = cearf.CEARFConfig(
        exclude_seen=domain not in REPEAT_PROTOCOL_DOMAINS)
    inner_index = cearf.CEARFIndex(
        inner_fit, data["n_items"], config)
    item_freq = Counter(
        item for sequence in inner_fit.values() for item in sequence)
    head_items = set(popularity_partition(
        item_freq, data["n_items"])[0].tolist())
    return {
        "data": data,
        "sessions": sessions,
        "inner_fit": inner_fit,
        "gate_queries": gate_queries,
        "gate_keys": keys,
        "gate_targets": targets_for(keys, gate_queries),
        "gate_memory": memory,
        "item_freq": item_freq,
        "head_items": head_items,
        "inner_index": inner_index,
        "short_context": config.short_context,
        "split": split_actual,
        "memory_path": memory_path,
    }


def _source_run_for_seed(domain_block: dict, seed: int) -> dict:
    matches = [
        run for run in domain_block.get("runs", [])
        if int(run["seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{domain_block.get('domain')} seed={seed}: "
            f"expected one completed source run, found {len(matches)}")
    return matches[0]


def validate_source_results(
        raw: dict,
        domains: Sequence[str],
        seeds: Sequence[int],
) -> None:
    for domain in domains:
        if domain not in raw:
            raise ValueError(f"source results missing {domain}")
        block = raw[domain]
        if str(block.get("protocol")) not in LEGACY_PROTOCOL_ALIASES:
            raise ValueError(
                f"{domain}: unsupported source protocol "
                f"{block.get('protocol')}")
        for seed in seeds:
            run = _source_run_for_seed(block, seed)
            manifest_path = resolve_source_path(run["manifest"])
            rank_path = resolve_source_path(run["rank_artifact"])
            if not manifest_path.exists():
                raise FileNotFoundError(manifest_path)
            if not rank_path.exists():
                raise FileNotFoundError(rank_path)


def validate_completed_control(
        domain: str,
        seed: int,
        completed_run: dict,
        source_run: dict,
) -> None:
    manifest_path = resolve_source_path(
        completed_run["frozen_control_manifest"])
    rank_path = resolve_source_path(completed_run["rank_artifact"])
    state_path = resolve_source_path(completed_run["policy_state_artifact"])
    for path in (manifest_path, rank_path, state_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    if (
            manifest.get("protocol") != CONTROL_PROTOCOL
            or manifest.get("domain") != domain
            or int(manifest.get("seed", -1)) != seed):
        raise ValueError(
            f"{domain} seed={seed}: completed control identity mismatch")
    source_manifest_path = resolve_source_path(source_run["manifest"])
    source_rank_path = resolve_source_path(source_run["rank_artifact"])
    inputs = manifest.get("inputs", {})
    if inputs.get("source_manifest_sha256") != _sha256_file(
            source_manifest_path):
        raise ValueError(
            f"{domain} seed={seed}: source manifest changed after controls")
    if inputs.get("source_rank_artifact_sha256") != _sha256_file(
            source_rank_path):
        raise ValueError(
            f"{domain} seed={seed}: source ranks changed after controls")
    if manifest.get(
            "fitted_policy_state_artifact_sha256") != _sha256_file(
                state_path):
        raise ValueError(
            f"{domain} seed={seed}: fitted control state changed")


def _control_state_fingerprint(
        dynamic_models: Mapping[str, TrainOnlyDynamicBeta],
        bucket_reports: Mapping[str, dict],
        global_beta: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(global_beta, dtype=np.float64).tobytes())
    for name in sorted(dynamic_models):
        digest.update(name.encode())
        digest.update(_mapping_array_fingerprint(
            dynamic_models[name].state_dict_numpy()).encode())
    digest.update(json.dumps(
        bucket_reports, sort_keys=True).encode())
    return digest.hexdigest()


def _control_state_payload(
        dynamic_models: Mapping[str, TrainOnlyDynamicBeta],
        head_tail_models: Mapping[str, TrainOnlyGlobalBeta | None],
        crossed_models: Mapping[str, TrainOnlyGlobalBeta | None],
        global_beta: float,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "oof_global_beta": np.asarray(global_beta, dtype=np.float32),
    }
    for family, models in (
            ("head_tail", head_tail_models),
            ("short_long_head_tail", crossed_models)):
        for label, model in models.items():
            value = global_beta if model is None else float(model.beta_)
            payload[f"{family}::{label}::beta"] = np.asarray(
                value, dtype=np.float32)
            payload[f"{family}::{label}::used_global_fallback"] = np.asarray(
                model is None, dtype=np.bool_)
    for control, model in dynamic_models.items():
        for name, value in model.state_dict_numpy().items():
            payload[f"{control}::{name}"] = value
    return payload


def run_seed(
        domain: str,
        seed: int,
        domain_block: dict,
        source_run: dict,
        reconstructed: dict,
        artifact_dir: Path,
        output_artifact_dir: Path,
) -> dict:
    source_manifest_path = resolve_source_path(source_run["manifest"])
    source_manifest = json.loads(source_manifest_path.read_text())
    if (
            str(source_manifest.get("protocol"))
            not in LEGACY_PROTOCOL_ALIASES
            or source_manifest.get("domain") != domain
            or int(source_manifest.get("seed", -1)) != seed):
        raise ValueError(f"{domain} seed={seed}: source manifest mismatch")
    for key in ("profile_query_fingerprint", "gate_query_fingerprint"):
        if (
                source_manifest.get("split", {}).get(key)
                != reconstructed["split"].get(key)):
            raise ValueError(
                f"{domain} seed={seed}: source-manifest {key} differs")
    if source_manifest.get("memory_profiles") != domain_block["profiles"]:
        raise ValueError(
            f"{domain} seed={seed}: source-manifest profiles differ")

    domain_dir = artifact_dir / domain.lower()
    prediction_dir = domain_dir / "predictions"
    gate_neural_path = (
        prediction_dir / f"seed{seed}_gate_oof_top120.npz")
    gate_neural = _load_npz(
        gate_neural_path, ("keys", "rankings", "fingerprint"))
    gate_keys = reconstructed["gate_keys"]
    if _string_keys(gate_neural["keys"]) != gate_keys:
        raise ValueError(f"{domain} seed={seed}: OOF neural order differs")
    candidate_width = int(source_manifest["candidate_width"])
    gate_memory = reconstructed["gate_memory"]
    if (
            gate_memory["selected"].shape
            != gate_neural["rankings"].shape
            or gate_memory["selected"].shape
            != (len(gate_keys), candidate_width)):
        raise ValueError(
            f"{domain} seed={seed}: OOF full-rank shape differs")
    gate_fingerprint = query_fingerprint(
        reconstructed["gate_queries"])
    if str(gate_neural["fingerprint"].item()) != gate_fingerprint:
        raise ValueError(
            f"{domain} seed={seed}: OOF neural fingerprint differs")

    gate_features = feature_matrix(
        reconstructed["gate_queries"],
        gate_keys,
        gate_memory,
        gate_neural["rankings"],
        reconstructed["item_freq"],
        reconstructed["head_items"],
        reconstructed["inner_index"],
    )
    independent_primary = primary_context_features(
        reconstructed["gate_queries"],
        gate_keys,
        reconstructed["item_freq"],
        reconstructed["head_items"],
    )
    if not np.array_equal(
            gate_features[:, PRIMARY_COLUMNS], independent_primary):
        raise ValueError(
            f"{domain} seed={seed}: OOF primary feature reconstruction differs")
    gate_targets = reconstructed["gate_targets"]

    global_model = TrainOnlyGlobalBeta(seed=seed)
    global_report = global_model.fit(
        gate_memory["selected"],
        gate_neural["rankings"],
        gate_targets,
    )
    source_global_beta = float(
        source_manifest["global_beta_training"]["beta"])
    if not np.isclose(
            float(global_model.beta_), source_global_beta,
            rtol=0.0, atol=1e-7):
        raise ValueError(
            f"{domain} seed={seed}: OOF global beta is not reproducible")

    head_tail_oof, crossed_oof = bucket_assignments(
        independent_primary, reconstructed["short_context"])
    head_tail_models, head_tail_report = fit_bucket_policy(
        head_tail_oof,
        ("head", "tail"),
        gate_memory["selected"],
        gate_neural["rankings"],
        gate_targets,
        float(global_model.beta_),
        seed,
    )
    crossed_models, crossed_report = fit_bucket_policy(
        crossed_oof,
        ("short_head", "short_tail", "long_head", "long_tail"),
        gate_memory["selected"],
        gate_neural["rankings"],
        gate_targets,
        float(global_model.beta_),
        seed,
    )

    dynamic_models = {}
    dynamic_reports = {}
    for name, spec in DYNAMIC_CONTROL_SPECS.items():
        model, report = fit_dynamic_control(
            spec,
            gate_features,
            gate_memory["selected"],
            gate_neural["rankings"],
            gate_targets,
            float(global_model.beta_),
            seed,
        )
        dynamic_models[name] = model
        dynamic_reports[name] = report

    test_memory_path = domain_dir / "test_memory.npz"
    test_neural_path = (
        prediction_dir / f"seed{seed}_test_top120.npz")
    primary_rank_path = resolve_source_path(source_run["rank_artifact"])
    test_memory = _load_npz(
        test_memory_path,
        ("keys", "fingerprint", "profiles", "selected"))
    test_neural = _load_npz(
        test_neural_path, ("keys", "rankings", "fingerprint"))
    primary_rank = _load_npz(
        primary_rank_path,
        ("test_keys", "test_memory_only_top20",
         "test_neural_only_top20", "test_features", "test_dynamic_beta"),
    )
    test_keys = assert_saved_rank_identity(
        test_memory, test_neural, primary_rank)
    if (
            test_memory["selected"].shape
            != test_neural["rankings"].shape
            or test_memory["selected"].shape
            != (len(test_keys), candidate_width)):
        raise ValueError(
            f"{domain} seed={seed}: test full-rank shape differs")
    expected_test_fingerprint = str(test_memory["fingerprint"].item())
    if str(test_neural["fingerprint"].item()) != expected_test_fingerprint:
        raise ValueError(
            f"{domain} seed={seed}: test cache fingerprints differ")
    if str(test_memory["profiles"].item()) != json.dumps(
            domain_block["profiles"], sort_keys=True):
        raise ValueError(
            f"{domain} seed={seed}: test memory profile identity differs")

    final_freq = Counter(
        item for sequence in reconstructed["sessions"].values()
        for item in sequence)
    final_head = set(popularity_partition(
        final_freq, reconstructed["data"]["n_items"])[0].tolist())
    test_primary = primary_context_features(
        reconstructed["data"]["test_queries"],
        test_keys,
        final_freq,
        final_head,
    )
    saved_test_features = np.asarray(
        primary_rank["test_features"], dtype=np.float32)
    if not np.array_equal(
            saved_test_features[:, PRIMARY_COLUMNS], test_primary):
        raise ValueError(
            f"{domain} seed={seed}: saved test features changed")

    head_tail_test, crossed_test = bucket_assignments(
        test_primary, reconstructed["short_context"])
    betas = {
        "oof_global": global_model.predict(len(test_keys)),
        "bucket_head_tail": predict_bucket_policy(
            head_tail_models,
            head_tail_test,
            float(global_model.beta_),
        ),
        "bucket_short_long_head_tail": predict_bucket_policy(
            crossed_models,
            crossed_test,
            float(global_model.beta_),
        ),
    }
    for name, spec in DYNAMIC_CONTROL_SPECS.items():
        betas[name] = dynamic_models[name].predict(
            saved_test_features[:, spec.columns])

    # The Delta=.10 control is an exact replay of the declared primary gate.
    if not np.allclose(
            betas["dynamic_delta_010"],
            primary_rank["test_dynamic_beta"],
            rtol=0.0,
            atol=1e-6):
        raise ValueError(
            f"{domain} seed={seed}: primary dynamic gate is not reproducible")
    betas["dynamic_beta_permuted"] = deterministic_beta_permutation(
        betas["dynamic_delta_010"], test_keys, seed)

    output_domain_dir = output_artifact_dir / domain.lower()
    output_domain_dir.mkdir(parents=True, exist_ok=True)
    control_state_path = (
        output_domain_dir / f"seed{seed}_allocation_controls_state.npz")
    np.savez_compressed(
        control_state_path,
        **_control_state_payload(
            dynamic_models,
            head_tail_models,
            crossed_models,
            float(global_model.beta_),
        ),
    )
    frozen_manifest_path = (
        output_domain_dir / f"seed{seed}_allocation_controls_manifest.json")
    training_reports = {
        "oof_global": global_report,
        "bucket_head_tail": head_tail_report,
        "bucket_short_long_head_tail": crossed_report,
        **dynamic_reports,
    }
    frozen_manifest = {
        "protocol": CONTROL_PROTOCOL,
        "source_protocol": str(domain_block["protocol"]),
        "domain": domain,
        "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_test_target_evaluation": True,
        "expert_training_or_prediction_performed": False,
        "allocation_fit_source": "OOF leave-last-out training queries only",
        "validation_labels_used_to_fit": False,
        "test_labels_used_to_fit": False,
        "test_labels_used_to_select": False,
        "global_fallback": "single OOF-trained global beta",
        "controls": {
            name: asdict(spec)
            for name, spec in DYNAMIC_CONTROL_SPECS.items()
        },
        "assignment_control": {
            "name": "dynamic_beta_permuted",
            "algorithm": (
                "SHA-256-derived deterministic permutation of frozen "
                "primary beta values"
            ),
            "preserves_beta_multiset_exactly": True,
            "uses_validation_or_test_targets": False,
            "permuted_beta_fingerprint": array_fingerprint(
                betas["dynamic_beta_permuted"]),
        },
        "training": training_reports,
        "inputs": {
            "source_results_protocol": str(domain_block["protocol"]),
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": _sha256_file(source_manifest_path),
            "source_rank_artifact": str(primary_rank_path),
            "source_rank_artifact_sha256": _sha256_file(primary_rank_path),
            "gate_query_fingerprint": gate_fingerprint,
            "gate_memory_path": str(reconstructed["memory_path"]),
            "gate_memory_selected_fingerprint": array_fingerprint(
                gate_memory["selected"]),
            "gate_neural_path": str(gate_neural_path),
            "gate_neural_fingerprint": array_fingerprint(
                gate_neural["rankings"]),
            "gate_features_fingerprint": array_fingerprint(gate_features),
            "test_query_fingerprint": expected_test_fingerprint,
            "test_memory_path": str(test_memory_path),
            "test_memory_selected_fingerprint": array_fingerprint(
                test_memory["selected"]),
            "test_neural_path": str(test_neural_path),
            "test_neural_fingerprint": array_fingerprint(
                test_neural["rankings"]),
            "test_features_fingerprint": array_fingerprint(
                saved_test_features),
            "test_key_count": len(test_keys),
        },
        "fitted_policy_state_fingerprint": _control_state_fingerprint(
            dynamic_models,
            {
                "head_tail": head_tail_report,
                "short_long_head_tail": crossed_report,
            },
            float(global_model.beta_),
        ),
        "fitted_policy_state_artifact": str(control_state_path),
        "fitted_policy_state_artifact_sha256": _sha256_file(
            control_state_path),
    }
    frozen_manifest_path.write_text(
        json.dumps(frozen_manifest, indent=2))

    metrics, rank_payload = evaluate_after_manifest(
        frozen_manifest_path,
        reconstructed["data"]["test_queries"],
        test_keys,
        test_memory["selected"],
        test_neural["rankings"],
        betas,
    )
    control_rank_path = (
        output_domain_dir / f"seed{seed}_allocation_controls_ranks.npz")
    np.savez_compressed(
        control_rank_path,
        test_keys=np.asarray(test_keys, dtype=str),
        source_test_query_fingerprint=np.asarray(
            expected_test_fingerprint),
        **rank_payload,
    )
    return {
        "seed": seed,
        "source_run_manifest": str(source_manifest_path),
        "frozen_control_manifest": str(frozen_manifest_path),
        "policy_state_artifact": str(control_state_path),
        "rank_artifact": str(control_rank_path),
        "training": training_reports,
        "test": {"metrics": metrics},
        "identity_checks": {
            "oof_query_order_exact": True,
            "oof_query_fingerprint_exact": True,
            "oof_primary_features_exact": True,
            "oof_global_beta_reproduced": True,
            "test_query_order_exact": True,
            "test_memory_top20_exact": True,
            "test_neural_top20_exact": True,
            "test_primary_features_exact": True,
            "primary_dynamic_delta_010_reproduced": True,
            "manifest_precedes_test_target_evaluation": True,
        },
    }


def _mean_std(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": [float(value) for value in array],
    }


def paired_assignment_summary(
        runs: Sequence[dict],
        metric: str,
        repetitions: int,
        seed: int,
) -> dict:
    """Compare learned beta assignment with its target-free permutation."""
    differences = []
    reference_keys = None
    for run in runs:
        rank_path = resolve_source_path(run["rank_artifact"])
        with np.load(rank_path, allow_pickle=False) as saved:
            keys = np.asarray(saved["test_keys"], dtype=str)
            if reference_keys is None:
                reference_keys = keys
            elif not np.array_equal(reference_keys, keys):
                raise ValueError(
                    "allocation-control test query order differs across seeds"
                )
            primary = _per_query_value(
                saved["test_dynamic_delta_010_rank"], metric)
            permuted = _per_query_value(
                saved["test_dynamic_beta_permuted_rank"], metric)
            differences.append(primary - permuted)
    per_query_seed_mean = np.mean(np.stack(differences), axis=0)
    output = _discrete_bootstrap(
        per_query_seed_mean, repetitions, seed)
    output.update({
        "metric": metric,
        "challenger": "dynamic_delta_010",
        "baseline": "dynamic_beta_permuted",
        "seeds": [int(run["seed"]) for run in runs],
        "aggregation": (
            "primary-minus-permuted per-query difference averaged across "
            "matched seeds, then paired query-level bootstrap; intervals "
            "condition on the observed fitted seeds"
        ),
        "interpretation": (
            "query-to-beta assignment signal with the beta multiset fixed"
        ),
    })
    return output


def summarize_results(
        results: dict,
        bootstrap_repetitions: int = 20_000,
) -> dict:
    summary = {
        "protocol": CONTROL_PROTOCOL,
        "bootstrap_unit": (
            "smallest recoverable test-query identifier; matched-seed "
            "outcomes carried jointly"
        ),
        "seed_aggregation": (
            "per-query differences averaged over observed fitted seeds "
            "before bootstrap; intervals condition on those seeds"
        ),
        "domains": {},
    }
    for domain in DOMAINS:
        if domain not in results:
            continue
        runs = sorted(
            results[domain]["runs"], key=lambda run: int(run["seed"]))
        if not runs:
            raise ValueError(f"{domain}: no completed control runs")
        for run in runs:
            missing = [
                name for name in CONTROL_ORDER
                if name not in run["test"]["metrics"]
            ]
            if missing:
                raise ValueError(
                    f"{domain} seed={run['seed']}: "
                    f"missing controls {missing}")
        methods = list(CONTROL_ORDER)
        domain_summary = {
            "seeds": [int(run["seed"]) for run in runs],
            "methods": {
                method: {
                    metric: _mean_std([
                        run["test"]["metrics"][method][metric]
                        for run in runs
                    ])
                    for metric in METRICS
                }
                for method in methods
            },
            "primary_gate_parameters": {
                "standardized_coefficients": {
                    feature: _mean_std([
                        run["training"]["dynamic_delta_010"][
                            "standardized_coefficients"][feature]
                        for run in runs
                    ])
                    for feature in (
                        "log_context_length",
                        "log_last_item_frequency",
                        "last_item_is_tail",
                    )
                },
                "bias": _mean_std([
                    run["training"]["dynamic_delta_010"]["bias"]
                    for run in runs
                ]),
            },
        }
        if bootstrap_repetitions > 0:
            domain_summary["assignment_paired"] = {
                metric: paired_assignment_summary(
                    runs,
                    metric,
                    bootstrap_repetitions,
                    20260901 + index,
                )
                for index, metric in enumerate(METRICS)
            }
        summary["domains"][domain] = domain_summary
    return summary


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def _tex_value(block: dict) -> str:
    return f"{block['mean']:.5f} ({block['std']:.5f})"


def _tex_mean(block: dict) -> str:
    return f"{block['mean']:.5f}"


def _tex_delta_interval(block: dict) -> str:
    low, high = block["cluster_bootstrap_ci95"]
    return (
        f"{block['difference']:+.5f} "
        f"[{low:+.5f},{high:+.5f}]"
    ).replace("+0.", "+.").replace("-0.", "-.")


def _tex_signed_five(value: float) -> str:
    if round(float(value), 5) == 0:
        return "0.00000"
    return f"{float(value):+.5f}"


def render_summary_tex(summary: dict) -> str:
    domain_labels = {
        "Video_Games": "Video Games",
        "Baby_Products": "Baby Products",
        "Diginetica_HID": "Diginetica",
    }
    labels = {
        "oof_global": "OOF global",
        "bucket_head_tail": "Head/tail scalars",
        "bucket_short_long_head_tail": r"Short/long $\times$ head/tail",
        "dynamic_delta_005": r"Dynamic $\Delta=.05$",
        "dynamic_delta_010": r"Dynamic $\Delta=.10$ (primary)",
        "dynamic_beta_permuted": r"Primary $\beta_q$ reassigned",
        "dynamic_delta_020": r"Dynamic $\Delta=.20$",
        "feature_length_only": "Length only",
        "feature_frequency_only": "Frequency only",
        "feature_tail_only": "Tail only",
        "feature_drop_length": "Drop length",
        "feature_drop_frequency": "Drop frequency",
        "feature_drop_tail": "Drop tail",
        "regularization_no_admission_cost": "No admission cost",
        "regularization_no_prior_penalty": "No prior penalty",
    }
    lines = [
        "% Auto-generated by run_dynamic_beta_allocation_controls.py.",
        "% Values are means over completed seeds unless stated otherwise.",
    ]
    lines.extend([
        r"\begingroup\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{llrrrr}",
        r"\caption{Training-only allocation controls across all domains. "
        r"Learned policies use frozen OOF training ranks; reassignment is a "
        r"fixed target-free control, and no validation or test labels enter "
        r"fitting or selection.}\label{tab:allocation-controls-all}\\",
        r"\toprule",
        r"Domain & Policy & R@6 & R@20 & nDCG@20 & $U$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{c}{\tablename\ \thetable{} -- continued}\\",
        r"\toprule",
        r"Domain & Policy & R@6 & R@20 & nDCG@20 & $U$ \\",
        r"\midrule",
        r"\endhead",
    ])
    for domain_index, domain in enumerate(DOMAINS):
        if domain not in summary["domains"]:
            continue
        methods = summary["domains"][domain]["methods"]
        for method in CONTROL_ORDER:
            if method not in methods:
                continue
            block = methods[method]
            lines.append(
                f"{domain_labels.get(domain, _tex_escape(domain))} & "
                f"{labels.get(method, _tex_escape(method))} & "
                f"{_tex_mean(block['recall@6'])} & "
                f"{_tex_mean(block['recall@20'])} & "
                f"{_tex_mean(block['ndcg@20'])} & "
                f"{_tex_mean(block['utility'])} \\\\"
            )
        if domain_index < len(DOMAINS) - 1:
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule",
        r"\end{longtable}",
        r"\noindent R@10, nDCG@6, and nDCG@10 are omitted because no "
        r"allocation-control claim depends on them.",
        r"\endgroup",
        "",
    ])
    if all(
        "assignment_paired" in summary["domains"].get(domain, {})
        for domain in DOMAINS
        if domain in summary["domains"]
    ):
        metric_names = (
            ("recall@20", r"R@20"),
            ("ndcg@6", r"nDCG@6"),
            ("ndcg@10", r"nDCG@10"),
            ("ndcg@20", r"nDCG@20"),
            ("utility", r"$U$"),
        )
        lines.extend([
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Learned assignment versus fixed reassignment across "
            r"all domains. Entries are primary minus reassigned after "
            r"matched-seed averaging; paired query-level 95\% bootstrap "
            r"intervals are unadjusted across metrics.}",
            r"\label{tab:beta-assignment-control}",
            r"\begin{tabular}{llrr}",
            r"\toprule",
            r"Domain & Metric & Difference & 95\% CI\\",
            r"\midrule",
        ])
        for domain_index, domain in enumerate(DOMAINS):
            if domain not in summary["domains"]:
                continue
            paired = summary["domains"][domain]["assignment_paired"]
            for metric, metric_label in metric_names:
                block = paired[metric]
                low, high = block["cluster_bootstrap_ci95"]
                lines.append(
                    f"{domain_labels.get(domain, _tex_escape(domain))} & "
                    f"{metric_label} & {_tex_signed_five(block['difference'])} & "
                    f"[{_tex_signed_five(low)}, {_tex_signed_five(high)}] \\\\"
                )
            if domain_index < len(DOMAINS) - 1:
                lines.append(r"\midrule")
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ])
    coefficient_labels = (
        ("log_context_length", r"Length"),
        ("log_last_item_frequency", r"Frequency"),
        ("last_item_is_tail", "Tail"),
    )
    lines.extend([
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Primary linear-gate parameters on standardized OOF "
        r"features, mean (sample SD) across seeds.}",
        r"\label{tab:primary-gate-coefficients}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        "Domain & "
        + " & ".join(label for _, label in coefficient_labels)
        + r" & Bias \\",
        r"\hline",
    ])
    for domain in DOMAINS:
        if domain not in summary["domains"]:
            continue
        parameters = summary["domains"][domain][
            "primary_gate_parameters"]
        coefficients = parameters["standardized_coefficients"]
        lines.append(
            f"{domain_labels.get(domain, _tex_escape(domain))} & "
            + " & ".join(
                _tex_value(coefficients[name])
                for name, _ in coefficient_labels
            )
            + f" & {_tex_value(parameters['bias'])} \\\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def write_summary_outputs(
        results: dict,
        summary_path: Path,
        tex_path: Path,
        bootstrap_repetitions: int = 20_000,
) -> dict:
    summary = summarize_results(results, bootstrap_repetitions)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    tex_path.write_text(render_summary_tex(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument(
        "--source-results", type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_results.json")
    parser.add_argument(
        "--source-artifact-dir", type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_artifacts")
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "dynamic_beta_allocation_controls_results.json")
    parser.add_argument(
        "--output-artifact-dir", type=Path,
        default=HERE / "dynamic_beta_allocation_controls_artifacts")
    parser.add_argument(
        "--summary", type=Path,
        default=HERE / "dynamic_beta_allocation_controls_summary.json")
    parser.add_argument(
        "--tex", type=Path,
        default=HERE / "paper"
        / "generated_dynamic_beta_allocation_controls.tex")
    parser.add_argument(
        "--bootstrap-repetitions", type=int, default=20_000)
    args = parser.parse_args()

    raw = json.loads(args.source_results.read_text())
    validate_source_results(raw, args.domains, args.seeds)
    results = (
        json.loads(args.output.read_text())
        if args.output.exists() else {}
    )

    for domain in args.domains:
        domain_block = raw[domain]
        first_run = _source_run_for_seed(domain_block, args.seeds[0])
        first_manifest = json.loads(resolve_source_path(
            first_run["manifest"]).read_text())
        reconstructed = reconstruct_oof(
            domain,
            domain_block,
            first_manifest,
            args.source_artifact_dir,
        )
        output_block = results.setdefault(domain, {
            "domain": domain,
            "protocol": CONTROL_PROTOCOL,
            "source_protocol": str(domain_block["protocol"]),
            "runs": [],
        })
        if output_block.get("protocol") != CONTROL_PROTOCOL:
            raise ValueError(f"{domain}: output protocol mismatch")
        completed = {
            int(run["seed"]): run for run in output_block["runs"]
        }
        for seed in args.seeds:
            if seed in completed:
                validate_completed_control(
                    domain,
                    seed,
                    completed[seed],
                    _source_run_for_seed(domain_block, seed),
                )
                print(
                    f"[ALLOC] {domain} seed={seed} already complete",
                    flush=True)
                continue
            print(
                f"[ALLOC] {domain} seed={seed}: fitting OOF controls",
                flush=True)
            source_run = _source_run_for_seed(domain_block, seed)
            result = run_seed(
                domain,
                seed,
                domain_block,
                source_run,
                reconstructed,
                args.source_artifact_dir,
                args.output_artifact_dir,
            )
            output_block["runs"].append(result)
            args.output.write_text(json.dumps(results, indent=2))
            primary = result["test"]["metrics"]["dynamic_delta_010"]
            print(
                f"[ALLOC] DONE {domain} seed={seed}: "
                f"R20={primary['recall@20']:.5f} "
                f"U={primary['utility']:.5f}",
                flush=True,
            )

    write_summary_outputs(
        results,
        args.summary,
        args.tex,
        args.bootstrap_repetitions,
    )
    print(f"[ALLOC] saved {args.output}", flush=True)
    print(f"[ALLOC] saved {args.tex}", flush=True)


if __name__ == "__main__":
    main()
