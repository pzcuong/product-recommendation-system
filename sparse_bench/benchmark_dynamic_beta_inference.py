#!/usr/bin/env python3
"""Benchmark the frozen training-only dynamic-beta inference path.

This benchmark is deliberately separate from ``benchmark_cearfn_inference``:
that script measured the retired validation-selected router.  Here, beta is
produced by the frozen bounded linear gate

    beta_q = beta_OOF + Delta * tanh(w^T z_q + b),

where ``z_q`` contains only target-free context features.  The default run is
CPU-only and reuses cached top-120 memory/PASGR ranks.  It therefore measures
the *new* per-query post-processing path exactly:

1. three context-feature lookups;
2. standardization and bounded linear beta assignment;
3. weighted reciprocal-rank fusion to top-20.

For context only, the output can also compose these new measurements with the
memory-retrieval and PASGR-prediction times from the earlier warm-expert
benchmark.  That value is explicitly labelled a cross-run estimate, never a
newly co-timed end-to-end measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Callable, Mapping, Sequence

import numpy as np

import loaders
from dynamic_beta import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    fuse_with_dynamic_beta,
)
from run_cearfn_evidence import popularity_partition


HERE = Path(__file__).resolve().parent
DEFAULT_DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")
DOMAIN_DIRS = {
    "Video_Games": "video_games",
    "Baby_Products": "baby_products",
    "Diginetica_HID": "diginetica_hid",
}
DOMAIN_MACRO_PREFIX = {
    "Video_Games": "Video",
    "Baby_Products": "Baby",
    "Diginetica_HID": "Digi",
}
PRIMARY_FEATURE_NAMES = tuple(
    FEATURE_NAMES[column] for column in FEATURE_GROUPS["context"]
)


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def hardware_description() -> str:
    cpu = _sysctl("machdep.cpu.brand_string")
    memory = _sysctl("hw.memsize")
    parts = [cpu or platform.processor() or platform.machine()]
    if memory and memory.isdigit():
        parts.append(f"{int(memory) / 2**30:.0f} GiB RAM")
    parts.append(f"{platform.system()} {platform.release()}")
    return ", ".join(parts)


def context_feature_matrix(
    queries: Mapping[str, Mapping[str, Sequence[int]]],
    keys: Sequence[str],
    item_frequency: Mapping[int, int],
    head_items: set[int],
) -> np.ndarray:
    """Compute the three target-free features used by the primary gate."""
    features = np.empty((len(keys), 3), dtype=np.float32)
    for row, uid0 in enumerate(keys):
        uid = str(uid0)
        context = queries[uid].get("context", ())
        length = len(context)
        last_item = int(context[-1]) if length else 0
        features[row, 0] = math.log1p(length)
        features[row, 1] = math.log1p(
            item_frequency.get(last_item, 0))
        features[row, 2] = float(
            bool(last_item) and last_item not in head_items)
    return features


def bounded_linear_beta(
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    global_beta: float,
    max_residual: float,
) -> np.ndarray:
    """NumPy inference equivalent of the frozen ``_BetaNetwork``."""
    standardized = (
        np.asarray(features, dtype=np.float32) - feature_mean
    ) / feature_scale
    logits = standardized @ weight.reshape(-1) + float(bias.reshape(-1)[0])
    beta = global_beta + max_residual * np.tanh(logits)
    return beta.astype(np.float32, copy=False)


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def timing_summary(seconds: Sequence[float], n_queries: int) -> dict:
    median = float(statistics.median(seconds))
    return {
        "repetitions_seconds": [float(value) for value in seconds],
        "median_seconds": median,
        "min_seconds": float(min(seconds)),
        "max_seconds": float(max(seconds)),
        "iqr_seconds": _percentile(seconds, 75) - _percentile(seconds, 25),
        "median_microseconds_per_query": 1e6 * median / n_queries,
        "median_queries_per_second": n_queries / median,
    }


def time_function(
    function: Callable[[], object],
    repetitions: int,
    n_queries: int,
) -> tuple[dict, object]:
    times: list[float] = []
    output: object = None
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            output = function()
            elapsed_ns = time.perf_counter_ns() - started
            times.append(elapsed_ns / 1e9)
    finally:
        if was_enabled:
            gc.enable()
    return timing_summary(times, n_queries), output


def _load_cached_inputs(
    artifact_root: Path,
    domain: str,
    seed: int,
) -> dict:
    domain_dir = artifact_root / DOMAIN_DIRS[domain]
    memory_path = domain_dir / "test_memory.npz"
    neural_path = (
        domain_dir / "predictions" / f"seed{seed}_test_top120.npz")
    gate_path = domain_dir / f"seed{seed}_dynamic_beta_gate.npz"
    ranks_path = domain_dir / f"seed{seed}_dynamic_beta_ranks.npz"
    missing = [
        str(path) for path in (
            memory_path, neural_path, gate_path, ranks_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "required frozen/cached artifacts are missing: "
            + ", ".join(missing))

    with np.load(memory_path) as saved:
        memory_keys = [str(value) for value in saved["keys"]]
        memory = saved["selected"].astype(np.int32, copy=True)
        memory_fingerprint = str(saved["fingerprint"].item())
    with np.load(neural_path) as saved:
        neural_keys = [str(value) for value in saved["keys"]]
        neural = saved["rankings"].astype(np.int32, copy=True)
        neural_fingerprint = str(saved["fingerprint"].item())
    with np.load(gate_path) as saved:
        gate = {key: saved[key].copy() for key in saved.files}
    with np.load(ranks_path) as saved:
        rank_keys = [str(value) for value in saved["test_keys"]]
        expected_features = saved["test_features"][
            :, FEATURE_GROUPS["context"]].astype(np.float32, copy=True)
        expected_betas = saved["test_dynamic_beta"].astype(
            np.float32, copy=True)
        expected_fused = saved["test_dynamic_top20"].astype(
            np.int32, copy=True)

    if memory_keys != neural_keys or memory_keys != rank_keys:
        raise RuntimeError(f"{domain}: cached test-query orders disagree")
    if memory_fingerprint != neural_fingerprint:
        raise RuntimeError(f"{domain}: cached query fingerprints disagree")
    if memory.shape != neural.shape or memory.shape[1] != 120:
        raise RuntimeError(
            f"{domain}: expected matched top-120 expert arrays, got "
            f"{memory.shape} and {neural.shape}")
    if gate["feature_mean"].shape != (3,):
        raise RuntimeError(
            f"{domain}: primary gate must use three features, got "
            f"{gate['feature_mean'].shape}")
    return {
        "keys": memory_keys,
        "memory": memory,
        "neural": neural,
        "gate": gate,
        "expected_features": expected_features,
        "expected_betas": expected_betas,
        "expected_fused": expected_fused,
        "paths": {
            "memory": str(memory_path),
            "neural": str(neural_path),
            "gate": str(gate_path),
            "ranks": str(ranks_path),
        },
        "query_fingerprint": memory_fingerprint,
    }


def _verification(
    features: np.ndarray,
    betas: np.ndarray,
    fused: np.ndarray,
    cached: dict,
) -> dict:
    feature_error = float(np.max(np.abs(
        features - cached["expected_features"])))
    beta_error = float(np.max(np.abs(
        betas - cached["expected_betas"])))
    fused_matches = bool(np.array_equal(
        fused, cached["expected_fused"]))
    if feature_error > 1e-6:
        raise RuntimeError(
            f"feature implementation mismatch: max error {feature_error}")
    if beta_error > 1e-6:
        raise RuntimeError(
            f"gate implementation mismatch: max error {beta_error}")
    if not fused_matches:
        raise RuntimeError("fusion output differs from frozen rank artifact")
    return {
        "feature_max_abs_error_vs_frozen_artifact": feature_error,
        "beta_max_abs_error_vs_frozen_artifact": beta_error,
        "fused_top20_exact_match_vs_frozen_artifact": fused_matches,
    }


def _reference_expert_times(
    reference: dict | None,
    domain: str,
    n_queries: int,
    new_postprocessing_seconds: float,
) -> dict | None:
    if reference is None or domain not in reference.get("domains", {}):
        return None
    old = reference["domains"][domain]
    if int(old["n_queries"]) != n_queries:
        raise RuntimeError(
            f"{domain}: reference benchmark query count differs "
            f"({old['n_queries']} != {n_queries})")
    memory_seconds = float(old["memory_seconds"])
    neural_seconds = float(old["neural_seconds"])
    estimate = memory_seconds + neural_seconds + new_postprocessing_seconds
    return {
        "status": "cross-run_component_composition_not_cotimed",
        "memory_seconds_from_reference_run": memory_seconds,
        "pasgr_seconds_from_reference_run": neural_seconds,
        "new_dynamic_postprocessing_median_seconds": (
            new_postprocessing_seconds),
        "estimated_total_seconds": estimate,
        "estimated_milliseconds_per_query": 1000.0 * estimate / n_queries,
        "estimated_queries_per_second": n_queries / estimate,
        "estimated_component_share": {
            "memory": memory_seconds / estimate,
            "pasgr": neural_seconds / estimate,
            "new_dynamic_postprocessing": (
                new_postprocessing_seconds / estimate),
        },
        "reference_source_protocol": reference.get("protocol", {}),
        "caveat": (
            "Memory and PASGR were measured in the earlier warm-expert run. "
            "The retired router timings and retired end-to-end total are not "
            "used. This composition is an estimate, not a newly co-timed "
            "dynamic-beta end-to-end measurement."
        ),
    }


def _tex_decimal(value: float, digits: int) -> str:
    rendered = f"{value:.{digits}f}"
    if rendered.startswith("-0."):
        return "-" + rendered[2:]
    if rendered.startswith("0."):
        return rendered[1:]
    return rendered


def _tex_scientific(value: float, digits: int = 2) -> str:
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def write_tex_macros(result: dict, destination: Path) -> None:
    """Write paper numbers directly from the verified benchmark artifact."""
    rows = [
        "% Generated by benchmark_dynamic_beta_inference.py; do not edit.",
    ]
    maximum_beta_error = 0.0
    exact_top20 = True
    for domain in DEFAULT_DOMAINS:
        if domain not in result["domains"]:
            continue
        block = result["domains"][domain]
        prefix = DOMAIN_MACRO_PREFIX[domain]
        headline = block["headline"]
        verification = block["verification"]
        maximum_beta_error = max(
            maximum_beta_error,
            float(verification[
                "beta_max_abs_error_vs_frozen_artifact"
            ]),
        )
        exact_top20 = exact_top20 and bool(
            verification[
                "fused_top20_exact_match_vs_frozen_artifact"
            ]
        )
        estimate = block.get("expert_inclusive_cross_run_estimate")
        total = (
            "\\textit{n/a}"
            if estimate is None
            else _tex_decimal(
                float(estimate["estimated_milliseconds_per_query"]), 3)
        )
        expert = "\\textit{n/a}"
        if estimate is not None:
            post_ms = (
                float(headline[
                    "dynamic_postprocessing_including_rrf_microseconds_per_query"
                ])
                / 1000.0
            )
            expert = _tex_decimal(
                float(estimate["estimated_milliseconds_per_query"]) - post_ms,
                3,
            )
        rows.extend([
            (
                f"\\newcommand{{\\DynamicRuntime{prefix}Queries}}{{"
                f"{int(block['n_queries']):,}"
                "}"
            ),
            (
                f"\\newcommand{{\\DynamicRuntime{prefix}GateUs}}{{"
                f"{_tex_decimal(float(headline[
                    'dynamic_allocation_feature_plus_gate_microseconds_per_query'
                ]), 3)}"
                "}"
            ),
            (
                f"\\newcommand{{\\DynamicRuntime{prefix}PostUs}}{{"
                f"{_tex_decimal(float(headline[
                    'dynamic_postprocessing_including_rrf_microseconds_per_query'
                ]), 2)}"
                "}"
            ),
            (
                f"\\newcommand{{\\DynamicRuntime{prefix}ExpertMs}}{{"
                f"{expert}"
                "}"
            ),
            (
                f"\\newcommand{{\\DynamicRuntime{prefix}TotalMs}}{{"
                f"{total}"
                "}"
            ),
        ])
    rows.extend([
        (
            "\\newcommand{\\DynamicRuntimeMaxBetaError}{"
            f"{_tex_scientific(maximum_beta_error)}"
            "}"
        ),
        (
            "\\newcommand{\\DynamicRuntimeExactTopTwenty}{"
            f"{'exact' if exact_top20 else 'not exact'}"
            "}"
        ),
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-queries", type=int, default=512)
    parser.add_argument("--max-residual", type=float, default=0.10)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_artifacts")
    parser.add_argument(
        "--reference-expert-benchmark", type=Path,
        default=HERE / "cearfn_inference_benchmark.json")
    parser.add_argument(
        "--no-reference-expert-times", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "dynamic_beta_inference_benchmark.json")
    parser.add_argument(
        "--tex-macro-output",
        type=Path,
        default=(
            HERE / "paper"
            / "generated_dynamic_beta_runtime_macros.tex"
        ),
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.warmup_queries < 1:
        parser.error("--warmup-queries must be positive")
    if not 0.0 < args.max_residual <= 0.5:
        parser.error("--max-residual must be in (0, 0.5]")
    unknown = [domain for domain in args.domains if domain not in DOMAIN_DIRS]
    if unknown:
        parser.error(f"unknown domains: {', '.join(unknown)}")

    reference = None
    if (
        not args.no_reference_expert_times
        and args.reference_expert_benchmark.exists()
    ):
        reference = json.loads(
            args.reference_expert_benchmark.read_text())

    result = {
        "protocol": {
            "name": (
                "CEARF-N training-only bounded dynamic-beta warm "
                "post-processing benchmark"
            ),
            "seed": args.seed,
            "hardware": hardware_description(),
            "clock": "time.perf_counter_ns (monotonic)",
            "repetitions": args.repetitions,
            "warmup_queries": args.warmup_queries,
            "reported_statistic": "median; all repetitions and IQR retained",
            "garbage_collection_during_timing": "disabled",
            "warm_state": (
                "test queries, frozen gate, and cached top-120 expert ranks "
                "resident in memory"
            ),
            "candidate_width_per_expert": 120,
            "output_width": 20,
            "rrf_constant": 20.0,
            "max_residual": args.max_residual,
            "effective_gate_features": list(PRIMARY_FEATURE_NAMES),
            "included_in_measured_dynamic_postprocessing": [
                "three target-free context features",
                "feature standardization",
                "bounded linear beta assignment",
                "weighted reciprocal-rank fusion to top-20",
            ],
            "excluded_from_measured_dynamic_postprocessing": [
                "data/artifact loading",
                "training and OOF calibration",
                "CEARF index construction and memory retrieval",
                "PASGR prediction",
                "metric computation and artifact writing",
                "warm-up",
            ],
            "expert_total_note": (
                "Any reported total that includes memory and PASGR is "
                "explicitly a cross-run estimate from raw expert component "
                "times; retired-router timings are not reused."
            ),
        },
        "domains": {},
    }

    for domain in args.domains:
        print(f"[DYNAMIC-INFER] === {domain} ===", flush=True)
        cached = _load_cached_inputs(
            args.artifact_root, domain, args.seed)
        data = loaders.ALL_LOADERS[domain]()
        keys = cached["keys"]
        queries = data["test_queries"]
        if any(uid not in queries for uid in keys):
            raise RuntimeError(f"{domain}: cached key absent from test queries")

        # Frequency/head preparation belongs to the frozen expert state and is
        # intentionally outside the per-query inference timer.
        frequency = Counter(
            int(item)
            for sequence in data["train_sessions"].values()
            for item in sequence
        )
        head = set(popularity_partition(
            frequency, data["n_items"])[0].tolist())
        gate = cached["gate"]
        feature_mean = gate["feature_mean"].astype(
            np.float32, copy=False)
        feature_scale = gate["feature_scale"].astype(
            np.float32, copy=False)
        weight = gate["model::network.weight"].astype(
            np.float32, copy=False)
        bias = gate["model::network.bias"].astype(
            np.float32, copy=False)
        global_beta = float(gate["global_beta"])

        def make_features() -> np.ndarray:
            return context_feature_matrix(
                queries, keys, frequency, head)

        # Untimed materialization supplies inputs for isolated component
        # timings and for exact verification against the frozen experiment.
        features = make_features()

        def make_betas() -> np.ndarray:
            return bounded_linear_beta(
                features, feature_mean, feature_scale, weight, bias,
                global_beta, args.max_residual)

        betas = make_betas()

        def make_fused() -> np.ndarray:
            return fuse_with_dynamic_beta(
                cached["memory"], cached["neural"], betas,
                topk=20, constant=20.0)

        warm_n = min(args.warmup_queries, len(keys))
        warm_features = context_feature_matrix(
            queries, keys[:warm_n], frequency, head)
        warm_betas = bounded_linear_beta(
            warm_features, feature_mean, feature_scale, weight, bias,
            global_beta, args.max_residual)
        fuse_with_dynamic_beta(
            cached["memory"][:warm_n],
            cached["neural"][:warm_n],
            warm_betas,
            topk=20,
            constant=20.0,
        )

        feature_timing, features_out = time_function(
            make_features, args.repetitions, len(keys))
        features = np.asarray(features_out)
        gate_timing, betas_out = time_function(
            make_betas, args.repetitions, len(keys))
        betas = np.asarray(betas_out)
        fusion_timing, _ = time_function(
            make_fused, args.repetitions, len(keys))

        def full_postprocessing() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            pipeline_features = make_features()
            pipeline_betas = bounded_linear_beta(
                pipeline_features,
                feature_mean,
                feature_scale,
                weight,
                bias,
                global_beta,
                args.max_residual,
            )
            pipeline_fused = fuse_with_dynamic_beta(
                cached["memory"],
                cached["neural"],
                pipeline_betas,
                topk=20,
                constant=20.0,
            )
            return pipeline_features, pipeline_betas, pipeline_fused

        pipeline_timing, pipeline_out = time_function(
            full_postprocessing, args.repetitions, len(keys))
        pipeline_features, pipeline_betas, pipeline_fused = pipeline_out
        verification = _verification(
            np.asarray(pipeline_features),
            np.asarray(pipeline_betas),
            np.asarray(pipeline_fused),
            cached,
        )
        reference_total = _reference_expert_times(
            reference,
            domain,
            len(keys),
            float(pipeline_timing["median_seconds"]),
        )
        allocation_us = (
            float(feature_timing["median_microseconds_per_query"])
            + float(gate_timing["median_microseconds_per_query"])
        )

        result["domains"][domain] = {
            "n_queries": len(keys),
            "n_items": int(data["n_items"]),
            "query_fingerprint": cached["query_fingerprint"],
            "global_beta": global_beta,
            "realized_beta": {
                "mean": float(np.mean(pipeline_betas)),
                "std": float(np.std(pipeline_betas)),
                "min": float(np.min(pipeline_betas)),
                "max": float(np.max(pipeline_betas)),
            },
            "timing": {
                "context_features": feature_timing,
                "bounded_linear_gate": gate_timing,
                "rrf_fusion": fusion_timing,
                "dynamic_postprocessing_pipeline": pipeline_timing,
            },
            "headline": {
                "dynamic_allocation_feature_plus_gate_microseconds_per_query": (
                    allocation_us),
                "dynamic_postprocessing_including_rrf_microseconds_per_query": (
                    float(pipeline_timing[
                        "median_microseconds_per_query"])),
                "dynamic_postprocessing_queries_per_second": float(
                    pipeline_timing["median_queries_per_second"]),
            },
            "verification": verification,
            "artifacts": cached["paths"],
            "expert_inclusive_cross_run_estimate": reference_total,
        }
        print(
            f"[DYNAMIC-INFER] {domain}: "
            f"{pipeline_timing['median_microseconds_per_query']:.2f} "
            "us/query dynamic post-processing",
            flush=True,
        )

    args.output.write_text(json.dumps(result, indent=2) + "\n")
    write_tex_macros(result, args.tex_macro_output)
    print(f"[DYNAMIC-INFER] saved {args.output}", flush=True)
    print(
        f"[DYNAMIC-INFER] saved {args.tex_macro_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
