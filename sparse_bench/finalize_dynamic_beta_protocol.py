#!/usr/bin/env python3
"""Finalize validation-split provenance without changing experimental values.

The dynamic-beta runner declares a deterministic 5,000-query validation
subset from each loader's larger validation-candidate mapping. Earlier v2
manifests used the ambiguous word ``official`` for that declared subset. This
utility reconstructs the split, verifies its fingerprints and zero overlap
with OOF calibration, and rewrites only protocol/provenance metadata.

Training reports, beta values, predictions, ranks, and evaluation metrics are
hashed before and after finalization and must remain byte-for-byte identical
under canonical JSON serialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cearf
import loaders
from run_dynamic_beta import (
    DOMAINS,
    canonical_validation_sources,
    make_training_oof_split,
)
from validation_protocol import hold_out_validation_targets


HERE = Path(__file__).resolve().parent
PROTOCOL = "dynamic-beta-train-only-v2-declared-validation-5k"


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def numerical_payload(results: dict) -> dict:
    return {
        domain: [
            {
                "seed": int(run["seed"]),
                "training": run["training"],
                "validation": run["validation"],
                "test": run["test"],
            }
            for run in results[domain]["runs"]
        ]
        for domain in DOMAINS
    }


def declared_validation_subset(data: dict, valid_cap: int) -> dict:
    candidates = data["valid_queries"]
    if len(candidates) <= valid_cap:
        return dict(candidates)
    keys = sorted(candidates, key=cearf._stable_fraction)[:valid_cap]
    return {uid: candidates[uid] for uid in keys}


def validation_target_boundary(
        sessions: dict,
        declared_queries: dict,
) -> dict:
    shortened = hold_out_validation_targets(sessions, declared_queries)
    sources = canonical_validation_sources(declared_queries)
    removed = 0
    already_outside = 0
    for uid, query in declared_queries.items():
        source = str(uid)[:-2] if str(uid).endswith("_v") else str(uid)
        targets = [int(item) for item in query.get("targets", ())]
        before = [int(item) for item in sessions.get(source, ())]
        after = [int(item) for item in shortened.get(source, ())]
        if targets and len(after) + 1 == len(before) and before[-1] in targets:
            removed += 1
        elif source in sources and targets and (
                not before or before[-1] not in targets):
            already_outside += 1
    return {
        "sessions_after_boundary": shortened,
        "declared_validation_target_events_removed": removed,
        "declared_validation_targets_already_outside_training_histories": (
            already_outside
        ),
    }


def reconstruct_domain_split(
        domain: str,
        valid_cap: int,
        oof_fraction: float,
        oof_cap: int,
        profile_cap: int,
) -> tuple[dict, dict]:
    data = loaders.ALL_LOADERS[domain]()
    candidate_queries = data["valid_queries"]
    candidate_sources = canonical_validation_sources(candidate_queries)
    declared_queries = declared_validation_subset(data, valid_cap)
    declared_sources = canonical_validation_sources(declared_queries)
    boundary = validation_target_boundary(
        data["train_sessions"], declared_queries)
    (
        _,
        profile_queries,
        gate_queries,
        split,
    ) = make_training_oof_split(
        boundary["sessions_after_boundary"],
        declared_sources,
        oof_fraction,
        oof_cap,
        profile_cap,
    )
    held_sources = {
        str(uid).split("::", 1)[1]
        for uid in (*profile_queries, *gate_queries)
    }
    split.update({
        "validation_definition": (
            "stable-hash declared subset selected before allocation fit"
        ),
        "validation_candidate_pool_queries": len(candidate_queries),
        "validation_candidate_pool_sources": len(candidate_sources),
        "declared_validation_queries": len(declared_queries),
        "declared_validation_sources": len(declared_sources),
        "validation_candidate_pool_source_overlap": len(
            held_sources & candidate_sources),
        "unselected_validation_candidates_remain_training_events": True,
        "declared_validation_target_events_removed_from_tuning_sessions": (
            boundary["declared_validation_target_events_removed"]
        ),
        "declared_validation_targets_already_outside_training_histories": (
            boundary[
                "declared_validation_targets_already_outside_training_histories"
            ]
        ),
    })
    audit = {
        "domain": domain,
        "candidate_pool_queries": len(candidate_queries),
        "declared_validation_queries": len(declared_queries),
        "declared_validation_source_overlap_with_oof": split[
            "declared_validation_source_overlap"],
        "candidate_pool_source_overlap_with_oof": split[
            "validation_candidate_pool_source_overlap"],
        "target_events_removed": split[
            "declared_validation_target_events_removed_from_tuning_sessions"],
        "targets_already_outside_training": split[
            "declared_validation_targets_already_outside_training_histories"],
        "profile_query_fingerprint": split["profile_query_fingerprint"],
        "gate_query_fingerprint": split["gate_query_fingerprint"],
    }
    return split, audit


def finalize(
        results: dict,
        valid_cap: int,
        oof_fraction: float,
        oof_cap: int,
        profile_cap: int,
        apply: bool,
) -> tuple[dict, dict]:
    before_hash = canonical_sha256(numerical_payload(results))
    output = json.loads(json.dumps(results))
    audits = {}
    for domain in DOMAINS:
        if domain not in output:
            raise ValueError(f"missing completed domain: {domain}")
        reconstructed, audit = reconstruct_domain_split(
            domain, valid_cap, oof_fraction, oof_cap, profile_cap)
        recorded = output[domain].get("split", {})
        for key in ("profile_query_fingerprint", "gate_query_fingerprint"):
            if recorded.get(key) != reconstructed[key]:
                raise ValueError(
                    f"{domain}: {key} differs from reconstructed protocol")
        if reconstructed["declared_validation_source_overlap"] != 0:
            raise ValueError(
                f"{domain}: declared validation overlaps OOF calibration")
        output[domain]["protocol"] = PROTOCOL
        output[domain]["split"] = reconstructed
        output[domain]["protocol_metadata_finalized"] = {
            "ambiguous_official_wording_removed": True,
            "experimental_values_changed": False,
        }
        for run in output[domain]["runs"]:
            manifest_path = Path(run["manifest"])
            manifest = json.loads(manifest_path.read_text())
            recorded_manifest_split = manifest.get("split", {})
            for key in (
                    "profile_query_fingerprint",
                    "gate_query_fingerprint"):
                if recorded_manifest_split.get(key) != reconstructed[key]:
                    raise ValueError(
                        f"{domain} seed {run['seed']}: manifest split mismatch")
            manifest["protocol"] = PROTOCOL
            manifest["split"] = reconstructed
            manifest.pop(
                "created_before_official_validation_prediction", None)
            manifest.pop(
                "frozen_before_official_validation_or_test_evaluation_under_protocol",
                None,
            )
            manifest[
                "frozen_before_declared_validation_or_test_evaluation"
            ] = True
            manifest["protocol_metadata_finalized"] = {
                "experimental_values_changed": False,
                "reason": (
                    "distinguish declared 5k validation from loader candidate pool"
                ),
            }
            if apply:
                manifest_path.write_text(json.dumps(manifest, indent=2))
        audits[domain] = audit
    after_hash = canonical_sha256(numerical_payload(output))
    if after_hash != before_hash:
        raise AssertionError("protocol finalization changed experimental values")
    report = {
        "protocol": PROTOCOL,
        "experimental_payload_sha256_before": before_hash,
        "experimental_payload_sha256_after": after_hash,
        "experimental_values_changed": False,
        "domains": audits,
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_results.json",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=HERE / "dynamic_beta_protocol_audit.json",
    )
    parser.add_argument("--valid-cap", type=int, default=5_000)
    parser.add_argument("--oof-fraction", type=float, default=.10)
    parser.add_argument("--oof-cap", type=int, default=5_000)
    parser.add_argument("--profile-cap", type=int, default=1_000)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    finalized, report = finalize(
        results,
        args.valid_cap,
        args.oof_fraction,
        args.oof_cap,
        args.profile_cap,
        args.apply,
    )
    if args.apply:
        backup = args.results.with_suffix(
            args.results.suffix + ".pre-protocol-finalization.bak")
        if not backup.exists():
            shutil.copy2(args.results, backup)
        args.results.write_text(json.dumps(finalized, indent=2))
        args.audit_output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
