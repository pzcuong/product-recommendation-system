#!/usr/bin/env python3
"""Audit the semantic-teacher cache selected by ``semantic_matrix``.

This script never edits experiment results or teacher arrays.  It records the
resolver precedence implemented in ``run_pasgr_full.py`` and fingerprints the
already-selected caches so manuscript labels can be traced to bytes on disk.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
RESOLVER = HERE / "run_pasgr_full.py"

DOMAIN_DECLARATIONS: dict[str, dict[str, Any]] = {
    "Video_Games": {
        "corrected_label": "E5-small cached item-text embeddings",
        "expected_selected_cache": "artifacts/Video_Games_e5_small.npy",
        "evidence": [
            "PILOT_RESULTS.md:31",
            "pilot_casm_seed42.json:9",
        ],
        "caveat": (
            "The cache predates this audit and has no encoder sidecar; the "
            "E5-small attribution is supported by the canonical experiment "
            "notes and cache name, while SHA-256 below fixes the exact bytes."
        ),
    },
    "Baby_Products": {
        "corrected_label": "TF-IDF/SVD cached item-text embeddings",
        "expected_selected_cache": "artifacts/Baby_Products_pasgr_semantic.npy",
        "evidence": [
            "PILOT_RESULTS.md:31",
            "pilot_casm_seed42.json:10",
        ],
        "caveat": None,
    },
    "Diginetica_HID": {
        "corrected_label": "TF-IDF/SVD cached product-name embeddings",
        "expected_selected_cache": "artifacts/Diginetica_HID_pasgr_semantic.npy",
        "evidence": [
            "run_pasgr_full.py:31-57",
            "loaders.py:542-550",
        ],
        "caveat": None,
    },
}


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolver_precedence() -> list[dict[str, str]]:
    source = RESOLVER.read_text()
    markers = [
        ("if legacy.exists():", "legacy_e5_small"),
        ("if artifact.exists():", "pasgr_semantic"),
        ("if texts:", "build_tfidf_svd_from_item_texts"),
        ("if categories:", "build_seeded_category_vectors"),
    ]
    located = []
    for marker, name in markers:
        offset = source.find(marker)
        if offset < 0:
            raise RuntimeError(f"resolver marker not found: {marker}")
        located.append((offset, name, marker))
    if located != sorted(located):
        raise RuntimeError("semantic_matrix resolver precedence has changed")
    return [
        {"priority": str(index), "name": name, "source_marker": marker}
        for index, (_, name, marker) in enumerate(located, start=1)
    ]


def audit_domain(domain: str, declaration: dict[str, Any]) -> dict[str, Any]:
    legacy = HERE / "artifacts" / f"{domain}_e5_small.npy"
    artifact = HERE / "artifacts" / f"{domain}_pasgr_semantic.npy"
    selected = legacy if legacy.exists() else artifact if artifact.exists() else None
    if selected is None:
        raise FileNotFoundError(f"no selected semantic cache for {domain}")
    expected = HERE / declaration["expected_selected_cache"]
    if selected.resolve() != expected.resolve():
        raise RuntimeError(
            f"{domain}: resolver selected {selected}, expected {expected}")
    array = np.load(selected, mmap_mode="r", allow_pickle=False)
    return {
        "corrected_teacher_label": declaration["corrected_label"],
        "selected_by_precedence": (
            "legacy_e5_small" if selected == legacy else "pasgr_semantic"),
        "selected_cache": str(selected.relative_to(HERE)),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "file_size_bytes": selected.stat().st_size,
        "sha256": sha256(selected),
        "evidence": declaration["evidence"],
        "caveat": declaration["caveat"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "teacher_resolver_provenance_audit.json",
    )
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_scope": (
            "No experiment result or teacher array is modified; this report "
            "fingerprints existing caches and the resolver order only."
        ),
        "resolver_source": str(RESOLVER.relative_to(HERE)),
        "resolver_precedence": resolver_precedence(),
        "domains": {
            domain: audit_domain(domain, declaration)
            for domain, declaration in DOMAIN_DECLARATIONS.items()
        },
        "known_label_correction": {
            "artifact": "narm_tfidf_fairness_nested_results.json",
            "issue": (
                "The runner's --teacher=tfidf label does not override the "
                "semantic_matrix cache resolver. Video_Games therefore used "
                "the higher-priority E5-small cache; Baby_Products used the "
                "TF-IDF/SVD cache."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
