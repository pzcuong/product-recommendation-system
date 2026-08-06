#!/usr/bin/env python3
"""Matched architecture-by-teacher audit over locked Amazon rank artifacts."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

import loaders
from run_stratified_evaluation import popularity_strata


HERE = Path(__file__).resolve().parent
SEEDS = (42, 123, 456)
TEACHERS = ("None", "TF-IDF/SVD", "MiniLM")
DOMAINS = {
    "Video Games": {
        "slug": "video_games",
        "loader": lambda: loaders.load_amazon("Video_Games"),
    },
    "Baby Products": {
        "slug": "baby_products",
        "loader": lambda: loaders.load_amazon("amazon_baby"),
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


def artifact_spec(model: str, teacher: str, slug: str, seed: int) -> tuple[Path, str]:
    if model == "CEARF-N":
        if teacher == "None":
            return (
                HERE / "cearfn_v2_nometa_artifacts" /
                f"{slug}_nometa_seed{seed}_ranks.npz",
                "regime_rank",
            )
        if teacher == "TF-IDF/SVD":
            return (
                HERE / "cearfn_v2_nested_artifacts" /
                f"{slug}_v2_seed{seed}_ranks.npz",
                "selected_rank",
            )
        return (
            HERE / "cearfn_v2_minilm_artifacts" /
            f"{slug}_v2_seed{seed}_ranks.npz",
            "selected_rank",
        )
    if teacher == "None":
        return (
            HERE / "paper_baseline_artifacts" /
            f"{slug}_full_narm_seed{seed}_ranks.npz",
            "ranks",
        )
    if teacher == "TF-IDF/SVD":
        return (
            HERE / "semantic_init_artifacts" /
            f"{slug}_narm_sem_seed{seed}_ranks.npz",
            "ranks",
        )
    return (
        HERE / "narm_minilm_fairness_artifacts" /
        f"{slug}_narm_minilm_seed{seed}_ranks.npz",
        "ranks",
    )


def summarize(ranks_by_seed: dict[int, np.ndarray], mask: np.ndarray) -> dict:
    per_seed = {}
    for seed in SEEDS:
        ranks = ranks_by_seed[seed][mask]
        hits = (ranks > 0) & (ranks <= 20)
        per_seed[str(seed)] = float(hits.mean())
    values = np.asarray(list(per_seed.values()), dtype=np.float64)
    return {
        "per_seed_recall@20": per_seed,
        "mean_recall@20": float(values.mean()),
        "std_recall@20": float(values.std(ddof=1)),
        "n_queries": int(mask.sum()),
    }


def effects(block: dict) -> dict:
    output = {
        "architecture_gap_cearfn_minus_narm": {},
        "teacher_effect_vs_none": {"CEARF-N": {}, "NARM": {}},
        "interaction_difference_in_differences": {},
        "minilm_minus_tfidf": {},
    }
    for teacher in TEACHERS:
        output["architecture_gap_cearfn_minus_narm"][teacher] = (
            block["CEARF-N"][teacher]["mean_recall@20"]
            - block["NARM"][teacher]["mean_recall@20"]
        )
    for model in ("CEARF-N", "NARM"):
        base = block[model]["None"]["mean_recall@20"]
        for teacher in TEACHERS[1:]:
            output["teacher_effect_vs_none"][model][teacher] = (
                block[model][teacher]["mean_recall@20"] - base
            )
    for teacher in TEACHERS[1:]:
        output["interaction_difference_in_differences"][teacher] = (
            output["teacher_effect_vs_none"]["CEARF-N"][teacher]
            - output["teacher_effect_vs_none"]["NARM"][teacher]
        )
    for model in ("CEARF-N", "NARM"):
        output["minilm_minus_tfidf"][model] = (
            block[model]["MiniLM"]["mean_recall@20"]
            - block[model]["TF-IDF/SVD"]["mean_recall@20"]
        )
    output["minilm_minus_tfidf"]["interaction"] = (
        output["minilm_minus_tfidf"]["CEARF-N"]
        - output["minilm_minus_tfidf"]["NARM"]
    )
    return output


def main() -> None:
    result = {
        "protocol": {
            "test_reselection": False,
            "evaluation": "full-catalogue Recall@20 over locked rank vectors",
            "factorial_axes": ["architecture: CEARF-N/NARM",
                               "teacher: None/TF-IDF-SVD/MiniLM"],
            "interaction": (
                "(CEARF teacher effect vs None) - "
                "(NARM teacher effect vs None)"
            ),
            "caveat": (
                "Descriptive system-level interaction, not a causal factorial "
                "decomposition. CEARF-N/None uses the regime router and the "
                "default PASGR configuration, whereas TF-IDF/SVD and MiniLM "
                "use teacher-specific validation-selected PASGR cells and "
                "router families; NARM also uses its condition-specific "
                "validation-selected checkpoint."
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
        pop_map, pop_meta = popularity_strata(
            Counter(data["item_freq"]), data["n_items"])
        labels = np.asarray(
            [pop_map.get(int(target), "tail") for target in targets],
            dtype=object,
        )
        fingerprint = query_fingerprint(data["test_queries"])
        ranks: dict[str, dict[str, dict[int, np.ndarray]]] = {
            model: {teacher: {} for teacher in TEACHERS}
            for model in ("CEARF-N", "NARM")
        }
        alignment_exceptions = []
        for model in ranks:
            for teacher in TEACHERS:
                for seed in SEEDS:
                    path, key = artifact_spec(model, teacher, spec["slug"], seed)
                    with np.load(path, allow_pickle=True) as saved:
                        artifact_fp = str(saved["test_fingerprint"].item())
                        if artifact_fp != fingerprint:
                            # The legacy TF-IDF NARM writer accidentally
                            # persisted a metric scalar in this field. Its
                            # generator (`predict_array`) nevertheless fixes
                            # query order as `sorted(test)` and computes ranks
                            # against targets in that same order. Accept only
                            # this known format and record the exception.
                            legacy_numeric = (
                                model == "NARM"
                                and teacher == "TF-IDF/SVD"
                                and np.asarray(saved["test_fingerprint"]).dtype.kind
                                in {"f", "i", "u"}
                            )
                            if not legacy_numeric:
                                raise ValueError(
                                    f"{path}: query fingerprint mismatch")
                            alignment_exceptions.append({
                                "artifact": str(path.relative_to(HERE)),
                                "stored_value": artifact_fp,
                                "accepted_because": (
                                    "legacy writer stored a metric scalar; "
                                    "source predict_array and targets_for both "
                                    "use sorted query identifiers"
                                ),
                            })
                        vector = np.asarray(saved[key], dtype=np.int32)
                    if len(vector) != len(keys):
                        raise ValueError(f"{path}: rank/query length mismatch")
                    ranks[model][teacher][seed] = vector

        masks = {
            "overall": np.ones(len(keys), dtype=bool),
            **{name: labels == name for name in ("head", "torso", "tail")},
        }
        domain_block = {
            "metadata": {
                "n_queries": len(keys),
                "query_fingerprint_sha256": fingerprint,
                "popularity_split": pop_meta,
                "alignment_exceptions": alignment_exceptions,
            },
            "overall": {},
            "popularity_strata": {},
        }
        for stratum, mask in masks.items():
            block = {
                model: {
                    teacher: summarize(ranks[model][teacher], mask)
                    for teacher in TEACHERS
                }
                for model in ("CEARF-N", "NARM")
            }
            payload = {"cells": block, "effects": effects(block)}
            if stratum == "overall":
                domain_block["overall"] = payload
            else:
                domain_block["popularity_strata"][stratum] = payload
        result["domains"][domain] = domain_block

    path = HERE / "teacher_interaction_audit.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
