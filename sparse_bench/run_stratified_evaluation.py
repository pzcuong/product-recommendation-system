#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import loaders


HERE = Path(__file__).resolve().parent
SEEDS = (42, 123, 456)
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")
PRIMARY_BASELINE = {
    "Video_Games": "narm",
    "Baby_Products": "sr_gnn",
    "Arts_Crafts_and_Sewing": "narm",
    "Diginetica_HID": "narm",
}
SEMANTIC_BASELINE = {
    "Video_Games": "narm_sem",
    "Baby_Products": "narm_sem",
    "Arts_Crafts_and_Sewing": "narm_sem",
}
DOMAIN_LOADERS = {
    "Video_Games": lambda: loaders.load_amazon("Video_Games"),
    "Baby_Products": lambda: loaders.load_amazon("amazon_baby"),
    "Arts_Crafts_and_Sewing": lambda: loaders.load_amazon("Arts_Crafts_and_Sewing"),
    "Diginetica_HID": loaders.load_diginetica_hid,
}
SLUG = {
    "Video_Games": "video_games",
    "Baby_Products": "baby_products",
    "Arts_Crafts_and_Sewing": "arts_crafts_and_sewing",
    "Diginetica_HID": "diginetica_hid",
}


def metrics_from_ranks(ranks: np.ndarray) -> dict[str, float | int]:
    ranks = np.asarray(ranks, dtype=np.int32)
    n = int(len(ranks))
    hit10 = (ranks > 0) & (ranks <= 10)
    hit20 = (ranks > 0) & (ranks <= 20)
    precision20 = hit20.astype(np.float64) / 20.0
    rr20 = np.zeros(n, dtype=np.float64)
    rr20[hit20] = 1.0 / ranks[hit20]
    ndcg20 = np.zeros(n, dtype=np.float64)
    ndcg20[hit20] = 1.0 / np.log2(ranks[hit20] + 1.0)
    return {
        "n": n,
        "hr@10": float(hit10.mean()) if n else 0.0,
        "recall@20": float(hit20.mean()) if n else 0.0,
        "precision@20": float(precision20.mean()) if n else 0.0,
        "mrr@20": float(rr20.mean()) if n else 0.0,
        "map@20": float(rr20.mean()) if n else 0.0,
        "ndcg@20": float(ndcg20.mean()) if n else 0.0,
    }


def popularity_strata(item_freq: dict[int, int], n_items: int) -> tuple[dict[int, str], dict[str, int]]:
    ordered = sorted(range(1, n_items), key=lambda item: (-item_freq.get(item, 0), item))
    n = len(ordered)
    head_cut = max(1, int(round(n * 0.2)))
    tail_start = min(n - 1, max(head_cut + 1, int(round(n * 0.8)))) if n > 1 else 1
    labels: dict[int, str] = {}
    for idx, item in enumerate(ordered):
        if idx < head_cut:
            labels[item] = "head"
        elif idx < tail_start:
            labels[item] = "torso"
        else:
            labels[item] = "tail"
    return labels, {
        "head_items": head_cut,
        "torso_items": max(tail_start - head_cut, 0),
        "tail_items": max(n - tail_start, 0),
    }


def context_bins(lengths: np.ndarray) -> tuple[list[str], dict[str, list[int]]]:
    lengths = np.asarray(lengths, dtype=np.int32)
    q1, q2 = np.quantile(lengths, [1 / 3, 2 / 3], method="nearest")
    q1 = int(q1)
    q2 = int(q2)
    labels = []
    for value in lengths:
        if value <= q1:
            labels.append("short")
        elif value <= q2:
            labels.append("medium")
        else:
            labels.append("long")
    return labels, {"short_max": q1, "medium_max": q2}


def load_rank_vector(path: Path, key: str = "ranks") -> np.ndarray:
    saved = np.load(path, allow_pickle=True)
    return np.asarray(saved[key], dtype=np.int32)


def load_cearfn_rank(domain: str, seed: int) -> np.ndarray:
    path = HERE / "cearfn_v2_nested_artifacts" / f"{SLUG[domain]}_v2_seed{seed}_ranks.npz"
    return load_rank_vector(path, key="selected_rank")


def load_primary_baseline_rank(domain: str, model: str, seed: int) -> np.ndarray:
    folder = HERE / ("paper_baseline_digi_nested_artifacts" if domain == "Diginetica_HID" else "paper_baseline_artifacts")
    path = folder / f"{SLUG[domain]}_full_{model}_seed{seed}_ranks.npz"
    return load_rank_vector(path)


def load_semantic_baseline_rank(domain: str, model: str, seed: int) -> np.ndarray:
    path = HERE / "semantic_init_artifacts" / f"{SLUG[domain]}_{model}_seed{seed}_ranks.npz"
    return load_rank_vector(path)


def aggregate_seed_metrics(rank_loader, domain: str, mask: np.ndarray) -> dict:
    per_seed = {}
    summary = {}
    for seed in SEEDS:
        ranks = rank_loader(domain, seed)[mask]
        per_seed[str(seed)] = metrics_from_ranks(ranks)
    for metric in ("hr@10", "recall@20", "precision@20", "mrr@20", "map@20", "ndcg@20"):
        values = np.asarray([per_seed[str(seed)][metric] for seed in SEEDS], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    summary["n_queries"] = int(mask.sum())
    return {"per_seed": per_seed, "summary": summary}


def compare_means(a: dict, b: dict) -> dict[str, float]:
    out = {}
    for metric in ("hr@10", "recall@20", "precision@20", "mrr@20", "map@20", "ndcg@20"):
        out[metric] = a["summary"][metric]["mean"] - b["summary"][metric]["mean"]
    return out


def main() -> None:
    output = HERE / "stratified_evaluation.json"
    results = {
        "protocol_note": {
            "single_positive_per_query": True,
            "map@20_equals_mrr@20": True,
            "context_bins": "per-domain terciles over observed context length",
            "popularity_bins": "item-frequency rank split into 20/60/20 head/torso/tail",
        },
        "domains": {},
    }
    for domain in DOMAINS:
        data = DOMAIN_LOADERS[domain]()
        # CEARF-N and neural-baseline rank artifacts use sorted query ids.
        keys = sorted(data["test_queries"])
        targets = np.asarray([int(data["test_queries"][k]["targets"][0]) for k in keys], dtype=np.int32)
        lengths = np.asarray([len(data["test_queries"][k]["context"]) for k in keys], dtype=np.int32)
        pop_label_map, pop_meta = popularity_strata(Counter(data["item_freq"]), data["n_items"])
        target_pop_labels = np.asarray([pop_label_map.get(int(t), "tail") for t in targets], dtype=object)
        context_labels, context_meta = context_bins(lengths)
        context_labels = np.asarray(context_labels, dtype=object)

        ce = aggregate_seed_metrics(lambda dom, seed: load_cearfn_rank(dom, seed), domain, np.ones(len(keys), dtype=bool))
        pb = aggregate_seed_metrics(
            lambda dom, seed: load_primary_baseline_rank(dom, PRIMARY_BASELINE[dom], seed),
            domain,
            np.ones(len(keys), dtype=bool),
        )
        domain_block = {
            "overall": {
                "cearfn": ce,
                "primary_baseline_name": PRIMARY_BASELINE[domain],
                "primary_baseline": pb,
                "delta_cearfn_minus_primary": compare_means(ce, pb),
            },
            "popularity_strata": {},
            "context_length_strata": {},
            "metadata": {
                "n_queries": len(keys),
                "popularity_split": pop_meta,
                "context_cutoffs": context_meta,
                "mean_context_length": float(lengths.mean()),
            },
        }
        if domain in SEMANTIC_BASELINE:
            sem = aggregate_seed_metrics(
                lambda dom, seed: load_semantic_baseline_rank(dom, SEMANTIC_BASELINE[dom], seed),
                domain,
                np.ones(len(keys), dtype=bool),
            )
            domain_block["overall"]["semantic_baseline_name"] = SEMANTIC_BASELINE[domain]
            domain_block["overall"]["semantic_baseline"] = sem
            domain_block["overall"]["delta_cearfn_minus_semantic"] = compare_means(ce, sem)

        for label in ("head", "torso", "tail"):
            mask = target_pop_labels == label
            ce_s = aggregate_seed_metrics(lambda dom, seed: load_cearfn_rank(dom, seed), domain, mask)
            pb_s = aggregate_seed_metrics(
                lambda dom, seed: load_primary_baseline_rank(dom, PRIMARY_BASELINE[dom], seed),
                domain,
                mask,
            )
            block = {
                "cearfn": ce_s,
                "primary_baseline": pb_s,
                "delta_cearfn_minus_primary": compare_means(ce_s, pb_s),
            }
            if domain in SEMANTIC_BASELINE:
                sem_s = aggregate_seed_metrics(
                    lambda dom, seed: load_semantic_baseline_rank(dom, SEMANTIC_BASELINE[dom], seed),
                    domain,
                    mask,
                )
                block["semantic_baseline"] = sem_s
                block["delta_cearfn_minus_semantic"] = compare_means(ce_s, sem_s)
            domain_block["popularity_strata"][label] = block

        for label in ("short", "medium", "long"):
            mask = context_labels == label
            ce_s = aggregate_seed_metrics(lambda dom, seed: load_cearfn_rank(dom, seed), domain, mask)
            pb_s = aggregate_seed_metrics(
                lambda dom, seed: load_primary_baseline_rank(dom, PRIMARY_BASELINE[dom], seed),
                domain,
                mask,
            )
            block = {
                "cearfn": ce_s,
                "primary_baseline": pb_s,
                "delta_cearfn_minus_primary": compare_means(ce_s, pb_s),
            }
            if domain in SEMANTIC_BASELINE:
                sem_s = aggregate_seed_metrics(
                    lambda dom, seed: load_semantic_baseline_rank(dom, SEMANTIC_BASELINE[dom], seed),
                    domain,
                    mask,
                )
                block["semantic_baseline"] = sem_s
                block["delta_cearfn_minus_semantic"] = compare_means(ce_s, sem_s)
            domain_block["context_length_strata"][label] = block

        results["domains"][domain] = domain_block

    output.write_text(json.dumps(results, indent=2))
    print(output)


if __name__ == "__main__":
    main()
