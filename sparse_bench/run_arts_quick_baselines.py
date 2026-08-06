#!/usr/bin/env python3
"""Quick baseline check for Arts_Crafts_and_Sewing with progress logging.

Purpose:
  - get a fast, paper-usable signal on the new public domain without waiting for
    the full neighborhood grid;
  - score one strong V-SKNN config discovered from the aborted validation sweep;
  - score the transition baseline as a lower-bound memory constituent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import cearf
import loaders
from neighborhood_baselines import NeighborhoodConfig, NeighborhoodIndex
from run_cearfn_evidence import metrics_from_ranks, ranks_at_20, targets_for
from validation_protocol import hold_out_validation_targets


HERE = Path(__file__).resolve().parent
DOMAIN = "Arts_Crafts_and_Sewing"
CHUNK = 5000


def evaluate_chunked(index: NeighborhoodIndex, queries: dict, cfg: NeighborhoodConfig,
                     *, label: str, partial_path: Path | None = None) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    keys = list(queries)
    targets = targets_for(keys, queries)
    top20 = np.empty((len(keys), 20), dtype=np.int32)
    for start in range(0, len(keys), CHUNK):
        end = min(start + CHUNK, len(keys))
        batch_keys = keys[start:end]
        batch_queries = {k: queries[k] for k in batch_keys}
        pred = index.predict(batch_queries, cfg, topk=20)
        top20[start:end] = np.asarray([pred[k][:20] for k in batch_keys], dtype=np.int32)
        print(f"[ARTS-QUICK] {label} {end}/{len(keys)}", flush=True)
        if partial_path is not None:
            partial_ranks = ranks_at_20(top20[:end], targets[:end])
            partial = {
                "label": label,
                "completed_queries": int(end),
                "total_queries": int(len(keys)),
                "metrics": metrics_from_ranks(partial_ranks),
            }
            partial_path.write_text(json.dumps(partial, indent=2))
    ranks = ranks_at_20(top20, targets)
    return metrics_from_ranks(ranks), top20, ranks, keys


def transition_chunked(index: cearf.CEARFIndex, queries: dict, *, exclude_seen: bool,
                       label: str, partial_path: Path | None = None) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    keys = list(queries)
    targets = targets_for(keys, queries)
    top20 = np.empty((len(keys), 20), dtype=np.int32)
    for start in range(0, len(keys), CHUNK):
        end = min(start + CHUNK, len(keys))
        for row, uid in enumerate(keys[start:end], start):
            q = queries[uid]
            transition, _, popularity = index.component_rankings(q.get("context", ()))
            blocked = set(q.get("context", ())) if exclude_seen else set()
            rank = [x for x in transition if x not in blocked]
            chosen = set(rank)
            rank.extend(x for x in popularity if x not in blocked and x not in chosen)
            top20[row] = np.asarray(rank[:20], dtype=np.int32)
        print(f"[ARTS-QUICK] {label} {end}/{len(keys)}", flush=True)
        if partial_path is not None:
            partial_ranks = ranks_at_20(top20[:end], targets[:end])
            partial = {
                "label": label,
                "completed_queries": int(end),
                "total_queries": int(len(keys)),
                "metrics": metrics_from_ranks(partial_ranks),
            }
            partial_path.write_text(json.dumps(partial, indent=2))
    ranks = ranks_at_20(top20, targets)
    return metrics_from_ranks(ranks), top20, ranks, keys


def main() -> None:
    data = loaders.ALL_LOADERS[DOMAIN]()
    if len(data["valid_queries"]) > 5000:
        keys = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
        validation = {k: data["valid_queries"][k] for k in keys}
    else:
        validation = data["valid_queries"]

    exclude_seen = True
    tune_sessions = hold_out_validation_targets(data["train_sessions"], validation)
    tune_index = NeighborhoodIndex(tune_sessions, data["n_items"])
    final_index = NeighborhoodIndex(data["train_sessions"], data["n_items"])
    vsknn_cfg = NeighborhoodConfig(
        method="vsknn", k=500, sample_size=5000,
        weighting="div", score_weighting="div",
        exclude_seen=exclude_seen)

    partial_dir = HERE / "arts_quick_partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    valid_metrics, _, _, _ = evaluate_chunked(
        tune_index, validation, vsknn_cfg, label="vsknn-valid",
        partial_path=partial_dir / "vsknn_valid_partial.json")
    test_metrics, test_top20, test_ranks, test_keys = evaluate_chunked(
        final_index, data["test_queries"], vsknn_cfg, label="vsknn-test",
        partial_path=partial_dir / "vsknn_test_partial.json")

    cearf_index = cearf.CEARFIndex(
        data["train_sessions"], data["n_items"],
        cearf.CEARFConfig(exclude_seen=exclude_seen))
    transition_metrics, transition_top20, transition_ranks, transition_keys = transition_chunked(
        cearf_index, data["test_queries"], exclude_seen=exclude_seen, label="transition-test",
        partial_path=partial_dir / "transition_test_partial.json")

    artifact_dir = HERE / "neighborhood_baseline_artifacts_ext"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / "arts_crafts_and_sewing_vsknn_ranks.npz",
        top20=test_top20.astype(np.int32),
        ranks=test_ranks.astype(np.uint8),
        keys=np.asarray(test_keys, dtype=str))
    np.savez_compressed(
        artifact_dir / "arts_crafts_and_sewing_transition_ranks.npz",
        top20=transition_top20.astype(np.int32),
        ranks=transition_ranks.astype(np.uint8),
        keys=np.asarray(transition_keys, dtype=str))

    output = {
        "domain": DOMAIN,
        "n_items": data["n_items"],
        "n_validation_queries": len(validation),
        "n_test_queries": len(data["test_queries"]),
        "vsknn_config": vsknn_cfg.__dict__,
        "vsknn_validation": valid_metrics,
        "vsknn_test": test_metrics,
        "transition_test": transition_metrics,
    }
    out_path = HERE / "arts_quick_baselines.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(out_path, flush=True)


if __name__ == "__main__":
    main()
