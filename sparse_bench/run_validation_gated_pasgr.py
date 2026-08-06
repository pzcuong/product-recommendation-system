#!/usr/bin/env python3
"""Validation-gated component selection for the PASGR neural residual.

Instead of hard-coding graph_weight/prototype_transport/contrastive_weight/
inbatch_weight to fixed defaults, this script sweeps a 16-cell grid on each
domain, fuses each candidate PASGR with the (already-tuned) CEARF memory, and
picks the configuration that maximises the same validation utility the rest
of the pipeline uses (0.5*R@6 + 0.5*R@20). The selected per-domain config is
written to ``pasgr_config_per_domain.json`` so downstream CEARF-N v2 runs can
reuse it without re-tuning.

Critical property: selection uses ONLY validation queries. Test labels never
influence the choice of graph/prototype/contrastive/inbatch switches.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import cearf
import loaders
import pasgr
from run_cearfn_ablation import fuse_matrix, select_beta
from run_cearfn_evidence import (
    load_or_build_memory, metrics_from_ranks, query_fingerprint,
    ranks_at_20, targets_for, tune_beta_arrays)
from run_pasgr_full import semantic_matrix
from validation_protocol import hold_out_validation_targets

HERE = Path(__file__).resolve().parent

# 16-cell grid: 2x2x2x2 over the four switches. Values mirror the existing
# ablation extremes (off vs default) so the chosen config is directly
# comparable to the always-on baseline reported in CEARFN_EVIDENCE_AUDIT.md.
GRID = list(itertools.product(
    (0.0, 0.35),               # graph_weight: off vs default
    (False, True),             # prototype_transport: off vs on
    (0.0, 0.15),               # contrastive_weight: off vs default
    (0.0, 0.10),               # inbatch_weight: off vs mild InfoNCE
))

DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID", "Tmall")
SEED = 42
EPOCHS = 4
REPEAT_PROTOCOL_DOMAINS = frozenset({"Diginetica_HID", "Tmall"})


def label_of(combination) -> str:
    gw, pt, cw, ib = combination
    parts = [
        f"graph{gw:g}",
        "proto" if pt else "noproto",
        f"contrast{cw:g}",
        f"inbatch{ib:g}",
    ]
    return "_".join(parts)


def validation_score(combination, selection):
    """Test-independent ordering; fewer active switches wins exact ties."""
    active = sum((combination[0] > 0, bool(combination[1]),
                  combination[2] > 0, combination[3] > 0))
    return (selection["utility"], selection["recall@20"],
            selection["recall@6"], -active)


def train_one(data, sessions, semantic, seed, epochs, combination):
    freq = Counter(x for sequence in sessions.values() for x in sequence)
    graph_weight, prototype_transport, contrastive_weight, inbatch_weight = combination
    config_data = dict(
        dim=64, prototypes=min(96, max(8, data["n_items"] // 250)),
        epochs=epochs, batch_size=512, hard_negatives=32,
        top_k=120, seed=seed,
        graph_weight=graph_weight,
        prototype_transport=prototype_transport,
        contrastive_weight=contrastive_weight,
        inbatch_weight=inbatch_weight,
    )
    config = pasgr.PASGRConfig(**config_data)
    assets = pasgr.build_prototype_graph_embeddings(
        sessions, data["n_items"], freq, semantic, config)
    return pasgr.train_pasgr(sessions, data["n_items"], freq, semantic,
                             config, prepared_assets=assets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--candidate-width", type=int, default=120)
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Optional subset of grid labels for recovery runs")
    parser.add_argument(
        "--semantic-dir", type=Path,
        help="Optional directory containing <domain>_minilm.npy teacher matrices.")
    parser.add_argument("--artifact-dir", type=Path,
                        default=HERE / "vgated_artifacts")
    parser.add_argument("--output", type=Path,
                        default=HERE / "pasgr_config_per_domain.json")
    args = parser.parse_args()

    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    for domain in args.domains:
        if domain in results and "selected" in results[domain]:
            print(f"[VGATED] {domain} already complete", flush=True)
            continue
        print(f"\n[VGATED] === {domain} ===", flush=True)
        data = loaders.ALL_LOADERS[domain]()
        # Match the evidence protocol's deterministic validation cap so the
        # locked Amazon memory cache can be reused (fingerprint is cap-sensitive).
        if len(data["valid_queries"]) > 5000:
            _vk = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
            data["valid_queries"] = {k: data["valid_queries"][k] for k in _vk}
        sessions = data["train_sessions"]
        tune_sessions = hold_out_validation_targets(
            sessions, data["valid_queries"])
        exclude_seen = domain not in REPEAT_PROTOCOL_DOMAINS
        config = cearf.CEARFConfig(exclude_seen=exclude_seen)
        tune_index = cearf.CEARFIndex(tune_sessions, data["n_items"], config)
        profiles, _ = cearf.tune_profiles(tune_index, data["valid_queries"])
        valid_memory = load_or_build_memory(
            args.artifact_dir / f"{domain.lower()}_nested_valid_memory.npz",
            tune_index, data["valid_queries"], profiles, args.candidate_width,
            f"{domain}-nested-valid")
        valid_keys = [str(x) for x in valid_memory["keys"]]
        valid_targets = targets_for(valid_keys, data["valid_queries"])
        if args.semantic_dir:
            semantic_path = args.semantic_dir / f"{domain.lower()}_minilm.npy"
            if not semantic_path.exists():
                raise FileNotFoundError(semantic_path)
            semantic = np.load(semantic_path).astype(np.float32)
        else:
            semantic = semantic_matrix(domain, data)
        if semantic is not None and semantic.shape[0] != data["n_items"]:
            raise ValueError(
                f"semantic rows ({semantic.shape[0]}) != n_items ({data['n_items']})")

        sweep_grid = [combo for combo in GRID
                      if args.labels is None or label_of(combo) in set(args.labels)]
        domain_result = results.get(domain, {
            "seed": args.seed, "epochs": args.epochs,
            "grid_size": len(sweep_grid), "grid": {}})
        if (domain_result.get("seed") != args.seed or
                domain_result.get("epochs") != args.epochs):
            raise RuntimeError(
                f"{domain}: partial sweep uses seed={domain_result.get('seed')} "
                f"epochs={domain_result.get('epochs')}; refusing to mix protocols")
        best = None  # (validation-only score, combination, label)
        for combo in sweep_grid:
            label = label_of(combo)
            if label in domain_result["grid"]:
                cell = domain_result["grid"][label]
                score = validation_score(combo, cell["validation"])
                candidate = (score, combo, label)
                if best is None or candidate[0] > best[0]:
                    best = candidate
                print(f"[VGATED] {domain} {label} already complete", flush=True)
                continue
            started = time.time()
            print(f"[VGATED] {domain} {label}", flush=True)
            model = train_one(data, tune_sessions, semantic, args.seed, args.epochs, combo)
            _, neural_valid = pasgr.predict_pasgr_array(
                model, data["valid_queries"], data["n_items"], args.candidate_width,
                exclude_seen=exclude_seen)
            beta, selection = select_beta(
                valid_memory["selected"], neural_valid, valid_targets,
                "rrf20", args.candidate_width)
            utility = selection["utility"]
            domain_result["grid"][label] = {
                "combination": {"graph_weight": combo[0],
                                "prototype_transport": combo[1],
                                "contrastive_weight": combo[2],
                                "inbatch_weight": combo[3]},
                "validation": selection,
                "seconds": time.time() - started,
            }
            print(f"[VGATED] DONE {domain} {label} valid_util={utility:.5f} "
                  f"({time.time()-started:.0f}s)", flush=True)
            score = validation_score(combo, selection)
            candidate = (score, combo, label)
            if best is None or candidate[0] > best[0]:
                best = candidate
            del model, neural_valid
            gc.collect()
            # Checkpoint after every cell to allow resume.
            results[domain] = domain_result
            args.output.write_text(json.dumps(results, indent=2))

        best_combo, best_label = best[1], best[2]
        # Test is scored once, only after validation has locked the cell.
        final_index = cearf.CEARFIndex(sessions, data["n_items"], config)
        test_memory = load_or_build_memory(
            args.artifact_dir / f"{domain.lower()}_nested_test_memory.npz",
            final_index, data["test_queries"], profiles, args.candidate_width,
            f"{domain}-test")
        test_keys = [str(x) for x in test_memory["keys"]]
        test_targets = targets_for(test_keys, data["test_queries"])
        final_model = train_one(
            data, sessions, semantic, args.seed, args.epochs, best_combo)
        _, neural_test = pasgr.predict_pasgr_array(
            final_model, data["test_queries"], data["n_items"],
            args.candidate_width, exclude_seen=exclude_seen)
        selected_beta = domain_result["grid"][best_label]["validation"]["beta"]
        fused_test = fuse_matrix(test_memory["selected"], neural_test,
                                 selected_beta, "rrf20", args.candidate_width)
        selected_test = metrics_from_ranks(ranks_at_20(fused_test, test_targets))
        domain_result["selected"] = {
            "label": best_label,
            "combination": {"graph_weight": best_combo[0],
                            "prototype_transport": best_combo[1],
                            "contrastive_weight": best_combo[2],
                            "inbatch_weight": best_combo[3]},
            "validation_utility": best[0][0],
            "test": selected_test,
        }
        # Baseline always-on for comparison (graph=.35, proto=True, contrast=.15, inbatch=0).
        baseline_label = label_of((0.35, True, 0.15, 0.0))
        if baseline_label in domain_result["grid"]:
            domain_result["baseline_always_on"] = {
                "label": baseline_label,
                "validation": domain_result["grid"][baseline_label]["validation"],
            }
        print(f"[VGATED] {domain} SELECTED {best_label} "
              f"valid_util={best[0][0]:.5f}", flush=True)
        results[domain] = domain_result
        args.output.write_text(json.dumps(results, indent=2))

    print(f"\n[VGATED] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
