#!/usr/bin/env python3
"""CEARF-N v2: integrates validation-gated PASGR + per-query β routing.

This runner is the canonical "final method" for the ADMA paper. Compared to
the v1 evidence run it:
  * trains PASGR with the per-domain validation-gated component configuration
    (loaded from `pasgr_config_per_domain.json`);
  * fuses memory + neural with the per-query β router (bucketed by default,
    with regime and continuous reported alongside for comparison);
  * runs 3 matched seeds and emits paired-bootstrap artifacts so the baseline
    paired analysis extends to v2.

Selection never touches test labels: the PASGR config and router are fit on
validation, then the test split is scored once.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import cearf
import loaders
import pasgr
from perquery_router import (
    BucketedRouter, ContinuousRouter, extract_features)
from run_cearfn import fuse, tune_beta
from run_cearfn_evidence import (
    metrics_from_ranks, popularity_partition,
    query_fingerprint, ranks_at_20, targets_for)
from run_pasgr_full import semantic_matrix
from validation_protocol import hold_out_validation_targets

HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Arts_Crafts_and_Sewing", "Diginetica_HID", "Tmall")
SEEDS = (42, 123, 456)
EPOCHS = 4
CANDIDATE_WIDTH = 120
# Datasets whose official protocol permits repeat consumption (target may
# already appear in the context). For these the CEARF memory index must NOT
# mask seen items, otherwise the comparison against baselines (which use the
# same repeat-aware protocol) silently caps recall.
REPEAT_PROTOCOL_DOMAINS = frozenset({"Diginetica_HID", "Tmall"})


def load_pasgr_config(domain: str, path: Path) -> dict:
    if not path.exists():
        print(f"[V2] WARNING: {path} missing; falling back to default PASGR config",
              flush=True)
        return {}
    data = json.loads(path.read_text())
    block = data.get(domain, {})
    sel = block.get("selected", {}).get("combination", {})
    if not sel:
        return {}
    return sel


def train_pasgr_v2(data, sessions, semantic, seed, epochs, gated_config):
    freq = Counter(x for sequence in sessions.values() for x in sequence)
    config_data = dict(
        dim=64, prototypes=min(96, max(8, data["n_items"] // 250)),
        epochs=epochs, batch_size=512, hard_negatives=32,
        top_k=120, seed=seed)
    config_data.update(gated_config)
    config = pasgr.PASGRConfig(**config_data)
    assets = pasgr.build_prototype_graph_embeddings(
        sessions, data["n_items"], freq, semantic, config)
    return pasgr.train_pasgr(sessions, data["n_items"], freq, semantic,
                             config, prepared_assets=assets)


def build_features(index, queries, freq, head_items):
    out = {}
    for uid, q in queries.items():
        ctx = q.get("context", ())
        comps = index.component_rankings(ctx)
        out[str(uid)] = extract_features(ctx, comps, freq, head_items)
    return out


def build_features_from_memory_arrays(
        queries, keys, memory_arrays, freq, head_items):
    """Reuse component ranks already produced by CEARF memory inference.

    ``build_memory_arrays`` stores the complete top-120 transition, session,
    and popularity lists consumed by ``extract_features``.  Reconstructing the
    same lists avoids a second index traversal for router features.
    """
    out = {}
    for row, uid in enumerate(keys):
        components = tuple(
            [int(item) for item in memory_arrays[name][row] if int(item) > 0]
            for name in ("transition", "session", "popularity")
        )
        context = queries[uid].get("context", ())
        out[str(uid)] = extract_features(
            context, components, freq, head_items)
    return out


def fuse_with_betas(memory, neural, keys, betas):
    output = np.empty((len(keys), 20), dtype=np.int32)
    for row, uid in enumerate(keys):
        output[row] = fuse(memory[row], neural[row], betas[uid])
    return output


def save_partial_json(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Journal-extension memory variants (DESIGN_CASM.md). The default variant
# "off" leaves the locked ADMA path above untouched; "raw-semantic" adds the
# unaligned teacher as the semantic_raw slot (control), "casm" adds the
# contrastively aligned memory as the casm slot.
# ---------------------------------------------------------------------------
V3_COMPONENT_NAMES = ("transition", "session", "popularity", "semantic_raw",
                      "repeat", "casm")


def build_memory_arrays_v3(index, queries: dict, profiles: dict, width: int,
                           label: str) -> dict[str, np.ndarray]:
    """Six-component analogue of run_cearfn_evidence.build_memory_arrays.

    CASM retrieval is batched (one chunked catalogue matmul for all queries)
    via CEARFIndexV3.component_rankings_batch; the fused "selected" column
    uses the 6-slot profile for the query's regime.
    """
    keys = sorted(queries)
    contexts = [queries[uid].get("context", ()) for uid in keys]
    arrays = {name: np.zeros((len(keys), width), dtype=np.int32)
              for name in V3_COMPONENT_NAMES + ("selected",)}
    batched = index.component_rankings_batch(contexts)
    for row, (uid, context, components) in enumerate(
            zip(keys, contexts, batched)):
        for name, ranking in zip(V3_COMPONENT_NAMES, components):
            take = min(width, len(ranking))
            arrays[name][row, :take] = ranking[:take]
        regime = ("short" if len(context) <= index.config.short_context
                  else "long")
        selected = index.fuse_rankings(context, components, profiles[regime],
                                       width)
        arrays["selected"][row, :len(selected)] = selected
        if (row + 1) % 10000 == 0:
            print(f"[V2] {label} memory={row + 1}/{len(keys)}", flush=True)
    arrays["keys"] = np.asarray(keys)
    return arrays


def load_or_build_memory_v3(path: Path, index, queries: dict, profiles: dict,
                            width: int, label: str) -> dict[str, np.ndarray]:
    """Fingerprinted cache identical in convention to load_or_build_memory."""
    fingerprint = query_fingerprint(queries)
    profile_json = json.dumps(profiles, sort_keys=True)
    if path.exists():
        with np.load(path) as saved:
            if (str(saved["fingerprint"].item()) == fingerprint and
                    str(saved["profiles"].item()) == profile_json):
                print(f"[V2] loading {path}", flush=True)
                return {key: saved[key] for key in saved.files
                        if key not in {"fingerprint", "profiles"}}
    arrays = build_memory_arrays_v3(index, queries, profiles, width, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, fingerprint=np.asarray(fingerprint),
                        profiles=np.asarray(profile_json))
    return arrays


def build_variant_indices(variant: str, domain: str, data: dict,
                          sessions: dict, tune_sessions: dict,
                          config, teacher: np.ndarray, artifact_dir: Path,
                          casm_kwargs: dict):
    """Return (tune_index, final_index) for an active variant.

    raw-semantic: identical batched retrieval path on the frozen teacher
    (semantic_raw slot) — no training, so the same memory serves both the
    tuning and the final index. casm: alignment head trained on
    tune_sessions for the validation-phase index and retrained on the full
    train_sessions for the single post-selection test index (nested
    protocol, DESIGN_CASM.md §2.3).
    """
    from casm import CASMemory, load_or_train_casm
    from cearf_v3_ext import CEARFIndexV3

    if teacher is None:
        raise ValueError(f"{domain}: no teacher matrix available "
                         f"for memory variant")
    if variant == "raw-semantic":
        # CASMemory.from_teacher exposes the same .ranking/.n_items interface
        # as SemanticMemory plus batched retrieval; identical lists, one
        # chunked matmul instead of one matvec per query.
        raw = CASMemory.from_teacher(teacher)
        tune_index = CEARFIndexV3(tune_sessions, data["n_items"], config,
                                  semantic=raw)
        final_index = CEARFIndexV3(sessions, data["n_items"], config,
                                   semantic=raw)
        return tune_index, final_index
    if variant == "casm":
        cache_dir = artifact_dir / "casm_cache"
        aligned_tune = load_or_train_casm(
            cache_dir, f"{domain.lower()}_tune", tune_sessions, teacher,
            **casm_kwargs)
        aligned_final = load_or_train_casm(
            cache_dir, f"{domain.lower()}_final", sessions, teacher,
            **casm_kwargs)
        tune_index = CEARFIndexV3(tune_sessions, data["n_items"], config,
                                  casm=CASMemory(aligned_tune))
        final_index = CEARFIndexV3(sessions, data["n_items"], config,
                                   casm=CASMemory(aligned_final))
        return tune_index, final_index
    raise ValueError(f"unknown memory variant {variant!r}")


def router_utility(ranking, targets):
    metrics = metrics_from_ranks(ranks_at_20(ranking, targets))
    return .5 * (metrics["recall@6"] + metrics["recall@20"]), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--candidate-width", type=int, default=CANDIDATE_WIDTH)
    parser.add_argument("--output", type=Path,
                        default=HERE / "cearfn_v2_results.json")
    parser.add_argument("--artifact-dir", type=Path,
                        default=HERE / "cearfn_v2_artifacts")
    parser.add_argument("--config-file", type=Path,
                        default=HERE / "pasgr_config_per_domain.json")
    parser.add_argument(
        "--semantic-dir", type=Path,
        help="Optional directory containing <domain>_minilm.npy teacher matrices.")
    parser.add_argument("--partial-dir", type=Path,
                        default=HERE / "cearfn_v2_partials")
    parser.add_argument("--force-router", choices=("regime", "bucketed", "continuous"),
                        default=None,
                        help="Optional exploratory override. If set, bypasses router-family "
                             "selection and reports the forced family as selected.")
    parser.add_argument("--memory-variant", choices=("off", "raw-semantic", "casm"),
                        default="off",
                        help="Journal-extension memory (DESIGN_CASM.md). 'off' (default) "
                             "reproduces the locked ADMA path exactly; 'raw-semantic' adds "
                             "the unaligned teacher control; 'casm' adds the aligned memory.")
    parser.add_argument("--casm-tau", type=float, default=0.07,
                        help="InfoNCE temperature (declared grid: 0.05/0.07/0.10).")
    parser.add_argument("--casm-epochs", type=int, default=5,
                        help="Alignment-head epochs (declared grid: 3/5/10).")
    parser.add_argument("--casm-pairs", type=int, default=2_000_000,
                        help="Sampled co-occurrence pairs per training run.")
    parser.add_argument("--casm-seed", type=int, default=42,
                        help="Alignment-head training seed.")
    parser.add_argument("--casm-mlp", action="store_true",
                        help="Use the 2-layer MLP ablation head instead of linear.")
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    for domain in args.domains:
        domain_started = time.time()
        data = loaders.ALL_LOADERS[domain]()
        if len(data["valid_queries"]) > 5000:
            _vk = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
            data["valid_queries"] = {k: data["valid_queries"][k] for k in _vk}
        sessions = data["train_sessions"]
        tune_sessions = hold_out_validation_targets(
            sessions, data["valid_queries"])
        freq = Counter(x for seq in sessions.values() for x in seq)
        exclude_seen = domain not in REPEAT_PROTOCOL_DOMAINS
        config = cearf.CEARFConfig(exclude_seen=exclude_seen)
        variant = args.memory_variant
        if variant == "off":
            # Locked ADMA path — must remain behaviour-identical.
            tune_index = cearf.CEARFIndex(tune_sessions, data["n_items"], config)
            final_index = cearf.CEARFIndex(sessions, data["n_items"], config)
            profiles, _ = cearf.tune_profiles(tune_index, data["valid_queries"])
            from run_cearfn_evidence import load_or_build_memory
            valid_memory = load_or_build_memory(
                args.artifact_dir / f"{domain.lower()}_nested_valid_memory.npz",
                tune_index, data["valid_queries"], profiles, args.candidate_width,
                f"{domain}-nested-valid")
            test_memory = load_or_build_memory(
                args.artifact_dir / f"{domain.lower()}_nested_test_memory.npz",
                final_index, data["test_queries"], profiles, args.candidate_width,
                f"{domain}-test")
        else:
            from cearf_v3_ext import tune_profiles_v3
            if args.semantic_dir:
                teacher_path = args.semantic_dir / f"{domain.lower()}_minilm.npy"
                if not teacher_path.exists():
                    raise FileNotFoundError(teacher_path)
                teacher = np.load(teacher_path).astype(np.float32)
            else:
                teacher = semantic_matrix(domain, data)
            casm_kwargs = dict(tau=args.casm_tau, epochs=args.casm_epochs,
                               n_pairs=args.casm_pairs, seed=args.casm_seed,
                               mlp=args.casm_mlp)
            tune_index, final_index = build_variant_indices(
                variant, domain, data, sessions, tune_sessions, config,
                teacher, args.artifact_dir, casm_kwargs)
            profiles, _ = tune_profiles_v3(tune_index, data["valid_queries"])
            tag = variant.replace("-", "_")
            valid_memory = load_or_build_memory_v3(
                args.artifact_dir /
                f"{domain.lower()}_{tag}_nested_valid_memory.npz",
                tune_index, data["valid_queries"], profiles,
                args.candidate_width, f"{domain}-{variant}-nested-valid")
            test_memory = load_or_build_memory_v3(
                args.artifact_dir /
                f"{domain.lower()}_{tag}_nested_test_memory.npz",
                final_index, data["test_queries"], profiles,
                args.candidate_width, f"{domain}-{variant}-test")

        valid_keys = [str(x) for x in valid_memory["keys"]]
        test_keys = [str(x) for x in test_memory["keys"]]
        head_items, _, _ = popularity_partition(freq, data["n_items"])
        head_set = set(head_items.tolist())
        test_targets = targets_for(test_keys, data["test_queries"])

        valid_features = build_features_from_memory_arrays(
            data["valid_queries"], valid_keys, valid_memory, freq, head_set)
        test_features = build_features_from_memory_arrays(
            data["test_queries"], test_keys, test_memory, freq, head_set)

        if args.semantic_dir:
            semantic_path = args.semantic_dir / f"{domain.lower()}_minilm.npy"
            if not semantic_path.exists():
                raise FileNotFoundError(semantic_path)
            semantic = np.load(semantic_path).astype(np.float32)
        else:
            semantic_path = None
            semantic = semantic_matrix(domain, data)
        if semantic is not None and semantic.shape[0] != data["n_items"]:
            raise ValueError(
                f"semantic rows ({semantic.shape[0]}) != n_items ({data['n_items']})")
        gated_config = load_pasgr_config(domain, args.config_file)
        print(f"[V2] {domain} gated PASGR config: {gated_config}", flush=True)
        domain_partial_dir = args.partial_dir / domain.lower()
        save_partial_json(domain_partial_dir / "domain_start.json", {
            "domain": domain,
            "status": "started",
            "n_train_sessions": len(sessions),
            "n_valid_queries": len(data["valid_queries"]),
            "n_test_queries": len(data["test_queries"]),
            "n_items": int(data["n_items"]),
            "gated_config": gated_config,
            "semantic_matrix": (str(semantic_path)
                                if semantic_path else "tfidf-svd-cache"),
        })

        # Active memory variants get their own results key so the locked
        # ADMA blocks (plain domain keys) are never touched.
        results_key = (domain if variant == "off"
                       else f"{domain}::{variant}")
        domain_block = results.get(results_key,
                                   {"runs": [], "pasgr_config": gated_config})
        completed = {int(r["seed"]) for r in domain_block["runs"]}
        for seed in args.seeds:
            if seed in completed:
                continue
            print(f"\n[V2] {domain} seed={seed}", flush=True)
            seed_started = time.time()
            seed_partial = domain_partial_dir / f"seed{seed}.json"
            save_partial_json(seed_partial, {
                "domain": domain,
                "seed": seed,
                "stage": "train_validation_model",
                "status": "running",
            })
            validation_model = train_pasgr_v2(
                data, tune_sessions, semantic, seed, args.epochs, gated_config)
            save_partial_json(seed_partial, {
                "domain": domain,
                "seed": seed,
                "stage": "predict_validation_neural",
                "status": "running",
            })
            _, neural_valid = pasgr.predict_pasgr_array(
                validation_model, data["valid_queries"], data["n_items"], args.candidate_width,
                exclude_seen=exclude_seen)
            del validation_model
            gc.collect()
            save_partial_json(seed_partial, {
                "domain": domain,
                "seed": seed,
                "stage": "train_test_model",
                "status": "running",
            })
            final_model = train_pasgr_v2(
                data, sessions, semantic, seed, args.epochs, gated_config)
            save_partial_json(seed_partial, {
                "domain": domain,
                "seed": seed,
                "stage": "predict_test_neural",
                "status": "running",
            })
            _, neural_test = pasgr.predict_pasgr_array(
                final_model, data["test_queries"], data["n_items"], args.candidate_width,
                exclude_seen=exclude_seen)

            valid_memory_dict = {uid: list(valid_memory["selected"][i])
                                 for i, uid in enumerate(valid_keys)}
            valid_neural_dict = {uid: list(neural_valid[i])
                                 for i, uid in enumerate(valid_keys)}

            # Nested validation: fit/calibrate router families on 80% and
            # select the family on a disjoint 20% before any test scoring.
            ordered = sorted(valid_keys, key=cearf._stable_fraction)
            split_at = max(1, min(len(ordered) - 1, int(.8 * len(ordered))))
            router_fit_keys, router_select_keys = ordered[:split_at], ordered[split_at:]
            fit_queries = {uid: data["valid_queries"][uid] for uid in router_fit_keys}
            fit_memory = {uid: valid_memory_dict[uid] for uid in router_fit_keys}
            fit_neural = {uid: valid_neural_dict[uid] for uid in router_fit_keys}
            fit_features = {uid: valid_features[uid] for uid in router_fit_keys}
            row_of = {uid: row for row, uid in enumerate(valid_keys)}
            select_memory = np.asarray([
                valid_memory["selected"][row_of[uid]] for uid in router_select_keys])
            select_neural = np.asarray([
                neural_valid[row_of[uid]] for uid in router_select_keys])
            select_queries = {uid: data["valid_queries"][uid]
                              for uid in router_select_keys}
            select_targets = targets_for(router_select_keys, select_queries)

            fit_betas, _ = tune_beta(
                fit_memory, fit_neural, fit_queries, config.short_context)
            regime_selection_betas = {
                uid: fit_betas["short" if valid_features[uid].length <=
                               config.short_context else "long"]
                for uid in router_select_keys}
            fit_bucketed = BucketedRouter(fallback_beta=fit_betas)
            fit_bucketed.fit(fit_queries, fit_memory, fit_neural,
                             router_fit_keys, fit_features)
            fit_continuous = ContinuousRouter(alpha=1.0)
            fit_continuous.fit(fit_queries, fit_memory, fit_neural,
                               router_fit_keys, fit_features)
            selection_rankings = {
                "regime": fuse_with_betas(
                    select_memory, select_neural, router_select_keys,
                    regime_selection_betas),
                "bucketed": fuse_with_betas(
                    select_memory, select_neural, router_select_keys,
                    {uid: fit_bucketed.beta_for(valid_features[uid])
                     for uid in router_select_keys}),
                "continuous": fuse_with_betas(
                    select_memory, select_neural, router_select_keys,
                    {uid: fit_continuous.beta_for(valid_features[uid])
                     for uid in router_select_keys}),
            }
            router_selection = {}
            preference = {"regime": 2, "bucketed": 1, "continuous": 0}
            best_router = None
            for router_name, ranking in selection_rankings.items():
                util, router_metrics = router_utility(ranking, select_targets)
                router_selection[router_name] = {"utility": util, **router_metrics}
                candidate = (util, router_metrics["recall@20"],
                             router_metrics["recall@6"], preference[router_name],
                             router_name)
                if best_router is None or candidate > best_router:
                    best_router = candidate
            selected_router = best_router[-1]
            if args.force_router is not None:
                selected_router = args.force_router
            save_partial_json(seed_partial, {
                "domain": domain,
                "seed": seed,
                "stage": "router_selection_complete",
                "status": "running",
                "selected_router": selected_router,
                "router_selection": router_selection,
            })

            # Regime baseline β for comparison.
            betas_regime, _ = tune_beta(valid_memory_dict, valid_neural_dict,
                                        data["valid_queries"], config.short_context)
            regime_per_uid = {
                uid: betas_regime["short" if f.length <= config.short_context else "long"]
                for uid, f in test_features.items()}
            fused_regime = fuse_with_betas(
                test_memory["selected"], neural_test, test_keys, regime_per_uid)

            # Refit all router families on full validation for reporting; only
            # the family selected above is the final method.
            bucketed = BucketedRouter(fallback_beta=betas_regime)
            bucketed.fit({uid: data["valid_queries"][uid] for uid in valid_keys},
                         valid_memory_dict, valid_neural_dict,
                         valid_keys, valid_features)
            bucketed_per_uid = {uid: bucketed.beta_for(test_features[uid])
                                for uid in test_keys}
            fused_bucketed = fuse_with_betas(
                test_memory["selected"], neural_test, test_keys, bucketed_per_uid)

            # Continuous router.
            continuous = ContinuousRouter(alpha=1.0)
            continuous.fit({uid: data["valid_queries"][uid] for uid in valid_keys},
                           valid_memory_dict, valid_neural_dict,
                           valid_keys, valid_features)
            continuous_per_uid = {uid: continuous.beta_for(test_features[uid])
                                  for uid in test_keys}
            fused_continuous = fuse_with_betas(
                test_memory["selected"], neural_test, test_keys, continuous_per_uid)
            selected_fused = {"regime": fused_regime,
                              "bucketed": fused_bucketed,
                              "continuous": fused_continuous}[selected_router]
            valid_regime_per_uid = {
                uid: betas_regime["short" if valid_features[uid].length <= config.short_context else "long"]
                for uid in valid_keys}
            fused_valid_regime = fuse_with_betas(
                valid_memory["selected"], neural_valid, valid_keys, valid_regime_per_uid)
            valid_bucketed_per_uid = {
                uid: bucketed.beta_for(valid_features[uid]) for uid in valid_keys}
            fused_valid_bucketed = fuse_with_betas(
                valid_memory["selected"], neural_valid, valid_keys, valid_bucketed_per_uid)
            valid_continuous_per_uid = {
                uid: continuous.beta_for(valid_features[uid]) for uid in valid_keys}
            fused_valid_continuous = fuse_with_betas(
                valid_memory["selected"], neural_valid, valid_keys, valid_continuous_per_uid)
            selected_valid_fused = {
                "regime": fused_valid_regime,
                "bucketed": fused_valid_bucketed,
                "continuous": fused_valid_continuous,
            }[selected_router]

            # Persist per-seed rank arrays for paired bootstrap. Active
            # memory variants get their own artifact name so the locked v2
            # rank files are never overwritten; the keys stay identical so
            # the guarded selector consumes both uniformly.
            variant_tag = ("" if variant == "off"
                           else f"_{variant.replace('-', '_')}")
            seed_artifact = (args.artifact_dir /
                             f"{domain.lower()}_v2{variant_tag}_seed{seed}_ranks.npz")
            np.savez_compressed(
                seed_artifact,
                valid_keys=np.asarray(valid_keys, dtype=str),
                test_keys=np.asarray(test_keys, dtype=str),
                valid_memory_top20=valid_memory["selected"][:, :20].astype(np.int32),
                valid_neural_top20=neural_valid[:, :20].astype(np.int32),
                valid_regime_top20=fused_valid_regime[:, :20].astype(np.int32),
                valid_bucketed_top20=fused_valid_bucketed[:, :20].astype(np.int32),
                valid_continuous_top20=fused_valid_continuous[:, :20].astype(np.int32),
                valid_selected_top20=selected_valid_fused[:, :20].astype(np.int32),
                memory_top20=test_memory["selected"][:, :20].astype(np.int32),
                neural_top20=neural_test[:, :20].astype(np.int32),
                regime_top20=fused_regime[:, :20].astype(np.int32),
                bucketed_top20=fused_bucketed[:, :20].astype(np.int32),
                continuous_top20=fused_continuous[:, :20].astype(np.int32),
                selected_top20=selected_fused[:, :20].astype(np.int32),
                valid_memory_rank=ranks_at_20(
                    valid_memory["selected"], targets_for(valid_keys, data["valid_queries"])
                ).astype(np.uint8),
                valid_neural_rank=ranks_at_20(
                    neural_valid, targets_for(valid_keys, data["valid_queries"])
                ).astype(np.uint8),
                valid_regime_rank=ranks_at_20(
                    fused_valid_regime, targets_for(valid_keys, data["valid_queries"])
                ).astype(np.uint8),
                valid_bucketed_rank=ranks_at_20(
                    fused_valid_bucketed, targets_for(valid_keys, data["valid_queries"])
                ).astype(np.uint8),
                valid_continuous_rank=ranks_at_20(
                    fused_valid_continuous, targets_for(valid_keys, data["valid_queries"])
                ).astype(np.uint8),
                valid_selected_rank=ranks_at_20(
                    selected_valid_fused, targets_for(valid_keys, data["valid_queries"])
                ).astype(np.uint8),
                memory_rank=ranks_at_20(
                    test_memory["selected"], test_targets).astype(np.uint8),
                neural_rank=ranks_at_20(
                    neural_test, test_targets).astype(np.uint8),
                regime_rank=ranks_at_20(fused_regime, test_targets).astype(np.uint8),
                bucketed_rank=ranks_at_20(fused_bucketed, test_targets).astype(np.uint8),
                continuous_rank=ranks_at_20(fused_continuous, test_targets).astype(np.uint8),
                selected_rank=ranks_at_20(selected_fused, test_targets).astype(np.uint8),
                selected_router=np.asarray(selected_router),
                test_fingerprint=np.asarray(query_fingerprint(data["test_queries"])))

            domain_block["runs"].append({
                "seed": seed,
                # Key added only for journal-extension variants so the
                # locked-path results JSON stays byte-compatible.
                **({"memory_variant": variant} if variant != "off" else {}),
                "memory_only": metrics_from_ranks(
                    ranks_at_20(test_memory["selected"], test_targets)),
                "neural_only": metrics_from_ranks(
                    ranks_at_20(neural_test, test_targets)),
                "regime": metrics_from_ranks(ranks_at_20(fused_regime, test_targets)),
                "bucketed": metrics_from_ranks(ranks_at_20(fused_bucketed, test_targets)),
                "continuous": metrics_from_ranks(ranks_at_20(fused_continuous, test_targets)),
                "selected_router": selected_router,
                "router_override": args.force_router,
                "router_selection": {"fit_queries": len(router_fit_keys),
                                     "selection_queries": len(router_select_keys),
                                     "variants": router_selection},
                "selected": metrics_from_ranks(
                    ranks_at_20(selected_fused, test_targets)),
                "rank_artifact": str(seed_artifact),
                "seconds": time.time() - seed_started,
            })
            r = domain_block["runs"][-1]
            save_partial_json(seed_partial, {
                "domain": domain,
                "seed": seed,
                "stage": "complete",
                "status": "complete",
                "selected_router": r["selected_router"],
                "selected_metrics": r["selected"],
                "memory_only_metrics": r["memory_only"],
                "neural_only_metrics": r["neural_only"],
                "seconds": r["seconds"],
            })
            results[results_key] = domain_block
            args.output.write_text(json.dumps(results, indent=2))
            print(f"[V2] DONE {domain} seed={seed} regime R@20="
                  f"{r['regime']['recall@20']:.5f} bucketed="
                  f"{r['bucketed']['recall@20']:.5f} continuous="
                  f"{r['continuous']['recall@20']:.5f}", flush=True)
            del final_model, neural_valid, neural_test
            gc.collect()

        domain_block["seconds_total"] = time.time() - domain_started
        results[results_key] = domain_block
        args.output.write_text(json.dumps(results, indent=2))

    print(f"\n[V2] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
