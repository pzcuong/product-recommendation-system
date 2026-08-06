#!/usr/bin/env python3
"""Paper-grade evidence suite for CEARF-N on the two positive Amazon domains.

The suite performs validation-only profile/beta selection, five independent
neural training seeds, component ablations, paired inference statistics, and
popularity head/tail analysis. Test labels are read only after every selection
decision for a seed has been made. Compact rank vectors and query-order caches
are persisted so every reported statistic remains independently auditable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import binomtest, t as student_t

import cearf
import loaders
import pasgr
from run_cearfn import BETAS, fuse, train_neural
from run_pasgr_full import semantic_matrix


HERE = Path(__file__).resolve().parent
DEFAULT_DOMAINS = ("Video_Games", "Baby_Products")
DEFAULT_SEEDS = (42, 43, 44, 123, 456)


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


def targets_for(keys: list[str], queries: dict) -> np.ndarray:
    targets = []
    for uid in keys:
        values = [int(x) for x in queries[uid].get("targets", ())]
        if len(values) != 1:
            raise ValueError(f"CEARF-N evidence expects one target, got {uid}={values}")
        targets.append(values[0])
    return np.asarray(targets, dtype=np.int32)


def ranks_at_20(rankings: np.ndarray, targets: np.ndarray) -> np.ndarray:
    top = rankings[:, :20]
    matches = top == targets[:, None]
    found = matches.any(axis=1)
    ranks = np.zeros(len(targets), dtype=np.uint8)
    ranks[found] = matches[found].argmax(axis=1).astype(np.uint8) + 1
    return ranks


def metrics_from_ranks(ranks: np.ndarray, mask: np.ndarray | None = None) -> dict:
    values = ranks if mask is None else ranks[mask]
    output: dict[str, float | int] = {"n": int(len(values))}
    for k in (6, 10, 20):
        hit = (values > 0) & (values <= k)
        gain = np.zeros(len(values), dtype=np.float64)
        gain[hit] = 1.0 / np.log2(values[hit].astype(np.float64) + 1.0)
        output[f"recall@{k}"] = float(hit.mean()) if len(values) else 0.0
        output[f"ndcg@{k}"] = float(gain.mean()) if len(values) else 0.0
    return output


def prediction_diagnostics(rankings: np.ndarray, tail_items: np.ndarray,
                           n_items: int) -> dict:
    top = rankings[:, :20]
    return {
        "catalog_coverage@20": float(len(np.unique(top)) / max(n_items - 1, 1)),
        "tail_exposure@20": float(np.isin(top, tail_items).mean()),
    }


def paired_recall_test(challenger: np.ndarray, baseline: np.ndarray,
                       k: int = 20, reps: int = 20000,
                       seed: int = 20260719) -> dict:
    hit_a = (challenger > 0) & (challenger <= k)
    hit_b = (baseline > 0) & (baseline <= k)
    positive = int(np.sum(hit_a & ~hit_b))
    negative = int(np.sum(~hit_a & hit_b))
    unchanged = int(len(hit_a) - positive - negative)
    rng = np.random.default_rng(seed)
    probabilities = np.asarray([positive, negative, unchanged], dtype=float) / len(hit_a)
    draws = rng.multinomial(len(hit_a), probabilities, size=reps)
    delta = (draws[:, 0] - draws[:, 1]) / len(hit_a)
    discordant = positive + negative
    pvalue = (float(binomtest(positive, discordant, 0.5).pvalue)
              if discordant else 1.0)
    return {
        "k": k,
        "challenger": float(hit_a.mean()),
        "baseline": float(hit_b.mean()),
        "difference": float(hit_a.mean() - hit_b.mean()),
        "paired_bootstrap_ci95": [float(x) for x in np.quantile(delta, [.025, .975])],
        "challenger_only": positive,
        "baseline_only": negative,
        "mcnemar_exact_p": pvalue,
        "bootstrap_repetitions": reps,
        "n": int(len(hit_a)),
    }


def aggregate_runs(runs: list[dict], method: str) -> dict:
    output: dict = {}
    for metric in ("recall@6", "ndcg@6", "recall@10", "ndcg@10",
                   "recall@20", "ndcg@20"):
        values = np.asarray([run[method][metric] for run in runs], dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output[metric] = {
            "mean": float(values.mean()),
            "std": std,
            "seed_ci95_half_width": float(
                student_t.ppf(.975, len(values) - 1) * std / math.sqrt(len(values)))
            if len(values) > 1 else 0.0,
            "seed_ci_method": "two-sided Student-t interval over independent seeds",
        }
    return output


def popularity_partition(freq: Counter, n_items: int) -> tuple[np.ndarray, np.ndarray, int]:
    ordered = sorted(range(1, n_items), key=lambda item: (-freq.get(item, 0), item))
    total = sum(freq.values())
    cumulative = 0
    split = 0
    for split, item in enumerate(ordered, 1):
        cumulative += freq.get(item, 0)
        if cumulative >= 0.80 * total:
            break
    head = np.asarray(ordered[:split], dtype=np.int32)
    tail = np.asarray(ordered[split:], dtype=np.int32)
    return head, tail, split


def build_memory_arrays(index: cearf.CEARFIndex, queries: dict,
                        profiles: dict, width: int, label: str) -> dict[str, np.ndarray]:
    keys = sorted(queries)
    arrays = {name: np.zeros((len(keys), width), dtype=np.int32)
              for name in ("transition", "session", "popularity", "selected")}
    for row, uid in enumerate(keys):
        context = queries[uid].get("context", ())
        components = index.component_rankings(context)
        for name, ranking in zip(("transition", "session", "popularity"), components):
            take = min(width, len(ranking))
            arrays[name][row, :take] = ranking[:take]
        regime = "short" if len(context) <= index.config.short_context else "long"
        selected = index.fuse_rankings(context, components, profiles[regime], width)
        arrays["selected"][row, :len(selected)] = selected
        if (row + 1) % 10000 == 0:
            print(f"[EVIDENCE] {label} memory={row + 1}/{len(keys)}", flush=True)
    arrays["keys"] = np.asarray(keys)
    return arrays


def load_or_build_memory(path: Path, index: cearf.CEARFIndex, queries: dict,
                         profiles: dict, width: int, label: str) -> dict[str, np.ndarray]:
    fingerprint = query_fingerprint(queries)
    profile_json = json.dumps(profiles, sort_keys=True)
    if path.exists():
        with np.load(path) as saved:
            if (str(saved["fingerprint"].item()) == fingerprint and
                    str(saved["profiles"].item()) == profile_json):
                print(f"[EVIDENCE] loading {path}", flush=True)
                return {key: saved[key] for key in saved.files
                        if key not in {"fingerprint", "profiles"}}
    arrays = build_memory_arrays(index, queries, profiles, width, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, fingerprint=np.asarray(fingerprint),
                        profiles=np.asarray(profile_json))
    return arrays


def tune_beta_arrays(memory: np.ndarray, neural: np.ndarray, keys: list[str],
                     queries: dict, short_context: int = 2) -> tuple[dict, dict]:
    selected: dict[str, float] = {}
    report: dict = {}
    targets = targets_for(keys, queries)
    lengths = np.asarray([len(queries[uid].get("context", ())) for uid in keys])
    for regime in ("short", "long"):
        mask = lengths <= short_context if regime == "short" else lengths > short_context
        rows = np.flatnonzero(mask)
        if not len(rows):
            selected[regime] = 0.0
            report[regime] = {"beta": 0.0, "n": 0}
            continue
        best = None
        for beta in BETAS:
            fused = np.asarray([
                fuse(memory[row], neural[row], beta, topk=20)
                for row in rows
            ], dtype=np.int32)
            ranks = ranks_at_20(fused, targets[rows])
            r6 = float(np.mean((ranks > 0) & (ranks <= 6)))
            r20 = float(np.mean(ranks > 0))
            candidate = (0.5 * r6 + 0.5 * r20, r20, r6, -beta, beta)
            if best is None or candidate > best:
                best = candidate
        selected[regime] = best[-1]
        report[regime] = {"beta": best[-1], "n": int(len(rows)),
                          "score": best[0], "recall@6": best[2],
                          "recall@20": best[1]}
    return selected, report


def fuse_test_arrays(memory: np.ndarray, neural: np.ndarray, keys: list[str],
                     queries: dict, betas: dict, short_context: int = 2) -> np.ndarray:
    output = np.empty((len(keys), 20), dtype=np.int32)
    for row, uid in enumerate(keys):
        regime = "short" if len(queries[uid].get("context", ())) <= short_context else "long"
        output[row] = fuse(memory[row], neural[row], betas[regime], topk=20)
    return output


def protocol_report(data: dict) -> dict:
    train_targets = Counter(x for seq in data["train_sessions"].values() for x in seq)
    valid = data.get("valid_queries", {})
    test = data["test_queries"]
    def violations(queries: dict) -> int:
        return sum(bool(set(q.get("context", ())) & set(q.get("targets", ())))
                   for q in queries.values())
    return {
        "train_sessions": len(data["train_sessions"]),
        "validation_queries": len(valid),
        "test_queries": len(test),
        "n_items_including_padding": data["n_items"],
        "validation_fingerprint_sha256": query_fingerprint(valid),
        "test_fingerprint_sha256": query_fingerprint(test),
        "target_in_context_violations": {
            "validation": violations(valid), "test": violations(test)},
        "test_target_unseen_in_train_rate": float(np.mean([
            train_targets.get(int(q["targets"][0]), 0) == 0 for q in test.values()])),
        "selection_uses_test_labels": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--candidate-width", type=int, default=120)
    parser.add_argument("--validation-cap", type=int, default=5000)
    parser.add_argument("--test-cap", type=int, default=0,
                        help="Deterministic query cap for smoke tests only")
    parser.add_argument("--output", type=Path,
                        default=HERE / "cearfn_evidence_results.json")
    parser.add_argument("--artifact-dir", type=Path,
                        default=HERE / "cearfn_evidence_artifacts")
    args = parser.parse_args()
    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    for domain in args.domains:
        started = time.time()
        data = loaders.ALL_LOADERS[domain]()
        if not data.get("valid_queries"):
            raise ValueError(f"{domain} requires an explicit validation split")
        if len(data["valid_queries"]) > args.validation_cap:
            valid_keys = sorted(data["valid_queries"], key=cearf._stable_fraction)[
                :args.validation_cap]
            data["valid_queries"] = {
                key: data["valid_queries"][key] for key in valid_keys}
        if args.test_cap:
            test_keys = sorted(data["test_queries"], key=cearf._stable_fraction)[:args.test_cap]
            data["test_queries"] = {key: data["test_queries"][key] for key in test_keys}
        sessions = data["train_sessions"]
        freq = Counter(x for seq in sessions.values() for x in seq)
        config = cearf.CEARFConfig(validation_cap=args.validation_cap)
        index = cearf.CEARFIndex(sessions, data["n_items"], config)
        profiles, profile_report = cearf.tune_profiles(index, data["valid_queries"])
        profile_names = {regime: profile_report[regime]["profile"] for regime in profiles}
        cache_tag = f"{domain.lower()}_{'smoke' + str(args.test_cap) if args.test_cap else 'full'}"
        valid_memory = load_or_build_memory(
            args.artifact_dir / f"{cache_tag}_valid_memory.npz", index,
            data["valid_queries"], profiles, args.candidate_width, f"{domain}-valid")
        test_memory = load_or_build_memory(
            args.artifact_dir / f"{cache_tag}_test_memory.npz", index,
            data["test_queries"], profiles, args.candidate_width, f"{domain}-test")
        valid_keys = [str(x) for x in valid_memory["keys"]]
        test_keys = [str(x) for x in test_memory["keys"]]
        valid_targets = targets_for(valid_keys, data["valid_queries"])
        test_targets = targets_for(test_keys, data["test_queries"])
        head_items, tail_items, head_count = popularity_partition(
            freq, data["n_items"])
        tail_mask = np.isin(test_targets, tail_items)
        head_mask = ~tail_mask

        memory_ranks = {
            name: ranks_at_20(test_memory[name], test_targets)
            for name in ("transition", "session", "popularity", "selected")
        }
        domain_result = results.get(domain, {})
        domain_result.update({
            "dataset": domain,
            "protocol": protocol_report(data),
            "profile_selection": profile_report,
            "selected_profile_names": profile_names,
            "popularity_partition": {
                "definition": "head is the smallest most-popular item set covering 80% of training interactions",
                "head_catalog_items": int(head_count),
                "tail_catalog_items": int(len(tail_items)),
                "head_test_queries": int(head_mask.sum()),
                "tail_test_queries": int(tail_mask.sum()),
            },
            "memory_ablations": {
                name: {
                    "overall": metrics_from_ranks(ranks),
                    "head_targets": metrics_from_ranks(ranks, head_mask),
                    "tail_targets": metrics_from_ranks(ranks, tail_mask),
                    "diagnostics": prediction_diagnostics(
                        test_memory[name], tail_items, data["n_items"]),
                }
                for name, ranks in memory_ranks.items()
            },
            "runs": domain_result.get("runs", []),
        })
        completed = {int(run["seed"]) for run in domain_result["runs"]}
        args.output.write_text(json.dumps({**results, domain: domain_result}, indent=2))
        semantic = semantic_matrix(domain, data)

        for seed in args.seeds:
            if seed in completed:
                print(f"[EVIDENCE] {domain} seed={seed} already complete", flush=True)
                continue
            seed_started = time.time()
            print(f"[EVIDENCE] {domain} seed={seed} train", flush=True)
            model = train_neural(domain, sessions, data, semantic, args.epochs, seed=seed)
            neural_valid_keys, neural_valid = pasgr.predict_pasgr_array(
                model, data["valid_queries"], data["n_items"], args.candidate_width)
            if neural_valid_keys != valid_keys:
                raise RuntimeError("validation prediction order mismatch")
            betas, beta_report = tune_beta_arrays(
                valid_memory["selected"], neural_valid, valid_keys,
                data["valid_queries"], config.short_context)
            print(f"[EVIDENCE] {domain} seed={seed} betas={beta_report}", flush=True)
            # Selection is complete here; only now produce official test ranks.
            neural_test_keys, neural_test = pasgr.predict_pasgr_array(
                model, data["test_queries"], data["n_items"], args.candidate_width)
            if neural_test_keys != test_keys:
                raise RuntimeError("test prediction order mismatch")
            fused = fuse_test_arrays(test_memory["selected"], neural_test, test_keys,
                                     data["test_queries"], betas, config.short_context)
            neural_ranks = ranks_at_20(neural_test, test_targets)
            fused_ranks = ranks_at_20(fused, test_targets)
            seed_artifact = args.artifact_dir / f"{cache_tag}_seed{seed}_ranks.npz"
            np.savez_compressed(
                seed_artifact, cearfn_rank=fused_ranks, pasgr_rank=neural_ranks,
                cearf_rank=memory_ranks["selected"], target_frequency=np.asarray(
                    [freq.get(int(x), 0) for x in test_targets], dtype=np.int32),
                test_fingerprint=np.asarray(query_fingerprint(data["test_queries"])))
            run = {
                "seed": seed,
                "beta_selection": beta_report,
                "CEARF-N": metrics_from_ranks(fused_ranks),
                "PASGR": metrics_from_ranks(neural_ranks),
                "CEARF": metrics_from_ranks(memory_ranks["selected"]),
                "groups": {
                    "CEARF-N": {
                        "head_targets": metrics_from_ranks(fused_ranks, head_mask),
                        "tail_targets": metrics_from_ranks(fused_ranks, tail_mask),
                    },
                    "PASGR": {
                        "head_targets": metrics_from_ranks(neural_ranks, head_mask),
                        "tail_targets": metrics_from_ranks(neural_ranks, tail_mask),
                    },
                },
                "diagnostics": {
                    "CEARF-N": prediction_diagnostics(fused, tail_items, data["n_items"]),
                    "PASGR": prediction_diagnostics(neural_test, tail_items, data["n_items"]),
                },
                "paired_tests": {
                    "vs_CEARF_R20": paired_recall_test(
                        fused_ranks, memory_ranks["selected"], 20, seed=seed),
                    "vs_PASGR_R20": paired_recall_test(
                        fused_ranks, neural_ranks, 20, seed=seed),
                },
                "rank_artifact": str(seed_artifact),
                "seconds": time.time() - seed_started,
            }
            domain_result["runs"].append(run)
            domain_result["aggregate"] = {
                "CEARF-N": aggregate_runs(domain_result["runs"], "CEARF-N"),
                "PASGR": aggregate_runs(domain_result["runs"], "PASGR"),
                "CEARF": aggregate_runs(domain_result["runs"], "CEARF"),
            }
            domain_result["seconds_latest_invocation"] = time.time() - started
            results[domain] = domain_result
            args.output.write_text(json.dumps(results, indent=2))
            print(f"[EVIDENCE] {domain} seed={seed} done "
                  f"R20={run['CEARF-N']['recall@20']:.6f} "
                  f"seconds={run['seconds']:.1f}", flush=True)
            del model, neural_valid, neural_test, fused
            gc.collect()
        if domain_result["runs"]:
            domain_result["aggregate"] = {
                "CEARF-N": aggregate_runs(domain_result["runs"], "CEARF-N"),
                "PASGR": aggregate_runs(domain_result["runs"], "PASGR"),
                "CEARF": aggregate_runs(domain_result["runs"], "CEARF"),
            }
        results[domain] = domain_result
        args.output.write_text(json.dumps(results, indent=2))
    print(f"[EVIDENCE] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
