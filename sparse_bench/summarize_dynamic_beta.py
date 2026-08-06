#!/usr/bin/env python3
"""Aggregate CEARF-N dynamic-beta runs and emit auditable paper tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DOMAINS = ("Video_Games", "Baby_Products", "Diginetica_HID")
METHODS = (
    "memory_only",
    "neural_only",
    "fixed_05",
    "oof_global",
    "oof_short_long",
    "dynamic",
    "dynamic_context_mlp",
    "dynamic_full_linear",
    "dynamic_full_mlp",
    "dynamic_without_cross_expert",
    "dynamic_without_memory_certainty",
)
CAPACITY_METHODS = (
    "dynamic",
    "dynamic_context_mlp",
    "dynamic_full_linear",
    "dynamic_full_mlp",
    "dynamic_without_cross_expert",
    "dynamic_without_memory_certainty",
)
RRF_SENSITIVITY = ("rrf_k10", "rrf_k20", "rrf_k60")
METRICS = (
    "recall@6",
    "ndcg@6",
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "utility",
)


def _mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": [float(value) for value in array],
    }


def validate_raw_results(
        raw: dict,
        allowed_seeds: set[int] | None,
) -> None:
    missing = [domain for domain in DOMAINS if domain not in raw]
    if missing:
        raise ValueError(f"missing completed domains: {missing}")
    protocols = {str(raw[domain].get("protocol")) for domain in DOMAINS}
    if len(protocols) != 1:
        raise ValueError(f"mixed domain protocols: {sorted(protocols)}")
    seed_sets = {}
    for domain in DOMAINS:
        runs = raw[domain].get("runs", [])
        seeds = [int(run["seed"]) for run in runs
                 if allowed_seeds is None
                 or int(run["seed"]) in allowed_seeds]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{domain}: duplicate completed seed")
        seed_sets[domain] = tuple(sorted(seeds))
        for run in runs:
            if allowed_seeds is not None and int(run["seed"]) not in allowed_seeds:
                continue
            manifest_path = Path(run["manifest"])
            rank_path = Path(run["rank_artifact"])
            if not manifest_path.exists() or not rank_path.exists():
                raise ValueError(
                    f"{domain} seed {run['seed']}: missing frozen artifact")
            manifest = json.loads(manifest_path.read_text())
            if (
                    str(manifest.get("protocol")) not in protocols
                    or str(manifest.get("domain")) != domain
                    or int(manifest.get("seed", -1)) != int(run["seed"])):
                raise ValueError(
                    f"{domain} seed {run['seed']}: manifest identity mismatch")
    if len(set(seed_sets.values())) != 1:
        raise ValueError(f"domains use different seed sets: {seed_sets}")
    if not next(iter(seed_sets.values())):
        raise ValueError("no completed seeds selected")


def _discrete_bootstrap(
        differences: np.ndarray,
        repetitions: int,
        seed: int,
) -> dict:
    """Paired query bootstrap and sign-flip via grouped discrete differences."""
    rounded = np.round(np.asarray(differences, dtype=np.float64), 12)
    values, counts = np.unique(rounded, return_counts=True)
    probabilities = counts / counts.sum()
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(
        len(rounded), probabilities, size=repetitions)
    bootstrap = draws @ values / len(rounded)

    absolute, absolute_counts = np.unique(
        np.abs(rounded[rounded != 0.0]), return_counts=True)
    if len(absolute):
        positive = np.column_stack([
            rng.binomial(int(count), .5, size=repetitions)
            for count in absolute_counts
        ])
        permuted = (
            ((2 * positive - absolute_counts[None, :])
             * absolute[None, :]).sum(axis=1)
            / len(rounded)
        )
        observed = abs(float(rounded.mean()))
        pvalue = float(
            (np.sum(np.abs(permuted) >= observed) + 1)
            / (repetitions + 1)
        )
    else:
        pvalue = 1.0
    return {
        "difference": float(rounded.mean()),
        "cluster_bootstrap_ci95": [
            float(value)
            for value in np.quantile(bootstrap, [.025, .975])
        ],
        "cluster_sign_flip_p": pvalue,
        "clusters": int(len(rounded)),
        "repetitions": repetitions,
        "support": {
            str(float(value)): int(count)
            for value, count in zip(values, counts)
        },
    }


def _per_query_value(ranks: np.ndarray, metric: str) -> np.ndarray:
    if metric.startswith("recall@"):
        cutoff = int(metric.split("@")[1])
        return ((ranks > 0) & (ranks <= cutoff)).astype(np.float64)
    if metric == "utility":
        return (
            .5 * ((ranks > 0) & (ranks <= 6))
            + .5 * ((ranks > 0) & (ranks <= 20))
        ).astype(np.float64)
    if metric.startswith("ndcg@"):
        cutoff = int(metric.split("@")[1])
        output = np.zeros(len(ranks), dtype=np.float64)
        hit = (ranks > 0) & (ranks <= cutoff)
        output[hit] = 1.0 / np.log2(ranks[hit].astype(np.float64) + 1.0)
        return output
    raise ValueError(metric)


def paired_across_seeds(
        runs: list[dict],
        challenger: str,
        baseline: str,
        metric: str,
        repetitions: int,
        seed: int,
) -> dict:
    differences = []
    reference_keys = None
    for run in runs:
        with np.load(run["rank_artifact"]) as saved:
            keys = np.asarray(saved["test_keys"], dtype=str)
            if reference_keys is None:
                reference_keys = keys
            elif not np.array_equal(reference_keys, keys):
                raise ValueError("test query order differs across seeds")
            challenger_rank = saved[f"test_{challenger}_rank"]
            baseline_rank = saved[f"test_{baseline}_rank"]
            differences.append(
                _per_query_value(challenger_rank, metric)
                - _per_query_value(baseline_rank, metric)
            )
    query_seed_mean = np.mean(np.stack(differences), axis=0)
    output = _discrete_bootstrap(
        query_seed_mean, repetitions, seed)
    output["metric"] = metric
    output["challenger"] = challenger
    output["baseline"] = baseline
    output["seeds"] = [int(run["seed"]) for run in runs]
    output["aggregation"] = (
        "per-query difference averaged across seeds, then query identifiers "
        "resampled with replacement"
    )
    return output


def summarize_domain(
        block: dict,
        allowed_seeds: set[int] | None,
        repetitions: int,
) -> dict:
    runs = sorted(
        (
            run for run in block["runs"]
            if allowed_seeds is None or int(run["seed"]) in allowed_seeds
        ),
        key=lambda run: int(run["seed"]),
    )
    if not runs:
        raise ValueError(f"{block['domain']}: no matching completed runs")
    aggregate = {}
    for method in METHODS:
        if method not in runs[0]["test"]["metrics"]:
            continue
        aggregate[method] = {
            metric: _mean_std([
                float(run["test"]["metrics"][method][metric])
                for run in runs
            ])
            for metric in METRICS
        }
        aggregate[method]["net_rescues_vs_memory"] = _mean_std([
            float(run["test"]["metrics"][method]["net_rescues_vs_memory"])
            for run in runs
        ])
        aggregate[method]["rescues_vs_memory"] = _mean_std([
            float(run["test"]["metrics"][method]["rescues_vs_memory"])
            for run in runs
        ])
        aggregate[method]["damage_vs_memory"] = _mean_std([
            float(run["test"]["metrics"][method]["damage_vs_memory"])
            for run in runs
        ])
    paired = {}
    for baseline in ("oof_global", "oof_short_long", "fixed_05",
                     "memory_only", "neural_only"):
        paired[baseline] = {
            metric: paired_across_seeds(
                runs,
                "dynamic",
                baseline,
                metric,
                repetitions,
                seed=20260729 + index,
            )
            for index, metric in enumerate((
                "recall@6",
                "ndcg@6",
                "recall@10",
                "ndcg@10",
                "recall@20",
                "ndcg@20",
                "utility",
            ))
        }
    beta = {
        key: _mean_std([
            float(run["test"]["metrics"]["dynamic"]["beta"][key])
            for run in runs
        ])
        for key in ("mean", "std", "q10", "median", "q90")
    }
    calibration = {
        "rows": _mean_std([
            float(run["training"]["global"]["n_calibration_queries"])
            for run in runs
        ]),
        "actionable": _mean_std([
            float(run["training"]["global"]["n_actionable_queries"])
            for run in runs
        ]),
        "actionable_share": _mean_std([
            float(run["training"]["global"]["actionable_share"])
            for run in runs
        ]),
        "global_beta": _mean_std([
            float(run["training"]["global"]["beta"])
            for run in runs
        ]),
    }
    fusion_sensitivity = {}
    for setting in RRF_SENSITIVITY:
        if all(
                setting in run["test"].get("fusion_sensitivity", {})
                for run in runs):
            fusion_sensitivity[setting] = {
                metric: _mean_std([
                    float(
                        run["test"]["fusion_sensitivity"][setting][metric])
                    for run in runs
                ])
                for metric in METRICS
            }
    dynamic_minus_global_net_rescues = _mean_std([
        float(
            run["test"]["metrics"]["dynamic"]["net_rescues_vs_memory"]
            - run["test"]["metrics"]["oof_global"]["net_rescues_vs_memory"]
        )
        for run in runs
    ])
    decile_runs = [
        run["test"]["metrics"]["dynamic"]["beta_deciles"]
        for run in runs
    ]
    beta_deciles = []
    for index in range(10):
        beta_deciles.append({
            "decile": index + 1,
            "n": _mean_std([
                float(block[index]["n"]) for block in decile_runs
            ]),
            "beta_mean": _mean_std([
                float(block[index]["beta_mean"]) for block in decile_runs
            ]),
            "realized_neural_minus_memory_ndcg20": _mean_std([
                float(
                    block[index]["realized_neural_minus_memory_ndcg20"])
                for block in decile_runs
            ]),
        })

    def weighted_decile_advantage(indices: range) -> list[float]:
        values = []
        for block in decile_runs:
            counts = np.asarray(
                [block[index]["n"] for index in indices],
                dtype=np.float64,
            )
            advantages = np.asarray([
                block[index]["realized_neural_minus_memory_ndcg20"]
                for index in indices
            ], dtype=np.float64)
            values.append(float(np.average(advantages, weights=counts)))
        return values

    low_advantage = weighted_decile_advantage(range(3))
    high_advantage = weighted_decile_advantage(range(7, 10))
    allocation_discrimination = {
        "low_beta_deciles_1_3": _mean_std(low_advantage),
        "high_beta_deciles_8_10": _mean_std(high_advantage),
        "high_minus_low": _mean_std([
            high - low
            for high, low in zip(high_advantage, low_advantage)
        ]),
    }
    return {
        "seeds": [int(run["seed"]) for run in runs],
        "n_seeds": len(runs),
        "aggregate": aggregate,
        "paired": paired,
        "dynamic_beta": beta,
        "calibration": calibration,
        "fusion_sensitivity": fusion_sensitivity,
        "dynamic_minus_global_net_rescues": (
            dynamic_minus_global_net_rescues
        ),
        "beta_deciles": beta_deciles,
        "allocation_discrimination": allocation_discrimination,
        "split": block.get("split", {}),
    }


def _cell(summary: dict, method: str, metric: str) -> str:
    value = summary["aggregate"][method][metric]
    mean = f"{value['mean']:.5f}".replace("0.", ".")
    if summary["n_seeds"] > 1:
        std = f"{value['std']:.5f}".replace("0.", ".")
        return f"{mean} ({std})"
    return mean


def _mean_cell(summary: dict, method: str, metric: str) -> str:
    value = summary["aggregate"][method][metric]["mean"]
    return f"{value:.5f}".replace("0.", ".")


def _delta_cell(summary: dict, baseline: str, metric: str) -> str:
    value = summary["paired"][baseline][metric]
    delta = value["difference"]
    low, high = value["cluster_bootstrap_ci95"]
    rendered = f"{delta:+.5f} [{low:+.5f},{high:+.5f}]"
    return rendered.replace("+0.", "+.").replace("-0.", "-.")


def _tex_decimal(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}".replace("0.", ".").replace("-0.", "-.")


def _mean_sd_cell(
        value: dict,
        digits: int = 5,
        integer: bool = False,
) -> str:
    if integer:
        mean = f"{value['mean']:.0f}"
        if len(value["values"]) > 1:
            return f"{mean} ({value['std']:.1f})"
        return mean
    mean = _tex_decimal(value["mean"], digits)
    if len(value["values"]) > 1:
        return f"{mean} ({_tex_decimal(value['std'], digits)})"
    return mean


def write_macros(summary: dict, destination: Path) -> None:
    domain_names = {
        "Video_Games": "Video",
        "Baby_Products": "Baby",
        "Diginetica_HID": "Digi",
    }
    capacity_names = {
        "dynamic": "Primary",
        "dynamic_context_mlp": "ContextMLP",
        "dynamic_full_linear": "FullLinear",
        "dynamic_full_mlp": "FullMLP",
        "dynamic_without_cross_expert": "NoCrossExpert",
        "dynamic_without_memory_certainty": "NoMemoryCertainty",
    }
    rrf_names = {
        "rrf_k10": "KTen",
        "rrf_k20": "KTwenty",
        "rrf_k60": "KSixty",
    }
    rows = [
        "% Generated by summarize_dynamic_beta.py; do not edit manually.",
        (
            "\\newcommand{\\DynamicSeedCount}{"
            f"{min(value['n_seeds'] for value in summary['domains'].values())}"
            "}"
        ),
    ]
    for domain, prefix in domain_names.items():
        block = summary["domains"][domain]
        dynamic = block["aggregate"]["dynamic"]
        rows.extend([
            (
                f"\\newcommand{{\\Dynamic{prefix}Rtwenty}}{{"
                f"{_tex_decimal(dynamic['recall@20']['mean'])}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}BetaMean}}{{"
                f"{_tex_decimal(block['dynamic_beta']['mean']['mean'], 3)}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}BetaWithinSD}}{{"
                f"{_tex_decimal(block['dynamic_beta']['std']['mean'], 3)}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}NetVsGlobal}}{{"
                f"{block['dynamic_minus_global_net_rescues']['mean']:.0f}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}LowBetaAdv}}{{"
                f"{_tex_decimal(block['allocation_discrimination']['low_beta_deciles_1_3']['mean'])}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}HighBetaAdv}}{{"
                f"{_tex_decimal(block['allocation_discrimination']['high_beta_deciles_8_10']['mean'])}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}AllocationGap}}{{"
                f"{_tex_decimal(block['allocation_discrimination']['high_minus_low']['mean'])}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}OOFRows}}{{"
                f"{_mean_sd_cell(block['calibration']['rows'], integer=True)}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}Actionable}}{{"
                f"{_mean_sd_cell(block['calibration']['actionable'], integer=True)}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}ActionableShare}}{{"
                f"{_mean_sd_cell(block['calibration']['actionable_share'], 4)}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}GlobalBeta}}{{"
                f"{_mean_sd_cell(block['calibration']['global_beta'], 5)}"
                "}"
            ),
        ])
        split = block.get("split", {})
        required_split = {
            "validation_candidate_pool_queries",
            "declared_validation_queries",
            "declared_validation_target_events_removed_from_tuning_sessions",
            "declared_validation_source_overlap",
            "validation_candidate_pool_source_overlap",
            "eligible_source_sessions",
            "profile_query_fingerprint",
            "gate_query_fingerprint",
        }
        missing_split = sorted(required_split - set(split))
        if missing_split:
            raise ValueError(
                f"{domain}: missing finalized split fields {missing_split}"
            )
        missing_rrf = sorted(set(rrf_names) - set(block["fusion_sensitivity"]))
        if missing_rrf:
            raise ValueError(
                f"{domain}: missing RRF sensitivity settings {missing_rrf}"
            )
        targets_removed = int(
            split[
                "declared_validation_target_events_removed_from_tuning_sessions"
            ]
        )
        candidate_oof_overlap = int(
            split["validation_candidate_pool_source_overlap"]
        )
        rows.extend([
            (
                f"\\newcommand{{\\Dynamic{prefix}CandidatePool}}{{"
                f"{int(split['validation_candidate_pool_queries']):,}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}DeclaredValidation}}{{"
                f"{int(split['declared_validation_queries']):,}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}TargetsRemoved}}{{"
                f"{targets_removed:,}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}DeclaredOOFOverlap}}{{"
                f"{int(split['declared_validation_source_overlap']):,}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}CandidateOOFOverlap}}{{"
                f"{candidate_oof_overlap:,}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}Eligible}}{{"
                f"{int(split['eligible_source_sessions']):,}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}ProfileHash}}{{"
                f"{str(split['profile_query_fingerprint'])[:8]}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}GateHash}}{{"
                f"{str(split['gate_query_fingerprint'])[:8]}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}Rescues}}{{"
                f"{_mean_sd_cell(dynamic['rescues_vs_memory'], integer=True)}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}Damage}}{{"
                f"{_mean_sd_cell(dynamic['damage_vs_memory'], integer=True)}"
                "}"
            ),
            (
                f"\\newcommand{{\\Dynamic{prefix}NetRescues}}{{"
                f"{_mean_sd_cell(dynamic['net_rescues_vs_memory'], integer=True)}"
                "}"
            ),
        ])
        for method, suffix in capacity_names.items():
            if method in block["aggregate"]:
                rows.append(
                    f"\\newcommand{{\\Dynamic{prefix}{suffix}U}}{{"
                    f"{_mean_sd_cell(block['aggregate'][method]['utility'])}"
                    "}"
                )
        for setting, suffix in rrf_names.items():
            value = _mean_sd_cell(
                block["fusion_sensitivity"][setting]["utility"])
            rows.append(
                f"\\newcommand{{\\Dynamic{prefix}{suffix}U}}{{"
                f"{value}"
                "}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + "\n")


def write_tex(summary: dict, destination: Path) -> None:
    labels = {
        "Video_Games": "Video",
        "Baby_Products": "Baby",
        "Diginetica_HID": "Diginetica",
    }
    rows = [
        "% Generated by summarize_dynamic_beta.py; do not edit manually.",
        "\\begin{table}[t]",
        "\\caption{Allocation ablation on full-catalogue test queries. Values are mean $\\pm$ SD over matched seeds; $U=.5R@6+.5R@20$. No row uses validation to fit or select $\\beta$.}",
        "\\label{tab:allocation}",
        "\\centering\\scriptsize",
        "\\setlength{\\tabcolsep}{2.7pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Domain & Allocation & R@6 & R@20 & nD@20 & $U$\\\\",
        "\\midrule",
    ]
    method_labels = {
        "memory_only": "Memory endpoint",
        "neural_only": "Neural endpoint",
        "fixed_05": "Fixed $.5$",
        "oof_global": "OOF global",
        "oof_short_long": "OOF short/long",
        "dynamic": "\\textbf{Dynamic $\\beta_q$}",
    }
    for domain in DOMAINS:
        block = summary["domains"][domain]
        for row_index, method in enumerate(method_labels):
            domain_label = labels[domain] if row_index == 0 else ""
            rows.append(
                f"{domain_label} & {method_labels[method]} & "
                f"{_cell(block, method, 'recall@6')} & "
                f"{_cell(block, method, 'recall@20')} & "
                f"{_cell(block, method, 'ndcg@20')} & "
                f"{_cell(block, method, 'utility')}\\\\"
            )
        if domain != DOMAINS[-1]:
            rows.append("\\midrule")
    rows.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        "\\caption{Dynamic $\\beta_q$ minus training-only global $\\beta$; paired query-level 95\\% bootstrap intervals after averaging matched-seed outcomes.}",
        "\\label{tab:dynamic-delta}",
        "\\centering\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        (
            "Domain & $\\Delta$R@6 & $\\Delta$R@20 & "
            "$\\Delta$nD@20 & $\\Delta U$\\\\"
        ),
        "\\midrule",
    ])
    for domain in DOMAINS:
        block = summary["domains"][domain]
        rows.append(
            f"{labels[domain]} & "
            f"{_delta_cell(block, 'oof_global', 'recall@6')} & "
            f"{_delta_cell(block, 'oof_global', 'recall@20')} & "
            f"{_delta_cell(block, 'oof_global', 'ndcg@20')} & "
            f"{_delta_cell(block, 'oof_global', 'utility')}\\\\"
        )
    rows.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
        "",
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows))


def write_main_tex(summary: dict, destination: Path) -> None:
    """Write the compact result table used by the eight-page paper."""
    labels = {
        "Video_Games": "Video",
        "Baby_Products": "Baby",
        "Diginetica_HID": "Diginetica",
    }
    rows = [
        "% Generated by summarize_dynamic_beta.py; do not edit manually.",
        "\\begin{table}[t]",
        (
            "\\caption{Compact allocation result on full-catalogue test "
            "queries. Values are matched-seed R@20 means. $\\Delta U$ is "
            "dynamic minus OOF-global with a paired query-level 95\\% "
            "bootstrap interval; complete cutoffs and seed SD are in the "
            "supplement.}"
        ),
        "\\label{tab:allocation-main}",
        "\\centering\\scriptsize",
        "\\setlength{\\tabcolsep}{2.7pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        (
            "Domain & Memory & Neural & Equal $.5$ & OOF global & "
            "OOF short/long & Dynamic & $\\Delta U$ [95\\% CI]\\\\"
        ),
        "\\midrule",
    ]
    for domain in DOMAINS:
        block = summary["domains"][domain]
        rows.append(
            f"{labels[domain]} & "
            f"{_mean_cell(block, 'memory_only', 'recall@20')} & "
            f"{_mean_cell(block, 'neural_only', 'recall@20')} & "
            f"{_mean_cell(block, 'fixed_05', 'recall@20')} & "
            f"{_mean_cell(block, 'oof_global', 'recall@20')} & "
            f"{_mean_cell(block, 'oof_short_long', 'recall@20')} & "
            f"{_mean_cell(block, 'dynamic', 'recall@20')} & "
            f"{_delta_cell(block, 'oof_global', 'utility')}\\\\"
        )
    rows.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
        "",
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows))


def write_full_metrics_tex(summary: dict, destination: Path) -> None:
    labels = {
        "Video_Games": "Video",
        "Baby_Products": "Baby",
        "Diginetica_HID": "Diginetica",
    }
    methods = {
        "fixed_05": "Fixed $.5$",
        "oof_global": "OOF global",
        "oof_short_long": "OOF short/long",
        "dynamic": "Dynamic $\\beta_q$",
    }
    rows = [
        "% Generated by summarize_dynamic_beta.py; do not edit manually.",
        "\\begin{table}[h]",
        (
            "\\caption{Complete allocation metrics on full-catalogue test "
            "queries. Values are mean (SD) over matched seeds.}"
        ),
        "\\label{tab:full-allocation-metrics}",
        "\\centering\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        (
            "Domain & Allocation & R@6 & nD@6 & R@10 & nD@10 "
            "& R@20 & nD@20 & $U$\\\\"
        ),
        "\\midrule",
    ]
    for domain in DOMAINS:
        block = summary["domains"][domain]
        for index, (method, method_label) in enumerate(methods.items()):
            domain_label = labels[domain] if index == 0 else ""
            rows.append(
                f"{domain_label} & {method_label} & "
                f"{_cell(block, method, 'recall@6')} & "
                f"{_cell(block, method, 'ndcg@6')} & "
                f"{_cell(block, method, 'recall@10')} & "
                f"{_cell(block, method, 'ndcg@10')} & "
                f"{_cell(block, method, 'recall@20')} & "
                f"{_cell(block, method, 'ndcg@20')} & "
                f"{_cell(block, method, 'utility')}\\\\"
            )
        if domain != DOMAINS[-1]:
            rows.append("\\midrule")
    rows.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
        "",
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=HERE / "dynamic_beta_trainonly_v2_results.json")
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "dynamic_beta_summary.json")
    parser.add_argument(
        "--tex-output", type=Path,
        default=HERE / "paper" / "generated_dynamic_beta_tables.tex")
    parser.add_argument(
        "--macro-output", type=Path,
        default=HERE / "paper" / "generated_dynamic_beta_macros.tex")
    parser.add_argument(
        "--full-metrics-tex-output",
        type=Path,
        default=(
            HERE / "paper" / "generated_dynamic_beta_full_metrics.tex"
        ),
    )
    parser.add_argument(
        "--main-tex-output",
        type=Path,
        default=(
            HERE / "paper" / "generated_dynamic_beta_main_table.tex"
        ),
    )
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    allowed = set(args.seeds) if args.seeds else None
    validate_raw_results(raw, allowed)
    summary = {
        "source": str(args.input),
        "bootstrap_unit": (
            "test query identifier; smallest recoverable unit, with all seed "
            "outcomes paired"
        ),
        "seed_aggregation": "matched per-query mean across seeds",
        "domains": {
            domain: summarize_domain(
                raw[domain], allowed, args.bootstrap_repetitions)
            for domain in DOMAINS
            if domain in raw
        },
    }
    args.output.write_text(json.dumps(summary, indent=2))
    if all(domain in summary["domains"] for domain in DOMAINS):
        write_macros(summary, args.macro_output)
        write_tex(summary, args.tex_output)
        write_main_tex(summary, args.main_tex_output)
        write_full_metrics_tex(summary, args.full_metrics_tex_output)
    print(args.output)
    print(args.macro_output)
    print(args.tex_output)
    print(args.main_tex_output)
    print(args.full_metrics_tex_output)


if __name__ == "__main__":
    main()
