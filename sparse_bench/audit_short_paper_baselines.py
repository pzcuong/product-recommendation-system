#!/usr/bin/env python3
"""Recompute and attribute baseline numbers used by the dynamic-beta paper.

The audit is intentionally read-only with respect to experiment artifacts.  It
recomputes metrics from persisted rank vectors, checks retained JSON values,
extracts selected epochs from checkpoints, and writes a separate report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CUTOFFS = (6, 10, 20)
METRICS = tuple(
    metric
    for cutoff in CUTOFFS
    for metric in (f"recall@{cutoff}", f"ndcg@{cutoff}")
)
SEEDS = (42, 123, 456)


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    local = HERE / candidate
    return local if local.exists() else candidate


def rank_metrics(ranks: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(ranks, dtype=np.float64)
    output: dict[str, float | int] = {"n": int(len(values))}
    for cutoff in CUTOFFS:
        hit = (values > 0) & (values <= cutoff)
        output[f"recall@{cutoff}"] = float(hit.mean())
        gain = np.zeros(len(values), dtype=np.float64)
        gain[hit] = 1.0 / np.log2(values[hit] + 1.0)
        output[f"ndcg@{cutoff}"] = float(gain.mean())
    return output


def assert_metrics(
    recomputed: dict[str, float | int],
    retained: dict[str, float | int],
    label: str,
) -> None:
    for key, value in recomputed.items():
        if key not in retained or abs(float(value) - float(retained[key])) > 1e-15:
            raise RuntimeError(
                f"{label}: retained {key}={retained.get(key)!r}, "
                f"recomputed={value!r}")


def aggregate(runs: list[dict[str, float | int]]) -> dict[str, Any]:
    output: dict[str, Any] = {"n": int(runs[0]["n"])}
    for metric in METRICS:
        values = np.asarray([run[metric] for run in runs], dtype=np.float64)
        output[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "per_seed": {
                str(seed): float(value) for seed, value in zip(SEEDS, values)
            },
        }
    return output


def audit_neighborhood() -> dict[str, Any]:
    result_path = HERE / "neighborhood_baseline_results.json"
    retained = json.loads(result_path.read_text())
    output: dict[str, Any] = {
        "source": str(result_path.relative_to(HERE)),
        "grid": {
            "V-SKNN": {
                "size": 8,
                "axes": {
                    "k": [100, 500],
                    "sample_size": [1000, 5000],
                    "weighting": ["div", "quadratic"],
                    "score_weighting": ["div"],
                },
            },
            "STAN": {
                "size": 16,
                "axes": {
                    "k": [100, 500],
                    "sample_size": [1000, 5000],
                    "lambda_spw": [1.02, 2.0],
                    "lambda_snh": [None, 5000.0],
                    "lambda_inh": [2.05],
                },
            },
            "selection_utility": "0.5 * Recall@6 + 0.5 * Recall@20",
            "source_code": "run_neighborhood_baselines.py:39-77",
        },
        "domains": {},
    }
    for domain, block in retained.items():
        methods: dict[str, Any] = {}
        for name, method in block["methods"].items():
            artifact = resolve(method["artifact"])
            with np.load(artifact, allow_pickle=False) as saved:
                recomputed = rank_metrics(saved["ranks"])
            assert_metrics(recomputed, method["test"], f"{domain}/{name}")
            methods[name] = {
                "artifact": str(artifact),
                "metrics": recomputed,
                "selected_config": method.get("selected_config"),
                "reported_seed_ids": list(method["per_seed"]),
                "seed_semantics": (
                    "deterministic copies, not independent stochastic fits"),
            }
        output["domains"][domain] = {
            "protocol": block["protocol"],
            "methods": methods,
        }
    return output


def checkpoint_epoch(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["epoch"])


def audit_amazon_id_neural() -> dict[str, Any]:
    domains = {
        "Video_Games": ("video_games", 94762),
        "Baby_Products": ("baby_products", 150777),
    }
    models = {
        "GRU4Rec": "gru4rec",
        "NARM": "narm",
        "SR-GNN": "sr_gnn",
        "SIGMA-compatible": "sigma_compatible",
    }
    output: dict[str, Any] = {
        "rank_root": "paper_baseline_artifacts",
        "protocol_evidence": [
            "run_paper_baselines.py",
            "loaders.py:201-322 (Amazon validation is an explicit split)",
            "v2_paired_amazon.json",
        ],
        "domains": {},
    }
    for domain, (prefix, expected_n) in domains.items():
        domain_output: dict[str, Any] = {}
        for model, slug in models.items():
            runs = []
            epochs: dict[str, int] = {}
            fingerprints: dict[str, str] = {}
            artifacts: dict[str, str] = {}
            for seed in SEEDS:
                rank_path = (
                    HERE / "paper_baseline_artifacts"
                    / f"{prefix}_full_{slug}_seed{seed}_ranks.npz"
                )
                checkpoint = rank_path.with_name(
                    rank_path.name.replace("_ranks.npz", ".pt"))
                with np.load(rank_path, allow_pickle=False) as saved:
                    runs.append(rank_metrics(saved["ranks"]))
                    fingerprints[str(seed)] = str(
                        saved["test_fingerprint"].item())
                if int(runs[-1]["n"]) != expected_n:
                    raise RuntimeError(f"{rank_path}: unexpected query count")
                epochs[str(seed)] = checkpoint_epoch(checkpoint)
                artifacts[str(seed)] = str(rank_path)
            domain_output[model] = {
                "aggregate": aggregate(runs),
                "selected_epoch": epochs,
                "rank_artifacts": artifacts,
                "test_fingerprint_sha256": fingerprints,
                "provenance": (
                    "repository-local full-catalog reimplementation"
                    if model != "SIGMA-compatible"
                    else "repository-local architecture-compatible proxy; "
                    "not official SIGMA code"
                ),
            }
        output["domains"][domain] = domain_output
    return output


def audit_diginetica_neural() -> dict[str, Any]:
    result_path = HERE / "paper_baseline_digi_nested.json"
    retained = json.loads(result_path.read_text())["Diginetica_HID"]
    output: dict[str, Any] = {
        "source": str(result_path.relative_to(HERE)),
        "protocol": retained["protocol"],
        "models": {},
    }
    for model in ("GRU4Rec", "NARM", "SR-GNN", "SIGMA-compatible"):
        runs = []
        paths: dict[str, str] = {}
        epochs: dict[str, int] = {}
        for run in retained["models"][model]["runs"]:
            path = resolve(run["rank_artifact"])
            with np.load(path, allow_pickle=False) as saved:
                recomputed = rank_metrics(saved["ranks"])
            assert_metrics(recomputed, run["test"], f"Diginetica/{model}")
            runs.append(recomputed)
            seed = str(run["seed"])
            paths[seed] = str(path)
            epochs[seed] = int(run["best_epoch"])
        output["models"][model] = {
            "aggregate": aggregate(runs),
            "selected_epoch": epochs,
            "rank_artifacts": paths,
            "retained_aggregate_verified": True,
            "provenance": retained["models"][model]["runs"][0]["provenance"],
        }
    return output


def audit_matched_teacher_narm() -> dict[str, Any]:
    result_path = HERE / "narm_tfidf_fairness_nested_results.json"
    retained = json.loads(result_path.read_text())
    corrected = {
        "Video_Games": "E5-small cached item-text embeddings",
        "Baby_Products": "TF-IDF/SVD cached item-text embeddings",
    }
    output: dict[str, Any] = {
        "source": str(result_path.relative_to(HERE)),
        "label_warning": (
            "The retained JSON says teacher=tfidf on both domains, but the "
            "resolver selects the higher-priority Video_Games E5-small cache. "
            "See teacher_resolver_provenance_audit.json."
        ),
        "domains": {},
    }
    for domain, block in retained.items():
        runs = []
        paths: dict[str, str] = {}
        epochs: dict[str, int] = {}
        for run in block["models"]["NARM"]["runs"]:
            path = resolve(run["rank_artifact"])
            with np.load(path, allow_pickle=False) as saved:
                recomputed = rank_metrics(saved["ranks"])
            assert_metrics(recomputed, run["test"], f"{domain}/NARM+teacher")
            runs.append(recomputed)
            seed = str(run["seed"])
            paths[seed] = str(path)
            epochs[seed] = int(run["best_epoch"])
        output["domains"][domain] = {
            "corrected_teacher": corrected[domain],
            "retained_semantic_shape": block["semantic_shape"],
            "projected_shape": block["projected_semantic_shape"],
            "protocol": block["protocol"],
            "aggregate": aggregate(runs),
            "selected_epoch": epochs,
            "rank_artifacts": paths,
        }
    return output


def audit_no_metadata() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for filename in (
        "cearfn_v2_nometa_results.json",
        "cearfn_v2_nometa_nested_results.json",
    ):
        path = HERE / filename
        retained = json.loads(path.read_text())
        domains: dict[str, Any] = {}
        for domain, block in retained.items():
            runs = [run["regime"] for run in block["runs"]]
            domains[domain] = {
                "seeds": [run["seed"] for run in block["runs"]],
                "aggregate": aggregate(runs),
                "pasgr_config": block.get("pasgr_config"),
                "method_family": (
                    "legacy CEARF-N v2 validation-selected short/long beta; "
                    "not the current training-only dynamic-beta method"
                ),
            }
        output[filename] = {"domains": domains}
    output["safety"] = {
        "current_dynamic_no_metadata_available": False,
        "safe_static_control": (
            "cearfn_v2_nometa_nested_results.json / Diginetica_HID only"),
        "unsafe": [
            (
                "Do not label either file as a no-metadata ablation of the "
                "current dynamic-beta method."
            ),
            (
                "Do not use the Diginetica block in "
                "cearfn_v2_nometa_results.json; PROTOCOL_CORRECTIONS.md marks "
                "unqualified historical Diginetica validation artifacts invalid."
            ),
            (
                "The Amazon blocks in cearfn_v2_nometa_results.json retain an "
                "empty pasgr_config and are not a matched current-PASGR control."
            ),
        ],
    }
    return output


def tex_escape(value: Any) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(source, target)
    return text


def metric_cells(
    aggregate_block: dict[str, Any],
    *,
    deterministic: bool = False,
) -> list[str]:
    cells = []
    for metric in METRICS:
        if deterministic:
            value = float(aggregate_block[metric])
            cells.append(f"{value:.5f}")
        else:
            mean = float(aggregate_block[metric]["mean"])
            std = float(aggregate_block[metric]["sample_std"])
            cells.append(f"{mean:.5f} $\\pm$ {std:.5f}")
    return cells


def write_tex(report: dict[str, Any], output: Path) -> None:
    domain_order = ("Video_Games", "Baby_Products", "Diginetica_HID")
    domain_label = {
        "Video_Games": "Video Games",
        "Baby_Products": "Baby Products",
        "Diginetica_HID": "Diginetica",
    }
    method_label = {
        "transition": r"Transition-only$^\dagger$",
        "vsknn": r"V-SKNN$^\dagger$",
        "stan": r"STAN$^\dagger$",
    }
    lines = [
        "% Generated by audit_short_paper_baselines.py; do not hand-edit.",
        r"\section{Audited External-Baseline Details}",
        (
            r"\noindent The following tables report the full-catalogue "
            r"baseline results used in the main-paper comparison. A dagger "
            r"marks deterministic methods: their rows for "
            r"seeds 42/123/456 are identical copies for schema compatibility, "
            r"not independent stochastic fits.  STAN uses loader session "
            r"order as its recency coordinate because original timestamps "
            r"are unavailable.  Baseline definitions follow V-SKNN "
            r"\cite{ludewig2018vsknn}, STAN \cite{garg2019stan}, GRU4Rec "
            r"\cite{hidasi2016gru4rec}, NARM \cite{li2017narm}, SR-GNN "
            r"\cite{wu2019srgnn}, and SIGMA \cite{sigma2025}; the last is "
            r"our compatible reimplementation."
        ),
        "",
        r"\begin{table}[h]",
        r"\caption{Validation grids and selected neighborhood configurations.}",
        r"\centering\scriptsize",
        r"\begin{tabular}{llp{.58\textwidth}}",
        r"\toprule",
        r"Domain & Method & Grid and selected configuration\\",
        r"\midrule",
    ]
    neighborhood = report["neighborhood_and_transition"]
    for domain in domain_order:
        methods = neighborhood["domains"][domain]["methods"]
        v = methods["vsknn"]["selected_config"]
        s = methods["stan"]["selected_config"]
        lines.append(
            f"{domain_label[domain]} & V-SKNN & "
            f"8 cells: $k\\in\\{{100,500\\}}$, sample "
            f"$\\in\\{{1000,5000\\}}$, weighting "
            f"$\\in\\{{\\mathrm{{div}},\\mathrm{{quadratic}}\\}}$; selected "
            f"$k={v['k']}$, sample $={v['sample_size']}$, "
            f"{tex_escape(v['weighting'])}, exclude-seen="
            f"{str(v['exclude_seen']).lower()}.\\\\"
        )
        lines.append(
            f"{domain_label[domain]} & STAN & "
            f"16 cells: $k\\in\\{{100,500\\}}$, sample "
            f"$\\in\\{{1000,5000\\}}$, "
            f"$\\lambda_{{spw}}\\in\\{{1.02,2.0\\}}$, "
            f"$\\lambda_{{snh}}\\in\\{{\\emptyset,5000\\}}$; selected "
            f"$k={s['k']}$, sample $={s['sample_size']}$, "
            f"$\\lambda_{{spw}}={s['lambda_spw']}$, "
            f"$\\lambda_{{snh}}=\\emptyset$, exclude-seen="
            f"{str(s['exclude_seen']).lower()}.\\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\par\smallskip\scriptsize Selection maximizes "
            r"$0.5R@6+0.5R@20$ on 5,000 declared validation queries. "
            r"Score weighting is \texttt{div}; "
            r"$\lambda_{inh}=2.05$ is fixed."
        ),
        r"\end{table}",
        "",
    ])

    amazon = report["id_only_neural_amazon"]["domains"]
    digi = report["id_only_neural_diginetica"]["models"]
    for domain in domain_order:
        metric_rows = []
        for method in ("transition", "vsknn", "stan"):
            block = neighborhood["domains"][domain]["methods"][method]
            metric_rows.append((
                method_label[method],
                metric_cells(block["metrics"], deterministic=True),
            ))
        neural = digi if domain == "Diginetica_HID" else amazon[domain]
        for method in ("GRU4Rec", "NARM", "SR-GNN", "SIGMA-compatible"):
            metric_rows.append((
                tex_escape(method),
                metric_cells(neural[method]["aggregate"]),
            ))
        lines.extend([
            r"\begin{table}[h]",
            (
                r"\caption{Compact full-catalog baseline metrics on "
                + domain_label[domain]
                + r". Neural entries are mean $\pm$ sample SD over seeds "
                r"42/123/456. R@10 and earlier-cutoff nDCG are omitted "
                r"for compactness.}"
            ),
            r"\centering\scriptsize",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Method & R@6 & R@20 & nDCG@20\\",
            r"\midrule",
        ])
        for label, cells in metric_rows:
            lines.append(
                label + " & " + " & ".join([cells[0], cells[4], cells[5]]) + r"\\")
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ])

    lines.extend([
        r"\begin{table}[h]",
        r"\caption{Validation-selected epochs for ID-only neural baselines.}",
        r"\centering\scriptsize",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Domain & GRU4Rec & NARM & SR-GNN & SIGMA-compatible\\",
        r"\midrule",
    ])
    for domain in domain_order:
        neural = digi if domain == "Diginetica_HID" else amazon[domain]
        epoch_cells = []
        for method in ("GRU4Rec", "NARM", "SR-GNN", "SIGMA-compatible"):
            selected = neural[method]["selected_epoch"]
            epoch_cells.append("/".join(str(selected[str(seed)]) for seed in SEEDS))
        lines.append(
            domain_label[domain] + " & " + " & ".join(epoch_cells) + r"\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\par\smallskip\scriptsize Shared fixed settings: 64 dimensions, "
            r"50-item maximum context, full-catalog cross-entropy, AdamW, "
            r"weight decay $10^{-5}$, gradient clipping at 5, batch 512 "
            r"(SR-GNN: 128), learning rate $10^{-3}$ "
            r"(SIGMA-compatible: $5\!\times\!10^{-4}$).  Only epoch is "
            r"validation-selected; these are our local "
            r"reimplementations, and SIGMA-compatible is not official SIGMA."
        ),
        r"\end{table}",
        "",
    ])

    teacher_path = HERE / "teacher_resolver_provenance_audit.json"
    teacher = json.loads(teacher_path.read_text())
    matched = report["matched_teacher_narm_amazon"]["domains"]
    provenance_labels = {
        "Video_Games": "E5-small cache",
        "Baby_Products": "TF-IDF/SVD cache",
        "Diginetica_HID": "TF-IDF/SVD product-name cache",
    }
    lines.extend([
        r"\begin{table}[h]",
        (
            r"\caption{Resolved semantic-teacher provenance. Reported "
            r"SHA-256 prefixes identify the cached matrices used.}"
        ),
        r"\centering\small",
        r"\begin{tabular}{lp{.40\textwidth}ll}",
        r"\toprule",
        r"Domain & Resolved teacher & Shape & SHA-256\\",
        r"\midrule",
    ])
    for domain in domain_order:
        provenance = teacher["domains"][domain]
        shape = r"$" + r"\times".join(str(x) for x in provenance["shape"]) + r"$"
        lines.append(
            f"{domain_label[domain]} & "
            f"{tex_escape(provenance_labels.get(domain, provenance['corrected_teacher_label']))} & {shape} & "
            f"\\texttt{{{provenance['sha256'][:12]}}}\\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[h]",
        (
            r"\caption{Matched-teacher NARM final-cutoff metrics on Amazon "
            r"domains. Values are mean $\pm$ sample SD over seeds 42/123/456; "
            r"no matched-teacher Diginetica run is retained.}"
        ),
        r"\centering\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Domain & R@20 & nDCG@20\\",
        r"\midrule",
    ])
    for domain in ("Video_Games", "Baby_Products"):
        cells = metric_cells(matched[domain]["aggregate"])
        lines.append(domain_label[domain] + " & " + " & ".join([cells[4], cells[5]]) + r"\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip\scriptsize Only the final cutoff is shown because "
        r"the matched-teacher claim concerns R@20 and nDCG@20.",
        (
            r"\par\smallskip\scriptsize The earlier fairness record "
            r"mislabels the Video Games teacher: resolving the loaded matrix gives the "
            r"higher-priority E5-small cache before TF--IDF/SVD. The Video "
            r"cache has no encoder sidecar; the reported SHA-256 prefix "
            r"identifies the matrix used. Exact matched-teacher NARM R@20 "
            r"means are .153869695 and .066278898."
        ),
        r"\end{table}",
        "",
        (
            r"\noindent\textbf{No-metadata scope.} The available no-metadata "
            r"results do not ablate the current training-only dynamic-$\beta$ "
            r"method; they belong to the earlier validation-selected "
            r"short/long-$\beta$ family. Only the nested Diginetica result is "
            r"protocol-safe as a static diagnostic."
        ),
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "short_paper_baseline_artifact_audit.json",
    )
    parser.add_argument(
        "--tex-output",
        type=Path,
        default=HERE / "paper" / "generated_dynamic_beta_baseline_audit.tex",
    )
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_scope": (
            "Original result JSON, checkpoints, ranks, and teacher arrays are "
            "never modified."
        ),
        "neighborhood_and_transition": audit_neighborhood(),
        "id_only_neural_amazon": audit_amazon_id_neural(),
        "id_only_neural_diginetica": audit_diginetica_neural(),
        "matched_teacher_narm_amazon": audit_matched_teacher_narm(),
        "no_metadata": audit_no_metadata(),
        "neural_training_configuration": {
            "fixed_not_grid_searched": True,
            "source": [
                "run_paper_baselines.py:118-218",
                "paper_models.py:18-178",
            ],
            "shared": {
                "embedding_dim": 64,
                "max_context_length": 50,
                "objective": "full-catalog cross-entropy",
                "optimizer": "AdamW",
                "weight_decay": 1e-5,
                "gradient_clip_norm": 5.0,
                "batch_size": 512,
                "SR-GNN_batch_size": 128,
                "learning_rate_GRU4Rec_NARM_SR-GNN": 1e-3,
                "learning_rate_SIGMA-compatible": 5e-4,
                "epoch_selection_utility": (
                    "0.5 * Recall@6 + 0.5 * Recall@20 on 5,000 validation queries"
                ),
            },
            "Diginetica_budget": {
                "max_epochs": 12,
                "patience": 3,
                "evidence": [
                    "CEARFN_FINDINGS.md:213-214",
                    "paper_baseline_digi_nested.json per-epoch histories",
                ],
            },
            "Amazon_budget_caveat": (
                "Selected epochs are retained in checkpoints and reported "
                "above, but the deleted/missing paper_baseline_results.json "
                "means the exact invocation-level max_epochs/patience values "
                "are not independently retained. Do not claim a uniform "
                "Amazon search grid beyond validation-selected epoch."
            ),
        },
        "unsafe_claims": [
            (
                "V-SKNN, STAN, and transition rows have three identical seed "
                "copies by design; they are not three independent fits."
            ),
            (
                "STAN uses deterministic loader session order rather than "
                "original wall-clock timestamps."
            ),
            (
                "The neural baselines are repository-local reimplementations; "
                "SIGMA-compatible is explicitly not official SIGMA."
            ),
            (
                "The neural baselines had fixed architecture hyperparameters "
                "and validation-selected epochs, not an exhaustive matched "
                "hyperparameter grid."
            ),
            (
                "Matched-teacher NARM exists only for the two Amazon domains; "
                "do not extend that attribution claim to Diginetica."
            ),
        ],
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    write_tex(report, args.tex_output)
    print(args.output)
    print(args.tex_output)


if __name__ == "__main__":
    main()
