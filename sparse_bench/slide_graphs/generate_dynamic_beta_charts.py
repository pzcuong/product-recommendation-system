#!/usr/bin/env python3
"""Render data-driven CEARF-N dynamic-beta charts for papers and slides.

The generator deliberately keeps all experimental values outside the source:

* aggregate test metrics come from the newest training-only result artifact;
* paired differences and query-level bootstrap intervals come from
  ``dynamic_beta_summary.json``;
* context plots use the rank artifact recorded by each completed run; and
* inference measurements come from the newest available benchmark JSON;
* external baselines are reconstructed from their JSON/NPZ artifacts; and
* expert-swap, fusion-operator, and allocation-control charts are emitted only
  when their raw three-domain results are complete.

Every figure is exported as SVG and PNG under a new output directory, leaving
the earlier validation-gating figures untouched.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "sparse_bench"
OUT = ROOT / "output" / "charts" / "cearfn_dynamic"
V2_RESULTS_PATH = BENCH / "dynamic_beta_trainonly_v2_results.json"
# The slide deck is a final-result artifact: never silently fall back to the
# earlier development run when the declared train-only-v2 result is absent.
RESULTS_PATH = V2_RESULTS_PATH
SUMMARY_PATH = BENCH / "dynamic_beta_summary.json"
NEIGHBORHOOD_BASELINE_PATH = BENCH / "neighborhood_baseline_results.json"
DIGI_NEURAL_BASELINE_PATH = BENCH / "paper_baseline_digi_nested.json"
AMAZON_NEURAL_ARTIFACT_DIR = BENCH / "paper_baseline_artifacts"
EXPERT_SWAP_PATH = BENCH / "dynamic_beta_expert_swap_results.json"
FUSION_CONTROL_PATH = (
    BENCH / "dynamic_beta_fusion_operator_control.json"
)
ALLOCATION_CONTROL_PATH = (
    BENCH / "dynamic_beta_allocation_controls_summary.json"
)

DOMAIN_LABELS = OrderedDict(
    [
        ("Video_Games", "Video Games"),
        ("Baby_Products", "Baby Products"),
        ("Diginetica_HID", "Diginetica"),
    ]
)
MODE_LABELS = OrderedDict(
    [
        ("memory_only", "Memory-only"),
        ("neural_only", "Neural-only"),
        ("fixed_05", r"Fixed $\beta=.5$"),
        ("oof_global", "OOF global"),
        ("oof_short_long", "OOF short/long"),
        ("dynamic", r"Dynamic $\beta_q$"),
    ]
)
MODE_COLORS = OrderedDict(
    [
        ("memory_only", "#5F6368"),
        ("neural_only", "#BDBDBD"),
        ("fixed_05", "#C8A46B"),
        ("oof_global", "#7F8C8D"),
        ("oof_short_long", "#4FA3A5"),
        ("dynamic", "#2B6CB0"),
    ]
)
PRIMARY_ALLOCATION_SEEDS = (42, 123, 456)
DOMAIN_COLORS = OrderedDict(
    [
        ("Video_Games", "#2B6CB0"),
        ("Baby_Products", "#0B8F55"),
        ("Diginetica_HID", "#E76F51"),
    ]
)
INK = "#171717"
MUTED = "#575757"
GRID = "#D7D7D7"
WHITE = "#FFFFFF"
POSITIVE = "#2B6CB0"
NEGATIVE = "#D05A4E"

ALLOCATION_CONTROL_PROTOCOL = (
    "dynamic-beta-allocation-controls-v2-assignment-shuffle"
)
ALLOCATION_CONTROL_SEEDS = (42, 123, 456)
BETA_DECILE_SEEDS = (42, 123, 456)
FULL_METRIC_PAIRED_SEEDS = (42, 123, 456)
FUSION_ABLATION_SEEDS = (42, 123, 456)
FUSION_ABLATION_MODES = OrderedDict(
    [
        ("memory_only", ("Memory-only", "#8D8D8D")),
        ("neural_only", ("Neural-only", "#BDBDBD")),
        ("dynamic", (r"Fused (dynamic $\beta_q$)", "#2B6CB0")),
    ]
)
FULL_METRIC_PAIRED_METRICS = OrderedDict(
    [
        ("recall@6", "Recall@6"),
        ("recall@10", "Recall@10"),
        ("recall@20", "Recall@20"),
        ("ndcg@6", "nDCG@6"),
        ("ndcg@10", "nDCG@10"),
        ("ndcg@20", "nDCG@20"),
    ]
)
ALLOCATION_CONTROL_METHODS = OrderedDict(
    [
        ("oof_global", "OOF global"),
        ("bucket_head_tail", "Head/tail"),
        (
            "bucket_short_long_head_tail",
            "Short/long ×\nhead/tail",
        ),
        ("dynamic_delta_005", r"Dynamic $\Delta=.05$"),
        (
            "dynamic_delta_010",
            "Dynamic "
            + r"$\Delta=.10$"
            + "\n(primary)",
        ),
        (
            "dynamic_beta_permuted",
            "Primary "
            + r"$\beta_q$"
            + "\npermuted",
        ),
        ("dynamic_delta_020", r"Dynamic $\Delta=.20$"),
    ]
)
ALLOCATION_CONTROL_METRICS = OrderedDict(
    [
        ("utility", "Utility"),
        ("ndcg@20", "nDCG@20"),
    ]
)
ASSIGNMENT_EFFECT_METRICS = OrderedDict(
    [
        ("ndcg@6", "nDCG@6"),
        ("ndcg@10", "nDCG@10"),
        ("ndcg@20", "nDCG@20"),
        ("utility", "Utility"),
    ]
)
PRIMARY_GATE_FEATURES = OrderedDict(
    [
        ("log_context_length", r"$\log(1+L)$"),
        (
            "log_last_item_frequency",
            r"$\log(1+\mathrm{last\ frequency})$",
        ),
        ("last_item_is_tail", "Tail indicator"),
    ]
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_runs(domain_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the last complete run for each seed, sorted by seed."""
    runs_by_seed: dict[int, dict[str, Any]] = {}
    for run in domain_payload.get("runs", []):
        metrics = run.get("test", {}).get("metrics", {})
        if "dynamic" not in metrics:
            continue
        runs_by_seed[int(run["seed"])] = run
    return [runs_by_seed[seed] for seed in sorted(runs_by_seed)]


def _runs_by_domain(results: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        domain: _unique_runs(results.get(domain, {}))
        for domain in DOMAIN_LABELS
    }


def _mean(values: Iterable[float]) -> float:
    data = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(data)) if len(data) else math.nan


def _std(values: Iterable[float]) -> float:
    data = np.asarray(list(values), dtype=np.float64)
    return float(np.std(data, ddof=1)) if len(data) > 1 else 0.0


def _aggregate_metric(
    runs: list[dict[str, Any]], mode: str, metric: str
) -> tuple[float, float]:
    values = [
        float(run["test"]["metrics"][mode][metric])
        for run in runs
        if metric in run["test"]["metrics"].get(mode, {})
    ]
    return _mean(values), _std(values)


def _seed_text(
    runs_by_domain: dict[str, list[dict[str, Any]]],
    domains: Iterable[str] | None = None,
) -> str:
    selected = list(domains or DOMAIN_LABELS.keys())
    parts = []
    for domain in selected:
        seeds = [int(run["seed"]) for run in runs_by_domain.get(domain, [])]
        joined = ",".join(map(str, seeds)) if seeds else "none"
        parts.append(f"{DOMAIN_LABELS[domain]} n={len(seeds)} ({joined})")
    return "Available completed seeds: " + " · ".join(parts)


def _summary_seed_text(summary: dict[str, Any]) -> str:
    parts = []
    for domain, label in DOMAIN_LABELS.items():
        payload = summary.get("domains", {}).get(domain, {})
        seeds = [int(seed) for seed in payload.get("seeds", [])]
        joined = ",".join(map(str, seeds)) if seeds else "none"
        parts.append(f"{label} n={len(seeds)} ({joined})")
    return "Paired-summary seeds: " + " · ".join(parts)


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "figure.titlesize": 22,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.0,
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "svg.fonttype": "none",
        }
    )


def _clean_axis(axis: mpl.axes.Axes, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)


def _add_footer(figure: mpl.figure.Figure, text: str) -> None:
    figure.text(
        0.995,
        0.012,
        text,
        ha="right",
        va="bottom",
        fontsize=8.3,
        color=MUTED,
        family="DejaVu Sans",
    )


def _save(
    figure: mpl.figure.Figure,
    stem: str,
    *,
    top: float = 0.84,
    bottom: float = 0.15,
) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(bottom=bottom, top=top)
    svg_path = OUT / f"{stem}.svg"
    png_path = OUT / f"{stem}.png"
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return [str(svg_path), str(png_path)]


def _diagram_box(
    axis: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = INK,
    fontsize: float = 10.5,
    linewidth: float = 1.2,
) -> None:
    """Draw one slide-safe rounded box in normalized axis coordinates."""
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
    )


def _diagram_arrow(
    axis: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    linestyle: str = "-",
    linewidth: float = 1.5,
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linestyle": linestyle,
            "linewidth": linewidth,
            "mutation_scale": 13,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def method_and_protocol_overview() -> list[str]:
    """Render conceptual method and leakage-boundary figures for slides."""
    outputs: list[str] = []

    figure, axis = plt.subplots(figsize=(16, 8.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.suptitle(
        r"CEARF-N learns a bounded query-wise allocation law",
        fontweight="bold",
    )
    axis.text(
        0.02,
        0.91,
        "TRAINING · source-disjoint out-of-fit supervision",
        fontsize=12,
        fontweight="bold",
        color=MUTED,
    )
    training_boxes = [
        (0.02, "Inner-fit\nsessions", "#E9F1FA"),
        (0.18, "Frozen expert\nrankings", "#F8EBDD"),
        (0.34, "OOF ranks\n+ targets", "#E9F1FA"),
        (
            0.50,
            "Stage 1\nlearn " + r"$\beta_{\mathrm{OOF}}$",
            "#F8EBDD",
        ),
        (
            0.66,
            "Stage 2\nlearn " + r"$(\mathbf{w},b)$",
            "#F8EBDD",
        ),
        (0.82, "Freeze gate\n+ manifest", "#E8F4EC"),
    ]
    for x, label, color in training_boxes:
        _diagram_box(
            axis, x, 0.69, 0.135, 0.13, label, facecolor=color
        )
    for left, right in zip(training_boxes, training_boxes[1:]):
        _diagram_arrow(
            axis,
            (left[0] + 0.135, 0.755),
            (right[0], 0.755),
        )

    _diagram_box(
        axis,
        0.33,
        0.48,
        0.34,
        0.095,
        "Declared-validation and test labels\nnever fit or select allocation",
        facecolor="#F7EAEA",
        edgecolor=NEGATIVE,
        fontsize=10.2,
    )
    axis.plot(
        [0.50, 0.50],
        [0.575, 0.66],
        color=NEGATIVE,
        linewidth=1.6,
        linestyle=(0, (3, 3)),
    )
    axis.text(
        0.50,
        0.625,
        "×",
        ha="center",
        va="center",
        fontsize=18,
        color=NEGATIVE,
        fontweight="bold",
    )

    axis.plot([0.02, 0.96], [0.43, 0.43], color=GRID, linewidth=1.2)
    axis.text(
        0.02,
        0.405,
        "INFERENCE · target-free per-query allocation",
        fontsize=12,
        fontweight="bold",
        color=MUTED,
    )
    _diagram_box(
        axis, 0.02, 0.12, 0.11, 0.12, "Query\nprefix $q$",
        facecolor="#E9F1FA",
    )
    _diagram_box(
        axis, 0.20, 0.15, 0.15, 0.10, "CEARF memory\n$\\pi_M$",
        facecolor="#F8EBDD",
    )
    _diagram_box(
        axis, 0.20, 0.02, 0.15, 0.10, "PASGR neural\n$\\pi_N$",
        facecolor="#F8EBDD",
    )
    _diagram_box(
        axis, 0.20, 0.28, 0.15, 0.10,
        "Context features\n$\\widetilde{\\mathbf{z}}_q$",
        facecolor="#E9F1FA",
    )
    _diagram_box(
        axis, 0.43, 0.28, 0.15, 0.10,
        "Frozen gate\n" + r"$\beta_q\in(0,1)$",
        facecolor="#E8F4EC",
    )
    _diagram_box(
        axis, 0.66, 0.10, 0.16, 0.16,
        "Weighted RRF\n"
        + r"$F_q=(1-\beta_q)\rho_M+\beta_q\rho_N$",
        facecolor="#E8F4EC",
        fontsize=9.6,
    )
    _diagram_box(
        axis, 0.89, 0.13, 0.09, 0.10, "Top-20",
        facecolor="#E8F4EC",
    )
    _diagram_arrow(axis, (0.13, 0.19), (0.20, 0.20))
    _diagram_arrow(axis, (0.13, 0.16), (0.20, 0.07))
    _diagram_arrow(axis, (0.13, 0.22), (0.20, 0.33))
    _diagram_arrow(axis, (0.35, 0.33), (0.43, 0.33))
    _diagram_arrow(axis, (0.35, 0.20), (0.66, 0.22))
    _diagram_arrow(axis, (0.35, 0.07), (0.66, 0.14))
    _diagram_arrow(axis, (0.58, 0.33), (0.66, 0.24))
    _diagram_arrow(axis, (0.82, 0.18), (0.89, 0.18))
    figure.text(
        0.50,
        0.022,
        r"Five optimized coefficients · rank-only expert interface · "
        r"$\Delta_{\mathrm{eff}}\leq .10$ bounds pair intervention",
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )
    outputs.extend(
        _save(
            figure,
            "00-method-architecture",
            top=0.90,
            bottom=0.08,
        )
    )

    figure, axis = plt.subplots(figsize=(16, 6.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.suptitle(
        "Leakage-safe allocation lineage",
        fontweight="bold",
    )
    steps = [
        ("1", "Declare 5k\nvalidation queries", "#F7EAEA"),
        ("2", "Hash-select 5k\ntraining sources", "#E9F1FA"),
        ("3", "Inner-fit experts\nwithout sources", "#F8EBDD"),
        ("4", "1k OOF:\nlock memory profile", "#E9F1FA"),
        ("5", "4k OOF:\nfit prior + gate", "#E9F1FA"),
        ("6", "Freeze manifest,\nfeatures and gate", "#E8F4EC"),
        ("7", "Final expert refit\nthen test", "#E8F4EC"),
    ]
    width = 0.12
    starts = np.linspace(0.025, 0.855, len(steps))
    for x, (number, label, color) in zip(starts, steps):
        _diagram_box(
            axis, float(x), 0.42, width, 0.24, label,
            facecolor=color, fontsize=9.7
        )
        axis.text(
            float(x) + width / 2,
            0.72,
            number,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )
    for left, right in zip(starts, starts[1:]):
        _diagram_arrow(
            axis,
            (float(left) + width, 0.54),
            (float(right), 0.54),
        )
    axis.text(
        0.50,
        0.25,
        "Profile-lock sources and gate-calibration sources are disjoint; "
        "neither appears in the corresponding inner expert fit.",
        ha="center",
        fontsize=11,
        color=INK,
    )
    axis.text(
        0.50,
        0.13,
        "Only expert refitting may use split-permitted data after allocation "
        "is frozen; validation/test targets never update β.",
        ha="center",
        fontsize=10.3,
        color=MUTED,
    )
    outputs.extend(
        _save(
            figure,
            "00-protocol-lineage",
            top=0.84,
            bottom=0.08,
        )
    )

    figure, axis = plt.subplots(figsize=(14, 8))
    boundary = np.linspace(0.0, 1.0, 400)
    axis.fill_between(
        boundary,
        0.0,
        boundary,
        color=POSITIVE,
        alpha=0.12,
    )
    axis.fill_between(
        boundary,
        boundary,
        1.0,
        color=NEGATIVE,
        alpha=0.10,
    )
    axis.plot(boundary, boundary, color=INK, linewidth=2.0)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel(
        r"Global pair margin  $|m_{\mathrm{OOF}}(a,b)|$  (larger $\rightarrow$)"
    )
    axis.set_ylabel(
        r"Permitted disagreement correction  "
        r"$|\delta_q|\,|D_q(a)-D_q(b)|$  (larger $\rightarrow$)"
    )
    figure.suptitle(
        "Bounded expert-disagreement certificate",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.91,
        (
            r"$F_q(a)-F_q(b)=m_{\mathrm{OOF}}(a,b)"
            r"+\delta_q[D_q(a)-D_q(b)]$,  "
            r"$|\delta_q|\leq\Delta_{\mathrm{eff}}$"
        ),
        ha="center",
        fontsize=12,
        color=INK,
    )
    axis.text(
        0.73,
        0.27,
        "CERTIFIED STABLE\nGlobal order cannot reverse",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=POSITIVE,
    )
    axis.text(
        0.27,
        0.73,
        "ADAPTIVE BAND\nOrder may change, but need not",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=NEGATIVE,
    )
    axis.annotate(
        r"certificate boundary:  $|m_{\mathrm{OOF}}|"
        r"=|\delta_q|\,|\Delta D_q|$",
        xy=(0.60, 0.60),
        xytext=(0.69, 0.78),
        arrowprops={"arrowstyle": "->", "color": MUTED},
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    axis.text(
        0.04,
        0.035,
        r"If $D_q(a)=D_q(b)$, allocation cannot change the pair.",
        ha="left",
        va="bottom",
        fontsize=11,
        color=INK,
    )
    _clean_axis(axis, grid_axis="both")
    _add_footer(
        figure,
        "Formal CEARF-N property · schematic axes, not empirical values · "
        "the certificate does not imply Recall/nDCG improvement.",
    )
    outputs.extend(
        _save(
            figure,
            "17-bounded-pair-certificate",
            top=0.86,
            bottom=0.14,
        )
    )
    return outputs


def _validate_primary_allocation_runs(
    runs_by_domain: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Validate the final six-way allocation comparison and its boundary."""
    expected_modes = tuple(MODE_LABELS)
    matched_seeds: dict[str, list[int]] = {}
    for domain in DOMAIN_LABELS:
        runs = runs_by_domain.get(domain, [])
        seeds = [int(run["seed"]) for run in runs]
        if seeds != list(PRIMARY_ALLOCATION_SEEDS):
            raise ValueError(
                f"{domain}: expected exactly seeds "
                f"{list(PRIMARY_ALLOCATION_SEEDS)}, got {seeds}"
            )
        matched_seeds[domain] = seeds
        for run in runs:
            metrics = run.get("test", {}).get("metrics", {})
            missing = [
                mode for mode in expected_modes if mode not in metrics
            ]
            if missing:
                raise ValueError(
                    f"{domain} seed={run['seed']}: missing allocation "
                    f"modes {missing}"
                )
            for mode in expected_modes:
                utility = metrics[mode].get("utility")
                if utility is None or not np.isfinite(float(utility)):
                    raise ValueError(
                        f"{domain} seed={run['seed']} mode={mode}: "
                        "missing or non-finite test utility"
                    )

            training = run.get("training", {})
            boundary_flags = [
                training.get("global", {}).get(
                    "training_uses_validation_labels"
                ),
                training.get("dynamic", {}).get(
                    "training_uses_validation_labels"
                ),
                training.get("short_long", {})
                .get("short", {})
                .get("training_uses_validation_labels"),
                training.get("short_long", {})
                .get("long", {})
                .get("training_uses_validation_labels"),
            ]
            if boundary_flags != [False, False, False, False]:
                raise ValueError(
                    f"{domain} seed={run['seed']}: allocation boundary "
                    "is not certified as validation-label-free"
                )

    return {
        "matched_seeds": matched_seeds,
        "methods": list(expected_modes),
        "metric": "utility = (Recall@6 + Recall@20) / 2",
        "aggregation": (
            "mean across matched seeds; error bars are sample SD"
        ),
        "allocation_boundary": (
            "OOF global, OOF short/long, and dynamic beta_q are fitted "
            "without validation labels"
        ),
    }


def allocation_modes(
    runs_by_domain: dict[str, list[dict[str, Any]]],
    *,
    source_path: Path = RESULTS_PATH,
) -> tuple[list[str], dict[str, Any]]:
    provenance = _validate_primary_allocation_runs(runs_by_domain)
    provenance["source"] = str(source_path.resolve())
    provenance["source_sha256"] = _sha256_file(source_path)

    figure, axes = plt.subplots(1, 3, figsize=(16, 8), constrained_layout=False)
    figure.suptitle(
        r"RQ1 — From expert endpoints to query-wise dynamic rank fusion",
        fontweight="bold",
    )

    for axis, (domain, label) in zip(axes, DOMAIN_LABELS.items()):
        runs = runs_by_domain[domain]
        means, errors = [], []
        for mode in MODE_LABELS:
            mean, error = _aggregate_metric(runs, mode, "utility")
            means.append(mean)
            errors.append(error)
        y = np.arange(len(MODE_LABELS))
        bars = axis.barh(
            y,
            means,
            xerr=errors,
            capsize=4,
            color=list(MODE_COLORS.values()),
            edgecolor=WHITE,
            linewidth=0.8,
        )
        upper = (
            max(
                mean + error
                for mean, error in zip(means, errors)
            )
            * 1.22
            if means and np.isfinite(means).all()
            else 1.0
        )
        axis.set_xlim(0, upper)
        axis.set_yticks(y)
        axis.set_yticklabels(
            [
                "Memory-only",
                "Neural-only",
                "Fixed\n" + r"$\beta=.5$",
                "OOF global",
                "OOF\nshort/long",
                "Dynamic\n" + r"$\beta_q$",
            ]
        )
        axis.invert_yaxis()
        axis.set_title(f"{label}\n3 matched seeds")
        axis.set_xlabel(
            r"Test utility $\frac{R@6+R@20}{2}$"
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.5f}" for value in means],
            padding=4,
            fontsize=8.5,
        )
        _clean_axis(axis, grid_axis="x")

    figure.text(
        0.5,
        0.885,
        (
            "Same frozen expert ranks in every row; OOF allocators use "
            "training labels only. Bars start at zero; whiskers are "
            "across-seed SD."
        ),
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    _add_footer(
        figure,
        f"Final source: {source_path.name} · "
        f"SHA-256 {provenance['source_sha256'][:12]}… · "
        + _seed_text(runs_by_domain),
    )
    outputs = _save(
        figure,
        "01-allocation-modes-utility",
        top=0.80,
        bottom=0.12,
    )
    return outputs, provenance


def _load_fusion_ablation_recall20(
    runs_by_domain: dict[str, list[dict[str, Any]]],
    *,
    source_path: Path = RESULTS_PATH,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
]:
    """Validate the matched-seed endpoint/fusion Recall@20 ablation."""
    data: dict[str, dict[str, dict[str, Any]]] = {}
    provenance: dict[str, Any] = {
        "source": str(source_path.resolve()),
        "source_sha256": _sha256_file(source_path),
        "metric": "recall@20",
        "matched_seeds": list(FUSION_ABLATION_SEEDS),
        "aggregation": (
            "mean across matched seeds; error bars are sample SD"
        ),
        "modes": list(FUSION_ABLATION_MODES),
        "domains": {},
    }

    for domain in DOMAIN_LABELS:
        runs = runs_by_domain.get(domain, [])
        seeds = tuple(int(run["seed"]) for run in runs)
        if seeds != FUSION_ABLATION_SEEDS:
            raise ValueError(
                f"{domain}: seeds are {list(seeds)}, expected exactly "
                f"{list(FUSION_ABLATION_SEEDS)}"
            )

        data[domain] = {}
        query_counts: dict[int, set[int]] = {
            seed: set() for seed in seeds
        }
        for mode in FUSION_ABLATION_MODES:
            values: list[float] = []
            for run in runs:
                seed = int(run["seed"])
                mode_metrics = (
                    run.get("test", {}).get("metrics", {}).get(mode)
                )
                if not isinstance(mode_metrics, dict):
                    raise ValueError(
                        f"{domain} seed={seed}: missing mode {mode}"
                    )
                value = float(mode_metrics.get("recall@20", math.nan))
                if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"{domain} seed={seed} mode={mode}: missing or "
                        "invalid recall@20"
                    )
                query_count = int(mode_metrics.get("n", 0))
                if query_count <= 0:
                    raise ValueError(
                        f"{domain} seed={seed} mode={mode}: positive "
                        "test-query count is required"
                    )
                query_counts[seed].add(query_count)
                values.append(value)

            data[domain][mode] = _metric_record(
                values,
                seeds=seeds,
                source=str(source_path.resolve()),
                derivation=(
                    "matched-seed test Recall@20 read directly from "
                    "dynamic_beta_trainonly_v2_results.json"
                ),
            )

        inconsistent_counts = {
            seed: sorted(counts)
            for seed, counts in query_counts.items()
            if len(counts) != 1
        }
        if inconsistent_counts:
            raise ValueError(
                f"{domain}: endpoint/fusion query counts differ within "
                f"seed: {inconsistent_counts}"
            )

        endpoint = max(
            ("memory_only", "neural_only"),
            key=lambda mode: data[domain][mode]["mean"],
        )
        best_endpoint = float(data[domain][endpoint]["mean"])
        fused = float(data[domain]["dynamic"]["mean"])
        gain = fused - best_endpoint
        relative_gain = gain / best_endpoint if best_endpoint > 0.0 else math.nan
        provenance["domains"][domain] = {
            "seeds": list(seeds),
            "test_query_count_by_seed": {
                str(seed): next(iter(query_counts[seed]))
                for seed in seeds
            },
            "statistics": data[domain],
            "best_endpoint": endpoint,
            "absolute_endpoint_gain": gain,
            "relative_endpoint_gain": relative_gain,
            "endpoint_gain_is_positive": gain > 0.0,
        }

    return data, provenance


def fusion_ablation_recall20(
    runs_by_domain: dict[str, list[dict[str, Any]]],
    *,
    source_path: Path = RESULTS_PATH,
) -> tuple[list[str], dict[str, Any]]:
    """Plot memory-only, neural-only, and dynamic fusion Recall@20."""
    data, provenance = _load_fusion_ablation_recall20(
        runs_by_domain,
        source_path=source_path,
    )
    figure, axis = plt.subplots(figsize=(16, 8))
    figure.suptitle(
        "Fusion ablation: expert endpoints vs dynamic CEARF-N",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.895,
        (
            "Recall@20 on exactly matched seeds 42/123/456; whiskers are "
            "sample SD and every label reports mean ± SD."
        ),
        ha="center",
        fontsize=11,
        color=MUTED,
    )

    x = np.arange(len(DOMAIN_LABELS), dtype=np.float64)
    width = 0.23
    offsets = (-width, 0.0, width)
    maximum = max(
        float(data[domain][mode]["mean"])
        + float(data[domain][mode]["std"])
        for domain in DOMAIN_LABELS
        for mode in FUSION_ABLATION_MODES
    )
    headroom = max(maximum * 0.18, 0.035)

    for offset, (mode, (label, color)) in zip(
        offsets, FUSION_ABLATION_MODES.items()
    ):
        means = [
            float(data[domain][mode]["mean"])
            for domain in DOMAIN_LABELS
        ]
        errors = [
            float(data[domain][mode]["std"])
            for domain in DOMAIN_LABELS
        ]
        bars = axis.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            color=color,
            edgecolor=WHITE,
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        for bar, mean, error in zip(bars, means, errors):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                mean + error + maximum * 0.012,
                f"{mean:.5f}\n±{error:.5f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color=INK,
            )

    for position, domain in enumerate(DOMAIN_LABELS):
        domain_provenance = provenance["domains"][domain]
        gain = float(domain_provenance["absolute_endpoint_gain"])
        relative = float(domain_provenance["relative_endpoint_gain"])
        fused = float(data[domain]["dynamic"]["mean"])
        fused_sd = float(data[domain]["dynamic"]["std"])
        is_positive = bool(
            domain_provenance["endpoint_gain_is_positive"]
        )
        marker = "↑" if is_positive else "↓"
        color = "#16733A" if is_positive else NEGATIVE
        axis.annotate(
            (
                f"{marker} {gain:+.5f} vs best endpoint"
                f"\n({relative:+.1%})"
            ),
            xy=(position + width, fused + fused_sd),
            xytext=(
                position + width,
                fused + fused_sd + headroom * 0.58,
            ),
            ha="center",
            va="bottom",
            fontsize=9.4,
            color=color,
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 1.4,
            },
        )

    axis.set_xticks(x)
    axis.set_xticklabels(list(DOMAIN_LABELS.values()))
    axis.set_ylabel("Recall@20")
    axis.set_ylim(0.0, maximum + headroom)
    axis.legend(
        loc="upper left",
        frameon=False,
        ncol=1,
    )
    _clean_axis(axis)
    _add_footer(
        figure,
        f"Only source: {source_path.name} · SHA-256 "
        f"{provenance['source_sha256'][:12]}… · exact matched seeds "
        + ",".join(map(str, provenance["matched_seeds"]))
        + " · endpoint gain = fused mean − max(memory, neural) mean.",
    )
    return (
        _save(
            figure,
            "14-fusion-ablation-recall20",
            top=0.80,
            bottom=0.13,
        ),
        provenance,
    )


def paired_delta(summary: dict[str, Any]) -> list[str]:
    metrics = OrderedDict(
        [
            ("recall@6", "Recall@6"),
            ("recall@20", "Recall@20"),
            ("utility", "Utility"),
        ]
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
    figure.suptitle(
        r"Dynamic $\beta_q$ minus training-only global $\beta$",
        fontweight="bold",
    )
    domain_order = list(DOMAIN_LABELS)
    y = np.arange(len(domain_order))

    for axis, (metric, metric_label) in zip(axes, metrics.items()):
        plotted = []
        for row, domain in enumerate(domain_order):
            item = (
                summary.get("domains", {})
                .get(domain, {})
                .get("paired", {})
                .get("oof_global", {})
                .get(metric)
            )
            if not item:
                continue
            difference = float(item["difference"]) * 1000.0
            ci = item.get("cluster_bootstrap_ci95")
            color = DOMAIN_COLORS[domain]
            if ci and len(ci) == 2:
                low, high = (float(ci[0]) * 1000.0, float(ci[1]) * 1000.0)
                axis.errorbar(
                    difference,
                    row,
                    xerr=[[difference - low], [high - difference]],
                    fmt="o",
                    markersize=8,
                    capsize=5,
                    color=color,
                    ecolor=color,
                    linewidth=2,
                )
            else:
                axis.plot(difference, row, "o", markersize=8, color=color)
            plotted.append((row, difference))
        axis.axvline(0, color=INK, linewidth=1.2)
        axis.set_title(metric_label)
        axis.set_xlabel(r"Paired difference $\times 10^3$")
        axis.set_yticks(y)
        axis.set_yticklabels([DOMAIN_LABELS[domain] for domain in domain_order])
        axis.invert_yaxis()
        _clean_axis(axis, grid_axis="x")
        xmin, xmax = axis.get_xlim()
        span = max(xmax - xmin, 1e-9)
        for row, difference in plotted:
            offset = 0.025 * span
            axis.text(
                difference + (offset if difference >= 0 else -offset),
                row - 0.14,
                f"{difference:+.2f}",
                ha="left" if difference >= 0 else "right",
                va="center",
                fontsize=9.5,
                color=INK,
            )

    figure.text(
        0.5,
        0.895,
        "Points are paired query-level effects; whiskers are query-level bootstrap 95% CIs.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    _add_footer(
        figure,
        "Source: dynamic_beta_summary.json · " + _summary_seed_text(summary),
    )
    return _save(figure, "02-dynamic-vs-global-paired-delta", top=0.82)


def rescue_damage(
    runs_by_domain: dict[str, list[dict[str, Any]]]
) -> list[str]:
    figure, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=False)
    figure.suptitle(
        "Query-level rescues and damage relative to memory-only",
        fontweight="bold",
    )
    compared_modes = OrderedDict(
        [("oof_global", "OOF global"), ("dynamic", r"Dynamic $\beta_q$")]
    )

    for axis, (domain, label) in zip(axes, DOMAIN_LABELS.items()):
        runs = runs_by_domain[domain]
        x = np.arange(len(compared_modes))
        rescues, damages = [], []
        for mode in compared_modes:
            rescues.append(
                100.0
                * _mean(
                    run["test"]["metrics"][mode]["rescue_rate"]
                    for run in runs
                )
            )
            damages.append(
                -100.0
                * _mean(
                    run["test"]["metrics"][mode]["damage_rate"]
                    for run in runs
                )
            )
        width = 0.34
        rescue_bars = axis.bar(
            x - width / 2,
            rescues,
            width,
            color=POSITIVE,
            label="Rescued target",
        )
        damage_bars = axis.bar(
            x + width / 2,
            damages,
            width,
            color=NEGATIVE,
            label="Damaged target",
        )
        axis.axhline(0, color=INK, linewidth=1.2)
        axis.set_xticks(x)
        axis.set_xticklabels(list(compared_modes.values()))
        axis.set_title(f"{label}\n{len(runs)} completed seed(s)")
        axis.set_ylabel("Share of test queries (%)" if axis is axes[0] else "")
        axis.bar_label(rescue_bars, labels=[f"{value:.2f}%" for value in rescues], padding=3)
        axis.bar_label(
            damage_bars,
            labels=[f"{abs(value):.2f}%" for value in damages],
            padding=3,
        )
        for index, (rescue, damage) in enumerate(zip(rescues, damages)):
            axis.text(
                index,
                min(damages) * 1.24,
                f"net {10.0 * (rescue + damage):+.1f}/1k",
                ha="center",
                va="top",
                fontsize=9.2,
                color=MUTED,
            )
        extent = max(max(rescues, default=0), abs(min(damages, default=0)), 0.1)
        axis.set_ylim(-extent * 1.58, extent * 1.35)
        _clean_axis(axis)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=2,
        frameon=False,
    )
    _add_footer(
        figure,
        "Positive = memory miss changed to hit; negative = memory hit changed to miss · "
        + _seed_text(runs_by_domain),
    )
    return _save(figure, "03-rescue-damage-vs-memory", top=0.77)


def beta_distribution(
    runs_by_domain: dict[str, list[dict[str, Any]]]
) -> list[str]:
    figure, axis = plt.subplots(1, 1, figsize=(16, 8))
    figure.suptitle(
        r"Learned test-time distribution of continuous $\beta_q$",
        fontweight="bold",
    )
    y = np.arange(len(DOMAIN_LABELS))

    for row, (domain, label) in enumerate(DOMAIN_LABELS.items()):
        runs = runs_by_domain[domain]
        beta_records = [
            run["test"]["metrics"]["dynamic"].get("beta", {}) for run in runs
        ]
        q10 = _mean(item["q10"] for item in beta_records)
        q25 = _mean(item["q25"] for item in beta_records)
        median = _mean(item["median"] for item in beta_records)
        q75 = _mean(item["q75"] for item in beta_records)
        q90 = _mean(item["q90"] for item in beta_records)
        color = DOMAIN_COLORS[domain]
        axis.plot([q10, q90], [row, row], color=color, linewidth=4, alpha=0.55)
        axis.plot([q25, q75], [row, row], color=color, linewidth=13, solid_capstyle="butt")
        axis.scatter([median], [row], s=105, color=WHITE, edgecolor=color, linewidth=2.5, zorder=3)
        axis.text(
            0.985,
            row,
            f"q10 / med / q90 = {q10:.3f} / {median:.3f} / {q90:.3f}",
            ha="right",
            va="center",
            fontsize=10,
            color=MUTED,
        )

    axis.set_xlim(0, 1)
    axis.set_xlabel(r"Neural allocation $\beta_q$ (full admissible range)")
    axis.set_yticks(y)
    axis.set_yticklabels(
        [
            f"{label}\n{len(runs_by_domain[domain])} seed(s)"
            for domain, label in DOMAIN_LABELS.items()
        ]
    )
    axis.invert_yaxis()
    axis.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
    axis.text(
        0.5,
        0.98,
        "equal allocation",
        ha="center",
        va="top",
        color=MUTED,
        transform=axis.get_xaxis_transform(),
    )
    _clean_axis(axis, grid_axis="x")
    figure.text(
        0.5,
        0.89,
        "Thin segment: q10–q90 · thick segment: q25–q75 · circle: median.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    _add_footer(
        figure,
        f"Source: {RESULTS_PATH.name} · axis intentionally spans β=0…1 · "
        + _seed_text(runs_by_domain),
    )
    return _save(figure, "04-dynamic-beta-distribution", top=0.82)


def _quantile_bin_effect(
    values: np.ndarray, beta: np.ndarray, bins: int = 4
) -> np.ndarray:
    """Return within-run mean-beta effects in ordered quantile groups."""
    values = np.asarray(values, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    groups = np.empty(len(values), dtype=np.int8)
    groups[order] = np.minimum(
        (np.arange(len(values), dtype=np.int64) * bins) // max(len(values), 1),
        bins - 1,
    )
    baseline = float(np.mean(beta))
    effects = np.full(bins, np.nan, dtype=np.float64)
    for group in range(bins):
        mask = groups == group
        if np.any(mask):
            effects[group] = float(np.mean(beta[mask]) - baseline)
    return effects


def _context_effects(
    runs_by_domain: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    effects: dict[str, dict[str, np.ndarray]] = {}
    artifact_counts: dict[str, int] = {}
    for domain, runs in runs_by_domain.items():
        length_effects = []
        frequency_effects = []
        tail_effects = []
        for run in runs:
            artifact_text = run.get("rank_artifact")
            if not artifact_text:
                continue
            artifact = Path(artifact_text)
            if not artifact.exists():
                continue
            with np.load(artifact, allow_pickle=False) as bundle:
                if "test_features" not in bundle or "test_dynamic_beta" not in bundle:
                    continue
                features = np.asarray(bundle["test_features"], dtype=np.float64)
                beta = np.asarray(bundle["test_dynamic_beta"], dtype=np.float64)
            if features.shape[1] < 3 or len(features) != len(beta):
                continue
            length_effects.append(_quantile_bin_effect(features[:, 0], beta))
            frequency_effects.append(_quantile_bin_effect(features[:, 1], beta))
            baseline = float(np.mean(beta))
            tail_indicator = features[:, 2] >= 0.5
            if np.any(~tail_indicator) and np.any(tail_indicator):
                tail_effects.append(
                    np.asarray(
                        [
                            float(np.mean(beta[~tail_indicator]) - baseline),
                            float(np.mean(beta[tail_indicator]) - baseline),
                        ]
                    )
                )
        artifact_counts[domain] = len(length_effects)
        effects[domain] = {
            "length": np.nanmean(length_effects, axis=0)
            if length_effects
            else np.full(4, np.nan),
            "frequency": np.nanmean(frequency_effects, axis=0)
            if frequency_effects
            else np.full(4, np.nan),
            "tail": np.nanmean(tail_effects, axis=0)
            if tail_effects
            else np.full(2, np.nan),
        }
    return effects, artifact_counts


def beta_context_behavior(
    runs_by_domain: dict[str, list[dict[str, Any]]]
) -> list[str]:
    effects, artifact_counts = _context_effects(runs_by_domain)
    figure, axes = plt.subplots(1, 3, figsize=(16, 8))
    figure.suptitle(
        r"How query context moves $\beta_q$ around its domain mean",
        fontweight="bold",
    )
    panels = [
        ("length", ["Q1\nshortest", "Q2", "Q3", "Q4\nlongest"], "Context length"),
        (
            "frequency",
            ["Q1\nrarest", "Q2", "Q3", "Q4\nmost frequent"],
            "Last-item frequency",
        ),
        ("tail", ["Head item", "Tail item"], "Last-item segment"),
    ]

    all_values = [
        value
        for domain in effects.values()
        for feature in domain.values()
        for value in feature
        if np.isfinite(value)
    ]
    y_extent = max(max(map(abs, all_values), default=0.005) * 1.35, 0.005)

    for axis, (key, categories, title) in zip(axes, panels):
        x = np.arange(len(categories))
        for domain, label in DOMAIN_LABELS.items():
            values = effects[domain][key]
            axis.plot(
                x,
                values,
                marker="o",
                markersize=7,
                linewidth=2.2,
                color=DOMAIN_COLORS[domain],
                label=label,
            )
        axis.axhline(0, color=INK, linewidth=1.1)
        axis.set_xticks(x)
        axis.set_xticklabels(categories)
        axis.set_ylim(-y_extent, y_extent)
        axis.set_title(title)
        axis.set_ylabel(r"Mean $\beta_q$ minus domain mean" if axis is axes[0] else "")
        _clean_axis(axis)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        ncol=3,
        frameon=False,
    )
    artifact_status = " · ".join(
        f"{DOMAIN_LABELS[domain]} n={count} artifact seed(s)"
        for domain, count in artifact_counts.items()
    )
    _add_footer(
        figure,
        "Within-domain equal-count quartiles; effects averaged equally across seeds · "
        + artifact_status,
    )
    return _save(figure, "05-dynamic-beta-context-behavior", top=0.77)


def rrf_sensitivity(
    runs_by_domain: dict[str, list[dict[str, Any]]]
) -> list[str]:
    keys = OrderedDict([("rrf_k10", "10"), ("rrf_k20", "20"), ("rrf_k60", "60")])
    figure, axes = plt.subplots(1, 3, figsize=(16, 8))
    figure.suptitle(
        "Reciprocal-rank constant sensitivity",
        fontweight="bold",
    )
    sensitivity_status = []

    for axis, (domain, label) in zip(axes, DOMAIN_LABELS.items()):
        eligible = [
            run
            for run in runs_by_domain[domain]
            if all(
                key in run.get("test", {}).get("fusion_sensitivity", {})
                for key in keys
            )
        ]
        sensitivity_status.append(
            f"{label} n={len(eligible)} ({','.join(str(run['seed']) for run in eligible) or 'none'})"
        )
        means = [
            _mean(
                run["test"]["fusion_sensitivity"][key]["utility"]
                for run in eligible
            )
            for key in keys
        ]
        errors = [
            _std(
                run["test"]["fusion_sensitivity"][key]["utility"]
                for run in eligible
            )
            for key in keys
        ]
        x = np.arange(len(keys))
        bars = axis.bar(
            x,
            means,
            yerr=errors if len(eligible) > 1 else None,
            capsize=4,
            color=["#9EB8D1", "#2B6CB0", "#7E93A8"],
        )
        upper = max(means) * 1.22 if means and np.isfinite(means).any() else 1.0
        axis.set_ylim(0, upper)
        axis.set_xticks(x)
        axis.set_xticklabels(list(keys.values()))
        axis.set_xlabel("RRF constant k")
        axis.set_ylabel(r"Utility $=\frac{R@6+R@20}{2}$" if axis is axes[0] else "")
        axis.set_title(f"{label}\n{len(eligible)} sensitivity seed(s)")
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in means], padding=4)
        _clean_axis(axis)

    figure.text(
        0.5,
        0.855,
        "All bars start at zero; k is changed and the continuous gate is refit on OOF training queries.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    _add_footer(
        figure,
        f"Source: {RESULTS_PATH.name} · " + " · ".join(sensitivity_status),
    )
    return _save(figure, "06-rrf-k-sensitivity", top=0.77)


def _find_inference_benchmark() -> Path | None:
    candidates = [
        BENCH / "dynamic_beta_inference_benchmark.json",
        BENCH / "cearfn_dynamic_inference_benchmark.json",
        BENCH / "cearfn_inference_benchmark.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def inference_performance(path: Path) -> list[str]:
    benchmark = _read_json(path)
    domains = benchmark.get("domains", {})
    available = [domain for domain in DOMAIN_LABELS if domain in domains]
    if not available:
        return []

    if all("timing" in domains[domain] for domain in available):
        return dynamic_inference_performance(path, benchmark, available)

    component_keys = OrderedDict(
        [
            ("memory_seconds", ("Memory retrieval", "#7F8C8D")),
            ("neural_seconds", ("Neural ranking", "#2B6CB0")),
            ("router_feature_seconds", ("Gate features", "#4FA3A5")),
            ("router_beta_seconds", (r"$\beta_q$ assignment", "#9B7EBD")),
            ("fusion_seconds", ("Rank fusion", "#E6A04B")),
        ]
    )
    figure, axis = plt.subplots(1, 1, figsize=(16, 8))
    figure.suptitle("Warm-state inference cost by component", fontweight="bold")
    y = np.arange(len(available))
    left = np.zeros(len(available), dtype=np.float64)

    for component, (label, color) in component_keys.items():
        values = np.asarray(
            [
                1000.0
                * float(domains[domain].get(component, 0.0))
                / max(int(domains[domain].get("n_queries", 0)), 1)
                for domain in available
            ]
        )
        axis.barh(y, values, left=left, color=color, label=label, height=0.58)
        left += values

    for row, domain in enumerate(available):
        reported_total = float(
            domains[domain].get("amortized_milliseconds_per_query", left[row])
        )
        throughput = float(domains[domain].get("queries_per_second", 0.0))
        axis.text(
            left[row] + max(left) * 0.015,
            row,
            f"{reported_total:.2f} ms/query · {throughput:.0f} q/s",
            va="center",
            ha="left",
            fontsize=11,
        )
    axis.set_yticks(y)
    axis.set_yticklabels([DOMAIN_LABELS[domain] for domain in available])
    axis.invert_yaxis()
    axis.set_xlim(0, max(left) * 1.42)
    axis.set_xlabel("Amortized milliseconds per query")
    _clean_axis(axis, grid_axis="x")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
        frameon=False,
    )

    protocol = benchmark.get("protocol", {})
    hardware = protocol.get("hardware", "hardware recorded in benchmark JSON")
    fallback_note = (
        "This is the existing router benchmark; regenerate after the dynamic-gate benchmark lands."
        if path.name == "cearfn_inference_benchmark.json"
        else "Dynamic-gate inference benchmark."
    )
    _add_footer(
        figure,
        f"Source: {path.name} · {hardware} · {fallback_note}",
    )
    return _save(figure, "07-inference-performance")


def dynamic_inference_performance(
    path: Path,
    benchmark: dict[str, Any],
    available: list[str],
) -> list[str]:
    """Render the new dynamic-gate timing schema without co-timing overclaim."""
    domains = benchmark["domains"]
    figure, (total_axis, post_axis) = plt.subplots(
        1,
        2,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [1.18, 1]},
    )
    figure.suptitle(
        "Warm-state inference: expert-inclusive estimate and measured dynamic post-processing",
        fontweight="bold",
    )
    y = np.arange(len(available))

    total_components = OrderedDict(
        [
            ("memory_seconds_from_reference_run", ("Memory retrieval", "#7F8C8D")),
            ("pasgr_seconds_from_reference_run", ("PASGR ranking", "#2B6CB0")),
            (
                "new_dynamic_postprocessing_median_seconds",
                ("Dynamic post-processing", "#4FA3A5"),
            ),
        ]
    )
    left = np.zeros(len(available), dtype=np.float64)
    for key, (label, color) in total_components.items():
        values = np.asarray(
            [
                1000.0
                * float(
                    domains[domain]
                    .get("expert_inclusive_cross_run_estimate", {})
                    .get(key, 0.0)
                )
                / max(int(domains[domain].get("n_queries", 0)), 1)
                for domain in available
            ]
        )
        total_axis.barh(y, values, left=left, color=color, label=label, height=0.58)
        left += values

    for row, domain in enumerate(available):
        estimate = domains[domain].get("expert_inclusive_cross_run_estimate", {})
        total_ms = float(estimate.get("estimated_milliseconds_per_query", left[row]))
        throughput = float(estimate.get("estimated_queries_per_second", 0.0))
        total_axis.text(
            left[row] + max(left) * 0.02,
            row,
            f"{total_ms:.2f} ms · {throughput:.0f} q/s",
            va="center",
            ha="left",
            fontsize=10.5,
        )
    total_axis.set_yticks(y)
    total_axis.set_yticklabels([DOMAIN_LABELS[domain] for domain in available])
    total_axis.invert_yaxis()
    total_axis.set_xlim(0, max(left) * 1.46)
    total_axis.set_xlabel("Estimated ms/query")
    total_axis.set_title("Expert-inclusive cross-run composition")
    _clean_axis(total_axis, grid_axis="x")
    total_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )

    post_components = OrderedDict(
        [
            (
                "context_features",
                ("Context features", "#7F8C8D"),
            ),
            (
                "bounded_linear_gate",
                (r"Bounded linear $\beta_q$", "#9B7EBD"),
            ),
            (
                "rrf_fusion",
                ("RRF fusion", "#4FA3A5"),
            ),
        ]
    )
    left_microseconds = np.zeros(len(available), dtype=np.float64)
    for key, (label, color) in post_components.items():
        values = np.asarray(
            [
                float(
                    domains[domain]
                    .get("timing", {})
                    .get(key, {})
                    .get("median_microseconds_per_query", 0.0)
                )
                for domain in available
            ]
        )
        post_axis.barh(
            y,
            values,
            left=left_microseconds,
            color=color,
            label=label,
            height=0.58,
        )
        left_microseconds += values

    for row, domain in enumerate(available):
        headline = domains[domain].get("headline", {})
        post = float(
            headline.get(
                "dynamic_postprocessing_including_rrf_microseconds_per_query",
                left_microseconds[row],
            )
        )
        allocation = float(
            headline.get(
                "dynamic_allocation_feature_plus_gate_microseconds_per_query",
                0.0,
            )
        )
        post_axis.text(
            left_microseconds[row] + max(left_microseconds) * 0.02,
            row,
            f"{post:.1f} μs total\n{allocation:.2f} μs allocation",
            va="center",
            ha="left",
            fontsize=9.5,
        )
    post_axis.set_yticks(y)
    post_axis.set_yticklabels([DOMAIN_LABELS[domain] for domain in available])
    post_axis.invert_yaxis()
    post_axis.set_xlim(0, max(left_microseconds) * 1.48)
    post_axis.set_xlabel("Measured μs/query")
    post_axis.set_title("Dynamic post-processing (median of repetitions)")
    _clean_axis(post_axis, grid_axis="x")
    post_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )

    protocol = benchmark.get("protocol", {})
    hardware = protocol.get("hardware", "hardware recorded in benchmark JSON")
    repetitions = protocol.get("repetitions", "recorded")
    _add_footer(
        figure,
        f"Source: {path.name} · {hardware} · {repetitions} timing repetitions · "
        "left panel composes separately timed warm-state experts and is explicitly an estimate.",
    )
    return _save(figure, "07-inference-performance", top=0.76)


EXTERNAL_METHOD_ORDER = OrderedDict(
    [
        ("transition", "Transition"),
        ("vsknn", "V-SKNN"),
        ("stan", "STAN"),
        ("GRU4Rec", "GRU4Rec"),
        ("SASRec", "SASRec"),
        ("NARM", "NARM"),
        ("SR-GNN", "SR-GNN"),
        ("SIGMA-compatible", "SIGMA-compatible"),
        ("CEARF-N", r"CEARF-N dynamic $\beta_q$"),
    ]
)
AMAZON_ARTIFACT_PREFIX = {
    "Video_Games": "video_games",
    "Baby_Products": "baby_products",
}
AMAZON_NEURAL_SLUGS = OrderedDict(
    [
        ("GRU4Rec", "gru4rec"),
        ("SASRec", "sasrec"),
        ("NARM", "narm"),
        ("SR-GNN", "sr_gnn"),
        ("SIGMA-compatible", "sigma_compatible"),
    ]
)


def _metric_record(
    values: Iterable[float],
    *,
    seeds: Iterable[int],
    source: str,
    derivation: str,
) -> dict[str, Any]:
    value_list = [float(value) for value in values]
    seed_list = [int(seed) for seed in seeds]
    return {
        "mean": _mean(value_list),
        "std": _std(value_list),
        "values": value_list,
        "seeds": seed_list,
        "n_seeds": len(seed_list),
        "source": source,
        "derivation": derivation,
    }


def _load_amazon_rank_baselines(
    domain: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    prefix = AMAZON_ARTIFACT_PREFIX[domain]
    records: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    for method, slug in AMAZON_NEURAL_SLUGS.items():
        pattern = f"{prefix}_full_{slug}_seed*_ranks.npz"
        files = sorted(
            AMAZON_NEURAL_ARTIFACT_DIR.glob(pattern),
            key=lambda path: int(
                re.search(r"_seed(\d+)_ranks\.npz$", path.name).group(1)
            ),
        )
        values: list[float] = []
        seeds: list[int] = []
        fingerprints: set[str] = set()
        used_files: list[str] = []
        for artifact in files:
            match = re.search(r"_seed(\d+)_ranks\.npz$", artifact.name)
            if match is None:
                continue
            with np.load(artifact, allow_pickle=False) as bundle:
                if "ranks" not in bundle:
                    continue
                ranks = np.asarray(bundle["ranks"])
                values.append(
                    float(np.mean((ranks > 0) & (ranks <= 20)))
                )
                if "test_fingerprint" in bundle:
                    fingerprints.add(str(bundle["test_fingerprint"].item()))
            seeds.append(int(match.group(1)))
            used_files.append(str(artifact))
        if not values:
            continue
        if len(fingerprints) > 1:
            raise RuntimeError(
                f"{method} {domain}: rank artifacts have inconsistent test "
                f"fingerprints: {sorted(fingerprints)}"
            )
        records[method] = _metric_record(
            values,
            seeds=seeds,
            source=str(AMAZON_NEURAL_ARTIFACT_DIR),
            derivation=(
                "Recall@20 recomputed as mean(0 < stored target rank <= 20)"
            ),
        )
        provenance[method] = {
            "files": used_files,
            "test_fingerprint": next(iter(fingerprints), None),
            "seeds": seeds,
            "derivation": records[method]["derivation"],
            "information_budget": "ID-only",
        }
    return records, provenance


def _load_external_baselines(
    summary: dict[str, Any],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    neighborhood = _read_json(NEIGHBORHOOD_BASELINE_PATH)
    digi_neural = _read_json(DIGI_NEURAL_BASELINE_PATH)
    data: dict[str, dict[str, dict[str, Any]]] = {
        domain: {} for domain in DOMAIN_LABELS
    }
    provenance: dict[str, Any] = {
        "neighborhood_baselines": {
            "source": str(NEIGHBORHOOD_BASELINE_PATH),
            "methods": ["V-SKNN", "STAN", "Transition"],
            "information_budget": "ID-only",
        },
        "amazon_neural_baselines": {
            "source_directory": str(AMAZON_NEURAL_ARTIFACT_DIR),
            "derivation": (
                "Recall@20 recomputed directly from persisted target-rank "
                "arrays; zero denotes target absent from top-20."
            ),
            "domains": {},
            "information_budget": "ID-only",
        },
        "diginetica_neural_baselines": {
            "source": str(DIGI_NEURAL_BASELINE_PATH),
            "derivation": "read aggregate Recall@20 mean/SD",
            "information_budget": "ID-only",
        },
        "cearfn_dynamic": {
            "source": str(SUMMARY_PATH),
            "derivation": "aggregate.dynamic.recall@20",
            "information_budget_by_domain": {
                "Video_Games": (
                    "cached 384-d teacher documented as E5-small"
                ),
                "Baby_Products": "128-d TF-IDF/SVD item-text teacher",
                "Diginetica_HID": (
                    "128-d TF-IDF/SVD product-name teacher"
                ),
            },
        },
        "metadata_disclosure": (
            "CEARF-N uses a cached text teacher on every domain "
            "(documented E5-small on Video; TF-IDF/SVD on Baby and "
            "Diginetica), "
            "while every external comparator in this chart is ID-only. "
            "This is a system comparison, not metadata-matched "
            "architecture attribution."
        ),
    }

    for domain in DOMAIN_LABELS:
        domain_methods = neighborhood.get(domain, {}).get("methods", {})
        for method in ("transition", "vsknn", "stan"):
            block = domain_methods.get(method)
            if block is None:
                continue
            per_seed = block.get("per_seed", {})
            ordered_seeds = sorted(int(seed) for seed in per_seed)
            values = [
                float(per_seed[str(seed)]["recall@20"])
                for seed in ordered_seeds
            ]
            if not values and "test" in block:
                values = [float(block["test"]["recall@20"])]
                ordered_seeds = []
            data[domain][method] = _metric_record(
                values,
                seeds=ordered_seeds,
                source=str(NEIGHBORHOOD_BASELINE_PATH),
                derivation=(
                    "read per-seed Recall@20; deterministic methods repeat "
                    "the same ranking across reported seed IDs"
                ),
            )

    for domain in AMAZON_ARTIFACT_PREFIX:
        amazon_records, amazon_provenance = _load_amazon_rank_baselines(
            domain
        )
        data[domain].update(amazon_records)
        provenance["amazon_neural_baselines"]["domains"][
            domain
        ] = amazon_provenance

    digi_models = (
        digi_neural.get("Diginetica_HID", {}).get("models", {})
    )
    digi_provenance: dict[str, Any] = {}
    for method in AMAZON_NEURAL_SLUGS:
        model = digi_models.get(method)
        if model is None:
            continue
        metric = model.get("aggregate", {}).get("recall@20", {})
        runs = model.get("runs", [])
        seeds = [int(run["seed"]) for run in runs]
        values = [
            float(run["test"]["recall@20"])
            for run in runs
            if "recall@20" in run.get("test", {})
        ]
        if values:
            record = _metric_record(
                values,
                seeds=seeds,
                source=str(DIGI_NEURAL_BASELINE_PATH),
                derivation="read per-run test Recall@20",
            )
        else:
            record = {
                "mean": float(metric["mean"]),
                "std": float(metric.get("std", 0.0)),
                "values": [],
                "seeds": seeds,
                "n_seeds": len(seeds),
                "source": str(DIGI_NEURAL_BASELINE_PATH),
                "derivation": "read aggregate Recall@20 mean/SD",
            }
        data["Diginetica_HID"][method] = record
        digi_provenance[method] = {
            "seeds": seeds,
            "derivation": record["derivation"],
        }
    provenance["diginetica_neural_baselines"][
        "models"
    ] = digi_provenance

    for domain in DOMAIN_LABELS:
        dynamic = (
            summary.get("domains", {})
            .get(domain, {})
            .get("aggregate", {})
            .get("dynamic", {})
            .get("recall@20")
        )
        if not dynamic:
            raise RuntimeError(
                f"{SUMMARY_PATH}: missing {domain} dynamic Recall@20"
            )
        seeds = [
            int(seed)
            for seed in summary["domains"][domain].get("seeds", [])
        ]
        values = [float(value) for value in dynamic.get("values", [])]
        data[domain]["CEARF-N"] = {
            "mean": float(dynamic["mean"]),
            "std": float(dynamic.get("std", 0.0)),
            "values": values,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "source": str(SUMMARY_PATH),
            "derivation": "read aggregate.dynamic.recall@20",
        }

    missing = {
        domain: [
            method
            for method in EXTERNAL_METHOD_ORDER
            if method not in data[domain]
        ]
        for domain in DOMAIN_LABELS
    }
    missing = {
        domain: methods for domain, methods in missing.items() if methods
    }
    if missing:
        raise RuntimeError(
            "External baseline chart is incomplete: "
            + json.dumps(missing, sort_keys=True)
        )
    return data, provenance


def external_baseline_comparison(
    summary: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    data, provenance = _load_external_baselines(summary)
    figure, axes = plt.subplots(
        1, 3, figsize=(16, 8), sharey=True, constrained_layout=False
    )
    figure.suptitle(
        "External baseline comparison on Recall@20",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.91,
        (
            "System-level disclosure: Video uses a cache documented as "
            "E5-small; Baby/Diginetica use TF-IDF/SVD; comparators are "
            "ID-only."
        ),
        ha="center",
        fontsize=11.2,
        color="#8A3B2A",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.865,
        "Blue = CEARF-N · orange = strongest external baseline · all bars start at zero.",
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )
    y = np.arange(len(EXTERNAL_METHOD_ORDER))

    for axis, (domain, label) in zip(axes, DOMAIN_LABELS.items()):
        means = [
            float(data[domain][method]["mean"])
            for method in EXTERNAL_METHOD_ORDER
        ]
        errors = [
            float(data[domain][method]["std"])
            for method in EXTERNAL_METHOD_ORDER
        ]
        external_methods = [
            method for method in EXTERNAL_METHOD_ORDER
            if method != "CEARF-N"
        ]
        best_external = max(
            external_methods,
            key=lambda method: float(data[domain][method]["mean"]),
        )
        colors = [
            (
                "#2B6CB0"
                if method == "CEARF-N"
                else "#E76F51"
                if method == best_external
                else "#929B9D"
            )
            for method in EXTERNAL_METHOD_ORDER
        ]
        bars = axis.barh(
            y,
            means,
            xerr=errors,
            capsize=3,
            color=colors,
            edgecolor=WHITE,
            linewidth=0.7,
            height=0.70,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in means],
            padding=3,
            fontsize=8.6,
        )
        dynamic_seeds = data[domain]["CEARF-N"]["seeds"]
        axis.set_title(
            f"{label}\nCEARF-N n={len(dynamic_seeds)} "
            f"({','.join(map(str, dynamic_seeds)) or 'none'})"
        )
        axis.set_xlabel("Recall@20")
        axis.set_xlim(0, max(means) * 1.28)
        axis.set_yticks(y)
        if axis is axes[0]:
            axis.set_yticklabels(list(EXTERNAL_METHOD_ORDER.values()))
        axis.invert_yaxis()
        _clean_axis(axis, grid_axis="x")

    source_text = (
        f"Sources: {NEIGHBORHOOD_BASELINE_PATH.name} · Amazon target-rank "
        f"artifacts · {DIGI_NEURAL_BASELINE_PATH.name} · "
        f"{SUMMARY_PATH.name} · error bars: across-seed SD"
    )
    _add_footer(figure, source_text)
    return (
        _save(
            figure,
            "08-external-baseline-comparison-recall20",
            top=0.79,
            bottom=0.14,
        ),
        provenance,
    )


def _load_expert_swap(
    path: Path,
) -> tuple[
    dict[str, dict[str, dict[str, dict[str, Any]]]],
    dict[str, Any],
]:
    raw = _read_json(path)
    protocol = "dynamic-beta-narm-expert-swap-v2-full-refit"
    modes = ("neural_only", "oof_global", "dynamic")
    output: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    provenance: dict[str, Any] = {
        "source": str(path),
        "protocol": protocol,
        "changed_factor": "PASGR -> ID-only NARM neural expert",
        "held_constant": (
            "CEARF memory, OOF split, continuous objective, bounded linear "
            "context gate, RRF k=20, and frozen full-catalog test protocol"
        ),
        "comparison_scope": (
            "system-level neural-expert sensitivity; Amazon is not a "
            "metadata-matched PASGR-vs-NARM architecture contrast"
        ),
        "domains": {},
    }
    for domain in DOMAIN_LABELS:
        domain_payload = raw.get(domain)
        if domain_payload is None:
            # The long expert-swap job checkpoints one completed domain at a
            # time.  Treat an in-progress artifact as incomplete so the
            # independent charts can still be regenerated.
            continue
        if domain_payload.get("protocol") != protocol:
            raise ValueError(f"{domain}: unexpected expert-swap protocol")
        runs_by_seed: dict[int, dict[str, Any]] = {}
        for run in domain_payload.get("runs", []):
            comparison = run.get("pasgr_comparison")
            if not comparison:
                continue
            reference = comparison.get("pasgr_reference", {})
            if not reference.get("test_metrics"):
                continue
            runs_by_seed[int(run["seed"])] = run
        runs = [runs_by_seed[seed] for seed in sorted(runs_by_seed)]
        if not runs:
            continue
        output[domain] = {"PASGR": {}, "NARM": {}}
        for expert in ("PASGR", "NARM"):
            for mode in modes:
                if expert == "PASGR":
                    values = [
                        float(
                            run["pasgr_comparison"]["pasgr_reference"][
                                "test_metrics"
                            ][mode]["recall@20"]
                        )
                        for run in runs
                    ]
                else:
                    values = [
                        float(
                            run["test"]["metrics"][mode]["recall@20"]
                        )
                        for run in runs
                    ]
                output[domain][expert][mode] = _metric_record(
                    values,
                    seeds=[int(run["seed"]) for run in runs],
                    source=str(path),
                    derivation=(
                        "matched-seed test Recall@20 from expert-swap run"
                    ),
                )
        provenance["domains"][domain] = {
            "seeds": [int(run["seed"]) for run in runs],
            "n_seeds": len(runs),
            "rank_artifacts": [
                run.get("rank_artifact") for run in runs
            ],
            "pasgr_rank_artifacts": [
                run["pasgr_comparison"]["pasgr_reference"].get(
                    "rank_artifact"
                )
                for run in runs
            ],
        }
    return output, provenance


def expert_swap_comparison(
    path: Path,
) -> tuple[list[str], dict[str, Any]] | None:
    data, provenance = _load_expert_swap(path)
    available = [domain for domain in DOMAIN_LABELS if domain in data]
    expected_domains = list(DOMAIN_LABELS)
    expected_seeds = [42, 123, 456]
    if available != expected_domains:
        return None
    if any(
        provenance["domains"][domain]["seeds"] != expected_seeds
        for domain in expected_domains
    ):
        return None

    figure, axes_raw = plt.subplots(
        1, len(available), figsize=(16, 8), squeeze=False
    )
    axes = list(axes_raw[0])
    modes = OrderedDict(
        [
            ("neural_only", "Neural-only"),
            ("oof_global", "OOF global"),
            ("dynamic", r"Dynamic $\beta_q$"),
        ]
    )
    experts = OrderedDict([("PASGR", "#2B6CB0"), ("NARM", "#E76F51")])
    figure.suptitle(
        "Neural-expert swap under the same training-only fusion protocol",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.90,
        (
            "Amazon is a system-sensitivity audit: PASGR may use semantic "
            "side information, whereas the swapped NARM expert is ID-only."
        ),
        ha="center",
        fontsize=11,
        color="#8A3B2A",
        fontweight="bold",
    )

    for axis, domain in zip(axes, available):
        x = np.arange(len(modes))
        width = 0.34
        for offset, (expert, color) in zip(
            (-width / 2, width / 2), experts.items()
        ):
            means = [
                data[domain][expert][mode]["mean"] for mode in modes
            ]
            errors = [
                data[domain][expert][mode]["std"] for mode in modes
            ]
            bars = axis.bar(
                x + offset,
                means,
                width,
                yerr=errors,
                capsize=3,
                color=color,
                label=expert,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.3f}" for value in means],
                padding=3,
                fontsize=8.8,
            )
        seeds = data[domain]["NARM"]["dynamic"]["seeds"]
        axis.set_title(
            f"{DOMAIN_LABELS[domain]}\nmatched n={len(seeds)} "
            f"({','.join(map(str, seeds))})"
        )
        axis.set_xticks(x)
        axis.set_xticklabels(list(modes.values()))
        maximum = max(
            data[domain][expert][mode]["mean"]
            for expert in experts
            for mode in modes
        )
        axis.set_ylim(0, maximum * 1.23)
        axis.set_ylabel("Recall@20" if axis is axes[0] else "")
        _clean_axis(axis)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        ncol=2,
        frameon=False,
    )
    _add_footer(
        figure,
        f"Source: {path.name} · error bars: matched-seed SD · "
        "only the neural expert changes; the gate is retrained on OOF "
        "training queries and NARM is freshly refit on complete training "
        "sessions for test.",
    )
    return (
        _save(
            figure,
            "09-expert-swap-pasgr-vs-narm",
            top=0.75,
            bottom=0.14,
        ),
        provenance,
    )


def _load_fusion_operator_control(
    path: Path,
) -> tuple[
    dict[str, dict[str, dict[str, dict[str, Any]]]],
    dict[str, Any],
]:
    raw = _read_json(path)
    expected_protocol = (
        "fusion-operator-control-v2-dynamic-and-equal-allocation"
    )
    if raw.get("protocol") != expected_protocol:
        raise ValueError(
            f"protocol is {raw.get('protocol')!r}, expected "
            f"{expected_protocol!r}"
        )
    operator = raw.get("operator_control", {})
    if operator.get("parameters_fit_or_selected") is not False:
        raise ValueError(
            "operator control does not certify zero refit/search/selection"
        )
    if operator.get("target_labels_used_to_form_rankings") is not False:
        raise ValueError(
            "operator control does not certify target-free ranking formation"
        )

    methods = (
        "weighted_rrf",
        "normalized_combsum",
        "fixed_05_weighted_rrf",
        "fixed_05_normalized_combsum",
    )
    metrics = ("recall@20", "ndcg@20")
    domains = raw.get("domains", {})
    missing_domains = [
        domain for domain in DOMAIN_LABELS if domain not in domains
    ]
    if missing_domains:
        raise ValueError(
            "missing domains: " + ", ".join(missing_domains)
        )

    output: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    provenance: dict[str, Any] = {
        "source": str(path),
        "protocol": expected_protocol,
        "operator_control": operator,
        "limitations": raw.get("limitations", []),
        "comparison": (
            "weighted RRF versus independently per-query, per-expert "
            "min-max normalized CombSUM"
        ),
        "allocations": (
            "each operator is evaluated under the identical frozen dynamic "
            "beta_q and, separately, under parameter-free beta=.5"
        ),
        "normalization_scope": (
            "each expert's persisted top-120; not the full catalogue"
        ),
        "domains": {},
    }
    seed_sets: list[tuple[int, ...]] = []
    for domain in DOMAIN_LABELS:
        block = domains[domain]
        runs = block.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"{domain} has no completed runs")
        seed_ids = [int(run["seed"]) for run in runs]
        if len(seed_ids) != len(set(seed_ids)):
            raise ValueError(f"{domain} contains duplicate seeds")
        runs = sorted(runs, key=lambda run: int(run["seed"]))
        seeds = [int(run["seed"]) for run in runs]
        seed_sets.append(tuple(seeds))
        output[domain] = {method: {} for method in methods}

        for run in runs:
            test_metrics = run.get("test", {}).get("metrics", {})
            for method in methods:
                if method not in test_metrics:
                    raise ValueError(
                        f"{domain} seed={run['seed']} missing {method}"
                    )
                for metric in metrics:
                    if metric not in test_metrics[method]:
                        raise ValueError(
                            f"{domain} seed={run['seed']} "
                            f"{method} missing {metric}"
                        )

        for method in methods:
            for metric in metrics:
                values = [
                    float(
                        run["test"]["metrics"][method][metric]
                    )
                    for run in runs
                ]
                output[domain][method][metric] = _metric_record(
                    values,
                    seeds=seeds,
                    source=str(path),
                    derivation=(
                        "matched-seed test metric from frozen-operator "
                        "control"
                    ),
                )
        provenance["domains"][domain] = {
            "seeds": seeds,
            "n_seeds": len(seeds),
            "rank_artifacts": [
                run.get("rank_artifact") for run in runs
            ],
            "rank_artifact_sha256": [
                run.get("rank_artifact_sha256") for run in runs
            ],
            "manifests": [
                run.get("manifest") for run in runs
            ],
            "manifest_sha256": [
                run.get("manifest_sha256") for run in runs
            ],
        }
    if len(set(seed_sets)) != 1:
        raise ValueError(
            "domain seed sets differ: "
            + ", ".join(
                f"{domain}={list(seed_set)}"
                for domain, seed_set in zip(
                    DOMAIN_LABELS, seed_sets
                )
            )
        )
    provenance["matched_seeds"] = list(seed_sets[0])
    return output, provenance


def fusion_operator_comparison(
    path: Path,
) -> tuple[list[str], dict[str, Any]]:
    data, provenance = _load_fusion_operator_control(path)
    operators = OrderedDict(
        [
            ("weighted_rrf", ("Weighted RRF", "#2B6CB0")),
            (
                "normalized_combsum",
                ("Normalized CombSUM", "#E76F51"),
            ),
        ]
    )
    allocations = OrderedDict(
        [
            ("dynamic", (r"Dynamic $\beta_q$", "")),
            ("fixed_05", (r"Equal $\beta=.5$", "fixed_05_")),
        ]
    )
    metrics = OrderedDict(
        [
            ("recall@20", "Recall@20"),
            ("ndcg@20", "nDCG@20"),
        ]
    )
    figure, axes = plt.subplots(
        2, 3, figsize=(16, 8), squeeze=False
    )
    figure.suptitle(
        "Fusion-operator control under dynamic and equal allocation",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.91,
        (
            "Each operator uses the same experts and candidate union; "
            "dynamic βq is frozen, while β=.5 is parameter-free."
        ),
        ha="center",
        fontsize=11.2,
        color=INK,
    )
    figure.text(
        0.5,
        0.865,
        (
            "CombSUM min-max normalization is independent per query and "
            "expert over persisted top-120 scores—not the full catalogue."
        ),
        ha="center",
        fontsize=10.5,
        color="#8A3B2A",
        fontweight="bold",
    )
    x = np.arange(len(allocations))
    width = 0.34

    for row, (metric, metric_label) in enumerate(metrics.items()):
        for column, (domain, domain_label) in enumerate(
            DOMAIN_LABELS.items()
        ):
            axis = axes[row, column]
            all_means = []
            all_errors = []
            for offset, (operator, (operator_label, color)) in zip(
                (-width / 2, width / 2), operators.items()
            ):
                means = [
                    data[domain][f"{prefix}{operator}"][metric]["mean"]
                    for _, prefix in allocations.values()
                ]
                errors = [
                    data[domain][f"{prefix}{operator}"][metric]["std"]
                    for _, prefix in allocations.values()
                ]
                all_means.extend(means)
                all_errors.extend(errors)
                bars = axis.bar(
                    x + offset,
                    means,
                    width,
                    yerr=errors,
                    capsize=3,
                    color=color,
                    label=operator_label,
                )
                axis.bar_label(
                    bars,
                    labels=[f"{value:.4f}" for value in means],
                    padding=3,
                    fontsize=8.3,
                )
            seeds = data[domain]["weighted_rrf"][metric]["seeds"]
            if row == 0:
                axis.set_title(
                    f"{domain_label}\nmatched n={len(seeds)} "
                    f"({','.join(map(str, seeds))})"
                )
            axis.set_xticks(x)
            axis.set_xticklabels(
                [label for label, _ in allocations.values()]
            )
            maximum = max(
                mean + error
                for mean, error in zip(all_means, all_errors)
            )
            axis.set_ylim(0, maximum * 1.28)
            if column == 0:
                axis.set_ylabel(metric_label)
            for position, (_, prefix) in enumerate(
                allocations.values()
            ):
                delta = (
                    data[domain][
                        f"{prefix}normalized_combsum"
                    ][metric]["mean"]
                    - data[domain][
                        f"{prefix}weighted_rrf"
                    ][metric]["mean"]
                )
                axis.text(
                    position,
                    maximum * 1.17,
                    f"Δ {delta:+.4f}",
                    ha="center",
                    va="top",
                    fontsize=8.3,
                    color=MUTED,
                )
            _clean_axis(axis)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles = [
            Patch(color=color, label=label)
            for label, color in operators.values()
        ]
        labels = [label for label, _ in operators.values()]
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.82),
        ncol=2,
        frameon=False,
    )
    figure.subplots_adjust(hspace=0.45, wspace=0.24)
    _add_footer(
        figure,
        f"Source: {path.name} · bars: matched-seed mean · error bars: "
        "seed SD · dynamic βq frozen; equal β=.5 · score normalization: "
        "per-query expert top-120.",
    )
    return (
        _save(
            figure,
            "10-fusion-operator-rrf-vs-combsum",
            top=0.73,
            bottom=0.13,
        ),
        provenance,
    )


def _load_allocation_control_summary(
    path: Path,
) -> tuple[
    dict[str, dict[str, dict[str, dict[str, Any]]]],
    dict[str, Any],
]:
    """Load a complete, internally consistent 3-domain control summary.

    The conditional chart must never make a partial run look complete.  This
    loader therefore requires the exact declared protocol, every requested
    method/metric, and the same canonical 42/123/456 seed set in all domains.
    It also recomputes each stored mean and sample SD from the per-seed values.
    """
    raw = _read_json(path)
    protocol = raw.get("protocol")
    if protocol != ALLOCATION_CONTROL_PROTOCOL:
        raise ValueError(
            f"protocol is {protocol!r}, expected "
            f"{ALLOCATION_CONTROL_PROTOCOL!r}"
        )

    domains = raw.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("domains must be an object")
    missing_domains = [
        domain for domain in DOMAIN_LABELS if domain not in domains
    ]
    if missing_domains:
        raise ValueError(
            "missing domains: " + ", ".join(missing_domains)
        )

    output: dict[
        str, dict[str, dict[str, dict[str, Any]]]
    ] = {}
    provenance: dict[str, Any] = {
        "source": str(path),
        "source_sha256": _sha256_file(path),
        "protocol": protocol,
        "matched_seeds": list(ALLOCATION_CONTROL_SEEDS),
        "fit_scope": (
            "all allocation policies fit only on frozen OOF training ranks; "
            "the permutation control preserves frozen beta values but breaks "
            "query assignment; validation/test labels do not enter either"
        ),
        "reference": "oof_global",
        "methods": list(ALLOCATION_CONTROL_METHODS),
        "metrics": list(ALLOCATION_CONTROL_METRICS),
        "domains": {},
    }

    for domain in DOMAIN_LABELS:
        block = domains[domain]
        if not isinstance(block, dict):
            raise ValueError(f"{domain}: payload must be an object")
        seeds = tuple(int(seed) for seed in block.get("seeds", []))
        if seeds != ALLOCATION_CONTROL_SEEDS:
            raise ValueError(
                f"{domain}: seeds are {list(seeds)}, expected exactly "
                f"{list(ALLOCATION_CONTROL_SEEDS)}"
            )
        methods = block.get("methods")
        if not isinstance(methods, dict):
            raise ValueError(f"{domain}: methods must be an object")

        output[domain] = {}
        for method in ALLOCATION_CONTROL_METHODS:
            method_block = methods.get(method)
            if not isinstance(method_block, dict):
                raise ValueError(f"{domain}: missing method {method}")
            output[domain][method] = {}
            for metric in ALLOCATION_CONTROL_METRICS:
                metric_block = method_block.get(metric)
                if not isinstance(metric_block, dict):
                    raise ValueError(
                        f"{domain} {method}: missing metric {metric}"
                    )
                values_raw = metric_block.get("values")
                if not isinstance(values_raw, list):
                    raise ValueError(
                        f"{domain} {method} {metric}: values must be a list"
                    )
                values = np.asarray(values_raw, dtype=np.float64)
                if len(values) != len(ALLOCATION_CONTROL_SEEDS):
                    raise ValueError(
                        f"{domain} {method} {metric}: expected "
                        f"{len(ALLOCATION_CONTROL_SEEDS)} values, found "
                        f"{len(values)}"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"{domain} {method} {metric}: non-finite value"
                    )
                if np.any((values < 0.0) | (values > 1.0)):
                    raise ValueError(
                        f"{domain} {method} {metric}: value outside [0,1]"
                    )

                stored_mean = float(metric_block.get("mean", math.nan))
                stored_std = float(metric_block.get("std", math.nan))
                expected_mean = float(np.mean(values))
                expected_std = float(np.std(values, ddof=1))
                if not math.isclose(
                    stored_mean,
                    expected_mean,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"{domain} {method} {metric}: stored mean "
                        f"{stored_mean} != recomputed {expected_mean}"
                    )
                if not math.isclose(
                    stored_std,
                    expected_std,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"{domain} {method} {metric}: stored SD "
                        f"{stored_std} != recomputed {expected_std}"
                    )
                output[domain][method][metric] = {
                    "mean": expected_mean,
                    "std": expected_std,
                    "values": values.tolist(),
                    "seeds": list(ALLOCATION_CONTROL_SEEDS),
                }
        provenance["domains"][domain] = {
            "seeds": list(seeds),
            "n_seeds": len(seeds),
        }
    return output, provenance


def allocation_control_comparison(
    path: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Plot matched-seed control deltas, conditional on a complete summary."""
    data, provenance = _load_allocation_control_summary(path)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        squeeze=False,
    )
    figure.suptitle(
        "Allocation capacity controls relative to OOF global",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.925,
        (
            "Every policy is fit on frozen training-OOF ranks; validation "
            "and test labels never enter allocation fitting."
        ),
        ha="center",
        fontsize=11.2,
        color=INK,
    )
    figure.text(
        0.5,
        0.888,
        (
            "Bars are matched-seed mean deltas; whiskers are sample SD "
            "across seeds 42/123/456. Blue improves; orange declines; "
            "gray marks the OOF-global reference."
        ),
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )

    methods = list(ALLOCATION_CONTROL_METHODS)
    y = np.arange(len(methods))
    delta_payload: dict[
        str, dict[str, dict[str, tuple[float, float]]]
    ] = {
        metric: {domain: {} for domain in DOMAIN_LABELS}
        for metric in ALLOCATION_CONTROL_METRICS
    }
    metric_extents: dict[str, float] = {}
    for metric in ALLOCATION_CONTROL_METRICS:
        largest = 0.0
        for domain in DOMAIN_LABELS:
            reference = np.asarray(
                data[domain]["oof_global"][metric]["values"],
                dtype=np.float64,
            )
            for method in methods:
                values = np.asarray(
                    data[domain][method][metric]["values"],
                    dtype=np.float64,
                )
                paired = values - reference
                delta_mean = float(np.mean(paired))
                delta_std = float(np.std(paired, ddof=1))
                delta_payload[metric][domain][method] = (
                    delta_mean,
                    delta_std,
                )
                largest = max(
                    largest,
                    abs(delta_mean) + delta_std,
                )
        metric_extents[metric] = max(largest * 1.42, 1e-5)

    for row, (metric, metric_label) in enumerate(
        ALLOCATION_CONTROL_METRICS.items()
    ):
        extent = metric_extents[metric]
        for column, (domain, domain_label) in enumerate(
            DOMAIN_LABELS.items()
        ):
            axis = axes[row, column]
            for position, method in enumerate(methods):
                delta_mean, delta_std = (
                    delta_payload[metric][domain][method]
                )
                if method == "oof_global":
                    axis.scatter(
                        [0.0],
                        [position],
                        marker="s",
                        s=42,
                        color="#7F8C8D",
                        zorder=4,
                    )
                    axis.text(
                        0.035 * extent,
                        position,
                        "reference",
                        ha="left",
                        va="center",
                        fontsize=8.3,
                        color=MUTED,
                    )
                    continue
                color = POSITIVE if delta_mean >= 0.0 else NEGATIVE
                axis.barh(
                    position,
                    delta_mean,
                    height=0.60,
                    color=color,
                    alpha=0.88,
                )
                axis.errorbar(
                    delta_mean,
                    position,
                    xerr=delta_std,
                    fmt="o",
                    markersize=4.5,
                    capsize=3,
                    linewidth=1.3,
                    color=color,
                    ecolor=color,
                    zorder=4,
                )
                label_offset = 0.035 * extent
                axis.text(
                    delta_mean
                    + (
                        label_offset
                        if delta_mean >= 0.0
                        else -label_offset
                    ),
                    position,
                    f"{delta_mean:+.5f}",
                    ha="left" if delta_mean >= 0.0 else "right",
                    va="center",
                    fontsize=8.1,
                    color=INK,
                )

            axis.axvline(0.0, color=INK, linewidth=1.0)
            axis.set_xlim(-extent, extent)
            axis.set_yticks(y)
            axis.set_yticklabels(
                list(ALLOCATION_CONTROL_METHODS.values()),
                fontsize=8.6,
            )
            axis.invert_yaxis()
            if row == 0:
                axis.set_title(
                    f"{domain_label}\nmatched n=3 (42,123,456)"
                )
            axis.set_xlabel(
                rf"$\Delta$ {metric_label} vs OOF global"
            )
            _clean_axis(axis, grid_axis="x")

    figure.subplots_adjust(hspace=0.48, wspace=0.34)
    _add_footer(
        figure,
        f"Source: {path.name} · SHA-256 "
        f"{provenance['source_sha256'][:12]}… · "
        f"protocol: {ALLOCATION_CONTROL_PROTOCOL} · "
        "complete matched 3-domain × 3-seed control artifact.",
    )
    return (
        _save(
            figure,
            "11-allocation-capacity-controls",
            top=0.78,
            bottom=0.13,
        ),
        provenance,
    )


def _load_assignment_mechanism_summary(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load paired assignment effects and standardized primary coefficients."""
    raw = _read_json(path)
    if raw.get("protocol") != ALLOCATION_CONTROL_PROTOCOL:
        raise ValueError("assignment summary protocol mismatch")
    domains = raw.get("domains", {})
    output: dict[str, Any] = {}
    provenance: dict[str, Any] = {
        "source": str(path),
        "source_sha256": _sha256_file(path),
        "protocol": ALLOCATION_CONTROL_PROTOCOL,
        "bootstrap_unit": raw.get("bootstrap_unit"),
        "seed_aggregation": raw.get("seed_aggregation"),
        "comparison": (
            "primary learned beta assignment minus a deterministic "
            "permutation of the exact same beta multiset"
        ),
        "domains": {},
    }
    for domain in DOMAIN_LABELS:
        block = domains.get(domain)
        if not isinstance(block, dict):
            raise ValueError(f"{domain}: missing assignment summary")
        seeds = [int(seed) for seed in block.get("seeds", [])]
        if seeds != list(ALLOCATION_CONTROL_SEEDS):
            raise ValueError(f"{domain}: assignment seed set differs")

        paired_raw = block.get("assignment_paired")
        if not isinstance(paired_raw, dict):
            raise ValueError(f"{domain}: assignment_paired is missing")
        paired: dict[str, Any] = {}
        for metric in ASSIGNMENT_EFFECT_METRICS:
            record = paired_raw.get(metric)
            if not isinstance(record, dict):
                raise ValueError(f"{domain}: missing paired {metric}")
            if (
                record.get("challenger") != "dynamic_delta_010"
                or record.get("baseline") != "dynamic_beta_permuted"
            ):
                raise ValueError(
                    f"{domain} {metric}: assignment contrast differs")
            effect = float(record.get("difference", math.nan))
            interval = np.asarray(
                record.get("cluster_bootstrap_ci95", []),
                dtype=np.float64,
            )
            if (
                not math.isfinite(effect)
                or interval.shape != (2,)
                or not np.isfinite(interval).all()
                or interval[0] > interval[1]
                or int(record.get("clusters", 0)) <= 0
                or int(record.get("repetitions", 0)) < 1_000
            ):
                raise ValueError(
                    f"{domain} {metric}: invalid paired assignment record")
            if [int(seed) for seed in record.get("seeds", [])] != seeds:
                raise ValueError(
                    f"{domain} {metric}: paired seeds differ")
            paired[metric] = {
                "difference": effect,
                "ci95": interval.tolist(),
                "clusters": int(record["clusters"]),
                "repetitions": int(record["repetitions"]),
            }

        parameter_raw = block.get("primary_gate_parameters", {})
        coefficient_raw = parameter_raw.get(
            "standardized_coefficients", {})
        coefficients: dict[str, Any] = {}
        for feature in PRIMARY_GATE_FEATURES:
            record = coefficient_raw.get(feature)
            if not isinstance(record, dict):
                raise ValueError(
                    f"{domain}: missing coefficient {feature}")
            values = np.asarray(record.get("values", []), dtype=np.float64)
            if (
                values.shape != (len(ALLOCATION_CONTROL_SEEDS),)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"{domain} {feature}: invalid coefficient values")
            mean = float(record.get("mean", math.nan))
            std = float(record.get("std", math.nan))
            expected_mean = float(values.mean())
            expected_std = float(values.std(ddof=1))
            if not (
                math.isclose(mean, expected_mean, abs_tol=1e-12)
                and math.isclose(std, expected_std, abs_tol=1e-12)
            ):
                raise ValueError(
                    f"{domain} {feature}: coefficient aggregate differs")
            coefficients[feature] = {
                "mean": mean,
                "std": std,
                "values": values.tolist(),
            }

        output[domain] = {
            "paired": paired,
            "coefficients": coefficients,
        }
        provenance["domains"][domain] = {
            "seeds": seeds,
            "n_seeds": len(seeds),
        }
    return output, provenance


def assignment_mechanism_charts(
    path: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Plot paired assignment effects and the four-parameter gate mechanism."""
    data, provenance = _load_assignment_mechanism_summary(path)

    figure, axes_raw = plt.subplots(
        1, 3, figsize=(16, 7.2), squeeze=False
    )
    axes = list(axes_raw[0])
    figure.suptitle(
        r"Does matching each query to its learned $\beta_q$ matter?",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.91,
        (
            "Primary minus one deterministic per-seed reassignment of the "
            r"exact same $\beta_q$ values; only query-to-weight matching "
            "changes."
        ),
        ha="center",
        fontsize=11.0,
        color=INK,
    )
    figure.text(
        0.5,
        0.87,
        (
            "Points average matched-seed query effects; whiskers are 95% "
            "query-level intervals conditional on those three fits and "
            "their fixed reassignments. "
            "Blue: CI above zero; gray: overlaps zero."
        ),
        ha="center",
        fontsize=10.3,
        color=MUTED,
    )
    scale = 1e4
    global_extent = max(
        abs(bound) * scale
        for domain in DOMAIN_LABELS
        for metric in ASSIGNMENT_EFFECT_METRICS
        for bound in data[domain]["paired"][metric]["ci95"]
    )
    global_extent = max(global_extent * 1.35, 0.5)
    y = np.arange(len(ASSIGNMENT_EFFECT_METRICS))
    for axis, (domain, domain_label) in zip(
        axes, DOMAIN_LABELS.items()
    ):
        for position, (metric, metric_label) in enumerate(
            ASSIGNMENT_EFFECT_METRICS.items()
        ):
            record = data[domain]["paired"][metric]
            value = record["difference"] * scale
            low, high = (
                bound * scale for bound in record["ci95"]
            )
            color = (
                POSITIVE if low > 0
                else NEGATIVE if high < 0
                else "#7F8C8D"
            )
            axis.errorbar(
                value,
                position,
                xerr=[[value - low], [high - value]],
                fmt="o",
                markersize=7,
                capsize=4,
                color=color,
                ecolor=color,
                linewidth=1.8,
            )
            axis.text(
                value + (0.035 * global_extent),
                position,
                f"{value:+.2f}",
                ha="left",
                va="center",
                fontsize=8.8,
                color=INK,
            )
        axis.axvline(0.0, color=INK, linewidth=1.0)
        axis.set_xlim(-global_extent, global_extent)
        axis.set_yticks(y)
        axis.set_yticklabels(
            list(ASSIGNMENT_EFFECT_METRICS.values()))
        axis.invert_yaxis()
        axis.set_title(
            f"{domain_label}\nmatched n=3 (42,123,456)")
        axis.set_xlabel(
            r"Primary $-$ permuted effect ($\times10^{-4}$)")
        _clean_axis(axis, grid_axis="x")
    _add_footer(
        figure,
        f"Source: {path.name} · SHA-256 "
        f"{provenance['source_sha256'][:12]}… · unadjusted across metrics; "
        "Video nDCG@20 is the pre-reported primary mechanism metric.",
    )
    assignment_outputs = _save(
        figure,
        "15-beta-assignment-paired-effects",
        top=0.75,
        bottom=0.15,
    )

    coefficient_figure, coefficient_axes_raw = plt.subplots(
        1, 3, figsize=(16, 7), squeeze=False
    )
    coefficient_axes = list(coefficient_axes_raw[0])
    coefficient_figure.suptitle(
        "What context shifts neural allocation?",
        fontweight="bold",
    )
    coefficient_figure.text(
        0.5,
        0.91,
        (
            "Primary linear-gate coefficients on standardized OOF features; "
            "positive values increase neural mass."
        ),
        ha="center",
        fontsize=11.0,
        color=INK,
    )
    coefficient_extent = max(
        abs(record[value])
        for domain in DOMAIN_LABELS
        for record in data[domain]["coefficients"].values()
        for value in ("mean", "std")
    )
    coefficient_extent = max(coefficient_extent * 1.9, 0.02)
    feature_y = np.arange(len(PRIMARY_GATE_FEATURES))
    for axis, (domain, domain_label) in zip(
        coefficient_axes, DOMAIN_LABELS.items()
    ):
        for position, (feature, feature_label) in enumerate(
            PRIMARY_GATE_FEATURES.items()
        ):
            record = data[domain]["coefficients"][feature]
            axis.errorbar(
                record["mean"],
                position,
                xerr=record["std"],
                fmt="o",
                markersize=7,
                capsize=4,
                color=DOMAIN_COLORS[domain],
                ecolor=DOMAIN_COLORS[domain],
                linewidth=1.8,
            )
            axis.text(
                record["mean"] + 0.035 * coefficient_extent,
                position,
                f"{record['mean']:+.3f}",
                ha="left",
                va="center",
                fontsize=8.8,
                color=INK,
            )
        axis.axvline(0.0, color=INK, linewidth=1.0)
        axis.set_xlim(-coefficient_extent, coefficient_extent)
        axis.set_yticks(feature_y)
        axis.set_yticklabels(list(PRIMARY_GATE_FEATURES.values()))
        axis.invert_yaxis()
        axis.set_title(
            f"{domain_label}\nmean ± seed SD")
        axis.set_xlabel("Standardized linear coefficient")
        _clean_axis(axis, grid_axis="x")
    _add_footer(
        coefficient_figure,
        f"Source: {path.name} · exact frozen gate states · "
        "coefficients describe allocation, not causal feature effects.",
    )
    coefficient_outputs = _save(
        coefficient_figure,
        "16-primary-gate-standardized-coefficients",
        top=0.80,
        bottom=0.14,
    )
    return assignment_outputs + coefficient_outputs, provenance


def _beta_decile_stat(
    record: Any,
    *,
    label: str,
    bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Validate one matched-seed statistic emitted by the primary summary."""
    if not isinstance(record, dict):
        raise ValueError(f"{label}: statistic must be an object")
    values_raw = record.get("values")
    if not isinstance(values_raw, list):
        raise ValueError(f"{label}: values must be a list")
    values = np.asarray(values_raw, dtype=np.float64)
    if values.shape != (len(BETA_DECILE_SEEDS),):
        raise ValueError(
            f"{label}: expected {len(BETA_DECILE_SEEDS)} matched "
            f"seed values, found {len(values)}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: non-finite value")
    if bounds is not None:
        lower, upper = bounds
        if np.any((values < lower) | (values > upper)):
            raise ValueError(
                f"{label}: value outside [{lower}, {upper}]"
            )

    stored_mean = float(record.get("mean", math.nan))
    stored_std = float(record.get("std", math.nan))
    expected_mean = float(np.mean(values))
    expected_std = float(np.std(values, ddof=1))
    if not math.isclose(
        stored_mean,
        expected_mean,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{label}: stored mean {stored_mean} != "
            f"recomputed {expected_mean}"
        )
    if not math.isclose(
        stored_std,
        expected_std,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{label}: stored SD {stored_std} != "
            f"recomputed {expected_std}"
        )
    return {
        "mean": expected_mean,
        "std": expected_std,
        "values": values.tolist(),
    }


def _load_beta_decile_mechanism(
    summary: dict[str, Any],
    *,
    path: Path = SUMMARY_PATH,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load a complete three-domain beta-decile mechanism audit.

    The chart is intentionally conditional: it is emitted only when all
    domains contain the same canonical three seeds, ten internally consistent
    beta deciles, and allocation-discrimination statistics that can be
    reconstructed exactly from those deciles.
    """
    domains = summary.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("summary domains must be an object")
    output: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {
        "source": str(path),
        "source_sha256": _sha256_file(path),
        "summary_source": summary.get("source"),
        "matched_seeds": list(BETA_DECILE_SEEDS),
        "x_definition": (
            "within-domain, within-seed equal-count deciles ordered by "
            "frozen test-time dynamic beta_q"
        ),
        "y_definition": (
            "realized neural-only minus memory-only nDCG@20 within each "
            "beta decile"
        ),
        "mechanism_test": (
            "higher beta deciles should order queries toward greater "
            "realized neural advantage; universal positive advantage is "
            "not required"
        ),
        "domains": {},
    }

    for domain in DOMAIN_LABELS:
        block = domains.get(domain)
        if not isinstance(block, dict):
            raise ValueError(f"{domain}: missing domain summary")
        seeds = tuple(int(seed) for seed in block.get("seeds", []))
        if seeds != BETA_DECILE_SEEDS:
            raise ValueError(
                f"{domain}: seeds are {list(seeds)}, expected exactly "
                f"{list(BETA_DECILE_SEEDS)}"
            )
        if int(block.get("n_seeds", -1)) != len(seeds):
            raise ValueError(
                f"{domain}: n_seeds does not match declared seeds"
            )

        deciles = block.get("beta_deciles")
        if not isinstance(deciles, list) or len(deciles) != 10:
            raise ValueError(
                f"{domain}: expected exactly ten beta_deciles"
            )
        parsed_deciles: list[dict[str, Any]] = []
        for index, decile in enumerate(deciles, start=1):
            if not isinstance(decile, dict):
                raise ValueError(
                    f"{domain} decile {index}: payload must be an object"
                )
            if int(decile.get("decile", -1)) != index:
                raise ValueError(
                    f"{domain}: beta deciles are not ordered 1...10"
                )
            count = _beta_decile_stat(
                decile.get("n"),
                label=f"{domain} decile {index} count",
            )
            if any(value <= 0 for value in count["values"]):
                raise ValueError(
                    f"{domain} decile {index}: empty seed decile"
                )
            beta_mean = _beta_decile_stat(
                decile.get("beta_mean"),
                label=f"{domain} decile {index} beta_mean",
                bounds=(0.0, 1.0),
            )
            advantage = _beta_decile_stat(
                decile.get(
                    "realized_neural_minus_memory_ndcg20"
                ),
                label=(
                    f"{domain} decile {index} realized neural-memory "
                    "nDCG@20"
                ),
                bounds=(-1.0, 1.0),
            )
            parsed_deciles.append(
                {
                    "decile": index,
                    "n": count,
                    "beta_mean": beta_mean,
                    "advantage": advantage,
                }
            )

        beta_by_seed = np.asarray(
            [item["beta_mean"]["values"] for item in parsed_deciles],
            dtype=np.float64,
        )
        if np.any(np.diff(beta_by_seed, axis=0) < -1e-12):
            raise ValueError(
                f"{domain}: beta_mean is not nondecreasing by decile"
            )

        discrimination_raw = block.get("allocation_discrimination")
        if not isinstance(discrimination_raw, dict):
            raise ValueError(
                f"{domain}: missing allocation_discrimination"
            )
        discrimination: dict[str, dict[str, Any]] = {}
        for key in (
            "low_beta_deciles_1_3",
            "high_beta_deciles_8_10",
            "high_minus_low",
        ):
            discrimination[key] = _beta_decile_stat(
                discrimination_raw.get(key),
                label=f"{domain} allocation_discrimination {key}",
                bounds=(-1.0, 1.0),
            )

        counts = np.asarray(
            [item["n"]["values"] for item in parsed_deciles],
            dtype=np.float64,
        )
        advantages = np.asarray(
            [item["advantage"]["values"] for item in parsed_deciles],
            dtype=np.float64,
        )
        recomputed_low = np.asarray(
            [
                np.average(
                    advantages[:3, seed_index],
                    weights=counts[:3, seed_index],
                )
                for seed_index in range(len(seeds))
            ],
            dtype=np.float64,
        )
        recomputed_high = np.asarray(
            [
                np.average(
                    advantages[7:, seed_index],
                    weights=counts[7:, seed_index],
                )
                for seed_index in range(len(seeds))
            ],
            dtype=np.float64,
        )
        expected_discrimination = {
            "low_beta_deciles_1_3": recomputed_low,
            "high_beta_deciles_8_10": recomputed_high,
            "high_minus_low": recomputed_high - recomputed_low,
        }
        for key, expected in expected_discrimination.items():
            observed = np.asarray(
                discrimination[key]["values"], dtype=np.float64
            )
            if not np.allclose(
                observed,
                expected,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{domain}: {key} does not reconstruct from "
                    "beta_deciles"
                )

        output[domain] = {
            "seeds": list(seeds),
            "deciles": parsed_deciles,
            "allocation_discrimination": discrimination,
        }
        provenance["domains"][domain] = {
            "seeds": list(seeds),
            "n_seeds": len(seeds),
            "queries_per_seed": [
                int(value) for value in np.sum(counts, axis=0)
            ],
            "mean_beta_decile_1": parsed_deciles[0][
                "beta_mean"
            ]["mean"],
            "mean_beta_decile_10": parsed_deciles[-1][
                "beta_mean"
            ]["mean"],
            "high_minus_low": discrimination["high_minus_low"],
        }
    return output, provenance


def beta_decile_mechanism(
    summary: dict[str, Any],
    *,
    path: Path = SUMMARY_PATH,
) -> tuple[list[str], dict[str, Any]]:
    """Plot realized expert advantage across learned-beta deciles."""
    data, provenance = _load_beta_decile_mechanism(
        summary, path=path
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 8),
        sharey=True,
    )
    figure.suptitle(
        r"Does dynamic $\beta_q$ order queries by realized neural advantage?",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.91,
        (
            "Mechanism criterion: higher β deciles should show greater "
            "neural−memory nDCG@20—not universal positivity."
        ),
        ha="center",
        fontsize=11.2,
        color=INK,
    )
    figure.text(
        0.5,
        0.872,
        (
            "Positive values favor the neural expert; negative values "
            "favor memory. Thin lines are seeds; bold line is their mean."
        ),
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )

    all_values = np.asarray(
        [
            value
            for domain in DOMAIN_LABELS
            for decile in data[domain]["deciles"]
            for value in decile["advantage"]["values"]
        ],
        dtype=np.float64,
    )
    extent = max(float(np.max(np.abs(all_values))) * 1.30, 1e-4)
    x = np.arange(1, 11)

    for axis, (domain, domain_label) in zip(
        axes, DOMAIN_LABELS.items()
    ):
        deciles = data[domain]["deciles"]
        seed_values = np.asarray(
            [item["advantage"]["values"] for item in deciles],
            dtype=np.float64,
        )
        means = np.asarray(
            [item["advantage"]["mean"] for item in deciles],
            dtype=np.float64,
        )
        errors = np.asarray(
            [item["advantage"]["std"] for item in deciles],
            dtype=np.float64,
        )
        beta_means = np.asarray(
            [item["beta_mean"]["mean"] for item in deciles],
            dtype=np.float64,
        )
        color = DOMAIN_COLORS[domain]
        for seed_index in range(seed_values.shape[1]):
            axis.plot(
                x,
                seed_values[:, seed_index],
                color=color,
                linewidth=1.0,
                alpha=0.22,
            )
        axis.errorbar(
            x,
            means,
            yerr=errors,
            color=color,
            marker="o",
            markersize=5.5,
            linewidth=2.4,
            capsize=2.5,
            label="Across-seed mean ± SD",
        )
        axis.axhline(0.0, color=INK, linewidth=1.15)
        axis.set_xlim(0.5, 10.5)
        axis.set_ylim(-extent, extent)
        axis.set_xticks(x)
        axis.set_xticklabels([f"D{index}" for index in x])
        axis.set_xlabel(r"Dynamic-$\beta_q$ decile (low $\rightarrow$ high)")
        axis.set_title(
            f"{domain_label}\n"
            f"mean β: {beta_means[0]:.3f} → {beta_means[-1]:.3f}"
        )
        if axis is axes[0]:
            axis.set_ylabel(
                "Realized neural − memory nDCG@20"
            )
        discrimination = data[domain][
            "allocation_discrimination"
        ]["high_minus_low"]
        gap = float(discrimination["mean"])
        gap_sd = float(discrimination["std"])
        axis.text(
            0.04,
            0.955,
            f"D8–D10 minus D1–D3: {gap:+.4f} ± {gap_sd:.4f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            color=POSITIVE if gap >= 0.0 else NEGATIVE,
        )
        _clean_axis(axis)

    _add_footer(
        figure,
        f"Source: {path.name} · SHA-256 "
        f"{provenance['source_sha256'][:12]}… · "
        "equal-count deciles formed within domain and seed · "
        "complete matched seeds "
        + ",".join(map(str, provenance["matched_seeds"]))
        + ".",
    )
    return (
        _save(
            figure,
            "12-beta-decile-mechanism-ndcg20",
            top=0.80,
            bottom=0.14,
        ),
        provenance,
    )


def _load_full_metric_paired_dashboard(
    summary: dict[str, Any],
    *,
    path: Path = SUMMARY_PATH,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
]:
    """Validate the complete six-metric dynamic-minus-global comparison.

    This chart is intentionally conditional.  A partial summary must not be
    rendered as a complete three-seed result, so every domain and metric must
    declare the exact canonical seeds and a finite query-level bootstrap CI.
    Values are retained in their signed, unscaled form until plotting.
    """
    domains = summary.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("summary domains must be an object")

    output: dict[str, dict[str, dict[str, Any]]] = {}
    provenance: dict[str, Any] = {
        "source": str(path),
        "source_sha256": _sha256_file(path),
        "summary_source": summary.get("source"),
        "bootstrap_unit": summary.get("bootstrap_unit"),
        "seed_aggregation": summary.get("seed_aggregation"),
        "matched_seeds": list(FULL_METRIC_PAIRED_SEEDS),
        "challenger": "dynamic",
        "baseline": "oof_global",
        "metrics": list(FULL_METRIC_PAIRED_METRICS),
        "display_scale": 1000.0,
        "ci": "paired query-level bootstrap 95%",
        "domains": {},
    }

    for domain in DOMAIN_LABELS:
        block = domains.get(domain)
        if not isinstance(block, dict):
            raise ValueError(f"{domain}: missing domain payload")
        seeds = tuple(int(seed) for seed in block.get("seeds", []))
        if seeds != FULL_METRIC_PAIRED_SEEDS:
            raise ValueError(
                f"{domain}: seeds are {list(seeds)}, expected exactly "
                f"{list(FULL_METRIC_PAIRED_SEEDS)}"
            )
        if int(block.get("n_seeds", -1)) != len(seeds):
            raise ValueError(
                f"{domain}: n_seeds does not match declared seeds"
            )
        paired = block.get("paired")
        if not isinstance(paired, dict):
            raise ValueError(f"{domain}: paired must be an object")
        comparison = paired.get("oof_global")
        if not isinstance(comparison, dict):
            raise ValueError(
                f"{domain}: missing paired.oof_global comparison"
            )

        output[domain] = {}
        domain_provenance: dict[str, Any] = {
            "seeds": list(seeds),
            "n_seeds": len(seeds),
            "metrics": {},
        }
        for metric in FULL_METRIC_PAIRED_METRICS:
            item = comparison.get(metric)
            if not isinstance(item, dict):
                raise ValueError(
                    f"{domain}: missing paired.oof_global.{metric}"
                )
            metric_seeds = tuple(
                int(seed) for seed in item.get("seeds", [])
            )
            if metric_seeds != FULL_METRIC_PAIRED_SEEDS:
                raise ValueError(
                    f"{domain} {metric}: paired seeds are "
                    f"{list(metric_seeds)}, expected exactly "
                    f"{list(FULL_METRIC_PAIRED_SEEDS)}"
                )
            if item.get("metric") != metric:
                raise ValueError(
                    f"{domain} {metric}: metric label is "
                    f"{item.get('metric')!r}"
                )
            if item.get("challenger") != "dynamic":
                raise ValueError(
                    f"{domain} {metric}: challenger must be 'dynamic'"
                )
            if item.get("baseline") != "oof_global":
                raise ValueError(
                    f"{domain} {metric}: baseline must be 'oof_global'"
                )

            difference = float(item.get("difference", math.nan))
            ci_raw = item.get("cluster_bootstrap_ci95")
            if (
                not isinstance(ci_raw, list)
                or len(ci_raw) != 2
            ):
                raise ValueError(
                    f"{domain} {metric}: complete two-sided "
                    "cluster_bootstrap_ci95 is required"
                )
            low, high = (float(ci_raw[0]), float(ci_raw[1]))
            values = np.asarray(
                [difference, low, high], dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{domain} {metric}: non-finite difference or CI"
                )
            if np.any((values < -1.0) | (values > 1.0)):
                raise ValueError(
                    f"{domain} {metric}: difference or CI outside [-1,1]"
                )
            if low > high:
                raise ValueError(
                    f"{domain} {metric}: CI lower bound exceeds upper"
                )
            if difference < low - 1e-12 or difference > high + 1e-12:
                raise ValueError(
                    f"{domain} {metric}: point estimate lies outside CI"
                )

            clusters = int(item.get("clusters", 0))
            repetitions = int(item.get("repetitions", 0))
            if clusters <= 0:
                raise ValueError(
                    f"{domain} {metric}: positive cluster count required"
                )
            if repetitions <= 0:
                raise ValueError(
                    f"{domain} {metric}: positive bootstrap repetitions "
                    "required"
                )

            output[domain][metric] = {
                "difference": difference,
                "cluster_bootstrap_ci95": [low, high],
                "clusters": clusters,
                "repetitions": repetitions,
                "seeds": list(metric_seeds),
            }
            domain_provenance["metrics"][metric] = {
                "clusters": clusters,
                "repetitions": repetitions,
            }
        provenance["domains"][domain] = domain_provenance

    return output, provenance


def full_metric_paired_dashboard(
    summary: dict[str, Any],
    *,
    path: Path = SUMMARY_PATH,
) -> tuple[list[str], dict[str, Any]]:
    """Plot signed paired effects for all Recall and nDCG cutoffs."""
    data, provenance = _load_full_metric_paired_dashboard(
        summary, path=path
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9.5),
        sharey=True,
        squeeze=False,
    )
    figure.suptitle(
        r"Dynamic $\beta_q$ minus training-only OOF-global allocation",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.922,
        (
            "Signed paired query effects across all reported ranking "
            "cutoffs; zero means no change."
        ),
        ha="center",
        fontsize=11.2,
        color=INK,
    )
    figure.text(
        0.5,
        0.888,
        (
            "Points are three-seed paired estimates; whiskers are "
            "query-level bootstrap 95% CIs."
        ),
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )

    metric_rows = [
        list(FULL_METRIC_PAIRED_METRICS.items())[:3],
        list(FULL_METRIC_PAIRED_METRICS.items())[3:],
    ]
    scale = float(provenance["display_scale"])
    row_extents: list[float] = []
    for metric_row in metric_rows:
        maximum = max(
            abs(value) * scale
            for metric, _ in metric_row
            for domain in DOMAIN_LABELS
            for value in (
                data[domain][metric]["difference"],
                *data[domain][metric]["cluster_bootstrap_ci95"],
            )
        )
        row_extents.append(max(maximum * 1.30, 0.10))

    y = np.arange(len(DOMAIN_LABELS))
    for row, metric_row in enumerate(metric_rows):
        extent = row_extents[row]
        for column, (metric, metric_label) in enumerate(metric_row):
            axis = axes[row, column]
            for position, domain in enumerate(DOMAIN_LABELS):
                item = data[domain][metric]
                difference = float(item["difference"]) * scale
                low, high = (
                    float(bound) * scale
                    for bound in item["cluster_bootstrap_ci95"]
                )
                color = DOMAIN_COLORS[domain]
                axis.errorbar(
                    difference,
                    position,
                    xerr=[
                        [difference - low],
                        [high - difference],
                    ],
                    fmt="o",
                    markersize=7.5,
                    capsize=4.5,
                    color=color,
                    ecolor=color,
                    linewidth=1.8,
                    zorder=3,
                )
                offset = 0.035 * extent
                axis.text(
                    difference
                    + (offset if difference >= 0.0 else -offset),
                    position - 0.14,
                    f"{difference:+.2f}",
                    ha="left" if difference >= 0.0 else "right",
                    va="center",
                    fontsize=8.9,
                    color=INK,
                )

            axis.axvline(0.0, color=INK, linewidth=1.15)
            axis.set_xlim(-extent, extent)
            axis.set_title(metric_label)
            axis.set_xlabel(r"Paired difference $\times 10^3$")
            axis.set_yticks(y)
            axis.set_yticklabels(list(DOMAIN_LABELS.values()))
            axis.tick_params(axis="y", labelleft=True)
            _clean_axis(axis, grid_axis="x")

    axes[0, 0].invert_yaxis()
    figure.text(
        0.012,
        0.68,
        "Recall",
        rotation=90,
        va="center",
        ha="left",
        fontsize=12.5,
        color=MUTED,
    )
    figure.text(
        0.012,
        0.29,
        "nDCG",
        rotation=90,
        va="center",
        ha="left",
        fontsize=12.5,
        color=MUTED,
    )
    figure.subplots_adjust(hspace=0.42, wspace=0.30)
    _add_footer(
        figure,
        f"Source: {path.name} · SHA-256 "
        f"{provenance['source_sha256'][:12]}… · exact matched seeds "
        + ",".join(map(str, provenance["matched_seeds"]))
        + " · signs and CI bounds preserved from paired.oof_global.",
    )
    return (
        _save(
            figure,
            "13-dynamic-vs-global-six-metric-paired-delta",
            top=0.81,
            bottom=0.13,
        ),
        provenance,
    )


def _write_contact_sheet(outputs: Iterable[str]) -> str:
    png_paths = sorted(
        {
            Path(output)
            for output in outputs
            if output.endswith(".png")
            and Path(output).name[:2].isdigit()
        }
    )
    if not png_paths:
        raise RuntimeError("cannot create contact sheet without PNG charts")
    thumb_width, thumb_height = 720, 390
    cell_width, cell_height = 760, 440
    cells: list[Image.Image] = []
    for path in png_paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.thumbnail(
            (thumb_width, thumb_height), Image.Resampling.LANCZOS
        )
        cell = Image.new("RGB", (cell_width, cell_height), WHITE)
        cell.paste(
            image,
            ((cell_width - image.width) // 2, 8),
        )
        ImageDraw.Draw(cell).text(
            (16, cell_height - 24), path.name, fill=INK
        )
        cells.append(cell)
    columns = 2
    rows = math.ceil(len(cells) / columns)
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "#E1E1E1",
    )
    for index, cell in enumerate(cells):
        sheet.paste(
            cell,
            (
                (index % columns) * cell_width,
                (index // columns) * cell_height,
            ),
        )
    destination = OUT / "contact-sheet.png"
    sheet.save(destination)
    return str(destination)


def main() -> None:
    _configure_style()
    results = _read_json(RESULTS_PATH)
    summary = _read_json(SUMMARY_PATH)
    runs_by_domain = _runs_by_domain(results)

    missing = [
        domain for domain, runs in runs_by_domain.items() if not runs
    ]
    if missing:
        raise RuntimeError(
            "No complete dynamic-beta runs for: " + ", ".join(missing)
        )

    outputs: list[str] = method_and_protocol_overview()
    (
        allocation_mode_outputs,
        allocation_mode_provenance,
    ) = allocation_modes(runs_by_domain, source_path=RESULTS_PATH)
    outputs.extend(allocation_mode_outputs)
    outputs.extend(paired_delta(summary))
    outputs.extend(rescue_damage(runs_by_domain))
    outputs.extend(beta_distribution(runs_by_domain))
    outputs.extend(beta_context_behavior(runs_by_domain))
    outputs.extend(rrf_sensitivity(runs_by_domain))

    inference_path = _find_inference_benchmark()
    if inference_path is not None:
        outputs.extend(inference_performance(inference_path))

    external_outputs, external_provenance = (
        external_baseline_comparison(summary)
    )
    outputs.extend(external_outputs)

    expert_swap_provenance = None
    expert_swap_status = "skipped: source JSON absent"
    if EXPERT_SWAP_PATH.exists():
        expert_swap_result = expert_swap_comparison(EXPERT_SWAP_PATH)
        if expert_swap_result is not None:
            expert_outputs, expert_swap_provenance = expert_swap_result
            outputs.extend(expert_outputs)
            expert_swap_status = "generated"
        else:
            expert_swap_status = "skipped: no matched complete runs"

    fusion_control_provenance = None
    fusion_control_status = "skipped: source JSON absent"
    if FUSION_CONTROL_PATH.exists():
        try:
            (
                fusion_outputs,
                fusion_control_provenance,
            ) = fusion_operator_comparison(FUSION_CONTROL_PATH)
        except (KeyError, TypeError, ValueError) as error:
            fusion_control_status = (
                "skipped: incomplete or invalid result: " + str(error)
            )
        else:
            outputs.extend(fusion_outputs)
            fusion_control_status = "generated"

    allocation_control_provenance = None
    allocation_control_status = "skipped: source JSON absent"
    assignment_mechanism_provenance = None
    assignment_mechanism_status = "skipped: source JSON absent"
    if ALLOCATION_CONTROL_PATH.exists():
        try:
            (
                allocation_outputs,
                allocation_control_provenance,
            ) = allocation_control_comparison(
                ALLOCATION_CONTROL_PATH
            )
        except (KeyError, TypeError, ValueError) as error:
            allocation_control_status = (
                "skipped: incomplete or invalid result: " + str(error)
            )
        else:
            outputs.extend(allocation_outputs)
            allocation_control_status = "generated"
        try:
            (
                assignment_outputs,
                assignment_mechanism_provenance,
            ) = assignment_mechanism_charts(
                ALLOCATION_CONTROL_PATH
            )
        except (KeyError, TypeError, ValueError) as error:
            assignment_mechanism_status = (
                "skipped: incomplete or invalid result: " + str(error)
            )
        else:
            outputs.extend(assignment_outputs)
            assignment_mechanism_status = "generated"

    beta_decile_provenance = None
    beta_decile_status = "skipped: incomplete summary"
    try:
        (
            beta_decile_outputs,
            beta_decile_provenance,
        ) = beta_decile_mechanism(summary)
    except (KeyError, TypeError, ValueError) as error:
        beta_decile_status = (
            "skipped: incomplete or invalid result: " + str(error)
        )
    else:
        outputs.extend(beta_decile_outputs)
        beta_decile_status = "generated"

    full_metric_paired_provenance = None
    full_metric_paired_status = "skipped: incomplete summary"
    try:
        (
            full_metric_paired_outputs,
            full_metric_paired_provenance,
        ) = full_metric_paired_dashboard(summary)
    except (KeyError, TypeError, ValueError) as error:
        full_metric_paired_status = (
            "skipped: incomplete or invalid result: " + str(error)
        )
    else:
        outputs.extend(full_metric_paired_outputs)
        full_metric_paired_status = "generated"

    (
        fusion_ablation_outputs,
        fusion_ablation_provenance,
    ) = fusion_ablation_recall20(
        runs_by_domain,
        source_path=RESULTS_PATH,
    )
    outputs.extend(fusion_ablation_outputs)

    contact_sheet = _write_contact_sheet(outputs)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "sources": {
            "results": str(RESULTS_PATH),
            "summary": str(SUMMARY_PATH),
            "inference": str(inference_path) if inference_path else None,
            "neighborhood_baselines": str(
                NEIGHBORHOOD_BASELINE_PATH
            ),
            "amazon_neural_baseline_artifacts": str(
                AMAZON_NEURAL_ARTIFACT_DIR
            ),
            "diginetica_neural_baselines": str(
                DIGI_NEURAL_BASELINE_PATH
            ),
            "expert_swap": (
                str(EXPERT_SWAP_PATH) if EXPERT_SWAP_PATH.exists() else None
            ),
            "fusion_operator_control": (
                str(FUSION_CONTROL_PATH)
                if FUSION_CONTROL_PATH.exists()
                else None
            ),
            "allocation_controls": (
                str(ALLOCATION_CONTROL_PATH)
                if ALLOCATION_CONTROL_PATH.exists()
                else None
            ),
        },
        "available_result_seeds": {
            domain: [int(run["seed"]) for run in runs]
            for domain, runs in runs_by_domain.items()
        },
        "paired_summary_seeds": {
            domain: summary.get("domains", {}).get(domain, {}).get("seeds", [])
            for domain in DOMAIN_LABELS
        },
        "outputs": outputs,
        "contact_sheet": contact_sheet,
        "chart_01_allocation_mode_provenance": (
            allocation_mode_provenance
        ),
        "chart_08_external_baseline_provenance": external_provenance,
        "chart_09_expert_swap_status": expert_swap_status,
        "chart_09_expert_swap_provenance": expert_swap_provenance,
        "chart_10_fusion_operator_status": fusion_control_status,
        "chart_10_fusion_operator_provenance": (
            fusion_control_provenance
        ),
        "chart_11_allocation_control_status": (
            allocation_control_status
        ),
        "chart_11_allocation_control_provenance": (
            allocation_control_provenance
        ),
        "chart_12_beta_decile_mechanism_status": (
            beta_decile_status
        ),
        "chart_12_beta_decile_mechanism_provenance": (
            beta_decile_provenance
        ),
        "chart_13_full_metric_paired_status": (
            full_metric_paired_status
        ),
        "chart_13_full_metric_paired_provenance": (
            full_metric_paired_provenance
        ),
        "chart_14_fusion_ablation_status": "generated",
        "chart_14_fusion_ablation_provenance": (
            fusion_ablation_provenance
        ),
        "chart_15_16_assignment_mechanism_status": (
            assignment_mechanism_status
        ),
        "chart_15_16_assignment_mechanism_provenance": (
            assignment_mechanism_provenance
        ),
        "chart_17_pair_certificate_status": "generated",
        "chart_17_pair_certificate_provenance": {
            "kind": "formal method schematic",
            "empirical_values": False,
            "certificate": (
                "|m_OOF(a,b)| > |delta_q| |D_q(a)-D_q(b)|"
            ),
            "limitation": (
                "pair-order certificate; not a Recall/nDCG guarantee"
            ),
        },
        "notes": [
            "No experimental value is embedded in the generator.",
            (
                "Chart 01 is generated only from "
                "dynamic_beta_trainonly_v2_results.json and requires "
                "all six endpoint/allocation modes for exact matched "
                "seeds 42/123/456 in every domain."
            ),
            "Allocation and sensitivity bars start at zero.",
            "Paired-difference axes are zero-referenced and report paired "
            "query-level CIs when present.",
            "Context effects are computed from rank artifacts referenced by completed runs.",
            (
                "CEARF-N uses a Video cache documented as E5-small and "
                "TF-IDF/SVD on Baby/Diginetica while chart-08 comparators "
                "are ID-only; chart "
                "08 is not metadata-matched architecture attribution."
            ),
            (
                "Chart 09 is conditional and is not emitted without all "
                "three domains and exact matched seeds 42/123/456 in "
                "dynamic_beta_expert_swap_results.json."
            ),
            (
                "Chart 10 is conditional and joins outputs/contact sheet "
                "only after all three fusion-control domains have the same "
                "non-empty seed set and complete weighted-RRF/normalized-"
                "CombSUM R@20 and nDCG@20 metrics."
            ),
            (
                "Chart 11 is conditional and joins outputs/contact sheet "
                "only when the allocation-control summary certifies the "
                "training-OOF protocol and exact seeds 42/123/456 for all "
                "three domains, with internally consistent per-seed "
                "utility and nDCG@20 values."
            ),
            (
                "Chart 12 is conditional and joins outputs/contact sheet "
                "only when dynamic_beta_summary.json contains internally "
                "reconstructable beta_deciles and "
                "allocation_discrimination for exact matched seeds "
                "42/123/456 in all three domains. Its mechanism criterion "
                "is ordering of realized neural-minus-memory nDCG@20, not "
                "universal positive expert advantage."
            ),
            (
                "Chart 13 is conditional and joins outputs/contact sheet "
                "only when paired.oof_global contains complete signed "
                "Recall@6/10/20 and nDCG@6/10/20 effects with query-level "
                "95% CIs for exact matched seeds 42/123/456 in all three "
                "domains."
            ),
            (
                "Chart 14 reads Recall@20 only from "
                "dynamic_beta_trainonly_v2_results.json, requires exact "
                "matched seeds 42/123/456 and consistent test-query "
                "counts, and reports mean ± sample SD. Its endpoint gain "
                "is the signed fused mean minus the better memory-only or "
                "neural-only mean."
            ),
            (
                "Charts 15--16 are conditional on the allocation-control "
                "summary containing exact primary-vs-permuted paired "
                "query-level effects and standardized primary-gate "
                "coefficients for seeds 42/123/456 in all domains."
            ),
            (
                "Chart 17 is a data-independent schematic of the formal "
                "pair-order certificate and contains no experimental value."
            ),
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "chart_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Generated {len(outputs)} chart files in {OUT}")
    print(f"Contact sheet: {contact_sheet}")
    print(f"Expert-swap chart: {expert_swap_status}")
    print(f"Fusion-operator chart: {fusion_control_status}")
    print(f"Allocation-control chart: {allocation_control_status}")
    print(f"Beta-decile mechanism chart: {beta_decile_status}")
    print(f"Full-metric paired chart: {full_metric_paired_status}")
    print("Fusion-ablation chart: generated")
    print(f"Manifest: {manifest_path}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
