from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import paired_bootstrap


def collect_runs(root):
    runs = []
    for path in Path(root).rglob("run.json"):
        row = json.loads(path.read_text()); row["artifact_dir"] = str(path.parent); runs.append(row)
    return runs


def aggregate_runs(runs, metric="recall@20"):
    groups = defaultdict(list)
    for run in runs:
        if run.get("status") == "complete":
            groups[(run.get("scale"), run.get("draw_seed"), run["variant"])].append(
                run["test_metrics"][metric])
    return [{"scale": k[0], "draw_seed": k[1], "variant": k[2], "metric": metric,
             "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0,
             "n_model_seeds": len(v)} for k, v in sorted(groups.items())]


def crossover(rows, baseline="gru4rec", challenger="pure_ssm"):
    values = {(r["draw_seed"], r["scale"], r["variant"]): r["mean"] for r in rows}
    draws = sorted({r["draw_seed"] for r in rows})
    output = []
    for draw in draws:
        points = []
        for scale in sorted({r["scale"] for r in rows if r["draw_seed"] == draw}):
            a, b = values.get((draw, scale, challenger)), values.get((draw, scale, baseline))
            if a is not None and b is not None: points.append((scale, a - b))
        crossings = []
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            if y1 == 0:
                crossings.append({"threshold": float(x1), "interval": [x1, x1],
                                  "direction": "touch"})
                continue
            if y1 * y2 < 0:
                threshold = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                crossings.append({"threshold": threshold, "interval": [x1, x2],
                                  "direction": "negative_to_positive" if y1 < y2 else "positive_to_negative"})
        sustained = None
        for crossing_index, crossing in enumerate(crossings):
            if crossing["direction"] != "negative_to_positive":
                continue
            later = [delta for scale, delta in points if scale >= crossing["interval"][1]]
            if later and all(delta > 0 for delta in later):
                sustained = crossing
                break
        output.append({"draw_seed": draw, "points": points, "crossings": crossings,
                       "sustained_crossover": sustained})
    valid = [x["sustained_crossover"]["threshold"] for x in output
             if x["sustained_crossover"] is not None]
    return {"baseline": baseline, "challenger": challenger, "draws": output,
            "n_draws_with_sustained_crossover": len(valid),
            "median_sustained_threshold": float(np.median(valid)) if valid else None,
            "definition": "A sustained crossover remains positive at every larger evaluated scale."}


def _query_ensemble(run_group, metric):
    """Average a metric per query across model seeds in one data draw."""
    values = defaultdict(list)
    for run in run_group:
        rows = json.loads((Path(run["artifact_dir"]) / "per_query.json").read_text())
        for row in rows:
            values[str(row["query_id"])].append(float(row[metric]))
    return [{"query_id": query_id, metric: float(np.mean(scores))}
            for query_id, scores in sorted(values.items())]


def paired_inference(runs, baseline="gru4rec", challenger="pure_ssm",
                     metric="recall@20", samples=10000, seed=2026):
    """Paired inference with data draws outside model-seed ensembles.

    Each data draw is treated as an independent outer replication. Within a
    draw, per-query outcomes are averaged over model seeds before bootstrap so
    queries are never incorrectly counted once per seed.
    """
    groups = defaultdict(list)
    for run in runs:
        if run.get("status") == "complete" and run.get("variant") in (baseline, challenger):
            groups[(run.get("scale"), run.get("draw_seed"), run["variant"])].append(run)

    draw_results = []
    for scale, draw in sorted({(key[0], key[1]) for key in groups}):
        base = groups.get((scale, draw, baseline), [])
        chall = groups.get((scale, draw, challenger), [])
        if not base or not chall:
            continue
        base_rows = _query_ensemble(base, metric)
        chall_rows = _query_ensemble(chall, metric)
        # paired_bootstrap returns A-B, so challenger is deliberately first.
        result = paired_bootstrap(chall_rows, base_rows, metric, samples,
                                  seed + int(scale) + int(draw))
        result.update({"scale": scale, "draw_seed": draw,
                       "n_baseline_seeds": len(base),
                       "n_challenger_seeds": len(chall)})
        draw_results.append(result)

    scales = []
    for scale in sorted({row["scale"] for row in draw_results}):
        rows = [row for row in draw_results if row["scale"] == scale]
        differences = np.asarray([row["difference"] for row in rows], dtype=float)
        scales.append({
            "scale": scale,
            "n_data_draws": len(rows),
            "mean_difference": float(differences.mean()),
            "std_across_draws": float(differences.std(ddof=1)) if len(rows) > 1 else 0.0,
            "draw_win_rate": float(np.mean(differences > 0)),
            "all_draws_positive": bool(np.all(differences > 0)),
            "all_draws_negative": bool(np.all(differences < 0)),
        })
    return {"baseline": baseline, "challenger": challenger, "metric": metric,
            "bootstrap_samples": samples, "draws": draw_results, "scales": scales,
            "interpretation": "Outer replications are data draws; model seeds are ensembled within each draw."}


def variance_decomposition(runs, variants=("gru4rec", "pure_ssm"), metric="recall@20"):
    """Method-of-moments split of draw and model-seed variance by scale/model."""
    grouped = defaultdict(lambda: defaultdict(list))
    for run in runs:
        if run.get("status") == "complete" and run.get("variant") in variants:
            grouped[(run.get("scale"), run["variant"])][run.get("draw_seed")].append(
                float(run["test_metrics"][metric]))
    output = []
    for (scale, variant), draws in sorted(grouped.items()):
        means = np.asarray([np.mean(values) for values in draws.values()], dtype=float)
        within_vars = [np.var(values, ddof=1) for values in draws.values() if len(values) > 1]
        within = float(np.mean(within_vars)) if within_vars else 0.0
        observed_between = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
        mean_seed_count = float(np.mean([len(values) for values in draws.values()]))
        between = max(0.0, observed_between - within / max(mean_seed_count, 1.0))
        total = between + within
        output.append({"scale": scale, "variant": variant, "metric": metric,
                       "n_data_draws": len(draws), "mean_model_seeds": mean_seed_count,
                       "between_draw_variance": between,
                       "within_draw_model_seed_variance": within,
                       "draw_variance_fraction": between / total if total else 0.0})
    return output


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
