from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from .analysis import (aggregate_runs, collect_runs, crossover, paired_inference,
                       variance_decomposition, write_csv)
from .audit import audit
from .data import coverage_stats, create_manifest, load_diginetica, materialize, write_json
from .runner import load_config, train_run

ROOT = Path(__file__).resolve().parent


def parse_ints(value): return [int(x) for x in value.split(",") if x]


def manifests(args):
    data = load_diginetica(); out = Path(args.output)
    for scale, draw in itertools.product(parse_ints(args.scales), parse_ints(args.draws)):
        manifest = create_manifest(data, scale, draw, args.test_limit, args.validation_fraction)
        path = out / f"scale_{scale}" / f"draw_{draw}.json"; write_json(path, manifest)
        train, _, test = materialize(data, manifest)
        write_json(path.with_name(f"draw_{draw}_coverage.json"), coverage_stats(train, test))


def run(args):
    data = load_diginetica(); manifest = json.loads(Path(args.manifest).read_text())
    train, val, test = materialize(data, manifest); config = load_config(args.config)
    out = Path(args.output) / f"scale_{manifest['scale']}" / f"draw_{manifest['draw_seed']}" \
        / args.variant / f"seed_{args.seed}"
    result = train_run(args.variant, train, val, test, data["n_items"], config, args.seed, out,
                       manifest["draw_seed"], manifest["scale"])
    print(json.dumps(result["test_metrics"], indent=2))


def suite(args):
    manifests_dir = Path(args.manifests); variants = args.variants.split(",")
    seeds = parse_ints(args.seeds)
    for path in sorted(manifests_dir.glob("scale_*/draw_*.json")):
        if path.name.endswith("coverage.json"): continue
        for variant, seed in itertools.product(variants, seeds):
            run(argparse.Namespace(manifest=str(path), config=args.config, output=args.output,
                                   variant=variant, seed=seed))


def tune(args):
    data = load_diginetica(); manifest = json.loads(Path(args.manifest).read_text())
    train, val, test = materialize(data, manifest); search = load_config(args.search_space)
    base = search["base"]; trials = search["trials"][:args.max_trials]
    for variant in args.variants.split(","):
        scored = []
        for index, override in enumerate(trials):
            config = {**base, **override}
            out = Path(args.output) / variant / f"trial_{index}"
            artifact = train_run(variant, train, val, test, data["n_items"], config,
                                 args.seed, out, manifest["draw_seed"], manifest["scale"], index)
            scored.append({"trial": index, "config": config,
                           "validation": artifact["best_validation_metric"]})
        scored.sort(key=lambda x: x["validation"], reverse=True)
        write_json(Path(args.output) / variant / "selection.json",
                   {"selection_uses_validation_only": True, "trials": scored,
                    "best_config": scored[0]["config"]})


def analyze(args):
    runs = collect_runs(args.results); rows = aggregate_runs(runs, args.metric)
    write_csv(Path(args.output) / "aggregated.csv", rows)
    write_json(Path(args.output) / "crossover.json",
               crossover(rows, args.baseline, args.challenger))
    write_json(Path(args.output) / "paired_inference.json",
               paired_inference(runs, args.baseline, args.challenger, args.metric,
                                args.bootstrap_samples))
    write_json(Path(args.output) / "variance_decomposition.json",
               variance_decomposition(runs, (args.baseline, args.challenger), args.metric))


def decision(args):
    rows = aggregate_runs(collect_runs(args.results), args.metric)
    lookup = {(r["scale"], r["variant"]): r["mean"] for r in rows if r["draw_seed"] == args.draw}
    scales = sorted({r["scale"] for r in rows})
    pure_reversal = any(lookup.get((s, "pure_ssm"), -1) > lookup.get((s, "gru4rec"), 2)
                        for s in scales[1:]) and lookup.get((scales[0], "pure_ssm"), 2) < lookup.get((scales[0], "gru4rec"), -1)
    hybrid_wins = any(lookup.get((s, "fe_gru_ssm"), -1) > lookup.get((s, "gru4rec"), 2) for s in scales)
    fe_not_worse = any(lookup.get((s, "fe_gru"), -1) >= lookup.get((s, "fe_gru_ssm"), 2) for s in scales)
    if pure_reversal: outcome, framing = "A", "selective state-space backbone"
    elif fe_not_worse: outcome, framing = "C", "SSM attribution unsupported; reframe or stop"
    elif hybrid_wins: outcome, framing = "B", "hybrid recurrent-state-space backbone"
    else: outcome, framing = "INCONCLUSIVE", "collect more evidence"
    write_json(Path(args.output), {"outcome": outcome, "framing": framing, "metric": args.metric,
                                   "warning": "Automated screen; confirm uncertainty across draws and seeds."})


def audit_cmd(args):
    result = audit(args.results); print(json.dumps(result, indent=2))
    if not result["ok"]: raise SystemExit(1)


def parser():
    p = argparse.ArgumentParser(description="Eight-phase SSM scaling-study pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("manifests", help="Phase 0/2: split manifests and coverage")
    x.add_argument("--scales", default="10000,12000,15000"); x.add_argument("--draws", default="42,123,456")
    x.add_argument("--test-limit", type=int, default=1500); x.add_argument("--validation-fraction", type=float, default=.1)
    x.add_argument("--output", default=str(ROOT / "artifacts/manifests")); x.set_defaults(func=manifests)
    x = sub.add_parser("run", help="Phase 1/3/5/6: one traceable training run")
    x.add_argument("--manifest", required=True); x.add_argument("--config", required=True)
    x.add_argument("--variant", choices=sorted(__import__("sparse_bench.scaling_study.models", fromlist=["SessionModel"]).SessionModel.VALID), required=True)
    x.add_argument("--seed", type=int, required=True); x.add_argument("--output", default=str(ROOT / "artifacts/runs")); x.set_defaults(func=run)
    x = sub.add_parser("suite", help="Phase 3/5: factorial ablation or crossover suite")
    x.add_argument("--manifests", required=True); x.add_argument("--config", required=True)
    x.add_argument("--variants", default="gru4rec,pure_ssm,fe_gru,fe_gru_ssm")
    x.add_argument("--seeds", default="42,123,456"); x.add_argument("--output", default=str(ROOT / "artifacts/runs")); x.set_defaults(func=suite)
    x = sub.add_parser("tune", help="Phase 4/6: equal-budget validation-only tuning")
    x.add_argument("--manifest", required=True); x.add_argument("--search-space", required=True)
    x.add_argument("--variants", default="gru4rec,pure_ssm,fe_gru,fe_gru_ssm,sasrec")
    x.add_argument("--max-trials", type=int, default=12); x.add_argument("--seed", type=int, default=42)
    x.add_argument("--output", default=str(ROOT / "artifacts/tuning")); x.set_defaults(func=tune)
    x = sub.add_parser("analyze", help="Phase 5: aggregate and estimate observed crossover")
    x.add_argument("--results", required=True); x.add_argument("--metric", default="recall@20")
    x.add_argument("--baseline", default="gru4rec"); x.add_argument("--challenger", default="pure_ssm")
    x.add_argument("--bootstrap-samples", type=int, default=10000)
    x.add_argument("--output", default=str(ROOT / "artifacts/analysis")); x.set_defaults(func=analyze)
    x = sub.add_parser("decision", help="Phase 3 gate: select scientifically valid framing")
    x.add_argument("--results", required=True); x.add_argument("--metric", default="recall@20")
    x.add_argument("--draw", type=int, default=42); x.add_argument("--output", default=str(ROOT / "artifacts/decision.json")); x.set_defaults(func=decision)
    x = sub.add_parser("audit", help="Phase 8: artifact completeness audit")
    x.add_argument("--results", required=True); x.set_defaults(func=audit_cmd)
    return p


def main():
    args = parser().parse_args(); args.func(args)


if __name__ == "__main__": main()
