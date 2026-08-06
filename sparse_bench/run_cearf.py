#!/usr/bin/env python3
"""Run CEARF and its matched retrieval baselines on unified domains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cearf
import loaders


HERE = Path(__file__).resolve().parent
DOMAINS = ["Rental_visit", "RetailRocket", "Video_Games", "Baby_Products"]


def metrics(predictions, queries):
    return cearf.ranking_metrics(predictions, queries)


def run_domain(domain: str, config: cearf.CEARFConfig) -> dict:
    started = time.time()
    data = loaders.ALL_LOADERS[domain]()
    sessions = data["train_sessions"]
    validation = data.get("valid_queries") or {}
    if validation:
        tune_sessions = sessions
        if len(validation) > config.validation_cap:
            keys = sorted(validation, key=cearf._stable_fraction)[:config.validation_cap]
            validation = {str(key): validation[key] for key in keys}
    else:
        tune_sessions, validation = cearf.make_validation_split(
            sessions, config.validation_fraction, config.validation_cap)
    print(f"[CEARF] {domain} tune_train={len(tune_sessions)} "
          f"valid={len(validation)} test={len(data['test_queries'])} "
          f"items={data['n_items']}", flush=True)
    tune_index = cearf.CEARFIndex(tune_sessions, data["n_items"], config)
    profiles, tuning = cearf.tune_profiles(tune_index, validation)
    print(f"[CEARF] {domain} profiles={tuning}", flush=True)
    del tune_index

    index = cearf.CEARFIndex(sessions, data["n_items"], config)
    test = data["test_queries"]
    predictions = {}
    transition_predictions = {}
    session_predictions = {}
    total = len(test)
    for row, (uid, query) in enumerate(test.items(), 1):
        context = query.get("context", [])
        rankings = index.component_rankings(context)
        regime = "short" if len(context) <= config.short_context else "long"
        predictions[str(uid)] = index.fuse_rankings(
            context, rankings, profiles[regime], 20)
        transition_predictions[str(uid)] = rankings[0][:20]
        session_predictions[str(uid)] = rankings[1][:20]
        if row % 10000 == 0:
            print(f"[CEARF] {domain} predicted={row}/{total}", flush=True)
    component_results = {
        "TransitionMemory": metrics(transition_predictions, test),
        "SessionMemory": metrics(session_predictions, test),
    }
    result = {
        "dataset": domain,
        "train_sessions": len(sessions),
        "validation_queries": len(validation),
        "test_queries": len(test),
        "n_items": data["n_items"],
        "tuning": tuning,
        "CEARF": metrics(predictions, test),
        "components": component_results,
        "seconds": time.time() - started,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", default=DOMAINS)
    parser.add_argument("--output", type=Path, default=HERE / "cearf_results.json")
    parser.add_argument("--validation-cap", type=int, default=5000)
    args = parser.parse_args()
    config = cearf.CEARFConfig(validation_cap=args.validation_cap)
    results = {}
    for domain in args.domains:
        results[domain] = run_domain(domain, config)
        args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
