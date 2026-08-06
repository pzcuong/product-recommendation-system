#!/usr/bin/env python3
"""Summarize the selected/runner-up/prototype-on PASGR stability audit."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "pasgr_stability_artifacts"
SEEDS = (42, 123, 456)
CELLS = {
    "Video_Games": {
        "selected": "graph0.35_noproto_contrast0.15_inbatch0.1",
        "runner_up": "graph0_noproto_contrast0.15_inbatch0.1",
        "best_prototype_on": "graph0.35_proto_contrast0_inbatch0",
    },
    "Baby_Products": {
        "selected": "graph0_noproto_contrast0.15_inbatch0",
        "runner_up": "graph0_noproto_contrast0.15_inbatch0.1",
        "best_prototype_on": "graph0_proto_contrast0.15_inbatch0.1",
    },
    "Diginetica_HID": {
        "selected": "graph0.35_noproto_contrast0_inbatch0.1",
        "runner_up_and_best_prototype_on":
            "graph0.35_proto_contrast0.15_inbatch0.1",
    },
}


def seed42_utilities() -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for domain, source in (
        ("Video_Games", HERE / "vgated_amazon.log"),
        ("Baby_Products", HERE / "vgated_baby.log"),
    ):
        domain_values = {}
        prefix = f"[VGATED] DONE {domain} "
        for line in source.read_text().splitlines():
            if not line.startswith(prefix):
                continue
            label, tail = line[len(prefix):].split(" valid_util=", 1)
            domain_values[label] = float(tail.split()[0])
        values[domain] = domain_values

    canonical = json.loads((HERE / "pasgr_config_per_domain.json").read_text())
    values["Diginetica_HID"] = {
        label: float(cell["validation"]["utility"])
        for label, cell in canonical["Diginetica_HID"]["grid"].items()
    }
    return values


def main() -> None:
    seed42 = seed42_utilities()
    result = {
        "purpose": (
            "Cross-seed stability of the seed-42 selected PASGR cell against "
            "the seed-42 runner-up and best prototype-on alternative."
        ),
        "selection_status": (
            "diagnostic only; the canonical cell remains locked from seed 42"
        ),
        "domains": {},
    }
    for domain, roles in CELLS.items():
        rows = {}
        for role, label in roles.items():
            utilities = {"42": seed42[domain][label]}
            for seed in SEEDS[1:]:
                path = OUT_DIR / f"{domain}_seed{seed}.json"
                block = json.loads(path.read_text())[domain]
                utilities[str(seed)] = float(
                    block["grid"][label]["validation"]["utility"]
                )
            utilities["mean"] = sum(utilities.values()) / len(SEEDS)
            rows[role] = {"label": label, "validation_utility": utilities}
        best_mean = max(rows, key=lambda role: rows[role]["validation_utility"]["mean"])
        result["domains"][domain] = {
            "cells": rows,
            "best_mean_role": best_mean,
            "seed42_selection_retained_by_mean": best_mean == "selected",
        }

    output = HERE / "pasgr_stability_audit.json"
    output.write_text(json.dumps(result, indent=2))
    print(output)


if __name__ == "__main__":
    main()
