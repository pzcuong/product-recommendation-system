#!/usr/bin/env python3
"""Recompute validation-to-test admission transfer from frozen result artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CORE_PATH = ROOT / "cearfn_v2_nested_results.json"
EXTERNAL_PATH = ROOT / "hard_gate_singleton_audit.json"
OUTPUT_PATH = ROOT / "validation_test_transfer_audit.json"


def utility(metrics: dict[str, float]) -> float:
    return 0.5 * metrics["recall@6"] + 0.5 * metrics["recall@20"]


def summarize(records: list[dict]) -> dict:
    admitted = [record for record in records if record["admitted"]]
    retained = [record for record in admitted if record["test_retention_margin"] >= 0]
    reversed_records = [record for record in admitted if record["reversal"]]
    n = len(admitted)
    return {
        "eligible_decisions": len(records),
        "admitted_decisions": n,
        "retained_nonnegative": len(retained),
        "reversals": len(reversed_records),
        "retention_rate": len(retained) / n,
        "reversal_rate": len(reversed_records) / n,
        "mean_validation_admission_margin": sum(
            record["validation_admission_margin"] for record in admitted
        )
        / n,
        "mean_test_retention_margin": sum(
            record["test_retention_margin"] for record in admitted
        )
        / n,
        "mean_transfer_gap_R_minus_A": sum(
            record["test_retention_margin"]
            - record["validation_admission_margin"]
            for record in admitted
        )
        / n,
    }


def main() -> None:
    core = json.loads(CORE_PATH.read_text())
    external = json.loads(EXTERNAL_PATH.read_text())

    router_records = []
    for domain, domain_data in core.items():
        for run in domain_data["runs"]:
            selected = run["selected_router"]
            variants = run["router_selection"]["variants"]
            admission_margin = variants[selected]["utility"] - variants["regime"]["utility"]
            retention_margin = utility(run[selected]) - utility(run["regime"])
            admitted = selected != "regime" and admission_margin > 0
            router_records.append(
                {
                    "stage": "adaptive_router",
                    "domain": domain,
                    "seed": run["seed"],
                    "selected": selected,
                    "null": "regime",
                    "validation_admission_margin": admission_margin,
                    "test_retention_margin": retention_margin,
                    "admitted": admitted,
                    "reversal": admitted and retention_margin < 0,
                }
            )

    external_records = []
    for domain, domain_data in external["domains"].items():
        for run in domain_data["runs"]:
            admission_margin = run["validation_admission_margin"]
            retention_margin = run["test_retention_margin"]
            admitted = admission_margin > 0
            external_records.append(
                {
                    "stage": "external_expert_gate",
                    "domain": domain,
                    "seed": run["seed"],
                    "selected": run["selected"],
                    "null": run["best_validation_singleton"],
                    "validation_admission_margin": admission_margin,
                    "test_retention_margin": retention_margin,
                    "admitted": admitted,
                    "reversal": admitted and retention_margin < 0,
                }
            )

    admitted_records = [
        record
        for record in router_records + external_records
        if record["admitted"]
    ]
    output = {
        "objective": "U = 0.5*Recall@6 + 0.5*Recall@20",
        "selection_uses_test_labels": False,
        "interpretation": (
            "Descriptive transfer audit over locked per-seed decisions; test "
            "outcomes are never used for reselection."
        ),
        "sources": [CORE_PATH.name, EXTERNAL_PATH.name],
        "stages": {
            "adaptive_router": {
                "summary": summarize(router_records),
                "records": router_records,
            },
            "external_expert_gate": {
                "summary": summarize(external_records),
                "records": external_records,
            },
        },
        "all_admitted_decisions": summarize(admitted_records),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
