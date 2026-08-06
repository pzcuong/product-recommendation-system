#!/usr/bin/env python3
"""Extract canonical CEARF-N wall-clock measurements without rerunning models."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent


def main() -> None:
    source = HERE / "cearfn_v2_nested_results.json"
    raw = json.loads(source.read_text())
    output = {
        "source": source.name,
        "hardware": "Apple M2 Pro, 32 GB; CEARF on one CPU process; PASGR on Metal",
        "timer_scope": {
            "locked_seed_pass": (
                "PASGR tune/full training, validation/test neural prediction, "
                "router fitting/selection, fusion, metrics, and rank-artifact write; "
                "excludes shared memory/index setup"
            ),
            "shared_setup": (
                "data loading, CEARF tune/final index construction, profile tuning, "
                "memory rank generation/cache loading, and query-feature construction"
            ),
            "complete_three_seed_invocation": (
                "shared setup plus the three locked-seed passes"
            ),
        },
        "domains": {},
    }
    for domain, block in raw.items():
        seed_seconds = [float(run["seconds"]) for run in block["runs"]]
        total_seconds = float(block["seconds_total"])
        shared_seconds = total_seconds - sum(seed_seconds)
        output["domains"][domain] = {
            "seed_seconds": seed_seconds,
            "mean_locked_seed_minutes": mean(seed_seconds) / 60,
            "shared_setup_minutes": shared_seconds / 60,
            "complete_three_seed_minutes": total_seconds / 60,
        }

    destination = HERE / "runtime_audit.json"
    destination.write_text(json.dumps(output, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
