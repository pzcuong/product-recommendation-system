from __future__ import annotations

import json
from pathlib import Path

REQUIRED = {"run.json", "rankings.json", "per_query.json", "checkpoint.pt"}


def audit(root):
    issues, complete = [], 0
    for path in Path(root).rglob("run.json"):
        complete += 1; directory = path.parent
        missing = REQUIRED - {x.name for x in directory.iterdir()}
        if missing: issues.append(f"{directory}: missing {sorted(missing)}")
        run = json.loads(path.read_text())
        for key in ("variant", "seed", "config", "parameters", "best_epoch", "test_metrics"):
            if key not in run: issues.append(f"{path}: missing field {key}")
        rows = json.loads((directory / "per_query.json").read_text()) if not missing else []
        if len(rows) != run.get("test_metrics", {}).get("n_queries"):
            issues.append(f"{directory}: query count mismatch")
    if complete == 0: issues.append("no completed run artifacts found")
    return {"ok": not issues, "runs_checked": complete, "issues": issues}
