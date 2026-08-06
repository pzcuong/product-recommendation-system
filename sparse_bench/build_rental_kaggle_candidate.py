#!/usr/bin/env python3
"""Build a conservative DualTwin + session-memory Kaggle candidate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def parse(frame):
    return {str(row.visit_id): str(row.product_ids).split()
            for row in frame.itertuples(index=False)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path,
                        default=Path("submission-044.csv"))
    parser.add_argument("--memory", type=Path,
                        default=Path("baseline_subs/submission_sknn.csv"))
    parser.add_argument("--primary-weight", type=float, default=0.85)
    parser.add_argument("--constant", type=float, default=20.0)
    parser.add_argument("--output", type=Path,
                        default=Path("kaggle_submission_cearf_dualtwin_candidate.csv"))
    args = parser.parse_args()
    primary_frame = pd.read_csv(args.primary, dtype=str, keep_default_na=False)
    memory_frame = pd.read_csv(args.memory, dtype=str, keep_default_na=False)
    primary, memory = parse(primary_frame), parse(memory_frame)
    if set(primary) != set(memory):
        raise ValueError("submission visit sets differ")
    popularity = Counter(item for ranking in primary.values() for item in ranking)
    fallback = [item for item, _ in popularity.most_common()]
    rows = []
    for uid in primary_frame["visit_id"].astype(str):
        score = defaultdict(float)
        for weight, ranking in (
                (args.primary_weight, primary[uid]),
                (1.0 - args.primary_weight, memory[uid])):
            for rank, item in enumerate(ranking, 1):
                score[item] += weight / (args.constant + rank)
        ranking = [item for item, _ in sorted(
            score.items(), key=lambda pair: (-pair[1], pair[0]))]
        ranking.extend(item for item in fallback if item not in set(ranking))
        rows.append({"visit_id": uid, "product_ids": " ".join(ranking[:6])})
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    changed = sum(primary[uid][:6] != output.iloc[row].product_ids.split()
                  for row, uid in enumerate(primary_frame["visit_id"].astype(str)))
    print({"output": str(args.output), "rows": len(output),
           "changed_rows_vs_primary": changed,
           "primary_weight": args.primary_weight}, flush=True)


if __name__ == "__main__":
    main()
