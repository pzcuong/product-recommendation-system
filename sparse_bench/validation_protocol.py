"""Leakage-safe construction of training sessions for validation queries."""
from __future__ import annotations

from typing import Mapping, Sequence


def hold_out_validation_targets(
    sessions: Mapping[str, Sequence[int]], validation: Mapping[str, dict]
) -> dict[str, list[int]]:
    """Remove an in-session validation target from its source training row.

    Diginetica validation keys use ``<source_session>_v``. Amazon validation
    targets are external to the training sequence, so those rows are left
    unchanged. The strict context/target check prevents accidental truncation.
    """
    output = {str(uid): [int(x) for x in seq] for uid, seq in sessions.items()}
    for query_id, query in validation.items():
        key = str(query_id)
        if not key.endswith("_v"):
            continue
        source = key[:-2]
        if source not in output:
            continue
        context = [int(x) for x in query.get("context", ())]
        targets = [int(x) for x in query.get("targets", ())]
        sequence = output[source]
        if len(targets) == 1 and sequence == context + targets:
            output[source] = context
    return output
