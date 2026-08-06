from collections import Counter

import cearf
from run_cearfn_evidence import build_memory_arrays, popularity_partition
from run_cearfn_v2 import build_features, build_features_from_memory_arrays


def test_router_features_reuse_memory_component_ranks_exactly():
    sessions = {
        "s1": [1, 2, 3, 4],
        "s2": [1, 3, 5],
        "s3": [2, 3, 6],
        "s4": [2, 5, 7],
    }
    queries = {
        "q1": {"context": [1, 2], "targets": [3]},
        "q2": {"context": [2, 3], "targets": [6]},
        "q3": {"context": [1, 3], "targets": [5]},
    }
    index = cearf.CEARFIndex(sessions, n_items=8)
    profiles = {
        "short": (0.5, 0.4, 0.1),
        "long": (0.6, 0.4, 0.0),
    }
    arrays = build_memory_arrays(
        index, queries, profiles, width=7, label="reuse-test")
    keys = [str(value) for value in arrays["keys"]]
    freq = Counter(item for sequence in sessions.values() for item in sequence)
    head, _, _ = popularity_partition(freq, n_items=8)
    head_set = set(head.tolist())

    direct = build_features(index, queries, freq, head_set)
    reused = build_features_from_memory_arrays(
        queries, keys, arrays, freq, head_set)

    assert reused == direct
