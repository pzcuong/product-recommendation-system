import numpy as np

import cearf
from cearf_v3_ext import (
    CEARFIndexV3,
    PROFILES_V3,
    SemanticMemory,
    repeat_ranking,
)


def test_v2_profiles_are_exact_v3_subfamily():
    # 6-slot family (transition, session, popularity, semantic_raw, repeat,
    # casm); legacy profiles are zero-padded so containment is preserved.
    for name, profile in cearf.PROFILES.items():
        assert PROFILES_V3[name] == tuple(profile) + (0.0, 0.0, 0.0)


def test_repeat_ranking_rewards_recency_and_frequency():
    ranking = repeat_ranking([2, 3, 2, 4], n_items=8)
    assert ranking[:2] == [4, 2]
    assert set(ranking) == {2, 3, 4}


def test_repeat_is_off_under_exclude_seen():
    sessions = {"s": [1, 2, 3]}
    index = CEARFIndexV3(
        sessions, 6, cearf.CEARFConfig(exclude_seen=True), use_repeat=True)
    components = index.component_rankings([1, 2])
    assert len(components) == 6
    repeat = components[4]
    assert repeat == []


def test_semantic_memory_does_not_return_padding_or_zero_vectors():
    vectors = np.asarray([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.9, 0.1],
    ], dtype=np.float32)
    memory = SemanticMemory(vectors, topn=3)
    assert memory.ranking([1], blocked=set()) == [1, 3]


def test_semantic_row_count_must_match_catalogue():
    vectors = np.zeros((4, 2), dtype=np.float32)
    try:
        CEARFIndexV3({"s": [1, 2]}, 5, semantic=SemanticMemory(vectors))
    except ValueError as exc:
        assert "semantic rows" in str(exc)
    else:
        raise AssertionError("expected semantic/catalogue mismatch to fail")
