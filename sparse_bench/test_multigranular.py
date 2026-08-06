from multigranular import MultiGranularIntentIndex, fuse_three


def test_exact_suffix_dominates_shorter_backoff():
    sessions = {
        "a": [1, 2, 9], "b": [1, 2, 9], "c": [7, 2, 8],
        "d": [1, 3, 4],
    }
    index = MultiGranularIntentIndex(sessions, 12)
    assert index.predict_one([1, 2], 3)[0] == 9


def test_repeated_target_is_retained():
    index = MultiGranularIntentIndex({"a": [1, 2, 1]}, 5)
    assert index.predict_one([1, 2], 1) == [1]


def test_three_way_fusion_rewards_consensus():
    fused = fuse_three([1, 2], [1, 3], [1, 4], (1/3, 1/3, 1/3), 2)
    assert fused[0] == 1
