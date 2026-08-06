import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from run_cearfn_evidence import query_fingerprint
from run_dynamic_beta_expert_swap import (
    PRIMARY_FEATURE_NAMES,
    align_rankings,
    canonical_profiles,
    context_feature_matrix,
    load_primary_memory_cache,
    load_prediction_cache,
    predict_array_target_free,
)


class DynamicBetaExpertSwapTests(unittest.TestCase):
    def test_profiles_remain_equal_after_json_round_trip(self):
        profiles = canonical_profiles(
            {
                "short": (0, 1, 0),
                "long": np.asarray([0.0, 1.0, 0.0]),
            }
        )
        self.assertEqual(json.loads(json.dumps(profiles)), profiles)
        self.assertEqual(
            profiles,
            {
                "short": [0.0, 1.0, 0.0],
                "long": [0.0, 1.0, 0.0],
            },
        )

    def test_narm_prediction_does_not_require_query_targets(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.0))

            def logits(self, contexts, lengths):
                del lengths
                base = torch.arange(
                    8, dtype=torch.float32, device=contexts.device)
                return base[None, :].repeat(len(contexts), 1)

        queries = {
            "q1": {"context": [1, 2]},
            "q2": {"context": [3]},
        }
        keys, rankings, _ = predict_array_target_free(
            DummyModel(),
            queries,
            n_items=8,
            topk=3,
            batch_size=2,
            exclude_seen=True,
        )
        self.assertEqual(keys, ["q1", "q2"])
        self.assertEqual(rankings.shape, (2, 3))
        self.assertNotIn(1, rankings[0])
        self.assertNotIn(2, rankings[0])

    def test_context_features_are_target_free_and_exact(self):
        queries_a = {
            "q1": {"context": [1, 2, 2], "targets": [8]},
            "q2": {"context": [], "targets": [9]},
        }
        queries_b = {
            key: {**value, "targets": [999]}
            for key, value in queries_a.items()
        }
        frequency = {2: 7}
        expected = np.asarray(
            [
                [math.log1p(3), math.log1p(7), 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        a = context_feature_matrix(
            queries_a, ["q1", "q2"], frequency, {1})
        b = context_feature_matrix(
            queries_b, ["q1", "q2"], frequency, {1})
        np.testing.assert_allclose(a, expected)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(
            PRIMARY_FEATURE_NAMES,
            (
                "log_context_length",
                "log_last_item_frequency",
                "last_item_is_tail",
            ),
        )

    def test_align_rankings_reorders_without_changing_rows(self):
        rankings = np.asarray([[30, 31], [10, 11], [20, 21]])
        aligned = align_rankings(
            ["a", "b", "c"], ["c", "a", "b"], rankings)
        np.testing.assert_array_equal(
            aligned,
            np.asarray([[10, 11], [20, 21], [30, 31]]),
        )

    def test_align_rankings_rejects_coverage_mismatch(self):
        with self.assertRaises(ValueError):
            align_rankings(
                ["a", "b"], ["a", "c"], np.asarray([[1], [2]]))

    def test_prediction_cache_checks_protocol_identity(self):
        queries = {
            "a": {"context": [1], "targets": [2]},
            "b": {"context": [3], "targets": [4]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            np.savez_compressed(
                path,
                keys=np.asarray(["a", "b"]),
                rankings=np.asarray([[5, 6], [7, 8]], dtype=np.int32),
                query_fingerprint=np.asarray(query_fingerprint(queries)),
                checkpoint_sha256=np.asarray("abc"),
                candidate_width=np.asarray(2),
                exclude_seen=np.asarray(True),
            )
            loaded = load_prediction_cache(
                path, queries, "abc", 2, True)
            self.assertIsNotNone(loaded)
            self.assertIsNone(load_prediction_cache(
                path, queries, "wrong", 2, True))
            self.assertIsNone(load_prediction_cache(
                path, queries, "abc", 2, False))

    def test_primary_memory_cache_is_strict_and_read_only(self):
        queries = {
            "a": {"context": [1], "targets": [2]},
            "b": {"context": [3], "targets": [4]},
        }
        profiles = {"short": "session", "long": "session"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.npz"
            np.savez_compressed(
                path,
                keys=np.asarray(["a", "b"]),
                selected=np.asarray([[5, 6], [7, 8]], dtype=np.int32),
                fingerprint=np.asarray(query_fingerprint(queries)),
                profiles=np.asarray(json.dumps(profiles, sort_keys=True)),
            )
            loaded = load_primary_memory_cache(
                path, queries, profiles, width=2)
            np.testing.assert_array_equal(
                loaded["selected"],
                np.asarray([[5, 6], [7, 8]], dtype=np.int32),
            )
            with self.assertRaises(RuntimeError):
                load_primary_memory_cache(
                    path, queries, {"short": "transition"}, width=2)


if __name__ == "__main__":
    unittest.main()
