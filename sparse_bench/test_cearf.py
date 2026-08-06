import unittest

import cearf
import pasgr
from collections import Counter
import numpy as np


class CEARFTests(unittest.TestCase):
    def setUp(self):
        self.sessions = {
            "a": [1, 2, 3], "b": [1, 2, 4], "c": [2, 5, 6],
            "d": [7, 2, 3], "e": [7, 5, 6],
        }

    def test_index_and_prediction(self):
        index = cearf.CEARFIndex(self.sessions, 8)
        ranking = index.predict_one([1, 2], cearf.PROFILES["balanced"], 5)
        self.assertTrue(ranking)
        self.assertNotIn(1, ranking)
        self.assertNotIn(2, ranking)
        self.assertEqual(len(ranking), len(set(ranking)))

    def test_validation_is_session_disjoint(self):
        train, valid = cearf.make_validation_split(self.sessions, .4, 2)
        self.assertEqual(len(valid), 2)
        for key, query in valid.items():
            self.assertEqual(train[key], query["context"])

    def test_tuning(self):
        index = cearf.CEARFIndex(self.sessions, 8)
        valid = {"q": {"context": [1, 2], "targets": [3]}}
        profiles, report = cearf.tune_profiles(index, valid)
        self.assertIn("short", profiles)
        self.assertEqual(report["short"]["n"], 1)

    def test_large_catalog_fallback(self):
        index = cearf.CEARFIndex({"a": [1, 2]}, 10000)
        ranking = index.predict_one([9999], cearf.PROFILES["balanced"], 20)
        self.assertEqual(len(ranking), 2)
        self.assertEqual(len(set(ranking)), 2)

    def test_repeat_policy_is_protocol_configurable(self):
        sessions = {"a": [1, 1, 2], "b": [1, 1, 3]}
        filtered = cearf.CEARFIndex(sessions, 10)
        repeated = cearf.CEARFIndex(
            sessions, 10, cearf.CEARFConfig(exclude_seen=False))
        self.assertNotIn(1, filtered.predict_one([1], cearf.PROFILES["transition"], 3))
        self.assertIn(1, repeated.predict_one([1], cearf.PROFILES["transition"], 3))

    def test_transition_window_accumulates_every_distance(self):
        index = cearf.CEARFIndex(
            {"a": [1, 2, 3]}, 10,
            cearf.CEARFConfig(window=2, exclude_seen=False))
        self.assertAlmostEqual(index.transition[2][3], 1.0)
        self.assertAlmostEqual(index.transition[1][3], 0.5)

    def test_pasgr_prototype_transport_is_ablatable(self):
        sessions = {"a": [1, 2, 3], "b": [1, 2, 4]}
        semantic = np.eye(6, dtype=np.float32)
        freq = Counter(x for seq in sessions.values() for x in seq)
        full = pasgr.build_prototype_graph_embeddings(
            sessions, 6, freq, semantic,
            pasgr.PASGRConfig(dim=6, prototypes=2, seed=1,
                              graph_weight=0, prototype_transport=True))[0]
        ablated = pasgr.build_prototype_graph_embeddings(
            sessions, 6, freq, semantic,
            pasgr.PASGRConfig(dim=6, prototypes=2, seed=1,
                              graph_weight=0, prototype_transport=False))[0]
        self.assertFalse(np.allclose(full[1:], ablated[1:]))


if __name__ == "__main__":
    unittest.main()
