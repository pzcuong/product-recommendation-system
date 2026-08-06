import unittest

import numpy as np

from dynamic_beta import (
    FEATURE_NAMES,
    TrainOnlyDynamicBeta,
    TrainOnlyGlobalBeta,
    fuse_with_dynamic_beta,
    rank_evidence_training_arrays,
)
from run_dynamic_beta import (
    canonical_validation_sources,
    make_training_oof_split,
)


class DynamicBetaTest(unittest.TestCase):
    def test_validation_sources_are_excluded_for_both_id_conventions(self):
        sources = canonical_validation_sources({
            "amazon-user": {"context": [1], "targets": [2]},
            "hid_train_7_v": {"context": [3], "targets": [4]},
        })
        self.assertEqual(sources, {"amazon-user", "hid_train_7"})
        sessions = {
            "amazon-user": [1, 2, 3],
            "hid_train_7": [3, 4, 5],
            "other-a": [6, 7, 8],
            "other-b": [9, 10, 11],
        }
        inner, profile, gate, report = make_training_oof_split(
            sessions,
            sources,
            fraction=1.0,
            cap=4,
            profile_cap=1,
        )
        held_queries = set(profile) | set(gate)
        self.assertFalse(any("amazon-user" in key for key in held_queries))
        self.assertFalse(any("hid_train_7" in key for key in held_queries))
        self.assertIn("amazon-user", inner)
        self.assertIn("hid_train_7", inner)
        self.assertEqual(report["declared_validation_source_overlap"], 0)

    def test_rank_evidence_distinguishes_memory_and_neural_rescues(self):
        memory = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
        neural = np.asarray([[9, 10, 11, 12], [1, 2, 3, 4]], dtype=np.int32)
        targets = np.asarray([1, 1], dtype=np.int32)
        target, _, _, actionable = rank_evidence_training_arrays(
            memory, neural, targets, hard_negatives=2)
        self.assertTrue(np.all(actionable))
        self.assertGreater(target[0, 0], target[0, 1])
        self.assertGreater(target[1, 1], target[1, 0])

    def test_gate_fit_predict_and_fuse(self):
        rng = np.random.default_rng(42)
        n = 200
        features = rng.standard_normal(
            (n, len(FEATURE_NAMES))).astype(np.float32)
        memory = np.asarray([
            rng.choice(np.arange(1, 300), size=120, replace=False)
            for _ in range(n)
        ], dtype=np.int32)
        neural = np.asarray([
            rng.choice(np.arange(1, 300), size=120, replace=False)
            for _ in range(n)
        ], dtype=np.int32)
        targets = np.asarray([
            memory[row, 0] if row < n // 2 else neural[row, 0]
            for row in range(n)
        ], dtype=np.int32)
        gate = TrainOnlyDynamicBeta(epochs=5, hidden=8, seed=42)
        report = gate.fit(features, memory, neural, targets)
        beta = gate.predict(features)
        self.assertEqual(len(beta), n)
        self.assertTrue(np.all(beta >= 0.0))
        self.assertTrue(np.all(beta <= 1.0))
        self.assertFalse(report["training_uses_validation_labels"])
        self.assertFalse(report["beta_search"])
        ranking = fuse_with_dynamic_beta(memory, neural, beta)
        self.assertEqual(ranking.shape, (n, 20))
        self.assertTrue(np.all(ranking > 0))
        self.assertTrue(np.all(ranking < 300))

    def test_neither_source_target_is_not_actionable(self):
        memory = np.asarray([[10, 20, 30, 40]], dtype=np.int32)
        neural = np.asarray([[50, 60, 70, 80]], dtype=np.int32)
        targets = np.asarray([99], dtype=np.int32)
        target, _, _, actionable = rank_evidence_training_arrays(
            memory, neural, targets, hard_negatives=2)
        self.assertTrue(np.all(target == 0.0))
        self.assertFalse(actionable[0])

    def test_global_beta_is_continuous_and_training_only(self):
        memory = np.tile(np.arange(1, 121), (20, 1)).astype(np.int32)
        neural = np.tile(np.arange(121, 241), (20, 1)).astype(np.int32)
        targets = np.asarray([1] * 10 + [121] * 10, dtype=np.int32)
        model = TrainOnlyGlobalBeta(epochs=5, seed=42)
        report = model.fit(memory, neural, targets)
        beta = model.predict(3)
        self.assertTrue(np.all(beta > 0.0))
        self.assertTrue(np.all(beta < 1.0))
        self.assertFalse(report["training_uses_validation_labels"])
        self.assertFalse(report["beta_search"])


if __name__ == "__main__":
    unittest.main()
