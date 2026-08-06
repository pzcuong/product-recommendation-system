import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_dynamic_beta_allocation_controls import (
    CONTROL_ORDER,
    DYNAMIC_CONTROL_SPECS,
    FEATURE_NAMES,
    assert_saved_rank_identity,
    bucket_assignments,
    deterministic_beta_permutation,
    evaluate_after_manifest,
    fit_bucket_policy,
    fit_dynamic_control,
    paired_assignment_summary,
    predict_bucket_policy,
    primary_context_features,
    render_summary_tex,
    summarize_results,
)


def _synthetic_experts(n: int = 24):
    memory = np.tile(np.arange(1, 41), (n, 1)).astype(np.int32)
    neural = np.tile(np.arange(41, 81), (n, 1)).astype(np.int32)
    targets = np.asarray([
        1 if row % 2 == 0 else 41 for row in range(n)
    ], dtype=np.int32)
    return memory, neural, targets


class DynamicBetaAllocationControlsTest(unittest.TestCase):
    def test_saved_rank_identity_rejects_any_expert_rank_change(self):
        memory = {
            "keys": np.asarray(["a", "b"]),
            "selected": np.asarray([
                np.arange(1, 21), np.arange(21, 41)
            ], dtype=np.int32),
        }
        neural = {
            "keys": np.asarray(["a", "b"]),
            "rankings": np.asarray([
                np.arange(41, 61), np.arange(61, 81)
            ], dtype=np.int32),
        }
        primary = {
            "test_keys": np.asarray(["a", "b"]),
            "test_memory_only_top20": memory["selected"].copy(),
            "test_neural_only_top20": neural["rankings"].copy(),
        }
        self.assertEqual(
            assert_saved_rank_identity(memory, neural, primary),
            ["a", "b"],
        )
        primary["test_neural_only_top20"][1, 0] = 999
        with self.assertRaisesRegex(ValueError, "neural ranks changed"):
            assert_saved_rank_identity(memory, neural, primary)

    def test_primary_features_and_bucket_labels_are_target_free(self):
        queries = {
            "a": {"context": [1], "targets": [999]},
            "b": {"context": [2, 3, 4], "targets": [998]},
            "c": {"context": [], "targets": [997]},
        }
        features = primary_context_features(
            queries,
            ["a", "b", "c"],
            {1: 10, 4: 1},
            {1},
        )
        self.assertEqual(features.shape, (3, 3))
        self.assertAlmostEqual(features[0, 0], np.log1p(1), places=6)
        self.assertAlmostEqual(features[0, 1], np.log1p(10), places=6)
        self.assertEqual(features[0, 2], 0.0)
        self.assertEqual(features[1, 2], 1.0)
        head_tail, crossed = bucket_assignments(features, short_context=2)
        np.testing.assert_array_equal(
            head_tail, ["head", "tail", "head"])
        np.testing.assert_array_equal(
            crossed, ["short_head", "long_tail", "short_head"])

        # Targets are deliberately changed; features and assignments must not.
        for query in queries.values():
            query["targets"] = [-1]
        repeated = primary_context_features(
            queries,
            ["a", "b", "c"],
            {1: 10, 4: 1},
            {1},
        )
        np.testing.assert_array_equal(features, repeated)

    def test_bucket_policy_uses_global_fallback_for_unactionable_bucket(self):
        memory, neural, targets = _synthetic_experts(12)
        labels = np.asarray(["head"] * 12)
        models, report = fit_bucket_policy(
            labels,
            ("head", "tail"),
            memory,
            neural,
            targets,
            fallback_beta=0.4,
            seed=42,
            epochs=2,
        )
        self.assertIsNotNone(models["head"])
        self.assertIsNone(models["tail"])
        self.assertTrue(report["tail"]["fallback_to_oof_global"])
        predicted = predict_bucket_policy(
            models,
            np.asarray(["head", "tail"]),
            fallback_beta=0.4,
        )
        self.assertAlmostEqual(predicted[1], 0.4, places=6)
        self.assertAlmostEqual(
            predicted[0], float(models["head"].beta_), places=6)

    def test_dynamic_control_factorial_has_requested_specs_and_bounds(self):
        expected = {
            "dynamic_delta_005",
            "dynamic_delta_010",
            "dynamic_delta_020",
            "feature_length_only",
            "feature_frequency_only",
            "feature_tail_only",
            "feature_drop_length",
            "feature_drop_frequency",
            "feature_drop_tail",
            "regularization_no_admission_cost",
            "regularization_no_prior_penalty",
        }
        self.assertEqual(set(DYNAMIC_CONTROL_SPECS), expected)
        self.assertEqual(
            DYNAMIC_CONTROL_SPECS[
                "regularization_no_admission_cost"].admission_cost,
            0.0,
        )
        self.assertEqual(
            DYNAMIC_CONTROL_SPECS[
                "regularization_no_prior_penalty"].prior_penalty,
            0.0,
        )

        rng = np.random.default_rng(7)
        memory, neural, targets = _synthetic_experts(24)
        features = rng.normal(size=(24, 16)).astype(np.float32)
        initial = 0.4
        for name in (
                "dynamic_delta_005",
                "feature_tail_only",
                "regularization_no_prior_penalty"):
            spec = DYNAMIC_CONTROL_SPECS[name]
            model, report = fit_dynamic_control(
                spec,
                features,
                memory,
                neural,
                targets,
                initial_beta=initial,
                seed=42,
                epochs=2,
            )
            beta = model.predict(features[:, spec.columns])
            self.assertTrue(np.all(beta >= initial - spec.max_residual - 1e-6))
            self.assertTrue(np.all(beta <= initial + spec.max_residual + 1e-6))
            self.assertFalse(report["training_uses_validation_labels"])
            self.assertFalse(report["beta_search"])
            self.assertEqual(
                len(report["feature_names"]), len(spec.columns))
            self.assertEqual(
                set(report["standardized_coefficients"]),
                {FEATURE_NAMES[column] for column in spec.columns},
            )

    def test_beta_permutation_preserves_values_but_breaks_assignment(self):
        beta = np.linspace(.2, .8, 20, dtype=np.float32)
        keys = [f"q{row}" for row in range(len(beta))]
        first = deterministic_beta_permutation(beta, keys, seed=42)
        second = deterministic_beta_permutation(beta, keys, seed=42)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.sort(first), np.sort(beta))
        self.assertFalse(np.array_equal(first, beta))

    def test_paired_assignment_summary_uses_primary_minus_permuted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranks.npz"
            np.savez_compressed(
                path,
                test_keys=np.asarray(["a", "b", "c"]),
                test_dynamic_delta_010_rank=np.asarray([1, 2, 0]),
                test_dynamic_beta_permuted_rank=np.asarray([2, 0, 1]),
            )
            summary = paired_assignment_summary(
                [{"seed": 42, "rank_artifact": str(path)}],
                "ndcg@20",
                repetitions=200,
                seed=7,
            )
            expected = np.mean([
                1.0 - 1.0 / np.log2(3.0),
                1.0 / np.log2(3.0),
                -1.0,
            ])
            self.assertAlmostEqual(summary["difference"], expected)
            self.assertEqual(
                summary["challenger"], "dynamic_delta_010")
            self.assertEqual(
                summary["baseline"], "dynamic_beta_permuted")

    def test_test_evaluation_is_guarded_by_preexisting_manifest(self):
        queries = {
            "q1": {"context": [1], "targets": [1]},
            "q2": {"context": [2], "targets": [41]},
        }
        memory, neural, _ = _synthetic_experts(2)
        beta = {"policy": np.asarray([0.4, 0.6], dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            with self.assertRaises(RuntimeError):
                evaluate_after_manifest(
                    manifest, queries, ["q1", "q2"],
                    memory, neural, beta)
            manifest.write_text(json.dumps({
                "frozen_before_test_target_evaluation": True,
                "inputs": {
                    "test_query_fingerprint": (
                        "definitely-the-wrong-fingerprint"
                    ),
                },
            }))
            with self.assertRaisesRegex(ValueError, "identity changed"):
                evaluate_after_manifest(
                    manifest, queries, ["q1", "q2"],
                    memory, neural, beta)

            from run_cearfn_evidence import query_fingerprint
            manifest.write_text(json.dumps({
                "frozen_before_test_target_evaluation": True,
                "inputs": {
                    "test_query_fingerprint": query_fingerprint(queries),
                },
            }))
            metrics, payload = evaluate_after_manifest(
                manifest, queries, ["q1", "q2"],
                memory, neural, beta)
            self.assertIn("policy", metrics)
            self.assertIn("test_policy_rank", payload)
            self.assertEqual(metrics["policy"]["n"], 2)

    def test_summary_contains_all_requested_metrics_and_tex(self):
        metric_block = {
            "recall@6": .1,
            "ndcg@6": .08,
            "recall@10": .12,
            "ndcg@10": .09,
            "recall@20": .15,
            "ndcg@20": .10,
            "utility": .125,
        }
        results = {
            "Video_Games": {
                "runs": [
                    {
                        "seed": 42,
                        "training": {
                            "dynamic_delta_010": {
                                "standardized_coefficients": {
                                    "log_context_length": .1,
                                    "log_last_item_frequency": .2,
                                    "last_item_is_tail": .3,
                                },
                                "bias": .4,
                            },
                        },
                        "test": {"metrics": {
                            name: dict(metric_block)
                            for name in CONTROL_ORDER
                        }},
                    },
                    {
                        "seed": 123,
                        "training": {
                            "dynamic_delta_010": {
                                "standardized_coefficients": {
                                    "log_context_length": .2,
                                    "log_last_item_frequency": .3,
                                    "last_item_is_tail": .4,
                                },
                                "bias": .5,
                            },
                        },
                        "test": {"metrics": {
                            name: dict(metric_block)
                            for name in CONTROL_ORDER
                        }},
                    },
                ],
            },
        }
        summary = summarize_results(results, bootstrap_repetitions=0)
        primary = summary["domains"]["Video_Games"]["methods"][
            "dynamic_delta_010"]
        self.assertEqual(
            set(primary),
            {
                "recall@6", "ndcg@6",
                "recall@10", "ndcg@10",
                "recall@20", "ndcg@20", "utility",
            },
        )
        tex = render_summary_tex(summary)
        self.assertIn(r"Dynamic $\Delta=.10$ (primary)", tex)
        self.assertIn(r"Primary $\beta_q$ reassigned", tex)
        self.assertIn("Primary linear-gate parameters", tex)
        self.assertIn("nDCG@20", tex)
        self.assertIn("no validation or test labels", tex)


if __name__ == "__main__":
    unittest.main()
