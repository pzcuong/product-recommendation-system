import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from slide_graphs import generate_dynamic_beta_charts as charts  # noqa: E402


def _metric(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "values": [float(value) for value in array],
    }


def _complete_summary():
    offsets = {
        "oof_global": 0.0,
        "bucket_head_tail": 0.0004,
        "bucket_short_long_head_tail": -0.0002,
        "dynamic_delta_005": 0.0001,
        "dynamic_delta_010": 0.0007,
        "dynamic_beta_permuted": 0.00005,
        "dynamic_delta_020": -0.0005,
    }
    domains = {}
    seed_noise = np.asarray([-0.0002, 0.0, 0.0002])
    for domain_index, domain in enumerate(charts.DOMAIN_LABELS):
        methods = {}
        for method_index, (method, offset) in enumerate(offsets.items()):
            scale = 1.0 + 0.05 * method_index
            utility = (
                0.10
                + 0.02 * domain_index
                + offset
                + scale * seed_noise
            )
            ndcg = (
                0.08
                + 0.015 * domain_index
                + 0.8 * offset
                + scale * seed_noise
            )
            methods[method] = {
                "utility": _metric(utility),
                "ndcg@20": _metric(ndcg),
            }
        domains[domain] = {
            "seeds": [42, 123, 456],
            "methods": methods,
            "assignment_paired": {
                metric: {
                    "difference": 0.0001 * (
                        metric_index + 1 + domain_index
                    ),
                    "cluster_bootstrap_ci95": [
                        -0.00002 + 0.0001 * (
                            metric_index + domain_index
                        ),
                        0.00022 + 0.0001 * (
                            metric_index + domain_index
                        ),
                    ],
                    "clusters": 100,
                    "repetitions": 2_000,
                    "challenger": "dynamic_delta_010",
                    "baseline": "dynamic_beta_permuted",
                    "seeds": [42, 123, 456],
                }
                for metric_index, metric in enumerate(
                    charts.ASSIGNMENT_EFFECT_METRICS
                )
            },
            "primary_gate_parameters": {
                "standardized_coefficients": {
                    feature: _metric(
                        np.asarray([-0.1, -0.08, -0.06])
                        + 0.01 * feature_index
                        + 0.005 * domain_index
                    )
                    for feature_index, feature in enumerate(
                        charts.PRIMARY_GATE_FEATURES
                    )
                },
                "bias": _metric([0.0, 0.01, -0.01]),
            },
        }
    return {
        "protocol": charts.ALLOCATION_CONTROL_PROTOCOL,
        "domains": domains,
    }


def _complete_paired_summary():
    metrics = list(charts.FULL_METRIC_PAIRED_METRICS)
    domains = {}
    for domain_index, domain in enumerate(charts.DOMAIN_LABELS):
        comparison = {}
        for metric_index, metric in enumerate(metrics):
            magnitude = (
                0.0001
                + 0.00002 * domain_index
                + 0.00001 * metric_index
            )
            difference = (
                -magnitude
                if (domain_index + metric_index) % 2
                else magnitude
            )
            comparison[metric] = {
                "difference": difference,
                "cluster_bootstrap_ci95": [
                    difference - 0.00005,
                    difference + 0.00006,
                ],
                "clusters": 1000 + domain_index,
                "repetitions": 20000,
                "metric": metric,
                "challenger": "dynamic",
                "baseline": "oof_global",
                "seeds": [42, 123, 456],
            }
        domains[domain] = {
            "seeds": [42, 123, 456],
            "n_seeds": 3,
            "paired": {"oof_global": comparison},
        }
    return {
        "source": "synthetic-primary-results.json",
        "bootstrap_unit": "query cluster",
        "seed_aggregation": "paired query mean across seeds",
        "domains": domains,
    }


def _complete_primary_results():
    domains = {}
    for domain_index, domain in enumerate(charts.DOMAIN_LABELS):
        runs = []
        for seed_index, seed in enumerate(
            charts.PRIMARY_ALLOCATION_SEEDS
        ):
            metrics = {}
            for mode_index, mode in enumerate(charts.MODE_LABELS):
                metrics[mode] = {
                    "utility": (
                        0.04
                        + 0.02 * domain_index
                        + 0.001 * mode_index
                        + 0.0001 * seed_index
                    ),
                    "recall@20": (
                        0.10
                        + 0.05 * domain_index
                        + 0.01 * mode_index
                        + 0.0002 * seed_index
                    ),
                    "n": 1000 + 100 * domain_index,
                }
            runs.append(
                {
                    "seed": seed,
                    "training": {
                        "global": {
                            "training_uses_validation_labels": False
                        },
                        "dynamic": {
                            "training_uses_validation_labels": False
                        },
                        "short_long": {
                            "short": {
                                "training_uses_validation_labels": False
                            },
                            "long": {
                                "training_uses_validation_labels": False
                            },
                        },
                    },
                    "test": {"metrics": metrics},
                }
            )
        domains[domain] = {"runs": runs}
    return domains


class PrimaryAllocationChartTests(unittest.TestCase):
    def _write_results(self, directory, payload):
        path = (
            Path(directory)
            / "dynamic_beta_trainonly_v2_results.json"
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_six_methods_validate_in_declared_order_and_render(self):
        payload = _complete_primary_results()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_results(directory, payload)
            runs = charts._runs_by_domain(payload)
            original_out = charts.OUT
            charts.OUT = Path(directory) / "charts"
            try:
                charts._configure_style()
                outputs, provenance = charts.allocation_modes(
                    runs, source_path=path
                )
            finally:
                charts.OUT = original_out

            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(Path(item).exists() for item in outputs))
            self.assertEqual(
                provenance["methods"],
                [
                    "memory_only",
                    "neural_only",
                    "fixed_05",
                    "oof_global",
                    "oof_short_long",
                    "dynamic",
                ],
            )
            self.assertEqual(
                provenance["matched_seeds"]["Video_Games"],
                [42, 123, 456],
            )
            self.assertEqual(
                provenance["source_sha256"],
                charts._sha256_file(path),
            )

    def test_missing_endpoint_is_rejected(self):
        payload = _complete_primary_results()
        del payload["Baby_Products"]["runs"][1]["test"]["metrics"][
            "neural_only"
        ]
        with self.assertRaisesRegex(
            ValueError, "missing allocation modes"
        ):
            charts._validate_primary_allocation_runs(
                charts._runs_by_domain(payload)
            )

    def test_noncanonical_seed_set_is_rejected(self):
        payload = _complete_primary_results()
        payload["Diginetica_HID"]["runs"].pop()
        with self.assertRaisesRegex(ValueError, "expected exactly seeds"):
            charts._validate_primary_allocation_runs(
                charts._runs_by_domain(payload)
            )

    def test_validation_label_use_is_rejected(self):
        payload = _complete_primary_results()
        payload["Video_Games"]["runs"][0]["training"]["dynamic"][
            "training_uses_validation_labels"
        ] = True
        with self.assertRaisesRegex(
            ValueError, "validation-label-free"
        ):
            charts._validate_primary_allocation_runs(
                charts._runs_by_domain(payload)
            )


class MethodOverviewChartTests(unittest.TestCase):
    def test_method_and_protocol_figures_render(self):
        with tempfile.TemporaryDirectory() as directory:
            original_out = charts.OUT
            charts.OUT = Path(directory) / "charts"
            try:
                charts._configure_style()
                outputs = charts.method_and_protocol_overview()
            finally:
                charts.OUT = original_out
            self.assertEqual(len(outputs), 6)
            self.assertTrue(all(Path(item).exists() for item in outputs))
            self.assertEqual(
                {Path(item).stem for item in outputs},
                {
                    "00-method-architecture",
                    "00-protocol-lineage",
                    "17-bounded-pair-certificate",
                },
            )


class AllocationControlChartTests(unittest.TestCase):
    def _write_summary(self, directory, payload):
        path = Path(directory) / "controls.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_complete_summary_validates_and_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_summary(directory, _complete_summary())
            data, provenance = charts._load_allocation_control_summary(
                path
            )
            self.assertEqual(
                provenance["matched_seeds"], [42, 123, 456]
            )
            self.assertEqual(
                set(data), set(charts.DOMAIN_LABELS)
            )

            original_out = charts.OUT
            charts.OUT = Path(directory) / "charts"
            try:
                charts._configure_style()
                outputs, chart_provenance = (
                    charts.allocation_control_comparison(path)
                )
            finally:
                charts.OUT = original_out
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(Path(item).exists() for item in outputs))
            self.assertEqual(
                chart_provenance["protocol"],
                charts.ALLOCATION_CONTROL_PROTOCOL,
            )

    def test_assignment_mechanism_summary_validates_and_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_summary(directory, _complete_summary())
            data, provenance = (
                charts._load_assignment_mechanism_summary(path)
            )
            self.assertEqual(
                data["Video_Games"]["paired"]["ndcg@20"][
                    "clusters"
                ],
                100,
            )
            original_out = charts.OUT
            charts.OUT = Path(directory) / "charts"
            try:
                charts._configure_style()
                outputs, chart_provenance = (
                    charts.assignment_mechanism_charts(path)
                )
            finally:
                charts.OUT = original_out
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(Path(item).exists() for item in outputs))
            self.assertEqual(
                chart_provenance["protocol"],
                charts.ALLOCATION_CONTROL_PROTOCOL,
            )
            self.assertEqual(
                provenance["domains"]["Baby_Products"]["seeds"],
                [42, 123, 456],
            )

    def test_incomplete_seed_set_is_rejected(self):
        payload = _complete_summary()
        payload["domains"]["Baby_Products"]["seeds"] = [42, 123]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_summary(directory, payload)
            with self.assertRaisesRegex(ValueError, "expected exactly"):
                charts._load_allocation_control_summary(path)

    def test_inconsistent_stored_statistic_is_rejected(self):
        payload = _complete_summary()
        payload["domains"]["Video_Games"]["methods"][
            "dynamic_delta_010"
        ]["utility"]["mean"] += 0.01
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_summary(directory, payload)
            with self.assertRaisesRegex(ValueError, "stored mean"):
                charts._load_allocation_control_summary(path)


class FusionAblationChartTests(unittest.TestCase):
    def _write_results(self, directory, payload):
        path = (
            Path(directory)
            / "dynamic_beta_trainonly_v2_results.json"
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_complete_results_preserve_values_and_render(self):
        payload = _complete_primary_results()
        expected_values = [
            run["test"]["metrics"]["dynamic"]["recall@20"]
            for run in payload["Video_Games"]["runs"]
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_results(directory, payload)
            runs = charts._runs_by_domain(payload)
            data, provenance = charts._load_fusion_ablation_recall20(
                runs,
                source_path=path,
            )
            self.assertEqual(
                data["Video_Games"]["dynamic"]["values"],
                expected_values,
            )
            self.assertEqual(
                provenance["matched_seeds"], [42, 123, 456]
            )
            self.assertTrue(
                provenance["domains"]["Video_Games"][
                    "endpoint_gain_is_positive"
                ]
            )
            self.assertEqual(
                provenance["source_sha256"],
                charts._sha256_file(path),
            )

            original_out = charts.OUT
            charts.OUT = Path(directory) / "charts"
            try:
                charts._configure_style()
                outputs, chart_provenance = (
                    charts.fusion_ablation_recall20(
                        runs,
                        source_path=path,
                    )
                )
            finally:
                charts.OUT = original_out
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(Path(item).exists() for item in outputs))
            self.assertEqual(
                chart_provenance["metric"], "recall@20"
            )

    def test_missing_recall20_is_rejected(self):
        payload = _complete_primary_results()
        del payload["Baby_Products"]["runs"][1]["test"]["metrics"][
            "neural_only"
        ]["recall@20"]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_results(directory, payload)
            with self.assertRaisesRegex(
                ValueError, "missing or invalid recall@20"
            ):
                charts._load_fusion_ablation_recall20(
                    charts._runs_by_domain(payload),
                    source_path=path,
                )

    def test_mismatched_query_count_is_rejected(self):
        payload = _complete_primary_results()
        payload["Diginetica_HID"]["runs"][0]["test"]["metrics"][
            "dynamic"
        ]["n"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_results(directory, payload)
            with self.assertRaisesRegex(
                ValueError, "query counts differ"
            ):
                charts._load_fusion_ablation_recall20(
                    charts._runs_by_domain(payload),
                    source_path=path,
                )


class FullMetricPairedDashboardTests(unittest.TestCase):
    def _write_summary(self, directory, payload):
        path = Path(directory) / "dynamic_beta_summary.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_complete_summary_validates_preserves_signs_and_renders(self):
        payload = _complete_paired_summary()
        expected = payload["domains"]["Baby_Products"]["paired"][
            "oof_global"
        ]["recall@6"]
        self.assertLess(expected["difference"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_summary(directory, payload)
            data, provenance = (
                charts._load_full_metric_paired_dashboard(
                    payload, path=path
                )
            )
            observed = data["Baby_Products"]["recall@6"]
            self.assertEqual(
                observed["difference"], expected["difference"]
            )
            self.assertEqual(
                observed["cluster_bootstrap_ci95"],
                expected["cluster_bootstrap_ci95"],
            )
            self.assertEqual(
                provenance["matched_seeds"], [42, 123, 456]
            )

            original_out = charts.OUT
            charts.OUT = Path(directory) / "charts"
            try:
                charts._configure_style()
                outputs, chart_provenance = (
                    charts.full_metric_paired_dashboard(
                        payload, path=path
                    )
                )
            finally:
                charts.OUT = original_out
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(Path(item).exists() for item in outputs))
            self.assertEqual(
                chart_provenance["metrics"],
                list(charts.FULL_METRIC_PAIRED_METRICS),
            )

    def test_noncanonical_seed_set_is_rejected(self):
        payload = _complete_paired_summary()
        payload["domains"]["Diginetica_HID"]["seeds"] = [42, 123]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_summary(directory, payload)
            with self.assertRaisesRegex(ValueError, "expected exactly"):
                charts._load_full_metric_paired_dashboard(
                    payload, path=path
                )

    def test_missing_ci_is_rejected(self):
        payload = _complete_paired_summary()
        del payload["domains"]["Video_Games"]["paired"][
            "oof_global"
        ]["ndcg@20"]["cluster_bootstrap_ci95"]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_summary(directory, payload)
            with self.assertRaisesRegex(
                ValueError, "cluster_bootstrap_ci95"
            ):
                charts._load_full_metric_paired_dashboard(
                    payload, path=path
                )


class ExpertSwapChartTests(unittest.TestCase):
    def test_in_progress_domain_checkpoint_is_skipped(self):
        payload = {
            "Video_Games": {
                "protocol": (
                    "dynamic-beta-narm-expert-swap-v2-full-refit"
                ),
                "runs": [],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert-swap.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(charts.expert_swap_comparison(path))

    def test_present_domain_with_wrong_protocol_is_rejected(self):
        payload = {
            "Video_Games": {
                "protocol": "stale-development-protocol",
                "runs": [],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert-swap.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "unexpected expert-swap protocol"
            ):
                charts.expert_swap_comparison(path)


if __name__ == "__main__":
    unittest.main()
