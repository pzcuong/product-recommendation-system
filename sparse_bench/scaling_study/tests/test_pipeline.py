import tempfile
import unittest
import json
from pathlib import Path

import torch

from sparse_bench.scaling_study.audit import audit
from sparse_bench.scaling_study.analysis import crossover, paired_inference, variance_decomposition
from sparse_bench.scaling_study.data import coverage_stats, create_manifest, materialize
from sparse_bench.scaling_study.metrics import aggregate, per_query
from sparse_bench.scaling_study.models import SessionModel
from sparse_bench.mamba4rec_reference import backend_status


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = {"train_sessions": {str(i): [1 + i % 4, 2 + i % 4, 6] for i in range(10)},
                     "test_queries": {"q1": {"context": [1], "targets": [2]},
                                      "q2": {"context": [2], "targets": [9]}}}

    def test_manifest_is_reproducible_and_disjoint(self):
        a = create_manifest(self.data, 8, 42, None, .25)
        b = create_manifest(self.data, 8, 42, None, .25)
        self.assertEqual(a, b)
        self.assertFalse(set(a["fit_session_ids"]) & set(a["validation_session_ids"]))
        fit, val, test = materialize(self.data, a)
        self.assertEqual(len(fit) + len(val), 8); self.assertEqual(len(test), 2)

    def test_variants_and_causal_sasrec_forward(self):
        seq = torch.tensor([[1, 2, 0], [2, 3, 4]]); lengths = torch.tensor([2, 3])
        for variant in SessionModel.VALID:
            model = SessionModel(variant, 10, dim=8, state=4, heads=2)
            self.assertEqual(tuple(model(seq, lengths).shape), (2, 10))
        self.assertFalse(any(isinstance(x, torch.nn.GRU) for x in SessionModel("pure_ssm", 10, dim=8).modules()))
        self.assertFalse(any(isinstance(x, torch.nn.GRU) for x in SessionModel("contractive_ssm", 10, dim=8).modules()))
        self.assertTrue(SessionModel("contractive_ssm", 10, dim=8).ssm[0].contractive)
        self.assertFalse(any(x.__class__.__name__ == "SelectiveSSMBlock" for x in SessionModel("fe_gru", 10, dim=8).modules()))

    def test_coverage_and_metrics(self):
        stats = coverage_stats(self.data["train_sessions"], self.data["test_queries"])
        self.assertEqual(stats["test_targets_seen"], 1)
        rows = per_query({"q1": [2, 3], "q2": [3, 4]}, self.data["test_queries"], {1, 2, 3})
        metrics = aggregate(rows)
        self.assertEqual(metrics["recall@20"], .5)
        self.assertEqual(metrics["seen_target_recall@20"], 1.)

    def test_empty_audit_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(audit(Path(directory))["ok"])

    def test_draw_aware_inference_and_variance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for draw, base, challenger in ((1, [0, 0], [1, 0]), (2, [0, 0], [1, 1])):
                for variant, values in (("gru4rec", base), ("pure_ssm", challenger)):
                    artifact = root / f"{draw}_{variant}"
                    artifact.mkdir()
                    rows = [{"query_id": str(i), "recall@20": value}
                            for i, value in enumerate(values)]
                    (artifact / "per_query.json").write_text(json.dumps(rows))
                    runs.append({"status": "complete", "scale": 100,
                                 "draw_seed": draw, "variant": variant,
                                 "seed": 42, "artifact_dir": str(artifact),
                                 "test_metrics": {"recall@20": sum(values) / len(values)}})
            inference = paired_inference(runs, samples=100, seed=1)
            self.assertEqual(inference["scales"][0]["draw_win_rate"], 1.0)
            self.assertTrue(inference["scales"][0]["all_draws_positive"])
            variance = variance_decomposition(runs)
            self.assertEqual(len(variance), 2)
            self.assertTrue(all(row["n_data_draws"] == 2 for row in variance))

    def test_crossover_requires_sustained_advantage(self):
        rows = []
        for scale, baseline, challenger in ((10, .2, .1), (20, .2, .3), (30, .3, .2)):
            rows.extend(({"scale": scale, "draw_seed": 1, "variant": "gru4rec", "mean": baseline},
                         {"scale": scale, "draw_seed": 1, "variant": "pure_ssm", "mean": challenger}))
        result = crossover(rows)
        self.assertEqual(len(result["draws"][0]["crossings"]), 2)
        self.assertIsNone(result["draws"][0]["sustained_crossover"])
        self.assertEqual(result["n_draws_with_sustained_crossover"], 0)

    def test_external_backend_status_is_explicit(self):
        status = backend_status()
        self.assertIsInstance(status.official_mamba4rec_ready, bool)


if __name__ == "__main__": unittest.main()
