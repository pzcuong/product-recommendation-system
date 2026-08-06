import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import cearf
from run_dynamic_beta_fusion_control import (
    load_score_cache,
    minmax_normalize_rows,
    normalized_combsum,
    reconstruct_cearf_final_scores,
    recover_pasgr_topk_scores,
    save_score_cache,
)
from summarize_dynamic_beta_fusion_control import paired_query_summary


class _ToyPASGR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.item = torch.nn.Embedding.from_pretrained(
            weights, freeze=False, padding_idx=0)
        self.config = SimpleNamespace(max_seq=3)

    def encode(self, contexts, lengths):
        rows = torch.arange(contexts.size(0), device=contexts.device)
        state = self.item(contexts[rows, lengths - 1])
        return F.normalize(state, dim=-1)


class DynamicBetaFusionControlTests(unittest.TestCase):
    def test_reconstructs_cearf_final_scores_and_zero_fallback(self):
        arrays = {
            "transition": np.asarray([[8, 9, 0, 0]], dtype=np.int32),
            "session": np.asarray([[3, 4, 5, 0]], dtype=np.int32),
            "popularity": np.asarray([[9, 8, 7, 6]], dtype=np.int32),
            "selected": np.asarray([[3, 4, 5, 9]], dtype=np.int32),
        }
        queries = {"q": {"context": [1]}}
        scores, report = reconstruct_cearf_final_scores(
            arrays,
            queries,
            ["q"],
            {"short": [0.0, 1.0, 0.0], "long": [1.0, 0.0, 0.0]},
            cearf.CEARFConfig(),
        )
        np.testing.assert_allclose(
            scores[0, :3],
            np.asarray([1 / 21, 1 / 22, 1 / 23]),
            rtol=1e-6,
        )
        self.assertEqual(float(scores[0, 3]), 0.0)
        self.assertEqual(
            report["fallback_items_with_native_score_zero"], 1)
        self.assertTrue(report["verified_native_order"])

    def test_reconstruction_rejects_inconsistent_selected_order(self):
        arrays = {
            "transition": np.asarray([[0, 0]], dtype=np.int32),
            "session": np.asarray([[3, 4]], dtype=np.int32),
            "popularity": np.asarray([[8, 9]], dtype=np.int32),
            "selected": np.asarray([[4, 3]], dtype=np.int32),
        }
        with self.assertRaises(AssertionError):
            reconstruct_cearf_final_scores(
                arrays,
                {"q": {"context": [1]}},
                ["q"],
                {
                    "short": [0.0, 1.0, 0.0],
                    "long": [0.0, 1.0, 0.0],
                },
                cearf.CEARFConfig(),
            )

    def test_minmax_is_per_row_and_zero_span_is_zero(self):
        items = np.asarray([[1, 2, 0], [3, 4, 0]], dtype=np.int32)
        scores = np.asarray([[2.0, 1.0, 99.0], [5.0, 5.0, 99.0]])
        normalized = minmax_normalize_rows(items, scores)
        np.testing.assert_allclose(
            normalized,
            np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )

    def test_combsum_uses_absent_zero_and_item_id_tie_break(self):
        ranking = normalized_combsum(
            np.asarray([[1, 2]], dtype=np.int32),
            np.asarray([[2.0, 1.0]], dtype=np.float32),
            np.asarray([[2, 3]], dtype=np.int32),
            np.asarray([[4.0, 2.0]], dtype=np.float32),
            np.asarray([0.5], dtype=np.float32),
            topk=3,
        )
        # item 1: .5, item 2: .5, item 3: 0; smaller ID wins tie.
        np.testing.assert_array_equal(
            ranking, np.asarray([[1, 2, 3]], dtype=np.int32))

    def test_pasgr_scores_reproduce_every_persisted_topk_id(self):
        model = _ToyPASGR()
        queries = {
            "a": {"context": [1]},
            "b": {"context": [3]},
        }
        keys = ["a", "b"]
        # Seen items are excluded. For q=a, 2 then 3; for q=b, 2 then 1.
        expected = np.asarray([[2, 3], [2, 1]], dtype=np.int32)
        scores = recover_pasgr_topk_scores(
            model,
            queries,
            keys,
            expected,
            n_items=5,
            exclude_seen=True,
            batch_size=2,
        )
        self.assertEqual(scores.shape, expected.shape)
        self.assertTrue(np.all(np.isfinite(scores)))
        wrong = expected.copy()
        wrong[0, 0] = 4
        with self.assertRaises(AssertionError):
            recover_pasgr_topk_scores(
                model,
                queries,
                keys,
                wrong,
                n_items=5,
                exclude_seen=True,
                batch_size=2,
            )

    def test_score_cache_requires_exact_protocol_identity(self):
        keys = ["a", "b"]
        items = np.asarray([[1, 2], [3, 4]], dtype=np.int32)
        scores = np.asarray([[0.5, 0.4], [0.9, 0.1]], dtype=np.float32)
        metadata = {
            "protocol": "p",
            "checkpoint_sha256": "abc",
            "candidate_width": 2,
            "exclude_seen": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            save_score_cache(
                path, metadata, keys, items, scores)
            loaded = load_score_cache(
                path, metadata, keys, items)
            np.testing.assert_array_equal(loaded, scores)
            incompatible = {**metadata, "checkpoint_sha256": "different"}
            self.assertIsNone(load_score_cache(
                path, incompatible, keys, items))

    def test_paired_operator_summary_uses_matched_query_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranks.npz"
            np.savez_compressed(
                path,
                test_keys=np.asarray(["a", "b", "c"]),
                test_weighted_rrf_rank=np.asarray([1, 0, 5]),
                test_normalized_combsum_rank=np.asarray([1, 4, 0]),
            )
            summary = paired_query_summary(
                [{"seed": 42, "rank_artifact": str(path)}],
                "recall@20",
                repetitions=200,
                seed=7,
            )
            # One rescue and one damage cancel exactly on matched queries.
            self.assertAlmostEqual(summary["difference"], 0.0)
            self.assertEqual(summary["clusters"], 3)


if __name__ == "__main__":
    unittest.main()
