import unittest

import numpy as np

from run_cearfn_evidence import (
    metrics_from_ranks, paired_recall_test, popularity_partition, ranks_at_20)


class CEARFNEvidenceTests(unittest.TestCase):
    def test_rank_extraction_and_metrics(self):
        rankings = np.asarray([[4, 2, 1], [3, 4, 5], [7, 8, 9]], dtype=np.int32)
        targets = np.asarray([2, 3, 1], dtype=np.int32)
        ranks = ranks_at_20(rankings, targets)
        np.testing.assert_array_equal(ranks, [2, 1, 0])
        self.assertAlmostEqual(metrics_from_ranks(ranks)["recall@20"], 2 / 3)

    def test_paired_test_keeps_query_pairing(self):
        challenger = np.asarray([1] * 80 + [0] * 20, dtype=np.uint8)
        baseline = np.asarray([1] * 50 + [0] * 50, dtype=np.uint8)
        report = paired_recall_test(challenger, baseline, reps=2000, seed=1)
        self.assertAlmostEqual(report["difference"], .30)
        self.assertGreater(report["paired_bootstrap_ci95"][0], 0)

    def test_popularity_partition_covers_eighty_percent(self):
        head, tail, split = popularity_partition({1: 8, 2: 1, 3: 1}, 4)
        np.testing.assert_array_equal(head, [1])
        np.testing.assert_array_equal(tail, [2, 3])
        self.assertEqual(split, 1)


if __name__ == "__main__":
    unittest.main()
