import unittest
import numpy as np
from paired_statistics import cluster_paired_recall


class PairedStatisticsTests(unittest.TestCase):
    def test_clusters_repeated_queries_across_seeds(self):
        challenger = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]])
        baseline = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        out = cluster_paired_recall(challenger, baseline, reps=1000, seed=1)
        self.assertEqual(out["n_queries"], 4)
        self.assertEqual(out["n_seeds"], 3)
        self.assertAlmostEqual(out["difference"], .5)


if __name__ == "__main__":
    unittest.main()
