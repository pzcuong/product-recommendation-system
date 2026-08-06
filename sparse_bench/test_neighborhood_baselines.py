import unittest

from neighborhood_baselines import NeighborhoodConfig, NeighborhoodIndex


class NeighborhoodBaselineTests(unittest.TestCase):
    def setUp(self):
        self.index = NeighborhoodIndex({
            "a": [1, 2, 3], "b": [1, 2, 4], "c": [5, 2, 6]}, 8)

    def test_vsknn_prefers_items_from_matching_recent_context(self):
        rank = self.index.predict_one(
            [1, 2], NeighborhoodConfig(method="vsknn", exclude_seen=True), 3)
        self.assertNotIn(1, rank)
        self.assertNotIn(2, rank)
        self.assertIn(rank[0], (3, 4))

    def test_stan_is_deterministic_and_unique(self):
        cfg = NeighborhoodConfig(method="stan", lambda_snh=None)
        one = self.index.predict_one([1, 2], cfg, 5)
        two = self.index.predict_one([1, 2], cfg, 5)
        self.assertEqual(one, two)
        self.assertEqual(len(one), len(set(one)))


if __name__ == "__main__":
    unittest.main()
