import unittest
from collections import Counter

from sparse_bench.vptr import fit_router, route, select_factual


class VPTRTest(unittest.TestCase):
    def test_validation_selects_expert_without_test_targets(self):
        valid = {str(i): {"context": [1, 2, 3], "targets": [9]} for i in range(50)}
        preds = {"SKNN": {u: [8, 7] for u in valid},
                 "Twin": {u: [9, 7] for u in valid}}
        router = fit_router(preds, valid, Counter({1: 9, 2: 8, 3: 7}),
                            min_samples=10, shrinkage=0)
        test = {"x": {"context": [1, 2, 3], "targets": [12345]}}
        out = route(router, {"SKNN": {"x": [8]}, "Twin": {"x": [9]}}, test)
        self.assertEqual(out["x"][0], 9)

    def test_small_stratum_abstains(self):
        valid = {"a": {"context": [1], "targets": [9]}}
        preds = {"SKNN": {"a": [8]}, "Twin": {"a": [9]}}
        router = fit_router(preds, valid, Counter({1: 2}), min_samples=40)
        self.assertEqual(router["policy"]["short|head"], "SKNN")

    def test_factual_anchor_is_validation_selected(self):
        valid = {str(i): {"context": [1], "targets": [9]} for i in range(10)}
        preds = {"SKNN": {u: [8] for u in valid},
                 "MostPop": {u: [9] for u in valid}}
        self.assertEqual(select_factual(preds, valid), "MostPop")


if __name__ == "__main__":
    unittest.main()
