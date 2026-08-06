import unittest
from collections import Counter

import ltr_fusion


class LTRFusionTests(unittest.TestCase):
    def test_matrix_never_injects_target(self):
        queries = {"q": {"context": [1, 2], "targets": [9]}}
        matrix = ltr_fusion.build_candidate_matrix(
            {"q": [3, 4]}, {"q": [4, 5]}, queries, Counter({3: 2}), set())
        self.assertEqual(matrix.candidates[0], [3, 4, 5])
        self.assertEqual(int(matrix.y.sum()), 0)

    def test_positive_is_labeled_only_when_retrieved(self):
        queries = {"q": {"context": [1, 2], "targets": [4]}}
        matrix = ltr_fusion.build_candidate_matrix(
            {"q": [3, 4]}, {"q": [4, 5]}, queries, Counter({4: 2}), {3})
        self.assertEqual(int(matrix.y.sum()), 1)
        self.assertEqual(matrix.x.shape[1], len(ltr_fusion.FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
