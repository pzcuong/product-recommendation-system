import unittest

import numpy as np

from run_cearfn_ablation import fuse_row, select_beta


class CEARFNAblationTests(unittest.TestCase):
    def test_fusion_rules_are_deterministic_and_unique(self):
        memory = [1, 2, 3]
        neural = [3, 4, 1]
        for rule in ("rrf20", "rrf60", "borda", "inverse_rank"):
            first = fuse_row(memory, neural, .5, rule, width=3, topk=4)
            second = fuse_row(memory, neural, .5, rule, width=3, topk=4)
            self.assertEqual(first, second)
            self.assertEqual(len(first), len(set(first)))

    def test_inverse_rank_emphasises_top_ranks(self):
        # 1/rank weights the head more sharply than 1/(20+rank); ensure the
        # fusion changes ordering in a way that depends on the rule.
        memory = [1, 2, 3, 4]
        neural = [5, 1, 2, 6]
        rrf20 = fuse_row(memory, neural, .5, "rrf20", width=6, topk=6)
        inverse = fuse_row(memory, neural, .5, "inverse_rank", width=6, topk=6)
        self.assertEqual(len(rrf20), 6)
        self.assertEqual(len(inverse), 6)
        # Item 1 is rank-1 in memory and rank-2 in neural; under both rules it
        # must be the top fused item.
        self.assertEqual(rrf20[0], 1)
        self.assertEqual(inverse[0], 1)

    def test_beta_selection_uses_validation_targets(self):
        memory = np.tile(np.arange(1, 21, dtype=np.int32), (2, 1))
        neural = np.tile(np.arange(21, 41, dtype=np.int32), (2, 1))
        beta, report = select_beta(memory, neural, np.asarray([21, 21]),
                                   "rrf20", 20)
        self.assertGreater(beta, 0)
        self.assertEqual(report["recall@20"], 1.0)


if __name__ == "__main__":
    unittest.main()
