import unittest

import hid_protocol
from run_cearfn_hid_protocol import evidence_regime, recurrence_rerank


class HIDProtocolTests(unittest.TestCase):
    def test_recurrence_rerank_can_restore_repeat_intent(self):
        ranking = [4, 5, 6, 7]
        reranked = recurrence_rerank(ranking, [1, 2, 1], gamma=2.0, topk=4)
        self.assertEqual(reranked[0], 1)
        self.assertEqual(len(reranked), len(set(reranked)))

    def test_evidence_regime_uses_only_observed_context(self):
        tail = {4}
        self.assertEqual(evidence_regime({"context": [5]}, tail), "short_tail")
        self.assertEqual(evidence_regime({"context": [2, 3, 4]}, tail), "long_head")

    def test_reconstruct_prefix_blocks(self):
        contexts = [[1, 2], [1], [4], [5, 6, 7], [5, 6], [5]]
        targets = [3, 2, 8, 9, 7, 6]
        sessions = hid_protocol._reconstruct_sessions(contexts, targets)
        self.assertEqual(list(sessions.values()), [[1, 2, 3], [4, 8], [5, 6, 7, 9]])

    def test_official_metric_index_shift(self):
        data = {
            "tail_score_indices": {2},
            "test_queries": {"q": {"context": [1], "targets": [3]}},
        }
        result = hid_protocol.official_metrics({"q": [3, 4]}, data, 2)
        self.assertEqual(result["HR@20"], 1.0)
        self.assertEqual(result["tHR@20"], 1.0)
        self.assertEqual(result["Tail@20"], .5)


if __name__ == "__main__":
    unittest.main()
