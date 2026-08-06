import unittest

import torch

from narm_mps import NARM, NARMConfig, collate_prefixes, expand_sessions


class NARMMPSTests(unittest.TestCase):
    def test_dense_logits_and_padding_mask(self):
        config = NARMConfig(n_items=12, dim=8, batch_size=2, epochs=1)
        model = NARM(config)
        contexts, lengths, targets = collate_prefixes([
            ([1, 2, 3], 4), ([2], 3)])
        logits = model.logits(contexts, lengths)
        self.assertEqual(tuple(logits.shape), (2, 12))
        self.assertTrue(torch.all(logits[:, 0] < -1e8))
        self.assertEqual(targets.tolist(), [4, 3])

    def test_expand_sessions_matches_prefix_protocol(self):
        contexts, targets = expand_sessions({"s": [1, 2, 3]})
        self.assertEqual(contexts, [[1], [1, 2]])
        self.assertEqual(targets, [2, 3])


if __name__ == "__main__":
    unittest.main()
