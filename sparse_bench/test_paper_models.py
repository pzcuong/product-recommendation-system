import unittest

import torch

from paper_models import build_model, model_logits


class PaperModelTests(unittest.TestCase):
    def test_all_models_return_full_catalog_logits(self):
        contexts = torch.tensor([[1, 2, 3], [4, 5, 0]])
        lengths = torch.tensor([3, 2])
        for name in ("GRU4Rec", "SASRec", "NARM", "SR-GNN", "SIGMA-compatible"):
            model = build_model(name, 12, 8)
            logits = model_logits(model, contexts, lengths)
            self.assertEqual(tuple(logits.shape), (2, 12), name)
            self.assertTrue(torch.isfinite(logits).all(), name)

    def test_sasrec_is_causal_for_last_state(self):
        model = build_model("SASRec", 12, 8).eval()
        short = torch.tensor([[1, 2, 0]])
        longer = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            first = model_logits(model, short, torch.tensor([2]))
            second = model_logits(model, longer, torch.tensor([2]))
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
