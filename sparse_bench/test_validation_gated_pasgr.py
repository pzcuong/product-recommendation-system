import unittest
from run_validation_gated_pasgr import validation_score


class ValidationGateTests(unittest.TestCase):
    def test_exact_validation_tie_prefers_simpler_cell(self):
        metrics = {"utility": .5, "recall@20": .6, "recall@6": .4}
        simple = (0.0, False, 0.0, 0.0)
        complex_ = (.35, True, .15, .10)
        self.assertGreater(validation_score(simple, metrics),
                           validation_score(complex_, metrics))


if __name__ == "__main__":
    unittest.main()
