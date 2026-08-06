import unittest

from benchmark_dynamic_beta_inference import _tex_decimal


class DynamicBetaInferenceFormattingTest(unittest.TestCase):
    def test_tex_decimal_only_compacts_a_leading_zero(self):
        self.assertEqual(_tex_decimal(0.988, 3), ".988")
        self.assertEqual(_tex_decimal(-0.125, 3), "-.125")
        self.assertEqual(_tex_decimal(70.134, 2), "70.13")
        self.assertEqual(_tex_decimal(100.01, 2), "100.01")


if __name__ == "__main__":
    unittest.main()
