import unittest

from backend.detector_service import (
    combined_fashion_single_false_ad_risk,
    combined_makeup_single_false_ad_risk,
    combined_single_false_ad_risk,
)


class ScoringFormulaTest(unittest.TestCase):
    def test_single_false_ad_risk_keeps_low_and_mid_cases_reasonable(self):
        cases = [
            ((0.90, 0.10, 0.10), 0.91),
            ((0.10, 0.85, 0.20), 0.91),
            ((0.30, 0.30, 0.30), 0.41),
            ((0.05, 0.10, 0.10), 0.06),
        ]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                self.assertAlmostEqual(combined_single_false_ad_risk(*inputs), expected, delta=0.025)
                self.assertAlmostEqual(combined_fashion_single_false_ad_risk(*inputs), expected, delta=0.025)

    def test_makeup_single_formula_reduces_ai_only_risk(self):
        fashion_risk = combined_fashion_single_false_ad_risk(0.90, 0.10, 0.10)
        makeup_risk = combined_makeup_single_false_ad_risk(0.90, 0.10, 0.10)
        self.assertGreater(fashion_risk, 0.85)
        self.assertLess(makeup_risk, 0.08)

    def test_makeup_single_formula_discounts_high_ai_probability(self):
        low_ai = combined_makeup_single_false_ad_risk(0.10, 0.70, 0.50)
        high_ai = combined_makeup_single_false_ad_risk(0.90, 0.70, 0.50)
        self.assertLess(high_ai, low_ai)

    def test_fashion_single_formula_raises_high_ai_probability(self):
        low_ai = combined_fashion_single_false_ad_risk(0.10, 0.20, 0.20)
        high_ai = combined_fashion_single_false_ad_risk(0.90, 0.20, 0.20)
        self.assertGreater(high_ai, low_ai)

    def test_makeup_single_formula_still_responds_to_retouch(self):
        low_retouch = combined_makeup_single_false_ad_risk(0.20, 0.10, 0.10)
        high_retouch = combined_makeup_single_false_ad_risk(0.20, 0.85, 0.60)
        self.assertLess(low_retouch, 0.12)
        self.assertGreater(high_retouch, 0.45)


if __name__ == "__main__":
    unittest.main()
