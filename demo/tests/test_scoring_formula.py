import unittest

from backend.detector_service import (
    combined_fashion_single_false_ad_risk,
    combined_makeup_single_false_ad_risk,
    combined_single_false_ad_risk,
    single_verdict,
)


class ScoringFormulaTest(unittest.TestCase):
    def test_legacy_single_formula_aliases_fashion_formula(self):
        cases = [
            (0.90, 0.10, 0.10),
            (0.10, 0.85, 0.20),
            (0.30, 0.30, 0.30),
            (0.05, 0.10, 0.10),
        ]
        for inputs in cases:
            with self.subTest(inputs=inputs):
                self.assertAlmostEqual(
                    combined_single_false_ad_risk(*inputs),
                    combined_fashion_single_false_ad_risk(*inputs),
                )

    def test_makeup_single_formula_raises_ai_or_retouch_risk(self):
        ai_risk = combined_makeup_single_false_ad_risk(0.90, 0.10, 0.10)
        retouch_risk = combined_makeup_single_false_ad_risk(0.10, 0.90, 0.20)
        combined_risk = combined_makeup_single_false_ad_risk(0.792, 0.308, 0.20)
        borderline_ai_low_retouch = combined_makeup_single_false_ad_risk(0.488, 0.022, 0.445)
        low_risk = combined_makeup_single_false_ad_risk(0.05, 0.10, 0.10)
        self.assertGreater(ai_risk, 0.65)
        self.assertGreater(retouch_risk, 0.65)
        self.assertGreater(combined_risk, 0.65)
        self.assertGreater(borderline_ai_low_retouch, 0.30)
        self.assertLess(borderline_ai_low_retouch, 0.40)
        self.assertLess(low_risk, 0.30)

    def test_fashion_single_formula_weights_ai_more_heavily(self):
        high_ai = combined_fashion_single_false_ad_risk(0.90, 0.10, 0.10)
        high_retouch_low_ai = combined_fashion_single_false_ad_risk(0.10, 0.90, 0.20)
        balanced = combined_fashion_single_false_ad_risk(0.50, 0.50, 0.30)
        low = combined_fashion_single_false_ad_risk(0.05, 0.10, 0.10)
        self.assertGreater(high_ai, 0.55)
        self.assertGreater(high_retouch_low_ai, 0.25)
        self.assertLess(high_retouch_low_ai, 0.55)
        self.assertGreater(balanced, 0.25)
        self.assertLess(balanced, 0.55)
        self.assertLess(low, 0.25)

    def test_single_verdict_uses_mode_specific_thresholds(self):
        makeup_risk = combined_makeup_single_false_ad_risk(0.90, 0.10, 0.10)
        fashion_risk = combined_fashion_single_false_ad_risk(0.90, 0.10, 0.10)
        self.assertEqual(single_verdict("makeup", makeup_risk), "High risk")
        self.assertEqual(single_verdict("fashion", fashion_risk), "High risk")
        self.assertEqual(single_verdict("fashion", 0.30), "Medium risk")
        self.assertEqual(single_verdict("makeup", 0.30), "Medium risk")


if __name__ == "__main__":
    unittest.main()
