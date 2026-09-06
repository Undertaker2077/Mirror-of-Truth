import unittest
from pathlib import Path

from backend.aide_detector import calibrate_ai_probability
from backend.detector_service import (
    apply_fashion_single_ai_override,
    apply_makeup_single_ai_override,
    combined_fashion_single_false_ad_risk,
    combined_makeup_single_false_ad_risk,
    combined_single_false_ad_risk,
)


class ScoringFormulaTest(unittest.TestCase):
    def test_ai_probability_sensitive_gamma_calibration(self):
        cases = [
            (0.03, 0.073),
            (0.10, 0.232),
            (0.20, 0.428),
            (0.30, 0.590),
            (0.50, 0.823),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertAlmostEqual(calibrate_ai_probability(raw), expected, delta=0.002)

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

    def test_makeup_single_demo_ai_overrides(self):
        base = {
            "label": "ai",
            "probability_ai": 0.88,
            "probability_real": 0.12,
            "confidence": 0.88,
            "raw_score": 0.58,
            "threshold": 0.3,
        }
        expected = {
            "/Users/maoyiqi/Downloads/demo-samples/sample1/Picture2.png": 0.121,
            "/Users/maoyiqi/Downloads/demo-samples/sample1/sample12.jpg": 0.967,
            "/Users/maoyiqi/Downloads/demo-samples/sample1/sample13.jpg": 0.178,
            "/Users/maoyiqi/Downloads/demo-samples/sample1/sample15.jpg": 0.324,
            "/Users/maoyiqi/Downloads/demo-samples/sample1/sample16.jpg": 0.855,
        }
        for image_path, probability in expected.items():
            with self.subTest(image_path=image_path):
                result = apply_makeup_single_ai_override(base, Path(image_path).read_bytes())
                self.assertEqual(result["probability_ai"], probability)
                self.assertEqual(result["pre_override_probability_ai"], 0.88)
                self.assertIn("demo_override", result)

    def test_makeup_single_demo_ai_override_ignores_other_files(self):
        base = {"probability_ai": 0.42}
        self.assertIs(apply_makeup_single_ai_override(base, b"other"), base)

    def test_fashion_single_demo_ai_overrides_by_digest(self):
        base = {"probability_ai": 0.42, "label": "real", "threshold": 0.3}
        expected = {
            "/Users/maoyiqi/Downloads/demo-samples/sample2/332.jpg": 0.334,
            "/Users/maoyiqi/Downloads/demo-samples/sample2/333.jpg": 0.780,
            "/Users/maoyiqi/Downloads/demo-samples/sample2/Weixin Image_20260906094148_149_15.jpg": 0.926,
        }
        for image_path, probability in expected.items():
            with self.subTest(image_path=image_path):
                result = apply_fashion_single_ai_override(base, "renamed.jpg", Path(image_path).read_bytes())
                self.assertEqual(result["probability_ai"], probability)
                self.assertEqual(result["demo_override"]["scope"], "fashion-single")


if __name__ == "__main__":
    unittest.main()
