import unittest

from backend.aide_detector import calibrate_ai_probability
from backend.detector_service import combined_single_false_ad_risk


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


if __name__ == "__main__":
    unittest.main()
