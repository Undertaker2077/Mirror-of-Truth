import unittest

from backend.face_alignment_mediapipe import FaceFeatures, compare_face_features


class FaceAlignmentGeometryTest(unittest.TestCase):
    def test_similar_face_features_are_aligned(self):
        before = FaceFeatures(
            bbox=(0.30, 0.20, 0.70, 0.78),
            center=(0.50, 0.49),
            angle_degrees=1.0,
            eye_distance=0.22,
            landmarks_count=478,
        )
        after = FaceFeatures(
            bbox=(0.31, 0.21, 0.71, 0.79),
            center=(0.51, 0.50),
            angle_degrees=3.0,
            eye_distance=0.22,
            landmarks_count=478,
        )
        result = compare_face_features(before, after)
        self.assertTrue(result["alignment_success"])
        self.assertEqual(result["face_angle_similarity"], "Similar")
        self.assertEqual(result["crop_similarity"], "Similar")

    def test_shifted_face_features_are_significant(self):
        before = FaceFeatures(
            bbox=(0.30, 0.20, 0.70, 0.78),
            center=(0.50, 0.49),
            angle_degrees=0.0,
            eye_distance=0.22,
            landmarks_count=478,
        )
        after = FaceFeatures(
            bbox=(0.42, 0.28, 0.90, 0.98),
            center=(0.66, 0.63),
            angle_degrees=15.0,
            eye_distance=0.28,
            landmarks_count=478,
        )
        result = compare_face_features(before, after)
        self.assertFalse(result["alignment_success"])
        self.assertEqual(result["face_angle_similarity"], "Significant")
        self.assertEqual(result["crop_similarity"], "Significant")


if __name__ == "__main__":
    unittest.main()
