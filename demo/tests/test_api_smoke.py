import io
import os
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

os.environ.pop("MIRROR_USE_REAL_AIDETECTOR", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.main import app  # noqa: E402


def tiny_png(color=(220, 205, 195)):
    image = Image.new("RGB", (64, 64), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ApiSmokeTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIn("beautyproof_unified", response.json())

    def test_single_image_analysis(self):
        response = self.client.post(
            "/api/analyze/single",
            data={"mode": "makeup", "backend": "ultra"},
            files={"image": ("ai-generated.png", tiny_png(), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("false_advertising_confidence", payload)
        self.assertIn("beautyproof_unified", payload)
        self.assertIn("visual_evidence", payload)
        self.assertGreaterEqual(payload["visual_evidence"]["integrity_score"], 0)
        self.assertLessEqual(payload["visual_evidence"]["integrity_score"], 1)
        self.assertTrue(payload["visual_evidence"]["manipulation_map"].endswith(".png"))
        self.assertEqual(payload["model_output"]["label"], "ai")
        self.assertNotIn("before_model_output", payload)
        self.assertNotIn("after_model_output", payload)
        self.assertNotIn("before_after_evidence", payload)

    def test_unified_route_requires_real_model_env(self):
        response = self.client.post(
            "/api/beautyproof/unified/analyze",
            files={"image": ("beauty_retouched.png", tiny_png(), "image/png")},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("BEAUTYPROOF_USE_UNIFIED=1", response.json()["detail"])

    def test_beautyproof_v2_contract(self):
        response = self.client.post(
            "/api/beautyproof/v2/detect",
            files={"image": ("beauty_retouched.png", tiny_png(), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        evidence = payload["visual_evidence"]
        self.assertEqual(evidence["model_version"], "BeautyProof-V2")
        self.assertIn(evidence["reliability"], {"Low", "Medium", "High"})
        self.assertEqual(set(evidence["metrics"]), {"skin_smoothing", "texture_loss", "whitening"})
        self.assertEqual(evidence["threshold"], 0.5)

    def test_before_after_analysis(self):
        response = self.client.post(
            "/api/analyze/before-after",
            data={"backend": "ultra"},
            files={
                "before_image": ("human.jpeg", tiny_png((120, 115, 112)), "image/png"),
                "after_image": ("ai_retouched.png", tiny_png((235, 225, 215)), "image/png"),
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("before_after_evidence", payload)
        self.assertIn("after_beautyproof_v2", payload)
        self.assertIn("visual_evidence", payload)
        self.assertEqual(payload["after_model_output"]["label"], "ai")


if __name__ == "__main__":
    unittest.main()
