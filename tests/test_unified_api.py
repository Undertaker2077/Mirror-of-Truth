from pathlib import Path

from beautyproof_api.unified import UnifiedBeautyProofAPI
from fastapi.testclient import TestClient
from beautyproof_api.server import app


def test_unified_api_returns_three_accepted_types_and_filters_eye_and_jawline(tmp_path: Path):
    image = tmp_path / "face.jpg"
    image.write_bytes(b"fixture")
    api = UnifiedBeautyProofAPI(
        v2_predictor=lambda _: {"probability": .91},
        type_predictor=lambda _: {
            "probabilities": {"skin_enhancement": .84, "face_slimming": .67, "facial_contouring": .71, "eye_enlargement": .99},
            "thresholds": {"skin_enhancement": .5, "face_slimming": .5, "facial_contouring": .5},
            "strengths": {"skin_enhancement": .6, "face_slimming": .5, "facial_contouring": .7},
            "strength": "medium",
        },
        region_predictor=lambda _: [
            {"name": "left_cheek", "confidence": .8, "polygon": [[.1, .2], [.2, .2], [.2, .3]]},
            {"name": "jawline", "confidence": .95, "polygon": [[.1, .8], [.5, .9], [.9, .8]]},
        ],
    )
    result = api.analyze(image)
    assert result["retouched"] is True
    assert [x["name"] for x in result["retouch_types"]] == ["skin_enhancement", "face_slimming", "facial_contouring"]
    assert [x["name"] for x in result["modified_regions"]] == ["left_cheek"]
    assert result["models"]["binary_detector"] == "BeautyProof-V2"


def test_v2_is_authoritative_when_regions_are_empty(tmp_path: Path):
    image = tmp_path / "face.jpg"; image.write_bytes(b"fixture")
    api = UnifiedBeautyProofAPI(
        v2_predictor=lambda _: {"probability": .8},
        type_predictor=lambda _: {"probabilities": {"skin_enhancement": .2}, "strengths": {}},
        region_predictor=lambda _: [],
    )
    result = api.analyze(image)
    assert result["retouched"] is True
    assert result["region_status"] == "not_localized"


def test_negative_v2_suppresses_downstream_claims(tmp_path: Path):
    image = tmp_path / "face.jpg"; image.write_bytes(b"fixture")
    api = UnifiedBeautyProofAPI(
        v2_predictor=lambda _: {"probability": .12},
        type_predictor=lambda _: {"probabilities": {"skin_enhancement": .99}, "strengths": {}},
        region_predictor=lambda _: [{"name": "nose", "confidence": .9, "polygon": []}],
    )
    result = api.analyze(image)
    assert result["retouched"] is False
    assert result["retouch_types"] == [] and result["modified_regions"] == []


def test_rest_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "api_version": "1.0.0"}

def test_rest_rejects_oversized_upload():
    response=TestClient(app).post("/v1/analyze",files={"image":("large.jpg",b"x"*(10*1024*1024+1),"image/jpeg")})
    assert response.status_code==413
