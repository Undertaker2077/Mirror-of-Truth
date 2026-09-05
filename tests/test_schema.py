import json

import pytest

from visual_evidence_schema import VisualEvidence


def test_visual_evidence_serializes_fixed_contract():
    evidence = VisualEvidence(
        integrity_score=0.73,
        reliability="Medium",
        metrics={"skin_smoothing": {"level": "Medium"}},
        manipulation_map="outputs/sample_heatmap.png",
    )
    payload = json.loads(evidence.to_json())
    assert payload["visual_evidence"]["integrity_score"] == 0.73
    assert payload["visual_evidence"]["reliability"] == "Medium"
    assert payload["visual_evidence"]["model_version"] == "BeautyProof-V2"


def test_visual_evidence_rejects_invalid_integrity_score():
    with pytest.raises(ValueError, match="between 0 and 1"):
        VisualEvidence(1.2, "Low", {}, None)

