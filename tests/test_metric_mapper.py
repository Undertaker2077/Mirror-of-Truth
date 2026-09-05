import pytest

from metric_mapper import map_business_metrics, reliability_level


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.49, "Low"), (0.50, "Medium"), (0.89, "Medium"), (0.90, "High")],
)
def test_score_boundaries_map_to_documented_levels(score, expected):
    metrics = map_business_metrics(score)
    assert metrics["skin_smoothing"]["level"] == expected
    assert metrics["texture_loss"]["level"] == expected
    assert metrics["whitening"]["level"] == expected
    assert all(item["basis"] == "binary_score_proxy" for item in metrics.values())


def test_mapper_rejects_scores_outside_probability_range():
    with pytest.raises(ValueError, match="between 0 and 1"):
        map_business_metrics(1.01)


@pytest.mark.parametrize(
    ("confidence", "concentration", "expected"),
    [(0.95, 0.75, "High"), (0.80, 0.50, "Medium"), (0.55, 0.10, "Low")],
)
def test_reliability_combines_confidence_and_map_concentration(
    confidence, concentration, expected
):
    assert reliability_level(confidence, concentration)["level"] == expected

