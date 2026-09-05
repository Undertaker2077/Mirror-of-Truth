"""Map BeautyProof V2's binary score to documented business-facing proxies."""

from __future__ import annotations


LOW_MAX = 0.50
HIGH_MIN = 0.90


def _validate_probability(value: float, name: str = "score") -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def score_level(score: float) -> str:
    score = _validate_probability(score)
    if score < LOW_MAX:
        return "Low"
    if score < HIGH_MIN:
        return "Medium"
    return "High"


def map_business_metrics(score: float) -> dict[str, dict[str, object]]:
    """Return proxy levels; V2 does not have independent manipulation heads."""
    score = _validate_probability(score)
    level = score_level(score)
    note = (
        "Proxy derived from the V2 binary retouch score; not an independent "
        "detector for this manipulation type."
    )
    return {
        name: {
            "level": level,
            "score": round(score, 6),
            "basis": "binary_score_proxy",
            "limitation": note,
        }
        for name in ("skin_smoothing", "texture_loss", "whitening")
    }


def reliability_level(confidence: float, map_concentration: float) -> dict[str, object]:
    confidence = _validate_probability(confidence, "confidence")
    map_concentration = _validate_probability(map_concentration, "map_concentration")
    combined = 0.7 * confidence + 0.3 * map_concentration
    level = "High" if combined >= 0.80 else "Medium" if combined >= 0.65 else "Low"
    return {
        "level": level,
        "score": round(combined, 6),
        "classification_confidence": round(confidence, 6),
        "map_concentration": round(map_concentration, 6),
        "method": "0.7 * classification_confidence + 0.3 * Grad-CAM concentration",
    }

