from __future__ import annotations

from pathlib import Path
from typing import Callable, Any

SUPPORTED_TYPES = ("skin_enhancement", "face_slimming", "facial_contouring")
SUPPORTED_REGIONS = ("forehead", "left_cheek", "right_cheek", "nose", "chin", "full_face")


class UnifiedBeautyProofAPI:
    """Combine V2 binary detection, effect classification and region segmentation."""

    def __init__(self, *, v2_predictor: Callable | None = None,
                 type_predictor: Callable | None = None,
                 region_predictor: Callable | None = None,
                 retouch_threshold: float = .5, type_threshold: float | None = None):
        if not 0 <= retouch_threshold <= 1 or (type_threshold is not None and not 0 <= type_threshold <= 1):
            raise ValueError("thresholds must be between 0 and 1")
        if v2_predictor is None or type_predictor is None or region_predictor is None:
            from .production import v2_predict, type_predict, region_predict
        self.v2_predictor = v2_predictor or v2_predict
        self.type_predictor = type_predictor or type_predict
        self.region_predictor = region_predictor or region_predict
        self.retouch_threshold = retouch_threshold
        self.type_threshold = type_threshold

    def analyze(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        v2 = self.v2_predictor(path)
        probability = float(v2["probability"])
        retouched = probability >= self.retouch_threshold
        type_output = self.type_predictor(path) if retouched else {"probabilities": {}, "strengths": {}}
        raw_regions = self.region_predictor(path) if retouched else []
        types = [{"name": name, "probability": round(float(type_output.get("probabilities", {}).get(name, 0)), 6),
                  "strength": round(float(type_output.get("strengths", {}).get(name, 0)), 6)}
                 for name in SUPPORTED_TYPES
                 if float(type_output.get("probabilities", {}).get(name, 0)) >= (self.type_threshold if self.type_threshold is not None else float(type_output.get("thresholds", {}).get(name, .5)))]
        regions = [r for r in raw_regions if r.get("name") in SUPPORTED_REGIONS]
        return {
            "schema_version": "1.0",
            "retouched": retouched,
            "retouch_probability": round(probability, 6),
            "retouch_types": types,
            "retouch_strength": max((x["strength"] for x in types), default=0.0),
            "modified_regions": regions,
            "region_status": "localized" if regions else ("not_localized" if retouched else "not_applicable"),
            "models": {"binary_detector": "BeautyProof-V2", "type_classifier": "BeautyProof-Retouch-Three-Type-V1",
                       "region_segmenter": "YOLO11n-Seg-30e"},
            "limitations": [
                "The three retouch types and region localization are validated primarily on synthetic data.",
                "A missing region does not override the V2 binary decision.",
                "This API does not assess makeup, eye enlargement, or product efficacy.",
            ],
        }
