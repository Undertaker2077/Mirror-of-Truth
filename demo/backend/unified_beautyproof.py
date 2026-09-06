from __future__ import annotations

import os
import sys
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from .beautyproof_v2 import HEATMAP_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _candidate_api_roots() -> list[Path]:
    roots: list[Path] = []
    env_path = os.getenv("BEAUTYPROOF_API_PATH")
    if env_path:
        roots.append(Path(env_path).expanduser())
    roots.extend(
        [
            PROJECT_ROOT.parent,
            PROJECT_ROOT,
            Path.cwd(),
        ]
    )
    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_roots.append(resolved)
    return unique_roots


def find_unified_api_root() -> Path | None:
    for root in _candidate_api_roots():
        if (root / "beautyproof_api" / "unified.py").exists():
            return root
    return None


@lru_cache(maxsize=1)
def _load_api():
    if os.getenv("BEAUTYPROOF_USE_UNIFIED") != "1":
        raise RuntimeError("set BEAUTYPROOF_USE_UNIFIED=1 to load BeautyProof Unified models")
    root = find_unified_api_root()
    if root is None:
        searched = ", ".join(str(path) for path in _candidate_api_roots())
        raise RuntimeError(
            "BeautyProof Unified API package not found. "
            "Set BEAUTYPROOF_API_PATH to the Mirror-of-Truth repo root. "
            f"Searched: {searched}"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from beautyproof_api import UnifiedBeautyProofAPI

    return UnifiedBeautyProofAPI(retouch_threshold=0.5, type_threshold=0.5)


def unified_status() -> dict[str, Any]:
    root = find_unified_api_root()
    model_paths = {}
    if root is not None:
        model_paths = {
            "v2": str(root / "models" / "retouch_detector_v2" / "best_model.pt"),
            "type_three_v1": str(root / "models" / "retouch_three_type_v1" / "best_model.pt"),
            "type_cnn_v1_legacy": str(root / "models" / "retouch_multitask_cnn_v1" / "best_model.pt"),
            "yolo_regions_v1": str(root / "models" / "yolo_retouch_regions_v1" / "best.pt"),
        }
    return {
        "package_present": root is not None,
        "api_root": str(root) if root else None,
        "real_model_env": "set BEAUTYPROOF_USE_UNIFIED=1 to load the unified BeautyProof pipeline",
        "real_model_enabled": os.getenv("BEAUTYPROOF_USE_UNIFIED") == "1",
        "model_paths": model_paths,
        "endpoint": "/api/beautyproof/unified/analyze",
        "compatible_endpoint": "/v1/analyze",
    }


def analyze_image_bytes(image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
    if not image_bytes:
        raise ValueError("empty image")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise OverflowError("image exceeds 10 MiB limit")

    suffix = Path(filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(image_bytes)
        temp_path = Path(handle.name)
    try:
        return _load_api().analyze(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def analyze_image_bytes_with_heatmap(image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
    if not image_bytes:
        raise ValueError("empty image")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise OverflowError("image exceeds 10 MiB limit")

    suffix = Path(filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(image_bytes)
        temp_path = Path(handle.name)
    try:
        result = _load_api().analyze(temp_path)
        result["_demo_heatmap"] = _generate_v2_heatmap(temp_path, filename)
        return result
    finally:
        temp_path.unlink(missing_ok=True)


def _generate_v2_heatmap(image_path: Path, filename: str | None) -> dict[str, Any] | None:
    root = find_unified_api_root()
    if root is None:
        return None

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    stem = Path(filename or image_path.name).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)[:70]
    output_stem = f"{safe_stem}_{uuid.uuid4().hex[:8]}"
    output_path = HEATMAP_ROOT / f"{output_stem}_heatmap.png"
    checkpoint_path = root / "models" / "retouch_detector_v2" / "best_model.pt"

    try:
        from model_inference import predict

        evidence = predict(image_path, checkpoint_path=checkpoint_path, output_dir=HEATMAP_ROOT)
        heatmap_path = evidence.get("manipulation_map") or evidence.get("visual_evidence", {}).get("manipulation_map")
        if not heatmap_path:
            raise KeyError("manipulation_map")
        generated_path = Path(heatmap_path)
        if generated_path.exists() and generated_path != output_path:
            generated_path.replace(output_path)
        return {
            "manipulation_map": str(output_path),
            "manipulation_map_url": f"/heatmaps/{output_path.name}",
            "source": "BeautyProof-V2 Grad-CAM",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "manipulation_map": None,
            "manipulation_map_url": None,
            "source": "unavailable",
            "error": f"{exc.__class__.__name__}: {exc}",
        }


class UnifiedBeautyProofService:
    def predict_bytes(self, image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
        try:
            result = analyze_image_bytes_with_heatmap(image_bytes, filename)
            public_result = dict(result)
            public_result.pop("_demo_heatmap", None)
            return {
                "status": "ok",
                "mock": False,
                "api_root": unified_status()["api_root"],
                "result": public_result,
                "visual_evidence": to_visual_evidence(result),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "unavailable",
                "mock": False,
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
                "setup": unified_status(),
                "result": None,
                "visual_evidence": None,
            }


def to_visual_evidence(result: dict[str, Any]) -> dict[str, Any]:
    probability = float(result.get("retouch_probability", 0.0))
    confidence = max(probability, 1.0 - probability)
    if confidence >= 0.85:
        reliability = "High"
    elif confidence >= 0.65:
        reliability = "Medium"
    else:
        reliability = "Low"

    type_probs = {item["name"]: float(item["probability"]) for item in result.get("retouch_types", [])}
    skin_probability = type_probs.get("skin_enhancement", type_probs.get("smoothing", probability))
    heatmap = result.get("_demo_heatmap") or {}
    return {
        "schema_version": result.get("schema_version", "1.0"),
        "model_version": "BeautyProof-Unified",
        "model_status": "real",
        "digital_retouch": bool(result.get("retouched", False)),
        "retouch_probability": round(probability, 6),
        "integrity_score": round(probability, 6),
        "confidence": round(confidence, 6),
        "threshold": 0.5,
        "reliability": reliability,
        "retouch_types": result.get("retouch_types", []),
        "retouch_strength": result.get("retouch_strength", "none"),
        "modified_regions": result.get("modified_regions", []),
        "region_status": result.get("region_status", "not_applicable"),
        "manipulation_map": heatmap.get("manipulation_map"),
        "manipulation_map_url": heatmap.get("manipulation_map_url"),
        "heatmap_source": heatmap.get("source"),
        "heatmap_error": heatmap.get("error"),
        "models": result.get("models", {}),
        "limitations": result.get("limitations", []),
        "metrics": {
            "skin_smoothing": {
                "probability": round(skin_probability, 6),
                "basis": "type_classifier" if "skin_enhancement" in type_probs or "smoothing" in type_probs else "binary_score_fallback",
            },
            "texture_loss": {
                "probability": round(skin_probability, 6),
                "basis": "skin_enhancement_proxy",
            },
            "whitening": {
                "probability": round(skin_probability, 6),
                "basis": "skin_enhancement_proxy",
            },
            "face_slimming": {
                "probability": round(type_probs.get("face_slimming", probability), 6),
                "basis": "type_classifier" if "face_slimming" in type_probs else "binary_score_fallback",
            },
            "facial_contouring": {
                "probability": round(type_probs.get("facial_contouring", probability), 6),
                "basis": "type_classifier" if "facial_contouring" in type_probs else "binary_score_fallback",
            },
        },
    }
