from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Any

import torch
from PIL import Image


MODEL_ID = os.getenv("MIRROR_HF3_MODEL_ID", "prithivMLmods/AI-vs-Deepfake-vs-Real")
THRESHOLD = float(os.getenv("MIRROR_HF3_THRESHOLD", "0.5"))


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _select_device() -> torch.device:
    requested = os.getenv("MIRROR_HF3_DEVICE", "").strip().lower()
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hf_three_way_status() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "threshold": THRESHOLD,
        "device": _select_device().type,
        "backend": "hf-ai-deepfake-real",
        "env_model_id": "optional MIRROR_HF3_MODEL_ID",
    }


@lru_cache(maxsize=1)
def _load_model():
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = _select_device()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model, device


def _normalized_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


class HuggingFaceThreeWayAIDetector:
    """Adapter for prithivMLmods/AI-vs-Deepfake-vs-Real.

    The public app contract remains binary via probability_ai/probability_real,
    while raw three-way probabilities are included for demo rendering.
    """

    def detect(self, image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
        processor, model, device = _load_model()
        with Image.open(io.BytesIO(image_bytes)) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()

        id2label = getattr(model.config, "id2label", {}) or {}
        class_scores: dict[str, float] = {}
        raw_class_scores: dict[str, float] = {}
        for index, score in enumerate(probs.tolist()):
            label = str(id2label.get(index, index))
            raw_class_scores[label] = _rounded(score)
            class_scores[_normalized_label(label)] = float(score)

        artificial = class_scores.get("artificial", 0.0)
        deepfake = class_scores.get("deepfake", 0.0)
        real = class_scores.get("real", 0.0)
        probability_ai = max(0.0, min(1.0, artificial + deepfake))
        probability_real = max(0.0, min(1.0, real if real else 1.0 - probability_ai))

        top_index = int(torch.argmax(probs).item())
        raw_label = str(id2label.get(top_index, top_index))
        normalized_top = _normalized_label(raw_label)
        label = "real" if normalized_top == "real" and probability_ai < THRESHOLD else "ai"

        return {
            "path": None,
            "filename": filename,
            "label": label,
            "probability_ai": _rounded(probability_ai),
            "probability_real": _rounded(probability_real),
            "confidence": _rounded(max(probability_ai, probability_real)),
            "raw_score": _rounded(probability_ai - THRESHOLD),
            "backend": "hf-ai-deepfake-real",
            "status": "ok",
            "mock": False,
            "source": MODEL_ID,
            "model_name": "AI-vs-Deepfake-vs-Real",
            "checkpoint": MODEL_ID,
            "threshold": THRESHOLD,
            "schema_version": "ai_detector.v1",
            "raw_label": raw_label,
            "probability_artificial": _rounded(artificial),
            "probability_deepfake": _rounded(deepfake),
            "class_probabilities": raw_class_scores,
        }
