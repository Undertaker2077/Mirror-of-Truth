from __future__ import annotations

import hashlib
import io
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "BeautyProof-V2"
THRESHOLD = 0.5
EXPECTED_SHA256 = "52F38353CEB4F20325B8AF84C0A0973FD48FEB57323B4429465D5C10FCFDC94D"
EXPECTED_BYTES = 16_328_815
CHECKPOINT_CANDIDATES = (
    PROJECT_ROOT / "best_model.pt",
    PROJECT_ROOT / "models" / "retouch_detector_v2" / "best_model.pt",
    PROJECT_ROOT.parent / "best_model.pt",
    PROJECT_ROOT.parent / "models" / "retouch_detector_v2" / "best_model.pt",
)
HEATMAP_ROOT = PROJECT_ROOT / "runtime" / "heatmaps"


PROXY_LIMITATION = (
    "Proxy derived from the V2 binary retouch score; not an independent detector "
    "for this manipulation type."
)
LIMITATIONS = [
    "Business metrics are proxies from one binary score.",
    "Grad-CAM is model attention, not pixel-level forensic ground truth.",
    "Do not use as the sole basis for legal, medical, hiring, or identity decisions.",
]


def _round(value: float) -> float:
    return round(float(value), 6)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _level(score: float) -> str:
    if score < 0.5:
        return "Low"
    if score < 0.9:
        return "Medium"
    return "High"


def _reliability_level(score: float) -> str:
    if score < 0.65:
        return "Low"
    if score < 0.8:
        return "Medium"
    return "High"


def _metric(score: float) -> dict[str, Any]:
    return {
        "level": _level(score),
        "score": _round(score),
        "basis": "binary_score_proxy",
        "limitation": PROXY_LIMITATION,
    }


def _find_checkpoint(explicit_path: str | Path | None = None) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _validate_checkpoint(path: Path) -> dict[str, Any]:
    size_bytes = path.stat().st_size
    sha256 = _sha256(path)
    return {
        "path": str(path),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "expected_size_bytes": EXPECTED_BYTES,
        "expected_sha256": EXPECTED_SHA256,
        "sha256_match": sha256 == EXPECTED_SHA256,
        "size_match": size_bytes == EXPECTED_BYTES,
    }


def _open_rgb(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _mock_score(image_bytes: bytes, filename: str | None) -> float:
    name = (filename or "").lower()
    if "retouch" in name or "beauty" in name or "ai" in name:
        return 0.78
    if "human" in name or "raw" in name or "before" in name:
        return 0.24
    digest = hashlib.sha256(image_bytes).digest()
    return 0.18 + (digest[1] / 255.0) * 0.68


def _make_mock_map(image: Image.Image) -> np.ndarray:
    gray = image.convert("L").resize((224, 224))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    array = np.asarray(edges, dtype=np.float32)
    if array.max() > array.min():
        array = (array - array.min()) / (array.max() - array.min())
    else:
        array = np.zeros_like(array)
    return 1.0 - array


def _map_concentration(cam: np.ndarray) -> float:
    flat = np.asarray(cam, dtype=np.float32).flatten()
    if flat.size == 0 or float(flat.max()) <= 0:
        return 0.0
    threshold = np.quantile(flat, 0.85)
    high = flat[flat >= threshold]
    return _clamp(float(high.mean() / (flat.mean() + 1e-6)) / 4.0)


def _save_heatmap_overlay(image: Image.Image, cam: np.ndarray, output_dir: Path, filename: str | None) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename or f"upload_{uuid.uuid4().hex}").stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)[:80]
    output_path = output_dir / f"{safe_stem}_heatmap.png"

    cam_img = Image.fromarray(np.uint8(_clamp_array(cam) * 255), mode="L").resize(image.size)
    red = Image.new("RGBA", image.size, (220, 38, 38, 0))
    red.putalpha(cam_img.point(lambda p: int(p * 0.52)))
    base = image.convert("RGBA")
    overlay = Image.alpha_composite(base, red).convert("RGB")
    overlay.save(output_path)
    return str(output_path)


def _clamp_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.max() > array.min():
        array = (array - array.min()) / (array.max() - array.min())
    return np.clip(array, 0.0, 1.0)


@lru_cache(maxsize=2)
def _load_model(checkpoint_path: str):
    import torch
    from torch import nn
    from torchvision import transforms
    from torchvision.models import efficientnet_b0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, preprocess, device, torch


def _predict_real(image: Image.Image, checkpoint_path: Path) -> tuple[float, np.ndarray]:
    model, preprocess, device, torch = _load_model(str(checkpoint_path))
    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    target_layer = model.features[-1]
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    try:
        batch = preprocess(image).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        logit = model(batch).flatten()[0]
        probability = torch.sigmoid(logit).item()
        logit.backward()
        if activations is None or gradients is None:
            cam = _make_mock_map(image)
        else:
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam_tensor = torch.relu((weights * activations).sum(dim=1, keepdim=False))[0]
            cam = cam_tensor.detach().cpu().numpy()
            cam = _clamp_array(cam)
        return float(probability), cam
    finally:
        handle_forward.remove()
        handle_backward.remove()


def _visual_evidence(
    *,
    score: float,
    manipulation_map: str | None,
    map_concentration: float,
    model_status: str,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    classification_confidence = max(score, 1.0 - score)
    reliability_score = 0.7 * classification_confidence + 0.3 * map_concentration
    heatmap_url = None
    if manipulation_map:
        heatmap_url = f"/heatmaps/{Path(manipulation_map).name}"
    return {
        "integrity_score": _round(score),
        "digital_retouch": score >= THRESHOLD,
        "retouch_probability": _round(score),
        "confidence": _round(classification_confidence),
        "reliability": _reliability_level(reliability_score),
        "metrics": {
            "skin_smoothing": _metric(score),
            "texture_loss": _metric(score),
            "whitening": _metric(score),
        },
        "manipulation_map": manipulation_map,
        "manipulation_map_url": heatmap_url,
        "reliability_details": {
            "level": _reliability_level(reliability_score),
            "score": _round(reliability_score),
            "classification_confidence": _round(classification_confidence),
            "map_concentration": _round(map_concentration),
            "method": "0.7 * classification_confidence + 0.3 * Grad-CAM concentration",
        },
        "model_version": MODEL_VERSION,
        "model_status": model_status,
        "checkpoint": checkpoint,
        "threshold": THRESHOLD,
        "limitations": LIMITATIONS,
    }


class BeautyProofV2Service:
    def __init__(self, checkpoint_path: str | Path | None = None, output_dir: str | Path | None = None) -> None:
        self.real_model_enabled = os.getenv("BEAUTYPROOF_USE_REAL") == "1"
        self.checkpoint_path = _find_checkpoint(checkpoint_path) if self.real_model_enabled else None
        self.output_dir = Path(output_dir) if output_dir else HEATMAP_ROOT
        self.allow_mock = os.getenv("BEAUTYPROOF_DISABLE_MOCK") != "1"

    def predict_bytes(self, image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
        image = _open_rgb(image_bytes)
        if self.checkpoint_path is None:
            if not self.allow_mock:
                return {
                    "error": {
                        "code": "MODEL_NOT_FOUND",
                        "message": "BeautyProof V2 checkpoint best_model.pt was not found.",
                        "retryable": False,
                    }
                }
            score = _mock_score(image_bytes, filename)
            cam = _make_mock_map(image)
            map_path = _save_heatmap_overlay(image, cam, self.output_dir, filename)
            return {
                "visual_evidence": _visual_evidence(
                    score=score,
                    manipulation_map=map_path,
                    map_concentration=_map_concentration(cam),
                    model_status="mock_pending_checkpoint",
                    checkpoint=None,
                )
            }

        checkpoint = _validate_checkpoint(self.checkpoint_path)
        if not checkpoint["sha256_match"]:
            return {
                "error": {
                    "code": "CHECKPOINT_HASH_MISMATCH",
                    "message": "BeautyProof V2 checkpoint SHA256 does not match the frozen interface document.",
                    "retryable": False,
                    "details": checkpoint,
                }
            }
        score, cam = _predict_real(image, self.checkpoint_path)
        map_path = _save_heatmap_overlay(image, cam, self.output_dir, filename)
        return {
            "visual_evidence": _visual_evidence(
                score=score,
                manipulation_map=map_path,
                map_concentration=_map_concentration(cam),
                model_status="real_checkpoint",
                checkpoint=checkpoint,
            )
        }
