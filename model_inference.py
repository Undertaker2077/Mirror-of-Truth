"""Engineering wrapper for BeautyProof V2 inference and visual evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

from heatmap_generator import generate_heatmap_overlay, map_concentration, normalize_map
from metric_mapper import map_business_metrics, reliability_level
from visual_evidence_schema import VisualEvidence


MODEL_CANDIDATES = (
    Path("best_model.pt"),
    Path("models/retouch_detector_v2/best_model.pt"),
)


def inference_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def resolve_checkpoint(checkpoint_path: str | Path | None = None) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("best_model.pt not found in repository root or models/retouch_detector_v2")


def load_model(checkpoint_path: str | Path, device: str) -> nn.Module:
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    checkpoint = torch.load(resolve_checkpoint(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def _gradcam(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    if not hasattr(model, "features"):
        return np.zeros((7, 7), dtype=np.float32)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    layer = model.features[-1]
    forward_handle = layer.register_forward_hook(
        lambda _module, _inputs, output: activations.__setitem__("value", output)
    )
    backward_handle = layer.register_full_backward_hook(
        lambda _module, _inputs, output: gradients.__setitem__("value", output[0])
    )
    try:
        model.zero_grad(set_to_none=True)
        model(tensor).flatten()[0].backward()
        weights = gradients["value"].mean((2, 3), keepdim=True)
        cam = torch.relu((weights * activations["value"]).sum(1)).squeeze()
        return normalize_map(cam.detach().cpu().numpy())
    finally:
        forward_handle.remove()
        backward_handle.remove()


def predict_raw(
    image_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    model: nn.Module | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = (model or load_model(resolve_checkpoint(checkpoint_path), device)).to(device).eval()
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        original_size = rgb.size
        tensor = inference_transform()(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(tensor).flatten()[0]).item()
    cam = _gradcam(model, tensor)
    full_map = cv2.resize(cam, original_size, interpolation=cv2.INTER_CUBIC)
    return {"score": float(score), "manipulation_map": normalize_map(full_map)}


def predict(
    image_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    image_path = Path(image_path)
    raw = predict_raw(image_path, checkpoint_path=checkpoint_path)
    heatmap_path = Path(output_dir) / f"{image_path.stem}_heatmap.png"
    generate_heatmap_overlay(image_path, raw["manipulation_map"], heatmap_path)
    confidence = max(raw["score"], 1.0 - raw["score"])
    reliability = reliability_level(confidence, map_concentration(raw["manipulation_map"]))
    evidence = VisualEvidence(
        integrity_score=round(raw["score"], 6),
        reliability=reliability["level"],
        reliability_details=reliability,
        metrics=map_business_metrics(raw["score"]),
        manipulation_map=heatmap_path.as_posix(),
    )
    return evidence.to_dict()

