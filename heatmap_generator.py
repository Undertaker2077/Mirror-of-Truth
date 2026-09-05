"""Render normalized manipulation maps as red-highlight overlays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def normalize_map(manipulation_map: np.ndarray) -> np.ndarray:
    values = np.asarray(manipulation_map, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("manipulation_map must be a non-empty 2D array")
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    minimum, maximum = float(values.min()), float(values.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(values)
    return np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)


def map_concentration(manipulation_map: np.ndarray) -> float:
    values = normalize_map(manipulation_map)
    total = float(values.sum())
    if total <= 1e-8:
        return 0.0
    probabilities = (values / total).ravel()
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    maximum_entropy = float(np.log(values.size))
    return float(np.clip(1.0 - entropy / maximum_entropy, 0.0, 1.0))


def generate_heatmap_overlay(
    image_path: str | Path,
    manipulation_map: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.45,
) -> Path:
    image_path, output_path = Path(image_path), Path(output_path)
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    normalized = normalize_map(manipulation_map)
    resized = cv2.resize(normalized, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_bgr = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 1.0 - alpha, heat_rgb, alpha, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path, format="PNG")
    return output_path

