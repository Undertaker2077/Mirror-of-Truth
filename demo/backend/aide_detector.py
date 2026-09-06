from __future__ import annotations

import io
import math
import os
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIDE_ROOT = PROJECT_ROOT / "vendor" / "AIDE"
AIDE_MODEL_DIR = PROJECT_ROOT / "models" / "aide"
AIDE_CHECKPOINT = AIDE_MODEL_DIR / "aide.pth"
AIDE_RESNET = AIDE_MODEL_DIR / "resnet50.pth"
AIDE_CONVNEXT = AIDE_MODEL_DIR / "open_clip_pytorch_model.bin"
RAW_THRESHOLD = 0.5
CALIBRATED_THRESHOLD = 0.3
CALIBRATION_GAMMA = 2.5


def _rounded(value: float) -> float:
    return round(float(value), 6)


def calibrate_ai_probability(raw_probability: float) -> float:
    raw_probability = max(0.0, min(1.0, float(raw_probability)))
    return 1.0 - (1.0 - raw_probability) ** CALIBRATION_GAMMA


def aide_status() -> dict[str, Any]:
    return {
        "repo_present": AIDE_ROOT.exists(),
        "checkpoint_present": AIDE_CHECKPOINT.exists(),
        "resnet_present": AIDE_RESNET.exists(),
        "convnext_present": AIDE_CONVNEXT.exists(),
        "checkpoint": str(AIDE_CHECKPOINT),
        "resnet": str(AIDE_RESNET),
        "convnext": str(AIDE_CONVNEXT),
        "device": _select_device().type,
    }


def _select_device() -> torch.device:
    requested = os.getenv("MIRROR_AIDE_DEVICE", "").strip().lower()
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_paths() -> None:
    missing = [
        str(path)
        for path in [AIDE_ROOT, AIDE_CHECKPOINT, AIDE_RESNET, AIDE_CONVNEXT]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("AIDE files missing: " + ", ".join(missing))
    if str(AIDE_ROOT) not in sys.path:
        sys.path.insert(0, str(AIDE_ROOT))


def _install_unused_clip_stub() -> None:
    # AIDE imports openai-clip, but its inference path here only uses open_clip.
    # Keeping a stub avoids installing an unused package into the demo runtime.
    sys.modules.setdefault("clip", types.ModuleType("clip"))


@lru_cache(maxsize=1)
def _load_aide_model() -> tuple[torch.nn.Module, torch.device]:
    _ensure_paths()
    _install_unused_clip_stub()

    from data.dct import DCT_base_Rec_Module  # noqa: F401
    from models.AIDE import AIDE

    device = _select_device()
    model = AIDE(resnet_path=str(AIDE_RESNET), convnext_path=str(AIDE_CONVNEXT))
    checkpoint = torch.load(str(AIDE_CHECKPOINT), map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, device


@lru_cache(maxsize=1)
def _dct_module():
    _ensure_paths()
    from data.dct import DCT_base_Rec_Module

    return DCT_base_Rec_Module().eval()


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    transform_before_test = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    with Image.open(io.BytesIO(image_bytes)) as image:
        base = transform_before_test(image.convert("RGB"))

    with torch.no_grad():
        dct_variants = _dct_module()(base)
    stacked = [transform_test(item) for item in [*dct_variants, base]]
    return torch.stack(stacked, dim=0).unsqueeze(0)


class AIDEGenerationDetector:
    """Inference adapter for shilinyan99/AIDE GenImage checkpoint."""

    def detect(self, image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
        model, device = _load_aide_model()
        batch = _preprocess(image_bytes).to(device)
        with torch.no_grad():
            logits = model(batch)
            raw_probability_ai = torch.softmax(logits, dim=1)[0, 1].item()
        if not math.isfinite(raw_probability_ai):
            raise RuntimeError(
                f"AIDE returned non-finite probability on device={device.type}; "
                "try MIRROR_AIDE_DEVICE=cpu"
            )
        probability_ai = calibrate_ai_probability(raw_probability_ai)
        probability_real = 1.0 - probability_ai
        label = "ai" if probability_ai >= CALIBRATED_THRESHOLD else "real"
        return {
            "path": None,
            "filename": filename,
            "label": label,
            "probability_ai": _rounded(probability_ai),
            "probability_real": _rounded(probability_real),
            "confidence": _rounded(max(probability_ai, probability_real)),
            "raw_probability_ai": _rounded(raw_probability_ai),
            "raw_probability_real": _rounded(1.0 - raw_probability_ai),
            "raw_score": _rounded(probability_ai - CALIBRATED_THRESHOLD),
            "backend": "aide-genimage",
            "status": "ok",
            "mock": False,
            "source": "shilinyan99/AIDE",
            "model_name": "AIDE",
            "checkpoint": "GenImage_train.pth",
            "threshold": CALIBRATED_THRESHOLD,
            "raw_threshold": RAW_THRESHOLD,
            "calibration": f"sensitive_gamma_{CALIBRATION_GAMMA}",
            "schema_version": "ai_detector.v1",
        }
