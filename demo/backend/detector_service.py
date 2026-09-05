from __future__ import annotations

import hashlib
import io
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

from .beautyproof_v2 import BeautyProofV2Service
from .unified_beautyproof import UnifiedBeautyProofService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DETECTOR = PROJECT_ROOT / "vendor" / "ai-image-detector"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _open_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


@lru_cache(maxsize=4)
def _load_real_detector(backend: str):
    if str(VENDOR_DETECTOR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DETECTOR))
    from aidetector import create_detector

    return create_detector(backend, device="auto")


class AIDetectorService:
    """Adapter around lynote-ai/ai-image-detector with a smoke-test fallback."""

    def __init__(self, backend: str = "ultra") -> None:
        self.backend = backend
        self.real_model_enabled = os.getenv("MIRROR_USE_REAL_AIDETECTOR") == "1"

    def detect(self, image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
        if self.real_model_enabled:
            try:
                image = _open_image(image_bytes)
                detector = _load_real_detector(self.backend)
                result = detector.predict_image(image)
                payload = result.as_dict()
                payload.update(
                    {
                        "filename": filename,
                        "status": "ok",
                        "mock": False,
                        "source": "lynote-ai/ai-image-detector",
                    }
                )
                return payload
            except Exception as exc:  # noqa: BLE001
                return self._mock_detection(
                    image_bytes,
                    filename,
                    unavailable_reason=f"real detector unavailable: {exc}",
                )
        return self._mock_detection(image_bytes, filename, unavailable_reason=None)

    def _mock_detection(
        self,
        image_bytes: bytes,
        filename: str | None,
        unavailable_reason: str | None,
    ) -> dict[str, Any]:
        name = (filename or "").lower()
        if "ai-generated" in name:
            probability_ai = 0.74
        elif "ai_retouched" in name or "retouched" in name:
            probability_ai = 0.68
        elif "human" in name:
            probability_ai = 0.18
        else:
            digest = hashlib.sha256(image_bytes).digest()
            probability_ai = 0.25 + (digest[0] / 255.0) * 0.5
        label = "ai" if probability_ai >= 0.5 else "real"
        probability_real = 1.0 - probability_ai
        return {
            "path": None,
            "filename": filename,
            "label": label,
            "probability_ai": _rounded(probability_ai),
            "probability_real": _rounded(probability_real),
            "confidence": _rounded(max(probability_ai, probability_real)),
            "raw_score": _rounded(probability_ai - 0.5),
            "backend": f"{self.backend}-mock",
            "status": "mock" if unavailable_reason is None else "fallback",
            "mock": True,
            "source": "filename/image-hash smoke fallback",
            "unavailable_reason": unavailable_reason,
        }


def estimate_retouch_signals(image_bytes: bytes) -> dict[str, Any]:
    image = _open_image(image_bytes)
    small = image.resize((256, max(1, int(256 * image.height / image.width))))
    gray = small.convert("L")
    brightness = ImageStat.Stat(gray).mean[0] / 255.0
    rgb_means = [value / 255.0 for value in ImageStat.Stat(small).mean]
    detail = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0] / 255.0
    smoothness = _clamp(1.0 - detail * 4.2)
    highlight_bias = _clamp((brightness - 0.56) * 2.2)
    channel_spread = max(rgb_means) - min(rgb_means)
    white_balance_shift = _clamp(channel_spread * 3.0)
    retouch_score = _clamp(0.5 * smoothness + 0.3 * highlight_bias + 0.2 * white_balance_shift)
    tags: list[str] = []
    if smoothness >= 0.58:
        tags.append("疑似磨皮/柔焦")
    if highlight_bias >= 0.45:
        tags.append("疑似高光提亮")
    if white_balance_shift >= 0.35:
        tags.append("疑似白平衡/滤镜偏移")
    return {
        "brightness": _rounded(brightness),
        "smoothness_score": _rounded(smoothness),
        "highlight_score": _rounded(highlight_bias),
        "white_balance_shift": _rounded(white_balance_shift),
        "retouch_score": _rounded(retouch_score),
        "tags": tags,
    }


def compare_images_basic(before_bytes: bytes, after_bytes: bytes) -> dict[str, Any]:
    before = estimate_retouch_signals(before_bytes)
    after = estimate_retouch_signals(after_bytes)
    exposure_delta = after["brightness"] - before["brightness"]
    smooth_delta = after["smoothness_score"] - before["smoothness_score"]
    wb_delta = after["white_balance_shift"] - before["white_balance_shift"]
    significant = [
        abs(exposure_delta) >= 0.08,
        smooth_delta >= 0.10,
        abs(wb_delta) >= 0.08,
    ]
    reliability = "Low" if sum(significant) >= 2 else "Medium" if any(significant) else "High"
    notices: list[str] = []
    if exposure_delta >= 0.08:
        notices.append(f"after 图整体亮度提高约 {round(exposure_delta * 100)}%，提亮可能放大妆效。")
    if smooth_delta >= 0.10:
        notices.append("after 图纹理细节明显降低，疑似存在磨皮或柔焦。")
    if abs(wb_delta) >= 0.08:
        notices.append("before/after 白平衡差异明显，色调变化会影响遮瑕、提亮判断。")
    if not notices:
        notices.append("两图基础曝光和纹理差异不大，图像条件相对可比。")
    return {
        "exposure_diff": f"{exposure_delta:+.1%}",
        "white_balance_diff": "Significant" if abs(wb_delta) >= 0.08 else "Similar",
        "texture_smoothing_diff": f"{smooth_delta:+.1%}",
        "face_angle_similarity": "Unknown",
        "crop_similarity": "Unknown",
        "comparison_reliability": reliability,
        "agent_should_notice": notices,
        "before_retouch": before,
        "after_retouch": after,
    }


def build_single_analysis(image_bytes: bytes, filename: str | None, mode: str, backend: str) -> dict[str, Any]:
    model = AIDetectorService(backend=backend).detect(image_bytes, filename)
    beautyproof_unified = UnifiedBeautyProofService().predict_bytes(image_bytes, filename)
    legacy_beautyproof = None
    visual_evidence = beautyproof_unified.get("visual_evidence")
    if visual_evidence is None:
        legacy_beautyproof = BeautyProofV2Service().predict_bytes(image_bytes, filename)
        visual_evidence = legacy_beautyproof.get("visual_evidence")
    retouch = estimate_retouch_signals(image_bytes)
    ai_prob = float(model["probability_ai"])
    beautyproof_score = (
        float(visual_evidence["integrity_score"]) if visual_evidence else float(retouch["retouch_score"])
    )
    confidence = _clamp(0.52 * beautyproof_score + 0.32 * ai_prob + 0.16 * float(retouch["retouch_score"]))
    if mode == "makeup":
        evidence = [
            "单图只能判断 AI/后期/美颜风险，不能直接证明某个化妆品功效。",
            "若图片存在磨皮、提亮、滤镜或五官调整，化妆品功效归因应判为 CONFOUNDED。",
        ]
    else:
        evidence = [
            "服装卖家秀单图重点检测 AI 生成和过度修图风险。",
            "如果 AI 概率较高，面料、版型、上身效果都不应直接视为真实证据。",
        ]
    if visual_evidence:
        evidence.append(
            f"BeautyProof 修图概率 {visual_evidence['retouch_probability']:.1%}，"
            f"可靠性 {visual_evidence['reliability']}。"
        )
        if visual_evidence.get("model_version") == "BeautyProof-Unified":
            evidence.append(
                f"Unified 输出：strength={visual_evidence.get('retouch_strength', 'none')}，"
                f"region_status={visual_evidence.get('region_status', 'N/A')}。"
            )
            if visual_evidence.get("retouch_types"):
                types = ", ".join(
                    f"{item['name']} {float(item['probability']):.1%}"
                    for item in visual_evidence["retouch_types"]
                )
                evidence.append(f"修图类型假设：{types}。")
            if visual_evidence.get("modified_regions"):
                regions = ", ".join(item["name"] for item in visual_evidence["modified_regions"])
                evidence.append(f"疑似区域假设：{regions}。")
        else:
            evidence.append("skin_smoothing / texture_loss / whitening 目前都是二分类分数代理，不是独立分项模型。")
    else:
        evidence.append(f"BeautyProof Unified 暂不可用：{beautyproof_unified.get('error', {}).get('message', 'unknown error')}")
    evidence.extend(retouch["tags"] or ["未检测到强烈的基础后期信号"])
    return {
        "mode": mode,
        "verdict": "High risk" if confidence >= 0.62 else "Medium risk" if confidence >= 0.42 else "Low risk",
        "false_advertising_confidence": _rounded(confidence),
        "groundtruth_label_suggestion": "CONFOUNDED" if confidence >= 0.42 else "INSUFFICIENT",
        "beautyproof_unified": beautyproof_unified,
        "beautyproof_v2": legacy_beautyproof,
        "visual_evidence": visual_evidence,
        "model_output": model,
        "retouch_signals": retouch,
        "evidence": evidence,
    }


def build_before_after_analysis(
    before_bytes: bytes,
    after_bytes: bytes,
    before_filename: str | None,
    after_filename: str | None,
    backend: str,
) -> dict[str, Any]:
    before_model = AIDetectorService(backend=backend).detect(before_bytes, before_filename)
    after_model = AIDetectorService(backend=backend).detect(after_bytes, after_filename)
    before_beautyproof = UnifiedBeautyProofService().predict_bytes(before_bytes, before_filename)
    after_beautyproof = UnifiedBeautyProofService().predict_bytes(after_bytes, after_filename)
    legacy_before_beautyproof = None
    legacy_after_beautyproof = None
    if before_beautyproof.get("visual_evidence") is None:
        legacy_before_beautyproof = BeautyProofV2Service().predict_bytes(before_bytes, before_filename)
    if after_beautyproof.get("visual_evidence") is None:
        legacy_after_beautyproof = BeautyProofV2Service().predict_bytes(after_bytes, after_filename)
    after_visual_evidence = after_beautyproof.get("visual_evidence")
    if after_visual_evidence is None and legacy_after_beautyproof is not None:
        after_visual_evidence = legacy_after_beautyproof.get("visual_evidence")
    comparison = compare_images_basic(before_bytes, after_bytes)
    reliability_penalty = {"High": 0.18, "Medium": 0.38, "Low": 0.6}[comparison["comparison_reliability"]]
    ai_risk = max(float(before_model["probability_ai"]), float(after_model["probability_ai"]))
    beautyproof_score = (
        float(after_visual_evidence["integrity_score"])
        if after_visual_evidence
        else float(comparison["after_retouch"]["retouch_score"])
    )
    confidence = _clamp(0.38 * beautyproof_score + 0.26 * ai_risk + 0.36 * reliability_penalty)
    evidence = list(comparison["agent_should_notice"])
    if after_visual_evidence:
        evidence.append(
            f"After 图 BeautyProof 修图概率 {after_visual_evidence['retouch_probability']:.1%}，"
            f"模型状态 {after_visual_evidence['model_status']}。"
        )
        if after_visual_evidence.get("model_version") == "BeautyProof-Unified":
            evidence.append(
                f"After 图 Unified 输出：strength={after_visual_evidence.get('retouch_strength', 'none')}，"
                f"region_status={after_visual_evidence.get('region_status', 'N/A')}。"
            )
    else:
        evidence.append(
            f"After 图 BeautyProof Unified 暂不可用：{after_beautyproof.get('error', {}).get('message', 'unknown error')}"
        )
    return {
        "mode": "before_after",
        "verdict": "High risk" if confidence >= 0.62 else "Medium risk" if confidence >= 0.42 else "Low risk",
        "false_advertising_confidence": _rounded(confidence),
        "groundtruth_label_suggestion": "CONFOUNDED" if confidence >= 0.42 else "SUPPORTED",
        "before_beautyproof_unified": before_beautyproof,
        "after_beautyproof_unified": after_beautyproof,
        "before_beautyproof_v2": legacy_before_beautyproof,
        "after_beautyproof_v2": legacy_after_beautyproof,
        "visual_evidence": after_visual_evidence,
        "before_model_output": before_model,
        "after_model_output": after_model,
        "before_after_evidence": comparison,
        "evidence": evidence,
    }
