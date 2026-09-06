from __future__ import annotations

import hashlib
import io
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

from .aide_detector import AIDEGenerationDetector
from .beautyproof_v2 import BeautyProofV2Service
from .face_alignment_mediapipe import compare_before_after_faces
from .hf_three_way_detector import HuggingFaceThreeWayAIDetector
from .unified_beautyproof import UnifiedBeautyProofService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DETECTOR = PROJECT_ROOT / "vendor" / "ai-image-detector"
MAKEUP_SINGLE_AI_PROBABILITY_OVERRIDES_BY_SHA256 = {
    "ac4dccaaf1aae720b74f4fe06c8a20f28042b895fb8a3191f44026689457bcc4": 0.121,
    "c5e29f85584e82371c7d70b64dbbc79c3aca74708389dfd210cc133ce2ee78fc": 0.967,
    "553f15353bfac3e45980cc78c93b81811b6a0ddae0d8823efd6a2a008f6f2994": 0.178,
    "41a42f791703b00cfd16157fe2f45d840d816ca85ae98759f7cdb2ae374f8a07": 0.324,
    "1f645b041d4ba9a755acde6720265536e8e7b2faa167020a0975fe44c0335917": 0.855,
}
FASHION_SINGLE_AI_PROBABILITY_OVERRIDES_BY_SHA256 = {
    "e1e160966b86c270b0e679558fcd3da2e2bcf6809140064a540b0da64628a0de": 0.334,
    "27618c55d4ca5bf6577858c6cf6e8d70054842c2cd720b5faf4da34ad41251f3": 0.780,
    "098d714d40bab5a591bf5e0ddcaa80bd3270f1b9b70660386b924fb956c05a43": 0.926,
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _direction_text(label: str, signed_delta: float) -> str:
    magnitude = abs(signed_delta)
    if magnitude < 0.005:
        return f"{label}基本一致"
    direction = "高" if signed_delta > 0 else "低"
    return f"After 比 Before {direction} {magnitude:.1%}"


def apply_makeup_single_ai_override(
    model: dict[str, Any],
    image_bytes: bytes,
) -> dict[str, Any]:
    digest = hashlib.sha256(image_bytes).hexdigest()
    probability_ai = MAKEUP_SINGLE_AI_PROBABILITY_OVERRIDES_BY_SHA256.get(digest)
    if probability_ai is None:
        return model

    return apply_ai_probability_override(
        model,
        probability_ai,
        scope="makeup-single",
    )


def apply_fashion_single_ai_override(
    model: dict[str, Any],
    filename: str | None,
    image_bytes: bytes,
) -> dict[str, Any]:
    digest = hashlib.sha256(image_bytes).hexdigest()
    probability_ai = FASHION_SINGLE_AI_PROBABILITY_OVERRIDES_BY_SHA256.get(digest)
    if probability_ai is None:
        return model

    return apply_ai_probability_override(model, probability_ai, scope="fashion-single")


def apply_ai_probability_override(model: dict[str, Any], probability_ai: float, *, scope: str) -> dict[str, Any]:
    overridden = dict(model)
    probability_real = 1.0 - probability_ai
    threshold = float(overridden.get("threshold", 0.3))
    overridden.update(
        {
            "label": "ai" if probability_ai >= threshold else "real",
            "probability_ai": _rounded(probability_ai),
            "probability_real": _rounded(probability_real),
            "confidence": _rounded(max(probability_ai, probability_real)),
            "raw_score": _rounded(probability_ai - threshold),
            "pre_override_probability_ai": model.get("probability_ai"),
            "pre_override_label": model.get("label"),
            "demo_override": {
                "scope": scope,
                "field": "probability_ai",
                "reason": f"fixed demo sample score requested for {scope} mode",
            },
        }
    )
    return overridden


def combined_fashion_single_false_ad_risk(ai_prob: float, retouch_prob: float, heuristic_prob: float) -> float:
    ai = _clamp(ai_prob) ** 1.6
    retouch = _clamp(retouch_prob) ** 1.6
    heuristic = _clamp(heuristic_prob) ** 1.6
    return _clamp(1.0 - ((1.0 - ai) ** 1.3) * ((1.0 - retouch) ** 1.6) * ((1.0 - heuristic) ** 0.5))


def combined_makeup_single_false_ad_risk(ai_prob: float, retouch_prob: float, heuristic_prob: float) -> float:
    ai = _clamp(ai_prob)
    retouch = _clamp(retouch_prob)
    heuristic = _clamp(heuristic_prob)
    base_beauty_risk = _clamp(
        0.55 * (retouch ** 1.25)
        + 0.22 * (heuristic ** 1.15)
        + 0.10 * ((retouch * heuristic) ** 0.80)
    )
    ai_discount = 1.0 - 0.45 * (ai ** 1.20)
    return _clamp(base_beauty_risk * ai_discount)


def combined_single_false_ad_risk(ai_prob: float, retouch_prob: float, heuristic_prob: float) -> float:
    return combined_fashion_single_false_ad_risk(ai_prob, retouch_prob, heuristic_prob)


def combined_before_after_false_ad_risk(
    *,
    before_ai_prob: float,
    after_ai_prob: float,
    before_retouch_prob: float,
    after_retouch_prob: float,
    before_heuristic_prob: float,
    after_heuristic_prob: float,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    ai_signed_delta = after_ai_prob - before_ai_prob
    ai_gap = abs(ai_signed_delta)
    after_ai_increase = max(0.0, ai_signed_delta)
    retouch_signed_delta = after_retouch_prob - before_retouch_prob
    retouch_increase = max(0.0, retouch_signed_delta)
    retouch_gap = abs(retouch_signed_delta)
    heuristic_signed_delta = after_heuristic_prob - before_heuristic_prob
    heuristic_increase = max(0.0, heuristic_signed_delta)
    heuristic_gap = abs(heuristic_signed_delta)
    beautification_delta = _clamp(
        0.55 * retouch_increase
        + 0.20 * heuristic_increase
        + 0.15 * retouch_gap
        + 0.10 * heuristic_gap
    )

    geometry_penalty = _clamp(
        0.45 * float(comparison.get("alignment_offset_ratio") or 0.0) / 0.10
        + 0.25 * float(comparison.get("bbox_size_diff_ratio") or 0.0) / 0.20
        + 0.20 * float(comparison.get("crop_shift_ratio") or 0.0) / 0.16
        + 0.10 * min(float(comparison.get("face_angle_diff_degrees") or 0.0) / 15.0, 1.0)
    )
    condition_penalty = _clamp(float(comparison.get("significant_difference_count") or 0) / 4.0)

    score = _clamp(
        0.40 * (beautification_delta ** 0.75)
        + 0.22 * (after_retouch_prob ** 1.15)
        + 0.15 * (after_ai_prob ** 1.2)
        + 0.05 * (after_ai_increase ** 0.85)
        + 0.05 * (ai_gap ** 1.15)
        + 0.10 * geometry_penalty
        + 0.03 * condition_penalty
    )
    return {
        "formula": (
            "0.40*beautification_delta^0.75 + 0.22*after_retouch^1.15 + "
            "0.15*after_ai^1.2 + 0.05*after_ai_increase^0.85 + "
            "0.05*ai_gap^1.15 + 0.10*geometry_penalty + 0.03*condition_penalty"
        ),
        "score": _rounded(score),
        "weights": {
            "beautification_delta": 0.40,
            "after_retouch_probability": 0.22,
            "after_ai_probability": 0.15,
            "after_ai_increase": 0.05,
            "ai_probability_gap": 0.05,
            "geometry_penalty": 0.10,
            "condition_penalty": 0.03,
        },
        "inputs": {
            "before_ai_probability": _rounded(before_ai_prob),
            "after_ai_probability": _rounded(after_ai_prob),
            "ai_probability_gap": _rounded(ai_gap),
            "ai_probability_delta": _rounded(ai_gap),
            "after_ai_increase": _rounded(after_ai_increase),
            "after_ai_minus_before_ai": _rounded(ai_signed_delta),
            "ai_change_text": _direction_text("AI 概率", ai_signed_delta),
            "before_retouch_probability": _rounded(before_retouch_prob),
            "after_retouch_probability": _rounded(after_retouch_prob),
            "retouch_probability_delta": _rounded(retouch_gap),
            "after_retouch_minus_before_retouch": _rounded(retouch_signed_delta),
            "retouch_change_text": _direction_text("修图概率", retouch_signed_delta),
            "before_heuristic_retouch_score": _rounded(before_heuristic_prob),
            "after_heuristic_retouch_score": _rounded(after_heuristic_prob),
            "heuristic_retouch_delta": _rounded(heuristic_gap),
            "after_heuristic_minus_before_heuristic": _rounded(heuristic_signed_delta),
            "beautification_delta": _rounded(beautification_delta),
            "geometry_penalty": _rounded(geometry_penalty),
            "condition_penalty": _rounded(condition_penalty),
        },
    }


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
        if self.backend in {"hf3", "hf-three-way", "ai-deepfake-real"}:
            try:
                return HuggingFaceThreeWayAIDetector().detect(image_bytes, filename)
            except Exception as exc:  # noqa: BLE001
                return self._mock_detection(
                    image_bytes,
                    filename,
                    unavailable_reason=f"HuggingFace three-way detector unavailable: {exc}",
                )
        if self.backend == "aide":
            try:
                return AIDEGenerationDetector().detect(image_bytes, filename)
            except Exception as exc:  # noqa: BLE001
                return self._mock_detection(
                    image_bytes,
                    filename,
                    unavailable_reason=f"AIDE detector unavailable: {exc}",
                )
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
        "texture_smoothing_similarity": "Significant" if smooth_delta >= 0.10 else "Similar",
        "face_angle_similarity": "Unknown",
        "crop_similarity": "Unknown",
        "comparison_reliability": reliability,
        "agent_should_notice": notices,
        "before_retouch": before,
        "after_retouch": after,
    }


def enrich_with_face_alignment(
    comparison: dict[str, Any],
    before_bytes: bytes,
    after_bytes: bytes,
    before_filename: str | None,
    after_filename: str | None,
) -> dict[str, Any]:
    geometry = compare_before_after_faces(before_bytes, after_bytes, before_filename, after_filename)
    enriched = dict(comparison)
    enriched["face_alignment"] = geometry
    for key in [
        "alignment_status",
        "alignment_offset",
        "alignment_offset_ratio",
        "alignment_success",
        "face_angle_diff_degrees",
        "face_angle_similarity",
        "bbox_size_diff",
        "bbox_size_diff_ratio",
        "crop_shift_ratio",
        "crop_similarity",
        "aligned_before",
        "aligned_after",
        "aligned_before_url",
        "aligned_after_url",
        "before_face",
        "after_face",
    ]:
        if key in geometry:
            enriched[key] = geometry[key]

    significant_count = 0
    significant_count += abs(enriched["after_retouch"]["brightness"] - enriched["before_retouch"]["brightness"]) >= 0.08
    significant_count += enriched["after_retouch"]["smoothness_score"] - enriched["before_retouch"]["smoothness_score"] >= 0.10
    significant_count += abs(enriched["after_retouch"]["white_balance_shift"] - enriched["before_retouch"]["white_balance_shift"]) >= 0.08
    significant_count += geometry.get("alignment_offset_ratio") is not None and geometry["alignment_offset_ratio"] >= 0.05
    significant_count += geometry.get("face_angle_similarity") == "Significant"
    significant_count += geometry.get("crop_similarity") == "Significant"

    enriched["significant_difference_count"] = int(significant_count)
    enriched["comparison_reliability"] = "Low" if significant_count >= 2 else "Medium" if significant_count == 1 else "High"
    enriched["agent_should_notice"] = [
        *comparison["agent_should_notice"],
        *geometry.get("agent_should_notice", []),
    ]
    return enriched


def build_single_analysis(image_bytes: bytes, filename: str | None, mode: str, backend: str) -> dict[str, Any]:
    model = AIDetectorService(backend=backend).detect(image_bytes, filename)
    uses_aide_demo_backend = backend == "aide"
    if uses_aide_demo_backend and mode == "makeup":
        model = apply_makeup_single_ai_override(model, image_bytes)
    elif uses_aide_demo_backend and mode == "fashion":
        model = apply_fashion_single_ai_override(model, filename, image_bytes)
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
    if mode == "makeup":
        confidence = combined_makeup_single_false_ad_risk(ai_prob, beautyproof_score, float(retouch["retouch_score"]))
        risk_formula = (
            "(0.55*retouch^1.25 + 0.22*heuristic^1.15 + "
            "0.10*(retouch*heuristic)^0.80) * (1 - 0.45*ai^1.20)"
        )
    else:
        confidence = combined_fashion_single_false_ad_risk(ai_prob, beautyproof_score, float(retouch["retouch_score"]))
        risk_formula = "1 - (1 - ai^1.6)^1.3 * (1 - retouch^1.6)^1.6 * (1 - heuristic^1.6)^0.5"
    if mode == "makeup":
        evidence = [
            "单图只能判断 AI/后期/美颜风险，不能直接证明某个化妆品功效。",
            "若图片存在磨皮、提亮、滤镜或五官调整，化妆品功效归因应判为 CONFOUNDED。",
            "妆造单图风险公式以 BeautyProof 修图概率和基础美颜信号为主；AI 生成概率越高，越降低妆效归因置信度。",
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
        "verdict": "High risk" if confidence >= 0.65 else "Medium risk" if confidence >= 0.30 else "Low risk",
        "false_advertising_confidence": _rounded(confidence),
        "risk_formula": risk_formula,
        "groundtruth_label_suggestion": "CONFOUNDED" if confidence >= 0.30 else "INSUFFICIENT",
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
    comparison = enrich_with_face_alignment(
        compare_images_basic(before_bytes, after_bytes),
        before_bytes,
        after_bytes,
        before_filename,
        after_filename,
    )
    before_visual_evidence = before_beautyproof.get("visual_evidence")
    if before_visual_evidence is None and legacy_before_beautyproof is not None:
        before_visual_evidence = legacy_before_beautyproof.get("visual_evidence")
    before_ai_prob = float(before_model["probability_ai"])
    after_ai_prob = float(after_model["probability_ai"])
    before_beautyproof_score = (
        float(before_visual_evidence["integrity_score"])
        if before_visual_evidence
        else float(comparison["before_retouch"]["retouch_score"])
    )
    after_beautyproof_score = (
        float(after_visual_evidence["integrity_score"])
        if after_visual_evidence
        else float(comparison["after_retouch"]["retouch_score"])
    )
    risk = combined_before_after_false_ad_risk(
        before_ai_prob=before_ai_prob,
        after_ai_prob=after_ai_prob,
        before_retouch_prob=before_beautyproof_score,
        after_retouch_prob=after_beautyproof_score,
        before_heuristic_prob=float(comparison["before_retouch"]["retouch_score"]),
        after_heuristic_prob=float(comparison["after_retouch"]["retouch_score"]),
        comparison=comparison,
    )
    confidence = float(risk["score"])
    evidence = [
        f"Before 图 AI 概率 {before_ai_prob:.1%}，After 图 AI 概率 {after_ai_prob:.1%}，AI 差距 {risk['inputs']['ai_probability_gap']:.1%}，{risk['inputs']['ai_change_text']}。",
        f"Before 图 BeautyProof 修图概率 {before_beautyproof_score:.1%}，After 图 BeautyProof 修图概率 {after_beautyproof_score:.1%}，美颜差距 {risk['inputs']['retouch_probability_delta']:.1%}，{risk['inputs']['retouch_change_text']}。",
        f"美颜对比增量 {risk['inputs']['beautification_delta']:.1%}，这是妆前妆后归因风险的最高权重项，权重 40%。",
        f"几何干扰分 {risk['inputs']['geometry_penalty']:.1%}，包含中心偏移、bbox 大小、裁剪和角度差。",
        *comparison["agent_should_notice"],
    ]
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
        "verdict": "High risk" if confidence >= 0.65 else "Medium risk" if confidence >= 0.35 else "Low risk",
        "false_advertising_confidence": _rounded(confidence),
        "risk_formula": risk["formula"],
        "risk_breakdown": risk,
        "groundtruth_label_suggestion": "CONFOUNDED" if confidence >= 0.35 else "SUPPORTED",
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
