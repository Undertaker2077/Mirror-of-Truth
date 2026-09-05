"""Stable JSON contract consumed by the upstream A service."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VisualEvidence:
    integrity_score: float
    reliability: str
    metrics: dict[str, Any]
    manipulation_map: str | None
    reliability_details: dict[str, Any] = field(default_factory=dict)
    model_version: str = "BeautyProof-V2"
    threshold: float = 0.5
    limitations: list[str] = field(
        default_factory=lambda: [
            "Business metrics are proxies from one binary score.",
            "Grad-CAM is model attention, not pixel-level forensic ground truth.",
            "Do not use as the sole basis for legal, medical, hiring, or identity decisions.",
        ]
    )

    def __post_init__(self) -> None:
        self.integrity_score = float(self.integrity_score)
        if not 0.0 <= self.integrity_score <= 1.0:
            raise ValueError("integrity_score must be between 0 and 1")
        if self.reliability not in {"Low", "Medium", "High"}:
            raise ValueError("reliability must be Low, Medium, or High")

    def to_dict(self) -> dict[str, Any]:
        return {"visual_evidence": asdict(self)}

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

