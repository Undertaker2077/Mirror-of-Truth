from pathlib import Path

import pytest
import torch
from PIL import Image

from model_inference import predict_raw


class ConstantModel(torch.nn.Module):
    def __init__(self, logit: float):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(logit))

    def forward(self, batch):
        return self.logit.expand(batch.shape[0], 1)


def test_predict_raw_returns_score_and_normalized_map(tmp_path: Path):
    image = tmp_path / "face.png"
    Image.new("RGB", (128, 96), "gray").save(image)
    result = predict_raw(image, model=ConstantModel(0.0), device="cpu")
    assert result["score"] == pytest.approx(0.5)
    assert result["manipulation_map"].shape == (96, 128)
    assert 0.0 <= float(result["manipulation_map"].min())
    assert float(result["manipulation_map"].max()) <= 1.0


def test_predict_raw_rejects_missing_image():
    with pytest.raises(FileNotFoundError):
        predict_raw("missing.png", model=ConstantModel(0.0), device="cpu")

