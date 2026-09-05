from pathlib import Path

import numpy as np
from PIL import Image

from heatmap_generator import generate_heatmap_overlay, map_concentration


def test_heatmap_overlay_preserves_original_dimensions(tmp_path: Path):
    source = tmp_path / "输入.png"
    output = tmp_path / "输出" / "热力图.png"
    Image.new("RGB", (320, 180), "gray").save(source)
    manipulation_map = np.zeros((7, 7), dtype=np.float32)
    manipulation_map[3, 3] = 1.0

    generated = generate_heatmap_overlay(source, manipulation_map, output)

    assert generated == output
    assert Image.open(output).size == (320, 180)


def test_map_concentration_is_higher_for_localized_evidence():
    localized = np.zeros((10, 10), dtype=np.float32)
    localized[4:6, 4:6] = 1.0
    diffuse = np.ones((10, 10), dtype=np.float32)
    assert map_concentration(localized) > map_concentration(diffuse)

