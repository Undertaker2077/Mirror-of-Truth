# Mirror of Truth — BeautyProof Unified API

This repository exposes one API over three complementary models: BeautyProof V2 for the authoritative binary retouch decision, Retouch Multitask CNN V1 for smoothing/whitening, and YOLO11n-Seg Regions V1 for facial-region localization.

> Slimming/jawline, makeup detection, and product-efficacy claims are intentionally not exposed. Type and region models were validated primarily on controlled synthetic edits.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

```python
from beautyproof_api import UnifiedBeautyProofAPI
result = UnifiedBeautyProofAPI().analyze("face.jpg")
```

```bash
python -m beautyproof_api.cli face.jpg --output result.json
uvicorn beautyproof_api.server:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/v1/analyze -F "image=@face.jpg"
```

See [Unified API documentation](docs/UNIFIED_BEAUTYPROOF_API.md) and [model card](docs/BEAUTYPROOF_UNIFIED_MODEL_CARD.md).

## Model files

| Component | Path |
|---|---|
| Binary V2 | `models/retouch_detector_v2/best_model.pt` |
| Type CNN | `models/retouch_multitask_cnn_v1/best_model.pt` |
| Region YOLO-Seg | `models/yolo_retouch_regions_v1/best.pt` |

## Tests

```bash
pytest tests/test_unified_api.py -q
```
