# BeautyProof Unified API Reference

## Decision contract

V2 is authoritative for `retouched`. Type and region models only enrich a positive V2 result. If V2 is positive and no supported region is found, the API returns `region_status: not_localized`.

Supported effect types are `smoothing` and `whitening`. Supported regions are `forehead`, `left_cheek`, `right_cheek`, `nose`, `chin`, and `full_face`. Slimming and jawline predictions are filtered before serialization.

## Python API

```python
from beautyproof_api import UnifiedBeautyProofAPI

api = UnifiedBeautyProofAPI(retouch_threshold=0.5, type_threshold=0.5)
result = api.analyze("face.jpg")
```

Models load lazily and are cached. CUDA device 0 is used when available; otherwise inference falls back to CPU.

## REST API

```bash
uvicorn beautyproof_api.server:app --host 0.0.0.0 --port 8000
```

### `GET /health`

Returns service availability. It is not a model-quality check.

### `POST /v1/analyze`

Content type: `multipart/form-data`; required field `image` is a JPEG or PNG face image.

```bash
curl -X POST http://localhost:8000/v1/analyze -F "image=@face.jpg"
```

### Response schema

```json
{
  "schema_version": "1.0",
  "retouched": true,
  "retouch_probability": 0.923114,
  "retouch_types": [{"name": "smoothing", "probability": 0.817331}],
  "retouch_strength": "medium",
  "modified_regions": [{
    "name": "left_cheek",
    "confidence": 0.741203,
    "polygon": [[0.21, 0.43], [0.46, 0.51], [0.38, 0.72]]
  }],
  "region_status": "localized",
  "models": {
    "binary_detector": "BeautyProof-V2",
    "type_classifier": "Retouch-Multitask-CNN-V1",
    "region_segmenter": "YOLO11n-Seg-30e"
  },
  "limitations": []
}
```

Polygon coordinates are normalized to `[0,1]` relative to input width and height.

`region_status` values:

- `localized`: at least one supported region was returned.
- `not_localized`: V2 is positive but localization is below threshold.
- `not_applicable`: V2 is negative and downstream claims are suppressed.

## CLI

```bash
python -m beautyproof_api.cli INPUT_IMAGE [--output result.json]
```

## Errors and integration rules

- Missing Python path: `FileNotFoundError`.
- Empty REST upload: HTTP 400.
- Unreadable image: HTTP 422.
- Uploads larger than 10 MiB: HTTP 413.
- Missing/incompatible checkpoint or hash mismatch: HTTP 503; verify model paths and hashes.
- Never infer “clean” from an empty `modified_regions` array.
- Do not convert smoothing/whitening probabilities into product-efficacy conclusions.
