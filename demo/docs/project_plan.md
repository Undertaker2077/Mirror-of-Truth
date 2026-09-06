# Mirror of Truth Demo Plan

## Goal

Build a Part B demo with three user-facing modes:

1. Makeup single-image check: AI generation plus over-beautification risk.
2. Fashion seller-image check: AI generation and over-editing risk.
3. Before/after makeup comparison: compare beauty intensity and local retouching clues.

## BeautyProof V2 Integration

BeautyProof V2 is the primary retouch detector for the MVP. The backend contract follows
the supplied interface document:

- model architecture: EfficientNet-B0 with one sigmoid output head
- labels: `no_retouch` / `digital_retouch`
- threshold: `0.5`
- output: top-level `visual_evidence`
- checkpoint lookup order:
  - `/Users/maoyiqi/Downloads/Mirror-of-Truth-demo/best_model.pt`
  - `/Users/maoyiqi/Downloads/Mirror-of-Truth-demo/models/retouch_detector_v2/best_model.pt`

Until the checkpoint is uploaded, the service returns a clearly marked
`model_status: mock_pending_checkpoint` payload so frontend/backend integration can be tested
without pretending that real inference has already run.

## AI Detector Integration

The default AI detector is the HuggingFace AI/Deepfake/Real classifier exposed through
`backend=hf3`. It returns `probability_ai` plus `probability_artificial`,
`probability_deepfake`, and `probability_real`. The vendored
`lynote-ai/ai-image-detector` project remains under `vendor/ai-image-detector` only as
the `backend=ultra` fallback. For development smoke tests, failed model loading returns
a deterministic mock response so frontend/backend wiring can still be verified.

## API

- `GET /api/health`
- `POST /api/beautyproof/v2/detect`
  - form fields: `image`
  - returns: `{ "visual_evidence": ... }`
- `POST /api/analyze/single`
  - form fields: `image`, `mode=makeup|fashion`, `backend=hf3|ultra`
- `POST /api/analyze/before-after`
  - form fields: `before_image`, `after_image`, `backend=hf3|ultra`

## Output Contract

The frontend renders:

- `false_advertising_confidence`
- `visual_evidence.integrity_score`
- `visual_evidence.reliability`
- `visual_evidence.manipulation_map_url`
- risk verdict
- model label and `probability_ai`
- retouch signals
- `before_after_evidence` JSON for A-side consumption

## Real Model Setup

Install dependencies and start the demo:

```bash
cd /Users/maoyiqi/Downloads/Mirror-of-Truth-demo
python3 -m pip install -r requirements.txt
BEAUTYPROOF_USE_UNIFIED=1 BEAUTYPROOF_API_PATH=/Users/maoyiqi/Downloads/Mirror-of-Truth python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```
