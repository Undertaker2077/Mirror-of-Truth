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

The vendored `lynote-ai/ai-image-detector` project is kept under `vendor/ai-image-detector`.
The backend adapter uses its documented Python API when `MIRROR_USE_REAL_AIDETECTOR=1`.
For development smoke tests, the adapter falls back to a deterministic mock response so
frontend/backend wiring can be verified without waiting for model dependencies or remote weights.

## API

- `GET /api/health`
- `POST /api/beautyproof/v2/detect`
  - form fields: `image`
  - returns: `{ "visual_evidence": ... }`
- `POST /api/analyze/single`
  - form fields: `image`, `mode=makeup|fashion`, `backend=ultra`
- `POST /api/analyze/before-after`
  - form fields: `before_image`, `after_image`, `backend=ultra`

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

Install the detector package if real inference is needed:

```bash
cd /Users/maoyiqi/Downloads/Mirror-of-Truth-demo
python3 -m pip install -e 'vendor/ai-image-detector[api]'
MIRROR_USE_REAL_AIDETECTOR=1 python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```
