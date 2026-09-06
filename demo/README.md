# Mirror of Truth Demo

Part B frontend/backend demo for visual false-advertising risk detection.
The MVP route combines HuggingFace AI/Deepfake/Real detection, BeautyProof
Unified retouch detection, and MediaPipe before/after face alignment.

## Run

```bash
cd Mirror-of-Truth-demo
python3 -m pip install -r requirements.txt
npm install --ignore-scripts
npm run build
BEAUTYPROOF_USE_UNIFIED=1 BEAUTYPROOF_API_PATH=.. python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

## Modes

- Makeup single image: AI-generated risk plus over-beautification signals.
- Fashion single image: AI-generated seller-image risk.
- Before/after makeup: exposure, white balance, smoothing, and retouching differences.

## Before/After Face Alignment

The before/after route returns `before_after_evidence` with:

```text
alignment_offset
face_angle_diff_degrees
face_angle_similarity
bbox_size_diff
crop_similarity
comparison_reliability
aligned_before_url
aligned_after_url
```

The demo uses a real MediaPipe FaceLandmarker worker for B1/B4 and writes
aligned preview images under `/aligned`. The worker runs in a Python 3.11
virtual environment so the main FastAPI process is protected from the native
MediaPipe crash seen in Python 3.13.

Create the worker environment:

```bash
python3.11 -m venv .venv-mediapipe
.venv-mediapipe/bin/python -m pip install --upgrade pip
.venv-mediapipe/bin/python -m pip install mediapipe==0.10.35 pillow numpy
```

Download the FaceLandmarker task model:

```text
mkdir -p models
curl -L --fail --output models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

If the worker or task model lives elsewhere, set:

```bash
export MEDIAPIPE_PYTHON=/path/to/python-with-mediapipe
export MEDIAPIPE_FACE_LANDMARKER_PATH=/path/to/face_landmarker.task
```

The OpenCV face-box fallback is only for local debugging and must be enabled
explicitly:

```bash
ALLOW_FACE_ALIGNMENT_FALLBACK=1 python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

## Frontend

The frontend is a Vue 3 + Vite app in `frontend/`.

Development server:

```bash
npm run dev
```

Production build served by FastAPI:

```bash
npm run build
BEAUTYPROOF_USE_UNIFIED=1 BEAUTYPROOF_API_PATH=.. python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

## Model Assets

The BeautyProof model weights are stored in the repository under `models/`.
The optional MediaPipe FaceLandmarker task file should be placed at:

```text
Mirror-of-Truth/
  demo/
    models/
      face_landmarker.task
```

If `demo/models/face_landmarker.task` is missing, download it with the command
shown in the face alignment section above.

## BeautyProof Unified Model

The demo can call your friend's unified package:

```python
from beautyproof_api import UnifiedBeautyProofAPI

result = UnifiedBeautyProofAPI(retouch_threshold=0.5, type_threshold=0.5).analyze("face.jpg")
```

Expected package and checkpoint layout:

```text
Mirror-of-Truth/
  beautyproof_api/
  models/
    retouch_detector_v2/best_model.pt
    retouch_three_type_v1/best_model.pt
    yolo_retouch_regions_v1/best.pt
  demo/
```

If this demo is not inside the `Mirror-of-Truth` repo root, point it at the repo:

```bash
export BEAUTYPROOF_API_PATH=/path/to/Mirror-of-Truth
```

Start with unified model loading enabled:

```bash
BEAUTYPROOF_USE_UNIFIED=1 BEAUTYPROOF_API_PATH=.. python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

Direct unified routes:

```bash
curl -s -F image=@example.jpg http://127.0.0.1:8765/api/beautyproof/unified/analyze
curl -s -F image=@example.jpg http://127.0.0.1:8765/v1/analyze
```

The response follows the unified contract: `retouched`,
`retouch_probability`, `retouch_types`, `retouch_strength`,
`modified_regions`, `region_status`, `models`, and `limitations`.

## BeautyProof V2 Fallback

Put the checkpoint in one of these locations:

```text
Mirror-of-Truth-demo/best_model.pt
Mirror-of-Truth-demo/models/retouch_detector_v2/best_model.pt
```

The backend verifies the expected SHA256:

```text
52F38353CEB4F20325B8AF84C0A0973FD48FEB57323B4429465D5C10FCFDC94D
```

Direct model route:

```bash
curl -s -F image=@example.jpg http://127.0.0.1:8765/api/beautyproof/v2/detect
```

Response contains top-level `visual_evidence` with `integrity_score`,
`retouch_probability`, `reliability`, proxy metrics, and `manipulation_map_url`.
By default the demo keeps BeautyProof in `mock_pending_checkpoint` mode so the AI
generation detector can be tested independently. Enable real BeautyProof inference with:

```bash
BEAUTYPROOF_USE_REAL=1 python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

## Real AI Detector

The default AI-generation detector is:

```text
prithivMLmods/AI-vs-Deepfake-vs-Real
```

Use it with:

```text
backend=hf3
```

It returns the binary-compatible `probability_ai` field plus three-way details:
`raw_label`, `probability_artificial`, `probability_deepfake`, and
`probability_real`.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Start the demo:

```bash
BEAUTYPROOF_USE_UNIFIED=1 BEAUTYPROOF_API_PATH=.. python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

Frontend defaults to `hf3`. The older `vendor/ai-image-detector` integration
remains available only as a fallback by passing `backend=ultra` with
`MIRROR_USE_REAL_AIDETECTOR=1`; otherwise it returns the deterministic smoke
fallback.

## Smoke Test

```bash
python3 -m unittest discover -s tests
curl -s http://127.0.0.1:8765/api/health
curl -s -F image=@vendor/ai-image-detector/test_images/ai_retouched.png http://127.0.0.1:8765/api/beautyproof/v2/detect
curl -s -F image=@vendor/ai-image-detector/test_images/ai-generated.png -F mode=makeup -F backend=hf3 http://127.0.0.1:8765/api/analyze/single
curl -s -F before_image=@vendor/ai-image-detector/test_images/human.jpeg -F after_image=@vendor/ai-image-detector/test_images/ai_retouched.png -F backend=hf3 http://127.0.0.1:8765/api/analyze/before-after
```
