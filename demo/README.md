# Mirror of Truth Demo

Part B frontend/backend demo for visual false-advertising risk detection.
The primary MVP model route is BeautyProof V2.

## Run

```bash
cd /Users/maoyiqi/Downloads/Mirror-of-Truth-demo
python3 -m pip install -r requirements.txt
npm install --ignore-scripts
npm run build
MIRROR_USE_REAL_AIDETECTOR=1 /Users/maoyiqi/.pyenv/versions/3.13.0/bin/python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

## Modes

- Makeup single image: AI-generated risk plus over-beautification signals.
- Fashion single image: AI-generated seller-image risk.
- Before/after makeup: exposure, white balance, smoothing, and retouching differences.

## Frontend

The frontend is a Vue 3 + Vite app in `frontend/`.

Development server:

```bash
npm run dev
```

Production build served by FastAPI:

```bash
npm run build
MIRROR_USE_REAL_AIDETECTOR=1 /Users/maoyiqi/.pyenv/versions/3.13.0/bin/python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

## BeautyProof V2 Model

Put the checkpoint in one of these locations:

```text
/Users/maoyiqi/Downloads/Mirror-of-Truth-demo/best_model.pt
/Users/maoyiqi/Downloads/Mirror-of-Truth-demo/models/retouch_detector_v2/best_model.pt
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

The vendored detector is in:

```text
vendor/ai-image-detector
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Start with real model loading:

```bash
MIRROR_USE_REAL_AIDETECTOR=1 python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

First real-model use may download Hugging Face weights and can take several minutes.

## Smoke Test

```bash
python3 -m unittest discover -s tests
curl -s http://127.0.0.1:8765/api/health
curl -s -F image=@vendor/ai-image-detector/test_images/ai_retouched.png http://127.0.0.1:8765/api/beautyproof/v2/detect
curl -s -F image=@vendor/ai-image-detector/test_images/ai-generated.png -F mode=makeup -F backend=ultra http://127.0.0.1:8765/api/analyze/single
curl -s -F before_image=@vendor/ai-image-detector/test_images/human.jpeg -F after_image=@vendor/ai-image-detector/test_images/ai_retouched.png -F backend=ultra http://127.0.0.1:8765/api/analyze/before-after
```
