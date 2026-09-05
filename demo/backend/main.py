from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .beautyproof_v2 import CHECKPOINT_CANDIDATES, HEATMAP_ROOT, BeautyProofV2Service
from .detector_service import build_before_after_analysis, build_single_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
VENDOR_DIR = PROJECT_ROOT / "vendor" / "ai-image-detector"

app = FastAPI(title="Mirror of Truth Part B Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
HEATMAP_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/heatmaps", StaticFiles(directory=str(HEATMAP_ROOT)), name="heatmaps")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    checkpoint_paths = [str(path) for path in CHECKPOINT_CANDIDATES]
    return {
        "status": "ok",
        "app": "Mirror of Truth Part B Demo",
        "beautyproof_v2": {
            "checkpoint_present": any(path.exists() for path in CHECKPOINT_CANDIDATES),
            "checkpoint_candidates": checkpoint_paths,
            "real_model_env": "set BEAUTYPROOF_USE_REAL=1 to load BeautyProof V2 checkpoint",
            "mock_mode": "default until BEAUTYPROOF_USE_REAL=1",
        },
        "ai_detector_repo_present": VENDOR_DIR.exists(),
        "real_detector_env": "set MIRROR_USE_REAL_AIDETECTOR=1 to load lynote-ai/ai-image-detector",
        "supported_modes": ["beautyproof_v2", "makeup_single", "fashion_single", "before_after"],
    }


@app.post("/api/beautyproof/v2/detect")
async def beautyproof_v2_detect(
    image: UploadFile = File(...),
) -> dict:
    image_bytes = await image.read()
    return BeautyProofV2Service().predict_bytes(image_bytes, image.filename)


@app.post("/api/analyze/single")
async def analyze_single(
    image: UploadFile = File(...),
    mode: str = Form("makeup"),
    backend: str = Form("ultra"),
) -> dict:
    image_bytes = await image.read()
    clean_mode = "fashion" if mode == "fashion" else "makeup"
    return build_single_analysis(image_bytes, image.filename, clean_mode, backend)


@app.post("/api/analyze/before-after")
async def analyze_before_after(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    backend: str = Form("ultra"),
) -> dict:
    before_bytes = await before_image.read()
    after_bytes = await after_image.read()
    return build_before_after_analysis(
        before_bytes,
        after_bytes,
        before_image.filename,
        after_image.filename,
        backend,
    )
