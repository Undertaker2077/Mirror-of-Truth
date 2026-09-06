from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .aide_detector import aide_status
from .beautyproof_v2 import CHECKPOINT_CANDIDATES, HEATMAP_ROOT, BeautyProofV2Service
from .detector_service import build_before_after_analysis, build_single_analysis
from .face_alignment_mediapipe import ALIGNED_ROOT, face_alignment_status
from .unified_beautyproof import (
    MAX_UPLOAD_BYTES,
    analyze_image_bytes,
    unified_status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
VENDOR_DIR = PROJECT_ROOT / "vendor" / "ai-image-detector"

app = FastAPI(title="Mirror of Truth Part B Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (FRONTEND_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")
elif FRONTEND_DIR.exists():
    app.mount("/src", StaticFiles(directory=str(FRONTEND_DIR / "src")), name="src")
HEATMAP_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/heatmaps", StaticFiles(directory=str(HEATMAP_ROOT)), name="heatmaps")
ALIGNED_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/aligned", StaticFiles(directory=str(ALIGNED_ROOT)), name="aligned")


@app.get("/")
def index() -> FileResponse:
    built_index = FRONTEND_DIST_DIR / "index.html"
    if built_index.exists():
        return FileResponse(built_index)
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
        "beautyproof_unified": unified_status(),
        "ai_detector_repo_present": VENDOR_DIR.exists(),
        "aide_detector": aide_status(),
        "face_alignment": face_alignment_status(),
        "real_detector_env": "AIDE is the default AI-generation detector; set backend=ultra with MIRROR_USE_REAL_AIDETECTOR=1 for lynote-ai fallback",
        "supported_modes": ["beautyproof_unified", "beautyproof_v2", "makeup_single", "fashion_single", "before_after"],
    }


@app.post("/api/beautyproof/v2/detect")
async def beautyproof_v2_detect(
    image: UploadFile = File(...),
) -> dict:
    image_bytes = await image.read()
    return BeautyProofV2Service().predict_bytes(image_bytes, image.filename)


@app.post("/api/beautyproof/unified/analyze")
async def beautyproof_unified_analyze(
    image: UploadFile = File(...),
) -> dict:
    image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds 10 MiB limit")
    try:
        return analyze_image_bytes(image_bytes, image.filename)
    except (FileNotFoundError, RuntimeError, ImportError, ModuleNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/analyze")
async def beautyproof_unified_compatible_analyze(
    image: UploadFile = File(...),
) -> dict:
    image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds 10 MiB limit")
    try:
        return analyze_image_bytes(image_bytes, image.filename)
    except (FileNotFoundError, RuntimeError, ImportError, ModuleNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/analyze/single")
async def analyze_single(
    image: UploadFile = File(...),
    mode: str = Form("makeup"),
    backend: str = Form("aide"),
) -> dict:
    image_bytes = await image.read()
    clean_mode = "fashion" if mode == "fashion" else "makeup"
    return build_single_analysis(image_bytes, image.filename, clean_mode, backend)


@app.post("/api/analyze/before-after")
async def analyze_before_after(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    backend: str = Form("aide"),
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
