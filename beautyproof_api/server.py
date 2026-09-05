from __future__ import annotations
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from .unified import UnifiedBeautyProofAPI

app=FastAPI(title="BeautyProof Unified API",version="1.0.0")
MAX_UPLOAD_BYTES=10*1024*1024

@app.get("/health")
def health(): return {"status":"ok","api_version":"1.0.0"}

@app.post("/v1/analyze")
async def analyze(image: UploadFile=File(...)):
    suffix=Path(image.filename or "image.jpg").suffix or ".jpg"
    data=await image.read(MAX_UPLOAD_BYTES+1)
    if not data: raise HTTPException(400,"empty image")
    if len(data)>MAX_UPLOAD_BYTES: raise HTTPException(413,"image exceeds 10 MiB limit")
    with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as f: f.write(data);path=Path(f.name)
    try:return UnifiedBeautyProofAPI().analyze(path)
    except (FileNotFoundError,RuntimeError) as exc: raise HTTPException(503,str(exc)) from exc
    except (ValueError,OSError) as exc: raise HTTPException(422,str(exc)) from exc
    finally:path.unlink(missing_ok=True)
