from __future__ import annotations

import io
import json
import math
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_ROOT = PROJECT_ROOT / "runtime" / "aligned"
DEFAULT_MEDIAPIPE_PYTHON = PROJECT_ROOT / ".venv-mediapipe" / "bin" / "python"
WORKER_SCRIPT = PROJECT_ROOT / "backend" / "mediapipe_face_worker.py"


@dataclass(frozen=True)
class FaceFeatures:
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    angle_degrees: float
    eye_distance: float
    landmarks_count: int
    provider: str = "unknown"


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _percent(value: float) -> str:
    return f"{value:+.1%}" if value < 0 else f"{value:.1%}"


def _safe_stem(filename: str | None) -> str:
    stem = Path(filename or f"upload_{uuid.uuid4().hex}").stem
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)[:70]


def _face_landmarker_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_model = os.getenv("MEDIAPIPE_FACE_LANDMARKER_PATH")
    if env_model:
        candidates.append(Path(env_model).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT / "models" / "face_landmarker.task",
            PROJECT_ROOT.parent / "models" / "face_landmarker.task",
        ]
    )
    api_root = os.getenv("BEAUTYPROOF_API_PATH")
    if api_root:
        candidates.append(Path(api_root).expanduser() / "models" / "face_landmarker.task")
    return [path.resolve() for path in candidates if path.is_file()]


def face_alignment_status() -> dict[str, Any]:
    task_model = next(iter(_face_landmarker_candidates()), None)
    worker_python = Path(os.getenv("MEDIAPIPE_PYTHON", str(DEFAULT_MEDIAPIPE_PYTHON))).expanduser()
    return {
        "provider": "MediaPipe FaceLandmarker subprocess",
        "mediapipe_task_model": str(task_model) if task_model else None,
        "mediapipe_worker_python": str(worker_python),
        "mediapipe_worker_present": worker_python.exists(),
        "mediapipe_required": os.getenv("ALLOW_FACE_ALIGNMENT_FALLBACK") != "1",
        "task_model_env": "optional MEDIAPIPE_FACE_LANDMARKER_PATH=/path/to/face_landmarker.task",
        "worker_python_env": "optional MEDIAPIPE_PYTHON=/path/to/python-with-mediapipe",
        "fallback_env": "set ALLOW_FACE_ALIGNMENT_FALLBACK=1 only for local debugging",
        "aligned_image_route": "/aligned",
    }


def detect_face_features(image_bytes: bytes) -> FaceFeatures:
    try:
        return _detect_with_mediapipe_worker(image_bytes)
    except Exception:
        if os.getenv("ALLOW_FACE_ALIGNMENT_FALLBACK") == "1":
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
            return _detect_with_opencv(rgb)
        raise


def _detect_with_mediapipe_worker(image_bytes: bytes) -> FaceFeatures:
    task_model = next(iter(_face_landmarker_candidates()), None)
    if task_model is None:
        raise RuntimeError("face_landmarker.task not found")
    worker_python = Path(os.getenv("MEDIAPIPE_PYTHON", str(DEFAULT_MEDIAPIPE_PYTHON))).expanduser()
    if not worker_python.exists():
        raise RuntimeError(f"MediaPipe worker python not found: {worker_python}")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(image_bytes)
        image_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [
                str(worker_python),
                str(WORKER_SCRIPT),
                "--image",
                str(image_path),
                "--model",
                str(task_model),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    finally:
        image_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        message = completed.stdout.strip() or completed.stderr.strip() or f"worker exited {completed.returncode}"
        raise RuntimeError(message)
    payload = json.loads(completed.stdout)
    if payload.get("status") != "ok":
        raise RuntimeError(payload.get("error", "MediaPipe worker failed"))
    features = payload["features"]
    return FaceFeatures(
        bbox=tuple(features["bbox"]),
        center=tuple(features["center"]),
        angle_degrees=float(features["angle_degrees"]),
        eye_distance=float(features["eye_distance"]),
        landmarks_count=int(features["landmarks_count"]),
        provider=features.get("provider", "mediapipe_tasks_subprocess"),
    )


def _detect_with_mediapipe_tasks(rgb: Image.Image, model_path: Path) -> FaceFeatures:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb))
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        raise ValueError("no face detected")
    landmarks = result.face_landmarks[0]
    return _features_from_landmarks(landmarks, "mediapipe_tasks")


def _detect_with_mediapipe_solutions(rgb: Image.Image) -> FaceFeatures:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is not installed") from exc

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(np.asarray(rgb))

    if not result.multi_face_landmarks:
        raise ValueError("no face detected")

    landmarks = result.multi_face_landmarks[0].landmark
    return _features_from_landmarks(landmarks, "mediapipe_solutions")


def _features_from_landmarks(landmarks, provider: str) -> FaceFeatures:
    xs = [max(0.0, min(1.0, point.x)) for point in landmarks]
    ys = [max(0.0, min(1.0, point.y)) for point in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    left_eye = _mean_point(landmarks, [33, 133])
    right_eye = _mean_point(landmarks, [362, 263])
    angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    eye_distance = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])

    return FaceFeatures(
        bbox=(_rounded(x_min), _rounded(y_min), _rounded(x_max), _rounded(y_max)),
        center=(_rounded((x_min + x_max) / 2), _rounded((y_min + y_max) / 2)),
        angle_degrees=_rounded(angle),
        eye_distance=_rounded(eye_distance),
        landmarks_count=len(landmarks),
        provider=provider,
    )


def _detect_with_opencv(rgb: Image.Image) -> FaceFeatures:
    import cv2

    image = np.asarray(rgb)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(48, 48))
    if len(faces) == 0:
        raise ValueError("no face detected")
    x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
    image_height, image_width = gray.shape
    x_min = x / image_width
    y_min = y / image_height
    x_max = (x + width) / image_width
    y_max = (y + height) / image_height
    return FaceFeatures(
        bbox=(_rounded(x_min), _rounded(y_min), _rounded(x_max), _rounded(y_max)),
        center=(_rounded((x_min + x_max) / 2), _rounded((y_min + y_max) / 2)),
        angle_degrees=0.0,
        eye_distance=_rounded((width / image_width) * 0.45),
        landmarks_count=0,
        provider="opencv_haar_fallback",
    )


def _mean_point(landmarks, indices: list[int]) -> tuple[float, float]:
    xs = [landmarks[index].x for index in indices]
    ys = [landmarks[index].y for index in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def compare_face_features(before: FaceFeatures, after: FaceFeatures) -> dict[str, Any]:
    before_width = before.bbox[2] - before.bbox[0]
    before_height = before.bbox[3] - before.bbox[1]
    after_width = after.bbox[2] - after.bbox[0]
    after_height = after.bbox[3] - after.bbox[1]
    before_diag = math.hypot(before_width, before_height)
    after_diag = math.hypot(after_width, after_height)
    avg_diag = max((before_diag + after_diag) / 2, 1e-6)

    center_delta = math.hypot(after.center[0] - before.center[0], after.center[1] - before.center[1])
    alignment_offset = center_delta / avg_diag
    face_angle_diff = abs(after.angle_degrees - before.angle_degrees)
    if face_angle_diff > 180:
        face_angle_diff = 360 - face_angle_diff
    bbox_size_diff = abs(after_diag - before_diag) / max(before_diag, 1e-6)
    crop_shift = max(
        abs(after.bbox[0] - before.bbox[0]),
        abs(after.bbox[1] - before.bbox[1]),
        abs(after.bbox[2] - before.bbox[2]),
        abs(after.bbox[3] - before.bbox[3]),
    )

    angle_similar = face_angle_diff < 8.0
    crop_similar = bbox_size_diff < 0.10 and crop_shift < 0.12
    aligned = alignment_offset < 0.05 and angle_similar and crop_similar

    return {
        "alignment_status": "Aligned" if aligned else "Misaligned",
        "alignment_offset": _percent(alignment_offset),
        "alignment_offset_ratio": _rounded(alignment_offset),
        "alignment_success": aligned,
        "face_angle_diff_degrees": _rounded(face_angle_diff),
        "face_angle_similarity": "Similar" if angle_similar else "Significant",
        "bbox_size_diff": _percent(bbox_size_diff),
        "bbox_size_diff_ratio": _rounded(bbox_size_diff),
        "crop_shift_ratio": _rounded(crop_shift),
        "crop_similarity": "Similar" if crop_similar else "Significant",
        "before_face": _serialize_features(before),
        "after_face": _serialize_features(after),
    }


def compare_before_after_faces(
    before_bytes: bytes,
    after_bytes: bytes,
    before_filename: str | None = None,
    after_filename: str | None = None,
) -> dict[str, Any]:
    try:
        before = detect_face_features(before_bytes)
        after = detect_face_features(after_bytes)
        geometry = compare_face_features(before, after)
        aligned = save_aligned_face_pair(before_bytes, after_bytes, before, after, before_filename, after_filename)
        geometry.update(aligned)
        geometry["status"] = "ok"
        geometry["agent_should_notice"] = _geometry_notices(geometry)
        return geometry
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "unavailable",
            "error": f"{exc.__class__.__name__}: {exc}",
            "alignment_status": "Unknown",
            "alignment_offset": "Unknown",
            "alignment_offset_ratio": None,
            "alignment_success": False,
            "face_angle_diff_degrees": None,
            "face_angle_similarity": "Unknown",
            "bbox_size_diff": "Unknown",
            "bbox_size_diff_ratio": None,
            "crop_shift_ratio": None,
            "crop_similarity": "Unknown",
            "aligned_before_url": None,
            "aligned_after_url": None,
            "agent_should_notice": ["未能检测到可用的人脸关键点，角度和裁剪可比性无法判断。"],
        }


def save_aligned_face_pair(
    before_bytes: bytes,
    after_bytes: bytes,
    before: FaceFeatures,
    after: FaceFeatures,
    before_filename: str | None,
    after_filename: str | None,
) -> dict[str, Any]:
    ALIGNED_ROOT.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    before_path = ALIGNED_ROOT / f"{_safe_stem(before_filename)}_{token}_aligned_before.png"
    after_path = ALIGNED_ROOT / f"{_safe_stem(after_filename)}_{token}_aligned_after.png"
    _aligned_crop(before_bytes, before).save(before_path)
    _aligned_crop(after_bytes, after).save(after_path)
    return {
        "aligned_before": str(before_path),
        "aligned_after": str(after_path),
        "aligned_before_url": f"/aligned/{before_path.name}",
        "aligned_after_url": f"/aligned/{after_path.name}",
    }


def _aligned_crop(image_bytes: bytes, features: FaceFeatures) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    corrected = rgb.rotate(-features.angle_degrees, resample=Image.Resampling.BICUBIC, expand=True)
    width, height = rgb.size
    rotated_width, rotated_height = corrected.size
    center_x = features.center[0] * width + (rotated_width - width) / 2
    center_y = features.center[1] * height + (rotated_height - height) / 2
    face_w = (features.bbox[2] - features.bbox[0]) * width
    face_h = (features.bbox[3] - features.bbox[1]) * height
    crop_size = max(face_w, face_h) * 1.45
    left = int(round(center_x - crop_size / 2))
    top = int(round(center_y - crop_size / 2))
    right = int(round(center_x + crop_size / 2))
    bottom = int(round(center_y + crop_size / 2))
    crop = corrected.crop((left, top, right, bottom))
    return ImageOps.pad(crop, (512, 512), color=(245, 247, 250), method=Image.Resampling.BICUBIC)


def _serialize_features(features: FaceFeatures) -> dict[str, Any]:
    return {
        "bbox": list(features.bbox),
        "center": list(features.center),
        "angle_degrees": features.angle_degrees,
        "eye_distance": features.eye_distance,
        "landmarks_count": features.landmarks_count,
        "provider": features.provider,
    }


def _geometry_notices(geometry: dict[str, Any]) -> list[str]:
    notices: list[str] = []
    if geometry["alignment_offset_ratio"] >= 0.05:
        notices.append(f"before/after 人脸中心偏移 {geometry['alignment_offset']}，超过 5% 对齐阈值。")
    if geometry["face_angle_similarity"] == "Significant":
        notices.append(f"before/after 人脸角度差 {geometry['face_angle_diff_degrees']}°，角度变化会影响妆效归因。")
    if geometry["crop_similarity"] == "Significant":
        notices.append(f"before/after 人脸框大小或裁剪差异明显，bbox 差异 {geometry['bbox_size_diff']}。")
    if not notices:
        notices.append("MediaPipe 人脸关键点显示两图角度、裁剪和中心位置相对可比。")
    return notices
