from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps


def rounded(value: float) -> float:
    return round(float(value), 6)


def mean_point(landmarks, indices: list[int]) -> tuple[float, float]:
    xs = [landmarks[index].x for index in indices]
    ys = [landmarks[index].y for index in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def detect(image_path: Path, model_path: Path) -> dict:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image))
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        raise ValueError("no face detected")

    landmarks = result.face_landmarks[0]
    xs = [max(0.0, min(1.0, point.x)) for point in landmarks]
    ys = [max(0.0, min(1.0, point.y)) for point in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    left_eye = mean_point(landmarks, [33, 133])
    right_eye = mean_point(landmarks, [362, 263])
    angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    eye_distance = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    return {
        "bbox": [rounded(x_min), rounded(y_min), rounded(x_max), rounded(y_max)],
        "center": [rounded((x_min + x_max) / 2), rounded((y_min + y_max) / 2)],
        "angle_degrees": rounded(angle),
        "eye_distance": rounded(eye_distance),
        "landmarks_count": len(landmarks),
        "provider": "mediapipe_tasks_subprocess",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps({"status": "ok", "features": detect(Path(args.image), Path(args.model))}))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
