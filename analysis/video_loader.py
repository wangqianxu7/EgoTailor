"""Load Ego4D videos by video_uid and extract frames for MLLM."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from analysis.config import EGO4D_VIDEO_ROOT, FRAME_MAX_SIZE


def resolve_video_path(video_uid: str, root: Path | None = None) -> Path | None:
    root = root or EGO4D_VIDEO_ROOT
    path = root / f"{video_uid}.mp4"
    return path if path.exists() else None


def get_video_duration_sec(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    if fps <= 0 or total <= 0:
        return 0.0
    return total / fps


def adaptive_frame_count(duration_sec: float) -> int:
    """Pick frame count from video duration.

    - < 5 min:  4–8 frames
    - 5–20 min: 8–16 frames  (lifelog clips usually fall here)
    - 20–30 min: 16 frames
    - > 30 min: 16–32 frames (caps at 32)
    """
    if duration_sec <= 0:
        return 8
    minutes = duration_sec / 60.0
    if minutes <= 5:
        return int(round(4 + (minutes / 5) * 4))
    if minutes <= 20:
        return int(round(8 + ((minutes - 5) / 15) * 8))
    if minutes <= 30:
        return 16
    # > 30 min: ramp 16 -> 32 over the next 30 minutes
    return min(32, int(round(16 + ((minutes - 30) / 30) * 16)))


def _resize_frame(frame: np.ndarray, max_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_frames(
    video_path: Path,
    num_frames: int | None = None,
    max_size: int = FRAME_MAX_SIZE,
    duration_sec: float | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"duration_sec": 0.0, "num_frames_requested": 0, "sampling": "failed"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if duration_sec is None and fps > 0 and total > 0:
        duration_sec = total / fps
    duration_sec = duration_sec or 0.0

    if num_frames is None:
        num_frames = adaptive_frame_count(duration_sec)
        sampling = "adaptive"
    else:
        sampling = "fixed"

    if total <= 0:
        cap.release()
        return [], {
            "duration_sec": duration_sec,
            "num_frames_requested": num_frames,
            "sampling": sampling,
        }

    num_frames = max(1, min(num_frames, total))
    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(_resize_frame(frame, max_size))
    cap.release()

    meta = {
        "duration_sec": round(duration_sec, 2),
        "duration_min": round(duration_sec / 60, 2),
        "total_video_frames": total,
        "fps": round(fps, 2),
        "num_frames_requested": num_frames,
        "num_frames_extracted": len(frames),
        "frame_indices": indices.tolist(),
        "sampling": sampling,
    }
    return frames, meta


def frames_to_base64_jpeg(frames: list[np.ndarray], quality: int = 85) -> list[str]:
    encoded: list[str] = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            encoded.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
    return encoded


def load_clip_visuals(
    video_uid: str,
    root: Path | None = None,
    num_frames: int | None = None,
    duration_hint_sec: float | None = None,
) -> dict[str, Any]:
    path = resolve_video_path(video_uid, root)
    if path is None:
        return {
            "video_uid": video_uid,
            "available": False,
            "path": None,
            "frames_b64": [],
            "frame_count": 0,
        }
    frames, sampling_meta = extract_frames(
        path,
        num_frames=num_frames,
        duration_sec=duration_hint_sec,
    )
    return {
        "video_uid": video_uid,
        "available": bool(frames),
        "path": str(path),
        "frames_b64": frames_to_base64_jpeg(frames),
        "frame_count": len(frames),
        **sampling_meta,
    }
