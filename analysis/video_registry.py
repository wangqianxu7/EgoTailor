"""Map lifelog video_uid entries to local Ego4D MP4 files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis.config import DEFAULT_LIFELOG, EGO4D_VIDEO_ROOT, VIDEO_REGISTRY_PATH
from analysis.lifelog_index import load_lifelog


def resolve_video_path(video_uid: str, root: Path | None = None) -> Path | None:
    root = root or EGO4D_VIDEO_ROOT
    path = root / f"{video_uid}.mp4"
    return path if path.exists() else None


def build_video_registry(
    lifelog_path: Path | None = None,
    video_root: Path | None = None,
) -> dict[str, Any]:
    lifelog_path = lifelog_path or DEFAULT_LIFELOG
    video_root = video_root or EGO4D_VIDEO_ROOT
    lifelog = load_lifelog(lifelog_path)

    entries: list[dict[str, Any]] = []
    uid_set: set[str] = set()

    for day in lifelog["days"]:
        meta = day["metadata"]
        for i, clip in enumerate(day["memory_content"]):
            uid = clip["video_uid"]
            path = resolve_video_path(uid, video_root)
            entries.append(
                {
                    "video_uid": uid,
                    "node_id": f"clip_{meta['day_index']:02d}_{i:03d}",
                    "day_index": meta["day_index"],
                    "calendar_date": meta["calendar_date"],
                    "day_theme": meta.get("day_theme", ""),
                    "slot_id": clip.get("slot_id", ""),
                    "start_timestamp": clip["start_timestamp"],
                    "end_timestamp": clip["end_timestamp"],
                    "duration": clip["duration"],
                    "plan_chunk": clip.get("plan_chunk", ""),
                    "consolidated_summary": clip.get("consolidated_summary", ""),
                    "video_scenarios": clip.get("video_scenarios", []),
                    "main_scene": clip.get("main_scene", ""),
                    "video_available": path is not None,
                    "video_path": str(path) if path else None,
                }
            )
            uid_set.add(uid)

    available = [e for e in entries if e["video_available"]]
    unique_available = {e["video_uid"] for e in available}

    registry = {
        "lifelog_path": str(lifelog_path),
        "video_root": str(video_root),
        "total_clips": len(entries),
        "clips_with_local_video": len(available),
        "unique_video_uids": len(uid_set),
        "unique_uids_with_local_video": len(unique_available),
        "coverage_ratio": round(len(unique_available) / len(uid_set), 4) if uid_set else 0.0,
        "entries": entries,
        "uid_to_path": {
            e["video_uid"]: e["video_path"]
            for e in entries
            if e["video_available"] and e["video_path"]
        },
    }
    return registry


def save_registry(registry: dict[str, Any], path: Path | None = None) -> Path:
    path = path or VIDEO_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    return path


def load_registry(path: Path | None = None) -> dict[str, Any]:
    path = path or VIDEO_REGISTRY_PATH
    with open(path) as f:
        return json.load(f)


def available_uids(registry: dict[str, Any]) -> set[str]:
    return set(registry.get("uid_to_path", {}).keys())
