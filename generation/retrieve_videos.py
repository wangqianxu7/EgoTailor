"""Video retrieval from Ego4D video_info.json.

Placement policy (one clip per daily-plan slot):
  - Prefer start_time == plan slot start.
  - If previous clip overran into this slot, chain: start == previous end.
  - If previous clip under-filled, next clip still starts at its own plan start.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from generation import config


def load_video_library() -> list[dict[str, Any]]:
    path = Path(config.EGO4DINFO_PATH) / "video_info.json"
    with open(path) as f:
        return json.load(f)


def scenario_iou(list_a: list[str], list_b: list[str]) -> float:
    sa, sb = set(list_a), set(list_b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def classify_time_from_hhmm(time_str: str) -> str:
    # Accept "HH:MM" or "HH:MM:SS"
    parts = time_str.split(":")
    t = datetime.strptime(f"{parts[0]}:{parts[1]}", "%H:%M").time()
    day_start = datetime.strptime("06:00", "%H:%M").time()
    twilight_start = datetime.strptime("17:00", "%H:%M").time()
    night_start = datetime.strptime("19:00", "%H:%M").time()
    if day_start < t < twilight_start:
        return "daytime"
    if twilight_start <= t < night_start:
        return "twilight"
    return "nighttime"


def filter_candidates(
    videos: list[dict[str, Any]],
    chunk: dict[str, Any],
    used_uids: set[str],
    location: str,
) -> list[dict[str, Any]]:
    target_scene = chunk.get("location", "indoor")
    target_time = chunk.get("time_period") or classify_time_from_hhmm(chunk["start_time"])
    target_scenarios = chunk.get("matched_scenarios", [])

    def passes(v: dict[str, Any], strict_location: bool, iou_thr: float) -> bool:
        if v["video_uid"] in used_uids:
            return False
        dur = v["video_duration"]
        if dur < config.MIN_VIDEO_DURATION or dur > config.MAX_VIDEO_DURATION:
            return False
        if strict_location and v["video_source"] != location:
            return False
        if v["main_scene"] not in (target_scene, "mixed"):
            return False
        if v["time_period"] not in (target_time, "not know"):
            return False
        if scenario_iou(target_scenarios, v["video_scenarios"]) < iou_thr:
            return False
        return True

    for strict, thr in [(True, config.IOU_THRESHOLD), (True, 0.0), (False, 0.0)]:
        cands = [v for v in videos if passes(v, strict, thr)]
        if cands:
            return cands
    return [
        v
        for v in videos
        if v["video_uid"] not in used_uids
        and v["video_source"] == location
        and config.MIN_VIDEO_DURATION <= v["video_duration"] <= config.MAX_VIDEO_DURATION
    ]


def score_candidate(video: dict[str, Any], chunk: dict[str, Any]) -> float:
    iou = scenario_iou(chunk.get("matched_scenarios", []), video["video_scenarios"])
    scene_bonus = 1.0 if video["main_scene"] == chunk.get("location") else 0.5
    dur = video["video_duration"]
    # Prefer clips that roughly fit the plan-slot duration when available
    slot_sec = _chunk_duration_sec(chunk)
    if slot_sec > 0:
        dur_score = 1.0 - abs(dur - min(slot_sec, 1800)) / max(slot_sec, 900)
    else:
        dur_score = 1.0 - abs(dur - 900) / 900
    return iou * 2 + scene_bonus + max(dur_score, 0)


def pick_video(
    candidates: list[dict[str, Any]],
    chunk: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda v: score_candidate(v, chunk), reverse=True)
    return rng.choice(ranked[: min(5, len(ranked))])


def _format_ts(date_str: str, time_str: str) -> str:
    return f"{date_str}T{time_str}"


def _to_hms(time_str: str) -> str:
    """Normalize 'HH:MM' or 'HH:MM:SS' -> 'HH:MM:SS'."""
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"
    if len(parts) == 3:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
    raise ValueError(f"Invalid time string: {time_str}")


def _add_seconds(date_str: str, time_str: str, seconds: float) -> tuple[str, str]:
    dt = datetime.strptime(f"{date_str} {_to_hms(time_str)}", "%Y-%m-%d %H:%M:%S")
    end = dt + timedelta(seconds=seconds)
    return end.strftime("%Y-%m-%d"), end.strftime("%H:%M:%S")


def _time_ge(a: str, b: str) -> bool:
    return _to_hms(a) >= _to_hms(b)


def _chunk_duration_sec(chunk: dict[str, Any]) -> float:
    try:
        start = datetime.strptime(_to_hms(chunk["start_time"]), "%H:%M:%S")
        end = datetime.strptime(_to_hms(chunk["end_time"]), "%H:%M:%S")
        delta = (end - start).total_seconds()
        return delta if delta > 0 else 0.0
    except (KeyError, ValueError):
        return 0.0


def _calendar_date(day_index: int) -> str:
    start = datetime.strptime(config.START_DATE, "%Y-%m-%d")
    return (start + timedelta(days=day_index)).strftime("%Y-%m-%d")


def build_day_lifelog(
    day_index: int,
    plan_chunks: list[dict[str, Any]],
    videos: list[dict[str, Any]],
    persona: dict[str, Any],
    global_used: set[str],
    rng: random.Random,
    day_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one day: exactly one video per daily-plan slot.

    Timing rules:
      - Prefer aligning each clip's start to the plan slot's start_time.
      - If the previous clip ended *after* the next plan's start, the next clip
        starts immediately at previous end (no forced gap).
      - If the previous clip ended *before* the next plan's start (under-filled),
        the next clip starts at the next plan's scheduled start_time.
    """
    date_str = _calendar_date(day_index)
    used_today: set[str] = set()
    memory_content: list[dict[str, Any]] = []
    scene_counts: dict[str, int] = defaultdict(int)

    # Cursor = absolute end of last placed clip on this calendar day (HH:MM:SS), or None
    cursor_end: str | None = None

    chrono_chunks = sorted(plan_chunks, key=lambda c: _to_hms(c["start_time"]))

    for chunk in chrono_chunks:
        cands = filter_candidates(videos, chunk, used_today | global_used, persona["location"])
        video = pick_video(cands, chunk, rng)
        if not video:
            continue

        plan_start = _to_hms(chunk["start_time"])
        plan_end = _to_hms(chunk["end_time"])

        if cursor_end is None or not _time_ge(cursor_end, plan_start):
            # Under-filled previous slot (or first clip): align to this plan's start
            start_time = plan_start
        else:
            # Previous overflowed past this plan start: chain immediately after it
            start_time = cursor_end

        start_date = date_str
        end_date, end_time = _add_seconds(start_date, start_time, video["video_duration"])
        # Keep cursor on the same calendar day clock for chaining
        if end_date != date_str:
            # Crossed midnight — clamp cursor to end_time still usable as HH:MM:SS next day edge
            cursor_end = end_time
        else:
            cursor_end = end_time

        memory_content.append(
            {
                "video_uid": video["video_uid"],
                "slot_id": chunk.get("slot_id"),
                "plan_chunk": chunk["plan_chunk"],
                "matched_scenarios": chunk.get("matched_scenarios", []),
                "video_scenarios": video["video_scenarios"],
                "main_scene": video["main_scene"],
                "consolidated_summary": video["consolidated_summary"].replace("C", "the character"),
                "plan_start_time": plan_start,
                "plan_end_time": plan_end,
                "start_time": start_time,
                "end_time": end_time,
                "start_timestamp": _format_ts(start_date, start_time),
                "end_timestamp": _format_ts(end_date, end_time),
                "duration": video["video_duration"],
                "video_source": video["video_source"],
                "time_period": video["time_period"],
            }
        )
        used_today.add(video["video_uid"])
        if not config.ALLOW_CROSS_DAY_REUSE:
            global_used.add(video["video_uid"])
        scene_counts[video["main_scene"]] += 1

    total_duration = sum(m["duration"] for m in memory_content)
    day_meta = day_meta or {}
    return {
        "metadata": {
            "persona_id": persona["persona_id"],
            "day_index": day_index,
            "calendar_date": date_str,
            "day_of_week": datetime.strptime(date_str, "%Y-%m-%d").strftime("%A"),
            "is_weekend": datetime.strptime(date_str, "%Y-%m-%d").weekday() >= 5,
            "target_seconds": config.TARGET_SECONDS_PER_DAY,
            "location": persona["location"],
            "day_theme": day_meta.get("day_theme", plan_chunks[0].get("day_theme", "") if plan_chunks else ""),
            "anomaly_events": day_meta.get("anomaly_events", []),
            "time_variations": day_meta.get("time_variations", {}),
            "placement_policy": "one_clip_per_plan_align_or_chain",
        },
        "daily_plan": [f"{c['plan_chunk']} ({c['start_time']}-{c['end_time']})" for c in plan_chunks],
        "memory_content": memory_content,
        "statistics": {
            "clip_count": len(memory_content),
            "plan_slot_count": len(chrono_chunks),
            "total_duration": total_duration,
            "total_duration_hours": round(total_duration / 3600, 2),
            "scene_distribution": dict(scene_counts),
            "selected_videos": [m["video_uid"] for m in memory_content],
            "unique_scenarios": sorted({s for m in memory_content for s in m["video_scenarios"]}),
        },
    }
