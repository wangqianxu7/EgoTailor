"""Video retrieval from Ego4D video_info.json with scene-diverse 8h/day selection."""

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
    t = datetime.strptime(time_str, "%H:%M").time()
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


def _add_seconds(date_str: str, time_str: str, seconds: float) -> tuple[str, str]:
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    end = dt + timedelta(seconds=seconds)
    return end.strftime("%Y-%m-%d"), end.strftime("%H:%M:%S")


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
    date_str = _calendar_date(day_index)
    used_today: set[str] = set()
    memory_content: list[dict[str, Any]] = []
    scene_counts: dict[str, int] = defaultdict(int)
    quota_remaining = dict(config.MIN_CLIPS_PER_SCENE)
    current_time = config.DAY_START_TIME

    def try_add_from_chunk(chunk: dict[str, Any], required_scene: str | None = None) -> bool:
        nonlocal current_time
        cands = filter_candidates(videos, chunk, used_today | global_used, persona["location"])
        if required_scene:
            cands = [v for v in cands if v["main_scene"] == required_scene] or cands
        video = pick_video(cands, chunk, rng)
        if not video:
            return False

        start_date, start_time = date_str, current_time
        end_date, end_time = _add_seconds(start_date, start_time, video["video_duration"])
        memory_content.append(
            {
                "video_uid": video["video_uid"],
                "slot_id": chunk.get("slot_id"),
                "plan_chunk": chunk["plan_chunk"],
                "matched_scenarios": chunk.get("matched_scenarios", []),
                "video_scenarios": video["video_scenarios"],
                "main_scene": video["main_scene"],
                "consolidated_summary": video["consolidated_summary"].replace("C", "the character"),
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
        _, current_time = _add_seconds(start_date, end_time, config.GAP_BETWEEN_CLIPS_SEC)
        return True

    for scene in ("outdoor", "mixed", "indoor"):
        while quota_remaining.get(scene, 0) > 0:
            if scene == "outdoor":
                matching_chunks = [c for c in plan_chunks if c.get("location") in ("outdoor", "mixed")] or plan_chunks
            else:
                matching_chunks = [c for c in plan_chunks if c.get("location") == scene] or plan_chunks
            rng.shuffle(matching_chunks)
            added = False
            for chunk in matching_chunks:
                req = scene if scene != "mixed" else None
                if try_add_from_chunk(chunk, required_scene=req):
                    quota_remaining[scene] -= 1
                    added = True
                    break
            if not added:
                break

    total_duration = sum(m["duration"] for m in memory_content)
    chunk_cycle = 0
    while total_duration < config.TARGET_SECONDS_PER_DAY and plan_chunks:
        chunk = plan_chunks[chunk_cycle % len(plan_chunks)]
        before = len(memory_content)
        try_add_from_chunk(chunk)
        if len(memory_content) == before:
            chunk_cycle += 1
            if chunk_cycle > len(plan_chunks) * 3:
                break
            continue
        total_duration = sum(m["duration"] for m in memory_content)
        chunk_cycle += 1

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
        },
        "daily_plan": [f"{c['plan_chunk']} ({c['start_time']}-{c['end_time']})" for c in plan_chunks],
        "memory_content": memory_content,
        "statistics": {
            "clip_count": len(memory_content),
            "total_duration": total_duration,
            "total_duration_hours": round(total_duration / 3600, 2),
            "scene_distribution": dict(scene_counts),
            "selected_videos": [m["video_uid"] for m in memory_content],
            "unique_scenarios": sorted({s for m in memory_content for s in m["video_scenarios"]}),
        },
    }
