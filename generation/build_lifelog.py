#!/usr/bin/env python3
"""
Build a 21-day × 8-hour first-person daily routine lifelog dataset.

Pipeline (mirrors X-LeBench, rule-based variant):
  Stage 1: Persona + 21-day daily plans + plan chunks
  Stage 2: Ego4D video library (video_info.json from X-LeBench)
  Stage 3: Per-day video retrieval with scene diversity + continuous timestamps

Usage:
  cd /root/EgoTailor
  python -m generation.build_lifelog [seed]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from generation import config, retrieve_videos, schedule_templates


def ensure_dirs() -> None:
    for p in (config.PERSONA_PATH, config.LIFELOG_PATH, config.DAYS_PATH):
        p.mkdir(parents=True, exist_ok=True)


def build_dataset(seed: int = 42) -> dict:
    rng = random.Random(seed)
    ensure_dirs()

    print("=" * 60)
    print(f"[EgoTailor] Building {config.TOTAL_DAYS}-day lifelog ({config.HOURS_PER_DAY}h/day)")
    print("=" * 60)

    persona = schedule_templates.build_persona()
    persona["daily_plans"] = {}
    persona["daily_plan_chunks"] = {}
    persona["day_metadata"] = {}
    for day_idx in range(config.TOTAL_DAYS):
        day_rng = random.Random(seed + day_idx * 997)
        persona["daily_plans"][str(day_idx)] = schedule_templates.build_daily_plan(day_idx, day_rng)
        persona["daily_plan_chunks"][str(day_idx)] = schedule_templates.build_daily_plan_chunks(
            day_idx, day_rng
        )
        persona["day_metadata"][str(day_idx)] = schedule_templates.build_day_metadata(day_idx, day_rng)

    persona_path = config.PERSONA_PATH / f"persona_{persona['persona_id']}.json"
    with open(persona_path, "w") as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)
    print(f"[Stage 1] Persona saved: {persona_path}")

    videos = retrieve_videos.load_video_library()
    print(f"[Stage 2] Loaded {len(videos)} Ego4D clips from {config.EGO4DINFO_PATH / 'video_info.json'}")

    global_used: set[str] = set()
    all_days = []
    aggregate_duration = 0.0
    aggregate_clips = 0
    all_scenes: dict[str, int] = {}

    for day_idx in range(config.TOTAL_DAYS):
        chunks = persona["daily_plan_chunks"][str(day_idx)]
        day_meta = persona["day_metadata"][str(day_idx)]
        day_log = retrieve_videos.build_day_lifelog(
            day_index=day_idx,
            plan_chunks=chunks,
            videos=videos,
            persona=persona,
            global_used=global_used,
            rng=rng,
            day_meta=day_meta,
        )
        all_days.append(day_log)
        aggregate_duration += day_log["statistics"]["total_duration"]
        aggregate_clips += day_log["statistics"]["clip_count"]
        for scene, cnt in day_log["statistics"]["scene_distribution"].items():
            all_scenes[scene] = all_scenes.get(scene, 0) + cnt

        day_path = config.DAYS_PATH / f"day_{day_idx:02d}_{day_log['metadata']['calendar_date']}.json"
        with open(day_path, "w") as f:
            json.dump(day_log, f, indent=2, ensure_ascii=False)

        stats = day_log["statistics"]
        theme = day_log["metadata"].get("day_theme", "")
        anomalies = day_log["metadata"].get("anomaly_events", [])
        anomaly_str = f" anomalies={anomalies}" if anomalies else ""
        print(
            f"  Day {day_idx + 1:2d} ({day_log['metadata']['calendar_date']} "
            f"{day_log['metadata']['day_of_week']:9s}) [{theme}]: "
            f"{stats['clip_count']:3d} clips, {stats['total_duration_hours']:.2f}h, "
            f"scenes={stats['scene_distribution']}{anomaly_str}"
        )

    lifelog = {
        "metadata": {
            "project": config.PROJECT_NAME,
            "version": config.VERSION,
            "persona_id": persona["persona_id"],
            "total_days": config.TOTAL_DAYS,
            "hours_per_day_target": config.HOURS_PER_DAY,
            "start_date": config.START_DATE,
            "location": persona["location"],
            "gen_way": "rule_based",
            "video_info_source": str(config.EGO4DINFO_PATH / "video_info.json"),
        },
        "persona_summary": {
            "mbti": persona["personality_traits"]["mbti_type"],
            "lifestyle": persona["lifestyle"],
            "hobbies": persona["hobbies"],
        },
        "days": all_days,
        "aggregate_statistics": {
            "total_clips": aggregate_clips,
            "total_duration_seconds": aggregate_duration,
            "total_duration_hours": round(aggregate_duration / 3600, 2),
            "avg_hours_per_day": round(aggregate_duration / 3600 / config.TOTAL_DAYS, 2),
            "scene_distribution": all_scenes,
            "unique_video_uids": len({m["video_uid"] for d in all_days for m in d["memory_content"]}),
        },
    }

    lifelog_path = config.LIFELOG_PATH / f"lifelog_{persona['persona_id']}_21d.json"
    with open(lifelog_path, "w") as f:
        json.dump(lifelog, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"[Done] Lifelog: {lifelog_path}")
    print(
        f"  Total: {lifelog['aggregate_statistics']['total_duration_hours']}h "
        f"({lifelog['aggregate_statistics']['total_clips']} clips, "
        f"{lifelog['aggregate_statistics']['unique_video_uids']} unique videos)"
    )
    print(f"  Scene coverage: {lifelog['aggregate_statistics']['scene_distribution']}")
    print("=" * 60)
    return lifelog


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    build_dataset(seed=seed)
