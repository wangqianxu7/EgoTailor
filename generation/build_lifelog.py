#!/usr/bin/env python3
"""
Build an N-day x 8-hour first-person daily routine lifelog dataset.
(Run length is config.TOTAL_DAYS; the corpus supports ~35 days before the
persona's scenario entropy falls through its guardrail — see persona_generator.)

Pipeline (mirrors X-LeBench, rule-based variant):
  Stage 1: Persona + daily plans + plan chunks
  Stage 2: Ego4D video library (video_info.json from X-LeBench)
  Stage 3: Per-day video retrieval with scene diversity + continuous timestamps

Stage 1 is corpus-driven by default: ``persona_generator`` reads what Ego4D
actually contains and describes someone who could have produced it, then
``schedule_from_quota`` gives that person a day. ``--legacy`` restores
``schedule_templates``, which writes schedules in scenarios chosen by hand and
consequently asks for footage the corpus does not have — it requested ``Bus``
23 times and ``Video call`` 13 times against a supply of zero, and only 41% of
its slots could be filled with a scenario-matched clip. It is kept for
comparison, not for use.

Usage:
  cd /root/EgoTailor
  python -m generation.build_lifelog [seed] [--legacy]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from generation import (
    config,
    persona_generator,
    retrieve_videos,
    schedule_from_quota,
    schedule_templates,
)


def ensure_dirs() -> None:
    for p in (config.PERSONA_PATH, config.LIFELOG_PATH, config.DAYS_PATH):
        p.mkdir(parents=True, exist_ok=True)


def build_legacy_persona(seed: int) -> dict:
    """Stage 1 via the hand-written scenario schedule. Kept only for comparison."""
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
    return persona


def write_portrait(spec: Any, path: Path) -> None:
    """The complete persona portrait: who this person is, and the evidence.

    One file, self-contained. Every quota row carries the measurements that
    justify it — how many clips in the pool could fill it, how many the run
    can draw without reuse, how long the median one runs — so the portrait
    can be checked against the corpus rather than taken on faith.
    """
    with open(path, "w") as f:
        json.dump(spec.to_dict(), f, indent=2, ensure_ascii=False)


def write_day_plan(day_log: dict, persona_id: str) -> Path:
    """One day's plan, each slot aligned to the clip that filled it.

    This is the file to read when checking the dataset by hand: planned
    activity, when it was planned, when the clip actually runs, the clip's uid
    and its own description, and the evidence the match is real — which
    scenarios the plan asked for, which the clip carries, their intersection
    and IoU, and which retrieval tier served it. Anything below
    ``T1_scenario_match`` means the request was relaxed to find a clip.
    """
    meta = day_log["metadata"]
    stats = day_log["statistics"]
    plan = {
        "persona_id": persona_id,
        "day_index": meta["day_index"],
        "date": meta["calendar_date"],
        "day_of_week": meta["day_of_week"],
        "is_weekend": meta["is_weekend"],
        "day_theme": meta["day_theme"],
        "anomaly_events": meta["anomaly_events"],
        "summary": {
            "slots_filled": f"{stats['clip_count']}/{stats['plan_slot_count']}",
            "fill_rate": stats["slot_fill_rate"],
            "hours": stats["total_duration_hours"],
            "retrieval_tiers": stats["retrieval_tier_counts"],
            "unfilled_slots": [u["plan_chunk"] for u in stats["unfilled_slots"]],
        },
        "plan": [
            {
                "slot_id": m["slot_id"],
                "activity": m["plan_chunk"],
                "planned": f"{m['plan_start_time'][:5]}-{m['plan_end_time'][:5]}",
                "actual": f"{m['start_time'][:5]}-{m['end_time'][:5]}",
                "minutes": round(m["duration"] / 60, 1),
                "video_uid": m["video_uid"],
                "video_description": m["consolidated_summary"],
                "video_scenarios": m["video_scenarios"],
                "main_scene": m["main_scene"],
                "time_period": m["time_period"],
                "video_source": m["video_source"],
                "requested_scenarios": m["requested_scenarios"],
                "matched_scenarios": m["matched_scenarios"],
                "scenario_iou": m["scenario_iou"],
                "retrieval_tier": m["retrieval_tier"],
                "start_timestamp": m["start_timestamp"],
                "end_timestamp": m["end_timestamp"],
            }
            for m in day_log["memory_content"]
        ],
    }
    # Namespaced by persona: the filename carries only a date, so two personas
    # generated in a row would silently overwrite each other's days while their
    # portraits and lifelogs sat side by side, un-clobbered and now mismatched.
    day_dir = config.DAYS_PATH / persona_id
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"day_{meta['day_index']:02d}_{meta['calendar_date']}.json"
    with open(path, "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    return path


def build_dataset(seed: int = 42, legacy: bool = False) -> dict:
    rng = random.Random(seed)
    ensure_dirs()

    print("=" * 60)
    print(f"[EgoTailor] Building {config.TOTAL_DAYS}-day lifelog ({config.HOURS_PER_DAY}h/day)")
    print("=" * 60)

    if legacy:
        spec = None
        persona = build_legacy_persona(seed)
        portrait_path = config.PERSONA_PATH / f"portrait_{persona['persona_id']}.json"
        with open(portrait_path, "w") as f:
            json.dump(
                {k: v for k, v in persona.items() if not k.startswith("daily_")},
                f,
                indent=2,
                ensure_ascii=False,
            )
    else:
        spec = persona_generator.generate(seed)
        persona = schedule_from_quota.build_persona(seed, spec)
        st = spec.stats
        print(
            f"[Stage 1] Persona: {' + '.join(spec.clusters)} — "
            f"{st['scenario_count']:.0f} scenarios, entropy {st['entropy_bits']:.2f} bits, "
            f"{st['hours_per_day']:.2f}h/day planned"
        )
        # Guardrails are the generator's contract; surfacing them only in its
        # own CLI meant a long run could silently produce a collapsed persona.
        # Past ~35 days the corpus cannot keep 25 scenarios distinct without
        # reusing clips, and entropy quietly falls through the floor.
        for problem in persona_generator.check(spec):
            print(f"[Stage 1] GUARDRAIL: {problem}")
        portrait_path = config.PERSONA_PATH / f"portrait_{persona['persona_id']}.json"
        write_portrait(spec, portrait_path)
    print(f"[Stage 1] Portrait: {portrait_path}")

    videos = retrieve_videos.load_video_library()
    print(f"[Stage 2] Loaded {len(videos)} Ego4D clips from {config.EGO4DINFO_PATH / 'video_info.json'}")

    global_used: set[str] = set()
    all_days = []
    aggregate_duration = 0.0
    aggregate_clips = 0
    aggregate_slots = 0
    all_scenes: dict[str, int] = {}
    all_tiers: dict[str, int] = {}

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
        aggregate_slots += day_log["statistics"]["plan_slot_count"]
        for scene, cnt in day_log["statistics"]["scene_distribution"].items():
            all_scenes[scene] = all_scenes.get(scene, 0) + cnt
        for tier, cnt in day_log["statistics"]["retrieval_tier_counts"].items():
            all_tiers[tier] = all_tiers.get(tier, 0) + cnt

        write_day_plan(day_log, persona["persona_id"])

        stats = day_log["statistics"]
        theme = day_log["metadata"].get("day_theme", "")
        anomalies = day_log["metadata"].get("anomaly_events", [])
        anomaly_str = f" anomalies={anomalies}" if anomalies else ""
        print(
            f"  Day {day_idx + 1:2d} ({day_log['metadata']['calendar_date']} "
            f"{day_log['metadata']['day_of_week']:9s}) [{theme}]: "
            f"{stats['clip_count']:3d}/{stats['plan_slot_count']:2d} slots "
            f"({stats['slot_fill_rate']:.0%}), {stats['total_duration_hours']:.2f}h, "
            f"tiers={stats['retrieval_tier_counts']}{anomaly_str}"
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
            "gen_way": "rule_based_legacy" if legacy else "corpus_driven",
            "quota_source": persona.get("quota_source", "generation/schedule_templates.py"),
            "min_retrieval_tier": config.MIN_RETRIEVAL_TIER,
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
            "total_plan_slots": aggregate_slots,
            "slot_fill_rate": round(aggregate_clips / aggregate_slots, 4) if aggregate_slots else 0.0,
            "retrieval_tier_counts": all_tiers,
            "total_duration_seconds": aggregate_duration,
            "total_duration_hours": round(aggregate_duration / 3600, 2),
            "avg_hours_per_day": round(aggregate_duration / 3600 / config.TOTAL_DAYS, 2),
            "scene_distribution": all_scenes,
            "unique_video_uids": len({m["video_uid"] for d in all_days for m in d["memory_content"]}),
        },
    }

    lifelog_path = config.LIFELOG_PATH / f"lifelog_{persona['persona_id']}_{config.TOTAL_DAYS}d.json"
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
    print(
        f"  Daily plans: {config.DAYS_PATH / persona['persona_id']}/day_NN_<date>.json "
        f"({config.TOTAL_DAYS} files)"
    )
    print("=" * 60)
    return lifelog


if __name__ == "__main__":
    args = sys.argv[1:]
    positional = [a for a in args if not a.startswith("--")]
    build_dataset(
        seed=int(positional[0]) if positional else 42,
        legacy="--legacy" in args,
    )
