"""Build hierarchical index (clip / hour / day / period) from lifelog JSON."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class ClipNode:
    node_id: str
    level: str = "clip"
    video_uid: str = ""
    day_index: int = 0
    calendar_date: str = ""
    slot_id: str = ""
    start_timestamp: str = ""
    end_timestamp: str = ""
    duration: float = 0.0
    main_scene: str = ""
    video_scenarios: list[str] = field(default_factory=list)
    plan_chunk: str = ""
    consolidated_summary: str = ""
    text: str = ""  # embedding / retrieval text


@dataclass
class HourNode:
    node_id: str
    level: str = "hour"
    calendar_date: str = ""
    hour: int = 0
    day_index: int = 0
    clip_ids: list[str] = field(default_factory=list)
    main_scenes: list[str] = field(default_factory=list)
    video_scenarios: list[str] = field(default_factory=list)
    start_timestamp: str = ""
    end_timestamp: str = ""
    text: str = ""


@dataclass
class DayNode:
    node_id: str
    level: str = "day"
    day_index: int = 0
    calendar_date: str = ""
    day_of_week: str = ""
    is_weekend: bool = False
    clip_count: int = 0
    total_duration_hours: float = 0.0
    scene_distribution: dict[str, int] = field(default_factory=dict)
    top_scenarios: list[str] = field(default_factory=list)
    daily_plan: list[str] = field(default_factory=list)
    clip_ids: list[str] = field(default_factory=list)
    text: str = ""


@dataclass
class PeriodNode:
    node_id: str
    level: str = "period"
    period_name: str = ""
    start_date: str = ""
    end_date: str = ""
    day_indices: list[int] = field(default_factory=list)
    clip_count: int = 0
    scene_distribution: dict[str, int] = field(default_factory=dict)
    top_scenarios: list[str] = field(default_factory=list)
    clip_ids: list[str] = field(default_factory=list)
    text: str = ""


def _clip_text(clip: dict[str, Any]) -> str:
    scenarios = ", ".join(clip.get("video_scenarios", []))
    return (
        f"[{clip.get('start_timestamp')} - {clip.get('end_timestamp')}] "
        f"scene={clip.get('main_scene')} slot={clip.get('slot_id')} "
        f"scenarios={scenarios} "
        f"plan={clip.get('plan_chunk')} "
        f"summary={clip.get('consolidated_summary')}"
    )


def load_lifelog(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def build_hierarchical_index(lifelog: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    clip_nodes: list[ClipNode] = []
    hour_buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"clips": [], "scenes": [], "scenarios": [], "day_index": 0, "date": ""}
    )
    day_nodes: list[DayNode] = []

    for day in lifelog["days"]:
        meta = day["metadata"]
        day_clip_ids: list[str] = []

        for i, clip in enumerate(day["memory_content"]):
            node_id = f"clip_{meta['day_index']:02d}_{i:03d}"
            cn = ClipNode(
                node_id=node_id,
                video_uid=clip["video_uid"],
                day_index=meta["day_index"],
                calendar_date=meta["calendar_date"],
                slot_id=clip.get("slot_id", ""),
                start_timestamp=clip["start_timestamp"],
                end_timestamp=clip["end_timestamp"],
                duration=clip["duration"],
                main_scene=clip.get("main_scene", ""),
                video_scenarios=clip.get("video_scenarios", []),
                plan_chunk=clip.get("plan_chunk", ""),
                consolidated_summary=clip.get("consolidated_summary", ""),
                text=_clip_text(clip),
            )
            clip_nodes.append(cn)
            day_clip_ids.append(node_id)

            start_dt = datetime.fromisoformat(clip["start_timestamp"])
            hour_key = f"{meta['calendar_date']}T{start_dt.hour:02d}"
            bucket = hour_buckets[hour_key]
            bucket["clips"].append(node_id)
            bucket["scenes"].append(clip.get("main_scene", ""))
            bucket["scenarios"].extend(clip.get("video_scenarios", []))
            bucket["day_index"] = meta["day_index"]
            bucket["date"] = meta["calendar_date"]
            bucket.setdefault("start_ts", clip["start_timestamp"])
            bucket["end_ts"] = clip["end_timestamp"]

        stats = day["statistics"]
        scenario_counter = Counter()
        for clip in day["memory_content"]:
            scenario_counter.update(clip.get("video_scenarios", []))

        dn = DayNode(
            node_id=f"day_{meta['day_index']:02d}",
            day_index=meta["day_index"],
            calendar_date=meta["calendar_date"],
            day_of_week=meta.get("day_of_week", ""),
            is_weekend=meta.get("is_weekend", False),
            clip_count=stats.get("clip_count", 0),
            total_duration_hours=stats.get("total_duration_hours", 0.0),
            scene_distribution=stats.get("scene_distribution", {}),
            top_scenarios=[s for s, _ in scenario_counter.most_common(8)],
            daily_plan=day.get("daily_plan", []),
            clip_ids=day_clip_ids,
            text=(
                f"Date {meta['calendar_date']} ({meta.get('day_of_week', '')}), "
                f"weekend={meta.get('is_weekend', False)}, "
                f"scenes={stats.get('scene_distribution', {})}, "
                f"top_scenarios={scenario_counter.most_common(8)}, "
                f"plan={' | '.join(day.get('daily_plan', [])[:6])}"
            ),
        )
        day_nodes.append(dn)

    hour_nodes: list[HourNode] = []
    for hour_key in sorted(hour_buckets.keys()):
        b = hour_buckets[hour_key]
        date_str, hour_str = hour_key.split("T")
        scenario_counter = Counter(b["scenarios"])
        hn = HourNode(
            node_id=f"hour_{hour_key.replace(':', '')}",
            calendar_date=date_str,
            hour=int(hour_str),
            day_index=b["day_index"],
            clip_ids=b["clips"],
            main_scenes=list(Counter(b["scenes"]).keys()),
            video_scenarios=[s for s, _ in scenario_counter.most_common(6)],
            start_timestamp=b.get("start_ts", hour_key),
            end_timestamp=b.get("end_ts", hour_key),
            text=(
                f"Hour block {hour_key}:00 on {date_str}, "
                f"clips={len(b['clips'])}, scenes={Counter(b['scenes'])}, "
                f"scenarios={scenario_counter.most_common(6)}"
            ),
        )
        hour_nodes.append(hn)

    period_nodes = _build_period_nodes(lifelog, day_nodes, clip_nodes)

    return {
        "clip": [asdict(n) for n in clip_nodes],
        "hour": [asdict(n) for n in hour_nodes],
        "day": [asdict(n) for n in day_nodes],
        "period": [asdict(n) for n in period_nodes],
    }


def _build_period_nodes(
    lifelog: dict[str, Any],
    day_nodes: list[DayNode],
    clip_nodes: list[ClipNode],
) -> list[PeriodNode]:
    periods: list[PeriodNode] = []
    days = lifelog["days"]
    start_date = lifelog["metadata"]["start_date"]

    # Week 1 / Week 2 / Week 3 windows
    week_ranges = [(0, 6, "week1"), (7, 13, "week2"), (14, 20, "week3")]
    for lo, hi, name in week_ranges:
        periods.append(_aggregate_period(name, lo, hi, days, day_nodes, clip_nodes))

    # Weekday vs weekend
    for name, is_wknd in [("weekdays", False), ("weekends", True)]:
        indices = [d.day_index for d in day_nodes if d.is_weekend == is_wknd]
        if indices:
            periods.append(
                _aggregate_period(
                    name, min(indices), max(indices), days, day_nodes, clip_nodes, day_filter=set(indices)
                )
            )

    # Full 21-day lifelog
    periods.append(_aggregate_period("full_21d", 0, 20, days, day_nodes, clip_nodes))
    return periods


def _aggregate_period(
    name: str,
    lo: int,
    hi: int,
    days: list[dict],
    day_nodes: list[DayNode],
    clip_nodes: list[ClipNode],
    day_filter: set[int] | None = None,
) -> PeriodNode:
    selected_days = [d for d in days if lo <= d["metadata"]["day_index"] <= hi]
    if day_filter is not None:
        selected_days = [d for d in selected_days if d["metadata"]["day_index"] in day_filter]

    scenario_counter: Counter = Counter()
    scene_counter: Counter = Counter()
    clip_ids: list[str] = []
    clip_count = 0
    dates: list[str] = []

    clip_by_day_idx: dict[int, list[ClipNode]] = defaultdict(list)
    for c in clip_nodes:
        clip_by_day_idx[c.day_index].append(c)

    for day in selected_days:
        idx = day["metadata"]["day_index"]
        dates.append(day["metadata"]["calendar_date"])
        for clip in day["memory_content"]:
            scenario_counter.update(clip.get("video_scenarios", []))
            scene_counter[clip.get("main_scene", "")] += 1
            clip_count += 1
        for c in clip_by_day_idx.get(idx, []):
            clip_ids.append(c.node_id)

    start_date = min(dates) if dates else ""
    end_date = max(dates) if dates else ""

    return PeriodNode(
        node_id=f"period_{name}",
        period_name=name,
        start_date=start_date,
        end_date=end_date,
        day_indices=sorted({d["metadata"]["day_index"] for d in selected_days}),
        clip_count=clip_count,
        scene_distribution=dict(scene_counter),
        top_scenarios=[s for s, _ in scenario_counter.most_common(12)],
        clip_ids=clip_ids[:50],
        text=(
            f"Period {name} ({start_date} to {end_date}): "
            f"clips={clip_count}, scenes={dict(scene_counter)}, "
            f"top_scenarios={scenario_counter.most_common(12)}"
        ),
    )


def save_index(index: dict[str, list[dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def load_index(path: Path) -> dict[str, list[dict]]:
    with open(path) as f:
        return json.load(f)
