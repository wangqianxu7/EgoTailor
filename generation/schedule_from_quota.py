#!/usr/bin/env python3
"""Narrative day structure driven by the corpus quota.

The old ``schedule_templates.py`` wrote schedules directly in scenarios, which
is how it ended up asking for ``Bus`` and ``Video call`` 23 and 13 times against
a supply of zero. This module keeps that file's structural instincts — a day has
a shape, weekdays differ from weekends, some days go wrong — but writes the
shape in *roles* (``hygiene``, ``cook``, ``desk``, ``social``) and lets the
quota decide which scenario fills each role.

Two independent knobs, which is the whole point:
  - change the quota       -> changes what the person does, and how often
  - edit DAY_SHAPES here   -> changes when it happens and in what order

Quota is conserved end to end: the weekly-quota instances are dealt across
the run and then *ordered*, never invented. A day cannot exceed what the pool
can serve, no matter what the skeleton asks for.

Where the quota comes from
--------------------------
By default, ``persona_generator.generate(seed)`` derives it from the corpus, so
a new seed produces a different person — a baker, a mechanic, a grad student —
and this module never notices the difference. ``--handwritten`` restores the
hand-tuned ``persona_quota.WEEK_QUOTA`` for comparison; both arrive here as the
same ``PersonaSpec`` shape.

  python -m generation.schedule_from_quota [seed] [--handwritten]
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from generation import config, persona_generator, persona_quota, retrieve_videos

# A band is a stretch of the day that accepts certain roles.
#   (name, start_min, end_min, time_period, roles_in_priority_order, max_slots)
# time_period must match the quota row's, so daytime work can never land in a
# band the corpus only has night footage for.
Band = tuple[str, int, int, str, tuple[str, ...], int]


def _m(h: int, m: int = 0) -> int:
    return h * 60 + m


WEEKDAY_SHAPE: list[Band] = [
    ("morning_rise", _m(6, 50), _m(7, 50), "daytime", ("hygiene", "coffee"), 2),
    ("breakfast", _m(7, 50), _m(9, 0), "daytime", ("cook", "meal"), 2),
    ("work_morning", _m(9, 0), _m(12, 0), "daytime", ("desk", "trade", "move", "chore"), 5),
    ("lunch", _m(12, 0), _m(13, 15), "daytime", ("cook", "meal"), 2),
    ("afternoon", _m(13, 15), _m(17, 0), "daytime",
     ("desk", "trade", "craft", "chore", "kids", "errand", "outdoor", "move", "exercise", "game"), 8),
    ("dusk", _m(17, 0), _m(19, 0), "twilight", ("errand", "move", "outdoor", "exercise"), 2),
    ("dinner", _m(19, 0), _m(20, 15), "nighttime", ("cook", "meal"), 3),
    ("evening", _m(20, 15), _m(23, 15), "nighttime",
     ("social", "wind_down", "game", "craft", "chore"), 8),
]

WEEKEND_SHAPE: list[Band] = [
    ("morning_rise", _m(8, 0), _m(9, 0), "daytime", ("hygiene", "coffee"), 2),
    ("brunch", _m(9, 0), _m(10, 30), "daytime", ("cook", "meal"), 3),
    ("midday", _m(10, 30), _m(13, 0), "daytime", ("craft", "outdoor", "kids", "chore", "exercise", "game"), 6),
    ("lunch", _m(13, 0), _m(14, 15), "daytime", ("cook", "meal"), 2),
    ("afternoon", _m(14, 15), _m(17, 0), "daytime",
     ("craft", "outdoor", "errand", "kids", "move", "desk", "trade", "chore", "exercise", "game"), 8),
    ("dusk", _m(17, 0), _m(19, 0), "twilight", ("move", "errand", "outdoor", "exercise"), 2),
    ("dinner", _m(19, 0), _m(20, 30), "nighttime", ("cook", "meal"), 3),
    ("evening", _m(20, 30), _m(23, 30), "nighttime",
     ("social", "game", "wind_down", "craft", "chore"), 8),
]

# Anomalies rewrite the shape rather than the quota — a rainy day still has to
# eat the same meals, it just cannot spend them in the garden.
ANOMALY_SHAPES: dict[str, list[Band]] = {
    # Rain: outdoor roles are unreachable, indoor craft/chore absorb the hours.
    "rain_day": [
        ("morning_rise", _m(7, 10), _m(8, 20), "daytime", ("hygiene", "coffee"), 2),
        ("breakfast", _m(8, 20), _m(9, 20), "daytime", ("cook", "meal"), 2),
        ("work_morning", _m(9, 20), _m(12, 15), "daytime", ("desk", "trade", "chore", "move"), 5),
        ("lunch", _m(12, 15), _m(13, 30), "daytime", ("cook", "meal"), 2),
        ("indoor_afternoon", _m(13, 30), _m(17, 30), "daytime",
         ("craft", "desk", "trade", "chore", "kids", "outdoor", "errand", "move", "exercise", "game"), 9),
        ("dusk_indoors", _m(17, 30), _m(19, 0), "twilight", ("move", "errand", "outdoor", "exercise"), 2),
        ("dinner", _m(19, 0), _m(20, 15), "nighttime", ("cook", "meal"), 3),
        ("evening", _m(20, 15), _m(23, 15), "nighttime",
         ("wind_down", "craft", "social", "game", "chore"), 8),
    ],
    # A deadline: desk work eats the afternoon and pushes into the night.
    "deadline_crunch": [
        ("morning_rise", _m(6, 40), _m(7, 40), "daytime", ("hygiene", "coffee"), 2),
        ("breakfast", _m(7, 40), _m(8, 30), "daytime", ("cook", "meal"), 2),
        ("work_morning", _m(8, 30), _m(12, 0), "daytime", ("desk", "trade", "move", "chore"), 6),
        ("lunch", _m(12, 0), _m(13, 0), "daytime", ("cook", "meal"), 2),
        ("work_afternoon", _m(13, 0), _m(17, 30), "daytime",
         ("desk", "trade", "chore", "move", "craft", "kids", "errand", "outdoor", "exercise", "game"), 8),
        ("dusk", _m(17, 30), _m(19, 0), "twilight", ("errand", "move", "outdoor", "exercise"), 2),
        ("late_dinner", _m(19, 30), _m(20, 30), "nighttime", ("cook", "meal"), 3),
        ("work_night", _m(20, 30), _m(23, 15), "nighttime",
         ("desk", "trade", "wind_down", "chore", "craft", "social", "game"), 7),
    ],
    # People over: cooking and company take the whole evening.
    "guests_over": [
        ("morning_rise", _m(7, 30), _m(8, 30), "daytime", ("hygiene", "coffee"), 2),
        ("breakfast", _m(8, 30), _m(9, 30), "daytime", ("cook", "meal"), 2),
        ("prep_morning", _m(9, 30), _m(12, 30), "daytime",
         ("chore", "craft", "errand", "desk", "trade", "move", "game"), 6),
        ("lunch", _m(12, 30), _m(13, 45), "daytime", ("cook", "meal"), 2),
        ("prep_afternoon", _m(13, 45), _m(17, 0), "daytime",
         ("cook", "chore", "errand", "kids", "craft", "outdoor", "move", "exercise", "game"), 8),
        ("dusk", _m(17, 0), _m(19, 0), "twilight", ("errand", "move", "outdoor", "exercise"), 2),
        ("dinner_party", _m(19, 0), _m(20, 45), "nighttime", ("cook", "meal", "social"), 4),
        ("evening_long", _m(20, 45), _m(23, 30), "nighttime",
         ("social", "game", "wind_down", "craft", "chore"), 8),
    ],
    # A day out: errands and walking replace the desk entirely.
    "day_out": [
        ("morning_rise", _m(7, 20), _m(8, 20), "daytime", ("hygiene", "coffee"), 2),
        ("breakfast", _m(8, 20), _m(9, 15), "daytime", ("cook", "meal"), 2),
        ("out_morning", _m(9, 15), _m(12, 30), "daytime",
         ("errand", "move", "outdoor", "kids", "exercise"), 6),
        ("lunch_out", _m(12, 30), _m(13, 45), "daytime", ("meal", "cook"), 2),
        ("out_afternoon", _m(13, 45), _m(17, 0), "daytime",
         ("move", "outdoor", "errand", "kids", "craft", "chore", "desk", "trade", "exercise", "game"), 8),
        ("dusk_return", _m(17, 0), _m(19, 0), "twilight", ("errand", "move", "outdoor", "exercise"), 2),
        ("dinner", _m(19, 0), _m(20, 15), "nighttime", ("cook", "meal"), 3),
        ("evening", _m(20, 15), _m(23, 0), "nighttime",
         ("wind_down", "social", "craft", "game", "chore"), 7),
    ],
    # Under the weather: low intensity, long evening, no outdoor work.
    "low_energy_day": [
        ("late_rise", _m(8, 30), _m(9, 40), "daytime", ("hygiene", "coffee"), 2),
        ("light_breakfast", _m(9, 40), _m(10, 40), "daytime", ("cook", "meal"), 2),
        ("quiet_morning", _m(10, 40), _m(12, 40), "daytime", ("wind_down", "craft", "chore", "game"), 4),
        ("lunch", _m(12, 40), _m(13, 45), "daytime", ("cook", "meal"), 2),
        ("rest_afternoon", _m(13, 45), _m(17, 30), "daytime",
         ("wind_down", "craft", "chore", "desk", "trade", "kids", "move", "errand", "outdoor", "exercise", "game"), 8),
        ("dusk_short", _m(17, 30), _m(19, 0), "twilight", ("move", "errand", "outdoor", "exercise"), 2),
        ("dinner", _m(19, 0), _m(20, 15), "nighttime", ("cook", "meal"), 3),
        ("early_evening", _m(20, 15), _m(23, 0), "nighttime",
         ("wind_down", "social", "craft", "game", "chore"), 7),
    ],
}

# day_index -> (theme, human-readable events). Mirrors the cadence of the old
# ANOMALY_DAYS: roughly one disrupted day a week, never two in a row.
ANOMALY_DAYS: dict[int, tuple[str, list[str]]] = {
    2: ("rain_day", ["rain_cancelled_outdoor"]),
    7: ("deadline_crunch", ["work_deadline_late_night"]),
    10: ("guests_over", ["friends_over_for_dinner"]),
    15: ("day_out", ["all_day_errands_and_walking"]),
    18: ("low_energy_day", ["under_the_weather"]),
}

WEEKDAY_THEMES = {
    0: "monday_desk_heavy",
    1: "tuesday_craft_afternoon",
    2: "wednesday_midweek",
    3: "thursday_desk_heavy",
    4: "friday_wind_down",
}
WEEKEND_THEMES = {5: "saturday_garden_and_making", 6: "sunday_slow_and_social"}

def _slug(scenario: str) -> str:
    head = scenario.split("(")[0].split("/")[0].strip().lower()
    return "_".join(head.replace("-", " ").split())[:24]


def seatable_role_periods() -> set[tuple[str, str]]:
    """Every (role, period) any day shape can seat.

    The skeleton's side of the contract. ``assign_to_bands`` requires an
    instance's period to match its band's, so a quota row whose (role, period)
    appears nowhere here is unplaceable by construction — dealt every week and
    dropped every week, while the persona description still claims it.

    This is the mirror image of the bug that started the redesign. The old
    ``schedule_templates`` asked the corpus for scenarios it did not have; a
    quota generated in ignorance of the skeleton supplies slots the day has no
    room for. Both end as a plan that lies about itself, so the generator reads
    this set and skips those cells.
    """
    seatable: set[tuple[str, str]] = set()
    for shape in [WEEKDAY_SHAPE, WEEKEND_SHAPE, *ANOMALY_SHAPES.values()]:
        for _, _, _, period, roles, _ in shape:
            seatable.update((role, period) for role in roles)
    return seatable


class HandwrittenSpec:
    """Adapter presenting ``persona_quota`` through the generated spec's shape.

    Kept so the hand-tuned table stays runnable and comparable — it is the
    reference the generator was built against, and a quota you can diff against
    a known-good one is worth more than a quota you can only admire.
    """

    persona_id = "egotailor_enfp"
    country_filter = persona_quota.POOL_COUNTRY_FILTER
    clusters = ["handwritten"]
    weekday_bias = persona_quota.WEEKDAY_BIAS

    def __init__(self) -> None:
        self.seed = -1
        self.week_quota = persona_quota.WEEK_QUOTA
        self.cells = []
        self.narrative = {
            "mbti_type": config.PERSONA_MBti,
            "character_traits": [
                "makes things with their hands",
                "feeds people as a love language",
                "sociable in the evening, solitary in the afternoon",
                "keeps a garden and a dog",
            ],
            "hobbies": [
                "cooking and baking",
                "knitting, sewing, drawing",
                "gardening and yardwork",
                "board games and cards with friends",
                "playing an instrument",
            ],
            "lifestyle": (
                "Works from home and spends most of the day around the house: cooking, "
                "making things by hand, keeping the place and the garden in order. "
                "Weekdays lean on desk work; weekends belong to the craft table and the "
                "garden. Evenings are the social half of the day — family, friends, "
                "board games, or a quiet hour with music and a book."
            ),
            "daily_routine": [
                "Weekday: rise ~07:00, breakfast, desk work through the morning",
                "Midday: cook and eat, then craft, chores or errands",
                "Weekend: later rise, brunch, long stretches in the garden or at the craft table",
                "Dusk: a walk or a drive as the light goes",
                "Evening: dinner, then family, friends, games, TV, reading or music",
            ],
        }
        self.stats = {}

    def role_of(self, scenario: str) -> str:
        return persona_quota.role_of(scenario)

    def phrase(self, scenario: str, time_period: str) -> str:
        return persona_generator.phrase_for(scenario, time_period)


def median_durations(spec: Any) -> dict[tuple[str, str, str], float]:
    """Median candidate duration (minutes) per quota row.

    The generated spec already measured this while sizing the quota, so reuse
    it — recomputing invites the two to drift apart, and a schedule laid out
    against different durations than the quota was budgeted with is exactly the
    kind of silent inconsistency this rewrite exists to remove. Only the
    hand-written table, which carries no measurements, needs the pool scan.
    """
    if getattr(spec, "cells", None):
        return {
            (c.scenario, c.location, c.time_period): c.median_min for c in spec.cells
        }

    pool = [
        v
        for v in retrieve_videos.load_video_library()
        if config.MIN_VIDEO_DURATION <= v["video_duration"] <= config.MAX_VIDEO_DURATION
        and (spec.country_filter is None or v["video_source"] == spec.country_filter)
    ]
    out: dict[tuple[str, str, str], float] = {}
    for scenarios, location, time_period, _ in spec.week_quota:
        cands = [
            v
            for v in pool
            if retrieve_videos.scenario_iou(scenarios, v["video_scenarios"])
            >= config.IOU_THRESHOLD
            and v["main_scene"] in (location, "mixed")
            and v["time_period"] in (time_period, "not know")
        ]
        durs = sorted(v["video_duration"] for v in cands)
        out[(scenarios[0], location, time_period)] = (
            durs[len(durs) // 2] / 60 if durs else 15.0
        )
    return out


def deal_week(
    spec: Any, rng: random.Random, medians: dict[tuple[str, str, str], float]
) -> list[list[dict[str, Any]]]:
    """Deal one week of quota across 7 days, weighted by each role's day bias.

    Weekday index 0-4 = Mon-Fri, 5-6 = Sat/Sun (START_DATE is a Monday).
    """
    week: list[list[dict[str, Any]]] = [[] for _ in range(7)]
    for scenarios, location, time_period, per_week in spec.week_quota:
        role = spec.role_of(scenarios[0])
        bias = spec.weekday_bias.get(role, 0.5)
        # Per-day preference, not a weekday/weekend mass split. Splitting mass
        # made 0.5 mean "half the instances into two weekend days", i.e. each
        # weekend day 2.5x likelier than each weekday — so every nominally
        # neutral role (meals, chores, cooking) quietly piled onto Saturday and
        # Sunday. As a per-day weight, 0.5 is genuinely uniform.
        weights = [bias] * 5 + [1.0 - bias] * 2
        for _ in range(per_week):
            # Prefer the lightest day among a weighted sample, so a scenario
            # never piles three instances onto one day by chance. Balancing on
            # (role, period) as well as scenario matters because the bands
            # consume that pair: `Cooking daytime` and `Cooking nighttime` are
            # separate quota rows, so counting scenarios alone happily gave one
            # day three dinners to cook and no breakfast.
            picks = rng.choices(range(7), weights=weights, k=3)
            day = min(picks, key=lambda d: (
                sum(1 for i in week[d] if i["scenarios"][0] == scenarios[0]),
                sum(
                    1
                    for i in week[d]
                    if i["role"] == role and i["time_period"] == time_period
                ),
                len(week[d]),
            ))
            week[day].append(
                {
                    "scenarios": scenarios,
                    "location": location,
                    "time_period": time_period,
                    "role": role,
                    "duration_min": medians.get((scenarios[0], location, time_period), 15.0),
                }
            )
    return week


def shape_for(day_index: int) -> tuple[list[Band], str, list[str]]:
    if day_index in ANOMALY_DAYS:
        theme, events = ANOMALY_DAYS[day_index]
        return ANOMALY_SHAPES[theme], theme, list(events)
    weekday = day_index % 7
    if weekday >= 5:
        return WEEKEND_SHAPE, WEEKEND_THEMES[weekday], []
    return WEEKDAY_SHAPE, WEEKDAY_THEMES[weekday], []


def _next_for_band(
    pool: list[dict[str, Any]],
    taken: list[dict[str, Any]],
    roles: tuple[str, ...],
    period: str,
) -> dict[str, Any] | None:
    """The instance this band should take next, or None.

    Prefers a role the band has taken *least* of so far, breaking ties by the
    band's declared role priority. Straight priority order lets the first role
    eat the band: the dinner band asks for ``("cook", "meal")`` and three
    cooking instances would fill it end to end, so the day cooked dinner three
    times and never ate it.
    """
    rank = {role: i for i, role in enumerate(roles)}
    used: dict[str, int] = defaultdict(int)
    for inst in taken:
        used[inst["role"]] += 1
    cands = [i for i in pool if i["time_period"] == period and i["role"] in rank]
    if not cands:
        return None
    return min(cands, key=lambda i: (used[i["role"]], rank[i["role"]]))


def assign_to_bands(
    instances: list[dict[str, Any]], shape: list[Band], rng: random.Random
) -> list[tuple[Band, list[dict[str, Any]]]]:
    """Place instances into bands, one round at a time.

    Every pass offers each band at most one more instance, so bands competing
    for the same (role, period) split it rather than racing for it. Filling
    each band to capacity before moving on — the obvious implementation, and
    the one this replaces — starves whatever comes later in the day: breakfast
    and lunch both want daytime ``cook``/``meal``, and breakfast, being first,
    took all of it. Eleven of twenty-one days had no lunch.

    Anything no band can seat is dropped rather than forced somewhere it does
    not belong. A dropped instance is honest; a misplaced one is the same bug
    this whole redesign exists to kill.
    """
    pool = list(instances)
    rng.shuffle(pool)
    chosen: list[list[dict[str, Any]]] = [[] for _ in shape]

    progress = True
    while progress:
        progress = False
        for i, band in enumerate(shape):
            _, _, _, band_period, roles, max_slots = band
            if len(chosen[i]) >= max_slots:
                continue
            pick = _next_for_band(pool, chosen[i], roles, band_period)
            if pick is None:
                continue
            pool.remove(pick)
            chosen[i].append(pick)
            progress = True

    # Spill pass: one instance over capacity is better than throwing quota away,
    # but only after every band has had its fair share above.
    for i, band in enumerate(shape):
        _, _, _, band_period, roles, max_slots = band
        while len(chosen[i]) < max_slots + 1:
            pick = _next_for_band(pool, chosen[i], roles, band_period)
            if pick is None:
                break
            pool.remove(pick)
            chosen[i].append(pick)

    return list(zip(shape, chosen))


def _order_within_band(band: Band, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order a band's items by role priority, never repeating a scenario back to back.

    Greedy skip-ahead rather than the previous insert-backwards patch, which
    could not help when the whole band was one scenario — it left
    ``Cooking, Cooking`` adjacent and only moved the third one.
    """
    _, _, _, _, roles, _ = band
    rank = {role: i for i, role in enumerate(roles)}
    remaining = sorted(items, key=lambda i: rank.get(i["role"], 99))
    out: list[dict[str, Any]] = []
    while remaining:
        pick = next(
            (
                i
                for i in remaining
                if not out or i["scenarios"][0] != out[-1]["scenarios"][0]
            ),
            remaining[0],  # everything left repeats the last one; take it anyway
        )
        remaining.remove(pick)
        out.append(pick)
    return out


def lay_out(items: list[dict[str, Any]], start: int, end: int) -> list[tuple[int, int]]:
    if not items:
        return []
    span = end - start
    total = sum(i["duration_min"] for i in items)
    scale = min(1.0, span / total) if total > 0 else 1.0
    gap = (span - total) / (len(items) + 1) if span > total else 0.0
    out: list[tuple[int, int]] = []
    cursor = start + gap
    for item in items:
        dur = item["duration_min"] * scale
        out.append((int(round(cursor)), int(round(cursor + dur))))
        cursor += dur + gap
    return out


def build_day_chunks(
    spec: Any, day_index: int, week: list[list[dict[str, Any]]], rng: random.Random
) -> tuple[list[dict[str, Any]], str, list[str]]:
    shape, theme, events = shape_for(day_index)
    instances = week[day_index % 7]
    filled = assign_to_bands(instances, shape, rng)

    jitter = rng.randint(-12, 12)
    chunks: list[dict[str, Any]] = []
    seen: dict[str, int] = defaultdict(int)

    for band, items in filled:
        name, start, end, band_period, _, _ = band
        if not items:
            continue
        items = _order_within_band(band, items)
        for item, (s, e) in zip(items, lay_out(items, start + jitter, end + jitter)):
            scenario = item["scenarios"][0]
            seen[scenario] += 1
            n = seen[scenario]
            s = max(0, min(s, 24 * 60 - 1))
            e = max(s + 1, min(e, 24 * 60 - 1))
            chunks.append(
                {
                    "slot_id": f"{_slug(scenario)}_{n}" if n > 1 else _slug(scenario),
                    "start_time": f"{s // 60:02d}:{s % 60:02d}",
                    "end_time": f"{e // 60:02d}:{e % 60:02d}",
                    "plan_chunk": spec.phrase(scenario, band_period),
                    "requested_scenarios": item["scenarios"],
                    "location": item["location"],
                    "time_period": band_period,
                    "band": name,
                    "day_theme": theme,
                    "anomaly_events": events,
                }
            )
    chunks.sort(key=lambda c: c["start_time"])
    return chunks, theme, events


def build_persona(seed: int = 42, spec: Any | None = None) -> dict[str, Any]:
    spec = spec or persona_generator.generate(seed)
    medians = median_durations(spec)
    weeks = [
        deal_week(spec, random.Random(seed + w * 7919), medians)
        for w in range(persona_generator.week_blocks())
    ]

    persona: dict[str, Any] = {
        "persona_id": spec.persona_id,
        # None = not pinned to one country; retrieve_videos skips the
        # video_source filter, keeping the whole pool in reach.
        "location": spec.country_filter,
        "gen_way": "corpus_driven_quota_with_narrative",
        "quota_source": (
            "generation/persona_quota.py"
            if isinstance(spec, HandwrittenSpec)
            else f"generation/persona_generator.py (seed {spec.seed})"
        ),
        "quota_clusters": spec.clusters,
        "quota_stats": spec.stats,
        "personality_traits": {
            "mbti_type": spec.narrative["mbti_type"],
            "character_traits": spec.narrative["character_traits"],
        },
        "lifestyle": spec.narrative["lifestyle"],
        "daily_routine": spec.narrative["daily_routine"],
        "hobbies": spec.narrative["hobbies"],
        "scheduled_anomalies": {
            str(k): {"theme": v[0], "events": v[1]} for k, v in ANOMALY_DAYS.items()
        },
        "daily_plans": {},
        "daily_plan_chunks": {},
        "day_metadata": {},
    }

    start = datetime.strptime(config.START_DATE, "%Y-%m-%d")
    for day_index in range(config.TOTAL_DAYS):
        week = weeks[day_index // 7]
        chunks, theme, events = build_day_chunks(
            spec, day_index, week, random.Random(seed + day_index * 997)
        )
        date = start + timedelta(days=day_index)
        persona["daily_plans"][str(day_index)] = [
            f"{c['plan_chunk']} at {c['start_time']} - {c['end_time']}" for c in chunks
        ]
        persona["daily_plan_chunks"][str(day_index)] = chunks
        persona["quota_offered"] = persona.get("quota_offered", 0) + len(week[day_index % 7])
        persona["day_metadata"][str(day_index)] = {
            "day_theme": theme,
            "anomaly_events": events,
            "time_variations": {},
            "day_of_week": date.strftime("%A"),
        }
    return persona


def main(argv: list[str]) -> None:
    positional = [a for a in argv if not a.startswith("--")]
    seed = int(positional[0]) if positional else 42
    spec = HandwrittenSpec() if "--handwritten" in argv else persona_generator.generate(seed)

    persona = build_persona(seed, spec)
    config.PERSONA_PATH.mkdir(parents=True, exist_ok=True)
    out = config.PERSONA_PATH / f"persona_{spec.persona_id}.json"
    with open(out, "w") as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)

    # Instances actually offered to the 21/30/N days, not the whole dealt
    # quota: the final week-block is only partly consumed when TOTAL_DAYS is
    # not a multiple of 7, and counting its unused days as "dropped" would
    # blame the schedule for arithmetic.
    quota_total = persona.get("quota_offered", 0) or 1
    slots = sum(len(v) for v in persona["daily_plan_chunks"].values())
    per_day = [len(v) for v in persona["daily_plan_chunks"].values()]
    print(f"persona: {out}")
    print(f"quota:   {' + '.join(spec.clusters)}")
    print(f"slots:   {slots}/{quota_total} of quota kept ({slots / quota_total:.1%})")
    print(f"         {slots / config.TOTAL_DAYS:.1f}/day, range {min(per_day)}-{max(per_day)}")
    themes = {d: persona["day_metadata"][str(d)]["day_theme"] for d in range(config.TOTAL_DAYS)}
    print(f"themes:  {len(set(themes.values()))} distinct, {len(ANOMALY_DAYS)} anomaly days")

    # Quota that no band could seat is dropped, by design — see assign_to_bands.
    # Reporting it here is what keeps that design honest rather than invisible.
    if slots < quota_total:
        dropped = quota_total - slots
        print(f"dropped: {dropped} instances had no band with room ({dropped / quota_total:.1%})")


if __name__ == "__main__":
    main(sys.argv[1:])
