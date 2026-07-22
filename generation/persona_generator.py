#!/usr/bin/env python3
"""Derive a persona from the corpus instead of writing one by hand.

Why this replaces the hand-written quota
----------------------------------------
``persona_quota.WEEK_QUOTA`` encodes a real insight — read what Ego4D actually
contains, then describe someone who could have produced it — but it encodes it
as 32 literal rows whose provenance lives in comments (``supply 492 cands 309``).
Those numbers were counted once, by hand, against one pool. Change
``MIN_VIDEO_DURATION``, change ``POOL_COUNTRY_FILTER``, swap in a different
``video_info.json``, and every one of them is quietly wrong while the file still
looks authoritative. A stale comment is worse than no comment: it reads as
verified.

This module computes them. Same insight, live numbers, and one thing the table
could never do — a different seed yields a different *person*, because the cast
is sampled from what the corpus supports rather than fixed at authoring time.

The split of labour
-------------------
Data decides **how much and when**: supply, solo rate, the time-of-day and
scene distribution, the median clip length, and the hard ceiling on how many
distinct clips a slot type can draw. All measured, none asserted.

``SEMANTICS`` decides **what to call it**: which narrative role a scenario
plays, which life-cluster it belongs to, how to phrase it in a plan. That is
meaning, not statistics, and no amount of counting recovers it. Keeping it in
one table makes the boundary explicit — everything below the table is derived,
everything in it is a human judgement you can argue with.

Method
------
1. Profile every scenario in the pool: supply, solo rate (share of clips where
   it appears alone — the signal for "can this carry a day by itself"), the
   time-of-day split, and per-(location, period) candidate counts computed with
   the *same* ``scenario_iou`` the retriever uses, so the ceiling is truthful.
2. Pick a cast. Anchor clusters (domestic / making / grounds / trade / study /
   fitness) are sampled by supply mass — a persona takes two or three and
   becomes a homemaker, a mechanic, or a lab rat accordingly. Tissue clusters
   (meals, hygiene, movement, social, leisure) are always present: everybody
   eats, walks and winds down.
3. Allocate weekly counts by water-filling. Weight per cell is
   ``sqrt(supply) x period_share x cluster_boost``; counts are clamped to
   ``candidates // week_blocks()`` so the run never has to reuse a clip, then repaired
   until the hour target, the max-share cap and the entropy floor all hold.

Guardrails (same ones the hand-written table carried, now enforced rather than
hoped for): no cell may be overdrawn, no single scenario may exceed
``MAX_SINGLE_SCENARIO_SHARE`` of slots, and scenario entropy must stay above
``MIN_SCENARIO_ENTROPY_BITS``. Optimising fill rate alone collapses to "all
Cooking"; these are what stop it.

Usage
-----
  python -m generation.persona_generator            # default seed, prints report
  python -m generation.persona_generator 7          # a different person
  python -m generation.persona_generator 7 --json   # dump the spec

``schedule_from_quota`` consumes the ``PersonaSpec`` this returns; the spec
exposes the same shape ``persona_quota`` did, so either can drive the schedule.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from generation import config, retrieve_videos

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# A cell needs this many distinct matching clips before it is worth scheduling.
# Below it, the row exists only to be starved or to force clip reuse.
MIN_CELL_CANDIDATES = 6

# A period is viable for a scenario when the scenario puts at least this
# fraction of the corpus-wide rate into it. Relative, not absolute, because the
# periods are wildly unequal: Ego4D is ~74% daytime, ~21% night, ~5% twilight.
# A flat threshold either erases twilight entirely or lets in four stray clips
# of "Cooking at 3am". Against the baseline, Walking on street's 23% twilight
# reads as the strong signal it is, while Cooking's 3.5% correctly does not.
PERIOD_VIABILITY_RATIO = 0.5

# Damping on supply. 0.5 (sqrt) keeps Cooking's 492 clips influential without
# letting them swallow the schedule; 1.0 would, 0.0 would ignore supply entirely.
SUPPLY_EXPONENT = 0.5

# Anchor-cluster cells get this much more weight than connective tissue. The
# spine of a day should be thicker than the threads between its vertebrae.
ANCHOR_BOOST = 1.35

# How many quota rows to keep. Comparable to the hand-written table's 32; more
# rows means thinner counts per row and a mushier character.
MAX_CELLS = 34

# Rows guaranteed to each period regardless of weight. Weight tracks clip mass,
# and twilight has almost none — sorting the whole cell list by it drops every
# dusk row and keeps a fourth indoor chore instead. Ego4D's twilight is thin,
# but a persona with literally no dusk is thinner than the data warrants.
MIN_CELLS_PER_PERIOD = {"daytime": 0, "twilight": 3, "nighttime": 8}

# Ceiling on a single row's weekly count. Without it the hour target is reached
# by repeating the shortest-median activity: Cooking's 12-minute median clips
# happily go to 28/week, i.e. four meals a day, because arithmetic has no idea
# what a day looks like.
MAX_TIMES_PER_WEEK = 14

# Stop growing once a day is this full, even if the hour target is unmet. The
# day shapes in schedule_from_quota can seat ~27 slots; quota beyond that is
# dealt out and then dropped on the floor, which reads as a fill-rate bug
# downstream rather than as the over-allocation it actually is.
MAX_SLOTS_PER_DAY = 26

# Solo rate above which a scenario can plausibly carry a stretch of day alone.
ANCHOR_SOLO_RATE = 0.55

# Guardrails. Inherited from persona_quota, now enforced in allocate().
MAX_SINGLE_SCENARIO_SHARE = 0.20
MIN_SCENARIO_ENTROPY_BITS = 4.0

# Country filter is intentionally off by default. Ego4D collects by
# country-theme, so pinning video_source trades coverage for a consistency the
# downstream VLM analysis never reads. Set to e.g. "USA" to reinstate it — the
# generator will re-derive everything against the smaller pool, which is the
# whole point of computing rather than hard-coding.
POOL_COUNTRY_FILTER: str | None = None

# Scenarios the corpus has but no persona should schedule, with the reason.
# Sleeping is the load-bearing one: it is 100% solo and looks like a perfect
# anchor, but 88% of its clips are *daytime* — they are naps, not nights. A
# generator that trusted solo rate alone would build a day around them.
EXCLUDED: dict[str, str] = {
    "Sleeping": "88% of clips are daytime naps, not nights — unschedulable as bedtime",
}


# ---------------------------------------------------------------------------
# The semantic layer — the part that cannot be counted
# ---------------------------------------------------------------------------

# Two scenario names are long enough to hurt the table's legibility.
_CONSTRUCTION = (
    "jobs related to construction/renovation company\n"
    "(Director of work, tiler, plumber, Electrician, Handyman, etc)"
)
_MAKER_LAB = (
    "Maker Lab (making items in different materials, wood plastic and also "
    "electronics), some overlap with construction etc. but benefit is all "
    "activities take place within a few rooms"
)


@dataclass(frozen=True)
class Sem:
    """What a scenario *means*: its narrative role, life-cluster, and phrasing.

    ``phrases`` is keyed by time period with ``"*"`` as the fallback, because a
    few activities read differently by light — a walk at dusk is not a walk at
    noon.
    """

    role: str
    cluster: str
    phrases: dict[str, str]


def _s(role: str, cluster: str, default: str, **by_period: str) -> Sem:
    return Sem(role, cluster, {"*": default, **by_period})


# Roles are a closed vocabulary: schedule_from_quota's day shapes are written in
# them, so adding one here without adding it to a band means those instances get
# silently dropped. Fourteen roles, same set the hand-written table used, plus
# `trade` and `exercise` for personas the old table could not express.
SEMANTICS: dict[str, Sem] = {
    # ---- domestic: the house and the people in it ----
    "Cooking": _s("cook", "domestic", "Cook a meal at home",
                  nighttime="Cook dinner", twilight="Start dinner early"),
    "Cleaning / laundry": _s("chore", "domestic", "Tidy up and do laundry",
                             nighttime="Evening tidy-up"),
    "Household management - caring for kids": _s("kids", "domestic", "Look after the kids"),
    "Fixing something in the home": _s("chore", "domestic", "Fix something around the house"),
    "Household cleaners": _s("chore", "domestic", "Deep-clean a room"),
    "Preparing hopot": _s("cook", "domestic", "Set up hotpot"),
    "Making a salad/sandwich": _s("cook", "domestic", "Put together something light"),

    # ---- making: hands, materials, a finished object ----
    "Crafting/knitting/sewing/drawing/painting": _s(
        "craft", "making", "Work on a craft project", nighttime="Wind down with handwork"),
    "Baker": _s("craft", "making", "Bake bread or pastries"),
    "Practicing a musical instrument": _s("craft", "making", "Practice an instrument"),
    _MAKER_LAB: _s("craft", "making", "Build something in the workshop"),
    "Assembling furniture": _s("craft", "making", "Assemble furniture"),
    "writing on book": _s("craft", "making", "Write by hand for a while"),
    "Assembling a puzzle": _s("game", "making", "Work on a puzzle"),

    # ---- grounds: outside the door but still home ----
    "Gardening": _s("outdoor", "grounds", "Tend the garden"),
    "Doing yardwork / shoveling snow": _s("outdoor", "grounds", "Do yardwork outside"),
    "Potting plants (indoor)": _s("outdoor", "grounds", "Repot the houseplants"),
    "Walking the dog / pet": _s("outdoor", "grounds", "Take the dog out"),
    "Playing with pets": _s("outdoor", "grounds", "Play with the animals"),
    "Washing the dog / pet, grooming horse": _s("outdoor", "grounds", "Wash and groom the dog"),
    "Going to the park": _s("outdoor", "grounds", "Head to the park"),

    # ---- trade: work that leaves the house and gets hands dirty ----
    "Car mechanic": _s("trade", "trade", "Work on a car"),
    "Scooter mechanic": _s("trade", "trade", "Work on a scooter"),
    "Bike mechanic": _s("trade", "trade", "Work on a bike"),
    "Carpenter": _s("trade", "trade", "Work at the bench"),
    _CONSTRUCTION: _s("trade", "trade", "Work on the renovation job"),
    "Farmer": _s("trade", "trade", "Work the land"),
    "Car/scooter washing": _s("trade", "trade", "Wash the vehicle down"),
    "Working in milktea shop": _s("trade", "trade", "Work a shift at the shop"),

    # ---- study: desk, screen, bench ----
    "Working at desk": _s("desk", "study", "Work at the desk from home",
                          nighttime="Late desk work"),
    "On a screen (phone/laptop)": _s("desk", "study", "Catch up on screen"),
    "biology experiments": _s("desk", "study", "Run an experiment"),
    "Labwork": _s("desk", "study", "Work in the lab"),
    "Talking to colleagues": _s("desk", "study", "Talk something through with colleagues"),
    "Participating in a meeting": _s("desk", "study", "Sit in on a meeting"),
    "Attending a TA session": _s("desk", "study", "Attend a session"),

    # ---- fitness ----
    "Working out at home": _s("exercise", "fitness", "Work out at home"),
    "Working out outside": _s("exercise", "fitness", "Train outside"),
    "Going to the gym - exercise machine, class, weights": _s("exercise", "fitness", "Go to the gym"),
    "Cycling / jogging": _s("exercise", "fitness", "Go for a run"),
    "Yoga practice": _s("exercise", "fitness", "Do a yoga session"),
    "Football": _s("exercise", "fitness", "Play football"),
    "BasketBall": _s("exercise", "fitness", "Play basketball"),
    "Golfing": _s("exercise", "fitness", "Play a round of golf"),

    # ---- tissue: table ----
    "Eating": _s("meal", "table", "Sit down for a meal", nighttime="Have dinner"),
    "Eating at a restaurant": _s("meal", "table", "Eat out"),
    "Making coffee": _s("coffee", "table", "Make coffee"),

    # ---- tissue: body ----
    "Daily hygiene": _s("hygiene", "body", "Morning hygiene routine",
                        nighttime="Get ready for bed"),

    # ---- tissue: movement ----
    "Walking on street": _s("move", "move", "Walk around the neighbourhood",
                            twilight="Evening walk as the light goes"),
    "Indoor Navigation (walking)": _s("move", "move", "Move around the house"),
    "Car - commuting, road trip": _s("errand", "move", "Drive an errand",
                                     twilight="Drive home at dusk"),
    "Bike": _s("move", "move", "Ride somewhere"),
    "Bus": _s("move", "move", "Take the bus"),
    "Skateboard/scooter": _s("move", "move", "Ride the scooter over"),

    # ---- tissue: errands ----
    "Grocery shopping indoors": _s("errand", "errand", "Pick up groceries"),
    "Clothes, other shopping": _s("errand", "errand", "Go shopping"),

    # ---- tissue: social ----
    "Talking with family members": _s("social", "social", "Catch up with family"),
    "Talking with friends/housemates": _s("social", "social", "Hang out with friends"),
    "Talking on the phone": _s("social", "social", "Take a phone call"),
    "Hosting a party": _s("social", "social", "Host people"),
    "Outdoor social (includes campfire)": _s("social", "social", "Sit outside with people"),
    "BBQ'ing/picnics": _s("social", "social", "Fire up the barbecue"),

    # ---- tissue: leisure ----
    "Watching tv": _s("wind_down", "leisure", "Watch TV",
                      daytime="Put something on to watch"),
    "Reading books": _s("wind_down", "leisure", "Read before bed",
                        daytime="Read for a while"),
    "Listening to music": _s("wind_down", "leisure", "Listen to music",
                             daytime="Put music on"),
    "Playing board games": _s("game", "leisure", "Play a board game"),
    "Playing cards": _s("game", "leisure", "Play cards"),
    "Playing games / video games": _s("game", "leisure", "Play video games"),
    "Play with cellphone": _s("wind_down", "leisure", "Scroll on the phone"),
}

# Clusters that can be a persona's spine, versus the ones everybody has.
ANCHOR_CLUSTERS = ("domestic", "making", "grounds", "trade", "study", "fitness")
TISSUE_CLUSTERS = ("table", "body", "move", "errand", "social", "leisure")

# How many *anchor* scenarios a cluster may contribute — the ones with a high
# solo rate, which read as identities rather than activities. Without this the
# generator hands one person every trade in the corpus: car mechanic and scooter
# mechanic and bike mechanic and carpenter and farmer, which is not a person.
# Domestic and grounds are uncapped because their anchors genuinely coexist —
# everyone who cooks also cleans.
CLUSTER_MAX_ANCHORS: dict[str, int] = {
    "trade": 2,      # one shop, maybe two related machines
    "making": 3,     # hobbies stack, but not indefinitely
    "study": 3,
    "fitness": 2,
    "domestic": 99,
    "grounds": 99,
}

# Which day type each role gravitates to. 1.0 = weekday only, 0.0 = weekend
# only, 0.5 = indifferent. A distribution hint for dealing the week — it never
# lets a day exceed its quota.
WEEKDAY_BIAS: dict[str, float] = {
    "hygiene": 0.6,
    "coffee": 0.6,
    "meal": 0.5,
    "cook": 0.5,
    "desk": 0.8,      # work concentrates on weekdays
    "trade": 0.85,    # a job even more so
    "chore": 0.5,
    "craft": 0.42,    # the hobby gets its longer hours at the weekend
    "outdoor": 0.4,   # garden and yard lean weekend
    "kids": 0.42,
    "errand": 0.55,
    "move": 0.5,
    "exercise": 0.5,
    "social": 0.4,    # friends come over at the weekend
    "game": 0.3,      # board games are mostly a weekend thing
    "wind_down": 0.6,
}

# One clause per cluster, used to write the persona's prose. Ordered
# (trait, hobby, routine fragment); any may be empty.
CLUSTER_VOICE: dict[str, tuple[str, str, str]] = {
    "domestic": ("feeds people as a love language", "cooking and keeping house",
                 "cooking and eating anchor the middle of the day"),
    "making": ("makes things with their hands", "craft work, baking, an instrument",
               "long stretches at the craft table"),
    "grounds": ("keeps a garden and an animal", "gardening, yardwork, the dog",
                "afternoons drift outside to the garden and the yard"),
    "trade": ("works with tools and gets their hands dirty", "tinkering with machines",
              "the working day is spent on the job, not at a desk"),
    "study": ("thinks in problems and needs a desk to do it", "reading and screen work",
              "mornings belong to focused desk work"),
    "fitness": ("moves their body daily and notices when they don't", "training and sport",
                "a training block most days"),
    "table": ("", "", ""),
    "body": ("", "", ""),
    "move": ("", "", ""),
    "errand": ("", "", ""),
    "social": ("sociable in the evening, solitary in the afternoon",
               "board games and long evenings with people", "evenings are the social half of the day"),
    "leisure": ("winds down slowly rather than all at once", "music, books, television",
                "the night ends quiet — music, a book, something on the screen"),
}


def semantics_of(scenario: str) -> Sem | None:
    return SEMANTICS.get(scenario)


def phrase_for(scenario: str, time_period: str) -> str:
    sem = SEMANTICS.get(scenario)
    if sem is None:
        return scenario.split("(")[0].strip()
    return sem.phrases.get(time_period, sem.phrases["*"])


# ---------------------------------------------------------------------------
# Stage 1 — profile the corpus
# ---------------------------------------------------------------------------


@dataclass
class ScenarioStats:
    scenario: str
    supply: int
    solo_rate: float
    period_counts: dict[str, int]      # dated clips only, for viability
    period_share: dict[str, float]

    @property
    def is_anchor(self) -> bool:
        return self.solo_rate >= ANCHOR_SOLO_RATE


def week_blocks() -> int:
    """How many weekly quotas get dealt to cover the run.

    A run of 30 days consumes four whole week-quotas plus two days of a fifth,
    so five are dealt and the last is only partly used. Ceilings divide by this
    rather than by the exact week count: erring high wastes a little supply,
    erring low means two clips get reused, and no-reuse is the property the
    whole allocation is built to guarantee.
    """
    return math.ceil(config.TOTAL_DAYS / 7)


def week_span() -> float:
    """Run length in weeks, fractional — what the hour budget is actually spread over."""
    return config.TOTAL_DAYS / 7


@dataclass
class Cell:
    """One prospective quota row, with the measurements that justify it."""

    scenario: str
    location: str
    time_period: str
    candidates: int
    median_min: float
    weight: float
    times_per_week: int = 0

    @property
    def ceiling(self) -> int:
        """Max times_per_week the run can serve without reusing a clip."""
        return self.candidates // week_blocks()

    @property
    def slots(self) -> int:
        return round(self.times_per_week * week_span())

    @property
    def minutes(self) -> float:
        return self.slots * self.median_min


def load_pool(country: str | None = POOL_COUNTRY_FILTER) -> list[dict[str, Any]]:
    return [
        v
        for v in retrieve_videos.load_video_library()
        if config.MIN_VIDEO_DURATION <= v["video_duration"] <= config.MAX_VIDEO_DURATION
        and (country is None or v["video_source"] == country)
    ]


def profile_corpus(pool: list[dict[str, Any]]) -> dict[str, ScenarioStats]:
    """Supply, solo rate and time-of-day split for every scenario in the pool.

    Solo rate is the share of a scenario's clips where it is the *only* label.
    It separates activities that can carry a stretch of day (Carpenter 98%,
    Crafting 96%) from ones that only ever texture it (Indoor Navigation 3%,
    Talking with family 3%). Scheduling a day out of pure tissue is what
    starved the original hand-written schedule.
    """
    supply: Counter[str] = Counter()
    solo: Counter[str] = Counter()
    periods: dict[str, Counter[str]] = defaultdict(Counter)

    for v in pool:
        scenarios = v["video_scenarios"]
        for s in scenarios:
            supply[s] += 1
            if v["time_period"] != "not know":
                periods[s][v["time_period"]] += 1
        if len(scenarios) == 1:
            solo[scenarios[0]] += 1

    stats: dict[str, ScenarioStats] = {}
    for s, n in supply.items():
        dated = sum(periods[s].values())
        stats[s] = ScenarioStats(
            scenario=s,
            supply=n,
            solo_rate=solo[s] / n,
            period_counts=dict(periods[s]),
            period_share={
                p: c / dated for p, c in periods[s].items()
            } if dated else {},
        )
    return stats


def _cell_candidates(
    pool: list[dict[str, Any]], scenario: str, location: str, period: str
) -> list[dict[str, Any]]:
    """Clips a slot of this shape could actually draw.

    Uses the retriever's own ``scenario_iou`` and the same location/period
    matching rules, so the ceiling this produces is the ceiling retrieval will
    hit — not an optimistic proxy for it.
    """
    return [
        v
        for v in pool
        if retrieve_videos.scenario_iou([scenario], v["video_scenarios"])
        >= config.IOU_THRESHOLD
        and v["main_scene"] in (location, "mixed")
        and v["time_period"] in (period, "not know")
    ]


def corpus_period_baseline(pool: list[dict[str, Any]]) -> dict[str, float]:
    """Share of dated clips in each period, corpus-wide.

    The reference every scenario's own split is judged against. Roughly
    daytime 0.74 / nighttime 0.21 / twilight 0.05 on the full pool.
    """
    counts = Counter(v["time_period"] for v in pool if v["time_period"] != "not know")
    total = sum(counts.values()) or 1
    return {p: c / total for p, c in counts.items()}


def build_cells(
    pool: list[dict[str, Any]],
    stats: dict[str, ScenarioStats],
    scenarios: Iterable[str],
    baseline: dict[str, float],
    seatable: set[tuple[str, str]] | None = None,
) -> list[Cell]:
    """Every viable (scenario, location, period) triple, with its weight.

    Location is chosen per cell by whichever bucket yields the most candidates
    — it is a retrieval key, not a claim about the world, and ``main_scene in
    (location, "mixed")`` means the three buckets are not symmetric.

    Viability is judged against the baseline; weight is not. Whether a period
    is *real* for a scenario is a relative question, but how many minutes it
    deserves once admitted is an absolute one — it follows the clip mass that
    actually sits there.
    """
    cells: list[Cell] = []
    for scenario in scenarios:
        st = stats[scenario]
        sem = SEMANTICS[scenario]
        boost = ANCHOR_BOOST if sem.cluster in ANCHOR_CLUSTERS else 1.0
        for period in ("daytime", "twilight", "nighttime"):
            share = st.period_share.get(period, 0.0)
            base = baseline.get(period, 0.0)
            if not base or share < PERIOD_VIABILITY_RATIO * base:
                continue
            # Quota the day skeleton cannot seat is quota that gets dealt and
            # then dropped — allocating it inflates the persona's description
            # against a schedule that never contains it.
            if seatable is not None and (sem.role, period) not in seatable:
                continue
            best: tuple[int, str, list[dict[str, Any]]] | None = None
            for location in ("indoor", "outdoor", "mixed"):
                cands = _cell_candidates(pool, scenario, location, period)
                if best is None or len(cands) > best[0]:
                    best = (len(cands), location, cands)
            assert best is not None
            count, location, cands = best
            if count < MIN_CELL_CANDIDATES:
                continue
            durs = sorted(v["video_duration"] for v in cands)
            cells.append(
                Cell(
                    scenario=scenario,
                    location=location,
                    time_period=period,
                    candidates=count,
                    median_min=durs[len(durs) // 2] / 60,
                    weight=(st.supply ** SUPPLY_EXPONENT) * share * boost,
                )
            )
    return cells


# ---------------------------------------------------------------------------
# Stage 2 — choose who this person is
# ---------------------------------------------------------------------------


def choose_clusters(stats: dict[str, ScenarioStats], rng: random.Random) -> list[str]:
    """Sample two or three anchor clusters, weighted by their supply mass.

    This is where the persona stops being one fixed character. Domestic carries
    ~790 clips so it wins often, and should — it is what Ego4D mostly is. But a
    seed that draws `trade` builds a mechanic instead, and the rest of the
    pipeline never notices the difference.
    """
    mass: Counter[str] = Counter()
    for scenario, st in stats.items():
        sem = SEMANTICS.get(scenario)
        if sem is None or sem.cluster not in ANCHOR_CLUSTERS or scenario in EXCLUDED:
            continue
        # Anchors count for more than tissue when sizing a cluster: a cluster of
        # never-solo scenarios cannot be a spine no matter how many clips it has.
        mass[sem.cluster] += st.supply * (1.0 if st.is_anchor else 0.35)

    available = [c for c in ANCHOR_CLUSTERS if mass[c] > 0]
    if not available:
        raise RuntimeError("no anchor cluster has any supply — is the pool empty?")

    k = min(rng.choice([2, 2, 3]), len(available))
    chosen: list[str] = []
    pool_ = list(available)
    for _ in range(k):
        weights = [mass[c] for c in pool_]
        pick = rng.choices(pool_, weights=weights, k=1)[0]
        chosen.append(pick)
        pool_.remove(pick)
    return chosen


def choose_cast(
    stats: dict[str, ScenarioStats], clusters: list[str], rng: random.Random
) -> list[str]:
    """Scenarios this persona may draw on.

    Tissue comes whole — everybody eats, walks and winds down. Anchor clusters
    are thinned to ``CLUSTER_MAX_ANCHORS`` identities apiece, sampled by supply,
    while their low-solo members pass through untouched: `Fixing something in
    the home` is something a person does, `Carpenter` is something a person *is*,
    and only the second kind needs rationing.
    """
    cast: list[str] = []
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for s in stats:
        if s in EXCLUDED or s not in SEMANTICS:
            continue
        by_cluster[SEMANTICS[s].cluster].append(s)

    for cluster in TISSUE_CLUSTERS:
        cast.extend(by_cluster.get(cluster, []))

    for cluster in clusters:
        members = by_cluster.get(cluster, [])
        anchors = [s for s in members if stats[s].is_anchor]
        cast.extend(s for s in members if not stats[s].is_anchor)

        limit = CLUSTER_MAX_ANCHORS.get(cluster, 3)
        if len(anchors) <= limit:
            cast.extend(anchors)
            continue
        remaining = list(anchors)
        for _ in range(limit):
            pick = rng.choices(remaining, weights=[stats[s].supply for s in remaining], k=1)[0]
            cast.append(pick)
            remaining.remove(pick)
    return cast


# ---------------------------------------------------------------------------
# Stage 3 — allocate the week
# ---------------------------------------------------------------------------


def _scenario_slots(cells: list[Cell]) -> Counter[str]:
    out: Counter[str] = Counter()
    for c in cells:
        out[c.scenario] += c.slots
    return out


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


def allocate(cells: list[Cell], target_minutes: float) -> list[Cell]:
    """Water-fill weekly counts towards the hour target, proportional to weight.

    Each step hands the next unit to whichever cell sits furthest *below* its
    weight-proportional share of minutes — largest-remainder, one unit at a
    time. Two rejected alternatives, both of which I wrote first:

      - "always feed the thinnest scenario" maximises entropy and destroys the
        character. It pushes every row to its ceiling and produces someone who
        is a car mechanic and a scooter mechanic and a carpenter and a farmer,
        each exactly eight times a week.
      - "compute shares, multiply, round" hits the hour target just as well but
        ignores the per-cell ceilings until the end, so the repair pass ends up
        doing all the real work anyway.

    The share cap still binds — it stops Cooking's 492 clips from eating the
    schedule — but it is a ceiling on the proportional result, not the
    objective.
    """
    cells = [c for c in cells if c.ceiling >= 1]
    if not cells:
        raise RuntimeError("no cell has enough candidates to fill even one slot/week")

    cells = select_cells(cells)
    for c in cells:
        c.times_per_week = 1

    total_weight = sum(c.weight for c in cells)
    slot_budget = MAX_SLOTS_PER_DAY * config.TOTAL_DAYS

    def minutes() -> float:
        return sum(c.minutes for c in cells)

    guard = 0
    while minutes() < target_minutes and guard < 10_000:
        guard += 1
        current = minutes() or 1.0
        slots = _scenario_slots(cells)
        total_slots = sum(slots.values())
        if total_slots >= slot_budget:
            break
        step = max(1, round(week_span()))  # slots one +1/week actually adds
        options = [
            c
            for c in cells
            if c.times_per_week < min(c.ceiling, MAX_TIMES_PER_WEEK)
            and (slots[c.scenario] + step) / (total_slots + step)
            <= MAX_SINGLE_SCENARIO_SHARE
        ]
        if not options:
            break
        # Deficit against the weight-proportional target, in shares of minutes.
        best = max(options, key=lambda c: c.weight / total_weight - c.minutes / current)
        best.times_per_week += 1

    return cells


def select_cells(cells: list[Cell]) -> list[Cell]:
    """Trim to MAX_CELLS by weight, after each period's reserved rows are safe.

    Thin counts spread over many rows make a character who does a little of
    everything and nothing in particular — hence the cap. But a plain
    weight-sort is biased against whole times of day, so each period claims its
    reserved rows first and the rest of the budget is contested globally.
    """
    by_period: dict[str, list[Cell]] = defaultdict(list)
    for c in cells:
        by_period[c.time_period].append(c)

    kept: list[Cell] = []
    for period, reserved in MIN_CELLS_PER_PERIOD.items():
        ranked = sorted(by_period.get(period, []), key=lambda c: c.weight, reverse=True)
        kept.extend(ranked[:reserved])

    keep_ids = {id(c) for c in kept}
    rest = sorted(
        (c for c in cells if id(c) not in keep_ids), key=lambda c: c.weight, reverse=True
    )
    kept.extend(rest[: max(0, MAX_CELLS - len(kept))])
    return kept


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


@dataclass
class PersonaSpec:
    """Everything downstream needs, with the evidence that produced it.

    Exposes the same surface ``persona_quota`` did (``week_quota``, ``role_of``,
    ``weekday_bias``, ``country_filter``) so ``schedule_from_quota`` can take
    either without caring which.
    """

    persona_id: str
    seed: int
    country_filter: str | None
    clusters: list[str]
    cells: list[Cell]
    narrative: dict[str, Any]
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def week_quota(self) -> list[tuple[list[str], str, str, int]]:
        return [
            ([c.scenario], c.location, c.time_period, c.times_per_week)
            for c in self.cells
        ]

    @property
    def weekday_bias(self) -> dict[str, float]:
        return WEEKDAY_BIAS

    def role_of(self, scenario: str) -> str:
        sem = SEMANTICS.get(scenario)
        return sem.role if sem else "misc"

    def phrase(self, scenario: str, time_period: str) -> str:
        return phrase_for(scenario, time_period)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "seed": self.seed,
            "country_filter": self.country_filter,
            "clusters": self.clusters,
            "narrative": self.narrative,
            "stats": self.stats,
            "quota": [
                {
                    "scenario": c.scenario,
                    "location": c.location,
                    "time_period": c.time_period,
                    "times_per_week": c.times_per_week,
                    "role": self.role_of(c.scenario),
                    # Provenance: measured, not asserted. If these look wrong,
                    # the pool changed — rerun rather than edit.
                    "candidates": c.candidates,
                    "ceiling_per_week": c.ceiling,
                    "median_minutes": round(c.median_min, 1),
                }
                for c in sorted(
                    self.cells, key=lambda c: (c.time_period, -c.times_per_week)
                )
            ],
        }


def narrate(clusters: list[str], cells: list[Cell]) -> dict[str, Any]:
    """Assemble the persona's prose from the clusters that actually got slots.

    Deterministic assembly, not generation: every clause is traceable to a
    cluster that survived allocation, so the description can never claim a
    hobby the schedule does not contain.
    """
    present = {SEMANTICS[c.scenario].cluster for c in cells if c.times_per_week > 0}
    ordered = [c for c in clusters if c in present] + [
        c for c in TISSUE_CLUSTERS if c in present
    ]

    traits = [CLUSTER_VOICE[c][0] for c in ordered if CLUSTER_VOICE[c][0]]
    hobbies = [CLUSTER_VOICE[c][1] for c in ordered if CLUSTER_VOICE[c][1]]
    routine = [CLUSTER_VOICE[c][2] for c in ordered if CLUSTER_VOICE[c][2]]

    by_period: Counter[str] = Counter()
    for c in cells:
        by_period[c.time_period] += c.slots
    total = sum(by_period.values()) or 1

    lifestyle = (
        "A life assembled from what the corpus can actually show: "
        + ", ".join(routine)
        + f". {by_period['daytime'] / total:.0%} of the day's slots fall in daylight, "
        f"{by_period['twilight'] / total:.0%} at dusk, "
        f"{by_period['nighttime'] / total:.0%} after dark — thin twilight is a "
        "property of Ego4D, not a choice."
    )

    return {
        "mbti_type": config.PERSONA_MBti,
        "character_traits": traits,
        "hobbies": hobbies,
        "lifestyle": lifestyle,
        "daily_routine": routine,
    }


def generate(
    seed: int = 42,
    country: str | None = POOL_COUNTRY_FILTER,
    seatable: set[tuple[str, str]] | None = None,
) -> PersonaSpec:
    """Derive a persona from the corpus.

    ``seatable`` is the set of (role, period) pairs the consuming schedule can
    actually place; pass ``schedule_from_quota.seatable_role_periods()`` to stop
    the generator budgeting hours the day has no room for. Imported lazily by
    default so this module stays runnable on its own.
    """
    if seatable is None:
        from generation import schedule_from_quota

        seatable = schedule_from_quota.seatable_role_periods()

    rng = random.Random(seed)
    pool = load_pool(country)
    stats = profile_corpus(pool)

    baseline = corpus_period_baseline(pool)
    clusters = choose_clusters(stats, rng)
    cast = choose_cast(stats, clusters, rng)
    cells = build_cells(pool, stats, cast, baseline, seatable)

    target = config.HOURS_PER_DAY * 60 * config.TOTAL_DAYS
    cells = allocate(cells, target)

    slots = _scenario_slots(cells)
    total_slots = sum(slots.values())
    minutes = sum(c.minutes for c in cells)
    entropy = _entropy(slots)
    max_share = max(slots.values()) / total_slots if total_slots else 0.0
    overdrawn = [c for c in cells if c.times_per_week > c.ceiling]

    spec = PersonaSpec(
        persona_id=f"egotailor_{'_'.join(clusters)}",
        seed=seed,
        country_filter=country,
        clusters=clusters,
        cells=cells,
        narrative={},
        stats={
            "pool_clips": len(pool),
            "quota_rows": len(cells),
            "total_slots": total_slots,
            "slots_per_day": total_slots / config.TOTAL_DAYS,
            "hours_per_day": minutes / 60 / config.TOTAL_DAYS,
            "scenario_count": len(slots),
            "entropy_bits": entropy,
            "max_scenario_share": max_share,
            "overdrawn_rows": len(overdrawn),
        },
    )
    spec.narrative = narrate(clusters, cells)
    return spec


def check(spec: PersonaSpec) -> list[str]:
    """Guardrail violations, empty if the spec is sound."""
    problems: list[str] = []
    s = spec.stats
    if s["overdrawn_rows"]:
        problems.append(f"{s['overdrawn_rows']} rows demand more clips than exist")
    if s["max_scenario_share"] > MAX_SINGLE_SCENARIO_SHARE:
        problems.append(
            f"max scenario share {s['max_scenario_share']:.1%} > "
            f"{MAX_SINGLE_SCENARIO_SHARE:.0%} — the persona is collapsing onto one activity"
        )
    if s["entropy_bits"] < MIN_SCENARIO_ENTROPY_BITS:
        problems.append(
            f"entropy {s['entropy_bits']:.2f} < {MIN_SCENARIO_ENTROPY_BITS} bits — too monotonous"
        )
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def report(spec: PersonaSpec) -> None:
    s = spec.stats
    print(f"persona:  {spec.persona_id}  (seed {spec.seed})")
    print(f"clusters: {' + '.join(spec.clusters)}")
    print(f"pool:     {int(s['pool_clips'])} clips (country filter: {spec.country_filter or 'none'})")
    print(
        f"quota:    {int(s['quota_rows'])} rows, {int(s['total_slots'])} slots over "
        f"{config.TOTAL_DAYS}d = {s['slots_per_day']:.1f}/day"
    )
    print(f"hours:    {s['hours_per_day']:.2f}/day (target {config.HOURS_PER_DAY})")
    print(
        f"spread:   {int(s['scenario_count'])} scenarios, entropy {s['entropy_bits']:.2f} bits, "
        f"max share {s['max_scenario_share']:.1%}"
    )
    print()
    print(f"{'scenario':52s} {'loc':7s} {'period':9s} {'/wk':>3s} {'cap':>4s} {'med':>5s}")
    for c in sorted(spec.cells, key=lambda c: (c.time_period, -c.times_per_week)):
        name = c.scenario.split("\n")[0][:52]
        print(
            f"{name:52s} {c.location:7s} {c.time_period:9s} "
            f"{c.times_per_week:3d} {c.ceiling:4d} {c.median_min:5.1f}"
        )
    print()
    print("traits:  " + "; ".join(spec.narrative["character_traits"]))
    print("hobbies: " + "; ".join(spec.narrative["hobbies"]))

    problems = check(spec)
    if problems:
        print()
        for p in problems:
            print(f"GUARDRAIL: {p}")
    else:
        print("\nall guardrails hold — no reuse required, spread within bounds")


def main(argv: list[str]) -> None:
    seed = 42
    positional = [a for a in argv if not a.startswith("--")]
    if positional:
        seed = int(positional[0])
    spec = generate(seed)
    if "--json" in argv:
        config.PERSONA_PATH.mkdir(parents=True, exist_ok=True)
        out = config.PERSONA_PATH / f"portrait_{spec.persona_id}.json"
        with open(out, "w") as f:
            json.dump(spec.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"wrote {out}")
    report(spec)


if __name__ == "__main__":
    main(sys.argv[1:])
