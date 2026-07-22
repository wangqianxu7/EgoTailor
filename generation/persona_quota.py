"""Corpus-driven scenario quota — the persona derived *from* Ego4D, not imposed on it.

Why this file exists
--------------------
The hand-written schedule in ``schedule_templates.py`` asks for scenarios that
the corpus barely has: ``Working at desk`` was requested 76 times against a
USA-pool supply of 2; ``Bus``, ``Video call``, ``Participating in a meeting``,
``Reading books`` and ``Working out at home`` have a USA supply of exactly 0.
Only 41% of its slots could be filled with a scenario-matched clip.

This quota inverts the direction: read what the pool actually contains, then
describe a person who could plausibly have produced it.

Method
------
1. Pool = all Ego4D clips with 120s <= duration <= 3600s, **no country filter**
   (1988 clips). Ego4D collects by country-theme, so pinning ``video_source``
   costs more than it buys — see the note at the bottom.
2. Scenarios split into two layers by *solo rate* (share of clips where the
   scenario appears without any other high-supply scenario):
     - "anchors"  — near-100% solo (Car mechanic, Farmer, Carpenter, Baker,
       Crafting 96%, construction 100%). One clip = one activity. These carry
       a day.
     - "connective tissue" — rarely solo (Indoor Navigation 3%, Talking with
       family 7%, On a screen 10%, Eating 28%). These never stand alone; they
       texture a day but cannot anchor it.
   A workable persona needs anchors for the spine and tissue for the weave.
   The old persona requested almost only tissue, which is why it starved.
3. Each entry's ``time_period`` is read off the corpus distribution rather than
   from common sense. Two places where those disagree, both load-bearing:
     - ``Sleeping`` is 88% *daytime* in Ego4D (naps, not nights) — so it is
       omitted entirely rather than scheduled at bedtime.
     - ``twilight`` is thin everywhere; only walking and commuting carry real
       twilight mass, so evening slots stay scarce (2.9% of the quota).

Who this describes
------------------
Someone who works from home, cooks a lot, makes things by hand, keeps a garden
and a dog, and has people over in the evening. ENFP survives intact — crafting,
baking, gardening and hosting board games is a coherent ENFP life. What is
dropped is the office-worker shell, which the corpus never supported.

Validation (python -m generation.persona_quota)
-----------------------------------------------
  numbers below were measured at TOTAL_DAYS = 21; rerun after changing it
  522 slots over 21 days (24.9/day), 8.21 h/day against the 8h target
  every slot has a distinct scenario-matched candidate — zero reuse required
  25 scenarios, entropy 4.19 bits, largest single scenario 17.2%

The entropy and max-share numbers are the guardrail: optimising fill rate alone
collapses to "all Cooking", so any future edit should keep max share under ~20%
and entropy above ~4 bits.

This module is data only — nothing generates schedules from it yet.
"""

from __future__ import annotations

# (scenarios, location, time_period, times_per_week)
#
# supply  = clips anywhere in the pool carrying this scenario
# cands   = clips that also match this row's location + time_period
#           (the real ceiling: times_per_week * 3 must stay <= cands for a
#            21-day run to need no clip reuse)
# median  = median candidate duration, minutes — drives the 8h/day arithmetic
WEEK_QUOTA: list[tuple[list[str], str, str, int]] = [
    # ---- anchors: high solo-rate, thick supply, these carry the day ----
    (["Cooking"], "indoor", "daytime", 20),                                    # supply 492  cands 309  ~12m
    (["Cooking"], "indoor", "nighttime", 10),                                  # supply 492  cands 189  ~13m
    (["Crafting/knitting/sewing/drawing/painting"], "indoor", "daytime", 12),   # supply 227  cands 149  ~19m
    (["Crafting/knitting/sewing/drawing/painting"], "indoor", "nighttime", 6),  # supply 227  cands  80  ~20m
    (["Cleaning / laundry"], "indoor", "daytime", 12),                          # supply 230  cands 146  ~20m
    (["Cleaning / laundry"], "indoor", "nighttime", 5),                         # supply 230  cands  78  ~16m
    (["Baker"], "indoor", "daytime", 4),                                        # supply  47  cands  40   ~9m
    (["Practicing a musical instrument"], "indoor", "daytime", 3),              # supply  35  cands  25  ~20m
    (["Gardening"], "outdoor", "daytime", 3),                                   # supply  35  cands  32  ~22m
    (["Doing yardwork / shoveling snow"], "outdoor", "daytime", 3),             # supply  34  cands  25  ~23m
    (["Household management - caring for kids"], "outdoor", "daytime", 2),      # supply  64  cands  31  ~12m

    # ---- connective tissue, daytime ----
    (["Eating"], "indoor", "daytime", 8),                                       # supply  78  cands  53  ~21m
    (["Daily hygiene"], "indoor", "daytime", 4),                                # supply  19  cands  13  ~15m
    (["Making coffee"], "indoor", "daytime", 4),                                # supply  16  cands  15  ~31m
    (["On a screen (phone/laptop)"], "indoor", "daytime", 5),                   # supply  42  cands  22  ~15m
    (["Working at desk"], "indoor", "daytime", 9),                              # supply  57  cands  41  ~20m
    (["Indoor Navigation (walking)"], "mixed", "daytime", 3),                   # supply  39  cands  10  ~20m
    (["Walking on street"], "mixed", "daytime", 6),                             # supply  65  cands  24  ~18m
    (["Grocery shopping indoors"], "mixed", "daytime", 2),                      # supply  18  cands   9  ~21m
    (["Walking the dog / pet"], "outdoor", "daytime", 4),                       # supply  21  cands  18  ~30m
    (["Car - commuting, road trip"], "mixed", "daytime", 4),                    # supply  44  cands  19  ~18m

    # ---- twilight: thin by nature; only walking/commuting have real mass ----
    (["Walking on street"], "mixed", "twilight", 3),                            # supply  65  cands  12  ~28m
    (["Car - commuting, road trip"], "mixed", "twilight", 2),                   # supply  44  cands   6  ~30m

    # ---- evening: corpus puts 50-62% of these scenarios at night ----
    (["Eating"], "indoor", "nighttime", 6),                                     # supply  78  cands  22  ~20m
    (["Talking with family members"], "indoor", "nighttime", 8),                # supply  60  cands  32  ~24m
    (["Talking with friends/housemates"], "indoor", "nighttime", 6),            # supply  46  cands  25  ~29m
    (["Watching tv"], "indoor", "nighttime", 7),                                # supply  37  cands  27  ~29m
    (["Reading books"], "indoor", "nighttime", 5),                              # supply  38  cands  20  ~23m
    (["Listening to music"], "indoor", "nighttime", 3),                         # supply  26  cands  13  ~21m
    (["Playing board games"], "indoor", "nighttime", 3),                        # supply  24  cands  11  ~26m
    (["Playing cards"], "indoor", "nighttime", 2),                              # supply  21  cands   9  ~19m
]

# Narrative role per scenario. The schedule skeleton is written in roles, not
# scenarios, so the story ("morning routine, then a work block, then lunch")
# stays separate from what the corpus can actually supply. Editing WEEK_QUOTA
# changes what fills a role; editing the skeleton changes when roles happen.
ROLE_OF: dict[str, str] = {
    "Daily hygiene": "hygiene",
    "Making coffee": "coffee",
    "Eating": "meal",
    "Cooking": "cook",
    "Working at desk": "desk",
    "On a screen (phone/laptop)": "desk",
    "Cleaning / laundry": "chore",
    "Crafting/knitting/sewing/drawing/painting": "craft",
    "Practicing a musical instrument": "craft",
    "Baker": "craft",
    "Gardening": "outdoor",
    "Doing yardwork / shoveling snow": "outdoor",
    "Walking the dog / pet": "outdoor",
    "Household management - caring for kids": "kids",
    "Grocery shopping indoors": "errand",
    "Car - commuting, road trip": "errand",
    "Walking on street": "move",
    "Indoor Navigation (walking)": "move",
    "Talking with family members": "social",
    "Talking with friends/housemates": "social",
    "Playing board games": "game",
    "Playing cards": "game",
    "Watching tv": "wind_down",
    "Reading books": "wind_down",
    "Listening to music": "wind_down",
}

# Which day type each role gravitates to when the week is dealt out.
# 1.0 = weekday only, 0.0 = weekend only, 0.5 = indifferent. Purely a
# distribution hint — it never lets a day exceed its quota.
WEEKDAY_BIAS: dict[str, float] = {
    "hygiene": 0.6,
    "coffee": 0.6,
    "meal": 0.5,
    "cook": 0.5,
    "desk": 0.8,     # work-from-home concentrates on weekdays
    "chore": 0.5,
    "craft": 0.42,   # the hobby gets its longer hours at the weekend
    "outdoor": 0.4,  # garden and yard lean weekend
    "kids": 0.42,
    "errand": 0.55,
    "move": 0.5,
    "social": 0.4,   # friends come over at the weekend
    "game": 0.3,     # board games are mostly a weekend thing
    "wind_down": 0.6,
}


def role_of(scenario: str) -> str:
    return ROLE_OF.get(scenario, "misc")


# Scenarios the pool is rich in but this persona deliberately skips, kept here
# so a future pass can reconsider rather than rediscover them. Each is a
# ~100%-solo occupational cluster: taking one on turns the persona into a
# tradesperson, which is a different (equally valid) character.
UNUSED_RICH_ANCHORS = {
    "jobs related to construction/renovation company": 102,
    "Carpenter": 65,
    "Car mechanic": 61,
    "Scooter mechanic": 46,
    "Farmer": 39,
    "Bike mechanic": 29,
}

# Guardrails for any future edit to WEEK_QUOTA. Optimising fill rate without
# these collapses the persona into a single high-supply scenario.
MAX_SINGLE_SCENARIO_SHARE = 0.20
MIN_SCENARIO_ENTROPY_BITS = 4.0

# Country filter is intentionally absent. Ego4D collects by country-theme, so
# pinning video_source trades coverage for a consistency the downstream VLM
# analysis never reads. For reference, against the *old* persona:
#   ALL 1988 clips -> 100%  |  KSA 333 -> 93.8%  |  USA 522 -> 91.0%
# USA holds the office-worker scenarios this persona no longer needs, and KSA
# is the pool that actually suits a desk-bound persona (Working at desk: 52 in
# KSA vs 2 in USA). Reinstate a filter only if geographic consistency becomes a
# stated requirement.
POOL_COUNTRY_FILTER: str | None = None


def validate(verbose: bool = True) -> dict[str, float]:
    """Check the quota against the live pool. Run after editing WEEK_QUOTA."""
    import collections
    import math

    from generation import config, persona_generator, retrieve_videos

    # Week-quotas needed to cover the run. Was hard-coded to 3, which silently
    # under-reported the moment TOTAL_DAYS stopped being 21: at 30 days it
    # claimed 522 slots over 30d (17.4/day, 5.75h/day) when the table actually
    # deals 5 week-blocks.
    blocks = persona_generator.week_blocks()

    pool = [
        v
        for v in retrieve_videos.load_video_library()
        if config.MIN_VIDEO_DURATION <= v["video_duration"] <= config.MAX_VIDEO_DURATION
        and (POOL_COUNTRY_FILTER is None or v["video_source"] == POOL_COUNTRY_FILTER)
    ]

    total_slots = sum(w for _, _, _, w in WEEK_QUOTA) * blocks
    per_scenario: dict[str, int] = collections.Counter()
    duration = 0.0
    overdrawn: list[tuple[str, int, int]] = []

    for scenarios, location, time_period, per_week in WEEK_QUOTA:
        demand = per_week * blocks
        cands = [
            v
            for v in pool
            if retrieve_videos.scenario_iou(scenarios, v["video_scenarios"]) >= config.IOU_THRESHOLD
            and v["main_scene"] in (location, "mixed")
            and v["time_period"] in (time_period, "not know")
        ]
        if demand > len(cands):
            overdrawn.append((scenarios[0], demand, len(cands)))
        if cands:
            durs = sorted(v["video_duration"] for v in cands)
            duration += demand * durs[len(durs) // 2]
        per_scenario[scenarios[0]] += demand

    entropy = -sum(
        (n / total_slots) * math.log2(n / total_slots) for n in per_scenario.values()
    )
    max_share = max(per_scenario.values()) / total_slots
    report = {
        "total_slots": total_slots,
        "slots_per_day": total_slots / config.TOTAL_DAYS,
        "hours_per_day": duration / 3600 / config.TOTAL_DAYS,
        "scenario_count": len(per_scenario),
        "entropy_bits": entropy,
        "max_scenario_share": max_share,
        "overdrawn": len(overdrawn),
    }

    if verbose:
        print(f"pool: {len(pool)} clips (country filter: {POOL_COUNTRY_FILTER or 'none'})")
        print(f"slots: {total_slots} over {config.TOTAL_DAYS}d = {report['slots_per_day']:.1f}/day")
        print(f"hours: {report['hours_per_day']:.2f}/day (target {config.HOURS_PER_DAY})")
        print(f"scenarios: {len(per_scenario)}  entropy: {entropy:.2f} bits  max share: {max_share:.1%}")
        if overdrawn:
            print("OVERDRAWN — these need clip reuse:")
            for name, demand, avail in overdrawn:
                print(f"  {name[:50]:50s} demand {demand} > candidates {avail}")
        else:
            print("no reuse required — every slot can draw a distinct clip")
        if max_share > MAX_SINGLE_SCENARIO_SHARE:
            print(f"GUARDRAIL: max share {max_share:.1%} > {MAX_SINGLE_SCENARIO_SHARE:.0%}")
        if entropy < MIN_SCENARIO_ENTROPY_BITS:
            print(f"GUARDRAIL: entropy {entropy:.2f} < {MIN_SCENARIO_ENTROPY_BITS}")

    return report


if __name__ == "__main__":
    validate()
