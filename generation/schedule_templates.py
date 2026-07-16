"""Daily activity templates with weekday/weekend variants, jitter, and anomaly events."""

from __future__ import annotations

import copy
import random
from datetime import datetime, timedelta
from typing import Any

from generation import config

Slot = dict[str, Any]

# ---------------------------------------------------------------------------
# Evening leisure pool (rotated; video call is only one option)
# ---------------------------------------------------------------------------
EVENING_LEISURE_OPTIONS: list[Slot] = [
    {
        "slot_id": "leisure_tv",
        "plan_chunk": "Watch TV series episodes and relax on the couch",
        "matched_scenarios": ["Watching tv"],
        "location": "indoor",
        "time_period": "nighttime",
        "duration_min": 75,
    },
    {
        "slot_id": "leisure_games",
        "plan_chunk": "Play video games at home",
        "matched_scenarios": ["Playing games / video games"],
        "location": "indoor",
        "time_period": "nighttime",
        "duration_min": 60,
    },
    {
        "slot_id": "leisure_read",
        "plan_chunk": "Read books and tech articles before bed",
        "matched_scenarios": ["Reading books", "On a screen (phone/laptop)"],
        "location": "indoor",
        "time_period": "nighttime",
        "duration_min": 50,
    },
    {
        "slot_id": "leisure_early_sleep",
        "plan_chunk": "Wind down early: light hygiene and prepare for sleep",
        "matched_scenarios": ["Daily hygiene", "Sleeping"],
        "location": "indoor",
        "time_period": "nighttime",
        "duration_min": 40,
    },
    {
        "slot_id": "leisure_video_call",
        "plan_chunk": "Video call with a close friend",
        "matched_scenarios": ["Video call", "Talking on the phone"],
        "location": "indoor",
        "time_period": "nighttime",
        "duration_min": 45,
    },
    {
        "slot_id": "leisure_music",
        "plan_chunk": "Listen to music and browse social media",
        "matched_scenarios": ["Listening to music", "On a screen (phone/laptop)"],
        "location": "indoor",
        "time_period": "nighttime",
        "duration_min": 55,
    },
]

EXERCISE_OPTIONS: list[Slot] = [
    {
        "slot_id": "exercise_gym",
        "plan_chunk": "Gym workout: weights and cardio machines",
        "matched_scenarios": ["Going to the gym - exercise machine, class, weights"],
        "location": "indoor",
        "time_period": "nighttime",
        "duration_min": 60,
    },
    {
        "slot_id": "exercise_home",
        "plan_chunk": "Home workout: bodyweight exercises and stretching",
        "matched_scenarios": ["Working out at home"],
        "location": "indoor",
        "time_period": "nighttime",
        "duration_min": 45,
    },
    {
        "slot_id": "exercise_run",
        "plan_chunk": "Evening jog or cycling around the neighborhood",
        "matched_scenarios": ["Cycling / jogging", "Working out outside"],
        "location": "outdoor",
        "time_period": "nighttime",
        "duration_min": 50,
    },
]


def _day_info(day_index: int) -> dict[str, Any]:
    start = datetime.strptime(config.START_DATE, "%Y-%m-%d")
    dt = start + timedelta(days=day_index)
    return {
        "day_index": day_index,
        "calendar_date": dt.strftime("%Y-%m-%d"),
        "weekday": dt.weekday(),  # 0=Mon
        "day_of_week": dt.strftime("%A"),
        "is_weekend": dt.weekday() >= 5,
        "week_num": day_index // 7 + 1,
        "sat_index": day_index // 7,  # which Saturday in the lifelog
    }


def _t(hh: int, mm: int) -> str:
    return f"{hh:02d}:{mm:02d}"


def _parse_t(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 60 + m


def _fmt_t(minutes: int) -> str:
    minutes = max(0, min(minutes, 23 * 60 + 59))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _shift_slot(slot: Slot, offset_min: int, duration_delta: int = 0) -> Slot:
    s = copy.deepcopy(slot)
    start = _parse_t(s["start"]) + offset_min
    end = _parse_t(s["end"]) + offset_min + duration_delta
    if end <= start:
        end = start + max(30, s.get("duration_min", 45))
    s["start"] = _fmt_t(start)
    s["end"] = _fmt_t(end)
    return s


def _slot(
    slot_id: str,
    start: str,
    end: str,
    plan_chunk: str,
    matched_scenarios: list[str],
    location: str,
    time_period: str,
) -> Slot:
    return {
        "slot_id": slot_id,
        "start": start,
        "end": end,
        "plan_chunk": plan_chunk,
        "matched_scenarios": matched_scenarios,
        "location": location,
        "time_period": time_period,
        "duration_min": _parse_t(end) - _parse_t(start),
    }


def _weekday_base(day_index: int, rng: random.Random) -> tuple[list[Slot], str, list[str]]:
    """Weekday schedule with fine-grained slots (one atomic activity each)."""
    info = _day_info(day_index)
    wd = info["weekday"]
    theme = "regular_workday"
    anomalies: list[str] = []

    # Default weekday: ~18 atomic plan slots for richer video diversity
    slots = [
        _slot("wake_hygiene", _t(7, 0), _t(7, 20),
              "Wake up and do morning hygiene",
              ["Daily hygiene"], "indoor", "daytime"),
        _slot("breakfast_coffee", _t(7, 20), _t(7, 45),
              "Prepare breakfast and make coffee at home",
              ["Cooking", "Making coffee"], "indoor", "daytime"),
        _slot("commute_morning", _t(8, 0), _t(8, 30),
              "Commute to office by car or bus",
              ["Car - commuting, road trip", "Bus"], "mixed", "daytime"),
        _slot("office_arrive", _t(8, 30), _t(8, 45),
              "Walk from parking or stop into the office building",
              ["Walking on street", "Indoor Navigation (walking)"], "mixed", "daytime"),
        _slot("work_emails", _t(9, 0), _t(9, 35),
              "Check emails and triage morning tasks at desk",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("work_standup", _t(9, 35), _t(10, 15),
              "Talk with colleagues and sync on morning priorities",
              ["Talking to colleagues", "Working at desk"], "indoor", "daytime"),
        _slot("work_focus_coding", _t(10, 45), _t(11, 30),
              "Focused desk work: coding and design docs",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("work_collab", _t(11, 30), _t(12, 0),
              "Whiteboard brainstorming and collaboration",
              ["Writing on whiteboard", "Talking to colleagues"], "indoor", "daytime"),
        _slot("lunch_walk", _t(12, 10), _t(12, 25),
              "Walk to cafeteria or nearby restaurant",
              ["Walking on street", "Indoor Navigation (walking)"], "mixed", "daytime"),
        _slot("lunch_eat", _t(12, 25), _t(13, 0),
              "Eat lunch at cafeteria or restaurant",
              ["Eating at the cafeteria", "Eating at a restaurant", "Eating"],
              "indoor", "daytime"),
        _slot("errand_walk", _t(13, 15), _t(13, 40),
              "Short outdoor walk after lunch",
              ["Walking on street"], "outdoor", "daytime"),
        _slot("errand_park", _t(13, 40), _t(14, 0),
              "Quick park break or outdoor errand",
              ["Going to the park", "Walking on street"], "outdoor", "daytime"),
        _slot("work_afternoon_desk", _t(14, 15), _t(15, 0),
              "Afternoon desk work and ticket handling",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("work_videocall", _t(15, 0), _t(15, 35),
              "Async video call and remote collaboration",
              ["Video call", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("work_afternoon_wrap", _t(15, 35), _t(16, 0),
              "Wrap up notes and prepare handoff for tomorrow",
              ["Working at desk", "Writing on whiteboard"], "indoor", "daytime"),
        _slot("commute_evening", _t(16, 15), _t(16, 40),
              "Commute home by car or bus",
              ["Car - commuting, road trip", "Bus"], "mixed", "daytime"),
        _slot("grocery_stop", _t(16, 40), _t(17, 5),
              "Stop by grocery store for dinner ingredients",
              ["Grocery shopping indoors", "Clothes, other shopping"], "indoor", "daytime"),
        _slot("cook_dinner", _t(17, 15), _t(17, 50),
              "Cook dinner at home",
              ["Cooking"], "indoor", "twilight"),
        _slot("home_social", _t(17, 50), _t(18, 15),
              "Talk with housemates and clean the kitchen",
              ["Talking with friends/housemates", "Cleaning / laundry"],
              "indoor", "twilight"),
    ]

    # --- Weekday-specific variants (replace by slot_id) ---
    if wd == 0:  # Monday: weekly meeting
        theme = "monday_weekly_meeting"
        _upsert(slots, _slot("work_emails", _t(9, 0), _t(9, 25),
                             "Skim emails before all-hands",
                             ["Working at desk", "On a screen (phone/laptop)"],
                             "indoor", "daytime"))
        _upsert(slots, _slot("work_standup", _t(9, 25), _t(10, 30),
                             "Weekly all-hands team meeting and sprint planning",
                             ["Participating in a meeting", "Talking to colleagues", "Working at desk"],
                             "indoor", "daytime"))

    elif wd == 1:  # Tuesday: deep work
        theme = "tuesday_deep_work"
        _upsert(slots, _slot("work_focus_coding", _t(10, 45), _t(11, 45),
                             "Deep focus coding block, headphones on",
                             ["Working at desk", "On a screen (phone/laptop)"],
                             "indoor", "daytime"))
        _upsert(slots, _slot("work_collab", _t(11, 45), _t(12, 15),
                             "Write design doc and review PRs",
                             ["Working at desk", "Writing on whiteboard"],
                             "indoor", "daytime"))

    elif wd == 2:
        theme = "wednesday_midweek"

    elif wd == 3:  # Thursday: project presentation
        theme = "thursday_project_review"
        _upsert(slots, _slot("work_afternoon_desk", _t(14, 15), _t(14, 45),
                             "Finalize slides before demo",
                             ["Working at desk", "On a screen (phone/laptop)"],
                             "indoor", "daytime"))
        _upsert(slots, _slot("work_videocall", _t(14, 45), _t(15, 45),
                             "Project presentation and demo to stakeholders",
                             ["Participating in a meeting", "Writing on whiteboard"],
                             "indoor", "daytime"))
        _upsert(slots, _slot("work_afternoon_wrap", _t(15, 45), _t(16, 15),
                             "Collect feedback and update action items",
                             ["Working at desk", "Talking to colleagues"],
                             "indoor", "daytime"))
        _upsert(slots, _slot("commute_evening", _t(16, 30), _t(16, 55),
                             "Later commute home after review",
                             ["Car - commuting, road trip", "Bus"], "mixed", "daytime"))
        _upsert(slots, _slot("grocery_stop", _t(16, 55), _t(17, 20),
                             "Quick grocery stop on the way home",
                             ["Grocery shopping indoors"], "indoor", "daytime"))

    elif wd == 4:  # Friday: team social
        theme = "friday_team_social"
        _upsert(slots, _slot("work_afternoon_desk", _t(14, 15), _t(14, 50),
                             "Wrap up weekly tickets",
                             ["Working at desk"], "indoor", "daytime"))
        _upsert(slots, _slot("work_videocall", _t(14, 50), _t(15, 20),
                             "Hand off unfinished tasks to teammates",
                             ["Talking to colleagues", "Working at desk"], "indoor", "daytime"))
        _upsert(slots, _slot("work_afternoon_wrap", _t(15, 20), _t(15, 45),
                             "Clear inbox before weekend",
                             ["On a screen (phone/laptop)", "Working at desk"],
                             "indoor", "daytime"))
        _upsert(slots, _slot("commute_evening", _t(16, 0), _t(16, 30),
                             "Commute downtown for team gathering",
                             ["Car - commuting, road trip", "Walking on street"],
                             "mixed", "daytime"))
        # Drop grocery; refresh at home instead
        slots = [s for s in slots if s["slot_id"] != "grocery_stop"]
        _upsert(slots, _slot("home_refresh", _t(17, 0), _t(17, 35),
                             "Quick refresh and light snack at home before heading out",
                             ["Daily hygiene", "Cooking"], "indoor", "twilight"))
        slots = [s for s in slots if s["slot_id"] not in ("cook_dinner", "home_social")]

    # Evening: exercise + leisure (skip gym on Friday)
    exercise = copy.deepcopy(rng.choice(EXERCISE_OPTIONS))
    if wd == 4:
        exercise = None
    leisure_idx = (day_index * 3 + wd) % len(EVENING_LEISURE_OPTIONS)
    leisure = copy.deepcopy(EVENING_LEISURE_OPTIONS[leisure_idx])

    if wd == 4:
        slots.append(_slot("team_dinner", _t(18, 30), _t(19, 45),
                           "Team dinner at restaurant with colleagues",
                           ["Eating at a restaurant", "Talking to colleagues"],
                           "indoor", "nighttime"))
        slots.append(_slot("friday_dessert", _t(19, 45), _t(20, 30),
                           "Dessert or drinks after dinner",
                           ["Eating at a restaurant", "Hanging out with friends at a bar"],
                           "indoor", "nighttime"))
        slots.append(_slot("friday_night_out", _t(20, 30), _t(22, 0),
                           "Casual hangout at bar with teammates",
                           ["Hanging out with friends at a bar", "Talking to colleagues"],
                           "indoor", "nighttime"))
    else:
        if exercise:
            ex_start = _t(19, 45) if wd != 3 else _t(20, 0)
            # Split exercise into arrive + workout when gym/outdoor
            if exercise["slot_id"] == "exercise_gym":
                slots.append(_slot("gym_arrive", ex_start, _fmt_t(_parse_t(ex_start) + 15),
                                   "Arrive at gym and warm up",
                                   ["Going to the gym - exercise machine, class, weights",
                                    "Walking on street"],
                                   "mixed", "nighttime"))
                slots.append(_slot("exercise_gym",
                                   _fmt_t(_parse_t(ex_start) + 15),
                                   _fmt_t(_parse_t(ex_start) + exercise.get("duration_min", 60)),
                                   "Gym workout: weights and cardio machines",
                                   exercise["matched_scenarios"],
                                   exercise["location"], exercise["time_period"]))
                lv_start = _fmt_t(_parse_t(ex_start) + exercise.get("duration_min", 60) + 15)
            elif exercise["slot_id"] == "exercise_run":
                slots.append(_slot("exercise_run", ex_start,
                                   _fmt_t(_parse_t(ex_start) + exercise.get("duration_min", 50)),
                                   exercise["plan_chunk"], exercise["matched_scenarios"],
                                   exercise["location"], exercise["time_period"]))
                lv_start = _fmt_t(_parse_t(ex_start) + exercise.get("duration_min", 50) + 15)
            else:
                slots.append(_slot(exercise["slot_id"], ex_start,
                                   _fmt_t(_parse_t(ex_start) + exercise.get("duration_min", 45)),
                                   exercise["plan_chunk"], exercise["matched_scenarios"],
                                   exercise["location"], exercise["time_period"]))
                lv_start = _fmt_t(_parse_t(ex_start) + exercise.get("duration_min", 45) + 15)
        else:
            lv_start = _t(19, 30)
        # Split leisure when it is long TV / games
        lv_dur = leisure.get("duration_min", 55)
        if leisure["slot_id"] in ("leisure_tv", "leisure_games") and lv_dur >= 50:
            mid = _fmt_t(_parse_t(lv_start) + lv_dur // 2)
            lv_end = _fmt_t(_parse_t(lv_start) + lv_dur)
            slots.append(_slot(leisure["slot_id"] + "_1", lv_start, mid,
                               leisure["plan_chunk"] + " (first half)",
                               leisure["matched_scenarios"],
                               leisure["location"], leisure["time_period"]))
            slots.append(_slot(leisure["slot_id"] + "_2", mid, lv_end,
                               leisure["plan_chunk"] + " (second half)",
                               leisure["matched_scenarios"],
                               leisure["location"], leisure["time_period"]))
        else:
            lv_end = _fmt_t(_parse_t(lv_start) + lv_dur)
            slots.append(_slot(leisure["slot_id"], lv_start, lv_end,
                               leisure["plan_chunk"], leisure["matched_scenarios"],
                               leisure["location"], leisure["time_period"]))

    slots.sort(key=lambda s: _parse_t(s["start"]))
    return slots, theme, anomalies


def _upsert(slots: list[Slot], new_slot: Slot) -> None:
    """Replace existing slot_id or append."""
    for i, s in enumerate(slots):
        if s["slot_id"] == new_slot["slot_id"]:
            slots[i] = new_slot
            return
    slots.append(new_slot)


def _weekend_base(day_index: int, rng: random.Random) -> tuple[list[Slot], str, list[str]]:
    info = _day_info(day_index)
    sat_idx = info["sat_index"]
    is_saturday = info["weekday"] == 5
    anomalies: list[str] = []

    if is_saturday:
        themes = [
            ("saturday_hiking", _saturday_hiking()),
            ("saturday_cycling_market", _saturday_cycling()),
            ("saturday_outdoor_expedition", _saturday_expedition()),
        ]
        theme, slots = themes[sat_idx % len(themes)]
    else:
        themes = [
            ("sunday_movie_marathon", _sunday_movie()),
            ("sunday_cozy_home", _sunday_cozy()),
            ("sunday_gaming_cooking", _sunday_gaming()),
        ]
        theme, slots = themes[sat_idx % len(themes)]

    return copy.deepcopy(slots), theme, anomalies


def _saturday_hiking() -> list[Slot]:
    return [
        _slot("wake_hygiene", _t(8, 0), _t(8, 25),
              "Early weekend wake-up and hygiene",
              ["Daily hygiene"], "indoor", "daytime"),
        _slot("pack_gear", _t(8, 25), _t(9, 0),
              "Pack snacks, water, and hiking gear",
              ["Cooking", "Making coffee"], "indoor", "daytime"),
        _slot("drive_trailhead", _t(9, 30), _t(10, 30),
              "Drive to trailhead for day hike",
              ["Car - commuting, road trip"], "mixed", "daytime"),
        _slot("hiking_ascent", _t(10, 45), _t(12, 15),
              "Hike uphill on mountain trail",
              ["Hiking", "Walking on street"], "outdoor", "daytime"),
        _slot("trail_views", _t(12, 15), _t(13, 15),
              "Pause for scenic views and photos",
              ["Hiking", "Tourism", "Taking photos in photography studio"],
              "outdoor", "daytime"),
        _slot("trail_lunch", _t(13, 30), _t(14, 15),
              "Picnic lunch on the trail",
              ["BBQ'ing/picnics", "Eating", "Outdoor cooking"], "outdoor", "daytime"),
        _slot("hiking_descent", _t(14, 30), _t(16, 15),
              "Descend trail back toward trailhead",
              ["Hiking", "Walking on street"], "outdoor", "daytime"),
        _slot("drive_home", _t(16, 45), _t(17, 45),
              "Drive home after the hike",
              ["Car - commuting, road trip"], "mixed", "twilight"),
        _slot("shower_stretch", _t(18, 15), _t(19, 0),
              "Shower and stretch sore muscles",
              ["Daily hygiene", "Working out at home"], "indoor", "nighttime"),
        _slot("simple_dinner", _t(19, 0), _t(19, 45),
              "Simple recovery dinner at home",
              ["Cooking", "Eating"], "indoor", "nighttime"),
        _slot("early_sleep", _t(20, 15), _t(21, 15),
              "Early to bed after exhausting hike",
              ["Sleeping", "Reading books"], "indoor", "nighttime"),
    ]


def _saturday_cycling() -> list[Slot]:
    return [
        _slot("wake_hygiene", _t(8, 30), _t(8, 55),
              "Leisurely wake-up and hygiene",
              ["Daily hygiene"], "indoor", "daytime"),
        _slot("brunch_home", _t(8, 55), _t(9, 35),
              "Brunch and coffee at home",
              ["Making coffee", "Cooking"], "indoor", "daytime"),
        _slot("cycling", _t(10, 0), _t(11, 30),
              "Cycling along riverside bike path",
              ["Cycling / jogging"], "outdoor", "daytime"),
        _slot("park_break", _t(11, 30), _t(12, 0),
              "Rest at park benches after cycling",
              ["Going to the park", "Walking on street"], "outdoor", "daytime"),
        _slot("farmers_market", _t(12, 20), _t(13, 10),
              "Browse farmers market stalls",
              ["Clothes, other shopping", "Walking on street"], "outdoor", "daytime"),
        _slot("buy_produce", _t(13, 10), _t(13, 40),
              "Buy fresh produce and snacks",
              ["Grocery shopping indoors", "Walking on street"], "mixed", "daytime"),
        _slot("home_cooking", _t(14, 10), _t(15, 10),
              "Cook with market finds",
              ["Cooking"], "indoor", "daytime"),
        _slot("home_craft", _t(15, 10), _t(15, 45),
              "Light crafting while food rests",
              ["Crafting/knitting/sewing/drawing/painting"], "indoor", "daytime"),
        _slot("social_outdoor", _t(16, 10), _t(17, 10),
              "Meet friends at park for frisbee",
              ["Frisbee", "Outdoor social (includes campfire)"], "outdoor", "daytime"),
        _slot("social_chat", _t(17, 10), _t(17, 40),
              "Chat with friends after the game",
              ["Talking with friends/housemates", "Walking on street"],
              "outdoor", "daytime"),
        _slot("dining_out", _t(19, 0), _t(20, 15),
              "Dinner at a new restaurant downtown",
              ["Eating at a restaurant"], "indoor", "nighttime"),
        _slot("board_games", _t(20, 30), _t(21, 45),
              "Play board games at friend's home",
              ["Playing board games", "Playing games / video games"],
              "indoor", "nighttime"),
    ]


def _saturday_expedition() -> list[Slot]:
    return [
        _slot("wake_hygiene", _t(7, 30), _t(7, 55),
              "Prepare for full-day outdoor expedition",
              ["Daily hygiene"], "indoor", "daytime"),
        _slot("pack_breakfast", _t(7, 55), _t(8, 30),
              "Quick breakfast and pack day bag",
              ["Cooking", "Making coffee"], "indoor", "daytime"),
        _slot("hiking_morning", _t(9, 0), _t(11, 0),
              "Morning hiking stretch",
              ["Hiking", "Walking on street"], "outdoor", "daytime"),
        _slot("nature_photos", _t(11, 0), _t(12, 0),
              "Nature photography and sightseeing",
              ["Taking photos in photography studio", "Tourism"],
              "outdoor", "daytime"),
        _slot("outdoor_lunch", _t(12, 15), _t(13, 0),
              "Eat at outdoor hawker center",
              ["Eating in hawker center", "Eating"], "outdoor", "daytime"),
        _slot("explore_walk", _t(13, 30), _t(15, 0),
              "Explore a new neighborhood on foot",
              ["Walking on street", "Tourism"], "outdoor", "daytime"),
        _slot("local_shops", _t(15, 0), _t(16, 15),
              "Visit local shops and browse",
              ["Clothes, other shopping", "Walking on street"], "mixed", "daytime"),
        _slot("bus_home", _t(16, 45), _t(17, 45),
              "Bus ride home, review photos on phone",
              ["Bus", "On a screen (phone/laptop)"], "mixed", "twilight"),
        _slot("cook_dinner", _t(18, 15), _t(19, 15),
              "Cook hearty dinner at home",
              ["Cooking"], "indoor", "nighttime"),
        _slot("share_photos", _t(19, 15), _t(20, 0),
              "Share photos with friends on a video call",
              ["Video call", "Talking with friends/housemates"],
              "indoor", "nighttime"),
    ]


def _sunday_movie() -> list[Slot]:
    return [
        _slot("wake_hygiene", _t(9, 30), _t(9, 55),
              "Sleep in, slow morning hygiene",
              ["Daily hygiene"], "indoor", "daytime"),
        _slot("lazy_breakfast", _t(9, 55), _t(10, 35),
              "Lazy breakfast in pajamas with coffee",
              ["Making coffee", "Cooking"], "indoor", "daytime"),
        _slot("laundry", _t(11, 0), _t(11, 35),
              "Start laundry load",
              ["Cleaning / laundry"], "indoor", "daytime"),
        _slot("tidy_apartment", _t(11, 35), _t(12, 10),
              "Tidying up apartment and ironing",
              ["Cleaning / laundry", "Ironing"], "indoor", "daytime"),
        _slot("movie_1", _t(12, 30), _t(14, 0),
              "First movie of the marathon",
              ["Watching tv"], "indoor", "daytime"),
        _slot("delivery_lunch", _t(14, 10), _t(14, 50),
              "Order delivery and eat on the couch",
              ["Eating", "Drive-thru food"], "indoor", "daytime"),
        _slot("movie_2", _t(15, 0), _t(16, 45),
              "Second movie in the series",
              ["Watching tv", "Watching movies at the cinema"],
              "indoor", "daytime"),
        _slot("movie_break", _t(16, 45), _t(17, 15),
              "Short stretch break between films",
              ["Working out at home", "On a screen (phone/laptop)"],
              "indoor", "twilight"),
        _slot("movie_3", _t(17, 15), _t(19, 0),
              "Continue movie marathon into evening",
              ["Watching tv"], "indoor", "twilight"),
        _slot("home_cooking", _t(19, 20), _t(20, 15),
              "Simple home-cooked dinner",
              ["Cooking"], "indoor", "nighttime"),
        _slot("leisure_read", _t(20, 45), _t(21, 45),
              "Read a few chapters before early sleep",
              ["Reading books"], "indoor", "nighttime"),
    ]


def _sunday_cozy() -> list[Slot]:
    return [
        _slot("wake_hygiene", _t(9, 0), _t(9, 25),
              "Slow morning hygiene",
              ["Daily hygiene"], "indoor", "daytime"),
        _slot("pour_over", _t(9, 25), _t(10, 0),
              "Pour-over coffee and podcast",
              ["Making coffee", "Listening to music"], "indoor", "daytime"),
        _slot("crafting", _t(10, 30), _t(11, 30),
              "Crafting project at the table",
              ["Crafting/knitting/sewing/drawing/painting"], "indoor", "daytime"),
        _slot("bookshelf", _t(11, 30), _t(12, 20),
              "Organize bookshelf and browse titles",
              ["Reading books"], "indoor", "daytime"),
        _slot("bake_bread", _t(13, 0), _t(14, 0),
              "Bake bread for lunch",
              ["Cooking", "Baker"], "indoor", "daytime"),
        _slot("elaborate_lunch", _t(14, 0), _t(14, 45),
              "Prepare and eat elaborate lunch",
              ["Cooking", "Eating"], "indoor", "daytime"),
        _slot("nap_relax", _t(15, 10), _t(16, 0),
              "Afternoon nap or quiet rest",
              ["Sleeping"], "indoor", "daytime"),
        _slot("phone_browse", _t(16, 0), _t(16, 40),
              "Browse phone and play with pet",
              ["Playing with pets", "On a screen (phone/laptop)"],
              "indoor", "daytime"),
        _slot("meal_prep", _t(17, 20), _t(18, 10),
              "Meal prep for the coming work week",
              ["Cooking"], "indoor", "twilight"),
        _slot("laundry_fold", _t(18, 10), _t(18, 40),
              "Fold laundry and tidy kitchen",
              ["Cleaning / laundry"], "indoor", "twilight"),
        _slot("skincare_journal", _t(19, 30), _t(20, 15),
              "Skincare and journaling",
              ["Daily hygiene", "Reading books"], "indoor", "nighttime"),
        _slot("early_sleep", _t(20, 15), _t(21, 0),
              "Early sleep routine",
              ["Sleeping"], "indoor", "nighttime"),
    ]


def _sunday_gaming() -> list[Slot]:
    return [
        _slot("wake_hygiene", _t(10, 0), _t(10, 25),
              "Late wake-up and hygiene",
              ["Daily hygiene"], "indoor", "daytime"),
        _slot("brunch_news", _t(10, 25), _t(11, 0),
              "Brunch and check gaming news online",
              ["Cooking", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("gaming_rpg", _t(11, 30), _t(13, 0),
              "Solo RPG gaming session",
              ["Playing games / video games"], "indoor", "daytime"),
        _slot("gaming_snack", _t(13, 0), _t(13, 30),
              "Snack break between game sessions",
              ["Eating", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("gaming_rpg_2", _t(13, 30), _t(14, 30),
              "Continue RPG story missions",
              ["Playing games / video games"], "indoor", "daytime"),
        _slot("online_multiplayer", _t(14, 45), _t(16, 15),
              "Online multiplayer with friends",
              ["Playing games / video games", "Video call"], "indoor", "daytime"),
        _slot("voice_chat", _t(16, 15), _t(16, 45),
              "Post-match voice chat with friends",
              ["Video call", "Talking on the phone"], "indoor", "daytime"),
        _slot("comfort_cook", _t(17, 10), _t(18, 10),
              "Cook comfort food for dinner",
              ["Cooking"], "indoor", "twilight"),
        _slot("gaming_evening", _t(19, 0), _t(20, 30),
              "Evening gaming: try a new game mode",
              ["Playing games / video games"], "indoor", "nighttime"),
        _slot("watch_streams", _t(20, 30), _t(21, 15),
              "Watch a short stream or related show",
              ["Watching tv", "On a screen (phone/laptop)"], "indoor", "nighttime"),
        _slot("leisure_music", _t(21, 30), _t(22, 15),
              "Wind down with music playlist",
              ["Listening to music"], "indoor", "nighttime"),
    ]


# ---------------------------------------------------------------------------
# Anomaly day schedules (override entire day)
# ---------------------------------------------------------------------------
def _anomaly_rain_day() -> list[Slot]:
    return [
        _slot("wake_hygiene", _t(7, 10), _t(7, 30),
              "Wake up to heavy rain",
              ["Daily hygiene"], "indoor", "daytime"),
        _slot("check_weather", _t(7, 30), _t(7, 50),
              "Check weather app and messages",
              ["On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("commute_morning", _t(8, 10), _t(8, 45),
              "Rainy commute with umbrella",
              ["Car - commuting, road trip", "Bus"], "mixed", "daytime"),
        _slot("office_arrive", _t(8, 45), _t(9, 0),
              "Walk into office with wet shoes",
              ["Walking on street", "Indoor Navigation (walking)"], "mixed", "daytime"),
        _slot("work_emails", _t(9, 10), _t(9, 50),
              "Desk work while many colleagues WFH",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("work_videocall", _t(9, 50), _t(10, 30),
              "Morning video sync with remote teammates",
              ["Video call", "Working at desk"], "indoor", "daytime"),
        _slot("work_focus_coding", _t(10, 45), _t(12, 0),
              "Focused coding, rain visible through window",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("lunch_eat", _t(12, 15), _t(13, 0),
              "Lunch at office cafeteria",
              ["Eating at the cafeteria", "Eating"], "indoor", "daytime"),
        _slot("indoor_coffee", _t(13, 15), _t(13, 45),
              "Cancelled outdoor walk; coffee break indoors",
              ["Making coffee", "Hanging out at a coffee shop"], "indoor", "daytime"),
        _slot("chat_colleagues", _t(13, 45), _t(14, 10),
              "Chat with colleagues by the window",
              ["Talking to colleagues"], "indoor", "daytime"),
        _slot("afternoon_meeting", _t(14, 15), _t(15, 15),
              "Afternoon meeting via video call",
              ["Video call", "Participating in a meeting"], "indoor", "daytime"),
        _slot("work_afternoon_wrap", _t(15, 15), _t(16, 0),
              "Finish tickets before heading home",
              ["Working at desk"], "indoor", "daytime"),
        _slot("commute_evening", _t(16, 15), _t(17, 0),
              "Rainy evening commute",
              ["Car - commuting, road trip", "Bus"], "mixed", "daytime"),
        _slot("cook_soup", _t(17, 20), _t(18, 0),
              "Cook warm soup for dinner",
              ["Cooking"], "indoor", "twilight"),
        _slot("dry_clothes", _t(18, 0), _t(18, 30),
              "Dry wet clothes and tidy entryway",
              ["Cleaning / laundry"], "indoor", "twilight"),
        _slot("leisure_tv", _t(19, 0), _t(20, 15),
              "Watch comfort TV show indoors",
              ["Watching tv"], "indoor", "nighttime"),
        _slot("leisure_read", _t(20, 30), _t(21, 30),
              "Read by the window while rain falls",
              ["Reading books"], "indoor", "nighttime"),
    ]


def _anomaly_business_trip() -> list[Slot]:
    return [
        _slot("wake_hotel", _t(6, 30), _t(6, 55),
              "Wake up in hotel and quick hygiene",
              ["Daily hygiene"], "indoor", "daytime"),
        _slot("pack_bag", _t(6, 55), _t(7, 20),
              "Pack bag and check room before checkout",
              ["Cleaning / laundry"], "indoor", "daytime"),
        _slot("taxi_airport", _t(7, 30), _t(8, 30),
              "Taxi to the airport",
              ["Car - commuting, road trip"], "mixed", "daytime"),
        _slot("airport_checkin", _t(8, 30), _t(9, 30),
              "Check in, security, wait at gate",
              ["Walking on street", "Talking on the phone"], "mixed", "daytime"),
        _slot("flight_work", _t(10, 0), _t(11, 30),
              "Work on laptop during flight",
              ["On a screen (phone/laptop)"], "mixed", "daytime"),
        _slot("flight_read", _t(11, 30), _t(12, 30),
              "Read on the plane for the remaining flight",
              ["Reading books"], "mixed", "daytime"),
        _slot("client_demo", _t(13, 30), _t(14, 45),
              "Client site product demo",
              ["Participating in a meeting", "Working at desk"], "indoor", "daytime"),
        _slot("client_discussion", _t(14, 45), _t(15, 30),
              "Follow-up discussion with client team",
              ["Talking to colleagues", "Participating in a meeting"],
              "indoor", "daytime"),
        _slot("business_lunch", _t(15, 45), _t(16, 45),
              "Business lunch with client team",
              ["Eating at a restaurant", "Talking to colleagues"], "indoor", "daytime"),
        _slot("hotel_emails", _t(17, 30), _t(18, 20),
              "Hotel room: send follow-up emails",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "twilight"),
        _slot("hotel_slides", _t(18, 20), _t(19, 0),
              "Update slides for tomorrow",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "twilight"),
        _slot("solo_dinner", _t(19, 30), _t(20, 30),
              "Solo dinner at hotel restaurant",
              ["Eating at a restaurant", "Eating"], "indoor", "nighttime"),
        _slot("hotel_hygiene", _t(21, 0), _t(21, 25),
              "Night hygiene in hotel bathroom",
              ["Daily hygiene"], "indoor", "nighttime"),
        _slot("hotel_sleep", _t(21, 25), _t(22, 0),
              "Early sleep in unfamiliar hotel room",
              ["Sleeping"], "indoor", "nighttime"),
    ]


def _anomaly_overtime() -> list[Slot]:
    return [
        _slot("wake_hygiene", _t(6, 50), _t(7, 15),
              "Wake early, anxious about deadline",
              ["Daily hygiene"], "indoor", "daytime"),
        _slot("coffee_rush", _t(7, 15), _t(7, 35),
              "Quick coffee before rushing out",
              ["Making coffee"], "indoor", "daytime"),
        _slot("commute_morning", _t(7, 50), _t(8, 30),
              "Quick commute to office",
              ["Car - commuting, road trip", "Bus"], "mixed", "daytime"),
        _slot("bugfix_1", _t(8, 45), _t(10, 15),
              "Fix critical bugs before presentation",
              ["Working at desk", "Fixing PC"], "indoor", "daytime"),
        _slot("bugfix_2", _t(10, 15), _t(12, 0),
              "Continue debugging and verifying patches",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("desk_lunch", _t(12, 10), _t(12, 40),
              "Quick desk lunch, no time to go out",
              ["Eating", "Working at desk"], "indoor", "daytime"),
        _slot("presentation", _t(13, 0), _t(14, 30),
              "High-stakes project presentation",
              ["Participating in a meeting", "Writing on whiteboard"],
              "indoor", "daytime"),
        _slot("post_review", _t(14, 30), _t(15, 15),
              "Collect leadership feedback",
              ["Talking to colleagues", "Participating in a meeting"],
              "indoor", "daytime"),
        _slot("overtime_patch", _t(15, 15), _t(17, 30),
              "Overtime: patch issues found during review",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("overtime_team", _t(17, 30), _t(19, 30),
              "Team stays late finishing remaining fixes",
              ["Working at desk", "Talking to colleagues", "Video call"],
              "indoor", "nighttime"),
        _slot("late_commute", _t(19, 45), _t(20, 30),
              "Late-night commute home",
              ["Car - commuting, road trip", "Bus"], "mixed", "nighttime"),
        _slot("takeout_dinner", _t(20, 45), _t(21, 25),
              "Order takeout and eat on the couch",
              ["Drive-thru food", "Eating"], "indoor", "nighttime"),
        _slot("crash_sleep", _t(21, 35), _t(22, 10),
              "Crash into bed immediately",
              ["Sleeping"], "indoor", "nighttime"),
    ]


def _anomaly_impromptu_date() -> list[Slot]:
    return [
        _slot("wake_hygiene", _t(7, 5), _t(7, 25),
              "Normal morning hygiene",
              ["Daily hygiene"], "indoor", "daytime"),
        _slot("breakfast_coffee", _t(7, 25), _t(7, 50),
              "Breakfast and coffee at home",
              ["Cooking", "Making coffee"], "indoor", "daytime"),
        _slot("commute_morning", _t(8, 5), _t(8, 45),
              "Commute to office",
              ["Car - commuting, road trip", "Bus"], "mixed", "daytime"),
        _slot("work_emails", _t(9, 0), _t(10, 0),
              "Morning inbox and tickets",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("work_standup", _t(10, 0), _t(11, 0),
              "Collaborate with teammates",
              ["Talking to colleagues", "Working at desk"], "indoor", "daytime"),
        _slot("work_focus_coding", _t(11, 0), _t(12, 0),
              "Finish a coding task before lunch",
              ["Working at desk"], "indoor", "daytime"),
        _slot("lunch_eat", _t(12, 15), _t(12, 55),
              "Lunch with coworker; get impromptu date invite",
              ["Eating at the cafeteria", "Talking to colleagues"], "indoor", "daytime"),
        _slot("work_afternoon_desk", _t(13, 15), _t(15, 0),
              "Afternoon desk work",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("work_afternoon_wrap", _t(15, 0), _t(17, 0),
              "Wrap work early enough to leave on time",
              ["Working at desk"], "indoor", "daytime"),
        _slot("commute_date", _t(17, 15), _t(17, 45),
              "Commute downtown for the date",
              ["Car - commuting, road trip", "Walking on street"], "mixed", "daytime"),
        _slot("date_dinner", _t(18, 0), _t(19, 30),
              "Impromptu dinner date at a nice restaurant",
              ["Eating at a restaurant", "Talking with friends/housemates"],
              "indoor", "nighttime"),
        _slot("date_dessert", _t(19, 30), _t(20, 0),
              "Dessert and conversation after dinner",
              ["Eating at a restaurant"], "indoor", "nighttime"),
        _slot("evening_walk", _t(20, 15), _t(21, 0),
              "After-dinner walk along the waterfront",
              ["Walking on street"], "outdoor", "nighttime"),
        _slot("home_hygiene", _t(21, 30), _t(21, 55),
              "Return home and get ready for bed",
              ["Daily hygiene"], "indoor", "nighttime"),
        _slot("early_sleep", _t(21, 55), _t(22, 20),
              "Sleep after a good evening",
              ["Sleeping"], "indoor", "nighttime"),
    ]


def _anomaly_doctor_visit() -> list[Slot]:
    return [
        _slot("wake_hygiene", _t(7, 0), _t(7, 20),
              "Wake up, remember doctor appointment",
              ["Daily hygiene"], "indoor", "daytime"),
        _slot("coffee_quick", _t(7, 20), _t(7, 40),
              "Quick coffee before leaving",
              ["Making coffee"], "indoor", "daytime"),
        _slot("commute_morning", _t(8, 0), _t(8, 35),
              "Commute to office first",
              ["Car - commuting, road trip"], "mixed", "daytime"),
        _slot("work_before_appt", _t(8, 45), _t(10, 15),
              "Work before leaving for appointment",
              ["Working at desk"], "indoor", "daytime"),
        _slot("work_emails", _t(10, 15), _t(11, 15),
              "Clear urgent emails before clinic",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("clinic_wait", _t(11, 45), _t(12, 20),
              "Wait at the clinic waiting room",
              ["Appointments: doctor, dentist", "On a screen (phone/laptop)"],
              "indoor", "daytime"),
        _slot("doctor_appointment", _t(12, 20), _t(13, 0),
              "Annual check-up with the doctor",
              ["Appointments: doctor, dentist", "Talking on the phone"],
              "indoor", "daytime"),
        _slot("lunch_near_clinic", _t(13, 15), _t(14, 0),
              "Grab lunch near the clinic",
              ["Eating at a restaurant", "Eating"], "indoor", "daytime"),
        _slot("work_afternoon_desk", _t(14, 15), _t(15, 30),
              "Return to office for afternoon work",
              ["Working at desk"], "indoor", "daytime"),
        _slot("work_videocall", _t(15, 30), _t(16, 20),
              "Catch up on a video call",
              ["Video call", "Working at desk"], "indoor", "daytime"),
        _slot("commute_evening", _t(16, 45), _t(17, 25),
              "Commute home",
              ["Car - commuting, road trip"], "mixed", "daytime"),
        _slot("light_dinner", _t(17, 45), _t(18, 30),
              "Cook a light dinner and rest",
              ["Cooking"], "indoor", "twilight"),
        _slot("gentle_stretch", _t(19, 0), _t(19, 40),
              "Gentle home stretching, skip intense workout",
              ["Working out at home", "Yoga practice"], "indoor", "nighttime"),
        _slot("leisure_tv", _t(20, 0), _t(21, 15),
              "Watch TV and relax",
              ["Watching tv"], "indoor", "nighttime"),
    ]


# Anomaly registry: day_index -> (theme, anomaly_labels, slot_builder)
ANOMALY_DAYS: dict[int, tuple[str, list[str], Any]] = {
    2: ("anomaly_rain_day", ["rain_cancelled_outdoor"], _anomaly_rain_day),
    7: ("anomaly_business_trip", ["business_trip_day1"], _anomaly_business_trip),
    10: ("anomaly_overtime", ["overtime_late_night"], _anomaly_overtime),
    15: ("anomaly_impromptu_date", ["impromptu_evening_date"], _anomaly_impromptu_date),
    16: ("anomaly_doctor_visit", ["doctor_appointment_midday"], _anomaly_doctor_visit),
}


def _apply_time_jitter(slots: list[Slot], rng: random.Random, is_weekend: bool) -> dict[str, int]:
    """Apply small random shifts to wake, commute, exercise timing."""
    wake_jitter = rng.randint(-15, 20) if not is_weekend else rng.randint(-20, 30)
    commute_jitter = rng.randint(-10, 15)
    exercise_jitter = rng.randint(-10, 20)

    jittered: list[Slot] = []
    for s in slots:
        offset = 0
        dur_delta = 0
        sid = s["slot_id"]
        if sid.startswith("wake_") or sid in ("breakfast_coffee", "lazy_breakfast", "brunch_home",
                                              "pour_over", "pack_gear", "pack_breakfast",
                                              "coffee_rush", "coffee_quick", "check_weather",
                                              "pack_bag"):
            offset = wake_jitter
        elif ("commute" in sid or "drive" in sid or "travel" in sid or "taxi" in sid
              or sid.startswith("bus_") or sid == "office_arrive" or sid == "airport_checkin"):
            offset = wake_jitter + commute_jitter
        elif sid.startswith("exercise") or sid.startswith("gym_") or sid.startswith("work_"):
            offset = wake_jitter + commute_jitter // 2
        elif sid.startswith("leisure") or sid.startswith("team") or sid.startswith("friday"):
            dur_delta = rng.randint(-10, 15)
            offset = wake_jitter + commute_jitter // 2
        else:
            offset = wake_jitter + commute_jitter // 2
        jittered.append(_shift_slot(s, offset, dur_delta))

    # Mutate in place so callers keep the jittered list
    slots[:] = jittered

    return {
        "wake_jitter_min": wake_jitter,
        "commute_jitter_min": commute_jitter,
        "exercise_jitter_min": exercise_jitter,
    }


def build_day_schedule(day_index: int, rng: random.Random | None = None) -> dict[str, Any]:
    rng = rng or random.Random(day_index + 42)
    info = _day_info(day_index)

    if day_index in ANOMALY_DAYS:
        theme, anomalies, builder = ANOMALY_DAYS[day_index]
        slots = builder()
    elif info["is_weekend"]:
        slots, theme, anomalies = _weekend_base(day_index, rng)
    else:
        slots, theme, anomalies = _weekday_base(day_index, rng)

    jitter = _apply_time_jitter(slots, rng, info["is_weekend"])

    return {
        "day_index": day_index,
        "calendar_date": info["calendar_date"],
        "day_of_week": info["day_of_week"],
        "is_weekend": info["is_weekend"],
        "day_theme": theme,
        "anomaly_events": anomalies,
        "time_variations": jitter,
        "slots": slots,
    }


def build_persona() -> dict[str, Any]:
    return {
        "persona_id": "egotailor_usa_enfp",
        "location": config.PERSONA_LOCATION,
        "gen_way": "rule_based_v2",
        "personality_traits": {
            "mbti_type": config.PERSONA_MBti,
            "character_traits": [
                "curious and energetic",
                "enjoys social interaction",
                "organized but flexible",
                "health-conscious",
            ],
        },
        "lifestyle": (
            "A 28-year-old software engineer living in a US city. "
            "Works hybrid with Monday stand-ups, Thursday project reviews, and Friday team dinners. "
            "Weekends alternate between outdoor adventures (Saturday) and cozy home days (Sunday). "
            "Life includes occasional overtime, business trips, and spontaneous social plans."
        ),
        "daily_routine": [
            "Wake 6:45-7:20 on weekdays (varies), 8:00-10:00 on weekends",
            "Monday weekly meeting, Thursday project presentation, Friday team dinner",
            "Saturday: hiking / cycling / outdoor themes by week",
            "Sunday: movie marathon / cozy home / gaming themes by week",
            "Evening mix: TV, gaming, reading, early sleep — not always social calls",
        ],
        "hobbies": [
            "hiking and cycling",
            "cooking and baking",
            "video games and TV series",
            "reading",
        ],
        "scheduled_anomalies": {
            str(k): {"theme": v[0], "events": v[1]}
            for k, v in ANOMALY_DAYS.items()
        },
    }


def build_daily_plan(day_index: int, rng: random.Random | None = None) -> list[str]:
    sched = build_day_schedule(day_index, rng)
    return [f"{s['plan_chunk']} at {s['start']} - {s['end']}" for s in sched["slots"]]


def build_daily_plan_chunks(day_index: int, rng: random.Random | None = None) -> list[dict[str, Any]]:
    sched = build_day_schedule(day_index, rng)
    return [
        {
            "slot_id": s["slot_id"],
            "start_time": s["start"],
            "end_time": s["end"],
            "plan_chunk": s["plan_chunk"],
            "matched_scenarios": s["matched_scenarios"],
            "location": s["location"],
            "time_period": s["time_period"],
            "day_theme": sched["day_theme"],
            "anomaly_events": sched["anomaly_events"],
        }
        for s in sched["slots"]
    ]


def build_day_metadata(day_index: int, rng: random.Random | None = None) -> dict[str, Any]:
    sched = build_day_schedule(day_index, rng)
    return {
        "day_theme": sched["day_theme"],
        "anomaly_events": sched["anomaly_events"],
        "time_variations": sched["time_variations"],
    }
