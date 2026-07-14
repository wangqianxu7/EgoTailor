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
    info = _day_info(day_index)
    wd = info["weekday"]
    theme = "regular_workday"
    anomalies: list[str] = []

    slots = [
        _slot("wake_home", _t(7, 0), _t(7, 45),
               "Wake up, daily hygiene, prepare breakfast and coffee at home",
               ["Daily hygiene", "Cooking", "Making coffee"], "indoor", "daytime"),
        _slot("commute_morning", _t(8, 0), _t(8, 40),
               "Commute to office by car or bus",
               ["Car - commuting, road trip", "Bus", "Walking on street"], "mixed", "daytime"),
        _slot("work_morning", _t(9, 0), _t(10, 30),
               "Work at desk, read emails, handle morning tasks",
               ["Working at desk", "On a screen (phone/laptop)", "Talking to colleagues"],
               "indoor", "daytime"),
        _slot("work_focus", _t(10, 45), _t(12, 0),
               "Focused desk work and collaboration",
               ["Working at desk", "Writing on whiteboard", "Labwork"], "indoor", "daytime"),
        _slot("lunch_out", _t(12, 15), _t(13, 0),
               "Lunch break at cafeteria or nearby restaurant",
               ["Eating at the cafeteria", "Eating at a restaurant", "Eating"],
               "indoor", "daytime"),
        _slot("errand_outdoor", _t(13, 15), _t(14, 0),
               "Short walk outside or quick errand in the park",
               ["Walking on street", "Going to the park"], "outdoor", "daytime"),
        _slot("work_afternoon", _t(14, 15), _t(16, 0),
               "Afternoon work session and async collaboration",
               ["Working at desk", "Video call", "On a screen (phone/laptop)"],
               "indoor", "daytime"),
        _slot("commute_evening", _t(16, 15), _t(17, 0),
               "Commute home, optionally stop by grocery store",
               ["Car - commuting, road trip", "Grocery shopping indoors", "Bus"],
               "mixed", "daytime"),
        _slot("home_evening", _t(17, 15), _t(18, 15),
               "Cook dinner, talk with housemates, clean kitchen",
               ["Cooking", "Talking with friends/housemates", "Cleaning / laundry"],
               "indoor", "twilight"),
    ]

    # --- Weekday-specific variants ---
    if wd == 0:  # Monday: weekly meeting
        theme = "monday_weekly_meeting"
        slots[2] = _slot("work_morning", _t(9, 0), _t(10, 30),
                         "Weekly all-hands team meeting and sprint planning",
                         ["Participating in a meeting", "Working at desk", "Talking to colleagues"],
                         "indoor", "daytime")

    elif wd == 1:  # Tuesday: deep work
        theme = "tuesday_deep_work"
        slots[3] = _slot("work_focus", _t(10, 45), _t(12, 15),
                         "Deep focus: coding, design docs, no meetings",
                         ["Working at desk", "On a screen (phone/laptop)", "Writing on whiteboard"],
                         "indoor", "daytime")

    elif wd == 2:  # Wednesday: mid-week
        theme = "wednesday_midweek"

    elif wd == 3:  # Thursday: project presentation
        theme = "thursday_project_review"
        slots[6] = _slot("work_afternoon", _t(14, 15), _t(16, 15),
                         "Project presentation and demo to stakeholders",
                         ["Participating in a meeting", "Writing on whiteboard", "Working at desk"],
                         "indoor", "daytime")
        slots[7] = _shift_slot(slots[7], 15)  # later commute

    elif wd == 4:  # Friday: team dinner
        theme = "friday_team_social"
        slots[6] = _slot("work_afternoon", _t(14, 15), _t(15, 45),
                         "Wrap up weekly tasks and hand off to teammates",
                         ["Working at desk", "Talking to colleagues"], "indoor", "daytime")
        slots[7] = _slot("commute_evening", _t(16, 0), _t(16, 45),
                         "Commute to downtown for team gathering",
                         ["Car - commuting, road trip", "Walking on street"], "mixed", "daytime")
        slots[8] = _slot("home_evening", _t(17, 0), _t(18, 0),
                         "Quick refresh at home before heading out",
                         ["Daily hygiene", "Cooking"], "indoor", "twilight")

    # Evening: pick exercise + leisure (varied)
    exercise = copy.deepcopy(rng.choice(EXERCISE_OPTIONS))
    if wd == 4:  # Friday: skip gym, go to team dinner instead
        exercise = None

    leisure_idx = (day_index * 3 + wd) % len(EVENING_LEISURE_OPTIONS)
    leisure = copy.deepcopy(EVENING_LEISURE_OPTIONS[leisure_idx])

    if wd == 4:
        slots.append(_slot("team_dinner", _t(18, 30), _t(20, 30),
                           "Team dinner at restaurant with colleagues",
                           ["Eating at a restaurant", "Talking to colleagues", "Attending a party"],
                           "indoor", "nighttime"))
        slots.append(_slot("friday_night_out", _t(20, 45), _t(22, 0),
                           "Casual hangout at bar or dessert shop with teammates",
                           ["Hanging out with friends at a bar", "Eating at a restaurant"],
                           "indoor", "nighttime"))
    else:
        if exercise:
            ex_start = _t(19, 45) if wd != 3 else _t(20, 0)
            ex_end = _fmt_t(_parse_t(ex_start) + exercise.get("duration_min", 55))
            slots.append(_slot(exercise["slot_id"], ex_start, ex_end,
                               exercise["plan_chunk"], exercise["matched_scenarios"],
                               exercise["location"], exercise["time_period"]))
            lv_start = _fmt_t(_parse_t(ex_end) + 15)
        else:
            lv_start = _t(19, 30)
        lv_end = _fmt_t(_parse_t(lv_start) + leisure.get("duration_min", 55))
        slots.append(_slot(leisure["slot_id"], lv_start, lv_end,
                           leisure["plan_chunk"], leisure["matched_scenarios"],
                           leisure["location"], leisure["time_period"]))

    return slots, theme, anomalies


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
        _slot("wake_home", _t(8, 0), _t(9, 0),
              "Early weekend wake-up, pack snacks and hiking gear",
              ["Daily hygiene", "Cooking", "Making coffee"], "indoor", "daytime"),
        _slot("drive_trailhead", _t(9, 30), _t(10, 30),
              "Drive to trailhead for day hike",
              ["Car - commuting, road trip"], "mixed", "daytime"),
        _slot("hiking", _t(10, 45), _t(13, 30),
              "Long hiking on mountain trail with scenic views",
              ["Hiking", "Walking on street"], "outdoor", "daytime"),
        _slot("trail_lunch", _t(13, 45), _t(14, 30),
              "Picnic lunch on the trail",
              ["BBQ'ing/picnics", "Eating", "Outdoor cooking"], "outdoor", "daytime"),
        _slot("hiking_return", _t(14, 45), _t(16, 30),
              "Continue hiking and descend back to trailhead",
              ["Hiking", "Tourism"], "outdoor", "daytime"),
        _slot("drive_home", _t(17, 0), _t(18, 0),
              "Drive home, tired but satisfied",
              ["Car - commuting, road trip"], "mixed", "twilight"),
        _slot("recovery_home", _t(18, 30), _t(20, 0),
              "Shower, simple dinner, stretch sore muscles",
              ["Cooking", "Daily hygiene", "Working out at home"], "indoor", "nighttime"),
        _slot("early_sleep", _t(20, 30), _t(21, 30),
              "Early to bed after exhausting hike",
              ["Sleeping", "Reading books"], "indoor", "nighttime"),
    ]


def _saturday_cycling() -> list[Slot]:
    return [
        _slot("wake_home", _t(8, 30), _t(9, 30),
              "Leisurely brunch at home",
              ["Making coffee", "Cooking", "Daily hygiene"], "indoor", "daytime"),
        _slot("cycling", _t(10, 0), _t(12, 0),
              "Cycling along riverside bike path",
              ["Cycling / jogging", "Going to the park"], "outdoor", "daytime"),
        _slot("farmers_market", _t(12, 30), _t(13, 30),
              "Browse farmers market, buy fresh produce",
              ["Grocery shopping indoors", "Clothes, other shopping", "Walking on street"],
              "outdoor", "daytime"),
        _slot("home_cooking", _t(14, 0), _t(15, 30),
              "Experiment with new recipes using market finds",
              ["Cooking", "Crafting/knitting/sewing/drawing/painting"], "indoor", "daytime"),
        _slot("social_outdoor", _t(16, 0), _t(17, 30),
              "Meet friends at park for frisbee and chat",
              ["Frisbee", "Outdoor social (includes campfire)", "Talking with friends/housemates"],
              "outdoor", "daytime"),
        _slot("dining_out", _t(19, 0), _t(20, 30),
              "Dinner at a new restaurant downtown",
              ["Eating at a restaurant"], "indoor", "nighttime"),
        _slot("leisure_games", _t(21, 0), _t(22, 0),
              "Play board games at friend's home",
              ["Playing board games", "Playing games / video games"], "indoor", "nighttime"),
    ]


def _saturday_expedition() -> list[Slot]:
    return [
        _slot("wake_home", _t(7, 30), _t(8, 30),
              "Prepare for full-day outdoor expedition",
              ["Daily hygiene", "Cooking"], "indoor", "daytime"),
        _slot("outdoor_activity", _t(9, 0), _t(12, 0),
              "Full-day outdoor adventure: hiking and nature photography",
              ["Hiking", "Taking photos in photography studio", "Tourism"],
              "outdoor", "daytime"),
        _slot("outdoor_lunch", _t(12, 15), _t(13, 0),
              "Eat at outdoor hawker center",
              ["Eating in hawker center", "Eating"], "outdoor", "daytime"),
        _slot("outdoor_afternoon", _t(13, 30), _t(16, 30),
              "Explore new neighborhood on foot, visit local shops",
              ["Walking on street", "Tourism", "Clothes, other shopping"],
              "outdoor", "daytime"),
        _slot("commute_home", _t(17, 0), _t(18, 0),
              "Bus ride home, review photos on phone",
              ["Bus", "On a screen (phone/laptop)"], "mixed", "twilight"),
        _slot("home_evening", _t(18, 30), _t(20, 0),
              "Cook hearty dinner, share photos with friends",
              ["Cooking", "Video call", "Talking with friends/housemates"],
              "indoor", "nighttime"),
    ]


def _sunday_movie() -> list[Slot]:
    return [
        _slot("wake_home", _t(9, 30), _t(10, 30),
              "Sleep in, lazy breakfast in pajamas",
              ["Daily hygiene", "Making coffee", "Cooking"], "indoor", "daytime"),
        _slot("home_chores", _t(11, 0), _t(12, 0),
              "Light laundry and tidying up apartment",
              ["Cleaning / laundry", "Ironing"], "indoor", "daytime"),
        _slot("movie_marathon", _t(12, 30), _t(15, 30),
              "Movie marathon at home on the couch",
              ["Watching tv", "Watching movies at the cinema"], "indoor", "daytime"),
        _slot("home_lunch", _t(15, 45), _t(16, 30),
              "Order delivery and eat while watching",
              ["Eating", "Drive-thru food"], "indoor", "daytime"),
        _slot("movie_afternoon", _t(16, 45), _t(19, 0),
              "Continue movie series, take short breaks",
              ["Watching tv"], "indoor", "twilight"),
        _slot("home_cooking", _t(19, 30), _t(20, 30),
              "Simple home-cooked dinner",
              ["Cooking"], "indoor", "nighttime"),
        _slot("leisure_read", _t(21, 0), _t(22, 0),
              "Read a few chapters before early sleep",
              ["Reading books"], "indoor", "nighttime"),
    ]


def _sunday_cozy() -> list[Slot]:
    return [
        _slot("wake_home", _t(9, 0), _t(10, 0),
              "Slow morning with pour-over coffee and podcast",
              ["Making coffee", "Listening to music", "Daily hygiene"],
              "indoor", "daytime"),
        _slot("home_hobby", _t(10, 30), _t(12, 30),
              "Crafting project and organizing bookshelf",
              ["Crafting/knitting/sewing/drawing/painting", "Reading books"],
              "indoor", "daytime"),
        _slot("home_cooking", _t(13, 0), _t(14, 30),
              "Bake bread and prepare elaborate lunch",
              ["Cooking", "Baker"], "indoor", "daytime"),
        _slot("indoor_relax", _t(15, 0), _t(17, 0),
              "Nap, browse phone, play with pet",
              ["Playing with pets", "On a screen (phone/laptop)", "Sleeping"],
              "indoor", "daytime"),
        _slot("home_chores", _t(17, 30), _t(18, 30),
              "Meal prep for the coming work week",
              ["Cooking", "Cleaning / laundry"], "indoor", "twilight"),
        _slot("leisure_early_sleep", _t(19, 30), _t(21, 0),
              "Early night routine: skincare, journal, sleep by 9:30",
              ["Daily hygiene", "Sleeping", "Reading books"], "indoor", "nighttime"),
    ]


def _sunday_gaming() -> list[Slot]:
    return [
        _slot("wake_home", _t(10, 0), _t(11, 0),
              "Brunch and check gaming news online",
              ["Cooking", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("gaming_session", _t(11, 30), _t(14, 0),
              "Long gaming session: new RPG release",
              ["Playing games / video games"], "indoor", "daytime"),
        _slot("online_gaming", _t(14, 30), _t(16, 30),
              "Online multiplayer with friends",
              ["Playing games / video games", "Video call"], "indoor", "daytime"),
        _slot("home_cooking", _t(17, 0), _t(18, 30),
              "Cook comfort food for dinner",
              ["Cooking"], "indoor", "twilight"),
        _slot("gaming_evening", _t(19, 0), _t(21, 0),
              "Continue gaming, try new game mode",
              ["Playing games / video games", "Watching tv"], "indoor", "nighttime"),
        _slot("leisure_music", _t(21, 30), _t(22, 30),
              "Wind down with music playlist",
              ["Listening to music"], "indoor", "nighttime"),
    ]


# ---------------------------------------------------------------------------
# Anomaly day schedules (override entire day)
# ---------------------------------------------------------------------------
def _anomaly_rain_day() -> list[Slot]:
    return [
        _slot("wake_home", _t(7, 10), _t(7, 50),
              "Wake up to heavy rain, check weather app",
              ["Daily hygiene", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("commute_morning", _t(8, 10), _t(8, 55),
              "Rainy commute with umbrella, traffic slower than usual",
              ["Car - commuting, road trip", "Bus", "Walking on street"],
              "mixed", "daytime"),
        _slot("work_morning", _t(9, 10), _t(10, 30),
              "Work at desk, many colleagues working from home",
              ["Working at desk", "Video call"], "indoor", "daytime"),
        _slot("work_focus", _t(10, 45), _t(12, 0),
              "Focused coding session, rain visible through window",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("lunch_indoor", _t(12, 15), _t(13, 0),
              "Lunch at office cafeteria, rain still pouring",
              ["Eating at the cafeteria", "Eating"], "indoor", "daytime"),
        _slot("rain_cancel_indoor", _t(13, 15), _t(14, 0),
              "Cancelled outdoor walk due to rain; coffee break indoors instead",
              ["Making coffee", "Talking to colleagues", "Hanging out at a coffee shop"],
              "indoor", "daytime"),
        _slot("work_afternoon", _t(14, 15), _t(16, 0),
              "Afternoon meetings via video call",
              ["Video call", "Participating in a meeting"], "indoor", "daytime"),
        _slot("commute_evening", _t(16, 15), _t(17, 5),
              "Rainy evening commute, wet shoes",
              ["Car - commuting, road trip", "Bus"], "mixed", "daytime"),
        _slot("home_evening", _t(17, 20), _t(18, 20),
              "Cook warm soup dinner, dry clothes",
              ["Cooking", "Cleaning / laundry"], "indoor", "twilight"),
        _slot("leisure_tv", _t(19, 0), _t(20, 30),
              "Watch comfort TV show, glad to stay indoors",
              ["Watching tv"], "indoor", "nighttime"),
        _slot("leisure_read", _t(20, 45), _t(21, 45),
              "Read by the window, rain still falling",
              ["Reading books"], "indoor", "nighttime"),
    ]


def _anomaly_business_trip() -> list[Slot]:
    return [
        _slot("wake_hotel", _t(6, 30), _t(7, 15),
              "Wake up in hotel, quick hygiene and pack bag",
              ["Daily hygiene", "Cleaning / laundry"], "indoor", "daytime"),
        _slot("travel_airport", _t(7, 30), _t(9, 30),
              "Taxi to airport, check in and wait at gate",
              ["Car - commuting, road trip", "Walking on street", "Talking on the phone"],
              "mixed", "daytime"),
        _slot("flight", _t(10, 0), _t(12, 30),
              "Flight to client city, work on laptop during flight",
              ["On a screen (phone/laptop)", "Reading books"], "mixed", "daytime"),
        _slot("client_meeting", _t(13, 30), _t(15, 30),
              "Client site visit and product demo meeting",
              ["Participating in a meeting", "Working at desk", "Talking to colleagues"],
              "indoor", "daytime"),
        _slot("business_lunch", _t(15, 45), _t(16, 45),
              "Business lunch with client team",
              ["Eating at a restaurant", "Talking to colleagues"], "indoor", "daytime"),
        _slot("hotel_work", _t(17, 30), _t(19, 0),
              "Work from hotel room: follow-up emails and slides",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "twilight"),
        _slot("solo_dinner", _t(19, 30), _t(20, 30),
              "Solo dinner at hotel restaurant",
              ["Eating at a restaurant", "Eating"], "indoor", "nighttime"),
        _slot("leisure_early_sleep", _t(21, 0), _t(22, 0),
              "Early sleep in unfamiliar hotel room",
              ["Sleeping", "Daily hygiene"], "indoor", "nighttime"),
    ]


def _anomaly_overtime() -> list[Slot]:
    return [
        _slot("wake_home", _t(6, 50), _t(7, 35),
              "Wake early, anxious about project deadline",
              ["Daily hygiene", "Making coffee"], "indoor", "daytime"),
        _slot("commute_morning", _t(7, 50), _t(8, 30),
              "Quick commute to office",
              ["Car - commuting, road trip", "Bus"], "mixed", "daytime"),
        _slot("work_morning", _t(8, 45), _t(12, 0),
              "Intense morning: fix critical bugs before presentation",
              ["Working at desk", "Fixing PC", "On a screen (phone/laptop)"],
              "indoor", "daytime"),
        _slot("lunch_quick", _t(12, 10), _t(12, 40),
              "Quick desk lunch, no time to go out",
              ["Eating", "Working at desk"], "indoor", "daytime"),
        _slot("project_presentation", _t(13, 0), _t(15, 0),
              "High-stakes project presentation to leadership",
              ["Participating in a meeting", "Writing on whiteboard"],
              "indoor", "daytime"),
        _slot("overtime_work", _t(15, 15), _t(19, 30),
              "Overtime: patch issues found during review, team stays late",
              ["Working at desk", "Talking to colleagues", "Video call"],
              "indoor", "nighttime"),
        _slot("late_commute", _t(19, 45), _t(20, 30),
              "Late night commute home, city lights",
              ["Car - commuting, road trip", "Bus"], "mixed", "nighttime"),
        _slot("takeout_dinner", _t(20, 45), _t(21, 30),
              "Exhausted, order takeout and eat on couch",
              ["Drive-thru food", "Eating"], "indoor", "nighttime"),
        _slot("leisure_early_sleep", _t(21, 45), _t(22, 15),
              "Crash into bed immediately",
              ["Sleeping"], "indoor", "nighttime"),
    ]


def _anomaly_impromptu_date() -> list[Slot]:
    return [
        _slot("wake_home", _t(7, 5), _t(7, 50),
              "Normal morning routine",
              ["Daily hygiene", "Cooking", "Making coffee"], "indoor", "daytime"),
        _slot("commute_morning", _t(8, 5), _t(8, 45),
              "Commute to office",
              ["Car - commuting, road trip", "Bus"], "mixed", "daytime"),
        _slot("work_morning", _t(9, 0), _t(12, 0),
              "Regular work morning",
              ["Working at desk", "Talking to colleagues"], "indoor", "daytime"),
        _slot("lunch_out", _t(12, 15), _t(13, 0),
              "Lunch with coworker, receive impromptu date invitation",
              ["Eating at the cafeteria", "Talking to colleagues"], "indoor", "daytime"),
        _slot("work_afternoon", _t(13, 15), _t(17, 0),
              "Afternoon work, leave on time for date",
              ["Working at desk", "On a screen (phone/laptop)"], "indoor", "daytime"),
        _slot("commute_date", _t(17, 15), _t(17, 45),
              "Commute to downtown restaurant",
              ["Car - commuting, road trip", "Walking on street"], "mixed", "daytime"),
        _slot("impromptu_date", _t(18, 0), _t(20, 0),
              "Impromptu dinner date at nice restaurant",
              ["Eating at a restaurant", "Talking with friends/housemates"],
              "indoor", "nighttime"),
        _slot("evening_walk", _t(20, 15), _t(21, 0),
              "After-dinner walk along the waterfront",
              ["Walking on street", "Talking on the phone"], "outdoor", "nighttime"),
        _slot("leisure_early_sleep", _t(21, 30), _t(22, 15),
              "Return home happy, get ready for bed",
              ["Daily hygiene", "Sleeping"], "indoor", "nighttime"),
    ]


def _anomaly_doctor_visit() -> list[Slot]:
    return [
        _slot("wake_home", _t(7, 0), _t(7, 40),
              "Wake up, remember doctor appointment today",
              ["Daily hygiene", "Making coffee"], "indoor", "daytime"),
        _slot("commute_morning", _t(8, 0), _t(8, 35),
              "Commute to office",
              ["Car - commuting, road trip"], "mixed", "daytime"),
        _slot("work_morning", _t(8, 45), _t(11, 30),
              "Work before leaving for appointment",
              ["Working at desk"], "indoor", "daytime"),
        _slot("doctor_appointment", _t(11, 45), _t(13, 0),
              "Doctor appointment: annual check-up",
              ["Appointments: doctor, dentist", "Talking on the phone"],
              "indoor", "daytime"),
        _slot("lunch_out", _t(13, 15), _t(14, 0),
              "Grab lunch near clinic",
              ["Eating at a restaurant", "Eating"], "indoor", "daytime"),
        _slot("work_afternoon", _t(14, 15), _t(16, 30),
              "Return to office for afternoon work",
              ["Working at desk", "Video call"], "indoor", "daytime"),
        _slot("commute_evening", _t(16, 45), _t(17, 30),
              "Commute home",
              ["Car - commuting, road trip"], "mixed", "daytime"),
        _slot("home_evening", _t(17, 45), _t(18, 45),
              "Cook light dinner, rest after appointment",
              ["Cooking"], "indoor", "twilight"),
        _slot("exercise_home", _t(19, 0), _t(19, 45),
              "Gentle home stretching, skip intense workout",
              ["Working out at home", "Yoga practice"], "indoor", "nighttime"),
        _slot("leisure_tv", _t(20, 0), _t(21, 30),
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

    jittered = []
    for s in slots:
        offset = 0
        dur_delta = 0
        sid = s["slot_id"]
        if sid == "wake_home" or sid == "wake_hotel":
            offset = wake_jitter
        elif "commute" in sid or "drive" in sid or "travel" in sid:
            offset = wake_jitter + commute_jitter
        elif sid.startswith("exercise") or sid.startswith("work_morning"):
            offset = wake_jitter + commute_jitter // 2
        elif sid.startswith("leisure") or sid.startswith("team") or sid.startswith("friday"):
            dur_delta = rng.randint(-10, 15)
            offset = wake_jitter + commute_jitter // 2
        else:
            offset = wake_jitter + commute_jitter // 2
        jittered.append(_shift_slot(s, offset, dur_delta))

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
