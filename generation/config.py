"""Configuration for the 21-day / 8-hour daily routine lifelog builder."""

from pathlib import Path

PROJECT_NAME = "EgoTailor"
VERSION = "2.0.0"

GENERATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GENERATION_DIR.parent
EGO4DINFO_PATH = PROJECT_ROOT.parent / "X-LeBench" / "generation" / "ego4d_info"

# Dataset scale
TOTAL_DAYS = 21
HOURS_PER_DAY = 8
TARGET_SECONDS_PER_DAY = HOURS_PER_DAY * 3600  # 28,800

# Calendar anchor (Day 1)
START_DATE = "2026-01-05"  # Monday
DAY_START_TIME = "07:00:00"
GAP_BETWEEN_CLIPS_SEC = 300  # 5 min between clips

# Persona (single subject)
PERSONA_LOCATION = "USA"
PERSONA_MBti = "ENFP"

# Scene coverage quotas per day
MIN_CLIPS_PER_SCENE = {
    "indoor": 4,
    "outdoor": 2,
    "mixed": 2,
}

# Video retrieval
IOU_THRESHOLD = 0.2
MAX_VIDEO_DURATION = 3600
MIN_VIDEO_DURATION = 120
ALLOW_CROSS_DAY_REUSE = False

# Output paths (under EgoTailor/output/)
OUTPUT_PATH = PROJECT_ROOT / "output"
PERSONA_PATH = OUTPUT_PATH / "persona"
LIFELOG_PATH = OUTPUT_PATH / "lifelog"
DAYS_PATH = OUTPUT_PATH / "days"
