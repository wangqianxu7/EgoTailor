"""Configuration for the 21-day / 8-hour daily routine lifelog builder."""

from pathlib import Path

PROJECT_NAME = "EgoTailor"
VERSION = "2.2.0"

GENERATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GENERATION_DIR.parent
# Local Ego4D metadata filtered to videos present under EGO4D_VIDEO_ROOT.
# Regenerate with: python scripts/filter_local_video_info.py
EGO4DINFO_PATH = GENERATION_DIR / "ego4d_info"
EGO4D_VIDEO_ROOT = Path("/mnt/data_oss/raw_data/Ego4d/v2/full_scale")
# Full upstream metadata (X-LeBench); used only by the filter script.
XLEBENCH_EGO4DINFO_PATH = PROJECT_ROOT.parent / "X-LeBench" / "generation" / "ego4d_info"

# Dataset scale
TOTAL_DAYS = 21
HOURS_PER_DAY = 8  # soft reference; actual hours = sum of 1 clip per plan slot
TARGET_SECONDS_PER_DAY = HOURS_PER_DAY * 3600  # 28,800 (metadata only)

# Calendar anchor (Day 1)
START_DATE = "2026-01-05"  # Monday
DAY_START_TIME = "07:00:00"
# Unused when placement_policy=one_clip_per_plan_align_or_chain (chain has 0 gap on overflow)
GAP_BETWEEN_CLIPS_SEC = 0

# Persona (single subject)
PERSONA_LOCATION = "USA"
PERSONA_MBti = "ENFP"

# Deprecated for packing; kept for backward compatibility / docs
MIN_CLIPS_PER_SCENE = {
    "indoor": 4,
    "outdoor": 2,
    "mixed": 2,
}

# One Ego4D video per daily-plan slot
ONE_CLIP_PER_PLAN = True

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
