"""Configuration for the N-day / 8-hour daily routine lifelog builder."""

from pathlib import Path

PROJECT_NAME = "EgoTailor"
# 3.0: Stage 1 is corpus-driven (persona_generator) rather than hand-written.
VERSION = "3.0.0"

GENERATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GENERATION_DIR.parent
# Local Ego4D metadata filtered to videos present under EGO4D_VIDEO_ROOT.
# Regenerate with: python scripts/filter_local_video_info.py
EGO4DINFO_PATH = GENERATION_DIR / "ego4d_info"
EGO4D_VIDEO_ROOT = Path("/mnt/data_oss/raw_data/Ego4d/v2/full_scale")
# (The upstream X-LeBench metadata path used to live here. Nothing read it —
# scripts/filter_local_video_info.py carries its own --source default.)

# Dataset scale. Ego4D supports ~35 days at ~26 slots/day without clip reuse;
# past that the persona collapses onto its deepest-supply scenarios and
# persona_generator.check() reports the entropy guardrail. See that module.
TOTAL_DAYS = 30
HOURS_PER_DAY = 8  # soft reference; actual hours = sum of 1 clip per plan slot
TARGET_SECONDS_PER_DAY = HOURS_PER_DAY * 3600  # 28,800 (metadata only)

# Calendar anchor (Day 1)
START_DATE = "2026-01-05"  # Monday
DAY_START_TIME = "07:00:00"

# Persona (single subject).
# PERSONA_LOCATION is read only by the legacy schedule_templates path. The
# corpus-driven generator leaves location unset: Ego4D collects by country
# theme, so pinning video_source costs coverage without buying anything the
# downstream analysis reads. See persona_generator.POOL_COUNTRY_FILTER.
PERSONA_LOCATION = "USA"
PERSONA_MBti = "ENFP"

# Removed here because nothing referenced them: GAP_BETWEEN_CLIPS_SEC (the
# chain placement policy leaves no gap), MIN_CLIPS_PER_SCENE (superseded by the
# quota's per-cell candidate ceilings), ONE_CLIP_PER_PLAN (always true).

# Video retrieval
IOU_THRESHOLD = 0.2
MAX_VIDEO_DURATION = 3600
MIN_VIDEO_DURATION = 120
ALLOW_CROSS_DAY_REUSE = False

# Retrieval fallback ladder, tried in order until a rung yields candidates.
# Each rung is (tier_name, strict_location, min_scenario_iou).
# Note T2+ set min_scenario_iou=0.0, i.e. the plan's requested scenarios are
# dropped entirely — a clip retrieved there is NOT scenario-aligned.
RETRIEVAL_TIERS = [
    ("T1_scenario_match", True, IOU_THRESHOLD),
    ("T2_any_scenario", True, 0.0),
    ("T3_any_location", False, 0.0),
    ("T4_unconstrained", True, 0.0),  # also drops scene/time filters
]

# Lowest tier whose clips may actually be placed. A slot whose best available
# tier ranks below this is left unfilled and recorded in
# statistics["unfilled_slots"] rather than filled with a mismatched clip.
# Set to "T4_unconstrained" to restore the pre-2.3 always-fill behaviour.
MIN_RETRIEVAL_TIER = "T1_scenario_match"

# Output paths (under EgoTailor/output/)
OUTPUT_PATH = PROJECT_ROOT / "output"
PERSONA_PATH = OUTPUT_PATH / "persona"
LIFELOG_PATH = OUTPUT_PATH / "lifelog"
DAYS_PATH = OUTPUT_PATH / "days"


def resolve_lifelog(explicit: "Path | str | None" = None) -> Path:
    """Locate the lifelog to read, or explain why it cannot be found.

    The filename carries both the persona id and the run length, and both
    change whenever the dataset is regenerated. Four call sites each hard-coded
    ``lifelog_egotailor_usa_enfp_21d.json`` as an argparse default, so every one
    of them went stale the moment TOTAL_DAYS or the persona changed — and
    stayed silently wrong, because a bad default only surfaces when somebody
    actually runs the script.

    Resolution order: an explicit path wins; then a run matching the configured
    TOTAL_DAYS; then, if exactly one lifelog exists, that one. Ambiguity raises
    with the candidates listed rather than picking for you — silently analysing
    the wrong persona is the failure this function exists to prevent.
    """
    if explicit is not None:
        return Path(explicit)

    found = sorted(LIFELOG_PATH.glob("lifelog_*d.json"))
    if not found:
        raise FileNotFoundError(
            f"No lifelog in {LIFELOG_PATH}. Build one first:\n"
            f"  python -m generation.build_lifelog\n"
            f"or pass an explicit --lifelog path."
        )

    matching = [p for p in found if p.stem.endswith(f"_{TOTAL_DAYS}d")]
    candidates = matching or found
    if len(candidates) == 1:
        return candidates[0]

    listing = "\n".join(f"  {p.name}" for p in candidates)
    raise FileNotFoundError(
        f"{len(candidates)} lifelogs in {LIFELOG_PATH} — pass --lifelog to choose:\n"
        f"{listing}"
    )
