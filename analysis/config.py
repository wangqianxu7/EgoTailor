"""Analysis pipeline configuration."""

from __future__ import annotations

import os
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
GENERATION_CONFIG = PROJECT_ROOT / "generation" / "config.py"

# Lifelog input
DEFAULT_LIFELOG = PROJECT_ROOT / "output" / "lifelog" / "lifelog_egotailor_usa_enfp_21d.json"

# Ego4D local video root: {video_uid}.mp4
EGO4D_VIDEO_ROOT = Path(os.getenv("EGO4D_VIDEO_ROOT", "/mnt/data_oss/raw_data/Ego4d/v2/full_scale"))

# vLLM OpenAI-compatible API (multimodal)
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8080/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen3-VL-8B-Instruct")
VLLM_TIMEOUT_SEC = int(os.getenv("VLLM_TIMEOUT_SEC", "180"))
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "1536"))

# Frame sampling for MLLM (None = adaptive by duration; set int to override)
FRAMES_PER_CLIP: int | None = (
    int(os.getenv("FRAMES_PER_CLIP")) if os.getenv("FRAMES_PER_CLIP") else None
)
FRAME_MAX_SIZE = int(os.getenv("FRAME_MAX_SIZE", "768"))  # max edge px

# Hierarchical RAG retrieval top-k
TOP_K = {
    "clip": 8,      # minute-level (~10-20 min clips)
    "hour": 4,      # hour blocks
    "day": 3,       # single days
    "period": 2,    # multi-day windows (week / full lifelog)
}

# Auto interest-mining queries (multi-level)
DEFAULT_INTEREST_QUERIES = [
    "outdoor activities exercise sports preferences",
    "food cooking eating dining habits",
    "work desk office meeting professional routines",
    "social interaction friends family communication",
    "commute transportation travel patterns",
    "home leisure entertainment hobbies relaxation",
    "shopping errands consumer behavior",
]

# Output
OUTPUT_PATH = PROJECT_ROOT / "output" / "analysis"
INDEX_PATH = OUTPUT_PATH / "hierarchical_index.json"
REPORT_PATH = OUTPUT_PATH / "interest_report.json"
VIDEO_REGISTRY_PATH = OUTPUT_PATH / "video_registry.json"
VLM_PROFILE_PATH = OUTPUT_PATH / "vlm_behavior_profile.json"
