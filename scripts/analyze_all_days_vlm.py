#!/usr/bin/env python3
"""
Full 21-day EgoTailor lifelog VLM analysis.

For every clip across all 21 days:
  1. Uniformly sample N frames (default 64) from the local Ego4D mp4
  2. Send frames to the local Qwen3-VL vLLM server (batched to fit context)
  3. Parse per-clip behavior / preference signals
  4. Synthesize a global behavior-preference profile

Features:
  - tqdm progress bars (overall + per-video frame batches)
  - incremental checkpoint / resume
  - optional --limit / --day filter for smoke tests

Usage (from EgoTailor root):
  cd /root/EgoTailor

  # Full 21-day run (558 videos, ~64 frames each; frames saved to /root/egodaily)
  python scripts/analyze_all_days_vlm.py

  # Smoke test first 3 clips
  python scripts/analyze_all_days_vlm.py --limit 3

  # Custom frames output dir
  python scripts/analyze_all_days_vlm.py --frames-dir /root/egodaily

  # Only day 0
  python scripts/analyze_all_days_vlm.py --day 0

  # Resume after interrupt
  python scripts/analyze_all_days_vlm.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Allow `python scripts/analyze_all_days_vlm.py` from EgoTailor root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generation.config import TOTAL_DAYS  # noqa: E402
from analysis.config import (  # noqa: E402
    DEFAULT_LIFELOG,
    resolve_lifelog,
    EGO4D_VIDEO_ROOT,
    OUTPUT_PATH,
    VLLM_API_BASE,
    VLLM_MODEL,
)
from analysis.mllm_client import VLLMClient  # noqa: E402
from analysis.video_loader import (  # noqa: E402
    extract_frames,
    frames_to_base64_jpeg,
    resolve_video_path,
)


CLIP_SYSTEM = """You are an expert egocentric (first-person) video captioner and behavior analyst.
You will see uniformly sampled frames from ONE Ego4D video clip, plus lifelog metadata.

Follow this TWO-STEP process strictly:

STEP 1 — CAPTION (required first):
Write ONE complete, fluent paragraph (4–8 sentences) that narrates what happens in the clip
from a first-person / observer perspective. The paragraph must be continuous prose — not bullets,
not fragmented labels. Cover: setting/place, main actions in temporal order, objects handled,
people/social interaction if any, and atmosphere or intent when visible. You may lightly use
the Ego4D caption and scheduled plan as hints, but ground the paragraph in what the frames show.

STEP 2 — EXTRACT (from the caption only):
Using ONLY the paragraph you just wrote in STEP 1 (plus obvious consistency with the frames),
extract structured fields. Do not invent activities that are absent from the caption.

Return valid JSON only (no markdown), with this schema:
{
  "caption": "ONE complete fluent paragraph describing the whole clip...",
  "observed_activities": ["short activity phrases extracted from the caption"],
  "objects_and_places": ["objects / places mentioned or clearly implied in the caption"],
  "social_interaction": "none|minimal|moderate|high",
  "behavior_tags": ["commuting", "cooking", "desk_work", "exercise", "shopping", "leisure", ...],
  "preference_hypotheses": [
    {"topic": "...", "evidence": "quote or paraphrase from the caption", "confidence": "high|medium|low"}
  ],
  "habit_signals": ["routine-like signals grounded in the caption"],
  "plan_vs_observation": "how the captioned scene relates to the scheduled plan_chunk",
  "notable_details": ["concrete details taken from the caption"]
}

Important:
- `caption` MUST be a single coherent paragraph of natural language.
- All list/tag fields MUST be derived from that caption; if unsure, omit rather than guess.
"""

BATCH_MERGE_SYSTEM = """You merge multiple partial frame-batch analyses of the SAME video clip.

TWO-STEP process:
1) First write ONE unified, complete, fluent paragraph `caption` that blends all partial captions
   into a single coherent narrative of the whole clip (no bullet lists).
2) Then EXTRACT all structured fields strictly from that merged caption.

Keep the same JSON schema as a single clip analysis (including required `caption`).
Deduplicate, resolve conflicts conservatively, and raise confidence only when evidence
repeats across batches. Return valid JSON only."""

DAY_SYNTHESIS_SYSTEM = """Synthesize one day's clip-level VLM analyses into daily behavior patterns.

Prefer using each clip's fluent `caption` paragraph as primary evidence; use extracted tags
only as secondary support.

Return valid JSON only:
{
  "day_summary": "one fluent paragraph summarizing the day",
  "dominant_activities": ["..."],
  "habit_candidates": [{"pattern": "...", "evidence": "..."}],
  "preference_signals": [{"topic": "...", "evidence": "...", "confidence": "high|medium|low"}],
  "scene_mix": "indoor/outdoor/mixed qualitative note",
  "anomalies": ["..."]
}"""

GLOBAL_SYNTHESIS_SYSTEM = """You are profiling ONE person's 21-day egocentric lifelog.
Given day-level summaries (each with a fluent day_summary paragraph) and aggregated clip statistics,
infer durable behavior preferences and habits. Prefer paragraph evidence over tag counts.

Return valid JSON only:
{
  "summary": "2-4 sentence overall profile as fluent prose",
  "core_interests": [
    {"topic": "...", "evidence": "...", "confidence": "high|medium|low", "supporting_video_uids": ["..."]}
  ],
  "habitual_patterns": [
    {"pattern": "...", "time_context": "weekday morning / weekend evening / ...", "frequency_hint": "..."}
  ],
  "preferences": [
    {"category": "food|work|leisure|social|mobility|home|...", "preference": "...", "reasoning": "..."}
  ],
  "weekday_vs_weekend": "...",
  "lifestyle_traits": ["..."],
  "data_quality_note": "coverage, vision limits, caption reliance"
}"""


# Output dir carries the run length: a 30-day run must not overwrite the
# 21-day results sitting next to it. These are expensive to recompute.
DEFAULT_OUT_DIR = OUTPUT_PATH / f"full_{TOTAL_DAYS}d_vlm"
DEFAULT_FRAMES_DIR = Path("/root/egodaily")
CHECKPOINT_NAME = "clip_analyses_checkpoint.jsonl"
REPORT_NAME = f"full_{TOTAL_DAYS}d_behavior_profile.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze all 21-day EgoTailor clips with VLM (64 frames/video).")
    p.add_argument("--lifelog", type=Path, default=DEFAULT_LIFELOG)
    p.add_argument("--video-root", type=Path, default=EGO4D_VIDEO_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--frames-dir",
        type=Path,
        default=DEFAULT_FRAMES_DIR,
        help="Save sampled frames under this root (default /root/egodaily)",
    )
    p.add_argument("--no-save-frames", action="store_true", help="Do not write frame JPEGs to --frames-dir")
    p.add_argument("--vllm-base", default=VLLM_API_BASE)
    p.add_argument("--model", default=VLLM_MODEL)
    p.add_argument("--frames", type=int, default=64, help="Uniform frames sampled per video (default 64)")
    p.add_argument(
        "--frame-batch",
        type=int,
        default=8,
        help="Frames per VLM call (default 8). 64 frames -> 8 batched calls per video.",
    )
    p.add_argument("--frame-max-size", type=int, default=448, help="Max image edge px (smaller = safer context)")
    p.add_argument("--day", type=int, nargs="*", default=None, help="Only analyze these day indices (0-20)")
    p.add_argument("--limit", type=int, default=None, help="Only analyze first N clips (smoke test)")
    p.add_argument("--resume", action="store_true", help="Skip video_uids already in checkpoint")
    p.add_argument("--skip-day-synthesis", action="store_true")
    p.add_argument("--skip-global-synthesis", action="store_true")
    p.add_argument("--save-every", type=int, default=1, help="Flush checkpoint every N clips")
    p.add_argument("--timeout", type=int, default=300, help="Per-request timeout seconds")
    return p.parse_args()


def clip_frames_dir(frames_root: Path, clip: dict[str, Any]) -> Path:
    """/root/egodaily/day_00_2026-01-05/<video_uid>/"""
    day_idx = int(clip.get("day_index") or 0)
    date = clip.get("calendar_date") or "unknown"
    return frames_root / f"day_{day_idx:02d}_{date}" / clip["video_uid"]


def save_clip_frames(
    frames_bgr: list[Any],
    clip: dict[str, Any],
    frames_root: Path,
    sampling_meta: dict[str, Any],
    quality: int = 90,
) -> dict[str, Any]:
    """Persist sampled frames as JPEG + meta.json. Returns save info."""
    import cv2

    out_dir = clip_frames_dir(frames_root, clip)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for i, frame in enumerate(frames_bgr):
        # frames from OpenCV are BGR; write as JPEG directly
        path = out_dir / f"frame_{i:03d}.jpg"
        ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            saved_paths.append(str(path))

    meta = {
        "video_uid": clip["video_uid"],
        "day_index": clip.get("day_index"),
        "calendar_date": clip.get("calendar_date"),
        "slot_id": clip.get("slot_id"),
        "plan_chunk": clip.get("plan_chunk"),
        "start_timestamp": clip.get("start_timestamp"),
        "end_timestamp": clip.get("end_timestamp"),
        "duration": clip.get("duration"),
        "main_scene": clip.get("main_scene"),
        "video_scenarios": clip.get("video_scenarios"),
        "consolidated_summary": clip.get("consolidated_summary"),
        "n_frames_saved": len(saved_paths),
        "frame_files": [Path(p).name for p in saved_paths],
        "sampling": sampling_meta,
        "frames_dir": str(out_dir),
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {
        "frames_dir": str(out_dir),
        "n_frames_saved": len(saved_paths),
        "meta_path": str(meta_path),
    }


def load_clips(lifelog_path: Path, days: list[int] | None, limit: int | None) -> list[dict[str, Any]]:
    with open(lifelog_path) as f:
        lifelog = json.load(f)

    clips: list[dict[str, Any]] = []
    for day in lifelog["days"]:
        meta = day["metadata"]
        day_idx = meta["day_index"]
        if days is not None and day_idx not in days:
            continue
        for clip in day["memory_content"]:
            clips.append(
                {
                    **clip,
                    "day_index": day_idx,
                    "calendar_date": meta.get("calendar_date"),
                    "day_of_week": meta.get("day_of_week"),
                    "is_weekend": meta.get("is_weekend"),
                    "day_theme": meta.get("day_theme"),
                    "anomaly_events": meta.get("anomaly_events", []),
                }
            )
    if limit is not None:
        clips = clips[:limit]
    return clips


def load_done_uids(checkpoint_path: Path) -> set[str]:
    done: set[str] = set()
    if not checkpoint_path.exists():
        return done
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = row.get("video_uid")
            if uid and row.get("status") in ("ok", "no_video", "error"):
                # resume only skips successful / terminal entries; allow retry of error if desired later
                if row.get("status") == "ok" or row.get("status") == "no_video":
                    done.add(uid)
    return done


def append_checkpoint(checkpoint_path: Path, row: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_checkpoint_rows(checkpoint_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not checkpoint_path.exists():
        return rows
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # keep last entry per uid
    by_uid: dict[str, dict[str, Any]] = {}
    for r in rows:
        uid = r.get("video_uid")
        if uid:
            by_uid[uid] = r
    return list(by_uid.values())


def safe_json_loads(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1]
            if t.startswith("json"):
                t = t[4:]
    return json.loads(t.strip())


def chunked(xs: list[Any], n: int) -> list[list[Any]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def build_clip_prompt(clip: dict[str, Any], batch_idx: int, n_batches: int) -> str:
    return (
        f"## Lifelog metadata\n"
        f"- video_uid: {clip['video_uid']}\n"
        f"- day_index: {clip.get('day_index')} date={clip.get('calendar_date')} "
        f"({clip.get('day_of_week')}, weekend={clip.get('is_weekend')})\n"
        f"- day_theme: {clip.get('day_theme')}\n"
        f"- anomalies: {clip.get('anomaly_events')}\n"
        f"- time: {clip.get('start_timestamp')} -> {clip.get('end_timestamp')}\n"
        f"- slot_id: {clip.get('slot_id')}\n"
        f"- scheduled plan: {clip.get('plan_chunk')}\n"
        f"- main_scene: {clip.get('main_scene')}\n"
        f"- video_scenarios: {', '.join(clip.get('video_scenarios') or [])}\n"
        f"- Ego4D caption (hint only): {clip.get('consolidated_summary')}\n"
        f"- duration_sec: {clip.get('duration')}\n\n"
        f"## Frame batch {batch_idx + 1}/{n_batches}\n"
        f"Instructions:\n"
        f"1) FIRST write a complete, fluent paragraph `caption` describing what happens "
        f"in these frames (continuous prose, 4–8 sentences).\n"
        f"2) THEN extract activities/tags/preferences FROM that caption.\n"
        f"Return JSON only, with `caption` as the first field."
    )


def analyze_one_clip(
    mllm: VLLMClient,
    clip: dict[str, Any],
    frames: int,
    frame_batch: int,
    frame_max_size: int,
    video_root: Path,
    pbar_desc: str,
    frames_dir: Path | None = None,
) -> dict[str, Any]:
    video_path = resolve_video_path(clip["video_uid"], video_root)
    frames_bgr: list[Any] = []
    sampling_meta: dict[str, Any] = {}
    if video_path is not None:
        frames_bgr, sampling_meta = extract_frames(
            video_path,
            num_frames=frames,
            max_size=frame_max_size,
            duration_sec=clip.get("duration"),
        )

    base: dict[str, Any] = {
        "video_uid": clip["video_uid"],
        "day_index": clip.get("day_index"),
        "calendar_date": clip.get("calendar_date"),
        "day_of_week": clip.get("day_of_week"),
        "slot_id": clip.get("slot_id"),
        "plan_chunk": clip.get("plan_chunk"),
        "main_scene": clip.get("main_scene"),
        "video_scenarios": clip.get("video_scenarios"),
        "consolidated_summary": clip.get("consolidated_summary"),
        "start_timestamp": clip.get("start_timestamp"),
        "end_timestamp": clip.get("end_timestamp"),
        "duration": clip.get("duration"),
        "day_theme": clip.get("day_theme"),
        "anomaly_events": clip.get("anomaly_events"),
        "video_path": str(video_path) if video_path else None,
        "frame_count": len(frames_bgr),
        "sampling": sampling_meta.get("sampling"),
        "duration_min": sampling_meta.get("duration_min"),
    }

    if frames_dir is not None and frames_bgr:
        save_info = save_clip_frames(frames_bgr, clip, frames_dir, sampling_meta)
        base["frames_saved_dir"] = save_info["frames_dir"]
        base["n_frames_saved"] = save_info["n_frames_saved"]

    if not frames_bgr:
        base["status"] = "no_video"
        base["vision_used"] = False
        # caption-only fallback
        text = mllm.chat_text(
            CLIP_SYSTEM,
            build_clip_prompt(clip, 0, 1)
            + "\n(No local frames available; write the fluent paragraph caption from metadata/"
            "Ego4D hint only, then extract fields from that caption.)",
            temperature=0.1,
        )
        try:
            base["analysis"] = safe_json_loads(text)
        except Exception:
            base["analysis"] = {"notable_details": [text]}
            base["raw_analysis"] = text
        return base

    frames_b64: list[str] = frames_to_base64_jpeg(frames_bgr)
    batches = chunked(frames_b64, max(1, frame_batch))
    partials: list[dict[str, Any]] = []

    batch_iter = tqdm(
        enumerate(batches),
        total=len(batches),
        desc=pbar_desc,
        leave=False,
        unit="batch",
        dynamic_ncols=True,
    )
    for bi, batch in batch_iter:
        prompt = build_clip_prompt(clip, bi, len(batches))
        try:
            partial = mllm.chat_vision_json(CLIP_SYSTEM, prompt, batch, temperature=0.1)
        except Exception as exc:
            raw = mllm.chat_vision(CLIP_SYSTEM, prompt, batch, temperature=0.1)
            try:
                partial = safe_json_loads(raw)
            except Exception:
                partial = {"notable_details": [raw], "parse_error": str(exc)}
        partials.append(partial)

    if len(partials) == 1:
        merged = partials[0]
    else:
        merge_user = (
            f"video_uid={clip['video_uid']}\n"
            f"plan={clip.get('plan_chunk')}\n"
            f"ego4d_caption_hint={clip.get('consolidated_summary')}\n\n"
            f"Merge all partial analyses. First produce ONE fluent paragraph `caption`, "
            f"then extract structured fields from it.\n\n"
            f"partial_analyses=\n{json.dumps(partials, ensure_ascii=False)[:12000]}"
        )
        try:
            merged = mllm.chat_json(BATCH_MERGE_SYSTEM, merge_user, temperature=0.1)
        except Exception:
            # heuristic merge
            merged = heuristic_merge(partials)

    base["status"] = "ok"
    base["vision_used"] = True
    base["n_frame_batches"] = len(batches)
    base["analysis"] = merged
    base["partial_analyses"] = partials if len(partials) > 1 else None
    return base


def heuristic_merge(partials: list[dict[str, Any]]) -> dict[str, Any]:
    activities: list[str] = []
    objects: list[str] = []
    tags: list[str] = []
    habits: list[str] = []
    prefs: list[dict[str, Any]] = []
    details: list[str] = []
    social_votes: list[str] = []
    captions: list[str] = []
    for p in partials:
        if not isinstance(p, dict):
            continue
        cap = p.get("caption")
        if isinstance(cap, str) and cap.strip():
            captions.append(cap.strip())
        activities.extend(p.get("observed_activities") or [])
        objects.extend(p.get("objects_and_places") or [])
        tags.extend(p.get("behavior_tags") or [])
        habits.extend(p.get("habit_signals") or [])
        prefs.extend(p.get("preference_hypotheses") or [])
        details.extend(p.get("notable_details") or [])
        if p.get("social_interaction"):
            social_votes.append(p["social_interaction"])

    def uniq(xs: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in xs:
            k = str(x).strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    # stitch partial captions into one paragraph when LLM merge fails
    if captions:
        merged_caption = " ".join(captions)
    else:
        merged_caption = ""

    social = Counter(social_votes).most_common(1)[0][0] if social_votes else "none"
    return {
        "caption": merged_caption,
        "observed_activities": uniq(activities)[:20],
        "objects_and_places": uniq(objects)[:20],
        "social_interaction": social,
        "behavior_tags": uniq(tags)[:20],
        "preference_hypotheses": prefs[:12],
        "habit_signals": uniq(habits)[:12],
        "plan_vs_observation": next(
            (p.get("plan_vs_observation") for p in partials if isinstance(p, dict) and p.get("plan_vs_observation")),
            "",
        ),
        "notable_details": uniq(details)[:20],
        "merge_mode": "heuristic",
    }


def synthesize_day(mllm: VLLMClient, day_idx: int, day_rows: list[dict[str, Any]]) -> dict[str, Any]:
    compact = []
    for r in day_rows:
        compact.append(
            {
                "video_uid": r.get("video_uid"),
                "time": f"{r.get('start_timestamp')}->{r.get('end_timestamp')}",
                "slot_id": r.get("slot_id"),
                "plan_chunk": r.get("plan_chunk"),
                "main_scene": r.get("main_scene"),
                "ego4d_caption_hint": r.get("consolidated_summary"),
                "vlm_caption": (r.get("analysis") or {}).get("caption")
                if isinstance(r.get("analysis"), dict)
                else None,
                "analysis": r.get("analysis"),
                "status": r.get("status"),
            }
        )
    meta = day_rows[0] if day_rows else {}
    user = (
        f"day_index={day_idx} date={meta.get('calendar_date')} "
        f"weekday={meta.get('day_of_week')} theme={meta.get('day_theme')} "
        f"anomalies={meta.get('anomaly_events')}\n\n"
        f"clips=\n{json.dumps(compact, ensure_ascii=False)[:14000]}"
    )
    try:
        return mllm.chat_json(DAY_SYNTHESIS_SYSTEM, user, temperature=0.2)
    except Exception as exc:
        text = mllm.chat_text(DAY_SYNTHESIS_SYSTEM, user, temperature=0.2)
        return {"day_summary": text, "parse_error": str(exc)}


def synthesize_global(
    mllm: VLLMClient,
    day_summaries: dict[str, Any],
    aggregate_stats: dict[str, Any],
) -> dict[str, Any]:
    user = json.dumps(
        {"aggregate_stats": aggregate_stats, "day_summaries": day_summaries},
        ensure_ascii=False,
    )[:16000]
    try:
        return mllm.chat_json(GLOBAL_SYNTHESIS_SYSTEM, user, temperature=0.2)
    except Exception as exc:
        text = mllm.chat_text(GLOBAL_SYNTHESIS_SYSTEM, user, temperature=0.2)
        return {"summary": text, "parse_error": str(exc)}


def aggregate_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tag_counter: Counter[str] = Counter()
    activity_counter: Counter[str] = Counter()
    social_counter: Counter[str] = Counter()
    scene_counter: Counter[str] = Counter()
    pref_topics: Counter[str] = Counter()
    ok = sum(1 for r in rows if r.get("status") == "ok")
    no_video = sum(1 for r in rows if r.get("status") == "no_video")
    err = sum(1 for r in rows if r.get("status") == "error")
    vision = sum(1 for r in rows if r.get("vision_used"))

    for r in rows:
        scene_counter[str(r.get("main_scene") or "unknown")] += 1
        a = r.get("analysis") or {}
        if not isinstance(a, dict):
            continue
        for t in a.get("behavior_tags") or []:
            tag_counter[str(t).lower()] += 1
        for act in a.get("observed_activities") or []:
            activity_counter[str(act).lower()] += 1
        if a.get("social_interaction"):
            social_counter[str(a["social_interaction"]).lower()] += 1
        for ph in a.get("preference_hypotheses") or []:
            if isinstance(ph, dict) and ph.get("topic"):
                pref_topics[str(ph["topic"]).lower()] += 1

    return {
        "n_clips": len(rows),
        "status_counts": {"ok": ok, "no_video": no_video, "error": err},
        "vision_used": vision,
        "top_behavior_tags": tag_counter.most_common(30),
        "top_activities": activity_counter.most_common(30),
        "social_interaction_dist": dict(social_counter),
        "scene_dist": dict(scene_counter),
        "top_preference_topics": pref_topics.most_common(30),
    }


def main() -> None:
    args = parse_args()
    args.lifelog = resolve_lifelog(args.lifelog)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME
    report_path = out_dir / REPORT_NAME

    # bump client timeout for heavy vision calls
    import analysis.config as cfg

    cfg.VLLM_TIMEOUT_SEC = args.timeout

    clips = load_clips(args.lifelog, args.day, args.limit)
    print("=" * 64)
    print("[EgoTailor] Full 21-day VLM behavior analysis")
    print(f"  lifelog     : {args.lifelog}")
    print(f"  video_root  : {args.video_root}")
    print(f"  clips       : {len(clips)}")
    print(f"  frames/clip : {args.frames} (batch={args.frame_batch}, max_edge={args.frame_max_size}px)")
    print(f"  vLLM        : {args.vllm_base}  model={args.model}")
    print(f"  out_dir     : {out_dir}")
    frames_dir = None if args.no_save_frames else Path(args.frames_dir)
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"  frames_dir  : {frames_dir}")
    else:
        print("  frames_dir  : (disabled)")
    print("=" * 64)

    mllm = VLLMClient(base_url=args.vllm_base, model=args.model)
    if not mllm.health_check():
        print(f"ERROR: vLLM not reachable at {args.vllm_base}")
        print("Start it first, e.g. bash /root/start_vllm_qwen3vl.sh")
        sys.exit(1)
    print("vLLM health check: OK")

    done = load_done_uids(checkpoint_path) if args.resume else set()
    if args.resume:
        print(f"Resume: {len(done)} clips already done in {checkpoint_path}")
    else:
        # fresh run: backup old checkpoint if present
        if checkpoint_path.exists():
            bak = checkpoint_path.with_suffix(f".bak_{int(time.time())}.jsonl")
            checkpoint_path.rename(bak)
            print(f"Previous checkpoint moved -> {bak}")

    todo = [c for c in clips if c["video_uid"] not in done]
    print(f"To analyze: {len(todo)} / {len(clips)}")

    t0 = time.time()
    errors = 0
    overall = tqdm(todo, desc="Clips", unit="clip", dynamic_ncols=True)
    for i, clip in enumerate(overall):
        uid_short = clip["video_uid"][:8]
        overall.set_postfix(
            day=clip.get("day_index"),
            uid=uid_short,
            err=errors,
            refresh=False,
        )
        try:
            row = analyze_one_clip(
                mllm=mllm,
                clip=clip,
                frames=args.frames,
                frame_batch=args.frame_batch,
                frame_max_size=args.frame_max_size,
                video_root=args.video_root,
                pbar_desc=f"  frames d{clip.get('day_index')} {uid_short}",
                frames_dir=frames_dir,
            )
            row["analyzed_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            errors += 1
            row = {
                "video_uid": clip["video_uid"],
                "day_index": clip.get("day_index"),
                "calendar_date": clip.get("calendar_date"),
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc()[-2000:],
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            }
        append_checkpoint(checkpoint_path, row)

        # drop heavy partials from memory path already written
        if (i + 1) % max(1, args.save_every) == 0:
            pass

    elapsed = time.time() - t0
    print(f"\nClip analysis finished in {elapsed / 60:.1f} min  errors={errors}")

    all_rows = read_checkpoint_rows(checkpoint_path)
    # keep only rows belonging to requested clip set
    wanted = {c["video_uid"] for c in clips}
    all_rows = [r for r in all_rows if r.get("video_uid") in wanted]
    print(f"Loaded {len(all_rows)} unique clip analyses from checkpoint")

    aggregate_stats = aggregate_from_rows(all_rows)
    print("Aggregate top tags:", aggregate_stats["top_behavior_tags"][:10])

    day_summaries: dict[str, Any] = {}
    if not args.skip_day_synthesis:
        by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in all_rows:
            if r.get("day_index") is not None:
                by_day[int(r["day_index"])].append(r)
        for day_idx in tqdm(sorted(by_day), desc="Day synthesis", unit="day", dynamic_ncols=True):
            day_summaries[str(day_idx)] = {
                "day_index": day_idx,
                "calendar_date": by_day[day_idx][0].get("calendar_date"),
                "day_theme": by_day[day_idx][0].get("day_theme"),
                "n_clips": len(by_day[day_idx]),
                "synthesis": synthesize_day(mllm, day_idx, by_day[day_idx]),
            }

    behavior_profile: dict[str, Any] | None = None
    if not args.skip_global_synthesis:
        print("Global preference synthesis...")
        behavior_profile = synthesize_global(mllm, day_summaries, aggregate_stats)

    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "lifelog": str(args.lifelog),
            "video_root": str(args.video_root),
            "vllm_base": args.vllm_base,
            "model": args.model,
            "frames_per_clip": args.frames,
            "frame_batch": args.frame_batch,
            "frame_max_size": args.frame_max_size,
            "frames_dir": str(frames_dir) if frames_dir else None,
            "n_clips_requested": len(clips),
            "n_clips_analyzed": len(all_rows),
            "elapsed_sec": round(elapsed, 1),
            "checkpoint": str(checkpoint_path),
        },
        "aggregate_stats": aggregate_stats,
        "day_summaries": day_summaries,
        "behavior_profile": behavior_profile,
        "clip_analyses": [
            {k: v for k, v in r.items() if k != "partial_analyses"} for r in all_rows
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 64)
    print(f"[Done] Report -> {report_path}")
    print(f"  clips analyzed : {len(all_rows)}")
    print(f"  vision_used    : {aggregate_stats['vision_used']}")
    print(f"  status         : {aggregate_stats['status_counts']}")
    if isinstance(behavior_profile, dict):
        print("\n=== Behavior Profile Summary ===")
        print(behavior_profile.get("summary", json.dumps(behavior_profile, ensure_ascii=False)[:2000]))
    print("=" * 64)


if __name__ == "__main__":
    main()
