#!/usr/bin/env python3
"""Fine-grained captioning of a day's Ego4D clips with the vLLM Omni model.

For every video in one day's lifelog plan:
  1. Sample frames from the local mp4 (dense, adaptive to duration).
  2. Split them into temporal batches that fit the server's context window and
     ask the model for a fine-grained, moment-to-moment description of each,
     tagged with the batch's approximate time window.
  3. Merge the batch descriptions into one continuous narrative for the clip.
  4. Write per-video JSON into Caption/day_NN/.

Why batches: a fine-grained caption needs many frames, but the context window
(even 32k) caps how many fit at once. Captioning windows and then stitching
keeps temporal detail without overflowing — and degrades gracefully to a single
call when the clip is short enough to send whole.

Usage (from EgoTailor root, with the vLLM server from start_vllm.sh running):

  # Day 0, the default video set for now
  python Caption/caption_videos.py

  # A different day, more frames, overwrite existing captions
  python Caption/caption_videos.py --day 3 --frames 32 --overwrite

  # Plumbing test without a server (no API calls, writes stub captions)
  python Caption/caption_videos.py --limit 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow `python Caption/caption_videos.py` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.config import (  # noqa: E402
    EGO4D_VIDEO_ROOT,
    VLLM_API_BASE,
    VLLM_API_KEY,
)
from analysis.mllm_client import VLLMClient  # noqa: E402
from analysis.video_loader import (  # noqa: E402
    extract_frames,
    frames_to_base64_jpeg,
    resolve_video_path,
)

CAPTION_DIR = Path(__file__).resolve().parent
# The Omni model this pipeline is built around (matches start_vllm.sh's
# served-model-name); override with VLLM_MODEL or --model.
DEFAULT_MODEL = os.getenv("VLLM_MODEL") or "Qwen3-Omni-30B-A3B-Instruct"

SEGMENT_SYSTEM = """You are an expert first-person (egocentric) video captioner and
behavior analyst. You are shown uniformly sampled frames from ONE segment of an
Ego4D clip, in temporal order. The segment covers approximately {window} of the clip.

Write a FINE-GRAINED, temporally-ordered description of this segment: what the
camera-wearer does moment to moment, the specific objects and tools they handle,
hand actions, the place and its layout, and notable transitions. Be concrete and
detailed, but stay faithful to the frames — never invent what is not visible. The
hints (Ego4D caption, planned activity) are weak priors only; ground everything in
the frames.

Two things to focus on beyond the description:
- CAMERA-WEARER GENDER, from HANDS ONLY. This is first-person video: the wearer is
  not in frame, but their own hands and forearms often are. When they are visible,
  infer the wearer's gender from hand cues only — hand shape and size, skin, nail
  polish, rings or other jewelry, forearm hair. Report "male", "female", or
  "unknown". Use ONLY hand/forearm evidence; if the hands are not clearly visible or
  the cues are ambiguous, report "unknown". Do NOT infer gender from the activity,
  the setting, or any other person — a person cooking is not therefore female.
- PREFERENCE / HABIT SIGNALS: things that hint at what this person likes or routinely
  does (a favourite dish, a preferred tool, a characteristic way of doing a task).
  Each is a hypothesis with visual evidence and a confidence — not an assertion.

Return valid JSON only (no markdown):
{
  "description": "one detailed, fluent paragraph for THIS segment",
  "actions": ["short ordered action phrases seen in this segment"],
  "objects": ["objects / tools handled or clearly visible"],
  "place": "the setting in a few words",
  "key_details": ["concrete notable details of the event"],
  "camera_wearer_gender": "male|female|unknown",
  "preference_signals": [
    {"topic": "e.g. likes spicy noodles", "category": "food|activity|social|other",
     "evidence": "what in the frames supports it", "confidence": "high|medium|low"}
  ]
}
Only if they are clearly present and relevant, you MAY also add "people_present"
(other people visible) and "food_and_consumables" (specific dishes/ingredients);
omit both when they do not apply — they are not the focus."""

MERGE_SYSTEM = """You merge ordered per-segment analyses of ONE first-person video
into a single fine-grained account of the whole clip.

First write ONE continuous, fluent, temporally-ordered narrative (several sentences)
that preserves the moment-to-moment detail of each segment and the transitions
between them — no bullet lists, no segment headers, covering the WHOLE clip start to
finish. Then consolidate the structured fields across all segments: de-duplicate,
keep temporal order, and raise a preference's confidence only when the same signal
recurs across segments.

For camera_wearer_gender: take the value the HAND evidence supports across segments;
if no segment saw the hands clearly, keep "unknown". Never upgrade "unknown" to a
guess from context.

Return valid JSON only (no markdown):
{
  "caption": "one continuous fine-grained narrative of the entire clip",
  "actions": ["ordered action phrases across the whole clip"],
  "objects": ["objects / tools across the whole clip"],
  "place": "the dominant setting",
  "key_details": ["concrete notable details across the clip"],
  "camera_wearer_gender": "male|female|unknown",
  "preference_signals": [{"topic": "...", "category": "...", "evidence": "...", "confidence": "high|medium|low"}]
}
Carry "people_present" and/or "food_and_consumables" only if segments reported them."""


def find_day_file(persona: str | None, day: int) -> Path:
    """Locate output/days/<persona>/day_NN_*.json, erroring clearly on ambiguity."""
    days_root = PROJECT_ROOT / "output" / "days"
    pattern = f"{persona}/day_{day:02d}_*.json" if persona else f"*/day_{day:02d}_*.json"
    matches = sorted(days_root.glob(pattern))
    if not matches:
        raise SystemExit(
            f"No day file for day {day} under {days_root} (persona={persona!r}).\n"
            f"Build the lifelog first:  python -m generation.build_lifelog"
        )
    if len(matches) > 1:
        personas = sorted({p.parent.name for p in matches})
        raise SystemExit(
            f"Several personas have day {day}: {personas}\n"
            f"Pick one with --persona <id>."
        )
    return matches[0]


def day_video_slots(day_file: Path) -> list[dict[str, Any]]:
    """One entry per unique video in the day, carrying the plan context we caption against."""
    day = json.loads(day_file.read_text())
    slots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for s in day.get("plan", []):
        uid = s.get("video_uid")
        if not uid or uid in seen:
            continue
        seen.add(uid)
        slots.append(
            {
                "video_uid": uid,
                "activity": s.get("activity", ""),
                "ego4d_caption": s.get("video_description", ""),
                "requested_scenarios": s.get("requested_scenarios", []),
                "main_scene": s.get("main_scene", ""),
                "start_timestamp": s.get("start_timestamp", ""),
                "minutes": s.get("minutes", 0),
            }
        )
    return slots


def place_video(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    """Put the source mp4 next to its caption for side-by-side review.

    symlink (default) is instant and free; copy is a real duplicate for when the
    folder will be moved off this machine. Existing links/copies are left alone
    unless --overwrite, so re-runs don't re-copy gigabytes.
    """
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def frame_timestamps(meta: dict[str, Any]) -> list[float]:
    """Map each extracted frame to its wall-clock second within the clip."""
    total = meta.get("total_video_frames") or 0
    dur = meta.get("duration_sec") or 0.0
    idxs = meta.get("frame_indices") or []
    if total <= 0 or dur <= 0 or not idxs:
        return [0.0] * len(idxs)
    return [idx / total * dur for idx in idxs]


def batch_ranges(n_frames: int, batch: int) -> list[tuple[int, int]]:
    return [(i, min(i + batch, n_frames)) for i in range(0, n_frames, max(1, batch))]


def _vision_json(
    client: VLLMClient,
    model: str,
    system: str,
    user_text: str,
    frames_b64: list[str],
    max_tokens: int,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """One multimodal call with our own max_tokens, tolerant of fenced JSON."""
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for b64 in frames_b64:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    resp = client.client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _parse_json(resp.choices[0].message.content or "")


def _text_json(
    client: VLLMClient,
    model: str,
    system: str,
    user_text: str,
    max_tokens: int,
    temperature: float = 0.2,
) -> dict[str, Any]:
    resp = client.client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _parse_json(resp.choices[0].message.content or "")


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Salvage the outermost object if the model wrapped it in prose.
        lo, hi = text.find("{"), text.rfind("}")
        if 0 <= lo < hi:
            return json.loads(text[lo : hi + 1])
        raise


def caption_one_video(
    client: VLLMClient | None,
    model: str,
    slot: dict[str, Any],
    video_root: Path,
    num_frames: int | None,
    batch: int,
    max_size: int,
    max_tokens: int,
    dry_run: bool,
) -> dict[str, Any]:
    uid = slot["video_uid"]
    base: dict[str, Any] = {
        "video_uid": uid,
        "activity": slot["activity"],
        "ego4d_caption": slot["ego4d_caption"],
        "requested_scenarios": slot["requested_scenarios"],
        "captioned_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
    }

    path = resolve_video_path(uid, video_root)
    if path is None:
        return {**base, "available": False, "error": f"mp4 not found under {video_root}"}

    frames, meta = extract_frames(path, num_frames=num_frames, max_size=max_size)
    if not frames:
        return {**base, "available": False, "error": "no frames extracted", "sampling": meta}

    b64 = frames_to_base64_jpeg(frames)
    times = frame_timestamps(meta)
    hint = (
        f"Ego4D caption (weak hint): {slot['ego4d_caption']}\n"
        f"Planned activity (weak hint): {slot['activity']}\n"
        f"Setting hint: {slot['main_scene']}\n"
        "Describe what the frames actually show."
    )

    if dry_run:
        return {
            **base,
            "available": True,
            "dry_run": True,
            "num_frames": len(frames),
            "duration_sec": meta.get("duration_sec"),
            "caption": "(dry-run: no model call)",
        }

    segments: list[dict[str, Any]] = []
    for lo, hi in batch_ranges(len(b64), batch):
        window = f"{_mmss(times[lo])}-{_mmss(times[hi - 1])}"
        system = SEGMENT_SYSTEM.format(window=window)
        seg = _vision_json(client, model, system, hint, b64[lo:hi], max_tokens)
        seg["time_window"] = window
        segments.append(seg)

    if len(segments) == 1:
        s0 = segments[0]
        merged = {
            "caption": s0.get("description", ""),
            "actions": s0.get("actions", []),
            "objects": s0.get("objects", []),
            "place": s0.get("place", ""),
            "key_details": s0.get("key_details", []),
            "camera_wearer_gender": s0.get("camera_wearer_gender", "unknown"),
            "preference_signals": s0.get("preference_signals", []),
        }
        for opt in ("people_present", "food_and_consumables"):
            if s0.get(opt):
                merged[opt] = s0[opt]
    else:
        # Merge from the full per-segment JSON so gender/preferences survive, not
        # just the prose — the whole-clip narrative is rebuilt from them.
        payload = json.dumps(
            [{"time_window": s.get("time_window"), **{k: v for k, v in s.items() if k != "time_window"}}
             for s in segments],
            ensure_ascii=False,
        )
        merged = _text_json(
            client, model, MERGE_SYSTEM,
            f"Ordered per-segment analyses of one clip:\n{payload}",
            max_tokens,
        )

    result = {
        **base,
        "available": True,
        "duration_sec": meta.get("duration_sec"),
        "num_frames": len(frames),
        "num_segments": len(segments),
        "caption": merged.get("caption", ""),
        "camera_wearer_gender": merged.get("camera_wearer_gender", "unknown"),
        "actions": merged.get("actions", []),
        "objects": merged.get("objects", []),
        "place": merged.get("place", ""),
        "key_details": merged.get("key_details", []),
        "preference_signals": merged.get("preference_signals", []),
        "segments": segments,
    }
    # Secondary fields: keep only when the model actually reported them.
    for opt in ("people_present", "food_and_consumables"):
        if merged.get(opt):
            result[opt] = merged[opt]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", type=int, default=0, help="Day index to caption (default 0)")
    ap.add_argument("--persona", default=None, help="Persona id, if output/days has more than one")
    ap.add_argument("--video-root", type=Path, default=EGO4D_VIDEO_ROOT)
    ap.add_argument("--vllm-base", default=VLLM_API_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--frames", type=int, default=None, help="Frames per clip (default: adaptive by duration)")
    ap.add_argument("--batch-frames", type=int, default=12, help="Frames per model call (default 12)")
    ap.add_argument("--max-size", type=int, default=768, help="Max frame edge in px (default 768)")
    ap.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens per call")
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: Caption/day_NN")
    ap.add_argument("--with-video", action="store_true",
                    help="Place each clip's mp4 next to its json (for side-by-side review)")
    ap.add_argument("--copy", action="store_true",
                    help="With --with-video, copy the mp4 instead of symlinking it")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N videos (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="Recaption videos that already have output")
    ap.add_argument("--dry-run", action="store_true", help="No API calls; write stub captions")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    day_file = find_day_file(args.persona, args.day)
    slots = day_video_slots(day_file)
    if args.limit:
        slots = slots[: args.limit]

    out_dir = args.out_dir or (CAPTION_DIR / f"day_{args.day:02d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    client: VLLMClient | None = None
    if not args.dry_run:
        client = VLLMClient(base_url=args.vllm_base, api_key=VLLM_API_KEY, model=args.model)
        if not client.health_check():
            raise SystemExit(
                f"vLLM server not reachable at {args.vllm_base}.\n"
                f"Start it first:  bash Caption/start_vllm.sh"
            )

    print(f"day file : {day_file}")
    print(f"videos   : {len(slots)}  (day {args.day})")
    print(f"model    : {args.model}  @ {args.vllm_base}")
    print(f"out dir  : {out_dir}")
    print("-" * 60)

    iterator = slots
    if not args.no_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(slots, desc=f"day {args.day}", unit="clip")
        except ImportError:
            pass

    video_mode = "copy" if args.copy else "symlink"
    done = skipped = failed = placed = 0
    index: list[dict[str, Any]] = []
    for slot in iterator:
        uid = slot["video_uid"]
        out_path = out_dir / f"{uid}.json"

        # Pair the mp4 with its caption for review, independent of captioning —
        # so the folder fills even when a caption is skipped on re-run.
        if args.with_video:
            src = resolve_video_path(uid, args.video_root)
            if src is not None:
                place_video(src, out_dir / f"{uid}{src.suffix}", video_mode, args.overwrite)
                placed += 1

        if out_path.exists() and not args.overwrite:
            skipped += 1
            index.append({"video_uid": uid, "status": "skipped", "file": out_path.name})
            continue
        t0 = time.time()
        try:
            result = caption_one_video(
                client, args.model, slot, args.video_root,
                args.frames, args.batch_frames, args.max_size, args.max_tokens, args.dry_run,
            )
            result["seconds"] = round(time.time() - t0, 1)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            if result.get("available"):
                done += 1
                status = "captioned"
            else:
                failed += 1
                status = "unavailable"
            index.append({"video_uid": uid, "status": status, "file": out_path.name,
                          "activity": slot["activity"]})
        except Exception as exc:  # noqa: BLE001
            failed += 1
            index.append({"video_uid": uid, "status": "error", "error": str(exc)})
            (out_dir / f"{uid}.error.txt").write_text(traceback.format_exc())

    (out_dir / "_index.json").write_text(
        json.dumps(
            {
                "day": args.day,
                "day_file": str(day_file),
                "model": args.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "counts": {"captioned": done, "skipped": skipped, "failed": failed},
                "videos": index,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("-" * 60)
    print(f"captioned {done} | skipped {skipped} | failed {failed}")
    if args.with_video:
        print(f"videos placed ({video_mode}): {placed}")
    print(f"index: {out_dir / '_index.json'}")


if __name__ == "__main__":
    main()
