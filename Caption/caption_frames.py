#!/usr/bin/env python3
"""Per-frame quick captioning: 3 random frames per clip, one line + gender each.

A lighter sibling of caption_videos.py. Instead of one merged narrative per clip,
this grabs a few random frames from each of a day's videos and, for EACH frame,
asks the model only for:
  - the important content of that single image (1-2 short sentences), and
  - whether the person's gender can be judged from it (male/female/unknown).

Every extracted frame is saved as a .jpg next to a single frames.json whose keys
are the exact image filenames, so an image and its caption line up one-to-one and
are trivial to eyeball.

Usage (from EgoTailor root, with a vLLM server running):

  # Day 0, 3 random frames per clip, into /root/caption_frames
  python Caption/caption_frames.py --model Qwen3-VL-8B-Instruct

  # Fewer clips / more frames while testing
  python Caption/caption_frames.py --model Qwen3-VL-8B-Instruct --limit 3 --frames 5

  # Plumbing test, no server (extracts+saves frames, stubs the captions)
  python Caption/caption_frames.py --limit 2 --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

# Allow `python Caption/caption_frames.py` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.config import (  # noqa: E402
    EGO4D_VIDEO_ROOT,
    FRAME_MAX_SIZE,
    VLLM_API_BASE,
    VLLM_API_KEY,
)
from analysis.mllm_client import VLLMClient  # noqa: E402
from analysis.video_loader import resolve_video_path  # noqa: E402

DEFAULT_OUT = Path("/root/caption_frames")
DEFAULT_MODEL = os.getenv("VLLM_MODEL") or "Qwen3-VL-8B-Instruct"

FRAME_SYSTEM = """You are analyzing ONE frame from a first-person (egocentric) video.

1. Briefly point out the IMPORTANT content of this image — the main objects, the
   action in progress, and the place — in 1-2 short sentences. Do not over-describe;
   name what matters, not every pixel.
2. Judge the person's gender, but ONLY from what is visible: the camera-wearer's own
   hands / forearms (shape, skin, nail polish, jewelry, arm hair) or another person
   clearly in frame. Report "male", "female", or "unknown". If no hands or person are
   clearly visible, or the cues are ambiguous, report "unknown". NEVER infer gender
   from the activity or the setting.

Return valid JSON only (no markdown):
{
  "description": "1-2 short sentences on the important content",
  "gender": "male|female|unknown",
  "gender_basis": "the visible cue used, or 'no person/hands visible' when unknown"
}"""


def find_day_file(persona: str | None, day: int) -> Path:
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
        raise SystemExit(f"Several personas have day {day}: {personas}\nPick one with --persona <id>.")
    return matches[0]


def day_video_slots(day_file: Path) -> list[dict[str, Any]]:
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
            }
        )
    return slots


def _mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _resize(frame, max_size: int):
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_random_frames(
    path: Path, k: int, rng: random.Random, max_size: int
) -> list[tuple[int, float, Any]]:
    """k random distinct frames as (frame_index, timestamp_sec, BGR image), time-ordered."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return []
    idxs = sorted(rng.sample(range(total), min(k, total)))
    out: list[tuple[int, float, Any]] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            out.append((idx, idx / fps if fps > 0 else 0.0, _resize(frame, max_size)))
    cap.release()
    return out


def _frame_to_b64(frame, quality: int = 85) -> str:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


def describe_frame(
    client: VLLMClient, model: str, b64: str, max_tokens: int, temperature: float = 0.2
) -> dict[str, Any]:
    resp = client.client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FRAME_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this single frame."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
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
        lo, hi = text.find("{"), text.rfind("}")
        if 0 <= lo < hi:
            return json.loads(text[lo : hi + 1])
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", type=int, default=0, help="Day index (default 0)")
    ap.add_argument("--persona", default=None, help="Persona id, if output/days has more than one")
    ap.add_argument("--video-root", type=Path, default=EGO4D_VIDEO_ROOT)
    ap.add_argument("--vllm-base", default=VLLM_API_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--frames", type=int, default=3, help="Random frames per clip (default 3)")
    ap.add_argument("--max-size", type=int, default=FRAME_MAX_SIZE, help="Max frame edge in px")
    ap.add_argument("--max-tokens", type=int, default=512, help="Max output tokens per frame")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Default: /root/caption_frames")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for frame selection")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N videos (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="No API calls; save frames, stub captions")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    day_file = find_day_file(args.persona, args.day)
    slots = day_video_slots(day_file)
    if args.limit:
        slots = slots[: args.limit]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    client: VLLMClient | None = None
    if not args.dry_run:
        client = VLLMClient(base_url=args.vllm_base, api_key=VLLM_API_KEY, model=args.model)
        if not client.health_check():
            raise SystemExit(
                f"vLLM server not reachable at {args.vllm_base}.\n"
                f"Start it first:  bash Caption/start_vllm_qwen3vl.sh"
            )

    print(f"day file : {day_file}")
    print(f"videos   : {len(slots)}  x {args.frames} frames  (day {args.day})")
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

    images: dict[str, dict[str, Any]] = {}
    n_frames = n_missing = n_failed = 0
    for slot in iterator:
        uid = slot["video_uid"]
        path = resolve_video_path(uid, args.video_root)
        if path is None:
            n_missing += 1
            continue
        try:
            frames = extract_random_frames(path, args.frames, rng, args.max_size)
        except Exception:  # noqa: BLE001
            n_missing += 1
            continue
        if not frames:
            n_missing += 1
            continue

        for n, (idx, t, frame) in enumerate(frames, start=1):
            name = f"{uid}_{n}.jpg"
            cv2.imwrite(str(out_dir / name), frame)  # BGR -> correct colors on disk
            entry: dict[str, Any] = {
                "video_uid": uid,
                "activity": slot["activity"],
                "frame_index": idx,
                "frame_time": _mmss(t),
            }
            if args.dry_run:
                entry.update({"description": "(dry-run: no model call)", "gender": "unknown"})
            else:
                try:
                    got = describe_frame(client, args.model, _frame_to_b64(frame), args.max_tokens)
                    entry.update(
                        {
                            "description": got.get("description", ""),
                            "gender": got.get("gender", "unknown"),
                            "gender_basis": got.get("gender_basis", ""),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    n_failed += 1
                    entry.update({"error": str(exc)})
                    (out_dir / f"{name}.error.txt").write_text(traceback.format_exc())
            images[name] = entry
            n_frames += 1

    result = {
        "meta": {
            "day": args.day,
            "day_file": str(day_file),
            "model": args.model,
            "seed": args.seed,
            "frames_per_video": args.frames,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "videos": len(slots),
                "images": n_frames,
                "videos_missing": n_missing,
                "frames_failed": n_failed,
            },
        },
        "images": images,
    }
    (out_dir / "frames.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("-" * 60)
    print(f"images {n_frames} | videos missing {n_missing} | frames failed {n_failed}")
    print(f"json: {out_dir / 'frames.json'}")


if __name__ == "__main__":
    main()
