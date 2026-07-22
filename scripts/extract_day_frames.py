#!/usr/bin/env python3
"""Extract uniformly sampled frames from each lifelog clip, ordered by day timeline.

For each day in the lifelog:
  1. Sort memory_content by start_timestamp
  2. For every video_uid, sample N frames (default 64) from the Ego4D mp4
  3. Map each frame onto the clip's [start_timestamp, end_timestamp] timeline
  4. Save JPEGs into one day folder, named so lexical order == timeline order

Layout:
  {output_dir}/
    day_00_2026-01-05/
      00000_20260105T065300_wake_hygiene_cee444e9_f00.jpg
      00001_20260105T065328_wake_hygiene_cee444e9_f01.jpg
      ...
      manifest.json
    day_01_...

Examples:
  python scripts/extract_day_frames.py --day 0 --frames 64

  python scripts/extract_day_frames.py --all-days --frames 32 \\
    --lifelog output/lifelog/lifelog_egotailor_usa_enfp_21d.json \\
    --video-root /mnt/data_oss/raw_data/Ego4d/v2/full_scale \\
    --output-dir /mnt/data/workspace/outputs/EgoTailor_30days_frames
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.video_loader import extract_frames, resolve_video_path  # noqa: E402

from generation.config import TOTAL_DAYS, resolve_lifelog  # noqa: E402

DEFAULT_VIDEO_ROOT = Path("/mnt/data_oss/raw_data/Ego4d/v2/full_scale")
# Output dir carries the run length: a 30-day run must not overwrite the
# 21-day results sitting next to it. These are expensive to recompute.
DEFAULT_OUTPUT_DIR = Path(f"/mnt/data/workspace/outputs/EgoTailor_{TOTAL_DAYS}days_frames")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract day-ordered frames from lifelog videos")
    p.add_argument("--lifelog", type=Path, default=None)
    p.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--day", type=int, nargs="*", default=None, help="Day indices (0-20)")
    p.add_argument("--all-days", action="store_true")
    p.add_argument("--frames", type=int, default=64, help="Frames per video clip (e.g. 32 or 64)")
    p.add_argument("--frame-max-size", type=int, default=448, help="Max image edge px")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument(
        "--also-per-clip",
        action="store_true",
        help="Also mirror frames under day_XX/clips/<slot>_<uid>/",
    )
    return p.parse_args()


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def frame_timestamps(start: datetime, end: datetime, n: int) -> list[datetime]:
    """Uniform timestamps across [start, end] for n sampled frames."""
    if n <= 0:
        return []
    if n == 1 or end <= start:
        return [start]
    total = (end - start).total_seconds()
    return [start + timedelta(seconds=total * i / (n - 1)) for i in range(n)]


def load_days(lifelog_path: Path, day_indices: list[int] | None, all_days: bool) -> list[dict[str, Any]]:
    data = json.loads(lifelog_path.read_text())
    days = data["days"]
    if all_days or day_indices is None:
        if day_indices is None and not all_days:
            raise SystemExit("Pass --day N [N ...] or --all-days")
        if all_days:
            return days
    wanted = set(day_indices or [])
    selected = [d for d in days if int(d["metadata"]["day_index"]) in wanted]
    missing = wanted - {int(d["metadata"]["day_index"]) for d in selected}
    if missing:
        raise SystemExit(f"day_index not found: {sorted(missing)}")
    return selected


def safe_token(text: str, max_len: int = 40) -> str:
    keep = []
    for ch in text.replace(" ", "_"):
        if ch.isalnum() or ch in ("_", "-"):
            keep.append(ch)
    s = "".join(keep).strip("_") or "clip"
    return s[:max_len]


def extract_day_frames(
    day: dict[str, Any],
    video_root: Path,
    output_dir: Path,
    frames_per_clip: int,
    frame_max_size: int,
    jpeg_quality: int,
    also_per_clip: bool,
) -> dict[str, Any]:
    meta = day["metadata"]
    day_idx = int(meta["day_index"])
    date = meta.get("calendar_date") or "unknown"
    day_name = f"day_{day_idx:02d}_{date}"
    day_dir = output_dir / day_name
    day_dir.mkdir(parents=True, exist_ok=True)

    clips = list(day["memory_content"])
    clips.sort(key=lambda c: c.get("start_timestamp") or c.get("start_time") or "")

    global_idx = 0
    frame_records: list[dict[str, Any]] = []
    missing_videos: list[dict[str, Any]] = []

    for clip_i, clip in enumerate(tqdm(clips, desc=day_name, unit="clip")):
        uid = clip["video_uid"]
        video_path = resolve_video_path(uid, video_root)
        slot = clip.get("slot_id") or f"clip{clip_i:02d}"
        start = parse_ts(clip.get("start_timestamp"))
        end = parse_ts(clip.get("end_timestamp"))
        if start is None:
            start = datetime(2026, 1, 1)
        if end is None:
            dur = float(clip.get("duration") or 0)
            end = start + timedelta(seconds=dur if dur > 0 else 1)

        if video_path is None:
            missing_videos.append({"video_uid": uid, "slot_id": slot, "start_timestamp": clip.get("start_timestamp")})
            continue

        frames_bgr, sampling = extract_frames(
            video_path,
            num_frames=frames_per_clip,
            max_size=frame_max_size,
            duration_sec=clip.get("duration"),
        )
        if not frames_bgr:
            missing_videos.append({"video_uid": uid, "slot_id": slot, "reason": "no_frames"})
            continue

        ts_list = frame_timestamps(start, end, len(frames_bgr))
        uid_short = uid[:8]
        slot_tok = safe_token(str(slot))

        clip_mirror_dir = None
        if also_per_clip:
            clip_mirror_dir = day_dir / "clips" / f"{clip_i:02d}_{slot_tok}_{uid_short}"
            clip_mirror_dir.mkdir(parents=True, exist_ok=True)

        for f_i, (frame, ts) in enumerate(zip(frames_bgr, ts_list)):
            ts_tag = ts.strftime("%Y%m%dT%H%M%S")
            fname = f"{global_idx:05d}_{ts_tag}_{slot_tok}_{uid_short}_f{f_i:02d}.jpg"
            out_path = day_dir / fname
            ok = cv2.imwrite(
                str(out_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if not ok:
                continue

            if clip_mirror_dir is not None:
                cv2.imwrite(
                    str(clip_mirror_dir / f"f{f_i:02d}.jpg"),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                )

            frame_records.append(
                {
                    "global_index": global_idx,
                    "file": fname,
                    "frame_timestamp": ts.isoformat(timespec="seconds"),
                    "clip_index": clip_i,
                    "frame_index_in_clip": f_i,
                    "video_uid": uid,
                    "slot_id": slot,
                    "plan_chunk": clip.get("plan_chunk"),
                    "clip_start_timestamp": clip.get("start_timestamp"),
                    "clip_end_timestamp": clip.get("end_timestamp"),
                    "video_frame_index": (sampling.get("frame_indices") or [None] * len(frames_bgr))[f_i],
                }
            )
            global_idx += 1

    manifest = {
        "day_index": day_idx,
        "calendar_date": date,
        "day_theme": meta.get("day_theme"),
        "day_of_week": meta.get("day_of_week"),
        "frames_per_clip": frames_per_clip,
        "frame_max_size": frame_max_size,
        "n_clips_in_lifelog": len(clips),
        "n_clips_with_frames": len({r["video_uid"] for r in frame_records}),
        "n_frames_saved": len(frame_records),
        "n_missing_videos": len(missing_videos),
        "missing_videos": missing_videos,
        "day_dir": str(day_dir),
        "frames": frame_records,
    }
    (day_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        f"[{day_name}] saved {len(frame_records)} frames from "
        f"{manifest['n_clips_with_frames']}/{len(clips)} clips -> {day_dir}"
    )
    return manifest


def main() -> None:
    args = parse_args()
    args.lifelog = resolve_lifelog(args.lifelog)
    if not args.all_days and args.day is None:
        args.day = [0]

    days = load_days(args.lifelog, args.day, args.all_days)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for day in days:
        m = extract_day_frames(
            day=day,
            video_root=args.video_root,
            output_dir=args.output_dir,
            frames_per_clip=args.frames,
            frame_max_size=args.frame_max_size,
            jpeg_quality=args.jpeg_quality,
            also_per_clip=args.also_per_clip,
        )
        summaries.append(
            {
                "day_index": m["day_index"],
                "calendar_date": m["calendar_date"],
                "n_frames_saved": m["n_frames_saved"],
                "n_clips_with_frames": m["n_clips_with_frames"],
                "n_missing_videos": m["n_missing_videos"],
                "day_dir": m["day_dir"],
            }
        )

    summary_path = args.output_dir / "extract_frames_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "lifelog": str(args.lifelog),
                "video_root": str(args.video_root),
                "frames_per_clip": args.frames,
                "n_days": len(summaries),
                "days": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
