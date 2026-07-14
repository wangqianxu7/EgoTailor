#!/usr/bin/env python3
"""Concatenate one day's lifelog Ego4D clips into a single MP4."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_DAY = Path("/root/EgoTailor/output/days/day_00_2026-01-05.json")
DEFAULT_VIDEO_ROOT = Path("/mnt/data_oss/raw_data/Ego4d/v2/full_scale")
DEFAULT_OUTPUT_DIR = Path("/root/egodaily")


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def stitch_day(
    day_json: Path,
    output_dir: Path,
    video_root: Path,
    skip_missing: bool = True,
) -> dict:
    day = json.loads(day_json.read_text())
    meta = day["metadata"]
    date = meta["calendar_date"]
    day_idx = meta["day_index"]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = output_dir / f"day_{day_idx:02d}_{date}.mp4"
    manifest_path = output_dir / f"day_{day_idx:02d}_{date}_manifest.json"
    concat_list = output_dir / f"day_{day_idx:02d}_{date}_concat.txt"

    included = []
    missing = []
    for i, clip in enumerate(day["memory_content"]):
        uid = clip["video_uid"]
        src = video_root / f"{uid}.mp4"
        entry = {
            "index": i,
            "video_uid": uid,
            "slot_id": clip.get("slot_id"),
            "start_timestamp": clip.get("start_timestamp"),
            "end_timestamp": clip.get("end_timestamp"),
            "duration_lifelog": clip.get("duration"),
            "plan_chunk": clip.get("plan_chunk"),
            "consolidated_summary": clip.get("consolidated_summary"),
        }
        if src.exists():
            entry["source_path"] = str(src)
            entry["duration_actual"] = probe_duration(src)
            included.append(entry)
        else:
            missing.append(entry)

    if not included:
        raise SystemExit("No local mp4 files found for this day.")

    with open(concat_list, "w") as f:
        for item in included:
            # ffmpeg concat demuxer requires escaped paths
            path = item["source_path"].replace("'", "'\\''")
            f.write(f"file '{path}'\n")

    print(f"Stitching {len(included)} clips -> {out_mp4}")
    if missing:
        print(f"  Warning: {len(missing)} clips missing locally (skipped)")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", str(out_mp4),
    ]
    subprocess.run(cmd, check=True)

    total_actual = sum(x["duration_actual"] for x in included)
    manifest = {
        "output_video": str(out_mp4),
        "calendar_date": date,
        "day_index": day_idx,
        "day_theme": meta.get("day_theme"),
        "clips_total": len(day["memory_content"]),
        "clips_included": len(included),
        "clips_missing": len(missing),
        "duration_lifelog_included_hours": round(
            sum(x["duration_lifelog"] for x in included) / 3600, 3
        ),
        "duration_actual_hours": round(total_actual / 3600, 3),
        "included_clips": included,
        "missing_clips": missing,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Done: {out_mp4} ({manifest['duration_actual_hours']:.2f}h)")
    print(f"Manifest: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch lifelog day videos")
    parser.add_argument("--day-json", type=Path, default=DEFAULT_DAY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    args = parser.parse_args()
    stitch_day(args.day_json, args.output_dir, args.video_root)


if __name__ == "__main__":
    main()
