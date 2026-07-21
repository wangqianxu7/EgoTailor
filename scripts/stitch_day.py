#!/usr/bin/env python3
"""Concatenate Ego4D clips for one or more lifelog days into day-level MP4s.

Input can be either:
  - full 21-day lifelog JSON (recommended): output/lifelog/lifelog_egotailor_usa_enfp_21d.json
  - single-day JSON: output/days/day_00_2026-01-05.json

Each day uses memory_content[*].video_uid -> {video_root}/{video_uid}.mp4,
stitched in lifelog order.

Default stitch mode drops audio and regenerates timestamps. Plain `-c copy`
across Ego4D clips often hits Non-monotonous DTS and fails with
"Error writing trailer: Invalid argument".

Examples (run on the server where Ego4D mp4s live):

  python scripts/stitch_day.py \
    --lifelog output/lifelog/lifelog_egotailor_usa_enfp_21d.json \
    --day 0 \
    --video-root /mnt/data_oss/raw_data/Ego4d/v2/full_scale \
    --output-dir /mnt/data/workspace/outputs/EgoTailor_21days_videos

  python scripts/stitch_day.py \
    --lifelog output/lifelog/lifelog_egotailor_usa_enfp_21d.json \
    --all-days \
    --video-root /mnt/data_oss/raw_data/Ego4d/v2/full_scale \
    --output-dir /mnt/data/workspace/outputs/EgoTailor_21days_videos
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_LIFELOG = Path("/root/EgoTailor/output/lifelog/lifelog_egotailor_usa_enfp_21d.json")
DEFAULT_DAY = Path("/root/EgoTailor/output/days/day_00_2026-01-05.json")
DEFAULT_VIDEO_ROOT = Path("/mnt/data_oss/raw_data/Ego4d/v2/full_scale")
DEFAULT_OUTPUT_DIR = Path("/mnt/data/workspace/outputs/EgoTailor_21days_videos")


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def load_days(
    lifelog: Path | None,
    day_json: Path | None,
    day_indices: list[int] | None,
    all_days: bool,
) -> list[dict[str, Any]]:
    """Return list of day objects (each with metadata + memory_content)."""
    if lifelog is not None:
        data = json.loads(lifelog.read_text())
        days = data["days"]
        if all_days:
            return days
        if day_indices is None:
            raise SystemExit("With --lifelog, pass --day N [N ...] or --all-days")
        wanted = set(day_indices)
        selected = [d for d in days if int(d["metadata"]["day_index"]) in wanted]
        missing = wanted - {int(d["metadata"]["day_index"]) for d in selected}
        if missing:
            raise SystemExit(f"day_index not found in lifelog: {sorted(missing)}")
        return selected

    if day_json is not None:
        return [json.loads(day_json.read_text())]

    raise SystemExit("Provide --lifelog or --day-json")


def run_ffmpeg(cmd: list[str]) -> None:
    print("  $", " ".join(cmd[:8]), "..." if len(cmd) > 8 else "")
    subprocess.run(cmd, check=True)


def ffmpeg_concat(
    sources: list[Path],
    out_mp4: Path,
    concat_list: Path,
    mode: str,
    keep_audio: bool,
) -> None:
    """Stitch sources into out_mp4.

    Modes:
      safe     — remux each clip to MPEG-TS (reset timestamps), then concat copy
      copy     — direct concat demuxer + stream copy (fragile on Ego4D)
      reencode — H.264/AAC re-encode (slow, most compatible)
    """
    with open(concat_list, "w") as f:
        for src in sources:
            path = str(src).replace("'", "'\\''")
            f.write(f"file '{path}'\n")

    if mode == "copy":
        cmd = [
            "ffmpeg", "-y", "-fflags", "+genpts",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
        ]
        if keep_audio:
            cmd += ["-c", "copy"]
        else:
            cmd += ["-map", "0:v:0", "-c", "copy", "-an"]
        cmd += ["-avoid_negative_ts", "make_zero", str(out_mp4)]
        run_ffmpeg(cmd)
        return

    if mode == "reencode":
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        ]
        if keep_audio:
            cmd += ["-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd += ["-movflags", "+faststart", str(out_mp4)]
        run_ffmpeg(cmd)
        return

    # mode == "safe": MPEG-TS intermediates avoid Non-monotonous DTS trailer failures
    tmp_root = Path(tempfile.mkdtemp(prefix="stitch_ts_", dir=str(out_mp4.parent)))
    try:
        ts_files: list[Path] = []
        ts_list = tmp_root / "ts_concat.txt"
        for i, src in enumerate(sources):
            ts_path = tmp_root / f"{i:04d}.ts"
            remux = [
                "ffmpeg", "-y", "-i", str(src),
                "-map", "0:v:0",
            ]
            if keep_audio:
                remux += ["-map", "0:a:0?"]
            remux += [
                "-c", "copy",
                "-bsf:v", "h264_mp4toannexb",
                "-f", "mpegts",
                str(ts_path),
            ]
            run_ffmpeg(remux)
            ts_files.append(ts_path)

        with open(ts_list, "w") as f:
            for ts in ts_files:
                path = str(ts).replace("'", "'\\''")
                f.write(f"file '{path}'\n")

        merge = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(ts_list),
            "-c", "copy",
        ]
        if keep_audio:
            merge += ["-bsf:a", "aac_adtstoasc"]
        else:
            merge += ["-an"]
        merge += ["-movflags", "+faststart", str(out_mp4)]
        run_ffmpeg(merge)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def stitch_one_day(
    day: dict[str, Any],
    output_dir: Path,
    video_root: Path,
    mode: str,
    keep_audio: bool,
) -> dict[str, Any]:
    meta = day["metadata"]
    date = meta["calendar_date"]
    day_idx = int(meta["day_index"])
    day_name = f"day_{day_idx:02d}_{date}"

    # Group by file format: videos/  manifests/  concat/
    videos_dir = output_dir / "videos"
    manifests_dir = output_dir / "manifests"
    concat_dir = output_dir / "concat"
    for d in (videos_dir, manifests_dir, concat_dir):
        d.mkdir(parents=True, exist_ok=True)

    out_mp4 = videos_dir / f"{day_name}.mp4"
    manifest_path = manifests_dir / f"{day_name}.json"
    concat_list = concat_dir / f"{day_name}.txt"

    included: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
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
        raise SystemExit(f"Day {day_idx} ({date}): no local mp4 files found under {video_root}")

    sources = [Path(item["source_path"]) for item in included]
    print(
        f"[day {day_idx:02d}] Stitching {len(included)}/{len(day['memory_content'])} clips "
        f"(mode={mode}, audio={'on' if keep_audio else 'off'}) -> {out_mp4}"
    )
    if missing:
        print(f"  Warning: {len(missing)} clips missing locally (skipped)")

    ffmpeg_concat(sources, out_mp4, concat_list, mode=mode, keep_audio=keep_audio)

    total_actual = sum(x["duration_actual"] for x in included)
    manifest = {
        "output_video": str(out_mp4),
        "concat_list": str(concat_list),
        "mode": mode,
        "keep_audio": keep_audio,
        "calendar_date": date,
        "day_index": day_idx,
        "day_theme": meta.get("day_theme"),
        "clips_total": len(day["memory_content"]),
        "clips_included": len(included),
        "clips_missing": len(missing),
        "duration_lifelog_included_hours": round(
            sum(x["duration_lifelog"] or 0 for x in included) / 3600, 3
        ),
        "duration_actual_hours": round(total_actual / 3600, 3),
        "included_clips": included,
        "missing_clips": missing,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  Done: {out_mp4} ({manifest['duration_actual_hours']:.2f}h)")
    print(f"  Manifest: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch lifelog day videos by video_uid")
    parser.add_argument(
        "--lifelog",
        type=Path,
        default=None,
        help="Full lifelog JSON (e.g. lifelog_egotailor_usa_enfp_21d.json)",
    )
    parser.add_argument(
        "--day-json",
        type=Path,
        default=None,
        help="Single-day JSON (legacy; same schema as lifelog['days'][i])",
    )
    parser.add_argument(
        "--day",
        type=int,
        nargs="*",
        default=None,
        help="Day index/indices to stitch when using --lifelog (0-20)",
    )
    parser.add_argument(
        "--all-days",
        action="store_true",
        help="Stitch every day in --lifelog",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument(
        "--mode",
        choices=["safe", "copy", "reencode"],
        default="safe",
        help="safe=MPEG-TS remux then concat (default); copy=fragile stream copy; "
             "reencode=libx264 (slow, most compatible)",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep audio tracks (default: drop audio; stream 0:1 DTS issues are common)",
    )
    args = parser.parse_args()

    # Backward-compatible defaults: if neither input given, use legacy day JSON
    lifelog = args.lifelog
    day_json = args.day_json
    if lifelog is None and day_json is None:
        if DEFAULT_LIFELOG.exists():
            lifelog = DEFAULT_LIFELOG
            if not args.all_days and args.day is None:
                args.day = [0]
        else:
            day_json = DEFAULT_DAY

    days = load_days(lifelog, day_json, args.day, args.all_days)
    manifests = []
    for day in days:
        manifests.append(
            stitch_one_day(
                day,
                args.output_dir,
                args.video_root,
                mode=args.mode,
                keep_audio=args.keep_audio,
            )
        )

    if len(manifests) > 1:
        summary_path = args.output_dir / "stitch_all_days_summary.json"
        summary = {
            "n_days": len(manifests),
            "mode": args.mode,
            "keep_audio": args.keep_audio,
            "days": [
                {
                    "day_index": m["day_index"],
                    "calendar_date": m["calendar_date"],
                    "output_video": m["output_video"],
                    "clips_included": m["clips_included"],
                    "clips_missing": m["clips_missing"],
                    "duration_actual_hours": m["duration_actual_hours"],
                }
                for m in manifests
            ],
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
