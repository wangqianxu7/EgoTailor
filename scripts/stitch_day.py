#!/usr/bin/env python3
"""Stitch Ego4D clips into day-level videos, keeping audio in sync.

Input: the lifelog JSON. Each day's memory_content[*].video_uid maps to
{video_root}/{video_uid}.mp4; clips are concatenated in lifelog order.

Why two passes (still only ONE video encode):

  pass 1 - normalize: clip -> norm/<uid>.mkv
      video: CFR h264, fixed WxH (letterboxed), yuv420p, one timebase
      audio: pcm_s16le 48k stereo through `aresample=async=1`, which pads or
             trims audio against the video clock, so intra-clip drift dies here.
             Clips without an audio track get anullsrc silence, otherwise the
             concat would shift every following clip's audio.
      Cached and parallel: a uid is encoded once even if it appears on
      several days; rerunning skips finished files.

  pass 2 - concat: concat demuxer, `-c:v copy -c:a aac`
      Video is copied (no second encode). Audio is encoded ONCE across the
      whole day, so there is no per-segment AAC priming delay to accumulate.
      That accumulation is what made sound drift later and later in the day.

Run on the server where the Ego4D mp4s live:

  python scripts/stitch_day.py --all-days \
    --video-root /mnt/data_oss/raw_data/Ego4d/v2/full_scale \
    --output-dir /mnt/data/workspace/outputs/EgoTailor_30days_videos \
    --jobs 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Allow `python scripts/<name>.py` from the EgoTailor root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generation.config import TOTAL_DAYS, resolve_lifelog  # noqa: E402

DEFAULT_VIDEO_ROOT = Path("/mnt/data_oss/raw_data/Ego4d/v2/full_scale")
# Output dir carries the run length: a 30-day run must not overwrite the
# 21-day results sitting next to it. These are expensive to recompute.
DEFAULT_OUTPUT_DIR = Path(f"/mnt/data/workspace/outputs/EgoTailor_{TOTAL_DAYS}days_videos")


# ---------------------------------------------------------------- ffprobe ---

def ffprobe(path: Path, *args: str) -> str:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", *args,
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True,
    )
    return out.strip()


def probe_duration(path: Path) -> float:
    """Seconds, or 0.0 if the file is too broken to answer."""
    try:
        return float(ffprobe(path, "-show_entries", "format=duration"))
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def has_audio(path: Path) -> bool:
    return bool(ffprobe(path, "-select_streams", "a:0",
                        "-show_entries", "stream=codec_type"))


def stream_duration(path: Path, stream: str) -> float:
    val = ffprobe(path, "-select_streams", stream,
                  "-show_entries", "stream=duration").splitlines()
    try:
        return float(val[0])
    except (IndexError, ValueError):
        return 0.0


# ----------------------------------------------------------------- ffmpeg ---

def run(cmd: list[str], dry_run: bool = False) -> None:
    if dry_run:
        print("  $", " ".join(cmd))
        return
    subprocess.run(
        [cmd[0], "-hide_banner", "-loglevel", "error", "-nostdin", *cmd[1:]],
        check=True,
    )


def normalize(src: Path, dst: Path, cfg: argparse.Namespace,
              tag: str = "") -> Path:
    """Encode one clip to the common format. Cached; atomic via .part file."""
    started = time.monotonic()
    tmp = dst.with_suffix(".part.mkv")
    w, h = cfg.width, cfg.height
    # Both streams are made endless (tpad clones the last frame, apad appends
    # silence) and then cut at the same `-t`, so every normalized clip has
    # video and audio of *identical* length. Ego4D files often ship streams of
    # unequal length; uncut, each mismatch would shove the whole rest of the
    # day's audio out of place.
    n_frames = max(1, round((stream_duration(src, "v:0") or probe_duration(src))
                            * cfg.fps))
    dur = n_frames / cfg.fps

    # A cache hit is only a hit if the file is actually whole: earlier runs
    # muxed straight onto object storage, which cannot seek back to write a
    # trailer, so the cache holds truncated clips that look fine by name.
    if dst.exists():
        if abs(probe_duration(dst) - dur) <= 1.0:
            print(f"  {tag} {dst.stem[:8]} cached")
            return dst
        print(f"  {tag} {dst.stem[:8]} cached copy is truncated, re-encoding")
        dst.unlink()
    vf = (
        f"fps={cfg.fps},"
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,format=yuv420p,tpad=stop=-1:stop_mode=clone"
    )

    cmd = ["ffmpeg", "-y", "-i", str(src)]
    audio_from_source = not cfg.mute and has_audio(src)
    if not audio_from_source:
        # Silence keeps every clip's audio track present and exactly as long as
        # its video; a missing track here would desync everything downstream.
        cmd += ["-f", "lavfi", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000"]

    cmd += ["-map", "0:v:0", "-vf", vf,
            "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
            "-g", str(cfg.fps * 2), "-pix_fmt", "yuv420p"]

    if audio_from_source:
        # async=1 puts audio back on the video clock: a late start is padded
        # with silence, an overrun is dropped.
        cmd += ["-map", "0:a:0", "-af", "aresample=async=1:first_pts=0,apad"]
    else:
        cmd += ["-map", "1:a:0"]
    # PCM has no encoder priming delay, so segment boundaries stay sample-exact.
    cmd += ["-t", f"{dur:.6f}",
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
            "-sn", "-dn", "-map_metadata", "-1", str(tmp)]

    run(cmd, cfg.dry_run)
    if not cfg.dry_run:
        # ffmpeg here can log "Error writing trailer" and still exit 0, so the
        # exit code is not evidence. Ask the file. A clip whose trailer never
        # landed has no usable duration, and letting it into the cache poisons
        # every day that uses this uid -- silently, forever, because the cache
        # check only looks at the filename.
        got = probe_duration(tmp)
        if abs(got - dur) > 1.0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"{src.name}: normalized file is {got:.1f}s, expected "
                f"{dur:.1f}s -- encode failed (disk full? check df -h)")
        tmp.rename(dst)
        print(f"  {tag} {dst.stem[:8]} {dur / 60:5.1f}min clip, "
              f"{'silent, ' if not audio_from_source else ''}"
              f"encoded in {time.monotonic() - started:.0f}s")
    return dst


def concat(parts: list[Path], listfile: Path, out: Path,
           cfg: argparse.Namespace) -> None:
    """Concat the normalized parts. Atomic via .part file, like normalize().

    An mp4's moov index is written last, so a killed ffmpeg leaves a file that
    no player can open. Under the final name that corpse also looks "done" to
    the skip check below, and every rerun would step over it. No +faststart:
    it rewrites the whole multi-GB file at the end just to move moov to the
    front, which only pays off for HTTP streaming. Add it later on a finished
    file with `ffmpeg -i in.mp4 -c copy -movflags +faststart out.mp4`.
    """
    listfile.write_text(
        "".join(f"file '{str(p.resolve())}'\n" for p in parts)
    )
    tmp = out.with_suffix(".part.mp4")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
         str(tmp)], cfg.dry_run)
    if not cfg.dry_run:
        tmp.rename(out)


# -------------------------------------------------------------------- day ---

def load_days(lifelog: Path, day_indices: list[int] | None,
              all_days: bool) -> list[dict[str, Any]]:
    days = json.loads(lifelog.read_text())["days"]
    if all_days:
        return days
    if not day_indices:
        raise SystemExit("Pass --day N [N ...] or --all-days")
    wanted = set(day_indices)
    picked = [d for d in days if int(d["metadata"]["day_index"]) in wanted]
    missing = wanted - {int(d["metadata"]["day_index"]) for d in picked}
    if missing:
        raise SystemExit(f"day_index not in lifelog: {sorted(missing)}")
    return picked


def stitch_day(day: dict[str, Any], cfg: argparse.Namespace,
               dirs: dict[str, Path]) -> dict[str, Any]:
    meta = day["metadata"]
    day_idx, date = int(meta["day_index"]), meta["calendar_date"]
    name = f"day_{day_idx:02d}_{date}"
    out = dirs["videos"] / f"{name}.mp4"

    clips, missing = [], []
    for i, clip in enumerate(day["memory_content"]):
        src = cfg.video_root / f"{clip['video_uid']}.mp4"
        entry = {
            "index": i,
            "video_uid": clip["video_uid"],
            "slot_id": clip.get("slot_id"),
            "plan_chunk": clip.get("plan_chunk"),
            "start_timestamp": clip.get("start_timestamp"),
            "end_timestamp": clip.get("end_timestamp"),
            "duration_lifelog": clip.get("duration"),
        }
        (clips if src.exists() else missing).append(entry)
        if src.exists():
            entry["source_path"] = str(src)

    if not clips:
        print(f"[day {day_idx:02d}] no local mp4 under {cfg.video_root}, skipped")
        return {}

    print(f"[day {day_idx:02d}] {date}: {len(clips)}/{len(day['memory_content'])}"
          f" clips -> {out.name}" + (f"  ({len(missing)} missing)" if missing else ""))

    if out.exists() and not cfg.overwrite:
        print("  already exists, skipped (use --overwrite to redo)")
        return {}

    # pass 1 (parallel, cached across days)
    todo = [(Path(c["source_path"]), dirs["norm"] / f"{c['video_uid']}.mkv")
            for c in clips]
    with ThreadPoolExecutor(max_workers=cfg.jobs) as pool:
        list(pool.map(
            lambda t: normalize(t[1][0], t[1][1], cfg,
                                f"[{t[0] + 1}/{len(todo)}]"),
            enumerate(todo)))

    # pass 2
    parts = [dst for _, dst in todo]
    print(f"  concat -> {out.name} (video copied, audio encoded once)")
    concat(parts, dirs["concat"] / f"{name}.txt", out, cfg)
    if cfg.dry_run:
        return {}

    v_dur, a_dur = stream_duration(out, "v:0"), stream_duration(out, "a:0")
    for c, (_, dst) in zip(clips, todo):
        c["duration_normalized"] = probe_duration(dst)
    # The parts are the ground truth: the day must be as long as its clips.
    expected = sum(c["duration_normalized"] for c in clips)
    complete = abs(v_dur - expected) <= 1.0
    if not complete:
        print(f"  !! WARNING {out.name} is {v_dur / 3600:.2f}h but its clips "
              f"sum to {expected / 3600:.2f}h -- output is truncated, "
              f"rerun this day with --overwrite")
    manifest = {
        "output_video": str(out),
        "day_index": day_idx,
        "calendar_date": date,
        "day_theme": meta.get("day_theme"),
        "clips_included": len(clips),
        "clips_missing": len(missing),
        "duration_hours": round(v_dur / 3600, 3),
        "duration_hours_expected": round(expected / 3600, 3),
        "complete": complete,
        "av_offset_seconds": round(a_dur - v_dur, 3),
        "profile": {"width": cfg.width, "height": cfg.height, "fps": cfg.fps,
                    "crf": cfg.crf, "muted": cfg.mute},
        "included_clips": clips,
        "missing_clips": missing,
    }
    path = dirs["manifests"] / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  done: {v_dur / 3600:.2f}h, audio-video offset "
          f"{a_dur - v_dur:+.3f}s -> {out}")
    return manifest


# ------------------------------------------------------------------- main ---

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lifelog", type=Path, default=None)
    p.add_argument("--day", type=int, nargs="*", help=f"day index/indices (0-{TOTAL_DAYS - 1})")
    p.add_argument("--all-days", action="store_true")
    p.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4),
                   help="parallel clip encodes in pass 1")
    p.add_argument("--size", default="960x720", help="normalized WxH (letterboxed)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--mute", action="store_true", help="silent output")
    p.add_argument("--overwrite", action="store_true", help="redo existing days")
    p.add_argument("--clean-temp", action="store_true",
                   help="delete the norm/ cache once all days are done "
                        "(default: keep it, so reruns are cheap)")
    p.add_argument("--dry-run", action="store_true", help="print ffmpeg only")
    cfg = p.parse_args()
    cfg.lifelog = resolve_lifelog(cfg.lifelog)
    cfg.width, cfg.height = (int(x) for x in cfg.size.lower().split("x"))

    dirs = {n: cfg.output_dir / n
            for n in ("videos", "manifests", "concat", "norm")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    manifests = [m for day in load_days(cfg.lifelog, cfg.day, cfg.all_days)
                 if (m := stitch_day(day, cfg, dirs))]

    if manifests:
        summary = cfg.output_dir / "stitch_summary.json"
        summary.write_text(json.dumps(
            {"n_days": len(manifests),
             "profile": manifests[0]["profile"],
             "days": [{k: m[k] for k in
                       ("day_index", "calendar_date", "output_video",
                        "clips_included", "clips_missing", "duration_hours",
                        "complete", "av_offset_seconds")} for m in manifests]},
            indent=2, ensure_ascii=False))
        print(f"summary: {summary}")

    if cfg.clean_temp and not cfg.dry_run:
        for f in dirs["norm"].glob("*.mkv"):
            f.unlink()
        print("norm cache cleaned")


if __name__ == "__main__":
    main()
