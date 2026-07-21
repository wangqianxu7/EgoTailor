#!/usr/bin/env python3
"""Copy each lifelog day's Ego4D clips into one folder, in time order.

No transcoding, no concatenation -- one directory per day, with filenames built
so that plain alphabetical order == chronological order:

    day_00_2026-01-05/
        00_0653_30m00s_wake-hygiene_cee444e9-6c62-4e35-a322-05b45f418718.mp4
        01_0723_24m02s_breakfast-coffee_201af2e0-72dd-466c-8280-8c3dc00a7981.mp4
        ...
        _index.json        per-clip uid / slot / timestamps / duration / size
        _playlist.m3u      open in VLC to watch the whole day back to back
        _concat.txt        ready for scripts/stitch_day.py if you stitch later

    index _ HHMM _ duration _ slot _ full video_uid

Copying is the default and it is not cheap: measured on this dataset the clips
run ~2.1 GB per hour (1.0 GB/h for a day spent sitting indoors, 5.2 GB/h for a
day of cycling -- egocentric motion drives the bitrate), so ~12.8 GB per day and
~269 GB for all 21 days. Free space is checked before each day starts, copies
run --jobs at a time, and an interrupted run resumes. Use --mode symlink for a
zero-byte layout that points back at the Ego4D mount instead.

    python scripts/collect_day_clips.py --all-days \
      --video-root /mnt/data_oss/raw_data/Ego4d/v2/full_scale \
      --output-dir /mnt/data/workspace/outputs/EgoTailor_21days_clips
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DEFAULT_LIFELOG = Path("output/lifelog/lifelog_egotailor_usa_enfp_21d.json")
DEFAULT_VIDEO_ROOT = Path("/mnt/data_oss/raw_data/Ego4d/v2/full_scale")
DEFAULT_OUTPUT_DIR = Path("/mnt/data/workspace/outputs/EgoTailor_21days_clips")

GB = 1024 ** 3


def probe_duration(path: Path) -> float | None:
    """Real duration in seconds, or None if ffprobe is unhappy / unavailable."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return round(float(out), 3)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def dur_tag(seconds: float | None) -> str:
    """1802.4 -> '30m02s'. Fixed width, so it never breaks the name sort."""
    if not seconds:
        return "__m__s"
    s = int(round(seconds))
    return f"{s // 60:02d}m{s % 60:02d}s"


def slug(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "clip").lower()).strip("-") or "clip"


def place(src: Path, dst: Path, mode: str) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)          # same filesystem only
    else:
        # copy to .part first: an interrupted run then leaves no half file that
        # the size check would mistake for a finished one.
        tmp = dst.with_name(dst.name + ".part")
        shutil.copyfile(src, tmp)
        shutil.copystat(src, tmp)
        tmp.rename(dst)


def collect_day(day: dict[str, Any], cfg: argparse.Namespace) -> dict[str, Any]:
    meta = day["metadata"]
    day_idx, date = int(meta["day_index"]), meta["calendar_date"]
    day_dir = cfg.output_dir / f"day_{day_idx:02d}_{date}"

    # The lifelog is already chronological, but sort anyway so the numeric
    # prefix can be trusted as ground truth.
    clips = sorted(day["memory_content"], key=lambda c: c["start_timestamp"])

    entries, missing = [], []
    for i, clip in enumerate(clips):
        uid = clip["video_uid"]
        src = cfg.video_root / f"{uid}.mp4"
        start = clip.get("start_timestamp", "")
        entry = {
            "index": i,
            "video_uid": uid,
            "slot_id": clip.get("slot_id"),
            "plan_chunk": clip.get("plan_chunk"),
            "start_timestamp": start,
            "end_timestamp": clip.get("end_timestamp"),
            "duration_lifelog": clip.get("duration"),
            "source_path": str(src),
        }
        if not src.exists():
            missing.append(entry)
            continue
        entry["size_bytes"] = src.stat().st_size
        entries.append(entry)

    # The real duration goes into the filename, so it must be known before the
    # name exists. ffprobe over a FUSE mount is slow enough to be worth running
    # in parallel; --no-probe falls back to the lifelog's planned duration.
    with ThreadPoolExecutor(max_workers=max(1, cfg.jobs)) as pool:
        probed = [None] * len(entries) if cfg.no_probe else list(pool.map(
            lambda e: probe_duration(Path(e["source_path"])), entries))
    for e, secs in zip(entries, probed):
        e["duration_actual"] = secs
        start = e["start_timestamp"]
        hhmm = start[11:16].replace(":", "") if len(start) >= 16 else f"{e['index']:04d}"
        e["filename"] = (f"{e['index']:02d}_{hhmm}_"
                         f"{dur_tag(secs or e['duration_lifelog'])}_"
                         f"{slug(e['slot_id'])}_{e['video_uid']}.mp4")

    need = sum(e["size_bytes"] for e in entries)
    print(f"[day {day_idx:02d}] {date} {meta.get('day_theme', ''):<28} "
          f"{len(entries):2d}/{len(clips):2d} clips, {need / GB:5.1f} GB"
          + (f"  ({len(missing)} MISSING)" if missing else ""))
    for e in missing:
        print(f"           missing: {e['video_uid']}  ({e['slot_id']})")
    if cfg.dry_run:
        return {"day_index": day_idx, "calendar_date": date, "dir": str(day_dir),
                "clips_present": len(entries), "clips_missing": len(missing),
                "bytes": need}

    day_dir.mkdir(parents=True, exist_ok=True)
    if cfg.mode == "copy":
        free = shutil.disk_usage(day_dir).free
        if free < need * 1.05:
            raise SystemExit(
                f"  Not enough space: need {need / GB:.1f} GB, "
                f"{free / GB:.1f} GB free on {day_dir}. "
                f"Free some up, or use --mode symlink.")

    def transfer(n: int, e: dict[str, Any]) -> int:
        src, dst = Path(e["source_path"]), day_dir / e["filename"]
        # Resume: a finished copy of the right size is left alone.
        if dst.exists() and not cfg.force and (
                cfg.mode != "copy" or dst.stat().st_size == e["size_bytes"]):
            print(f"  [{n:2d}/{len(entries)}] {e['filename']}  already there")
            return 0
        t0 = time.monotonic()
        place(src, dst, cfg.mode)
        dt = max(time.monotonic() - t0, 1e-9)
        rate = f", {e['size_bytes'] / 1e6 / dt:.0f} MB/s" if dt > 0.5 else ""
        print(f"  [{n:2d}/{len(entries)}] {e['filename']}  "
              f"{e['size_bytes'] / GB:.2f} GB{rate}")
        return e["size_bytes"]

    # Object-storage mounts rarely saturate on one stream, so copies run
    # concurrently. shutil releases the GIL on the read/write syscalls.
    t_day = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, cfg.jobs)) as pool:
        moved = sum(pool.map(lambda a: transfer(*a), enumerate(entries, 1)))
    elapsed = time.monotonic() - t_day
    if moved:
        print(f"           {moved / GB:.1f} GB in {elapsed / 60:.1f} min "
              f"= {moved / 1e6 / elapsed:.0f} MB/s aggregate ({cfg.jobs} jobs)")

    (day_dir / "_index.json").write_text(json.dumps({
        "day_index": day_idx,
        "calendar_date": date,
        "day_of_week": meta.get("day_of_week"),
        "day_theme": meta.get("day_theme"),
        "mode": cfg.mode,
        "clips_total": len(clips),
        "clips_present": len(entries),
        "clips_missing": len(missing),
        "bytes": need,
        "clips": entries,
        "missing_clips": missing,
    }, indent=2, ensure_ascii=False))

    # #EXTINF wants integer seconds; -1 means "unknown", which VLC accepts.
    lines = [f"#EXTM3U\n# day {day_idx:02d} {date} {meta.get('day_theme', '')}\n"]
    for e in entries:
        secs = int(e.get("duration_actual") or e.get("duration_lifelog") or -1)
        lines.append(f"#EXTINF:{secs},{e['start_timestamp'][11:16]} "
                     f"{e.get('plan_chunk') or e.get('slot_id')}\n{e['filename']}\n")
    (day_dir / "_playlist.m3u").write_text("".join(lines))
    (day_dir / "_concat.txt").write_text(
        "".join(f"file '{e['filename']}'\n" for e in entries))

    return {"day_index": day_idx, "calendar_date": date, "dir": str(day_dir),
            "clips_present": len(entries), "clips_missing": len(missing),
            "bytes": need}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lifelog", type=Path, default=DEFAULT_LIFELOG)
    p.add_argument("--day", type=int, nargs="*", help="day index/indices (0-20)")
    p.add_argument("--all-days", action="store_true")
    p.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--mode", choices=["copy", "symlink", "hardlink"],
                   default="copy", help="copy = real files (default); symlink = "
                                        "zero-byte pointers into --video-root")
    p.add_argument("--jobs", type=int, default=4,
                   help="parallel copies; object-storage mounts usually go "
                        "several times faster with 4-8 (default: 4)")
    p.add_argument("--no-probe", action="store_true",
                   help="skip ffprobe; filename duration comes from the lifelog")
    p.add_argument("--force", action="store_true", help="re-copy files already there")
    p.add_argument("--dry-run", action="store_true",
                   help="report clips and disk space, write nothing")
    cfg = p.parse_args()

    if not cfg.lifelog.exists():
        raise SystemExit(f"Lifelog not found: {cfg.lifelog.resolve()}\n"
                         f"(the default is relative -- run from the repo root, "
                         f"or pass --lifelog /abs/path.json)")
    if not cfg.video_root.is_dir():
        raise SystemExit(f"--video-root is not a directory: {cfg.video_root}")
    days = json.loads(cfg.lifelog.read_text())["days"]
    if not cfg.all_days:
        if not cfg.day:
            raise SystemExit("Pass --day N [N ...] or --all-days")
        wanted = set(cfg.day)
        days = [d for d in days if int(d["metadata"]["day_index"]) in wanted]
        if len(days) != len(wanted):
            raise SystemExit(f"day_index not in lifelog: "
                             f"{sorted(wanted - {int(d['metadata']['day_index']) for d in days})}")

    if not cfg.dry_run:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary = [collect_day(d, cfg) for d in days]

    present = sum(s["clips_present"] for s in summary)
    absent = sum(s["clips_missing"] for s in summary)
    total = sum(s["bytes"] for s in summary)
    print(f"\n{len(summary)} days, {present} clips, {absent} missing, "
          f"{total / GB:.1f} GB ({cfg.mode})")
    if cfg.dry_run:
        print("dry run: nothing written")
        return
    path = cfg.output_dir / "_collect_summary.json"
    path.write_text(json.dumps(
        {"mode": cfg.mode, "video_root": str(cfg.video_root),
         "clips_present": present, "clips_missing": absent,
         "bytes": total, "days": summary}, indent=2, ensure_ascii=False))
    print(f"summary: {path}")


if __name__ == "__main__":
    main()
