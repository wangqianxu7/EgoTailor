#!/usr/bin/env python3
"""Filter X-LeBench video_info.json to clips present under the local Ego4D root.

Usage:
  cd /root/EgoTailor
  python scripts/filter_local_video_info.py
  python scripts/filter_local_video_info.py \
    --source ../X-LeBench/generation/ego4d_info/video_info.json \
    --video-root /mnt/data_oss/raw_data/Ego4d/v2/full_scale \
    --out-dir generation/ego4d_info
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SOURCE = Path(__file__).resolve().parents[1].parent / "X-LeBench" / "generation" / "ego4d_info" / "video_info.json"
DEFAULT_VIDEO_ROOT = Path("/mnt/data_oss/raw_data/Ego4d/v2/full_scale")
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "generation" / "ego4d_info"


def filter_video_info(
    source: Path,
    video_root: Path,
    out_dir: Path,
    copy_refs: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    local_uids = {p.stem for p in video_root.glob("*.mp4")}

    with open(source) as f:
        videos = json.load(f)

    filtered = []
    for v in videos:
        uid = v["video_uid"]
        if uid not in local_uids:
            continue
        entry = dict(v)
        entry["local_video_path"] = str(video_root / f"{uid}.mp4")
        filtered.append(entry)

    out_path = out_dir / "video_info.json"
    with open(out_path, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    with open(out_dir / "available_video_uids.json", "w") as f:
        json.dump([v["video_uid"] for v in filtered], f, indent=2)

    source_uids = {v["video_uid"] for v in videos}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video_info": str(source.resolve()),
        "ego4d_video_root": str(video_root.resolve()),
        "source_count": len(videos),
        "local_mp4_count": len(local_uids),
        "filtered_count": len(filtered),
        "coverage_vs_source": round(len(filtered) / len(videos), 4) if videos else 0,
        "local_only_not_in_video_info": len(local_uids - source_uids),
        "by_source": dict(Counter(v["video_source"] for v in filtered)),
        "by_main_scene": dict(Counter(v["main_scene"] for v in filtered)),
        "by_time_period": dict(Counter(v["time_period"] for v in filtered)),
        "usa_count": sum(1 for v in filtered if v["video_source"] == "USA"),
        "duration_stats": {
            "min": min((v["video_duration"] for v in filtered), default=0),
            "max": max((v["video_duration"] for v in filtered), default=0),
            "mean": round(sum(v["video_duration"] for v in filtered) / len(filtered), 2) if filtered else 0,
            "total_hours": round(sum(v["video_duration"] for v in filtered) / 3600, 2) if filtered else 0,
        },
    }
    with open(out_dir / "filter_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if copy_refs:
        for name in ("ref_scenarios_list.json", "univ_loc_map.json"):
            src = source.parent / name
            if src.exists():
                (out_dir / name).write_bytes(src.read_bytes())

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-copy-refs", action="store_true")
    args = parser.parse_args()

    manifest = filter_video_info(
        source=args.source,
        video_root=args.video_root,
        out_dir=args.out_dir,
        copy_refs=not args.no_copy_refs,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nWrote filtered library -> {args.out_dir / 'video_info.json'}")


if __name__ == "__main__":
    main()
