#!/usr/bin/env python3
"""
Hierarchical RAG + vLLM multimodal interest/preference analysis for EgoTailor lifelogs.

Examples:
  # Build index
  python -m analysis.run_analysis build-index

  # Auto mine interests (multi-day RAG + vLLM)
  python -m analysis.run_analysis mine-interests --auto

  # Query-specific analysis
  python -m analysis.run_analysis mine-interests -q "outdoor hiking exercise"

  # Analyze single clip by video_uid
  python -m analysis.run_analysis analyze-clip --video-uid 40b86eb9-a408-4119-ba7c-402b050be506

  # Dry-run without vLLM (stats + text fallback)
  python -m analysis.run_analysis mine-interests --auto --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analysis.analyzer import InterestAnalyzer, build_rag_from_lifelog, load_rag
from analysis.config import (
    DEFAULT_LIFELOG,
    INDEX_PATH,
    REPORT_PATH,
    VIDEO_REGISTRY_PATH,
    VLM_PROFILE_PATH,
    VLLM_API_BASE,
    VLLM_MODEL,
)
from analysis.mllm_client import VLLMClient
from analysis.video_registry import build_video_registry, save_registry
from analysis.vlm_profiler import VLMBehaviorProfiler, save_profile


def cmd_build_index(args: argparse.Namespace) -> None:
    lifelog_path = Path(args.lifelog)
    index_path = Path(args.index)
    print(f"Building hierarchical index from {lifelog_path}")
    rag = build_rag_from_lifelog(lifelog_path, index_path)
    for level, nodes in rag.index.items():
        print(f"  {level}: {len(nodes)} nodes")
    print(f"Saved -> {index_path}")


def _get_rag(args: argparse.Namespace):
    index_path = Path(args.index)
    if not index_path.exists() or args.rebuild_index:
        return build_rag_from_lifelog(Path(args.lifelog), index_path)
    return load_rag(index_path)


def _get_mllm(args: argparse.Namespace) -> VLLMClient | None:
    if args.dry_run:
        print("[dry-run] Skipping vLLM calls")
        return None
    client = VLLMClient(base_url=args.vllm_base, model=args.model)
    if client.health_check():
        print(f"vLLM connected: {args.vllm_base} model={args.model}")
        return client
    print(f"Warning: vLLM not reachable at {args.vllm_base}, falling back to stats/text-only")
    return None


def cmd_mine_interests(args: argparse.Namespace) -> None:
    rag = _get_rag(args)
    mllm = _get_mllm(args)
    analyzer = InterestAnalyzer(
        rag=rag,
        mllm=mllm,
        use_vision=not args.no_vision,
        max_clips_per_query=args.max_clips,
    )
    queries = args.queries if args.queries else None
    report = analyzer.mine_interests(queries=queries, auto=args.auto)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved -> {out}")
    profile = report.get("user_profile", {})
    print("\n=== User Profile Summary ===")
    print(profile.get("summary", profile))


def cmd_analyze_clip(args: argparse.Namespace) -> None:
    rag = _get_rag(args)
    node = rag.get_clip_by_video_uid(args.video_uid)
    if node is None:
        print(f"video_uid not found in lifelog: {args.video_uid}")
        sys.exit(1)
    mllm = _get_mllm(args)
    analyzer = InterestAnalyzer(rag=rag, mllm=mllm, use_vision=not args.no_vision)
    from analysis.video_loader import load_clip_visuals

    visuals = load_clip_visuals(args.video_uid)
    print(f"Video available: {visuals['available']} path={visuals.get('path')}")
    result = analyzer.analyze_clip(node, frames_b64=visuals.get("frames_b64") or None)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_retrieve(args: argparse.Namespace) -> None:
    rag = _get_rag(args)
    for q in args.queries:
        r = rag.retrieve(q)
        print(f"\n=== Query: {q} ===")
        print(r.to_context_text()[:3000])


def cmd_build_registry(args: argparse.Namespace) -> None:
    registry = build_video_registry(Path(args.lifelog))
    path = save_registry(registry, Path(args.registry))
    print(f"Video registry saved -> {path}")
    print(
        f"  {registry['clips_with_local_video']}/{registry['total_clips']} clips have local mp4 "
        f"({registry['coverage_ratio']*100:.1f}% unique uid coverage)"
    )


def cmd_vlm_profile(args: argparse.Namespace) -> None:
    rag = _get_rag(args)
    mllm = _get_mllm(args)
    if mllm is None:
        print("vlm-profile requires a running vLLM server (remove --dry-run)")
        sys.exit(1)

    registry_path = Path(args.registry)
    if registry_path.exists() and not args.rebuild_registry:
        import json
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = build_video_registry(Path(args.lifelog))
        save_registry(registry, registry_path)
    print(
        f"Vision coverage: {registry['clips_with_local_video']}/{registry['total_clips']} clips "
        f"({registry['unique_uids_with_local_video']} unique uids)"
    )

    profiler = VLMBehaviorProfiler(
        rag=rag,
        mllm=mllm,
        registry=registry,
        frames_per_clip=args.frames,
        vision_first=True,
        show_progress=not args.no_progress,
    )
    queries = args.queries
    if args.auto:
        from analysis.config import DEFAULT_INTEREST_QUERIES
        queries = DEFAULT_INTEREST_QUERIES
    elif not queries:
        queries = [
            "outdoor hiking exercise preferences",
            "food cooking eating habits",
            "work office meeting routines",
            "home leisure entertainment",
        ]

    print(f"Running VLM profile: {len(queries)} queries x {args.max_clips} clips/query")
    report = profiler.run_profile(queries=queries, max_clips_per_query=args.max_clips)
    out = save_profile(report, Path(args.output))
    print(f"Profile saved -> {out}")
    print(f"  Vision analyses: {report['vision_analyses_count']}")
    profile = report.get("behavior_profile", {})
    print("\n=== Behavior Profile Summary ===")
    if isinstance(profile, dict):
        print(profile.get("summary", json.dumps(profile, indent=2)[:1500]))
    else:
        print(profile)


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lifelog", default=str(DEFAULT_LIFELOG))
    common.add_argument("--index", default=str(INDEX_PATH))
    common.add_argument("--output", default=str(REPORT_PATH))
    common.add_argument("--vllm-base", default=VLLM_API_BASE)
    common.add_argument("--model", default=VLLM_MODEL)
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--no-vision", action="store_true")
    common.add_argument("--rebuild-index", action="store_true")
    common.add_argument("--registry", default=str(VIDEO_REGISTRY_PATH))
    common.add_argument("--rebuild-registry", action="store_true")
    common.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Fixed frames per clip (default: adaptive by video duration)",
    )
    common.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")

    parser = argparse.ArgumentParser(description="EgoTailor hierarchical RAG + MLLM analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-index", parents=[common], help="Build hierarchical RAG index")
    p_build.set_defaults(func=cmd_build_index)

    p_mine = sub.add_parser("mine-interests", parents=[common], help="Mine user interests via hierarchical RAG")
    p_mine.add_argument("-q", "--queries", nargs="+", default=None)
    p_mine.add_argument("--auto", action="store_true", help="Run default multi-topic queries")
    p_mine.add_argument("--max-clips", type=int, default=3, help="MLLM clip analyses per query")
    p_mine.set_defaults(func=cmd_mine_interests)

    p_clip = sub.add_parser("analyze-clip", parents=[common], help="Analyze one clip by video_uid")
    p_clip.add_argument("--video-uid", required=True)
    p_clip.set_defaults(func=cmd_analyze_clip)

    p_ret = sub.add_parser("retrieve", parents=[common], help="Preview RAG retrieval (no MLLM)")
    p_ret.add_argument("-q", "--queries", nargs="+", required=True)
    p_ret.set_defaults(func=cmd_retrieve)

    p_reg = sub.add_parser("build-registry", parents=[common], help="Map lifelog video_uid -> Ego4D mp4")
    p_reg.set_defaults(func=cmd_build_registry)

    p_vlm = sub.add_parser("vlm-profile", parents=[common], help="RAG + Ego4D video VLM behavior profiling")
    p_vlm.add_argument("-q", "--queries", nargs="+", default=None)
    p_vlm.add_argument("--auto", action="store_true", help="All default interest queries")
    p_vlm.add_argument("--max-clips", type=int, default=2, help="VLM clip analyses per query (vision-first)")
    p_vlm.set_defaults(func=cmd_vlm_profile, output=str(VLM_PROFILE_PATH))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
