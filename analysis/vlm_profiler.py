"""Fine-grained behavior/preference profiling: RAG context + Ego4D video + vLLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from analysis.config import (
    DEFAULT_INTEREST_QUERIES,
    FRAMES_PER_CLIP,
    VLM_PROFILE_PATH,
)
from analysis.mllm_client import VLLMClient
from analysis.rag_retriever import HierarchicalRAG, RetrievalResult
from analysis.video_loader import load_clip_visuals
from analysis.video_registry import build_video_registry, save_registry

CLIP_VLM_SYSTEM = """You are an expert egocentric-behavior analyst.
Given sampled frames from a first-person Ego4D video clip AND structured lifelog/RAG metadata,
infer fine-grained user behaviors, interests, and preferences.

Return valid JSON only:
{
  "observed_activities": ["..."],
  "objects_and_places": ["..."],
  "social_interaction": "none|minimal|moderate|high",
  "behavior_tags": ["commuting", "cooking", ...],
  "preference_hypotheses": [
    {"topic": "...", "evidence": "...", "confidence": "high|medium|low"}
  ],
  "plan_vs_observation": "how observed video relates to scheduled plan_chunk",
  "notable_details": ["..."]
}"""

SYNTHESIS_VLM_SYSTEM = """Synthesize hierarchical RAG + VLM clip analyses into a user behavior & preference profile.
Return valid JSON:
{
  "summary": "...",
  "core_interests": [{"topic", "evidence", "confidence", "supporting_video_uids"}],
  "habitual_patterns": [{"pattern", "time_context", "frequency_hint"}],
  "preferences": [{"category", "preference", "reasoning"}],
  "weekday_vs_weekend": "...",
  "anomaly_insights": ["..."],
  "data_quality_note": "vision coverage and limitations"
}"""


class VLMBehaviorProfiler:
    def __init__(
        self,
        rag: HierarchicalRAG,
        mllm: VLLMClient,
        registry: dict[str, Any],
        frames_per_clip: int | None = FRAMES_PER_CLIP,
        vision_first: bool = True,
        show_progress: bool = True,
    ):
        self.rag = rag
        self.mllm = mllm
        self.registry = registry
        self.frames_per_clip = frames_per_clip
        self.vision_first = vision_first
        self.show_progress = show_progress
        self._available = set(registry.get("uid_to_path", {}))

    def _log(self, msg: str) -> None:
        print(msg)

    def _prioritize_clips(self, retrieval: RetrievalResult, max_clips: int) -> list[dict[str, Any]]:
        clips = retrieval.all_clip_nodes()
        if self.vision_first:
            with_video = [c for c in clips if c.get("video_uid") in self._available]
            without = [c for c in clips if c.get("video_uid") not in self._available]
            clips = with_video + without
        return clips[:max_clips]

    def analyze_clip_vlm(
        self,
        clip: dict[str, Any],
        rag_context: str,
    ) -> dict[str, Any]:
        uid = clip["video_uid"]
        visuals = load_clip_visuals(
            uid,
            num_frames=self.frames_per_clip,
            duration_hint_sec=clip.get("duration"),
        )
        meta = (
            f"## Lifelog metadata\n"
            f"- video_uid: {uid}\n"
            f"- time: {clip.get('start_timestamp')} -> {clip.get('end_timestamp')}\n"
            f"- day: {clip.get('calendar_date')} slot={clip.get('slot_id')}\n"
            f"- scene: {clip.get('main_scene')}\n"
            f"- scenarios: {', '.join(clip.get('video_scenarios', []))}\n"
            f"- scheduled plan: {clip.get('plan_chunk')}\n"
            f"- Ego4D caption: {clip.get('consolidated_summary')}\n\n"
            f"## Hierarchical RAG context\n{rag_context[:2500]}\n"
        )
        result: dict[str, Any] = {
            "video_uid": uid,
            "node_id": clip.get("node_id"),
            "video_available": visuals["available"],
            "video_path": visuals.get("path"),
            "frame_count": visuals.get("frame_count", 0),
            "duration_min": visuals.get("duration_min"),
            "sampling": visuals.get("sampling"),
        }
        if visuals["available"] and visuals["frames_b64"]:
            try:
                result["analysis"] = self.mllm.chat_vision_json(
                    CLIP_VLM_SYSTEM,
                    meta + "\nAnalyze the attached egocentric frames.",
                    visuals["frames_b64"],
                )
                result["vision_used"] = True
            except (json.JSONDecodeError, Exception) as exc:
                raw = self.mllm.chat_vision(
                    CLIP_VLM_SYSTEM,
                    meta + "\nAnalyze the attached egocentric frames.",
                    visuals["frames_b64"],
                )
                result["vision_used"] = True
                result["raw_analysis"] = raw
                result["analysis"] = {"notable_details": [raw], "parse_error": str(exc)}
        else:
            result["vision_used"] = False
            text = self.mllm.chat_text(
                CLIP_VLM_SYSTEM,
                meta + "\n(No local video; analyze from metadata and caption only. Return JSON.)",
                temperature=0.1,
            )
            result["raw_analysis"] = text
            try:
                t = text.strip()
                if t.startswith("```"):
                    t = t.split("```")[1]
                    if t.startswith("json"):
                        t = t[4:]
                result["analysis"] = json.loads(t.strip())
            except json.JSONDecodeError:
                result["analysis"] = {"notable_details": [text]}

        return result

    def analyze_query(
        self,
        query: str,
        max_clips: int = 3,
        query_idx: int = 0,
        n_queries: int = 1,
        step_bar: tqdm | None = None,
    ) -> dict[str, Any]:
        retrieval = self.rag.retrieve(query)
        rag_ctx = retrieval.to_context_text()
        out: dict[str, Any] = {
            "query": query,
            "rag_context": rag_ctx,
            "rag_hits": {
                "period": len(retrieval.period_hits),
                "day": len(retrieval.day_hits),
                "hour": len(retrieval.hour_hits),
                "clip": len(retrieval.clip_hits),
            },
            "text_levels": {},
            "clip_analyses": [],
        }

        text_steps = [
            ("period", retrieval.period_hits, "Summarize multi-day behavior patterns for the query."),
            ("day", retrieval.day_hits, "Summarize daily-level behavior patterns for the query."),
            ("hour", retrieval.hour_hits, "Summarize hour-block behavior patterns for the query."),
        ]
        for level, hits, system in text_steps:
            if not hits:
                continue
            if step_bar is not None:
                step_bar.set_postfix(stage=f"RAG/{level}", refresh=True)
            ctx = "\n".join(h["node"].get("text", "") for h in hits)
            out["text_levels"][level] = self.mllm.chat_text(
                system,
                f"Query: {query}\n\n{ctx}",
            )
            if step_bar is not None:
                step_bar.update(1)

        clips = self._prioritize_clips(retrieval, max_clips)
        for ci, clip in enumerate(clips):
            has_vid = clip.get("video_uid") in self._available
            if step_bar is not None:
                step_bar.set_postfix(
                    stage=f"clip {ci + 1}/{len(clips)}",
                    vision=has_vid,
                    uid=clip["video_uid"][:8],
                    refresh=True,
                )
            analysis = self.analyze_clip_vlm(clip, rag_ctx)
            out["clip_analyses"].append(analysis)
            mode = "vision" if analysis.get("vision_used") else "text"
            frames = analysis.get("frame_count", 0)
            self._log(
                f"  [{query_idx + 1}/{n_queries}] clip {ci + 1}/{len(clips)} [{mode}] "
                f"uid={clip['video_uid'][:8]}... frames={frames} slot={clip.get('slot_id', '')}"
            )
            if step_bar is not None:
                step_bar.update(1)

        return out

    def run_profile(
        self,
        queries: list[str] | None = None,
        max_clips_per_query: int = 2,
    ) -> dict[str, Any]:
        queries = queries or DEFAULT_INTEREST_QUERIES
        stats = self.rag.stats_summary()
        n_queries = len(queries)
        steps_per_query = 3 + max_clips_per_query  # upper bound for inner bar

        self._log(
            f"Start: {n_queries} queries x {max_clips_per_query} clips/query "
            f"({self.registry['clips_with_local_video']}/{self.registry['total_clips']} clips with local video)"
        )

        query_results: list[dict[str, Any]] = []
        query_iter: Any = queries
        if self.show_progress:
            query_iter = tqdm(
                queries,
                desc="VLM profile",
                unit="query",
                dynamic_ncols=True,
            )

        for qi, q in enumerate(query_iter):
            short_q = q if len(q) <= 40 else q[:37] + "..."
            if self.show_progress and isinstance(query_iter, tqdm):
                query_iter.set_postfix_str(short_q)

            step_bar = None
            if self.show_progress:
                step_bar = tqdm(
                    total=steps_per_query,
                    desc=f"  steps [{qi + 1}/{n_queries}]",
                    unit="step",
                    leave=False,
                    dynamic_ncols=True,
                )

            query_results.append(
                self.analyze_query(
                    q,
                    max_clips_per_query,
                    query_idx=qi,
                    n_queries=n_queries,
                    step_bar=step_bar,
                )
            )
            if step_bar is not None:
                step_bar.close()

        if self.show_progress:
            synth_bar = tqdm(total=1, desc="  synthesis", unit="step", leave=False)
        else:
            synth_bar = None

        synthesis_input = {
            "registry_summary": {
                "total_clips": self.registry["total_clips"],
                "clips_with_local_video": self.registry["clips_with_local_video"],
                "coverage_ratio": self.registry["coverage_ratio"],
            },
            "stats": stats,
            "query_results": query_results,
        }
        try:
            profile = self.mllm.chat_json(
                SYNTHESIS_VLM_SYSTEM,
                json.dumps(synthesis_input, ensure_ascii=False)[:14000],
                temperature=0.2,
            )
        except json.JSONDecodeError:
            text = self.mllm.chat_text(
                SYNTHESIS_VLM_SYSTEM,
                json.dumps(synthesis_input, ensure_ascii=False)[:14000],
            )
            profile = {"summary": text}

        if synth_bar is not None:
            synth_bar.update(1)
            synth_bar.close()

        vision_calls = sum(
            1
            for qr in query_results
            for ca in qr["clip_analyses"]
            if ca.get("vision_used")
        )
        return {
            "registry_summary": synthesis_input["registry_summary"],
            "stats": stats,
            "queries": queries,
            "query_results": query_results,
            "behavior_profile": profile,
            "vision_analyses_count": vision_calls,
        }


def build_registry_and_save(lifelog_path: Path) -> dict[str, Any]:
    registry = build_video_registry(lifelog_path)
    save_registry(registry)
    return registry


def save_profile(report: dict[str, Any], path: Path | None = None) -> Path:
    path = path or VLM_PROFILE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path
