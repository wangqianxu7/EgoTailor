"""Interest and preference mining via hierarchical RAG + vLLM multimodal analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis.config import DEFAULT_INTEREST_QUERIES, FRAMES_PER_CLIP, REPORT_PATH
from analysis.lifelog_index import build_hierarchical_index, load_index, load_lifelog, save_index
from analysis.mllm_client import VLLMClient
from analysis.rag_retriever import HierarchicalRAG, RetrievalResult
from analysis.video_loader import load_clip_visuals

CLIP_SYSTEM = (
    "You are an egocentric lifelog analyst. Given clip metadata and optional video frames, "
    "identify concrete user interests, preferences, habits, and behavioral cues. "
    "Be specific and ground claims in evidence. Output concise bullet points."
)

HOUR_SYSTEM = (
    "You analyze hour-level egocentric activity blocks. Summarize behavioral patterns, "
    "transitions, and emerging preferences within this hour window."
)

DAY_SYSTEM = (
    "You analyze one day of egocentric lifelog. Identify daily routines, dominant interests, "
    "and notable preference signals comparing plan vs observed activities."
)

PERIOD_SYSTEM = (
    "You analyze multi-day egocentric lifelog patterns. Identify long-term habits, recurring "
    "interests, lifestyle preferences, and weekday/weekend differences."
)

SYNTHESIS_SYSTEM = (
    "You synthesize hierarchical lifelog analyses into a structured user interest & preference profile. "
    "Return valid JSON with keys: "
    "interests (list of {topic, evidence, confidence}), "
    "habits (list of {pattern, frequency_hint, time_context}), "
    "preferences (list of {category, preference, supporting_clips}), "
    "summary (string)."
)


class InterestAnalyzer:
    def __init__(
        self,
        rag: HierarchicalRAG,
        mllm: VLLMClient | None = None,
        use_vision: bool = True,
        max_clips_per_query: int = 3,
    ):
        self.rag = rag
        self.mllm = mllm
        self.use_vision = use_vision
        self.max_clips_per_query = max_clips_per_query

    def analyze_clip(
        self,
        clip_node: dict[str, Any],
        frames_b64: list[str] | None = None,
    ) -> dict[str, Any]:
        meta = (
            f"Time: {clip_node.get('start_timestamp')} - {clip_node.get('end_timestamp')}\n"
            f"Scene: {clip_node.get('main_scene')}\n"
            f"Slot: {clip_node.get('slot_id')}\n"
            f"Scenarios: {', '.join(clip_node.get('video_scenarios', []))}\n"
            f"Plan: {clip_node.get('plan_chunk')}\n"
            f"Caption: {clip_node.get('consolidated_summary')}\n"
            f"video_uid: {clip_node.get('video_uid')}\n"
        )
        if self.mllm is None:
            return {
                "level": "clip",
                "video_uid": clip_node.get("video_uid"),
                "analysis": f"[text-only fallback] {clip_node.get('consolidated_summary')}",
                "vision_used": False,
            }

        user_prompt = (
            "Analyze this egocentric clip for user interests and preferences.\n\n" + meta
        )
        vision_used = False
        if self.use_vision and frames_b64:
            analysis = self.mllm.chat_vision(CLIP_SYSTEM, user_prompt, frames_b64)
            vision_used = True
        else:
            analysis = self.mllm.chat_text(CLIP_SYSTEM, user_prompt)

        return {
            "level": "clip",
            "video_uid": clip_node.get("video_uid"),
            "node_id": clip_node.get("node_id"),
            "analysis": analysis,
            "vision_used": vision_used,
        }

    def analyze_retrieval(self, retrieval: RetrievalResult) -> dict[str, Any]:
        out: dict[str, Any] = {"query": retrieval.query, "levels": {}}

        if self.mllm:
            if retrieval.period_hits:
                ctx = "\n".join(h["node"].get("text", "") for h in retrieval.period_hits)
                out["levels"]["period"] = self.mllm.chat_text(
                    PERIOD_SYSTEM, f"Query: {retrieval.query}\n\nContext:\n{ctx}"
                )
            if retrieval.day_hits:
                ctx = "\n".join(h["node"].get("text", "") for h in retrieval.day_hits)
                out["levels"]["day"] = self.mllm.chat_text(
                    DAY_SYSTEM, f"Query: {retrieval.query}\n\nContext:\n{ctx}"
                )
            if retrieval.hour_hits:
                ctx = "\n".join(h["node"].get("text", "") for h in retrieval.hour_hits)
                out["levels"]["hour"] = self.mllm.chat_text(
                    HOUR_SYSTEM, f"Query: {retrieval.query}\n\nContext:\n{ctx}"
                )

        clip_analyses = []
        for clip in retrieval.all_clip_nodes()[: self.max_clips_per_query]:
            visuals = load_clip_visuals(
                clip["video_uid"],
                num_frames=FRAMES_PER_CLIP,
                duration_hint_sec=clip.get("duration"),
            )
            frames = visuals["frames_b64"] if visuals["available"] else None
            clip_analyses.append(self.analyze_clip(clip, frames_b64=frames))
        out["levels"]["clip"] = clip_analyses
        out["rag_context"] = retrieval.to_context_text()
        return out

    def synthesize_profile(self, query_results: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "statistical_prior": stats,
            "hierarchical_analyses": query_results,
        }
        if self.mllm is None:
            return {
                "summary": "MLLM unavailable; statistical prior only.",
                "interests": [{"topic": s[0], "evidence": f"count={s[1]}", "confidence": "medium"} for s in stats.get("top_scenarios", [])[:10]],
                "habits": [],
                "preferences": [],
                "mode": "stats_only",
            }
        user = (
            "Synthesize the following hierarchical analyses and statistics into a user profile.\n\n"
            + json.dumps(payload, indent=2, ensure_ascii=False)[:12000]
        )
        try:
            profile = self.mllm.chat_json(SYNTHESIS_SYSTEM, user)
            profile["mode"] = "mllm"
            return profile
        except json.JSONDecodeError:
            text = self.mllm.chat_text(SYNTHESIS_SYSTEM, user)
            return {"summary": text, "mode": "mllm_text"}

    def mine_interests(
        self,
        queries: list[str] | None = None,
        auto: bool = False,
    ) -> dict[str, Any]:
        queries = queries or (DEFAULT_INTEREST_QUERIES if auto else ["user interests preferences habits"])
        stats = self.rag.stats_summary()
        query_results = []
        for q in queries:
            retrieval = self.rag.retrieve(q)
            query_results.append(self.analyze_retrieval(retrieval))
        profile = self.synthesize_profile(query_results, stats)
        return {
            "stats": stats,
            "queries": queries,
            "query_analyses": query_results,
            "user_profile": profile,
        }


def build_rag_from_lifelog(lifelog_path: Path, index_path: Path) -> HierarchicalRAG:
    lifelog = load_lifelog(lifelog_path)
    index = build_hierarchical_index(lifelog)
    save_index(index, index_path)
    return HierarchicalRAG(index)


def load_rag(index_path: Path) -> HierarchicalRAG:
    return HierarchicalRAG(load_index(index_path))
