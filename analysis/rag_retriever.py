"""Hierarchical RAG retriever over clip / hour / day / period levels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analysis.config import TOP_K
from analysis.embeddings import TextEmbedder


@dataclass
class RetrievalResult:
    query: str
    clip_hits: list[dict[str, Any]] = field(default_factory=list)
    hour_hits: list[dict[str, Any]] = field(default_factory=list)
    day_hits: list[dict[str, Any]] = field(default_factory=list)
    period_hits: list[dict[str, Any]] = field(default_factory=list)

    def all_clip_nodes(self) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for hit in self.clip_hits:
            nid = hit["node"]["node_id"]
            if nid not in seen:
                seen.add(nid)
                out.append(hit["node"])
        return out

    def to_context_text(self) -> str:
        sections = []
        for level, hits, label in [
            ("period", self.period_hits, "Multi-day patterns"),
            ("day", self.day_hits, "Daily summaries"),
            ("hour", self.hour_hits, "Hour blocks"),
            ("clip", self.clip_hits, "Clip-level evidence"),
        ]:
            if not hits:
                continue
            lines = [f"## {label}"]
            for h in hits:
                node = h["node"]
                score = h["score"]
                lines.append(f"- [{level}|score={score:.3f}] {node.get('text', '')[:500]}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)


class HierarchicalRAG:
    def __init__(self, index: dict[str, list[dict[str, Any]]]):
        self.index = index
        self.embedders: dict[str, TextEmbedder] = {}
        self._clip_by_id = {n["node_id"]: n for n in index["clip"]}
        self._build_embedders()

    def _build_embedders(self) -> None:
        for level in ("clip", "hour", "day", "period"):
            nodes = self.index[level]
            texts = [n.get("text", "") for n in nodes]
            emb = TextEmbedder()
            emb.fit(texts)
            self.embedders[level] = emb

    def retrieve(
        self,
        query: str,
        top_k: dict[str, int] | None = None,
        expand_clips_from_higher_levels: bool = True,
    ) -> RetrievalResult:
        k = top_k or TOP_K
        result = RetrievalResult(query=query)

        for level, attr in [
            ("clip", "clip_hits"),
            ("hour", "hour_hits"),
            ("day", "day_hits"),
            ("period", "period_hits"),
        ]:
            hits = self._search_level(level, query, k.get(level, 5))
            setattr(result, attr, hits)

        if expand_clips_from_higher_levels:
            extra_clip_ids: list[str] = []
            for hit in result.hour_hits + result.day_hits + result.period_hits:
                extra_clip_ids.extend(hit["node"].get("clip_ids", []))
            existing = {h["node"]["node_id"] for h in result.clip_hits}
            for cid in extra_clip_ids[: k.get("clip", 8)]:
                if cid in existing:
                    continue
                node = self._clip_by_id.get(cid)
                if node:
                    result.clip_hits.append({"node": node, "score": 0.0, "expanded": True})
                    existing.add(cid)

        return result

    def _search_level(self, level: str, query: str, top_k: int) -> list[dict[str, Any]]:
        nodes = self.index[level]
        emb = self.embedders[level]
        ranked = emb.search(query, top_k=min(top_k, len(nodes)))
        return [{"node": nodes[i], "score": score} for i, score in ranked]

    def get_clip_by_video_uid(self, video_uid: str) -> dict[str, Any] | None:
        for node in self.index["clip"]:
            if node["video_uid"] == video_uid:
                return node
        return None

    def stats_summary(self) -> dict[str, Any]:
        from collections import Counter

        scenario_counter: Counter = Counter()
        scene_counter: Counter = Counter()
        slot_counter: Counter = Counter()
        for node in self.index["clip"]:
            scenario_counter.update(node.get("video_scenarios", []))
            scene_counter[node.get("main_scene", "")] += 1
            slot_counter[node.get("slot_id", "")] += 1
        return {
            "total_clips": len(self.index["clip"]),
            "top_scenarios": scenario_counter.most_common(15),
            "scene_distribution": dict(scene_counter),
            "top_slots": slot_counter.most_common(10),
        }
