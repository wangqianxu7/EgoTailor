"""vLLM OpenAI-compatible client for multimodal analysis."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from analysis.config import (
    VLLM_API_BASE,
    VLLM_API_KEY,
    VLLM_MAX_TOKENS,
    VLLM_MODEL,
    VLLM_TIMEOUT_SEC,
)


class VLLMClient:
    """Talk to a vLLM server exposing OpenAI-compatible chat/completions."""

    def __init__(
        self,
        base_url: str = VLLM_API_BASE,
        api_key: str = VLLM_API_KEY,
        model: str = VLLM_MODEL,
    ):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=VLLM_TIMEOUT_SEC)

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def chat_text(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=VLLM_MAX_TOKENS,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def chat_vision(
        self,
        system: str,
        user_text: str,
        frames_b64: list[str],
        temperature: float = 0.2,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for b64 in frames_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            max_tokens=VLLM_MAX_TOKENS,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def chat_vision_json(
        self,
        system: str,
        user_text: str,
        frames_b64: list[str],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        raw = self.chat_vision(system, user_text, frames_b64, temperature=temperature)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    def chat_json(self, system: str, user: str, **kwargs: Any) -> dict[str, Any]:
        raw = self.chat_text(system, user, **kwargs)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
