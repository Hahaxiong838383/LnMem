"""OpenAI-compatible Gemini proxy client for parser_v0_llm LLMClient."""
from __future__ import annotations

import os
from typing import Any

import requests


class GeminiProxyClient:
    """Call Gemini proxy via OpenAI Chat Completions. 调 23.226.135.149:4000。"""

    def __init__(
        self,
        endpoint: str = "http://23.226.135.149:4000",
        model: str = "gemini-3-flash-preview",
        bearer_token: str | None = None,
        timeout_s: float = 15.0,
        max_tokens: int = 4096,
        reasoning_effort: str = "minimal",
    ) -> None:
        token = bearer_token or os.getenv("GEMINI_PROXY_TOKEN")
        if not token:
            raise ValueError("GeminiProxyClient requires bearer token or GEMINI_PROXY_TOKEN")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.bearer_token = token
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def __call__(self, prompt: str) -> str:
        """Return assistant text content. 发 OpenAI 兼容 chat/completions 请求。"""
        response = requests.post(
            f"{self.endpoint}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort,
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        return self._extract_content(payload)

    @staticmethod
    def _extract_content(payload: dict[str, Any]) -> str:
        """Extract text content from OpenAI-style response. 兼容 str / content parts。"""
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("chat completion response missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("chat completion response missing message")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            if text_parts:
                return "".join(text_parts)
        raise ValueError("chat completion response missing textual content")
