"""LLM client factory for demo parser; env-driven HTTP callable."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable


LLMClient = Callable[[str], str]


def make_llm_client() -> LLMClient:
    """Create a Gemini-proxy compatible callable from environment variables."""
    token = os.getenv("GEMINI_PROXY_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing GEMINI_PROXY_TOKEN. Set it before launching the demo."
        )

    endpoint = os.getenv("GEMINI_PROXY_ENDPOINT", "http://23.226.135.149:4000/v1/chat/completions")
    model = os.getenv("LNMEM_PARSER_MODEL", "gemini-3-flash-preview")
    # ↑ 15s 默认 timeout 不够留 thinking 余地，但开了 reasoning_effort=minimal 后 ~2s 内必返
    timeout_s = float(os.getenv("LNMEM_PARSER_TIMEOUT_SECONDS", "20"))
    # max_tokens + reasoning_effort 是修复 bug 的关键（5/21 14:12 demo timeout 根因）：
    #   gemini-3-flash-preview 默认强制 thinking → completion_tokens=2 / reasoning_tokens=44
    #   max_tokens 太小 → thinking 吃光 → 无 content 返回 → KeyError 'message'
    #   reasoning_effort=minimal → thinking 跳过 → 2s 内返回 JSON
    max_tokens = int(os.getenv("LNMEM_PARSER_MAX_TOKENS", "4096"))
    reasoning_effort = os.getenv("LNMEM_PARSER_REASONING_EFFORT", "minimal")

    def _call(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("Gemini proxy returned no choices")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Gemini proxy returned non-string message content")
        return content

    return _call
