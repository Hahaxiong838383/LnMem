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
    timeout_s = float(os.getenv("LNMEM_PARSER_TIMEOUT_SECONDS", "15"))

    def _call(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
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
