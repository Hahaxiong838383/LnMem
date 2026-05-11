"""LLM-based importance judge that outputs scalar gates in [0,1]."""
from __future__ import annotations

from typing import Any

from poc.src.memory.memory_lnn import DEFAULT_TAU_SECONDS

_FALLBACK_GATE = 0.5
_STAGE_NAMES = ("m2_short_term", "m3_working", "m4_long_term", "m5_lifelong")


class ImportanceJudge:
    """Wrap LLM calls for gate scoring while keeping memory dynamics model-free."""

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    def _parse_gate(self, content: str) -> float:
        """Parse and clamp one scalar gate value from model text output."""
        first_token = content.strip().split()[0]
        return float(min(1.0, max(0.0, float(first_token))))

    def _invoke(self, prompt: str) -> str:
        """Call client with common completion/chat surfaces and return text."""
        if hasattr(self.llm_client, "complete"):
            result = self.llm_client.complete(prompt=prompt)
            return str(getattr(result, "text", result))
        if hasattr(self.llm_client, "chat"):
            result = self.llm_client.chat(messages=[{"role": "user", "content": prompt}])
            return str(getattr(result, "text", result))
        raise RuntimeError("llm_client must expose complete(...) or chat(...)")

    def judge_gate(
        self,
        m_summary: str,
        next_stage_name: str,
        persona_summary: str,
        tau: float,
    ) -> float:
        """Return one gate scalar g in [0,1] for stage-to-stage diffusion."""
        prompt = (
            "You are a memory-importance judge. Output only one float in [0,1].\\n"
            f"next_stage={next_stage_name}, tau_seconds={tau}\\n"
            f"persona={persona_summary}\\n"
            f"memory_summary={m_summary}"
        )
        try:
            return self._parse_gate(self._invoke(prompt))
        except Exception:
            return _FALLBACK_GATE

    def judge_all_gates(self, ms_summaries: list[str], persona_summary: str) -> list[float]:
        """Return four gates in order: g12, g23, g34, g45."""
        if len(ms_summaries) < 4:
            raise ValueError("ms_summaries must contain at least four stage summaries")

        gates: list[float] = []
        for idx in range(4):
            gates.append(
                self.judge_gate(
                    m_summary=ms_summaries[idx],
                    next_stage_name=_STAGE_NAMES[idx],
                    persona_summary=persona_summary,
                    tau=DEFAULT_TAU_SECONDS[idx + 1],
                )
            )
        return gates


__all__ = ["ImportanceJudge"]
