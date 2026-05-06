"""Tests for Day 3 script generation; 覆盖 brief 抽取、生成、校验与批量落盘。"""
from __future__ import annotations

from pathlib import Path

from poc.src.data.event_schema import Script
from poc.src.data.script_validator import validate_script
from poc.src.generate_scripts import (
    HeuristicLLMClient,
    LLMClient,
    generate_day_script,
    load_persona_brief,
    run_batch,
)


PERSONA_001 = Path("poc/data/personas/vboss_001.md")
PERSONA_002 = Path("poc/data/personas/vboss_002.md")


def test_load_persona_brief_vboss_001_returns_nonempty_str() -> None:
    """Brief should be non-empty; 画像摘要必须可用且不为空。"""
    brief = load_persona_brief(PERSONA_001)
    assert isinstance(brief, str)
    assert len(brief) > 50
    assert len(brief) <= 1500


def test_load_persona_brief_includes_hexaco_keywords() -> None:
    """Brief includes HEXACO keys; 摘要中包含关键人格字段。"""
    brief = load_persona_brief(PERSONA_001)
    assert "hexaco" in brief.lower()
    assert "conscientiousness" in brief
    assert "emotionality" in brief


def test_heuristic_llm_returns_event_dicts_for_8h() -> None:
    """Heuristic output should cover 8h; 事件总时长约 8h 且数量充足。"""
    llm = HeuristicLLMClient()
    brief = load_persona_brief(PERSONA_001)
    events = llm.generate_script(brief, day_index=0, target_hours=8.0, seed=7)
    assert events
    assert isinstance(events[0], dict)
    total_duration = sum(float(e["duration_s"]) for e in events)
    assert abs(total_duration - 8.0 * 3600.0) < 1.0
    assert len(events) >= 30


def test_generate_day_script_passes_validator() -> None:
    """Generated script must pass validator; 生成结果必须通过契约校验。"""
    script = generate_day_script(
        persona_path=PERSONA_001,
        day_index=0,
        llm=HeuristicLLMClient(),
        target_hours=8.0,
        seed=42,
    )
    report = validate_script(script)
    assert report.passed, f"{[(i.code, i.message) for i in report.issues]}"


def test_generate_day_script_state_diversity() -> None:
    """Generated states should be diverse; 至少包含 3 类状态。"""
    script = generate_day_script(
        persona_path=PERSONA_002,
        day_index=0,
        llm=HeuristicLLMClient(),
        target_hours=8.0,
        seed=11,
    )
    states = {event.state.value for event in script.events}
    assert len(states) >= 3


class StubLLM:
    """Stub LLM for batch write test; 测试 run_batch 的注入客户端。"""

    def generate_script(
        self,
        persona_brief: str,
        day_index: int,
        target_hours: float,
        seed: int,
    ) -> list[dict]:
        return HeuristicLLMClient().generate_script(
            persona_brief=persona_brief,
            day_index=day_index,
            target_hours=target_hours,
            seed=seed + 100,
        )


def test_run_batch_writes_n_files(tmp_path: Path) -> None:
    """Batch should write one file per persona; 批量输出文件数应与画像数一致。"""
    llm: LLMClient = StubLLM()
    out_paths = run_batch(
        persona_paths=[PERSONA_001, PERSONA_002],
        day_index=0,
        llm=llm,
        out_dir=tmp_path,
        seed=1,
    )
    assert len(out_paths) == 2
    for path in out_paths:
        assert path.exists()
        loaded = Script.from_json(path.read_text(encoding="utf-8"))
        assert loaded.persona_id in {"vboss_001", "vboss_002"}
