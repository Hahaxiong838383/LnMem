"""Tests for Day 3 script generation; 覆盖 brief 抽取、生成、校验与批量落盘。"""
from __future__ import annotations

from statistics import pvariance
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


def _make_persona_brief(
    persona_id: str,
    hexaco: dict[str, float],
    profession: str = "产品负责人",
) -> str:
    return f"""[frontmatter]
id: {persona_id}
name: Test Persona
hexaco:
  honesty_humility: {hexaco["honesty_humility"]}
  emotionality: {hexaco["emotionality"]}
  extraversion: {hexaco["extraversion"]}
  agreeableness: {hexaco["agreeableness"]}
  conscientiousness: {hexaco["conscientiousness"]}
  openness: {hexaco["openness"]}
attachment:
  anxiety: 0.5
demographics:
  profession: {profession}
  commute_minutes: 10
multimodal_signal_density: medium
[layer4_if_then]
placeholder
[layer6_multimodal]
placeholder
"""


def _aggregate_metrics(
    llm: HeuristicLLMClient,
    brief: str,
    seeds: tuple[int, ...],
) -> dict[str, float]:
    counts = {
        "work_focus": 0.0,
        "work_meeting": 0.0,
        "break_long": 0.0,
        "leisure": 0.0,
        "transition": 0.0,
        "emotional_negative": 0.0,
        "switch_total": 0.0,
    }
    utterances: list[float] = []
    for day_idx, seed in enumerate(seeds):
        events = llm.generate_script(brief, day_index=day_idx, target_hours=8.0, seed=seed)
        for event in events:
            counts.setdefault(event["activity"], 0.0)
            counts[event["activity"]] += 1.0
            if event["state"] == "emotional_negative":
                counts["emotional_negative"] += 1.0
            counts["switch_total"] += float(event["multimodal_hint"]["app_switch_count_5min"])
            utterances.append(float(event["multimodal_hint"]["utterance_density"]))
    counts["utterance_var"] = pvariance(utterances)
    counts["unique_activities"] = float(
        sum(1 for key in ("work_focus", "work_meeting", "break_long", "leisure", "transition") if counts[key] > 0)
    )
    return counts


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


def test_hexaco_honesty_and_agreeableness_are_applied() -> None:
    """H/A dimensions must shift behavior; H/A 维度应改变活动与状态分布。"""
    llm = HeuristicLLMClient()
    seeds = (3, 5, 7, 11, 13, 17)
    base = {
        "honesty_humility": 0.5,
        "emotionality": 0.55,
        "extraversion": 0.55,
        "agreeableness": 0.5,
        "conscientiousness": 0.6,
        "openness": 0.55,
    }

    low_h = dict(base)
    high_h = dict(base)
    low_h["honesty_humility"] = 0.2
    high_h["honesty_humility"] = 0.85
    low_h_metrics = _aggregate_metrics(llm, _make_persona_brief("low_h", low_h), seeds)
    high_h_metrics = _aggregate_metrics(llm, _make_persona_brief("high_h", high_h), seeds)
    assert low_h_metrics["transition"] > high_h_metrics["transition"]
    assert low_h_metrics["work_meeting"] >= high_h_metrics["work_meeting"]
    assert low_h_metrics["emotional_negative"] > high_h_metrics["emotional_negative"]
    assert low_h_metrics["switch_total"] > high_h_metrics["switch_total"]

    low_a = dict(base)
    high_a = dict(base)
    low_a["agreeableness"] = 0.2
    high_a["agreeableness"] = 0.85
    low_a_metrics = _aggregate_metrics(llm, _make_persona_brief("low_a", low_a), seeds)
    high_a_metrics = _aggregate_metrics(llm, _make_persona_brief("high_a", high_a), seeds)
    assert high_a_metrics["work_meeting"] + high_a_metrics["leisure"] > low_a_metrics["work_meeting"] + low_a_metrics["leisure"]
    assert low_a_metrics["work_focus"] > high_a_metrics["work_focus"]


def test_hexaco_openness_controls_spectrum_and_utterance_variance() -> None:
    """Openness must affect exploration and signal variance; O 维应影响活动谱与语音密度方差。"""
    llm = HeuristicLLMClient()
    seeds = (2, 4, 6, 8, 10, 12)
    base = {
        "honesty_humility": 0.6,
        "emotionality": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.55,
        "conscientiousness": 0.6,
        "openness": 0.5,
    }
    low_o = dict(base)
    high_o = dict(base)
    low_o["openness"] = 0.2
    high_o["openness"] = 0.9
    low_o_metrics = _aggregate_metrics(llm, _make_persona_brief("low_o", low_o, profession="技术经理"), seeds)
    high_o_metrics = _aggregate_metrics(llm, _make_persona_brief("high_o", high_o, profession="技术经理"), seeds)
    assert high_o_metrics["break_long"] + high_o_metrics["leisure"] > low_o_metrics["break_long"] + low_o_metrics["leisure"]
    assert high_o_metrics["utterance_var"] > low_o_metrics["utterance_var"]
    assert high_o_metrics["unique_activities"] >= low_o_metrics["unique_activities"]


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
