"""Day 3 Lane D 测试 — dataset_report 聚合（v0.1）。"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from poc.src.data.event_schema import (
    Activity,
    MultimodalHint,
    Script,
    ScriptEvent,
    State,
)
from poc.src.dataset_report import (
    build_report,
    render_markdown,
    write_markdown_report,
)


def _make_event(pid: str, idx: int, ts_min: int, state: State) -> ScriptEvent:
    return ScriptEvent(
        event_id=f"{pid}_d0_e{idx:03d}",
        ts=f"2026-05-06T09:{ts_min:02d}:00",
        persona_id=pid,
        day_index=0,
        duration_s=600.0,
        activity=Activity.WORK_FOCUS,
        state=state,
        multimodal_hint=MultimodalHint(),
    )


def _write_script(path: Path, pid: str, states: list[State]) -> None:
    events = [
        _make_event(pid, i, i * 10, st) for i, st in enumerate(states)
    ]
    s = Script(
        persona_id=pid,
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=tuple(events),
    )
    path.write_text(s.to_json(), encoding="utf-8")


def _write_persona(path: Path, pid: str, hexaco: dict[str, float]) -> None:
    """写一个最小画像 markdown（仅 frontmatter，足够 dataset_report 提取 hexaco）。"""
    fm_lines = [
        "---",
        f"id: {pid}",
        f"name: {pid}",
        "type_label: 测试画像",
        "hexaco:",
    ]
    for k in (
        "honesty_humility",
        "emotionality",
        "extraversion",
        "agreeableness",
        "conscientiousness",
        "openness",
    ):
        fm_lines.append(f"  {k}: {hexaco[k]}")
    fm_lines.append("attachment:")
    fm_lines.append("  type: secure")
    fm_lines.append("---")
    fm_lines.append("")
    fm_lines.append("# Body")
    path.write_text("\n".join(fm_lines), encoding="utf-8")


# ---------- 基础聚合 ----------


def test_build_report_two_personas_count() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        scripts_dir = td_path / "synthetic"
        scripts_dir.mkdir()
        personas_dir = td_path / "personas"
        personas_dir.mkdir()

        _write_script(
            scripts_dir / "s1.json",
            "vboss_a",
            [State.FOCUSED, State.NEUTRAL, State.LOW_ENERGY],
        )
        _write_script(
            scripts_dir / "s2.json",
            "vboss_b",
            [State.DISTRACTED, State.FATIGUED, State.EMOTIONAL_POSITIVE],
        )

        report = build_report(
            list(scripts_dir.glob("*.json")), personas_dir
        )
        assert report.n_personas == 2
        assert report.n_scripts == 2
        assert report.total_duration_h == 1.0  # 6 events × 600s = 3600s = 1h


def test_build_report_state_distribution_per_persona() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        scripts_dir = td_path / "synthetic"
        scripts_dir.mkdir()
        personas_dir = td_path / "personas"
        personas_dir.mkdir()

        _write_script(
            scripts_dir / "s1.json",
            "vboss_a",
            [State.FOCUSED, State.FOCUSED, State.NEUTRAL],
        )

        report = build_report(
            list(scripts_dir.glob("*.json")), personas_dir
        )
        assert len(report.per_persona) == 1
        stats = report.per_persona[0]
        assert stats.persona_id == "vboss_a"
        assert stats.state_counts["focused"] == 2
        assert stats.state_counts["neutral"] == 1


# ---------- HEXACO 矩阵 ----------


def test_hexaco_extracted_from_persona_md() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        scripts_dir = td_path / "synthetic"
        scripts_dir.mkdir()
        personas_dir = td_path / "personas"
        personas_dir.mkdir()

        _write_script(
            scripts_dir / "s1.json",
            "vboss_a",
            [State.FOCUSED, State.NEUTRAL, State.LOW_ENERGY],
        )
        _write_persona(
            personas_dir / "vboss_a.md",
            "vboss_a",
            {
                "honesty_humility": 0.7,
                "emotionality": 0.4,
                "extraversion": 0.3,
                "agreeableness": 0.6,
                "conscientiousness": 0.85,
                "openness": 0.7,
            },
        )

        report = build_report(
            list(scripts_dir.glob("*.json")), personas_dir
        )
        stats = report.per_persona[0]
        assert stats.hexaco is not None
        assert abs(stats.hexaco["conscientiousness"] - 0.85) < 1e-6


def test_hexaco_pairwise_distance_computed() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        scripts_dir = td_path / "synthetic"
        scripts_dir.mkdir()
        personas_dir = td_path / "personas"
        personas_dir.mkdir()

        _write_script(scripts_dir / "a.json", "vboss_a", [State.FOCUSED] * 3)
        _write_script(scripts_dir / "b.json", "vboss_b", [State.NEUTRAL] * 3)
        _write_persona(
            personas_dir / "vboss_a.md",
            "vboss_a",
            {
                "honesty_humility": 0.7,
                "emotionality": 0.4,
                "extraversion": 0.3,
                "agreeableness": 0.6,
                "conscientiousness": 0.85,
                "openness": 0.7,
            },
        )
        _write_persona(
            personas_dir / "vboss_b.md",
            "vboss_b",
            {
                "honesty_humility": 0.5,
                "emotionality": 0.7,
                "extraversion": 0.8,
                "agreeableness": 0.5,
                "conscientiousness": 0.4,
                "openness": 0.6,
            },
        )

        report = build_report(
            list(scripts_dir.glob("*.json")), personas_dir
        )
        d_ab = report.hexaco_pairwise_l2["vboss_a"]["vboss_b"]
        d_ba = report.hexaco_pairwise_l2["vboss_b"]["vboss_a"]
        assert abs(d_ab - d_ba) < 1e-9
        # 应该 > 0.4（两画像差异显著）
        assert d_ab > 0.4


# ---------- markdown 渲染 ----------


def test_render_markdown_contains_section_headers() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        scripts_dir = td_path / "synthetic"
        scripts_dir.mkdir()
        personas_dir = td_path / "personas"
        personas_dir.mkdir()

        _write_script(
            scripts_dir / "s1.json", "vboss_a", [State.FOCUSED, State.NEUTRAL]
        )

        report = build_report(
            list(scripts_dir.glob("*.json")), personas_dir
        )
        md = render_markdown(report)
        assert "# 数据集质量报告" in md
        assert "## 1. 画像总览" in md
        assert "## 2. 状态分布" in md
        assert "## 3. 活动分布" in md


def test_write_markdown_report_creates_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        scripts_dir = td_path / "synthetic"
        scripts_dir.mkdir()
        personas_dir = td_path / "personas"
        personas_dir.mkdir()

        _write_script(scripts_dir / "s1.json", "vboss_a", [State.FOCUSED] * 3)

        report = build_report(
            list(scripts_dir.glob("*.json")), personas_dir
        )
        out = td_path / "report.md"
        result = write_markdown_report(report, out)
        assert result == out
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "vboss_a" in text


def test_build_report_records_parse_failed() -> None:
    """codex F1-03 review fix: 损坏 JSON 不能 silent skip，必须计入 SCRIPT_PARSE_FAILED。"""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        scripts_dir = td_path / "synthetic"
        scripts_dir.mkdir()
        personas_dir = td_path / "personas"
        personas_dir.mkdir()

        _write_script(scripts_dir / "good.json", "vboss_a", [State.FOCUSED, State.NEUTRAL])
        (scripts_dir / "bad.json").write_text(
            '{"persona_id": "vboss_b", "events": [', encoding="utf-8"
        )
        report = build_report(
            sorted(scripts_dir.glob("*.json")), personas_dir
        )
        assert report.n_scripts == 1
        assert report.issue_code_counts.get("SCRIPT_PARSE_FAILED") == 1
        assert any("SCRIPT_PARSE_FAILED" in i for i in report.issues)


def test_build_report_records_missing_file() -> None:
    """codex F1-03 fix: 缺失文件路径要计入 SCRIPT_FILE_MISSING。"""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ghost = td_path / "ghost.json"  # 不创建
        report = build_report([ghost], td_path)
        assert report.issue_code_counts.get("SCRIPT_FILE_MISSING") == 1


def test_empty_script_list_yields_zero_personas() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        report = build_report([], td_path)
        assert report.n_personas == 0
        assert report.n_scripts == 0
        assert report.total_duration_h == 0.0
