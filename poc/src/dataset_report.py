"""Day 3 Lane D — 数据集质量聚合报告（v0.1 / 2026-05-06）。

输入：
- synthetic 剧本目录（``poc/data/synthetic/script_*.json``）
- 画像目录（``poc/data/personas/vboss_*.md``）

输出：
- ``DatasetReport`` dataclass
- markdown 报告文本（默认写到 ``poc/data/synthetic/report.md``）

报告内容：
1. 总览：N 画像 / N 剧本 / 总时长 / 平均时长
2. 状态分布：每画像 8 类 State 帧数 + 比例
3. 活动分布：每画像 8 类 Activity 帧数
4. HEXACO 6 维两两欧式距离矩阵（与 data-generation-todo.md §4 标准 1 对齐）
5. 校验问题聚合（错误码 → 命中数）

零外部依赖 — 不用 numpy/pandas，纯标准库。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data.event_schema import Script
from .data.script_validator import (
    ScriptValidationReport,
    validate_script,
)


# ---------- 画像 frontmatter 提取（轻量，不依赖 PyYAML） ----------


HEXACO_KEYS = (
    "honesty_humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness",
)


def _extract_frontmatter_block(text: str) -> str:
    """从 markdown 提取 YAML frontmatter block 文本（不解析）。"""
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    return m.group(1) if m else ""


def _extract_hexaco(persona_path: Path) -> dict[str, float] | None:
    """轻量 parse hexaco 6 维（避免引入 pyyaml 依赖）。"""
    if not persona_path.exists():
        return None
    text = persona_path.read_text(encoding="utf-8")
    fm = _extract_frontmatter_block(text)
    if not fm:
        return None

    in_hexaco = False
    out: dict[str, float] = {}
    for line in fm.splitlines():
        stripped = line.strip()
        if stripped == "hexaco:":
            in_hexaco = True
            continue
        if in_hexaco:
            if not line.startswith(" ") and stripped:
                in_hexaco = False
                continue
            m = re.match(r"^\s+(\w+):\s*([0-9.]+)", line)
            if m:
                key, val = m.group(1), m.group(2)
                if key in HEXACO_KEYS:
                    try:
                        out[key] = float(val)
                    except ValueError:
                        pass
    return out if len(out) == 6 else None


def _hexaco_l2(a: dict[str, float], b: dict[str, float]) -> float:
    """6 维欧式距离（HEXACO 全 0-1 正值，余弦距离不合适）。"""
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in HEXACO_KEYS))


# ---------- 报告数据结构 ----------


@dataclass(frozen=True)
class PersonaScriptStats:
    persona_id: str
    n_scripts: int
    n_events: int
    total_duration_h: float
    state_counts: dict[str, int]
    activity_counts: dict[str, int]
    hexaco: dict[str, float] | None     # 来自画像 frontmatter，可缺
    validation_issues: tuple[str, ...]  # 仅 error code（warning 不进 summary）


@dataclass(frozen=True)
class DatasetReport:
    n_personas: int
    n_scripts: int
    total_duration_h: float
    per_persona: tuple[PersonaScriptStats, ...]
    hexaco_pairwise_l2: dict[str, dict[str, float]]   # {pid_a: {pid_b: l2}}
    issue_code_counts: dict[str, int]
    state_classes: tuple[str, ...] = (
        "low_energy",
        "focused",
        "distracted",
        "fatigued",
        "emotional_positive",
        "emotional_negative",
        "neutral",
        "transitioning",
    )
    activity_classes: tuple[str, ...] = (
        "work_focus",
        "work_meeting",
        "break_short",
        "break_long",
        "commute",
        "leisure",
        "sleep",
        "transition",
    )
    issues: tuple[str, ...] = field(default_factory=tuple)


# ---------- 主入口 ----------


def build_report(
    script_paths: list[Path],
    personas_dir: Path,
) -> DatasetReport:
    """聚合多个 Script JSON → DatasetReport。"""
    persona_scripts: dict[str, list[Script]] = {}
    persona_validations: dict[str, list[ScriptValidationReport]] = {}

    for path in script_paths:
        if not path.exists():
            continue
        try:
            s = Script.from_json(path.read_text(encoding="utf-8"))
        except (ValueError, KeyError):
            continue
        persona_scripts.setdefault(s.persona_id, []).append(s)
        persona_validations.setdefault(s.persona_id, []).append(validate_script(s))

    per_persona: list[PersonaScriptStats] = []
    issue_counter: Counter[str] = Counter()
    total_duration_s = 0.0
    n_total_scripts = 0

    for pid in sorted(persona_scripts.keys()):
        scripts = persona_scripts[pid]
        validations = persona_validations[pid]

        n_events = sum(len(s.events) for s in scripts)
        duration_s = sum(ev.duration_s for s in scripts for ev in s.events)
        state_counts: Counter[str] = Counter()
        activity_counts: Counter[str] = Counter()
        for s in scripts:
            for ev in s.events:
                state_counts[ev.state.value] += 1
                activity_counts[ev.activity.value] += 1

        error_codes: list[str] = []
        for v in validations:
            for issue in v.issues:
                if issue.severity == "error":
                    error_codes.append(issue.code)
                    issue_counter[issue.code] += 1

        hexaco = _extract_hexaco(personas_dir / f"{pid}.md")

        per_persona.append(
            PersonaScriptStats(
                persona_id=pid,
                n_scripts=len(scripts),
                n_events=n_events,
                total_duration_h=duration_s / 3600.0,
                state_counts=dict(state_counts),
                activity_counts=dict(activity_counts),
                hexaco=hexaco,
                validation_issues=tuple(error_codes),
            )
        )
        total_duration_s += duration_s
        n_total_scripts += len(scripts)

    # HEXACO 两两距离矩阵
    pairwise: dict[str, dict[str, float]] = {}
    for i, a in enumerate(per_persona):
        if a.hexaco is None:
            continue
        pairwise.setdefault(a.persona_id, {})
        for b in per_persona:
            if b.hexaco is None or a.persona_id == b.persona_id:
                continue
            pairwise[a.persona_id][b.persona_id] = _hexaco_l2(a.hexaco, b.hexaco)

    return DatasetReport(
        n_personas=len(per_persona),
        n_scripts=n_total_scripts,
        total_duration_h=total_duration_s / 3600.0,
        per_persona=tuple(per_persona),
        hexaco_pairwise_l2=pairwise,
        issue_code_counts=dict(issue_counter),
    )


def render_markdown(report: DatasetReport) -> str:
    """渲染 DatasetReport 为人类可读的 markdown。"""
    lines: list[str] = []
    lines.append("# 数据集质量报告")
    lines.append("")
    lines.append(f"- 画像数: **{report.n_personas}**")
    lines.append(f"- 剧本数: **{report.n_scripts}**")
    lines.append(f"- 总时长: **{report.total_duration_h:.2f}h**")
    lines.append(
        f"- 平均时长/画像: **"
        f"{(report.total_duration_h / report.n_personas):.2f}h**"
        if report.n_personas > 0
        else "- 平均时长/画像: N/A"
    )
    lines.append("")

    # 总览表
    lines.append("## 1. 画像总览")
    lines.append("")
    lines.append("| persona_id | scripts | events | duration_h | errors |")
    lines.append("|------------|---------|--------|-----------|--------|")
    for p in report.per_persona:
        lines.append(
            f"| {p.persona_id} | {p.n_scripts} | {p.n_events} | "
            f"{p.total_duration_h:.2f} | {len(p.validation_issues)} |"
        )
    lines.append("")

    # 状态分布
    lines.append("## 2. 状态分布（事件计数）")
    lines.append("")
    header = "| persona_id | " + " | ".join(report.state_classes) + " |"
    sep = "|" + "---|" * (len(report.state_classes) + 1)
    lines.append(header)
    lines.append(sep)
    for p in report.per_persona:
        cells = " | ".join(str(p.state_counts.get(s, 0)) for s in report.state_classes)
        lines.append(f"| {p.persona_id} | {cells} |")
    lines.append("")

    # Activity 分布
    lines.append("## 3. 活动分布（事件计数）")
    lines.append("")
    header = "| persona_id | " + " | ".join(report.activity_classes) + " |"
    sep = "|" + "---|" * (len(report.activity_classes) + 1)
    lines.append(header)
    lines.append(sep)
    for p in report.per_persona:
        cells = " | ".join(
            str(p.activity_counts.get(a, 0)) for a in report.activity_classes
        )
        lines.append(f"| {p.persona_id} | {cells} |")
    lines.append("")

    # HEXACO 距离矩阵
    if report.hexaco_pairwise_l2:
        ids = sorted(report.hexaco_pairwise_l2.keys())
        lines.append("## 4. HEXACO 6 维欧式距离矩阵")
        lines.append("")
        lines.append("> 标准（data-generation-todo.md §4 #1）：两两 ≥ 0.4。")
        lines.append("")
        header = "| | " + " | ".join(ids) + " |"
        sep = "|" + "---|" * (len(ids) + 1)
        lines.append(header)
        lines.append(sep)
        any_below = False
        for a in ids:
            cells: list[str] = []
            for b in ids:
                if a == b:
                    cells.append("—")
                else:
                    d = report.hexaco_pairwise_l2.get(a, {}).get(b)
                    if d is None:
                        cells.append("?")
                    else:
                        marker = "" if d >= 0.4 else " ⚠"
                        if d < 0.4:
                            any_below = True
                        cells.append(f"{d:.3f}{marker}")
            lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
        lines.append("")
        if any_below:
            lines.append("> ⚠ 标记的对距离 < 0.4，画像差异化不足。")
            lines.append("")

    # 问题汇总
    lines.append("## 5. 校验问题汇总")
    lines.append("")
    if not report.issue_code_counts:
        lines.append("✅ 无错误。")
    else:
        lines.append("| code | count |")
        lines.append("|------|-------|")
        for code, n in sorted(
            report.issue_code_counts.items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"| {code} | {n} |")
    lines.append("")

    return "\n".join(lines) + "\n"


def write_markdown_report(report: DatasetReport, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(report), encoding="utf-8")
    return out_path


__all__ = [
    "PersonaScriptStats",
    "DatasetReport",
    "build_report",
    "render_markdown",
    "write_markdown_report",
]
