#!/usr/bin/env python3
"""schema-guided LLM prompt spike for universal_grammar_parser (M0 W2 P-1).

PoC 资源分配铁律：优先验证效果未知的内容。
万能语法 parser 整个模块的核心 bet 是：**LLM 能否在中文场景下稳定按
schema_v1 (20 槽) 输出可信 JSON**。其他（batch / postprocess / 后端选型）
都是已知工程。本 spike 在写完整 parser_v0_llm 实现之前先做这一验证。

用法：
    GEMINI_PROXY_KEY=xxx .venv-poc-fresh/bin/python3 spikes/parser_prompt_spike.py

输出：每个测试句子 × 每个 LLM → JSON 解析结果，最终统计：
- json_parse_rate: 输出能 JSON.parse 的比例
- p0_recall: P0 4 槽（who/what/when/where）非 null 比例
- avg_slots_filled: 20 槽填充率
- 人眼标注的"明显错"案例

判定标准（spike PASS = 走 parser_v0_llm 完整实现 / spike FAIL = ADR 报告）：
- json_parse_rate ≥ 90%
- p0_recall ≥ 0.6（P0 召回，产品定义 §10.1 不可妥协）
- avg_slots_filled ≥ 8/20（40%）
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts.schema_v1 import SCHEMA_V1, P0_SLOTS, NUM_SLOTS  # noqa: E402


# ---------- 测试集（5 句覆盖不同复杂度）----------

TEST_SENTENCES = [
    {
        "text": "我昨晚 8 点跟王总在朝阳门吃了饭聊融资",
        "user_id": "vboss_self",
        "expect_p0": ["who_actor", "what_action", "when_time", "where_location"],
        "notes": "P0 全填 + listener=王总 + topic=融资 + multi entities",
    },
    {
        "text": "明天提醒我下午 3 点交报告",
        "user_id": "vboss_self",
        "expect_p0": ["who_actor", "what_action", "when_time"],
        "notes": "intent=request + future temporal + 缺 location（合理 null）",
    },
    {
        "text": "我有点累，想休息一下",
        "user_id": "vboss_self",
        "expect_p0": ["who_actor"],
        "notes": "sentiment=负 + intent=plan + minimal entities",
    },
    {
        "text": "刚和我妈打完电话，她说外婆住院了",
        "user_id": "vboss_self",
        "expect_p0": ["who_actor", "what_action", "when_time"],
        "notes": "social_link=妈/外婆 + causal_link 隐式 + 复合事件",
    },
    {
        "text": "把那个 PDF 给我发邮件",
        "user_id": "vboss_self",
        "expect_p0": ["what_action"],
        "notes": "intent=request + modality=text + pronoun（那个 PDF）",
    },
]


# ---------- LLM 后端 ----------

GEMINI_PROXY_URL = "http://23.226.135.149:4000/v1/chat/completions"


def _http_post_json(url: str, headers: dict, body: dict, timeout_s: float = 60.0) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks（thinking 模型 quirk）。"""
    while "<think>" in text and "</think>" in text:
        s = text.index("<think>")
        e = text.index("</think>", s) + len("</think>")
        text = text[:s] + text[e:]
    return text.strip()


def call_gemini(prompt: str, model: str = "gemini-3.1-flash-lite-preview") -> tuple[str, str]:
    """返回 (content, backend_label)。"""
    key = os.environ.get("GEMINI_PROXY_KEY")
    if not key:
        raise RuntimeError("GEMINI_PROXY_KEY env not set")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000,
        "temperature": 0.0,
    }
    resp = _http_post_json(
        GEMINI_PROXY_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        body=body,
        timeout_s=60.0,
    )
    content = resp["choices"][0]["message"]["content"]
    return _strip_thinking(content), f"gemini:{model}"


# ---------- Schema-guided prompt ----------

def build_schema_prompt(text: str, user_id: str) -> str:
    """schema-guided prompt：给 LLM schema_v1 完整描述 + 强制 JSON 输出。"""
    slot_specs = []
    for sid, meta in SCHEMA_V1.items():
        slot_specs.append(
            f'  - "{meta.name}" ({meta.cn_name}, {meta.value_type}, P{int(meta.priority)}): {meta.description}'
        )
    schema_block = "\n".join(slot_specs)

    return f"""你是一位中文事件解析专家。把下面这句话按 LnMem 万能语法 schema_v1 拆成 20 槽 JSON。

# Schema (20 槽，按英文 name 作为 JSON key)
{schema_block}

# 输入
- user_id: {user_id}
- text: "{text}"
- timestamp: 解析时假设是当前时刻（UTC），相对时间（如"昨晚"）按这个基准换算成 ISO8601

# 输出要求
- **严格 JSON only**（不要任何前后说明 / markdown 包裹）
- 顶层结构：{{"slots": {{slot_name: {{value, confidence, raw_text}}}}}}
- value 类型：
  - entity: 字符串（人名/物名/地点）
  - span: 字符串（动作/话题等短语）
  - scalar: 数值（sentiment ∈ [-1,1]）或 ISO8601（when_time）
  - categorical: 字符串（intent ∈ ask/inform/request/complain/share/plan/reflect/other 等）
  - enum: 字符串（modality ∈ speech/text/vision/multimodal）
  - link: list of event_id strings（无关联则空 list）
- confidence: [0, 1] 你对这个抽取的把握程度
- raw_text: 原文中对应的字符片段（无对应则 null）
- **缺失的槽位**：value=null, confidence=0.0, raw_text=null
- P0 4 槽（who_actor, what_action, when_time, where_location）尽量填，无法推断时 null

# 示例输出格式
{{"slots": {{
    "who_actor": {{"value": "我", "confidence": 0.95, "raw_text": "我"}},
    "what_action": {{"value": "吃饭", "confidence": 0.9, "raw_text": "吃了饭"}},
    "when_time": {{"value": "2026-05-11T20:00:00+08:00", "confidence": 0.85, "raw_text": "昨晚 8 点"}},
    ...
    "spatial_link": {{"value": [], "confidence": 0.0, "raw_text": null}}
}}}}

请解析。"""


# ---------- 评估 ----------

@dataclass
class SpikeResult:
    sentence: str
    backend: str
    raw_output: str
    parsed_ok: bool
    parse_error: str | None = None
    slots_filled: int = 0
    p0_filled: int = 0
    p0_expected: list[str] = field(default_factory=list)
    p0_hit: list[str] = field(default_factory=list)
    p0_miss: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


def evaluate_one(
    sentence: dict,
    backend_caller,
    backend_label: str,
) -> SpikeResult:
    prompt = build_schema_prompt(sentence["text"], sentence["user_id"])
    t0 = time.time()
    try:
        raw, backend = backend_caller(prompt)
    except Exception as e:
        return SpikeResult(
            sentence=sentence["text"], backend=backend_label,
            raw_output=f"<<call failed: {type(e).__name__}: {e}>>",
            parsed_ok=False, parse_error=str(e),
            elapsed_s=time.time() - t0,
            p0_expected=sentence["expect_p0"],
        )
    elapsed = time.time() - t0

    # 提取 JSON（防 markdown 包裹）
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # strip markdown fence
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return SpikeResult(
            sentence=sentence["text"], backend=backend,
            raw_output=raw, parsed_ok=False, parse_error=str(e),
            elapsed_s=elapsed, p0_expected=sentence["expect_p0"],
        )

    slots = parsed.get("slots", {}) if isinstance(parsed, dict) else {}
    if not isinstance(slots, dict):
        return SpikeResult(
            sentence=sentence["text"], backend=backend,
            raw_output=raw, parsed_ok=False,
            parse_error=f"top-level 'slots' missing or not dict (got {type(slots).__name__})",
            elapsed_s=elapsed, p0_expected=sentence["expect_p0"],
        )

    # 统计填充
    filled_names = {
        name for name, sv in slots.items()
        if isinstance(sv, dict) and sv.get("value") is not None
        and not (isinstance(sv.get("value"), list) and len(sv["value"]) == 0)
    }
    p0_names = {meta.name for sid, meta in SCHEMA_V1.items() if sid in P0_SLOTS}
    p0_filled = filled_names & p0_names

    expected_p0 = set(sentence["expect_p0"])
    p0_hit = sorted(expected_p0 & p0_filled)
    p0_miss = sorted(expected_p0 - p0_filled)

    return SpikeResult(
        sentence=sentence["text"], backend=backend,
        raw_output=raw, parsed_ok=True, parse_error=None,
        slots_filled=len(filled_names), p0_filled=len(p0_filled),
        p0_expected=sentence["expect_p0"],
        p0_hit=p0_hit, p0_miss=p0_miss,
        elapsed_s=elapsed,
    )


def run_spike() -> dict:
    """跑全套 5 句 × N backend，返回汇总。"""
    backends = [
        ("gemini:3.1-flash-lite", lambda p: call_gemini(p, "gemini-3.1-flash-lite-preview")),
        ("gemini:2.5-flash",       lambda p: call_gemini(p, "gemini-2.5-flash")),
    ]

    all_results: list[SpikeResult] = []
    for backend_label, caller in backends:
        print(f"\n{'='*70}\nBackend: {backend_label}\n{'='*70}")
        for s in TEST_SENTENCES:
            print(f"\n📝 {s['text']}")
            print(f"   expect P0: {s['expect_p0']}  notes: {s['notes']}")
            r = evaluate_one(s, caller, backend_label)
            print(f"   parsed_ok={r.parsed_ok}  slots_filled={r.slots_filled}/{NUM_SLOTS}"
                  f"  p0_hit={r.p0_hit}  p0_miss={r.p0_miss}"
                  f"  elapsed={r.elapsed_s:.1f}s")
            if not r.parsed_ok:
                print(f"   ❌ parse_error: {r.parse_error}")
                print(f"   raw[:200]: {r.raw_output[:200]}")
            all_results.append(r)

    # 汇总
    total = len(all_results)
    parsed = sum(1 for r in all_results if r.parsed_ok)
    parse_rate = parsed / total
    avg_slots = sum(r.slots_filled for r in all_results) / total
    p0_recall = (
        sum(len(r.p0_hit) / max(len(r.p0_expected), 1) for r in all_results) / total
    )

    summary = {
        "n_total": total,
        "json_parse_rate": parse_rate,
        "avg_slots_filled": avg_slots,
        "avg_slots_pct": avg_slots / NUM_SLOTS,
        "p0_recall_avg": p0_recall,
        "by_backend": {},
    }
    for backend_label, _ in backends:
        bs = [r for r in all_results if backend_label in r.backend]
        if not bs:
            continue
        summary["by_backend"][backend_label] = {
            "n": len(bs),
            "parsed_ok": sum(1 for r in bs if r.parsed_ok),
            "avg_slots_filled": sum(r.slots_filled for r in bs) / len(bs),
            "p0_recall": sum(len(r.p0_hit) / max(len(r.p0_expected), 1) for r in bs) / len(bs),
            "avg_elapsed_s": sum(r.elapsed_s for r in bs) / len(bs),
        }

    print(f"\n{'='*70}\n📊 Spike Summary\n{'='*70}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 判定
    PASS_PARSE = 0.90
    PASS_P0 = 0.60
    PASS_AVG_SLOTS = 8 / NUM_SLOTS
    print(f"\n判定：")
    print(f"  json_parse_rate {parse_rate*100:.0f}% {'✅' if parse_rate >= PASS_PARSE else '❌'} (≥{PASS_PARSE*100:.0f}%)")
    print(f"  p0_recall {p0_recall*100:.0f}% {'✅' if p0_recall >= PASS_P0 else '❌'} (≥{PASS_P0*100:.0f}%)")
    print(f"  avg_slots_filled {avg_slots:.1f}/{NUM_SLOTS} {'✅' if (avg_slots/NUM_SLOTS) >= PASS_AVG_SLOTS else '❌'} (≥{PASS_AVG_SLOTS*100:.0f}%)")
    if parse_rate >= PASS_PARSE and p0_recall >= PASS_P0 and (avg_slots/NUM_SLOTS) >= PASS_AVG_SLOTS:
        print(f"\n🎉 SPIKE PASS — 进 parser_v0_llm 完整实现")
        summary["verdict"] = "PASS"
    else:
        print(f"\n❌ SPIKE FAIL — 写 ADR 报告，提替代方案（gliner / multi-LLM ensemble / human-in-the-loop）")
        summary["verdict"] = "FAIL"

    return summary


if __name__ == "__main__":
    out = run_spike()
    out_path = Path("/tmp/parser_prompt_spike_result.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📄 详细结果落盘：{out_path}")
