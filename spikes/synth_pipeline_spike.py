#!/usr/bin/env python3
"""synthetic_data_generator type 1 端到端 spike (M0 W2 P-1, 2026-05-12).

PoC 资源分配铁律：优先验证未知。本 spike 验证 synthetic_data_generator
最核心 bet：

    "用 LLM 生成的合成对话经过 parser 后能否产出可训 LNN 的 EventTensor 集合，
     且满足产品定义 §6.3 硬阈值"

测试链路：
    LLM (gemini-3.1-flash-lite) — 生成 N 条 (text, ground_truth_slots) 对
        ↓
    parser (universal_grammar_parser, 已 spike 验证 5/5)
        ↓
    评估：
      - P0 完整度（产品定义 §6.3 硬阈值 ≥ 95%）
      - 跨 LLM 一致率（≥ 75%）— 不同 LLM 生成同 spec → parser 后槽位一致
      - 场景多样性（覆盖产品定义 §6.1 类型 1 的 8 个典型场景）

判定（PASS = 进 type1 完整实现）：
- p0_completeness ≥ 0.80（PoC 阶段放宽，产品要求 0.95）
- scene_diversity ≥ 5/8 场景
- 跨 LLM 一致率 ≥ 0.65（PoC 放宽）
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts.event_tensor import EventTensor  # noqa: E402
from contracts.protocols import ParseRequest  # noqa: E402
from contracts.schema_v1 import (  # noqa: E402
    P0_SLOTS, SCHEMA_V1, SlotID, SlotPriority,
)
from universal_grammar_parser.config import ParserConfig  # noqa: E402
from universal_grammar_parser.parser_v0_llm import SchemaGuidedLLMParser  # noqa: E402


# ---------- 8 个产品定义 §6.1 典型场景 ----------

SCENES = [
    ("早晨规划", "今天的工作和生活计划"),
    ("通勤", "去某个地方的交通过程"),
    ("工作专注", "专心写代码 / 文档 / 设计"),
    ("情绪低谷", "感到累 / 压力大 / 沮丧"),
    ("夜间反思", "睡前回顾今天 / 思考人生"),
    ("家庭琐事", "和家人互动 / 处理琐碎家务"),
    ("健康话题", "运动 / 看医生 / 体检"),
    ("关系处理", "和朋友 / 同事 / 上下级互动"),
]


# ---------- LLM 后端 ----------

GEMINI_PROXY_URL = "http://23.226.135.149:4000/v1/chat/completions"


def _http_post_json(url: str, headers: dict, body: dict, timeout_s: float = 60.0) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _strip_thinking(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        s = text.index("<think>")
        e = text.index("</think>", s) + len("</think>")
        text = text[:s] + text[e:]
    return text.strip()


def call_gemini(prompt: str, model: str = "gemini-3.1-flash-lite-preview", max_tokens: int = 4000) -> str:
    key = os.environ.get("GEMINI_PROXY_KEY")
    if not key:
        raise RuntimeError("GEMINI_PROXY_KEY env not set")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,  # 多样性需要 temp > 0
    }
    resp = _http_post_json(
        GEMINI_PROXY_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        body=body,
        timeout_s=60.0,
    )
    return _strip_thinking(resp["choices"][0]["message"]["content"])


# ---------- type 1 generator prompt ----------

def build_synth_prompt(scene_name: str, scene_desc: str, persona: str, day_iso: str) -> str:
    """让 LLM 生成"自然事件描述 + ground truth 槽位"。

    输出 JSON：{
        "text": "自然语言事件描述（中文，1-2 句）",
        "ground_truth_slots": {who/what/when/where/...: value, ...}
    }
    """
    return f"""你是 LnMem 万能语法合成数据生成器。生成一条 **{scene_name}** 场景的自然事件描述。

# 场景
- 名称: {scene_name}
- 描述: {scene_desc}
- 主人: {persona}
- 日期基准: {day_iso}

# 产出要求

输出 **严格 JSON**，结构：
```json
{{
  "text": "自然语言事件描述，1-2 句中文，符合"我..."第一人称叙事，可含具体地点/人物/时间词",
  "ground_truth_slots": {{
    "who_actor": "...",
    "what_action": "...",
    "when_time": "...",
    "where_location": "...",
    "listener": "...（如有）",
    "topic": "...（如有）",
    "intent": "...（ask/inform/request/share/plan/reflect/complain/other）",
    "sentiment": -1.0 到 1.0 的数值,
    "modality": "speech/text/vision/multimodal"
  }}
}}
```

约束：
- text 必须是**自然中文叙述**（不是清单/JSON 内嵌），让 parser 后端能从中抽出 ground_truth_slots
- ground_truth_slots 是你**生成时的 intent**（你写 text 时心里想到的槽位 ground truth）
- P0 4 槽（who/what/when/where）尽量都填
- text 含具体细节（不要"我做了某事"这种空泛描述）
- 不要 markdown 包裹，**直接 JSON**

请生成（场景：{scene_name}）："""


# ---------- 评估 ----------

@dataclass
class SynthSample:
    scene: str
    text: str
    ground_truth_slots: dict[str, Any]
    parser_event: EventTensor | None = None
    parser_p0_count: int = 0
    parser_total_filled: int = 0
    parse_error: str | None = None
    elapsed_gen_s: float = 0.0
    elapsed_parse_s: float = 0.0


def gen_one_sample(scene: tuple[str, str], persona: str, day_iso: str,
                   gen_model: str = "gemini-3.1-flash-lite-preview") -> SynthSample:
    """生成一条样本（不含 parser）。"""
    scene_name, scene_desc = scene
    prompt = build_synth_prompt(scene_name, scene_desc, persona, day_iso)
    t0 = time.time()
    raw = call_gemini(prompt, model=gen_model)
    elapsed = time.time() - t0

    # 解析生成的 JSON
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        gen_obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return SynthSample(
            scene=scene_name, text=f"<<gen failed: {e}>>",
            ground_truth_slots={}, parse_error=f"gen_json: {e}",
            elapsed_gen_s=elapsed,
        )
    text = gen_obj.get("text", "")
    gt = gen_obj.get("ground_truth_slots", {})
    return SynthSample(
        scene=scene_name, text=text, ground_truth_slots=gt,
        elapsed_gen_s=elapsed,
    )


def parse_sample_with_parser(sample: SynthSample, parser: SchemaGuidedLLMParser, persona: str) -> SynthSample:
    """跑 parser 解析 sample.text → EventTensor，更新 sample。"""
    if not sample.text or sample.parse_error:
        return sample
    req = ParseRequest(
        text=sample.text, user_id=persona,
        timestamp=datetime.now(timezone.utc),
    )
    t0 = time.time()
    try:
        event = parser.parse(req)
    except Exception as e:  # noqa: BLE001
        sample.parse_error = f"parser: {type(e).__name__}: {e}"
        return sample
    sample.elapsed_parse_s = time.time() - t0
    sample.parser_event = event
    counts = event.filled_priority_count()
    sample.parser_p0_count = counts["P0"]
    sample.parser_total_filled = sum(counts.values())
    return sample


def eval_cross_llm_consistency(text: str, persona: str, parsers: list[SchemaGuidedLLMParser]) -> dict:
    """同 text 跑多个 parser → 槽位一致率。"""
    req = ParseRequest(text=text, user_id=persona, timestamp=datetime.now(timezone.utc))
    events = []
    for p in parsers:
        try:
            events.append(p.parse(req))
        except Exception:
            events.append(None)

    if not events or any(e is None for e in events):
        return {"valid": False, "reason": "parser_failure"}

    # 比较 P0 槽位 value 是否一致（仅看是否都填了 + value 是否相等）
    p0_consistency = []
    for sid in P0_SLOTS:
        values = []
        for e in events:
            sv = e.get(sid)
            values.append(sv.value if sv else None)
        # 一致：所有非 None 值相同 OR 全 None
        non_none = [v for v in values if v is not None]
        if not non_none:
            p0_consistency.append({"slot": SCHEMA_V1[sid].name, "agree": True, "all_null": True})
        else:
            agree = all(v == non_none[0] for v in non_none) and len(non_none) == len(values)
            p0_consistency.append({
                "slot": SCHEMA_V1[sid].name, "agree": agree,
                "values": [str(v)[:30] for v in values],
            })
    agree_rate = sum(1 for c in p0_consistency if c["agree"]) / len(p0_consistency)
    return {"valid": True, "p0_agree_rate": agree_rate, "details": p0_consistency}


# ---------- spike main ----------

def run_spike(n_per_scene: int = 1, persona: str = "vboss_self") -> dict:
    """跑 8 场景 × N 样本 spike。"""
    day_iso = datetime.now(timezone.utc).date().isoformat()

    # parser 用 gemini-3.1-flash-lite-preview (一致主路径)
    parser_main = SchemaGuidedLLMParser(
        llm_client=lambda p: call_gemini(p, "gemini-3.1-flash-lite-preview", max_tokens=8000),
        config=ParserConfig(llm_model="gemini-3.1-flash-lite-preview", max_retries=1),
        sleep_fn=lambda s: None,
    )
    # 跨 LLM 一致率用 gemini-2.5-flash 作 second parser（如能加 Codex/MiniMax 更好，PoC 简化）
    parser_second = SchemaGuidedLLMParser(
        llm_client=lambda p: call_gemini(p, "gemini-2.5-flash", max_tokens=8000),
        config=ParserConfig(llm_model="gemini-2.5-flash", max_retries=1),
        sleep_fn=lambda s: None,
    )

    samples: list[SynthSample] = []
    for scene in SCENES:
        for _ in range(n_per_scene):
            print(f"\n📝 生成 [{scene[0]}]...")
            s = gen_one_sample(scene, persona, day_iso)
            print(f"   text: {s.text[:80]}{'...' if len(s.text) > 80 else ''}")
            if s.parse_error:
                print(f"   ❌ {s.parse_error}")
                samples.append(s)
                continue
            print(f"   ground_truth keys: {list(s.ground_truth_slots.keys())}")
            print(f"   解析中...")
            s = parse_sample_with_parser(s, parser_main, persona)
            if s.parse_error:
                print(f"   ❌ parser: {s.parse_error}")
            else:
                print(f"   parser_p0={s.parser_p0_count}/4 total_filled={s.parser_total_filled}/20"
                      f" (gen {s.elapsed_gen_s:.1f}s + parse {s.elapsed_parse_s:.1f}s)")
            samples.append(s)

    # 汇总评估
    n_total = len(samples)
    n_parse_ok = sum(1 for s in samples if s.parser_event is not None)
    p0_complete = sum(1 for s in samples if s.parser_p0_count == 4)
    p0_completeness = p0_complete / n_total
    avg_p0 = sum(s.parser_p0_count for s in samples) / n_total
    avg_total = sum(s.parser_total_filled for s in samples) / n_total
    scenes_covered = len({s.scene for s in samples if s.parser_event is not None and s.parser_p0_count >= 2})

    # 跨 LLM 一致率 — 只跑前 3 个成功样本（节省 API quota）
    print(f"\n{'='*70}\n跨 LLM 一致率评估（前 3 个样本 × main + 2.5-flash parser）\n{'='*70}")
    consistency_results = []
    eligible = [s for s in samples if s.parser_event is not None][:3]
    for s in eligible:
        cr = eval_cross_llm_consistency(s.text, persona, [parser_main, parser_second])
        print(f"  [{s.scene}] valid={cr.get('valid')} p0_agree={cr.get('p0_agree_rate', 0):.2f}")
        if cr.get("valid"):
            consistency_results.append(cr["p0_agree_rate"])
    avg_consistency = sum(consistency_results) / len(consistency_results) if consistency_results else 0.0

    summary = {
        "n_total": n_total,
        "n_parse_ok": n_parse_ok,
        "p0_completeness": p0_completeness,
        "avg_p0_filled": avg_p0,
        "avg_total_filled": avg_total,
        "scenes_covered": scenes_covered,
        "scenes_total": len(SCENES),
        "cross_llm_consistency": avg_consistency,
        "cross_llm_n_samples": len(consistency_results),
    }

    print(f"\n{'='*70}\n📊 Spike Summary\n{'='*70}")
    print(json.dumps(summary, indent=2))

    # 判定（PoC 阈值放宽，详见模块 docstring）
    PASS_P0 = 0.80
    PASS_SCENES = 5
    PASS_CONSISTENCY = 0.65

    print(f"\n判定（PoC 阈值，产品要求更高）：")
    print(f"  p0_completeness {p0_completeness*100:.0f}% {'✅' if p0_completeness >= PASS_P0 else '❌'} (PoC ≥{PASS_P0*100:.0f}%, prod §6.3 ≥95%)")
    print(f"  scene_diversity {scenes_covered}/8 {'✅' if scenes_covered >= PASS_SCENES else '❌'} (≥{PASS_SCENES}/8)")
    print(f"  cross_llm_consistency {avg_consistency*100:.0f}% {'✅' if avg_consistency >= PASS_CONSISTENCY else '❌'} (PoC ≥{PASS_CONSISTENCY*100:.0f}%, prod §6.3 ≥75%)")

    if (p0_completeness >= PASS_P0 and scenes_covered >= PASS_SCENES
            and avg_consistency >= PASS_CONSISTENCY):
        print(f"\n🎉 SPIKE PASS — 进 type1_llm 完整实现")
        summary["verdict"] = "PASS"
    else:
        print(f"\n❌ SPIKE FAIL — ADR 报告，改 generator prompt 或换 LLM")
        summary["verdict"] = "FAIL"

    return summary


if __name__ == "__main__":
    out = run_spike(n_per_scene=1)
    out_path = Path("/tmp/synth_pipeline_spike_result.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📄 详细结果落盘：{out_path}")
