"""LLM-driven 表情 profile 生成器（P-1 LNN-FIRST 强制）。

读取 poc/data/personas/*.json 画像（HEXACO + 依恋 + Schwartz + Layer 6 多模态签名），
调用 LLM 推导每个 persona 的 ExpressionProfile：
  - base ∈ [0,1]^6     — 中性基线（从 HEXACO 推导）
  - state_offset ∈ [-1,+1]^(8×6)  — 8 状态各自的偏移
  - tau ∈ [0.5, 20]^6  — 时间常数（从 HEXACO 推导）

输出：poc/data/personas/expression_profiles/<persona_id>.json

P-1 准则：LLM 在此**不参与表情演化逻辑**，只做"重要性裁判"型的参数推导。
真正的演化仍由 LTC ODE 驱动（见 expression_dynamics.py）。

用法：
  python -m poc.src.generate_expression_profiles --personas poc/data/personas \
      --out poc/data/personas/expression_profiles \
      --theory-table poc/docs/hexaco_au_tau_theory.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from poc.src.data.llm_clients import _retry_post_json, _strip_thinking
from poc.src.lnn.expression_dynamics import (
    EXPRESSION_DIMS,
    STATE_LABELS,
    ExpressionProfile,
    save_profile,
)


# ---------- LLM Chat Helper（OpenAI 兼容）----------


def _chat_minimax(prompt: str, max_tokens: int = 4000) -> str:
    """直调 MiniMax-M2.7 chat completions，返回 content 字符串。"""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY not set")
    body = {
        "model": "MiniMax-M2.7",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    resp = _retry_post_json(
        url="https://api.minimaxi.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        body=body,
        timeout_s=180.0,  # M2.7 thinking 模型可能 60s+
        max_attempts=2,
    )
    content = resp["choices"][0]["message"]["content"]
    return _strip_thinking(content)


def _chat_gemini_proxy(prompt: str, max_tokens: int = 16000, model: str = "gemini-2.5-flash") -> str:
    """直调 23.226.135.149:4000 Gemini Proxy chat completions。

    默认 gemini-2.5-flash（更快、thinking 占用少）。
    max_tokens 必须留足空间给 reasoning_tokens（thinking 模型特性）。
    """
    proxy_key = os.environ.get("GEMINI_PROXY_KEY")
    if not proxy_key:
        raise RuntimeError("GEMINI_PROXY_KEY not set")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    resp = _retry_post_json(
        url="http://23.226.135.149:4000/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {proxy_key}",
            "Content-Type": "application/json",
        },
        body=body,
        timeout_s=120.0,
        max_attempts=3,
    )
    content = resp["choices"][0]["message"]["content"]
    return _strip_thinking(content)


def chat_with_fallback(prompt: str, max_tokens: int = 16000) -> tuple[str, str]:
    """尝试 gemini-2.5-flash → fallback minimax，返回 (content, backend_used)。

    优先 gemini-flash 因为 minimax M2.7 thinking 模式经常 60s 超时。
    """
    try:
        return _chat_gemini_proxy(prompt, max_tokens, "gemini-2.5-flash"), "gemini-flash"
    except Exception as e:
        print(f"  [fallback] gemini-flash failed: {e}, trying minimax")
    return _chat_minimax(prompt, max_tokens), "minimax"


# ---------- LLM Prompt 模板 ----------


PROMPT_TEMPLATE = """你是一位 FACS（Facial Action Coding System）+ HEXACO 心理学专家，正在为虚拟主人画像推导 LTC（Liquid Time-constant）表情演化参数。

## 任务

根据下面的 persona 画像，推导这个 persona 在 LTC 表情动力系统中的 3 组参数：

```
LTC 公式：d(expr_i)/dt = (A_i(state) - expr_i) / τ_i
其中  A_i(state) = base_i + state_offset_i[state]
```

需要你输出：
1. **base**：6 个表情维度在"中性"状态下的基线值（base ∈ [0,1]）
2. **state_offset**：8 个状态 × 6 个表情维度的偏移量（Δ ∈ [-1,+1]，但要保证 base+Δ ∈ [0,1]）
3. **tau**：6 个表情维度的时间常数（τ ∈ [0.5, 20.0] 秒，越小反应越快）

## 6 个表情维度

- `smile_intensity`: 微笑强度（AU6+AU12，嘴角上扬 + 脸颊抬升）
- `brow_furrow`: 皱眉（AU4，内眉聚拢）
- `cheek_raise`: 脸颊抬升（AU6，Duchenne 微笑标志）
- `lip_corner_depress`: 嘴角下沉（AU15，悲伤标志）
- `jaw_drop`: 下巴张开（AU26，惊讶/投入）
- `mouth_open_ratio`: 嘴巴纵横比

## 8 个状态

low_energy, focused, distracted, fatigued, emotional_positive, emotional_negative, neutral, transitioning

## 心理学理论参考（gemini 调研产出）

{theory_table}

## Persona 画像（markdown 格式）

```markdown
{persona_md}
```

## 输出格式（严格 JSON）

{{
  "persona_id": "{persona_id}",
  "rationale": "≤200 字说明：根据这个画像的 HEXACO/依恋/Schwartz 特征，推导出的表情风格是什么样（例：内敛/外显/反应快/反应慢）",
  "base": {{
    "smile_intensity": 0.xx, "brow_furrow": 0.xx, "cheek_raise": 0.xx,
    "lip_corner_depress": 0.xx, "jaw_drop": 0.xx, "mouth_open_ratio": 0.xx
  }},
  "state_offset": {{
    "low_energy":         {{"smile_intensity": +0.xx, "brow_furrow": +0.xx, ...}},
    "focused":            {{...}},
    "distracted":         {{...}},
    "fatigued":           {{...}},
    "emotional_positive": {{...}},
    "emotional_negative": {{...}},
    "neutral":            {{"smile_intensity": 0.0, ...}},  // neutral 通常 Δ=0
    "transitioning":      {{...}}
  }},
  "tau": {{
    "smile_intensity": x.x, "brow_furrow": x.x, "cheek_raise": x.x,
    "lip_corner_depress": x.x, "jaw_drop": x.x, "mouth_open_ratio": x.x
  }}
}}

## 推导原则

1. **HEXACO 高 X（外向）+ 低 N（神经质）**：base_smile 高（0.10-0.20），τ_smile 短（1-3s），表情外显
2. **HEXACO 高 N（神经质）+ 低 X（外向）**：base_brow_furrow 高（0.15-0.25），τ 长（5-10s），表情内敛
3. **HEXACO 高 H（诚实-谦逊）**：base_smile 低（0.03-0.08），微笑克制不张扬
4. **HEXACO 高 C（尽责）**：focused 状态下 brow_furrow Δ 偏大（+0.2-+0.4），专注时眉头紧锁
5. **依恋安全型**：emotional_positive Δ_smile 大（+0.4-+0.6），愿意外显积极
6. **依恋回避型**：所有 Δ 都偏小，表情压抑
7. **base + Δ 必须 ∈ [0, 1]**，超出范围视为推导错误

只输出 JSON，不要任何额外说明。
"""


def build_prompt(persona_md: str, persona_id: str, theory_table: str) -> str:
    return PROMPT_TEMPLATE.format(
        persona_id=persona_id,
        persona_md=persona_md,
        theory_table=theory_table,
    )


def parse_llm_output(raw: str, persona_id: str) -> ExpressionProfile:
    """解析 LLM 返回的 JSON 为 ExpressionProfile，校验数值范围。"""
    # 容错：去掉可能的 ```json``` 包裹
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    # 容错：剥掉行尾 // 注释（gemini/minimax 偶尔返回带注释的 JSON）
    cleaned = re.sub(r"//[^\n]*", "", cleaned)
    # 容错：剥掉 trailing commas（{...,} 或 [...,]）
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    # 容错：截取首个 { 到最后一个 } 之间的内容
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # debug：把原始内容落盘便于分析
        debug_path = Path("/tmp") / f"llm_raw_{persona_id}.txt"
        debug_path.write_text(raw, encoding="utf-8")
        raise

    # 校验完整性
    base = data["base"]
    state_offset = data["state_offset"]
    tau = data["tau"]

    for dim in EXPRESSION_DIMS:
        assert 0.0 <= base[dim] <= 1.0, f"base[{dim}] = {base[dim]} 超出 [0,1]"
        assert 0.5 <= tau[dim] <= 20.0, f"tau[{dim}] = {tau[dim]} 超出 [0.5, 20]"

    # 容差 0.05 + 自动 clamp（LLM 偶尔输出微负值，是合理的边界情况）
    fixed_offset: dict[str, dict[str, float]] = {}
    for state in STATE_LABELS:
        assert state in state_offset, f"missing state_offset[{state}]"
        fixed_offset[state] = {}
        for dim in EXPRESSION_DIMS:
            offset_val = float(state_offset[state][dim])
            target = base[dim] + offset_val
            if target < -0.15 or target > 1.15:
                raise AssertionError(
                    f"base+offset[{state}][{dim}] = {target:.3f} 严重超出 [0,1]（>0.15 偏差）"
                )
            # 边界自动修正：若超 [0,1] 则 clip offset 到边界
            if target < 0.0:
                offset_val = -base[dim]
            elif target > 1.0:
                offset_val = 1.0 - base[dim]
            fixed_offset[state][dim] = offset_val

    return ExpressionProfile(
        persona_id=persona_id,
        base=dict(base),
        state_offset=fixed_offset,
        tau=dict(tau),
    )


def generate_one(
    persona_path: Path,
    out_dir: Path,
    theory_table: str,
) -> tuple[ExpressionProfile, str]:
    """读 persona markdown → 调 LLM → 解析 → 落盘 ExpressionProfile。

    返回 (profile, backend_name)。
    """
    persona_md = persona_path.read_text(encoding="utf-8")
    persona_id = persona_path.stem  # 例：vboss_001

    prompt = build_prompt(persona_md, persona_id, theory_table)
    raw, backend = chat_with_fallback(prompt, max_tokens=16000)

    profile = parse_llm_output(raw, persona_id)
    out_path = out_dir / f"{persona_id}.json"
    save_profile(profile, out_path)
    return profile, backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", type=Path, required=True,
                        help="poc/data/personas")
    parser.add_argument("--out", type=Path, required=True,
                        help="poc/data/personas/expression_profiles")
    parser.add_argument("--theory-table", type=Path, required=True,
                        help="gemini 调研产出的 HEXACO×AU×τ 映射表 markdown")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制处理 N 个画像（0 = 全部）")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    theory_table = args.theory_table.read_text(encoding="utf-8")

    # 支持 .md 或 .json 画像（实际 30 vboss personas 是 markdown）
    persona_files = sorted(args.personas.glob("vboss_*.md"))
    if not persona_files:
        persona_files = sorted(args.personas.glob("vboss_*.json"))
    # 排除 vboss_self（老板自己）— 仅处理 vboss_001..030
    persona_files = [p for p in persona_files if "self" not in p.stem]

    if args.limit:
        persona_files = persona_files[:args.limit]

    print(f"[generate_profiles] {len(persona_files)} personas to process")

    success = 0
    for i, pf in enumerate(persona_files, 1):
        try:
            profile, backend = generate_one(pf, args.out, theory_table)
            print(f"  [{i}/{len(persona_files)}] {pf.stem} [{backend}]: "
                  f"base_smile={profile.base['smile_intensity']:.2f} "
                  f"τ_smile={profile.tau['smile_intensity']:.1f}s")
            success += 1
        except Exception as e:
            print(f"  [{i}/{len(persona_files)}] {pf.stem}: ❌ {e}")

    print(f"[generate_profiles] {success}/{len(persona_files)} success → {args.out}")


if __name__ == "__main__":
    main()
