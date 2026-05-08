---
version: v1.0
audience: cc / LLM persona profile 生成器
purpose: HEXACO × FACS × LTC τ 的理论映射表（gemini-3-pro-preview 调研产出）
status: 稳定 v1.0（2026-05-08）
parent: ./memory-dynamics.md（5 阶段记忆系统）+ ../src/lnn/expression_dynamics.py
references:
  - Ekman & Friesen (1978) Facial Action Coding System
  - Ashton & Lee (2007) HEXACO model
  - Scherer (2009) Component Process Model
  - ../../docs/principles/lnn-first.md（**P-1 最高准则**）
---

# HEXACO × FACS 表情 × LTC 时间常数理论映射表

> 本表用于驱动 LLM（MiniMax-M2.7 / Gemini）为 30 虚拟主人画像生成 ExpressionProfile。
> LLM 不参与表情演化主逻辑（那是 LTC ODE 的事），只贡献"per-persona × per-feature 参数"。
>
> **来源**：gemini-3-pro-preview 2026-05-08 一次性调研产出，cc 主线归档。
> **使用方式**：本文件被 `poc/src/generate_expression_profiles.py` 注入到 LLM prompt 的 `{theory_table}` 槽。

基于心理学与 FACS，为 LTC 模型提供人格面部参数的理论基准。

## 1. HEXACO 因子对中性基线 (base) 的影响

| HEXACO 因子 (分高时) | smile_intensity | brow_furrow | cheek_raise | lip_corner_depress | jaw_drop | mouth_open_ratio |
|---|---|---|---|---|---|---|
| **H: Honesty-Humility (谦逊)** | ↓ (克制) | — | ↓ | — | ↓ | ↓ |
| **E: Emotionality (易感)** | ↓ | ↑↑ (愁眉) | — | ↑ | — | — |
| **X: Extraversion (外向)** | ↑↑ (爱笑) | ↓ | ↑ | ↓ | ↑ | ↑ |
| **A: Agreeableness (宜人)** | ↑ (亲和) | ↓ | ↑ | — | — | — |
| **C: Conscientiousness (尽责)** | ↓ (严肃) | ↑ | ↓ | — | ↓ | ↓ |
| **O: Openness (开放)** | ↑ | ↑ (深思) | ↑ | — | ↑ | ↑ |

## 2. HEXACO 因子对时间常数 (τ) 的影响

| HEXACO 因子 (分高时) | τ_persona 影响趋势 (微分方程响应速率) | 行为学表现解读 |
|---|---|---|
| **H: Honesty-Humility** | τ ↑ (长, >5s) | 情绪表达稳健克制，不轻易产生剧烈表情变化。 |
| **E: Emotionality** | τ ↓↓ (极短, <1s) | 情绪高度易感，表情随刺激迅速切换（极短时间常数）。 |
| **X: Extraversion** | τ ↓ (短, 1-2s) | 情绪上脸快，对外在刺激呈现迅速且外显的表情反馈。 |
| **A: Agreeableness** | τ 中 (中, 2-5s) | 根据社交情境节奏平缓响应，具备良好的自适应调节能力。 |
| **C: Conscientiousness**| τ ↑↑ (极长, >8s) | 极度严谨克制，表情常驻基线，情绪波动需要极大的积分累积。 |
| **O: Openness** | τ ↓ (短, 1-2s) | 易被新奇刺激吸引，微表情和肌肉神经反应灵敏且快速。 |

## 3. 不同状态 (State) 下的偏移量 (Δ) 参考范围

*注：基线值 `base` 加上特定状态的 `Δ` 后，其结果需保证在 `[0, 1]` 之间（负号表示低于基线减弱）。*

| State (状态) | smile_intensity | brow_furrow | cheek_raise | lip_corner_depress | jaw_drop | mouth_open_ratio |
|---|---|---|---|---|---|---|
| **neutral** | [-0.05, +0.05] | [-0.05, +0.05] | [-0.05, +0.05] | [-0.05, +0.05] | [0, 0] | [0, 0] |
| **focused** | [-0.1, +0.05] | [+0.2, +0.5] | [0, 0] | [0, +0.1] | [-0.1, 0] | [-0.1, 0] |
| **emotional_positive** | [+0.3, +0.8] | [-0.2, -0.1] | [+0.2, +0.6] | [-0.2, 0] | [0, +0.2] | [+0.1, +0.4] |
| **emotional_negative** | [-0.2, 0] | [+0.4, +0.8] | [-0.1, 0] | [+0.3, +0.7] | [0, +0.1] | [-0.1, +0.1] |
| **fatigued** | [-0.1, 0] | [+0.1, +0.3] | [-0.1, 0] | [+0.1, +0.3] | [+0.1, +0.3] | [+0.1, +0.2] |
| **low_energy** | [-0.1, 0] | [0, +0.1] | [-0.1, 0] | [+0.1, +0.2] | [0, +0.1] | [0, +0.1] |
| **distracted** | [0, +0.1] | [-0.1, +0.1] | [0, 0] | [0, 0] | [+0.1, +0.2] | [+0.1, +0.3] |
| **transitioning** | [-0.1, +0.1] | [-0.1, +0.1] | [-0.1, +0.1] | [-0.1, +0.1] | [0, +0.1] | [0, +0.1] |

## 4. FACS 心理学权威引用

1. **Ekman & Friesen (1978)**: *Facial Action Coding System (FACS)* 确立了肌肉运动 (AU) 与表情的生理关联，例如 AU6 和 AU12 联动表征真实的积极体验 (Duchenne 微笑)。
2. **Ashton & Lee (2007)**: *HEXACO 模型* 强调了诚实-谦逊与情绪性维度的独立性，支持不同人格因子在微表情幅度与响应灵敏度上的生理分离。
3. **Scherer (2009)**: *Component Process Model* 指出表情演化是一个动态的连续过程，为采用基于时间常数 (τ) 的连续微分方程取代阶跃状态切换提供了坚实的理论支撑。

提示：请大模型在生成 persona profile 时，严格按照上表方向将定性理论转化为数值参数，以供 LTC 微分方程运行。

---

署名：gemini-3-pro-preview（2026-05-08 调研）+ cc 归档
