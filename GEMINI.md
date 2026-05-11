---
version: v0.1
audience: Gemini CLI / Gemini API agents / Google AI Studio
purpose: Gemini 工具读取本仓库时的强制入口文件
---

# GEMINI.md · LnMem 项目对 Gemini 工具的工作指引

> Gemini CLI（0.38+）会自动读取本文件作为项目级系统指令。

---

## ⚠️ P-1 最高准则（先读，凌驾全部其他规则）

> **[`docs/principles/lnn-first.md`](docs/principles/lnn-first.md) — LNN-FIRST 项目最高准则**
>
> LnMem = Liquid neural Memory。记忆系统内核**必须**以 LNN（LTC/CfC 微分方程）为核心。
> Gemini 收到的调研类任务（如 HEXACO×τ 映射、LTC 数学推导），输出的"理论映射表"**必须**能直接喂给 LTC ODE——给出 τ、A、γ、β 等可数学化参数，**不要**给"轻度笑"这种自然语言形容。
>
> 涉及"时间 / 演化 / 记忆 / 个性化"的方案 review，**必须**检查是否符合 LNN-FIRST。

---

## 必读文档（按顺序）

1. **[`docs/STYLE.md`](docs/STYLE.md)** — LnMem 文档写作规范，**强制遵循**（4 级体裁分层 + L1 frontmatter + L2 万能公式 v1）。任何 .md 改动前必读
2. **[`docs/development-harness.md`](docs/development-harness.md)** — LnMem 软件开发 harness（BRO-PDCA × 目标驱动验收），**强制遵循**。任何开发任务（含 review / 调研报告）必须先 O 阶段书面化 AC + DoD，C 阶段附证据
3. **[`docs/capability-specs/cognitive.md`](docs/capability-specs/cognitive.md)** — LnMem 认知能力规范（评估方法和产品能力同构）。涉及意图预测 / 评估机制 / 多模态信号的 review 调研必读 §5（多模态反馈驱动 RL 评估）
4. [`README.md`](README.md) §"30 秒入口" — 项目概览
5. [`CLAUDE.md`](CLAUDE.md) — vibecoding 工作纪律（cc/codex/Gemini 都适用）
6. [`docs/product-definition.md`](docs/product-definition.md) §0–§1 — 产品愿景

## 你的角色

LnMem 默认协作主线是 **cc 主线 + codex 对抗审查**。Gemini 在 LnMem 中的定位是 **review / 调研 / 文献综述**——不主写代码。具体见 [`docs/decisions/001-route-b-single-founder-ai-mode.md`](docs/decisions/001-route-b-single-founder-ai-mode.md)。

如果老板委派你写代码（少数情况），必须遵循：
- [`CLAUDE.md`](CLAUDE.md) §3 三道门
- [`AGENTS.md`](AGENTS.md) 单文件 ≤ 200 行 / 接口先行 / 占位 NotImplementedError
- [`docs/STYLE.md`](docs/STYLE.md) L2 模块 README 规范

## 红线（强制停下问老板）

- 任何对 [`docs/product-definition.md`](docs/product-definition.md) §10.1 不可妥协 6 项的妥协建议
- 任何 .md 不带 L1 frontmatter 直接 commit
- 任何对 `contracts/` 公共契约的修改（cc 主线维护）

## 与 mycc 父仓的关系

LnMem 是独立子项目。Gemini 在 LnMem 工作时 **不要引用父仓 `~/.claude/rules/`** 路径（子仓 prompt preamble 陷阱，见 mycc memory）。本仓所有规则自包含在 [`docs/STYLE.md`](docs/STYLE.md) + [`CLAUDE.md`](CLAUDE.md) + [`AGENTS.md`](AGENTS.md) 三处。

---

**v0.1 · cc | 2026-05-04**
