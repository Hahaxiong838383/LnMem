# GitHub Copilot · LnMem Project Instructions

> GitHub Copilot Chat 自动读取本文件作为项目级 system prompt（自 2024 起支持）。

## Mandatory Reads (in order)

1. **[`docs/STYLE.md`](../docs/STYLE.md)** — LnMem documentation writing standard. **Strict adherence required.** Covers L1-L4 doc tier system + L1 frontmatter + L2 Universal Formula v1 (RULE-ID + 4-dim priority + Why).
2. [`README.md`](../README.md) "30-second entry" section — project overview
3. [`CLAUDE.md`](../CLAUDE.md) — vibecoding workflow rules (3-gate PR review)
4. [`docs/product-definition.md`](../docs/product-definition.md) §0–§1 — product vision

## Hard Rules

- **Any `.md` file you create/edit must include L1 YAML frontmatter** (version / audience / purpose minimum). No exceptions for L4 narrative docs.
- **No compromise on the 6 non-negotiables** ([`docs/product-definition.md`](../docs/product-definition.md) §10.1) without owner approval
- **Single file ≤ 200 lines** (see [`AGENTS.md`](../AGENTS.md))
- **Interface first**: check [`contracts/protocols.py`](../contracts/protocols.py) before writing any module class
- **All PRs require**: cc implementation + codex adversarial review + human engineer signature

## Project Context

LnMem is a companion-AI memory system using Liquid Neural Networks (LNN) + Universal Grammar 20-slot schema + Graphiti bi-temporal KG. Default collaboration mode is **single founder + AI** (Boss + cc-mainline + codex-adversarial). Copilot's role is **inline code completion** — for design decisions, defer to cc/codex.

## Anti-patterns (forbidden)

- Force L4 narrative docs (product/research/teaching) into L2 RULE-ID format — destroys readability
- Skip L1 frontmatter on any new `.md` — CI should reject (pre-commit hook planned for M0 W3)
- Reference parent repo `~/.claude/rules/` paths — LnMem is self-contained

---

**v0.1 · 2026-05-04 · Aligned with [`docs/STYLE.md`](../docs/STYLE.md) v0.1**
