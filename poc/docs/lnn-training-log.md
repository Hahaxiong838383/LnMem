---
name: LNN 训练实验日志 schema
audience: cc（实验执行）+ codex（对抗审查）+ 老板（review）
purpose: 每次 LNN 训练实验的 hypothesis / metric_before / metric_after / status 落档
status: v0.1 schema（2026-05-05 Day 0 起草，Day 5 起开始追加实验记录）
parent: ./poc-spec.md
references:
  - ~/.claude/rules/atomic-experiments.md（原子实验协议）
---

# LNN 训练实验日志

> 对接全局 `~/.claude/rules/atomic-experiments.md` 原子实验协议。每次实验**必须先在此 log 写 hypothesis + metric_before**，再开训。

---

## Schema（TSV 风格 markdown 表）

| iteration | timestamp | commit | task | hypothesis | metric_before | metric_after | delta | status | notes |
|-----------|-----------|--------|------|-----------|---------------|--------------|-------|--------|-------|
| 0 | 待填 | 待填 | 基线测量 | — | — | — | — | baseline | PoC 启动前 |

**status 取值**（仅 4 种）：
- `kept` — 实验成功，保留 commit
- `reverted` — 实验失败，git revert
- `fixed` — 修复型实验
- `skipped` — 因外部原因未跑

**task 取值**：
- `state_recognition` — StateRecognitionLNN
- `boundary_detection` — BoundaryDetectionLNN
- `personalized_param` — 个性化参数训练
- `5stage_memory` — 5 阶段记忆 LLM prompt 调优
- `multimodal_fusion` — 多模态融合层调优

---

## 实验记录追加规则

1. 每次启训前先追加一行：填 `iteration / timestamp / commit / task / hypothesis / metric_before`
2. 训练完成后回填：`metric_after / delta / status / notes`
3. 实验失败 (`reverted`) 必须填 notes 说明失败原因

---

## 实验摘要（PoC 结束时填）

PoC 完成后总结：
- 总实验数
- `kept` / `reverted` / `fixed` / `skipped` 各多少
- 最大单次改善
- 关键转折实验（哪几次改变了模型走向）

---

署名：cc（Claude Opus 4.7）| 2026-05-05 v0.1 schema 起草
