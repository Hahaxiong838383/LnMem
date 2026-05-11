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
| 0 | 2026-05-08 | — | 基线测量 | — | state_acc=0%, bnd_f1=0% | — | — | baseline | Day 5 训练前 |
| 1 | 2026-05-08 | f156e18 | state_recognition | CfC + τ-bias + FiLM, 20 epochs, lr=1e-3, stride=600 应达 ≥30% acc | state_acc=0% | state_acc=70.2%, bnd_f1=0.295 | +70.2% / +0.295 | kept | AC state ≥30% ✅, AC bnd ≥0.5 ❌; boundary 根因=pos_weight=5.0 不足(正样本0.3%) |
| 2 | 2026-05-08 | 08f44b5 | state_recognition | 验证: confusion matrix + threshold sweep + human verify | val_acc=70.2% | focused F1=0.87, transitioning F1=0.01 | — | kept | neutral 吞噬 transitioning/low_energy; 最优 boundary 阈值=0.45→F1=0.3123(+5.9%) |
| 3 | 2026-05-08 | 137878c | multimodal_fusion | 真人验证: 三模态采集(face+mic+app)应覆盖所有状态 | 4/12 维锁死 0.5 | 12/12 维真实, 但无表情特征 | — | kept | 发现 feature_dim 设计缺陷: 无 AU6/AU12 → 不能检测安静的积极; Day 6 扩展 |
| 4 | 2026-05-08 | — | multimodal_fusion | feature_dim 12→18: +6 AU 表情特征(smile/brow/cheek/lip/jaw/mouth)应修复 emotion 盲区 | 12 维无表情 | 18 维全通路, 138/138 tests PASS | +6 dim | kept | cc 主线 plumbing + codex AU extraction, 待重训验证 |
| 5 | 2026-05-08 | — | multimodal_fusion + P-1 LNN-FIRST | LTC 演化替代静态查表：每个 persona 有自己的 (base, Δ, τ) 18 个数值，d(expr)/dt = (A(state)-expr)/τ 半隐式欧拉演化 | 静态查表 | 30/30 personas LTC profile + 900 信号重生成 + 半隐式欧拉数值稳定 | per-persona | kept | LNN 个性化首次真正生效；vboss_007 vs vboss_001 反应速度差 12×，外显程度差 14× |
| 6 | 2026-05-08 | — | 5stage_memory | 把 5 阶段记忆建模为 5 个 LTC 单元串联：m₁(τ=30s) → m₂(5min) → m₃(1h) → m₄(1day) → m₅(∞ when confirmed)，重要性门 g 由 LLM 裁判输出 | 待实现 | memory_lnn.py 10/10 tests PASS + demo 4 项物理验证通过 | new module | kept | LnMem 项目核心模块；记忆从"LLM 文本游戏"升级为"动力学系统" |

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
