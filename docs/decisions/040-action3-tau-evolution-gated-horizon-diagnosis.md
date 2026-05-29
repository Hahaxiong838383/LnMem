# ADR-040: 动作 3（τ 跨数量级演化）门控决策 — horizon 诊断 + 三方对抗审收敛

**状态**：accepted
**日期**：2026-05-29
**决策者**：川哥（钦定先诊断再决定碰不碰 τ）/ cc（实验主案 + 综合）/ gemini-3.5-high（对抗审推翻 cc 实验结论）/ codex-gpt-5.4（前置代码对抗审）
**触发**：川哥要求"改善 LTC 记忆核心让 τ 真演化"→ cc team 分析建议"先诊断 horizon 再决定" → cc 跑 lag ablation/window sweep → gemini 对抗审推翻 cc 实验（循环论证）→ 收敛为门控 Action 3
**关联**：[`ADR-027`](027-dt-aware-ode-and-tau-explosion.md)（τ 解放 [0.96,34]）、[`ADR-028`](028-lnn-role-redefine-online-intent.md)（τ 解放预测反而差→转 intent）、[`ADR-022`](022-tau-bias-gradient-blackhole-fix.md)（τ 梯度黑洞）、[`ADR-039`](039-deterministic-retrain-proves-data-bottleneck.md)（数据瓶颈）、[`tau-spectrum-audit.md`](../realtime/tau-spectrum-audit.md)（F6）

---

## 背景

川哥诉求："改善 LTC 记忆核心，让 τ 真正演化，兑现'记忆会进化'承诺。" cc 经三方 team 分析建议：**不急着碰 τ，先诊断当前 intent 任务到底需不需要 >8s 感受野**（因为 ADR-028 已证 τ 解放反而预测差，gemini 指出 τ 是时域尺度不是语义载体，codex 用代码证明现网"τ"是 timespans 门控 + 8 帧硬墙更早卡死）。

## 实验：horizon 诊断（lag ablation + window sweep）

cc 写 [`scripts/diag_horizon_ablation.py`](../../scripts/diag_horizon_ablation.py)，用 daemon 实际 ckpt(59cd1d9c) + 279 条 au_window(len≥8)：

**lag ablation**（保留最近 k 帧，早期 prefix→persistence）：
| k帧 | state L1 mean | intent L1 mean | state L1 max | intent L1 max |
|---|---|---|---|---|
| 1 | 0.5448 | 0.2428 | 1.99 | 1.30 |
| 2 | 0.3600 | 0.1798 | 1.96 | 1.30 |
| 4 | 0.1882 | 0.0854 | 1.85 | 0.88 |
| 8 | 0.0376 | 0.0180 | 1.46 | 0.54 |

**window sweep**（截断最近 w 帧 vs full）：
| w帧 | state L1 mean | intent L1 mean | 适用样本 |
|---|---|---|---|
| 2 | 0.4078 | 0.1867 | 279 |
| 4 | 0.2128 | 0.0981 | 279 |
| 8 | 0.0398 | 0.0181 | 248 |
| 12 | 0.0204 | 0.0085 | 103 |
| 16 | 0.0002 | 0.0002 | 30 |

**cc 初步解读（已被推翻）**：有效时程 8-12s，分钟级 τ 过度设计，τ 降 P3。

## gemini 对抗审推翻 cc 实验（关键，不可删除的教训）

gemini-3.5-high 判定 **cc 实验结论可信度=低**，4 个致命缺陷：

1. **循环论证（最致命）**：用 τ 已收敛到短时程（std 0.013）的模型测"需不需要长历史"，得出"不需要"。= "因为我聋所以听不到声音，从而证明世界上没有声波"。短 τ 模型对 >3-5s 输入已物理指数衰减，本就测不出长程需求。
2. **数据截断 + 幸存者偏差**：au_window 只采 10s，物理过滤掉长时程意图（如"1 分钟前系统报错→现在叹气"）。"10 秒鱼塘没捞到鲸鱼"≠"没有鲸鱼"。
3. **L1 均值陷阱**：均值 0.04 小，但 intent L1 max=0.54（某类概率漂移 0.27 → argmax 翻转）、state L1 max=1.46。误触发率正由这些临界样本决定，均值掩盖 max 翻转危险。
4. **persistence 填充欺骗性**：常值填充把 LTC 推向固定点吸引子，人为抹平早期动态，低估早期帧作用（zero/shuffle 填充结论会不同）。

## cc 承认 + 一处保留（CLAUDE.md §3.1 冲突诊断）

**cc 承认**：缺陷 1-4 全部成立。"分钟级 τ 过度设计"结论**撤回**。实验只能说"当前 10s 数据 + 当前短 τ 模型下 8 帧捕获绝大部分输出"，**不能外推到"任务不需要长时程"**。

**cc 保留（独立判断）**：gemini 主张"用户分钟级意图客观存在"同样是**未证实断言**，它也没证明 intent 任务真需要分钟级。**双方都不该预设答案**——这正引出门控路线里的中立 oracle 裁判。

## 决策：门控 Action 3（gated，不砍不养老）

```
Phase 1 · 数据基建 (打破 10s 天花板) — 本 ADR 同步实施
  au_window 多尺度: 最近 10s@1Hz (10帧) + 10-120s@0.1Hz (~11帧) ≈ 21 帧
  inferencer _recent_temporal 已缓存 300 帧, 只需改 intent_feedback_server
  的 seconds=10 → 多尺度采样, 向后兼容现有 [{ts_ns, au}] 格式

Phase 2 · 中立裁判 (gemini 二审升级: 信息论优先于模型, 彻底绕开循环论证)
  2a. 信息论互信息基线 (最便宜最中立, 零模型):
      算条件互信息 I(X_[-120s,-10s] ; Y_t | X_[-10s,0s])
      → 纯数据指标, 不训任何模型, 不受 τ / 调优 / 数据量影响
      → 趋于 0 → 长时程对当前标签无信息, 直接判定, 连 oracle 都不用训
      → gemini 二审钦定: oracle 模型本身可能是新循环论证(未调优长程模型低性能
        ≠ 长程无用), 互信息是唯一绝对中立的裁判
  2b. (仅当 2a 互信息 > 0) 才训 oracle 模型验证可学习性:
      Transformer/TCN, 但必须用 Δt embedding / 连续时间位置编码
      (gemini 二审: 21 帧非等间距裸拼会让算子把 1s/10s 跨度等同, 破坏平移不变性)
  2c. 必加 0s Reactive baseline (单帧无历史):
      gemini 二审 R5.2: 若单帧就解释 95% 意图, 10s vs 120s 之争是伪命题
      → 对照梯度: 0s → 10s → 120s, 看每段边际增益

Phase 3 · 条件激活 (gate, gemini 二审修正指标)
  gate 不用全局 F1 5% (忽略类别不平衡):
  → 长程相关意图子集 Macro-F1 提升 > 10% 且 Bootstrap 95% CI 不含 0 (显著性)
  → 且 10s 相比 0s 提升 > 2% (否则任务本身不需要时序, 重审 intent 定义)
  满足 → Action 3 (τ 跨数量级 [0.5s,120s]) 升 P0
  不满足 → 确认短时程够, 正面回答 LNN-FIRST 存在性 (为何不用 GRU/TCN?)
```

**动作 3 状态**：从 cc 上一轮"降 P3 养老"修正为 **门控任务**——阻断器=数据基建，触发器=oracle F1 提升。

## 瓶颈优先级（修正版）

| 级 | 项 | 状态 |
|---|---|---|
| P0 | 数据（responses 攒标签） | 川哥进行中（notify_watcher 修复后 352→404 已增长） |
| P0 | **Phase 1 数据基建（扩 au_window 多尺度）** | 本 ADR 实施 |
| P1 | Phase 2 oracle 中立验证 | 等 Phase 1 数据积累 |
| P2 | FiLM 在线学习（个性化"懂你"，正交于时程） | 待 |
| P3/gated | Action 3 τ 跨数量级 | Phase 2 gate 后定 |

## 不可妥协

- **信息论互信息 (Phase 2a) 是第一裁判**，优先于任何模型——唯一不受 τ/调优/数据量影响的中立指标。
- **不预设长时程需求**——双方（cc"8s 够" / gemini"分钟级存在"）都是未证实断言，互信息客观裁判。
- Phase 1 改 logging **向后兼容**，不破坏现有 intent_feedback 格式 / 训练 pipeline。
- 长程数据喂模型**必须带 Δt / ts_ns**，不准非等间距裸拼。

## gemini 二审 (ADR-040) 采纳 + 分歧 (CLAUDE.md §3.1)

**采纳**（与场景无关的真洞察）：
1. ✅ 信息论互信息基线（Phase 2a）——比 oracle 模型更中立，gemini 自己点破"oracle 也可能新循环论证"
2. ✅ 0s Reactive baseline（Phase 2c）——漏掉的最底层对照
3. ✅ Δt embedding（Phase 2b）——非等间距不能裸拼
4. ✅ F1 指标改 Macro-F1 子集 + Bootstrap 显著性（intent 三类不平衡）
5. ✅ 标注未来信息泄露审视——LnMem 标注是川哥**实时** confirm/correct（非事后看动作标注），泄露风险低于 gemini 设想，但 Phase 2 训练前需确认标注时刻对齐
6. ✅ 冷启动降级（历史不足 120s 的推理降级）

**分歧（不采纳，gemini 场景幻觉）**：
- ❌ 机械臂碰撞 / 控制闭环 IoU / 10Hz 实时 latency / 意图振荡 chattering / ROS2 gRPC——**LnMem 是桌面伴侣 AI 的情绪/意图识别，不是机器人控制**。gemini 这次把场景幻想成机器人控制，这些约束不适用。
- ⚠️ 闭环因果反馈：LnMem intent 预测触发弹窗确有轻微闭环，但有 cooldown 节流，无 chattering 风险，降级处理即可

## 验证 / 复盘

- **Phase 1 DoD**：au_window 覆盖 ≤120s 多尺度，帧数 ≤ 30，现有训练 pipeline 不报错，新旧格式兼容
- **Phase 2 gate 指标**：长程 oracle vs 短程 baseline 的 intent F1 gap ≥ 5%（绑定真实标签，不是 L1 自比）
- **何时知道对**：Phase 2 oracle 给出客观答案那一刻，"τ 要不要演化"之争终结
- **回退**：Phase 1 改坏采集 → git 回滚；数据格式不兼容 → 加版本字段

## 关键教训（沉淀）

1. **诊断实验也要防循环论证**：用受测对象自身的退化版本去测"需不需要某能力"，必然得到有偏结论。要用不受该能力限制的 oracle。
2. **cc 单轮自评仍乐观**：lag ablation 一跑出"8 帧够"就下"过度设计"结论，被 gemini 一审推翻。重要结论必须对抗审（印证 `feedback_cc_gemini_adversarial_review_power`）。
3. **L1 均值不能掩盖 max 翻转**：分类任务看临界样本翻转率，不看平均漂移。

---

**本 ADR 版本：v1.0 · 2026-05-29**
**实验脚本**：[`scripts/diag_horizon_ablation.py`](../../scripts/diag_horizon_ablation.py)（保留供 Phase 2 LTC 侧基线复用）
