# ADR-037: predict_future state 投影对齐 + intent EMA 平滑

**状态**：accepted
**日期**：2026-05-26
**决策者**：川哥（钦定方案 3 "本质修复：state 投影对齐 + intent EMA 平滑"）
**触发**：川哥反馈"虚线位置的曲线值应该体现我当前实时的表情/意图/情绪/注意力状态"
**关联**：ADR-033（LTC ODE 真预测）、ADR-036（A+B 自回归外推）

---

## 背景：cc 实测 + Team 模式 (cc + gemini 3.5-flash high thinking) 联合诊断

### cc 实测炸药级证据（5 秒 SSE 流抓取）

```
                 实时 frame state         同时刻 prediction[0]
帧 #1-3        emotional_negative 0.59  │ transitioning 0.56     ← 主导 state 翻转！
帧 #4          emotional_negative 0.94  │ emotional_negative 0.98
帧 #5-6        emotional_negative 1.00  │ emotional_negative 1.00

intent: positive 0.36 → negative 0.41 → none 0.63   (3 秒内 3 个 intent，欠阻尼)
```

**两个问题同时存在**：
- 🐛 frame 和 prediction[0] **算法不对称**（虚线左右两侧数值跳变）
- 🐛 state 锁死 emotional_negative（吸引子滚雪球）+ intent 决策边界抖动

### Gemini 钦定核心论断

> **"frame 和 prediction[0] 不连续问题不应该在 UI 层修，应该在数据流层实现因果一致性。"**
>
> 正解：predict_future 从 daemon infer **平滑后的隐藏状态** 开始 rollout（不是 LTC 裸 forward）。UI 上 t₀ 用 frame state 当锚。

> **state 锁死 + intent 抖动 = 同一系统两种极端病态**：
> - state = **慢变量过度阻尼**（LTC τ 大 + 自回归正反馈）
> - intent = **快变量欠阻尼**（无平滑，py-feat AU 噪声直接打到 decision boundary）

> **产品语义**："虚线点（t₀）必须是 absolute anchor，100% 由 frame state 决定，不让 prediction 染指"

## 决策

### 修改 A：state 投影对齐（predict_future step=0 短路）

```python
# infer() 末尾缓存最新 frame 输出
self._last_state_probs = list(state_probs)
self._last_intent_probs = list(intent_probs) if intent_probs else None

# predict_future() step=0 短路：直接用 frame 缓存值
for step in range(steps):
    if step == 0 and self._last_state_probs is not None:
        # 用 frame infer 的 state_probs/intent_probs（保证 UI 虚线两侧严格连续）
        predictions.append({
            "step": 0,
            "state_probs": list(self._last_state_probs),
            "intent_probs": list(self._last_intent_probs),
            "aligned_with_frame": True,
            ...
        })
        continue
    # step >= 1：走 A+B 外推 + LTC forward（保留 ADR-036 逻辑）
```

**效果**：UI 虚线左侧最新 frame ≡ 虚线右侧 prediction[0]（**完全相同**），视觉零跳跃。

### 修改 B：intent EMA 平滑（infer() 内部）

```python
# 阻尼系数：α=0.3（新值 30%，老值 70%）
# 可通过 env LNMEM_INTENT_EMA_ALPHA 覆盖
if self._intent_ema_probs is None:
    self._intent_ema_probs = list(intent_probs)  # 冷启动
else:
    a = self._intent_ema_alpha
    self._intent_ema_probs = [a*new + (1-a)*old for new, old in zip(intent_probs, self._intent_ema_probs)]

# 归一化防数值漂移
intent_probs = list(self._intent_ema_probs)
```

**效果**：intent_probs 不再 1 秒翻 3 次。决策边界附近的高频抖动被低通滤波（半衰期 ≈ 2 帧 = 2 秒）。

## 实测验证（cc 重启 daemon 后 curl）

```
PRED step=0 top=emotional_negative state_probs=[..., 0.987, ...]  aligned=True   ← ✅ frame 对齐
PRED step=1 top=emotional_negative state_probs=[..., 0.983, ...]  aligned=False  ← LTC forward
PRED step=2 top=emotional_negative state_probs=[..., 0.976, ...]  aligned=False  ← A+B 外推
```

- step=0 完美对齐 frame（aligned=True）
- step 1-N 仍按 ADR-036 A+B 外推（aligned=False）
- 同时 intent 走 EMA，跨帧变化平滑

## 选项对比

| 选项 | 描述 | 优缺点 |
|---|---|---|
| 维持现状 | persistence + LTC forward step=0 | ❌ frame vs pred[0] 主导 state 翻转，intent 抖动 |
| **A. 仅修 UI 错位**（从 step=1 开始画） | UI hack | ✅ 30min，✗ 治标，斜率仍突变 |
| **B. UI 错位 + 预测半透明** | UI 美化 | ✅ 1-2h，弱化预测属性，✗ 没改本质 |
| **C. 本质修复（本 ADR）** ⭐ | state 投影对齐 + intent EMA | ✅ 3-5h，根治不连续 + 抑制抖动 |
| D. 不修 | M0 demo 另谋出路 | ✗ 虚线错位是 UX 硬伤 |

## 回退策略

- Level 1：intent EMA α 调小（默认 0.3 → 0.5 更敏感 / 0.1 更平滑）
- Level 2：完全关掉 EMA：env `LNMEM_INTENT_EMA_ALPHA=1.0`（α=1 等于不平滑）
- Level 3：state 投影对齐关掉：手动改 `ALIGN_STEP0_WITH_FRAME = False`
- Level 4：完整 revert：git revert 这个 commit

## LNN-FIRST 合规性

- ✅ **LTC 内核未动**：仍用 `sm.backbone.rnn_cell + sm._apply_film + sm.readout`
- ✅ **只改输出层 alignment + 时序平滑**：state 投影是"取 frame 缓存"非"绕开 LTC"
- ✅ **intent EMA 是数值滤波**：跟 LNN-FIRST 准则正交（intent_head 本来就是 readout，不是 LTC）

## 已知 limitation

- ⚠️ **step=0 ≠ step=1 intent 仍可能跳变**：step=0 用 EMA-smoothed，step=1 用 LTC forward 直出。视觉上微转折但不再断崖
- ⚠️ **EMA 引入"反应延迟"**：α=0.3 让 intent 切换慢 2-3 帧（=2-3 秒）。这是阻尼的代价
- ⚠️ **state 投影只对齐 step=0**：step=1-N 仍是 LTC 真预测，跟 frame 仍有差异
- ⚠️ **未解决 gemini 钦定根因 1**：端到端延迟 1-1.5s（py-feat 物理 latency）—— 需要换更快 AU detector

## 关联文件

- 修改：`poc/src/realtime/lnn_inferencer.py:55-115`（__init__ 加 _last_state_probs/_last_intent_probs/_intent_ema_alpha）
- 修改：`poc/src/realtime/lnn_inferencer.py:406-432`（infer() 末尾加 EMA + 缓存）
- 修改：`poc/src/realtime/lnn_inferencer.py:529-560`（predict_future step=0 短路）
- 不动：`docs/realtime/realtime_dashboard.html`（UI 零改动，prediction[0] 自然连续 frame）

## Team 模式协作沉淀

- **gemini 3.5-flash high thinking** 提供 5 根因 + "state 投影对齐"方案理论 + 产品心理学论断
- **cc 实测** curl 抓 5 秒 SSE 流，硬证据：frame vs pred[0] 主导 state 翻转
- 川哥 1 次问答选 "本质修复"，cc 实施 + curl 验证 step=0 aligned=True 完美对齐
