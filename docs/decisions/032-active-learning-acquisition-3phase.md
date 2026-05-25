# ADR-032: 主动询问从硬编码规则改 LNN 自学习 3 阶段路线

**状态**：accepted (Phase 1+2 立即实施)
**日期**：2026-05-25
**决策者**：川哥（5/25 13:25 钦定 Phase 1+2 路径 + 二元 reward）
**触发**：川哥指示"不是直接设置规则，应该是通过 LNN 的自学习，来调整主动询问的时机和频次"

---

## 背景

v2 / v2.1 watcher 触发逻辑仍是 cc 写死的 6 触发器 + 3 个 cooldown 阈值。川哥反馈本质问题：**任何 cooldown 数字都是 cc 拍脑袋，不是 learning**。

真正的 active learning 应该是：LNN 自己看历史（问了 → 川哥怎么响应 → 反馈是否纠错模型）→ 学一个 should_query(state, history) → [0,1] 概率函数。

cc 的角色：只设计输入特征 + reward 信号，**不设计决策规则**。

## 决策

3 阶段路线：

| Phase | 目标 | 数据需求 | 复杂度 | 何时做 |
|-------|------|---------|--------|--------|
| **Phase 1** | Online cooldown adaptation (最小 learning) | 实时 N=20 即可起 | 30min | **立即** |
| **Phase 2** | query_decisions.jsonl 数据收集 | 持续累积 | 30min | **立即** |
| **Phase 3** | Acquisition model (NN) 替代 6 触发器 | ≥500 (state, query, response, reward) | 3-5 天 + ADR | M0 demo 后 |

## Phase 1+2 实施细节（5/25 13:30 落地）

### Phase 2: query_decisions.jsonl 数据收集

每次弹窗（无论是否响应）写一条到 `~/.lnmem/data/query_decisions/YYYY-MM-DD.jsonl`：

```json
{
  "ts_ns": 1779683...,
  "ui_session": "notify-watcher-v2-...",
  "trigger_reason": "A",
  "lnn_state": {
    "phase2_state": "emotional_negative",
    "intent_pred": "none",
    "evidence_str": "intent_top2_diff=0.038 phase2_margin=0.486"
  },
  "user_response": "positive | negative | none | gave_up | canceled",
  "corrected": "positive (或空)",
  "response_time_s": 5.2,
  "reward": 0 | 1
}
```

### 二元 reward（川哥钦定）

- **响应了** (corrected ∈ {positive, negative, none}) → **reward = 1**
- **gave_up** 或 **canceled** → **reward = 0**

（备选"加权 reward"：纠错+2/确认+1/gave_up-1/取消-2，要多按钮 UI 支持，Phase 1 不做）

### Phase 1: Online cooldown adaptation

```python
# 滑动窗口最近 N=20 次弹窗的 reward
rewards = read_last_n(query_decisions.jsonl, N=20)
avg_reward = sum(rewards) / len(rewards)

if avg_reward < 0.3:
    cooldown × 1.5    # 川哥嫌烦，拉长
elif avg_reward > 0.7:
    cooldown × 0.8    # 川哥积极参与，缩短
# 中间区不变

cooldown = clamp(cooldown, [180, 3600])  # 防失控
```

**关键参数**（plist EnvironmentVariables）：
- `ADAPT_WINDOW_N=20`
- `ADAPT_LOW_THRESH=0.3` / `ADAPT_HIGH_THRESH=0.7`
- `ADAPT_MIN_COOLDOWN=180` / `ADAPT_MAX_COOLDOWN=3600`
- `ADAPT_EVERY_N=5`（每 5 次弹窗重算）

**持久化**：`~/.lnmem/watcher_state/adapted_cooldown` 存最近 cooldown 值，watcher 重启复用。

### 主循环改动

```bash
EFFECTIVE_COOLDOWN=$(读上次 adapted 或 GLOBAL_COOLDOWN_S 默认 300s)

while true:
    if elapsed < EFFECTIVE_COOLDOWN: continue
    ...
    trigger_dialog 同步等待  # 不再 &
    TRIGGER_COUNT += 1
    if TRIGGER_COUNT % ADAPT_EVERY_N == 0:
        EFFECTIVE_COOLDOWN = adapt_cooldown(EFFECTIVE_COOLDOWN)
        save EFFECTIVE_COOLDOWN
```

## 为什么选 Phase 1+2 立即做

| 选项 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **Phase 1+2 立即** ⭐ | 30min 落 / 实时收敛 / 同时积数据 | 不是真 NN learning（cooldown × 1.5/0.8 仍是规则）| ⭐⭐⭐⭐⭐ |
| 直接 Phase 3 spike | 优雅 | 3-5 天 / mock 数据测的代码上线后可能不准 | ⭐⭐ |
| 只做 Phase 2 数据收集 | 保守 | 当前 watcher 还是写死规则，川哥仍嫌烦 | ⭐⭐⭐ |

## Phase 3 设计预案（M0 demo 后启动，需独立 ADR）

```python
# acquisition_head.py
class AcquisitionHead(nn.Module):
    def __init__(self, input_dim=...):
        # 输入特征:
        #   phase2_state (one-hot 8)
        #   intent_probs (3)
        #   margin, entropy, top_prob
        #   ckpt_age (归一化)
        #   历史 N 次弹窗这个 state 的次数
        #   上次 corrected != predicted (binary)
        #   距上次弹窗时间
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)  # should_query 概率
```

**训练**（Studio nightly train 同步训）：
- 数据：query_decisions.jsonl 累积的 (state, reward) 对
- loss = BCE(should_query, reward)
- 优化器：Adam 1e-4
- 每次训 50 epoch

**部署**：watcher 主循环每帧用 `should_query >= 0.5` 替代 6 触发器 + cooldown。6 触发器降级为冷启动 fallback（model 没数据时用）。

**verify**：1 周 retro 看 acquisition model vs cooldown adaptation 的 user response rate。

## 监测指标（1 周后 retro）

- Cooldown 收敛轨迹：第 1/3/7 天的 EFFECTIVE_COOLDOWN 实际值
- Avg reward 变化趋势：是否随 cooldown 调整而稳定到 [0.5, 0.7]
- 弹窗频率：每天总次数 vs DAILY_MAX
- query_decisions.jsonl 累积量：何时达 500 条触发 Phase 3

## 选项与对比

| 学习方法 | 数据需求 | 实施时间 | learning 程度 |
|---------|---------|---------|------------|
| Cooldown adaptation (Phase 1) | 20 条即起 | 30min | 弱（规则 × 系数）|
| Contextual bandit (Thompson) | ≥200 条 | 1-2 天 | 中（LR + Beta posterior）|
| **Acquisition NN (Phase 3)** | ≥500 条 | 3-5 天 | 强（小 MLP）|
| 完整 RL (DDPG/PPO) | ≥10000 条 | 2 周 | 最强但 overkill |

Phase 3 选 Acquisition NN（不上完整 RL）：数据量门槛适中 + 训练快 + 可解释（输入特征明确）。

---

**关联 ADR**：ADR-028 LNN 角色重定义 / ADR-029 Sprint 2 / ADR-031 v2 触发器
**关联文件**：`scripts/notify_watcher.sh` (v2.2) / `scripts/launchd/com.lnmem.notify-watcher.plist`
**关联数据**：`~/.lnmem/data/query_decisions/YYYY-MM-DD.jsonl`
