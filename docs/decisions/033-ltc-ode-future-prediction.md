# ADR-033: LTC ODE 真预测 + 8 类合并成情绪+注意力两组

**状态**：accepted（Team 模式 cc + codex 并行实施）
**日期**：2026-05-25
**决策者**：川哥（5/25 14:15 钦定 "上 LTC ODE，做烂了 git revert"）
**触发**：UI 状态演化曲线 v0 用 mean 外推（假预测），川哥要求真 LNN 预测 + 8 类太杂乱合并

---

## 背景

v0 简单外推（最近 N 帧 mean）实现了"看起来像预测"，但本质是 hold-current-mean 不是真 LNN 预测。无法预测状态切换时机。

LNN 本质是 LTC ODE，理论上能从当前 hidden state h(t) 用 ODE solver 推 h(t+Δt)。这正是 Liquid Neural Network 设计初衷。

第二个问题：8 类 phase2 状态视觉上太杂乱，川哥钦定合并成两组。

## 决策

### 决策 A：LTC ODE 真预测

新增 daemon endpoint `/predict_future`：
```
GET /predict_future?steps=10
```

实现：
- 取最近一帧的 `fused_vector` (18 维) + `hidden_state` (64 维)
- 假设未来 N 步 input 持续 = 最后一帧（**persistence assumption**）
- for i in 1..N:
  - `hidden = state_model.backbone.rnn_cell(last_input, hidden, timespans)`
  - `hidden = state_model._apply_film(hidden, gamma, beta)`
  - `logits_t = state_model.readout(hidden)` → softmax → `state_probs_t (8 维)`
  - 过 `intent_head(hidden)` → softmax → `intent_probs_t (3 维)`
- 返回 list of 预测帧

**为什么 persistence assumption 合理**：
- 短期（5-10s）输入特征确实不剧烈变化
- LTC 内部状态会自然演化（即使 input 不变，hidden 也按 ODE 时间常数 τ 漂移）
- 这能捕获 "疲劳累积 / 注意力衰减" 这种内在动力学

**limitation（写在 endpoint response 元数据）**：
- 不能预测刺激驱动的状态突变（如突然有人敲门）
- 长期预测会衰减到 ODE attractor（不是真未来）
- 5-15s 内可信，>30s 不要看

### 决策 B：8 类合并成情绪 + 注意力两组，加 valence 标签

模型层不动，仍输出 8 类。UI 层做 mapping：

| 原 phase2 state | 分组 | valence 标签 | UI 颜色 |
|----------------|------|-------------|---------|
| `emotional_positive` | **情绪** | ✅ 正向 | 绿 |
| `emotional_negative` | **情绪** | ❌ 负向 | 红 |
| `neutral` | **情绪** | ⚪ 中性 | 灰 |
| `focused` | **注意力** | ✅ 正向 | 蓝 |
| `distracted` | **注意力** | ⚠️ 偏负 | 橙 |
| `fatigued` | **注意力** | ❌ 负向 | 紫 |
| `low_energy` | **注意力** | ❌ 负向 | 深灰 |
| `transitioning` | **注意力** | ⚪ 中性 | 粉 |

UI 展示：
- **情绪条**（占满宽度，3 类色块时间线）
- **注意力条**（占满宽度，5 类色块时间线）
- 每个色块 hover 显示 valence 标签

## 选项与对比（LTC ODE 实施方式）

| 选项 | 实现 | 优点 | 缺点 |
|------|------|------|------|
| **A: Persistence assumption** ⭐ | 未来 input = last_input | 简单实现 / 利用真 ODE | 不能预测外部刺激 |
| B: 时间延展 + Zero input | 未来 input = 0 | 纯 ODE 演化 | 实际不真实 |
| C: 训 forecasting head | 单独训未来预测 head | 真预测 | 需新模型 + 数据 + 1 周训练 |
| D: 直接外推 hidden trajectory | 用 hidden 历史的导数外推 | 不调模型 | 失去 LTC 非线性优势 |

A 立即可行 + 利用现有 ckpt + 无需重训。Phase 3 后续可升 C。

## 实施分工（Team 模式）

| 任务 | 负责 | 工时 |
|------|------|------|
| LTC-1 ADR-033（本文档）| cc | 30min |
| LTC-2 daemon `predict_future` 函数 + HTTP endpoint | **codex** (gpt-5.3-codex high) | 45min |
| LTC-3 UI 8→2 组合并 + valence | cc | 30min |
| LTC-4 UI 接 endpoint 替 v0 mean | cc | 30min |
| LTC-5 端到端验收 + commit | cc | 15min |

并行：LTC-2 / LTC-3 同时跑（codex 写 daemon，cc 写 UI 互不依赖）

## 回退策略（川哥钦定 "做烂了回退"）

**Level 1：UI 显示降级**
- `/predict_future` 调用失败 → 自动 fallback 用 v0 mean 外推
- UI 标 "⚠️ 真预测离线，用 mean 外推 fallback"

**Level 2：endpoint 关闭**
- daemon 加 env `LNMEM_DISABLE_PREDICT_FUTURE=1` 关掉 endpoint，client 自动 fallback

**Level 3：完整 revert**
```bash
git revert <ADR-033-commit-hash>
```
回到 v0 mean 外推状态。

## 验收标准

- [ ] `curl http://127.0.0.1:9876/predict_future?steps=10` 返合理 JSON（10 个未来帧）
- [ ] 未来 state_probs sum=1.0 ± 0.01（softmax 正确）
- [ ] 单元测试：mock state_model 跑 5 步外推不崩
- [ ] UI 看到曲线右侧是真预测（与 mean 外推视觉差异明显）
- [ ] 7 天无 ckpt 兼容性问题（intent_head reload 后预测 endpoint 仍工作）

## 监测（1 周后 retro）

- daemon `/predict_future` p95 延迟（应 < 100ms）
- 预测准确率：5s 后实际 state vs 5s 前预测 state（top-1 accuracy）
- 川哥主观打分：预测看起来"有道理" 1-5 分

---

**关联 ADR**：ADR-028 LNN 角色重定义 / ADR-029 Sprint 2 / ADR-031 active learning v2 / ADR-032 cooldown adaptation

**关联文件（实施时填）**：
- `poc/src/realtime/lnn_inferencer.py` (codex 加 predict_future 方法)
- `poc/src/realtime/daemon_main.py` (codex 加 HTTP endpoint)
- `streamlit_ui/intent_console.py` (cc 改 UI)
