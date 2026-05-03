# ncps τ 偏置 spike · 三阶段渐进破雷实验

> **目的**：验证 v0.3.1 LNN 原生轻量个性化三件套中"per-user τ 偏置"在 [mlech26l/ncps](https://github.com/mlech26l/ncps) 库上的工程可行性。
>
> **触发**：v0.3.1 把 LoRA 换成 τ 偏置 + FiLM + 槽位偏好向量，但**没人在 ncps 库上做过 per-user τ 偏置注入**。这是个真假两态的工程风险。
>
> **执行人**：cc 主线 + codex 对抗审 + 老板拍板（v1.1 单人 + AI 协作模式）。
> **执行硬件**：Studio M2 Max 64GB（CPU/MPS 都可，spike 规模小）。
> **执行时机**：M0 W1（一周内必须出三阶段结论）。

---

## 三阶段结构

```
Day 1 ─────────────► Day 2-3 ─────────────► Day 4-5
能跑通 (24h)         训练稳定性 (48h)        表达能力 (48h)
   │                    │                       │
   红 → 立即触发 D5      红 → Plan B          红 → Plan B
```

任一阶段红 → 立即停下推进，按 [`docs/memory-architecture.md`](../../docs/memory-architecture.md) §7.5.2 Plan B 优先级评估替代方案：

```
P1 FiLM 单独 > P2 BitFit + LoRA(rank=2) > P3 回退原 LoRA(rank=8) — 单人模式不可行
```

---

## 三阶段验收门槛

| 阶段 | 文件 | 验证目标 | 红 / 黄 / 绿 |
|------|------|---------|--------------|
| Day 1 | [`day1_smoke.py`](day1_smoke.py) | LTC cell 加 `tau_offset` 参数 + forward + backward + 单步梯度 | 跑不通 = 红 / 能跑 = 绿 |
| Day 2-3 | [`day2_3_stability.py`](day2_3_stability.py) | 在合成正弦序列上训练 1000 步，对比有/无 τ_offset 的 loss 曲线 | 发散 > 30% = 红 / 10-30% = 黄 / < 10% = 绿 |
| Day 4-5 | [`day4_5_capacity.py`](day4_5_capacity.py) | 用 5 个 spike 内自包含的 prompt 数据（`synthetic_personas.py`）训练，看 τ_offset 是否能区分 5 种行为模式 | 区分度 < 60% = 红 / 60-80% = 黄 / ≥ 80% = 绿 |

---

## 安装

```bash
# 在 spikes/ncps_tau_bias/ 目录下
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **注意**：`ncps` 库的具体 LTCCell API 在不同版本可能有细微差异，每个 day 脚本里的 `# TODO[ncps-api]` 标记位置需要按实际版本调整 `tau_offset` 注入方式。这是 spike 的一部分——验证 API 兼容性也是 Day 1 的一部分。

---

## 运行

```bash
# Day 1（24h 内）
python day1_smoke.py

# Day 2-3（48h 内，绿 / 黄 才进入）
python day2_3_stability.py --output day2_3_results.json

# Day 4-5（48h 内，绿 / 黄 才进入）
python day4_5_capacity.py --output day4_5_results.json
```

---

## Plan B 触发规则

任一阶段红 → **立即在 [`docs/decisions/`](../../docs/decisions/) 写一份新 ADR**（编号 `002-` 起），格式参考 [`000-template.md`](../../docs/decisions/000-template.md)：

- 哪个阶段红
- 失败现象（loss 曲线 / error trace / 区分度数字）
- 选择哪条 Plan B 路径（P1 FiLM / P2 BitFit+LoRA(2)）
- 后续 M3 个性化栈方案怎么改

---

## 执行后产物

- `day1_results.txt` / `day2_3_results.json` / `day4_5_results.json` — 各阶段结果
- 更新 [`docs/memory-architecture.md`](../../docs/memory-architecture.md) §7.5 中的"待校验项"为实测数字
- 如果三阶段全绿 → 工程可行性破雷成功，可放心进入 M1 LNN 雏形

---

**spike 起源**：v1.1 工程计划评审中老板挑战 cc 原版"24h 一阶段"的局限性。详见对话 2026-05-03。
