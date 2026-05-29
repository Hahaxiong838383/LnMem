# ADR-039: 确定性复现实验证明数据瓶颈 + intent_head/LTC 核心学习 gap

**状态**：accepted
**日期**：2026-05-29
**决策者**：川哥（要求手动训练验证链路改善作用）/ cc（执行 + 实测 + 归因）
**触发**：5/29 川哥要求"手动让 Studio 训练一次，结果返回 Air，验证这条链路对 LNN/LTC 预测/记忆真起改善作用"
**关联**：[`ADR-038`](038-path-d-plus-plus-demo-engineering-foundation.md)（路径 D++）、[`tau-spectrum-audit.md`](../realtime/tau-spectrum-audit.md)（F6）、notify_watcher 日期固化 fix（commit ce182e2）

---

## 背景

5/29 排查 Studio 训练状态时发现：`total_feedback=352, last_trained=352, increment=0`，Studio 已 40+ 小时因双门"增量≥10"正确 SKIP 训练。根因链：
1. notify_watcher 跨天日期固化 bug → 5/25 起 4 天 0 弹窗（已修，commit ce182e2）
2. → 0 新反馈标签 → increment=0
3. → Studio 无新数据可训

川哥指出"我之前给过很多真实反馈数据"（intent_feedback 累计 352 条，含 60 条真人纠正），要求**手动 `--force` 强制训练一次**，跑通 Studio→ckpt→Air 全链路，验证训练对项目的改善作用。

---

## 实验与发现

### 链路验证（全通）

```
Studio --force 训练 (09:03:58→09:04:09, 11秒)
  → intent_head_20260529-090358.pt (20 epoch, train acc 0.55→0.72)
  → latest symlink atomic 更新
  → studio-ibro (经 mini 雷电桥 192.168.100.2) pull 回 Air
  → Air ~/.lnmem/ckpt/intent_head_latest.pt 指向新 ckpt ✅
```

工程链路每一环打通，无缺陷。

### 关键发现：零改善（铁证）

| 对比项 | 0527 旧 ckpt | 0529 新 ckpt |
|---|---|---|
| dataset_size | 1387 | 1387 |
| seed | 42 | 42 |
| epochs | 20 | 20 |
| final loss | 0.2120055289000337 | 0.2120055289000337 |
| final acc | 0.7180966113914924 | 0.7180966113914924 |
| **权重 L2 diff** | — | **0.000000（逐字节相同）** |

`final_metrics` 逐位相同到小数点后 16 位，权重张量 L2 diff = 0。**同数据(1387 样本) + 同 seed(42) + 同超参 = 确定性复现。新 ckpt 与 5/27 逐字节完全一样。**

---

## 决策（结论）

### 结论 1：当前瓶颈是数据，不是训练链路也不是算法

```
链路 ✅ 完美    算法 ✅ 在学(train acc 0.72)    数据 ❌ 零增量
                                                    ↑
                        responses=0 → increment=0 → 重训=确定性复制
```

用实验钉死：
- 训练链路工程完美（Studio→ckpt→Air 全通）
- 算法本身在学（intent_head train acc 0.72）
- **但 responses=0 → 无新标签 → 重训只是复现，weights 零变化**

→ 修 notify_watcher 弹窗（已做）+ 川哥真去响应反馈（进行中），是让 weights 真正改变的唯一正路。F3 合成数据不能替代（会 OOD，且非新真实数据，见 ADR-038 §F3 + 上一轮 F3 讨论）。

### 结论 2：nightly-train 只训 intent_head，不训 LTC 记忆核心

这条链路训的是 `intent_head`（intent 分类头），**不是 LNN/LTC backbone 的 τ-bias / 记忆动力学核心**。

- intent_head 走 `train_intent_classifier.py`（监督训练分类头）
- LTC 记忆核心走 `online_adapter.tau_drift`（在线学习路径）
- F6 audit 实测：`online_tau_drift` mean ~4e-8，**几乎冻结**

→ **即便有新数据，这条链路改善的也只是 intent 分类，不是"记忆会进化"承诺对应的 LTC τ 演化。** LnMem 核心承诺 ②"它会变化"在 τ 维度上当前还没真正跑起来。这是 F6 audit + ADR-038 §F1 + 本 ADR 共同指向的同一个深层问题。

### 决策动作

| 动作 | 负责 | 状态 |
|---|---|---|
| 1. 修 notify_watcher 弹窗让新数据能进 | cc | ✅ 已修 (ce182e2) |
| 2. 川哥真去响应反馈攒新标签（≥10 条触发训练） | 川哥 | 🔄 进行中 |
| 3. 攒够后再 --force 训一次，验证 weights L2 diff >0 真改善 | cc+川哥 | ⏸ 等动作 2 |
| 4. **改善 LTC 记忆核心（让 tau_drift 真动）= τ 跨数量级 spike** | cc team 分析中 | 🔬 ADR-040 |

---

## 后果

### 正面
- 用最小实验（11 秒训练 + ckpt 对比）无可辩驳地定位了瓶颈，避免在链路/算法上浪费时间
- 厘清 intent_head 学习 vs LTC 核心学习的根本 gap，为动作 4（τ spike）提供动机

### 负面 / 风险
- 暴露 LnMem "记忆会进化"承诺当前在 τ 维度未兑现，是 demo 必须诚实面对的（已纳入 F6 §6 答辩底牌）

### 触发的下游决策
- 动作 4（LTC 记忆核心 τ 跨数量级 spike）→ 启动 ADR-040（team 模式分析中）

---

## 验证 / 复盘

- **何时知道结论对**：动作 2 攒够新标签后再训，若 weights L2 diff >0 且 final acc 变化 → 证明"数据进来就能改善"成立
- **回退**：无需回退（纯诊断性实验，未改生产数据）

---

## 复现命令

```bash
# 手动 --force 训练 (经 mini 雷电桥)
ssh mini "ssh fu-121@192.168.100.2 'cd ~/lnmem-poc && ./scripts/studio_nightly_train.sh --force'"
# 拉回 Air
LNMEM_STUDIO_HOST=studio-ibro bash scripts/pull_ckpt_from_studio.sh
# 对比新旧 ckpt 权重 diff
.venv-poc-fresh/bin/python -c "
import torch
old=torch.load('~/.lnmem/ckpt/intent_head_20260527-160005.pt',map_location='cpu',weights_only=False)
new=torch.load('~/.lnmem/ckpt/intent_head_20260529-090358.pt',map_location='cpu',weights_only=False)
print('final_metrics 相同:', old['final_metrics']==new['final_metrics'])
"
```

---

**本 ADR 版本：v1.0 · 2026-05-29**
