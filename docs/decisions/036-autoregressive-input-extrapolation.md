# ADR-036: predict_future 自回归 input 外推（替换 persistence assumption）

**状态**：accepted
**日期**：2026-05-26
**决策者**：川哥（钦定 "改自回归外推 + 5s→1s" → "A+B 混合最鲁棒"）
**触发**：川哥反馈"表情切换时预测曲线没明显连续变化"
**关联**：ADR-033（LTC ODE persistence 初版）

---

## 背景：cc 实测 + Team 模式 (cc + gemini 3.5-flash high) 联合诊断

### cc 实测炸药级证据（curl 旧 daemon）
```
step  predicted_state   top_prob  entropy
0     transitioning     0.349    1.828
9     transitioning     0.351    1.825
8 类 state_probs step0→step9 最大变化 = 0.0037 (0.37%)
intent 3 类完全一致 [0.297, 0.333, 0.370]
```

**结论：LTC 10 步外推几乎是直线，第 2 步就到稳态了。** 这是"看似真预测、实则 hold-current-state"——比 v0 mean 外推强不了多少。

### Gemini 3.5-flash high thinking 钦定根因 1（致命）

> **Persistence Assumption → ODE 趋向稳态吸引子**
> 在 `predict_future` 中，未来 10 步全部使用最后一帧 input。对于基于 ODE 的 LTC/CfC 模型，**当输入 u 为常数时，非线性动力系统会迅速衰减并收敛到该输入对应的稳态吸引子**。因此预测曲线在第 2 步之后就会迅速变平，失去动态特征。
>
> "持久化假设 + 10 步 LTC 外推这套范式，绝对无法预测刺激驱动的突变。"

## 决策：A+B 混合自回归外推

### 算法

```python
# 每步算下一帧 input
input[t+1] = input[t] + α·velocity - β·(input[t] - baseline)
velocity = velocity * 0.8   # 衰减防爆炸

# 初始化
velocity = (history[-1] - history[-VELOCITY_WINDOW]) / (VELOCITY_WINDOW - 1)
baseline = mean(history)    # history 是最近 10s 窗口

# 系数
α = 0.5    # trend 保持
β = 0.1    # decay 回 baseline
VELOCITY_DECAY = 0.8
VELOCITY_WINDOW = 5
```

### 物理意义（生物学合理性）

- **trend 保持**（α）：表情动作的物理惯性。比如刚开始笑，未来 1-2s 笑容会继续加深
- **decay 回 baseline**（β）：表情的本质是"短期偏起 baseline"。30s 内必会回到中性 baseline
- **velocity 衰减**（0.8）：trend 预测随时间衰减——10s 后的 trend 不应该用现在的 trend 直接外推

### 配合改动

- `_SSE_PREDICTION_INTERVAL_S`: **5.0 → 1.0**（跟 frame 推送同步，消除"延时感"）

## 选项对比（Team 评估时考虑过）

| 选项 | 描述 | 优缺点 |
|---|---|---|
| 维持 persistence（ADR-033） | input 常数 | ❌ 实测 LTC 收敛吸引子 0.37% 变化 |
| **A. Input 1阶导数外推** | velocity 持续 | ✅ 简单，✗ 长期 trend 爆炸 |
| **B. Decay 回 baseline** | 缓慢回归 | ✅ 生物学合理，✗ 无短期动态 |
| **A+B 混合** ⭐ | trend + decay | ✅ 最鲁棒，川哥钦定 |
| 训独立 forecasting head | DLinear/PatchTST | ✗ 需 1 周重训 |
| 反喂 predicted state 回 input | state→input 反向映射 | ✗ 维度不匹配（8→18），需训反向 head |

## 实测验证（cc 重启 daemon 后 curl）

| 指标 | 旧 persistence | 新 A+B 混合 | 改善 |
|---|---|---|---|
| 8 类 state 最大变化 | **0.0037** | **0.2363** | **64×** |
| 8 类平均变化 | 0.0011 | 0.0591 | 53× |
| top_prob 演化轨迹 | 0.349 → 0.352 | 0.762 → 0.998 | 真有动态 |
| entropy | 1.825 → 1.825 | 1.199 → 0.021 | 从模糊到高置信 |
| intent 最大变化 | ~0 | 0.2603 | 64× |

## 回退策略

- Level 1：A+B 参数不好调 → 改成单纯 B（decay only），α=0
- Level 2：完全 revert：恢复 ADR-033 persistence 写法
- Level 3：弃用 predict_future，UI 不再画预测线（dashboard 只画历史）

## LNN-FIRST 合规性检查

- ✅ **未替换 LTC 内核**：仍用 `sm.backbone.rnn_cell + sm._apply_film + sm.readout` 完整 LTC ODE forward
- ✅ **只是改 input 时序假设**：从 "未来 input 常数" 改成 "未来 input 有动态"
- ✅ **不引入 LLM / 规则系统主逻辑**：α/β/decay 是工程性外推系数，不是状态机
- ✅ 符合 ADR-033 §C "Phase 3 后续可升 forecasting head" 但更轻量

## 已知 limitation

- ⚠️ **不能预测刺激驱动的突变**（Gemini 钦定）：用户突然惊讶大笑这种 input 外部刺激模型无法预知
- ⚠️ **客户端 ts_ns 仍硬编码 1e9 ns/step**（gemini 根因 2 未修）：UI 时间轴标注 vs LTC 实际 timespan 单位不一致。下一个 ADR 解决
- ⚠️ **A+B 系数是手调的**：α=0.5/β=0.1 是 cc 估值，未通过数据 sweep 调优
- ⚠️ **训练时模型从未见过这种 input 外推**：但 LTC ODE 性质让它能合理处理任何 input（这是 LNN 优势）

## 后续路径（Phase 3+）

1. 修客户端时间轴：daemon `predict_future` response 返回真实 timespan，客户端按真实秒数画
2. 数据 sweep 调 α/β（拿不同 persona 真人数据找最优）
3. 训独立 forecasting head（DLinear 轻量级，1 周训练）

## 关联文件

- 修改：`poc/src/realtime/lnn_inferencer.py:438-518`（predict_future A+B 外推）
- 修改：`poc/src/realtime/daemon_main.py:61-62`（_SSE_PREDICTION_INTERVAL_S 5.0→1.0）
- 不动：`docs/realtime/realtime_dashboard.html`（客户端时序假设 ts_ns 仍硬编码 1s/step，下一个 ADR 修）

## Team 模式协作沉淀

- **gemini 3.5-flash high thinking** 提供致命根因 1 的理论分析（ODE 吸引子收敛）+ 4 方案对比
- **cc 实测**用 curl 拿到 step0→step9 变化 0.0037 的硬证据，跟 gemini 理论完美闭环
- **codex 5.3 high** 派出但落盘失败（输出未到 /tmp/codex_predict_diagnose_output.md），可能是 stdout 走丢
- 川哥 2 次问答选 "A+B 混合最鲁棒"，cc 实施 + curl 验证 64× 改善
