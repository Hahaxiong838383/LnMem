---
name: LnMem Day 15 · 实时大屏 + LTC 预测栈大改造
description: 单日 9 commit + 4 ADR，从 dashboard 卡死到 LTC 预测对齐 frame 全栈整改
type: progress
created: 2026-05-26
phase: M0 W3 · Realtime Stack Overhaul
---

# Day 15 — 实时栈整改全日报告

> 2026-05-26 单日（21:00-22:00 CST 收尾），围绕川哥 5 轮反馈的 dashboard / LTC 预测问题，
> 完成 **9 commit + 4 ADR (034→037) + 3 个 mycc 记忆教训沉淀**。
> Team 模式 (cc + gemini-3.5-flash high thinking) 跑了 **3 次完整深度诊断**。

---

## A. 单日全部 commit（9 个）

| # | Commit | 概述 | ADR |
|---|---|---|---|
| 1 | `891e24a` | 历史/预测 label 改 HTML overlay（chart annotation v3 limitation） | — |
| 2 | `7593f48` | label 修正定位 + z:100 + drawTime:'afterDraw'（中间方案，未根治） | — |
| 3 | `ae53216` | legend 移左侧对齐 + 历史/预测 label 浮到"现在"虚线正上方 | — |
| 4 | `f334674` | EventSource → fetch+ReadableStream 重写客户端 SSE（Team 模式钦定方案 B） | **ADR-035** |
| 5 | `57592d0` | daemon /dashboard 加 Cache-Control no-cache header（11 commit 真根因） | — |
| 6 | `fa58f07` | predict_future A+B 自回归 input 外推 + SSE 推送 5s→1s | **ADR-036** |
| 7 | `279da28` | refreshAllCharts rAF 节流（修 ADR-036 引入的 backlog） | — |
| 8 | `f889f93` | state 投影对齐 + intent EMA 平滑（虚线两侧严格连续） | **ADR-037** |

---

## B. 4 个 ADR 决策记录

### ADR-034: SSE realtime dashboard 初版（前置，5/25 已 accepted）
- daemon `/signals/stream` SSE endpoint
- 独立 HTML + Chart.js（脱离 streamlit 解决闪烁）

### ADR-035: 客户端 EventSource → fetch+ReadableStream 重写
**触发**：cc 24h 改 11 commit 修了 6 类根因（Cache-Control / 心跳 / 数值精度 / x 轴翻转 / annotation 加载 / 时序）仍卡

**Team 模式（cc + gemini）钦定 5 个深层根因**：
1. rAF 异步死锁 + 心跳盲区
2. HTTP/1.1 6 连接池耗尽 + TIME_WAIT silent pending
3. Canvas Context Lost
4. V8 JIT 去优化 + GC 绞杀
5. TCP window=0 让 EventSource C++ 层死锁

**修复**：fetch + ReadableStream + AbortController + 5 道防御层（gen 号 / 心跳 / health 探针 / contextlost listener）

### ADR-036: A+B 自回归 input 外推（替换 persistence）
**触发**：川哥反馈"表情切换时预测曲线没明显连续变化"

**cc 实测炸药级证据**：旧 predict_future 10 步 state 最大变化 = 0.0037（LTC 第 2 步就到吸引子）

**Gemini 钦定致命根因**：persistence + LTC ODE = 自治系统迅速收敛到稳态吸引子

**修复**：input[t+1] = input[t] + α·velocity - β·(input[t] - baseline)，velocity *= 0.8

**实测验证**：state 变化 0.0037 → 0.2363（**64×**），entropy 1.83 → 0.02（高置信）

### ADR-037: state 投影对齐 + intent EMA 平滑
**触发**：川哥反馈"虚线位置应体现我当前实时表情/意图/情绪/注意力"

**cc 实测**：同一时刻 frame state=emotional_negative 0.59 vs prediction[0]=transitioning 0.56（**主导 state 翻转**）+ intent 1 秒翻 3 次

**Gemini 钦定**：state 慢变量过度阻尼 vs intent 快变量欠阻尼共存

**修复**：
- predict_future step=0 短路：直接用 frame 缓存的 `_last_state_probs/_last_intent_probs`（aligned_with_frame=True）
- infer 内 intent EMA α=0.3（可 env LNMEM_INTENT_EMA_ALPHA 覆盖）

**实测验证**：step=0 完美 = frame state，step 1-N 仍走 LTC forward + A+B 外推

---

## C. 服务端架构现状（Day 15 末）

```
摄像头 → CaptureWorker (1Hz)
           ↓
       pyfeat MP pool → 18-dim AU vector
           ↓
       LNN inferencer (CfC backbone + state head + intent head)
           ↓ infer()
       _last_state_probs / _last_intent_probs（ADR-037 缓存）
           ↓ EMA intent 平滑
       InferenceResult → SSE /signals/stream
           ↓ 1Hz frame + 1Hz prediction
       客户端 fetch + ReadableStream (ADR-035)
           ↓ rAF 节流（最多 60fps chart.update）
       3× Chart.js stacked area chart (情绪/注意力/意图)
           ↓
       HTML overlay "历史 / 预测" label (DOM, z:10)
```

每 1s 一个完整循环（采集 → 推断 → 推送 → 渲染）。

---

## D. 实测端到端验证

| 验收点 | 结果 |
|---|---|
| dashboard 1Hz 实时刷新 | ✅ rAF 节流后 backlog 永不积压 |
| 虚线两侧曲线值连续 | ✅ step=0 aligned=True 完美对齐 frame |
| intent 不再 1 秒翻 3 次 | ✅ EMA α=0.3 引入 ~2 帧阻尼 |
| LTC 预测有动态轨迹 | ✅ A+B 外推 step=1-N state 变化 64× |
| daemon 长跑无 backlog | ✅ TCP 流控正常，CPU 395%（pyfeat 占大头） |

---

## E. 关键架构教训（已沉淀到 mycc memory）

1. **EventSource 是生产长跑黑盒陷阱**（5 个无声死锁路径，gemini 抓深层根因 cc 24h 想不到）→ fetch+ReadableStream
2. **chartjs-plugin-annotation v3 的 label clip:false 实测仍被裁** → HTML overlay 100% 可靠
3. **修了 N 次同类症状还不根治 = 该重构（不是修第 N+1 个）** → 启动 Team 模式跳出局部最优
4. **Persistence + LTC ODE = 自治系统趋向吸引子（必演变成直线）** → A+B 输入外推
5. **infer vs predict_future 两条算法路径必导致不连续** → state 投影对齐
6. **Team 模式（cc + codex + gemini）抓深层系统问题 ROI 远超 cc 单干**

---

**下一步交接**：详见 `docs/handoff/2026-05-26-realtime-stack-handoff.md`
