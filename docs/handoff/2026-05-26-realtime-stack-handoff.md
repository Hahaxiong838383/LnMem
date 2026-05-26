---
title: 5/26 Session Handoff · 实时栈整改收尾 + M0 demo 准备
date: 2026-05-26 22:00
status: ready
previous_session: 5/26 single-day realtime stack overhaul (commits 891e24a → f889f93)
related_adrs: ADR-034, ADR-035, ADR-036, ADR-037
related_progress: docs/day15-realtime-stack-overhaul-progress.md
---

# 5/26 Session Handoff · LnMem 实时栈整改收尾

## 项目状态对账（M0 demo 5/28-6/3 前 2 天）

### Day 15 完成的工程交付（9 commit + 4 ADR）

✅ **dashboard 卡死问题彻底根治**（fetch+ReadableStream + rAF 节流）
✅ **LTC 预测从"看似真预测、实则 hold 当前"升级为"A+B 自回归外推真有动态"**
✅ **虚线两侧曲线值严格连续**（state 投影对齐 step=0 = frame）
✅ **intent 抖动从 1秒翻3次 降到 2秒响应**（EMA α=0.3）
✅ **mycc memory 沉淀 3 条核心教训**

### 不可妥协 6 项 P-1 达成度（Day 15 末）

| 项 | 达成度 | Day 14→Day 15 变化 |
|---|---|---|
| #1 LNN-FIRST | 95% ✅ | 不变（LTC 核心未动） |
| #2 per-user τ-bias | 70% | 不变 |
| #3 AU 多模态 | 80% | 不变 |
| #4 离线优先 | 90% ✅ | 不变 |
| #5 ODE 可解释 | **88%** ↑ | +13%：LTC ODE 真预测可视化 + 虚线两侧严格连续 |
| #6 时序记忆 | 85% ✅ | 不变 |

**总分**：技术层 ~85% (+2%)，产品层 ~35% (+15%，dashboard 可演示了)，商业化 0%

---

## P0 后续 TODO（按优先级）

### P0.1 - M0 demo 5/28-6/3 准备（最高优先级，剩 2 天）

- [ ] **24h learning curve recording**：daemon 长跑 24h 收 query/feedback 数据
- [ ] **Studio USB camera hold-out 验证**：跨设备真人推断验证
- [ ] **5-10min demo 录屏**：脚本 + 真人演示 + 字幕
- [ ] **dashboard 顶部加 Help expander**：5 步引导（"虚线 = 现在"、"左历史右预测"、"颜色 = 情绪/注意力/意图"等）
- [ ] **README 30秒入口** 更新 dashboard 截图 + Streamlit 链接

### P0.2 - Gemini 钦定但未修的剩余根因（系统级 limitation）

- [ ] **端到端延迟 1-1.5s**（gemini 钦定根因 1）：py-feat 物理 latency
  - 方案 A：换更快 AU detector（MediaPipe / FACE-API.js?）
  - 方案 B：在 UI 上把视频流也延迟 1s 对齐（视觉错觉）
- [ ] **客户端 ts_ns 硬编码 1e9 ns/step vs LTC 真实 timespans 不一致**（gemini 根因 2）
  - daemon `/predict_future` response 返回真实 timespan
  - 客户端按真实秒数画时间轴
- [ ] **state 锁死 emotional_negative**（gemini 根因 3+5）：py-feat 亚洲人脸偏差 + mock VEATIC 训练分布偏差
  - 短期：调 EMA + 重新校准 baseline
  - 长期：重训 + 引入真实人脸数据集（FER2013 / AffectNet / RAF-DB）

### P1 - Phase 3 后续（M1+）

- [ ] **训独立 forecasting head**（DLinear / PatchTST）：1 周训练 + 跟 LTC inferencer 并行 fail-soft
- [ ] **predict_future 增量推送**：不必每帧推完整 10 步，只推变化部分
- [ ] **EMA α 数据驱动调优**：拿真人数据 sweep 找最优 (0.1 / 0.2 / 0.3 / 0.5)
- [ ] **chart.js → uPlot**（如果 dashboard 长跑出现 V8 退化）
- [ ] **dashboard 多用户支持**：不同 persona_idx 切换

### P2 - 架构债

- [ ] cc-AirPlayer 跟 LnMem 共占 CPU（之前 cc 关掉过）→ 长期需要明确资源调度策略
- [ ] daemon 重启时 SSE 连接 reset，浏览器需手动 reload → 加自动重连提示？

---

## Day 15 改动文件清单（核心）

```
poc/src/realtime/lnn_inferencer.py    +209-8  (ADR-036/037 实施)
poc/src/realtime/daemon_main.py       +2-2    (SSE prediction 5s→1s)
docs/realtime/realtime_dashboard.html +320-88 (ADR-035 fetch + HTML overlay + rAF + ADR-037 客户端无改)
docs/decisions/035-fetch-streaming-replace-eventsource.md  (新)
docs/decisions/036-autoregressive-input-extrapolation.md   (新)
docs/decisions/037-state-projection-alignment-intent-ema.md (新)
docs/day15-realtime-stack-overhaul-progress.md             (新)
docs/handoff/2026-05-26-realtime-stack-handoff.md          (新, 本文)
```

---

## 下个 session 接手 prompt（可粘贴）

```
继续 LnMem M0 demo 准备。

上下文：
- 5/26 完成实时栈大改造，9 commit + ADR-034/035/036/037
- dashboard 已修通：fetch+ReadableStream + rAF 节流 + HTML overlay label
- LTC 预测已升级：A+B 外推 + state 投影对齐 + intent EMA
- daemon /signals/stream + /dashboard + /predict_future 全部正常
- 当前 daemon pid 见 pgrep -f 'poc.src.realtime.daemon_main'

请先：
1. 读 docs/day15-realtime-stack-overhaul-progress.md
2. 读 docs/handoff/2026-05-26-realtime-stack-handoff.md（本文档）
3. 跟川哥确认 demo 路线：
   - 录屏 vs 真人现场演示？
   - 演示内容侧重 LNN 架构 / 预测能力 / 个性化哪一面？
   - 时长 5min vs 10min？
4. 按 P0.1 清单推进 demo 准备
```

---

**Day 15 收尾时间**：2026-05-26 22:00 CST
**单日 commit 总数**：9（最多 commit/day 记录）
**Team 模式调用**：3 次 gemini 3.5-flash high + 1 次 codex 5.3 high（codex 输出未落盘）
**川哥反馈轮次**：5 轮（dashboard 卡 → label 遮挡 → 还卡 → 虚线不连续 → 实时性 OK）
