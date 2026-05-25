# ADR-034: 实时大屏方案 SSE + 独立 HTML（脱离 streamlit 重画）

**状态**：accepted (Team 模式 cc + codex 并行实施)
**日期**：2026-05-25
**决策者**：川哥（5/25 17:18 钦定方案 E "独立 HTML + WebSocket"，cc 优化为 SSE）
**触发**：streamlit altair_chart 重画 dom 闪烁问题，streamlit-autorefresh + @st.fragment 都治标不治本

---

## 背景

川哥反馈 UI 曲线持续闪烁。根因：streamlit altair_chart 每次 rerun 都要 dom 清空-重画，与刷新机制（time.sleep / fragment / autorefresh）都无关——这是 streamlit 框架本质行为。

要彻底无闪 = **必须脱离 streamlit 重画机制**。

川哥钦定方案 E：独立 HTML + WebSocket。cc 优化为 **SSE**（更简单，单向推送够用）。

## 决策

### 双端共存架构

| 端 | 用途 | 框架 | 端口 |
|----|------|------|------|
| **Streamlit (8520)** | 管理面板 | streamlit | 反馈队列 / Studio panel / 注解 / 配置 |
| **独立 HTML 实时大屏（新）** | 实时演化曲线 | 纯 HTML + Chart.js CDN | http://127.0.0.1:9876/dashboard |

两端**并存不冲突**。Streamlit 适合"操作 / 历史 / 配置"，HTML 大屏适合"看实时 / 录 demo / 投屏"。

### SSE 而非 WebSocket（cc 优化）

**为什么 SSE**：
- daemon 现有 `ThreadingHTTPServer` 直接支持，**不需要装 `websockets` 包**
- 浏览器原生 `EventSource` API（IE 不支持，Chrome/Safari/Firefox 全支持）
- 单向 server → client，正好符合需求（无需双向通信）
- 自动重连（client 断线后自动 retry）
- HTTP 基础，企业代理 / VPN 兼容性好

**WebSocket 不需要的复杂性**：握手协议 / 双向消息路由 / 心跳维持。

### Endpoint Schema

```
GET /signals/stream
Headers:
  Accept: text/event-stream
  Cache-Control: no-cache

Response (持续推送):
Content-Type: text/event-stream

data: {"type":"frame","ts_ns":...,"lnn":{...}}\n\n
data: {"type":"frame","ts_ns":...,"lnn":{...}}\n\n
data: {"type":"prediction","ts_ns":...,"predictions":[...]}\n\n
data: {"type":"frame",...}\n\n
```

**推送频率**：
- type=frame: 每 200ms 一次（5Hz，匹配 daemon 1fps 推断节奏，每帧推 + 重复推保活）
- type=prediction: 每 5s 一次（LTC predict_future 调用 + 推送）
- heartbeat: 每 30s 推 `: heartbeat\n\n` 防 proxy 超时

### HTML 大屏 UI

`docs/realtime/realtime_dashboard.html`（自包含单文件）：
- 3 个图（与现 streamlit 对齐）：
  1. 🎯 意图概率（3 类 stacked area）
  2. 😊 情绪（3 类）
  3. 🎯 注意力（5 类）
- 右 1/3 预测段，红虚线"现在"，黄色背景带
- Chart.js 4.x via CDN（jsdelivr）
- 增量更新：`chart.data.datasets[i].data.push({x,y}); chart.update('none')`
- 滑窗：只保留最近 N 帧 (N=300)，超出 shift()
- 60fps 平滑（requestAnimationFrame）

### 数据流

```
daemon LNN 算出新 frame
   ↓ (signal_pipeline.py)
daemon SSE handler: 拉最近 N 帧 + LTC 预测 → 推送 JSON
   ↓ (HTTP /signals/stream)
浏览器 EventSource onmessage
   ↓
JavaScript updateCharts(data): chart.data.push + chart.update('none')
   ↓
Chart.js 增量重绘（dom 不清空，平滑过渡）
```

## 选项与对比

| 选项 | 复杂度 | 工程量 | 优势 | 劣势 |
|------|--------|--------|------|------|
| WebSocket + websockets 包 | 高 | 中 | 双向通信 | 装包 / 握手协议 |
| **SSE + 纯 HTTP** ⭐ | 低 | 小 | 不装包 / 自动重连 | 单向（够用）|
| Long polling | 中 | 中 | 兼容老浏览器 | 性能差 |

## 实施分工

| 任务 | 负责 | 工时 |
|------|------|------|
| WS-1 ADR-034（本文档） | cc | 30min |
| WS-2 daemon SSE endpoint `/signals/stream` + HTML 路由 `/dashboard` | **codex** (gpt-5.3-codex high) | 1h |
| WS-3 `realtime_dashboard.html` 单文件 (Chart.js) | cc | 1.5h |
| WS-4 端到端验收 + commit | cc | 30min |

并行：WS-2 / WS-3 同时跑。

## 回退策略

- Level 1：HTML 大屏出 bug → 川哥关浏览器 tab，继续用 Streamlit
- Level 2：daemon SSE endpoint 关：env `LNMEM_DISABLE_SSE=1` + 重启
- Level 3：完整 revert: `git revert <commit-hash>`

## 验收标准

- [ ] http://127.0.0.1:9876/dashboard 打开见 3 个空 chart
- [ ] WebSocket 连接成功（开发者工具 Network → eventsource 200）
- [ ] 5Hz 平滑更新（点开数据点能看到 ts 递增 200ms）
- [ ] 关闭 tab 重开自动重连
- [ ] 与 Streamlit (8520) 并行打开两端都工作
- [ ] 浏览器 CPU < 5%（chart.update('none') 高效）

---

**关联 ADR**：ADR-033 LTC ODE 真预测（数据源）
**关联文件（实施时填）**：
- `poc/src/realtime/daemon_main.py` (codex 加 SSE endpoint)
- `docs/realtime/realtime_dashboard.html` (cc 写大屏)
