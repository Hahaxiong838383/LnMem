# ADR-035: 客户端 SSE 层 EventSource → fetch + ReadableStream（方案 B）

**状态**：accepted
**日期**：2026-05-26
**决策者**：川哥（钦定方案 B "fetch+ReadableStream 重写"）
**触发**：3 条曲线卡死 11 commit 修了 6 类根因仍不根治
**关联**：ADR-034（SSE 实时大屏初版）

---

## 背景：为什么 EventSource 不靠谱

ADR-034 上线后，川哥反复反馈"3 条曲线卡住不更新，最后时间停在 4:52:43 PM"。cc 24h 内修了 11 个 commit、6 类根因：

| Commit | 修了什么 | 结果 |
|---|---|---|
| 57592d0 | daemon `/dashboard` Cache-Control no-cache | ✅ 但仍卡 |
| 58da68a | safeNsToMs 防 JS Number 2^53 精度坍塌 | ✅ 但仍卡 |
| 4c77830 | predDurationMs ≤ 0 让 x 轴翻转 | ✅ 但仍卡 |
| 58da68a | `_lastMessageTime` 移到 try 末尾防心跳假活 | ✅ 但仍卡 |
| ab91069 | try/catch 包 annotation/chart.update | ✅ 但仍卡 |
| aa7deee | annotation plugin 全局名兼容 | ✅ 但仍卡 |

**川哥反馈："修了那么多还是卡，team 模式整体看一下有没靠谱方案"**

## Team 模式诊断（cc + codex + gemini-3.5-flash high thinking）

### 已验证事实
- daemon SSE 推送 **完全正常**（curl 8s 拿到 12 个 frame，ts_ns 持续递增）
- `/dashboard` Cache-Control header 已生效
- daemon 进程 39 min, 377% CPU, RSS 518MB（CPU 高但服务稳）
- **当前 0 个浏览器 ESTABLISHED 连接**（只剩 14 个 TIME_WAIT）

### Gemini 给出的 5 个深层无声死锁原因

1. **rAF 异步死锁 + 心跳盲区**：`chart.update()` 异步走 requestAnimationFrame；高 CPU 下 rAF 可能被静默丢弃；`_lastMessageTime` 仍更新 → **心跳永远不重连** = 完美无声死锁
2. **HTTP/1.1 6 连接池耗尽**：`EventSource.close()` 进 TIME_WAIT 不立即释放；重连几次池满后 `new EventSource()` 进 Pending 队列**不报错不超时**
3. **Canvas Context Lost**：高内存压力下 OS 回收显存，drawing call 变 no-op
4. **V8 JIT 去优化 + GC 绞杀**：`lastPredTs=undefined`（cc 本轮发现的真 bug，未声明全局）让数据形状改变触发多态去优化（10-100× 慢）
5. **TCP Window 锁死**：`ThreadingHTTPServer` 无流控，客户端 GC 暂停几十 ms 让接收 buffer 满 → TCP window=0 → EventSource C++ 层死锁

### cc 本轮新发现 + 已修的真 bug
- `setNowAnnotation` 第 392 行 `xMax: lastPredTs` —— **未声明变量**（非 strict）
- 改成 `xMax: xMaxMs`（同函数里已算好的对称值）

## 决策：方案 B（fetch + ReadableStream 自解 SSE 帧）

### 选项对比

| 方案 | 稳定性 | 工时 | 根治网络死锁 | 根治渲染卡死 |
|---|:---:|:---:|:---:|:---:|
| A. 继续打补丁 | ⭐⭐⭐ | 2-4h | ❌ | ❌ |
| **B. fetch + ReadableStream** ⭐ | ⭐⭐⭐⭐ | 2-3h | ✅ | ❌ |
| C. WebSocket（含后端） | ⭐⭐⭐⭐⭐ | 8-12h | ✅ | ❌ |
| D. Chart.js → uPlot | ⭐⭐⭐⭐⭐ | 12-16h | ❌ | ✅ |

### 为什么选 B 不选 C/D

- **服务端零改动**（SSE 协议帧不变）—— 节省 daemon 部署/守护工程
- **fetch + AbortController** 完全掌控 reader 生命周期，绕过 EventSource 黑盒
- **2-3h 工时**符合 M0 demo 6/3 时间窗
- 渲染层（Chart.js）暂不换：3 个图量小，先看网络层修复效果；如仍卡再上 uPlot

## 实施细节

### 客户端核心变化

**旧版**（删）：
```js
const es = new EventSource(SSE_URL);
es.onmessage = (e) => { /* 处理 */ };
es.onerror = (err) => { /* 自动重连黑盒 */ };
```

**新版**（fetch 流式自解）：
```js
async function connect() {
  const myGen = ++_connectGen;  // 防旧 reader 跟新 reader 抢
  const ctrl = new AbortController();
  const resp = await fetch(SSE_URL, {
    headers: { Accept: 'text/event-stream' },
    signal: ctrl.signal,
    cache: 'no-store',  // 强制新 TCP 连接，避开 6 池上限
  });
  const reader = resp.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  while (myGen === _connectGen) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // 按 \n\n 切 SSE 帧，提取 "data: " 行 → processFrame()
  }
  // 自动重连
  if (myGen === _connectGen) setTimeout(connect, 1000);
}
```

### 4 道防御层（堆叠）
1. **AbortController 主动 abort**：释放底层 socket 绕过 TIME_WAIT
2. **_connectGen 单调递增**：旧 reader 检测 gen 失配立即退出（防泄漏）
3. **5s 心跳监控**：>10s 没数据 → 主动 abort + 重连
4. **30s `/health` 探针**：独立 SSE 之外验证 daemon 真活
5. **Canvas contextlost 监听**：触发即 reload 页面（gemini 钦定根因 3）

## 回退策略

- Level 1：客户端 fetch 出 bug → revert 这个 commit，旧 EventSource 代码 git 留存
- Level 2：完全弃用 → 川哥用 streamlit 8520 看曲线（兜底永远存在）

## 验收标准

- [ ] Cmd+Shift+R 后浏览器开 dashboard，✅ 标签变 "已连接 (fetch v2)"
- [ ] 30s 持续观察，3 条曲线时间戳每秒前进（不卡）
- [ ] DevTools Network → /signals/stream 一直是 (pending) 状态，下行流量持续
- [ ] /health 30s 一次成功（无 ❌ 探针失败 log）
- [ ] 关 tab 重开自动重连 OK
- [ ] 浏览器后台 5 min 回前台立即重连 OK

## 关联文件

- 修改：`docs/realtime/realtime_dashboard.html` (488-685 行，整段 SSE 客户端重写)
- 不动：`poc/src/realtime/daemon_main.py`（服务端零改动）
- 审计报告：cc + gemini 3.5-flash 高 thinking + codex 5.3-codex high 三引擎对照

## 复盘教训（写给未来 cc）

1. **EventSource 在生产长跑场景是黑盒陷阱**：6 连接上限 + 无主动超时 + 浏览器自动重连无 hook
2. **rAF + 同步心跳 = 假活死锁**：UI 渲染异步管线必须有独立健康探针（fetch /health）
3. **修了 6 个 bug 还不根治 → 该重构而不是再修第 7 个**
4. **Team 模式 (cc+codex+gemini) 抓深层架构问题比 cc 自调更准** — 这次 gemini 5 个深层原因 cc 全没想到
