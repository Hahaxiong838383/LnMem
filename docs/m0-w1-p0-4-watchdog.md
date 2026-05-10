---
name: M0 W1 P0-4 daemon 跨日稳定性 + RSS watchdog 设计
description: PoC Day 14 RSS 警报触发的 M0 W1 第一任务（cc 主线部分）
type: design-doc
created: 2026-05-10
phase: M0 W1
prereq: 川哥 PoC 三道门签字 ✅（2026-05-10 19:30 CST）
---

# M0 W1 P0-4 daemon 跨日稳定性 + RSS watchdog 设计

> **背景**：PoC Day 14 Loop 11 (19:29 CST) RSS 30min 内从 1.28GB 暴涨到 **2.19GB（+71%）突破 2GB 警戒**，同时 triggered 暴涨 +41，揭示 daemon V0.2 在持续运行 7+ 小时后存在内存压力问题。M0 W1 第一任务：watchdog + 24h 长跑稳定性。
>
> **cc 主线可做**（无需川哥配合）：watchdog 脚本设计 + launchd plist 模板 + log rotation 配置；**需川哥配合**：launchd 实际 install/load（TCC 授权 + 川哥常用终端启动）。

---

## A. 问题陈述

### A.1 PoC 期间观察到的 RSS 模式

11 时点（5/10 12:55 - 19:29，6.5h）：

```
12:55  1.86GB ← daemon 起步早期高点（py-feat 模型加载 + 初始化）
14:32  1.42GB ← 回落
15:43  1.10GB ← 最低点（GC 后稳态）
16:24  1.16GB
17:11  1.10GB
17:46  1.41GB ← Loop 8 spike
18:23  1.32GB ← Loop 9 回落
18:57  1.28GB ← Loop 10 持续回落
🚨 19:29  2.19GB ← Loop 11 30min 内 +71% 暴涨突破警戒
```

**两种行为并存**：
1. **非单调波动**（1.10-1.86GB 多次往返）→ frame buffer / VLM async cache 周期 GC
2. **突发 spike**（30min +71%）→ **真实 leak 或队列积压未排查**

### A.2 异常事件相关性

Loop 11 同时段：
- triggered: 78 → 119（+41 异常密集）
- rate_blocked: 66 → 107（+41 同步）
- queried: 12 → 12（0 新弹窗到用户，限速吃满）
- 反馈累积: 14 不变

**推测**：daemon 进入"高触发 + 高限速 + 0 user-actionable"状态，**VLM async 任务在 rate_blocked 路径下未释放**？或 frame buffer 在持续触发下不 GC？

---

## B. 设计目标

### B.1 验收 DoD

- [ ] daemon 连续运行 ≥ 72h，0 ERROR / 0 OOM / 0 数据丢失
- [ ] RSS 24h 后 < 1.5GB（稳态）
- [ ] RSS > 2GB 自动告警 + watchdog 重启
- [ ] launchd daemon 模式（系统重启自动恢复）
- [ ] log rotation（避免日志爆磁盘）
- [ ] 重启后 session_id 持久化（不破坏 frame_id 跨 session 索引）

### B.2 不破坏

- 不破坏 PoC 已建工程链（OnlineLoRAAdapter / feedback_to_batch / signals_loader）
- 不破坏 daemon V0.2 反馈 jsonl / signals jsonl 格式（V0.3 改 trigger 才动）
- 不破坏川哥常用终端 TCC 授权链

---

## C. 实施路径（cc 主线 vs 川哥配合）

### C.1 cc 主线可做（本 turn 起）

#### scripts/daemon_watchdog.sh
监控 daemon PID + RSS + 自动告警

```bash
#!/usr/bin/env bash
# LnMem daemon watchdog
# 用法：./daemon_watchdog.sh （前台监控）
#      或 launchd 后台运行（推荐，配合 com.lnmem.daemon.plist）
#
# 行为：
# 1. 每 60s 检查 daemon PID 存活
# 2. 检查 RSS：> 2GB 警告日志 / > 3GB 触发重启 hook
# 3. 检查 last_frame_ts：> 5min 无新帧 → daemon 假死，重启
# 4. log rotate：当 ~/.lnmem/data/signals_*.jsonl > 500MB 触发 archive
```

#### scripts/com.lnmem.daemon.plist
launchd 模板，需川哥 `launchctl load` 启动

#### scripts/log_rotation.sh
signals/feedback/skipped jsonl 按日期 + 大小 archive

### C.2 需川哥配合（M0 W1 第一周内）

- launchd plist install + load（TCC 授权 + 川哥常用终端）
- daemon V0.2 当前 PID 15934 重启（释放当前 2.19GB RSS）：
  ```bash
  # 川哥从他常用终端：
  pkill -f "poc.src.realtime.daemon_main"
  # 等 5s
  cd /tmp/lnmem-fix/LnMem
  source .venv-poc-fresh/bin/activate
  python -m poc.src.realtime.daemon_main &
  ```

### C.3 M0 W1 不做（推迟 W2）

- daemon V0.3 trigger 阈值平衡（user_triggered_mismatch channel）→ M0 W1 P0-1 单独排期
- feedback v2 missing mask → M0 W1 P0-2

---

## D. 验证方案

### D.1 单元测试（cc 主线）

- watchdog 脚本：mock daemon PID / 模拟 RSS 突破阈值 / 验证告警触发
- log_rotation：模拟 jsonl > 500MB / 验证 archive 命名 + 旧文件保留

### D.2 集成测试（需川哥环境）

- launchd 模式启动 daemon → kill -9 daemon → watchdog 检测 + 自动重启
- daemon 主动模拟内存压力（注入 fake VLM cache 持续涨）→ watchdog 触发 RSS 阈值告警

### D.3 24h 长跑（M0 W1 第一周末）

- launchd 模式 daemon 运行 72h
- watchdog 日志监控 RSS 时序
- 期望：RSS 稳态 < 1.5GB / 0 OOM / 0 daemon crash

---

## E. 风险

1. **TCC 限制**：cc 测试 watchdog 脚本时若涉及摄像头需川哥 TCC 终端启动（与 PoC 一致）
2. **launchd plist 调试成本**：macOS launchd 错误信息含糊（log 在 ~/Library/Logs/com.apple.xpc.launchd/），需川哥配合迭代
3. **重启对反馈累积影响**：当前 daemon RSS 2.19GB 重启后丢失内存中的 frame buffer / VLM cache，但 jsonl 已落盘，不影响数据完整性

---

## F. 时间盒

| 阶段 | 工时 |
|------|------|
| C.1 cc 主线脚本（watchdog + plist + rotation） | 2-3h |
| C.2 川哥配合 install/load | 30min（一次到位）|
| D.1 单元测试 | 2-3h |
| D.2 集成测试 | 1-2h（需川哥环境）|
| D.3 24h 长跑 | 72h（被动等待）|
| **总计** | **5-8h cc + 24-72h 长跑** |

预计 M0 W1 第 1-2 天完成 C+D；第 3-5 天长跑观察。
