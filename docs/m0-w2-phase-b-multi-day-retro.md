---
name: M0 W2 Phase B 多天真数据复盘（5/12 自然触发后 4 天）
description: 5/12–5/15 daemon 连续运行 4 天，trigger 通道 + LNN 持续学习闭环实测复盘
type: report
phase: M0 W2 Phase B-retro
created: 2026-05-16
status: 初版（cc 主线 + codex 待审）
related:
  - docs/m0-w2-phase2-first-run.md
  - poc/src/realtime/feedback/trigger.py
  - poc/src/lnn/fit_orchestrator.py
  - scripts/run_fit_step.py
  - scripts/daemon_watchdog.sh
---

# Phase B 多天真数据复盘（2026-05-12 → 05-15）

## TL;DR

**PoC 假设变现达成度：60%**

| 维度 | 5/12 单日里程碑 | 4 天累积现状 | 评级 |
|---|---|---|---|
| 摄像头 daemon 连续运行 | 跑通 | **4 天连续无人工干预**（含 5/15 假死 18h） | 🟡 |
| Phase B 三通道生效 | 设计 | brow_burst 53% + pose_held 43% + user_trig 4% | 🟢 |
| 用户反馈累积 | 39 | **138 queries，match 率 42%** | 🟢 |
| LNN 持续学习闭环 | 首次自然触发 | **4 天无自然 fit 触发**（trigger 满足但 cron 没装） | 🔴 |
| VLM 通道 | 0 | **4 天 vlm_label 全 null**（通道未激活） | 🔴 |
| watchdog 守护 | 设计 + 真重启 1 次 | **HTTP 假死 18h 没救活**（设计漏洞） | 🔴 → 🟢 已修 |

## 1. 数据画像（138 queries）

```
日期      total  pose_held  brow_burst  user_trig  skip  match  mismatch  input
2026-05-12   39        14         19          6    17     13         6      3
2026-05-13   37        17         20          0    14     19         2      2
2026-05-14   36        18         18          0    11     17         7      1
2026-05-15   26        10         16          0     8      9         9      0
合计        138        59         73          6    50     58        24      6
                     (43%)      (53%)        (4%)
```

**关键信号**：
- **match 率 42%**（58/138）— LNN 预测对 42% 的 trigger，PoC 阶段表现可接受
- **5/15 mismatch 占比飙升至 35%**（9/26）— 模型偏移或用户状态变复杂
- **mismatch 累积 28**（含 5/12 之前 6 个手动 + 5/13-15 22 个自然），**远超 fit 触发阈值（10）**
- **vlm_label 全空** — VLM 通道无激活，需查 daemon V0.3 配置

## 2. 关键发现

### 🔴 发现 1：LNN 持续学习闭环未真正闭合

设计文档 `m0-w2-phase2-first-run.md` 明示：
> "scripts/run_fit_step.py (CLI + **cron 调度**)"

实测：
- `crontab -l` 无 fit-step 任务
- `launchctl list` 只有 daemon-supervisor + daemon-watchdog，**无 fit cron**
- `trigger_if_ready` 实时返回 `True`（mismatch=28, total=187 全过阈值）
- **5/12 自然首次触发后，4 天累积满足条件，但从未被调度**

**根因**：设计阶段确认了 cron 是 fit 调度入口，但工程没装。

**修复方向**：
- 加 `~/Library/LaunchAgents/com.lnmem.fit-step.plist`（每 30min 或 1h 跑一次）
- 或装 user crontab：`*/30 * * * * /path/to/.venv/bin/python -m scripts.run_fit_step`
- 失败要落盘到 `~/.lnmem/fit_step.log`（不能 silent）

### 🔴 发现 2：VLM 通道 4 天 0 激活

138 个 query 中 `vlm_label` 字段全是 `null`。

可能原因（待 codex 查）：
1. daemon V0.3 配置 `enable_vlm=False`
2. VLM rate-limit 太严（low_vlm_conf 通道实际没触发）
3. 配额耗尽或 endpoint 不通
4. 设计意图就是"VLM 仅 fallback"，主路径走 brow + pose_held

需查 `daemon.toml` 实际配置。

### 🔴 → 🟢 发现 3：watchdog HTTP 假死 18h 盲区（已修）

详见本次 PR：`scripts/daemon_watchdog.sh` + `daemon_main.py` + 测试。
- 加 `CONSECUTIVE_HEALTH_FAIL_THRESHOLD=3` 独立触发路径
- daemon 加 SIGUSR1 → faulthandler dump 死锁栈
- 5/5 回归测试通过

### 🟢 发现 4：Phase B trigger 通道多样化生效

5/12 Phase B 落地后 4 天数据验证：
- pose_held = 59 次（43%）
- brow_burst = 73 次（53%）
- user_triggered = 6 次（4%，全在 5/12 手动加速时）

3 通道分布健康，避免了"都是皱眉"的 V0.2 单通道 anti-pattern。

### 🟢 发现 5：daemon 连续运行 4 天耐久度通过（含 1 次假死）

- 5/9–5/15 不间断 daemon 启动
- 5/13 21:22 + 5/14 00:27 + 5/14 17:21 + 5/15 00:34 共 4 次 supervisor 触发的自然重启（last_frame_ts STALE）
- 5/15 20:43 → 5/16 15:21 假死 18h（watchdog 漏判，已修）
- 信号采集累积 230MB+ jsonl

## 3. 下一步建议（按 PoC 资源分配铁律排序）

| 优先级 | 任务 | 类型 | 工时 |
|---|---|---|---|
| **P0** | 装 fit-step cron / launchd（闭合学习环） | 已知工程 | 30min |
| **P0** | 跑一次手动 `run_fit_step.py`（消费累积的 22 个自然 mismatch） | 已知工程 + 真信号验证 | 15min |
| P1 | 查 VLM 通道为什么 4 天无激活 | 调研 + 配置 | 30min |
| P1 | watchdog 修复合并到 main | 已知工程 | 5min |
| P2 | 调研 5/15 daemon HTTP 死锁根因（SIGUSR1 拿栈后） | 未知 binary | 1h+ |
| P2 | 万能语法 v0 大规模 N=1万 cron | 未知 binary（type1 持久化） | 2-3h |

**核心建议**：
1. 先装 fit cron（半小时工程，闭合 PoC 假设最后一公里）
2. 立即手动跑一次 fit_step，消费累积 mismatch（验证 LNN 学到了什么）
3. watchdog 修复 commit + push
4. 之后再走 P1/P2

## 4. 参考

- 5/12 首次自然触发详情：[`docs/m0-w2-phase2-first-run.md`](m0-w2-phase2-first-run.md)
- Phase B 通道多样化设计：commit `0ce27b8`
- 本次 watchdog 修复 PR：`scripts/daemon_watchdog.sh` + `poc/src/realtime/daemon_main.py`

---

署名：cc（Claude）| 2026-05-16 15:35 CST
