---
title: 2026-05-25 中午 P0-1 守护全装 + Active Learning v2 上线
date: 2026-05-25
session: 12:00-12:35
author: cc + codex (gpt-5.3-codex high) + gemini-3.5-flash (high reasoning)
type: report
---

# 2026-05-25 中午复盘：P0-1 守护全装 + Active Learning v2 上线

> **35 分钟内完成 3 个 commit / 4 个 ADR 决策 / 1 次川哥反转 / 7 个 codex 对抗审 finding 全采纳。**
> 关键转折：watcher 范式从"阈值触发"改"active learning"，弹窗显示理由。

---

## 一、三段核心交付（按时间线）

### 12:08 commit `b649815` · P0-1 launchd 全装 + daemon ckpt watcher + ADR-030 24h 决策

**完成的 8 件事**：
1. 修 `install_launchd_studio.sh` unload 错误识别（codex 钦定）
2. 写备用 8h plist `.alt` 供切换
3. 装 Air 端 2 个 LaunchAgent（sync + pull）
4. rsync + ssh 装 Studio 端 nightly-train LaunchAgent
5. 端到端验收：Air sync (90 条) → Studio --force train (7s) → Air pull → daemon /health intent_head_enabled=true
6. 写 ADR-030 24h 决策（cc/codex/gemini 三方对抗审）
7. 补 daemon `_ckpt_watcher_loop` 60s 周期热加载 thread
8. 实测 ckpt sha8 cdaa4d69→98227c93 自动 reload

**抓到的 5 个 bug**：
- `sync_daemon_to_studio.sh`: `[ -n "$DRY_RUN" ] && echo` + `set -e + pipefail` 导致非 dry-run 必败（隐性 bug 几天没人发现）
- `studio_nightly_train.sh`: git pull 失败崩流程（Studio rsync 部署的 .git 是 Air worktree 路径）
- `studio_nightly_train.sh`: FEEDBACK_MIN=50 统计绝对值会反复训同批数据（应 cursor-based 增量）
- `studio_nightly_train.sh`: symlink 用 ln -sf 非原子（应 tmp+mv）
- daemon: inferencer 有 reload_ckpt 但没人调（接口存在 ≠ 链路通）

### 12:18 commit `a1f1ac8` · 川哥反转 → 8h 调度 + UI list/dict 兼容

**决策反转**：cc 在 12:08 写完 ADR-030 钦定 24h，川哥 12:18 一句话「按照 8 小时间隔执行吧」推翻。

cc 立刻执行：
1. cp 8h.alt → 主 plist
2. plist 加 `LNMEM_FEEDBACK_MIN_INCREMENT=10` + `LNMEM_MIN_HOURS_BETWEEN_TRAINS=6` env（覆盖默认 30/20）
3. rsync 到 Studio + 重装 launchd
4. ssh dry-run 注入 env 验证「新增 0 < 10」← 阈值 10 已生效
5. ADR-030 追加 superseded by 川哥反转 + 回滚方法（5 秒切回 24h）
6. Bonus 修了 UI `intent_probs` list vs dict 兼容 AttributeError（daemon 启用 intent_head 后字段类型变了）

**教训**：cc 不能擅自决策时机/阈值这种"价值取舍"——gemini 的"长期最优"和川哥的"短期 demo 体验"价值排序不同，由人决定。

### 12:35 commit `fec97d6` · notify_watcher v2 active learning + ADR-031

**川哥钦定（5/25 12:25）**：
> "对于弹窗，我希望不是按照设定阈值，而是 LNN 判断需要我反馈时，弹窗让我反馈并附上理由，为什么这次需要我反馈，我的反馈起到什么作用。"

cc 设计 6 触发器 → 派 codex 对抗审 → 5 个 finding + 2 个遗漏全采纳：

**触发器优先级（单选不 OR，codex 钦定）**：
| 优先级 | 触发器 | 信号 |
|--------|--------|------|
| F | 冷启动校准 | ckpt 5min 内换过 + 有不确定信号 |
| A | 意图不确定 | intent_probs top-2 差 < 0.10 持续 3 帧 |
| B | 表情不确定 | phase2 margin<0.30 或 entropy>0.60 持续 3 帧 |
| C | 状态突变 | 两次高置信 top_prob≥0.80 但 state 不同 |
| D | 稀疏类样本 | 此 state 历史 feedback < 5 条 |
| E | 历史标注分歧 | 同 state 历史 ≥ 10 条且主类占比 40-60% |

**文案诚实性范式**（codex 第 4 点钦定）：
```
🤔 LNN 主动求反馈

【为什么这次问你】意图不确定
【依据】连续 3 帧三类概率差 < 0.10
        intent_top2_diff=0.038 phase2_margin=0.486
【你的反馈如何帮我】你告诉我到底是 positive/negative/none，
                    我下次同情境能学会
```

禁止"我觉得"心理化拟人。多触发器并发只展示最高优先级 1 个主因。

**部署**：`com.lnmem.notify-watcher.plist` KeepAlive=true + RunAtLoad=true + Throttle 60s + 全部阈值写 EnvironmentVariables（不依赖 shell env，重启行为一致）。

**修了 1 个 bug**：watcher 第一次启动看到 `last_ckpt_sha=""` 误把 "ckpt 刚换" 触发 F → 修为仅当 last_sha 非空且与当前不同才更新 change_ts。

---

## 二、6 个 LaunchAgent 守护全景

| # | 服务 | 调度 | 用途 |
|---|------|------|------|
| 1 | `com.lnmem.daemon-supervisor` | 已有 | daemon 业务进程拉起 |
| 2 | `com.lnmem.daemon-watchdog` | 已有 | daemon 健康检查 |
| 3 | `com.lnmem.sync-daemon-to-studio` | 3600s | Air → Studio rsync 数据 |
| 4 | `com.lnmem.pull-ckpt-from-studio` | 600s | Air ← Studio ckpt 拉回 |
| 5 | `com.lnmem.notify-watcher` | KeepAlive | active learning 弹窗 |
| 6 | `com.lnmem.studio-nightly-train`（Studio） | **8h (00/08/16)** | 训练 + cursor 双门 |

川哥关机/重登后**全部自动起**，cc 无需手动维护。

---

## 三、端到端真闭环图

```
川哥点反馈 → daemon /intent_feedback → ~/.lnmem/data/intent_feedback/*.jsonl
            ↓ (Air sync LaunchAgent 每 1h)
Studio ~/lnmem-daemon-sync/ → ssh trigger nightly-train (8h: 00/08/16)
            ↓ 双门: 增量≥10 + 距上次≥6h
intent_head 训练 (5-15min on M2 Max) → ckpt + atomic symlink
            ↓ (Air pull LaunchAgent 每 10min)
~/.lnmem/ckpt/intent_head_latest.pt
            ↓ (daemon ckpt_watcher 60s 周期)
inferencer._load_intent_head_ckpt → 1s 软重启线性过渡
            ↓
新 /signals/recent intent_pred = 真预测（不再 None）
            ↓ (notify_watcher v2 active learning)
LNN 真求反馈时弹窗 + 显示理由 → 川哥点 → 循环
```

**链路所有环节均有 LaunchAgent / supervisor 守护**：崩了自动起，无单点故障。

---

## 四、5 个 mycc memory 教训（已沉淀）

1. **`feedback_bash_short_circuit_pipefail_trap.md`** — `[ -n "$X" ] && echo` 在 set -e + pipefail 下短路必退
2. **`feedback_launchd_env_vs_shell_env.md`** — 长跑守护进程 env 必须写 plist
3. **`feedback_active_learning_trigger_design.md`** — 主动反馈 7 大坑（codex 钦定）
4. **`feedback_ckpt_hot_reload_pattern.md`** — ML daemon ckpt 热加载范式（接口存在 ≠ 链路通）
5. **`feedback_rsync_deployed_git_repo_pitfall.md`** — rsync .git worktree 部署后必崩

---

## 五、35 分钟内的 Team 协作记录

| 时间 | cc | codex | gemini | 川哥 |
|------|----|----|----|----|
| 12:00 | 读 ROADMAP + 列 TODO | | | |
| 12:02 | 派 codex 评审 launchd 方案 | gpt-5.3-codex high evaluating | gemini-3.5-flash high evaluating | 决策 8h vs 24h |
| 12:05 | | 输出 5 finding + 8h plist XML | 输出强反对 8h（灾难性遗忘）| |
| 12:08 | 综合三方 → 采纳 24h | | | |
| 12:10 | 装 launchd + 端到端验收 | | | |
| 12:15 | 实测 sha8 变化 + commit b649815 | | | |
| 12:18 | | | | **决策反转 → 8h** |
| 12:19 | 切 plist + 阈值 10/6h + push a1f1ac8 | | | |
| 12:25 | | | | **指令 active learning + 理由文案** |
| 12:27 | 设计 6 触发器 + 派 codex 对抗审 | gpt-5.3-codex high evaluating | | |
| 12:30 | | 输出 5 finding + 2 遗漏（信号质量 / 连续性）| | |
| 12:32 | 写 v2 watcher（200 行 bash + Python）+ plist | | | |
| 12:34 | 修首启 bug + 实测触发 F→C + push fec97d6 | | | |
| 12:35 | 写 ADR-031 + 5 个 mycc memory + 本报告 | | | |

**35 分钟 / 3 commit / 2 ADR / 1 范式转变 / 7 codex finding 全采纳 / 5 memory 沉淀 / 1 端到端验证**

Team 模式效率峰值。

---

## 六、关键经验萃取

### 经验 1：cc 不能擅自决策"价值取舍"

12:08 cc 综合 codex+gemini 选了 24h（gemini 论据扎实）。12:18 川哥一句话反转。

cc 必须意识到：技术评审给"长期最优"，但**用户/产品方有自己的价值排序**（demo 体验 > 长期模型稳定性）。技术决策应**呈现选项 + 利弊 + 推荐**，但**不要替用户拍板**。

ADR-030 的好处是反转成本极低——5 秒切 plist，因为决策记录完整。

### 经验 2：codex 对抗审在"规则设计"任务上 ROI 最高

写代码用 cc + 简单 codex 足够。但 **active learning 触发规则**这种"看似简单但藏 7 个坑"的设计任务，必须派 codex 对抗审。

cc 第一版 6 触发器**全部命中陷阱**：
- 高置信无定义
- OR 触发重复
- 小样本噪声
- 长尾尖刺
- 文案伪解释

codex 一轮 5 finding + 主动指出 2 个遗漏（信号质量异常 / 时间连续性），全部采纳后整改进 v2。

**规则**：任何"基于多信号 + 多条件 + 时间逻辑"的设计，写代码前必须派 codex 当对手挑刺。

### 经验 3：接口存在 ≠ 链路通

LnMem 5/24 ADR-029 写了 Sprint 2 pipeline：训练 → ckpt → rsync → 热重启。`inferencer.reload_ckpt()` 有，rsync 有，但 daemon 没有 file watcher 调 reload。

整条链路看起来通，实际断在 daemon 端。直到 5/25 P0-1 端到端验证才发现。

**铁律**：任何"自动更新"系统必须有 file watcher / event listener / poll loop 守护。光暴露接口不算实现。

### 经验 4：长跑守护进程必须用 plist 持久化 env

watcher v1 用了 cc 之前 session 手动 export 的 `NOTIFY_THRESHOLD=0.7 COOLDOWN=60s`，跑了 2h45min 直到川哥发现弹窗频率不对。重启后会丢失这些 env，恢复默认 0.85/180s。

v2 全部阈值写 plist 的 EnvironmentVariables，重启行为一致。

**规则**：任何超过 1 小时不停的进程，env 必须落 plist。临时 export 只用于 5 分钟内测试。

### 经验 5：文案诚实性是 active learning 关键设计

codex 第 4 点：watcher 看到的是字段值，**不是 LNN 内部"主观意图"**。把规则命中写成"我觉得困惑" = LLM 风格的伪解释 = 失真。

修复：
- 禁止心理化拟人
- 必须可追溯到字段+数值
- 多触发器并发只展示 1 个最高优先级

这条经验可推广到任何 AI agent 给人类的解释：**字段+数值是诚实，自然语言形容词是表演**。

---

## 七、给下个 AI session 的接手清单

**1 分钟掌握当前**：
```bash
# 看 6 个 LaunchAgent 状态
launchctl list | grep com.lnmem

# 看 daemon 健康 + intent_head 状态
curl -s http://127.0.0.1:9876/health | python3 -m json.tool

# 看反馈累积
wc -l ~/.lnmem/data/intent_feedback/*.jsonl

# 看 v2 watcher 实时触发
tail -f ~/.lnmem/logs/notify_watcher/launchd.out.log
```

**3 个关键 ADR 必读**：
- ADR-028 LNN 角色重定义（v0.1 战略转折）
- ADR-030 Studio 训练 8h 调度（5/25 决策）
- ADR-031 主动反馈 active learning 范式（5/25 决策）

**下一波等待**：
- 16:00 Studio 8h 自动训第一次（验证 launchd 链路）
- 川哥继续点 v2 弹窗（验证 active learning 触发逻辑）
- 1 周后 retro：catastrophic forgetting 监测 + 触发器分布

---

**署名**：cc（Claude）+ codex (gpt-5.3-codex high) + gemini-3.5-flash (high reasoning) | 2026-05-25 12:35
