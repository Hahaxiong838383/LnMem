# LnMem HANDOFF — 2026-05-10 21:38 暂停 → 2026-05-11 主 session 继续

> 川哥 21:38 CST 暂停 /loop dynamic，明天主 session 继续。本文件供下一个 cc session 接手。

---

## 1. 一键续接（明天主 session 第一步）

```bash
cd /Users/rayman.chen/mycc/2-Projects/LnMem-poc
git fetch origin
git reset --hard origin/poc/lnn-validation   # 拿 Day 14 全部 16 commit
git log --oneline -5                          # 应看到 90471af 在 HEAD
```

**注意**：cc 14 天工作 worktree 在 `/tmp/lnmem-fix/LnMem`（macOS /tmp 重启清空），但**所有 commit 已 push 到 origin/poc/lnn-validation**，川哥 worktree 拉取即同步。

---

## 2. 当前状态（2026-05-10 21:38 暂停时刻）

### 工程交付
- **PoC 三道门完结** ✅：cc → codex (12 finding 全修) → 川哥（19:30 方案 C 同意）
- **M0 W1 P0-4 watchdog**：5 文件交付 + 川哥 19:40 install + 20:50 首次实战 WARN 触发 + 1.5h+ launchctl 后台稳定
- **mycc PR #5**：https://github.com/Hahaxiong838383/mycc/pull/5（PoC 完结里程碑事件，待川哥 merge）

### Runtime 状态
- daemon V0.2 PID **15934** 健康（uptime 9.52h，RSS 1.38GB ✅）
- watchdog PID **55269**（launchctl com.lnmem.daemon-watchdog，全 DEBUG 一次 WARN 自恢复）
- venv `.venv-poc-fresh`
- 最新 ckpt: `poc/data/checkpoints/day12_with_self/checkpoint_day12.pt`
- 最新 commit: `90471af` poc/lnn-validation

### 反馈累积（21:34 校准）
- **19 总** = match×6 / mismatch×1 / skip×10 / input×2
- mismatch 仍 1（5+ 天 0 增长 = 结构性短缺铁证）
- skip 占 53%（trigger 设计偏 low_vlm_conf 路径，川哥对此主要 skip）

---

## 3. 川哥未答的问题（明天主 session 优先）

**21:00 CST 川哥问**："每次触发询问我，都是提示皱眉 1.0，为什么都一样？这个过程 LNN 有自学习么？"

**cc 答**（详见 conversation 21:00-21:05）：
1. **trigger 阈值是 brow_furrow > 0.85**（[trigger.py:27](poc/src/realtime/feedback/trigger.py:27)），所以触发出来的本来就是高皱眉，分布天然集中在 0.9-1.0
2. **18 次 trigger 实际两类**：low_vlm_conf 11 次（brow=0，川哥 skip 居多没在意）+ brow_burst 7 次（brow≥0.85，川哥认真回应）
3. **当前 LNN 没在跑自学习**：`grep online_trainer in poc/src/realtime/` 零命中，daemon V0.2 没接 online_trainer，反馈只落盘没读训练
4. **mismatch=1 也卡 ≥10 启动门槛**：即使接入了 trigger_if_ready 也不会触发 fit_step

**cc 反问**："要我现在动手起 daemon V0.3 P0-1 设计文档吗？（user_triggered_mismatch channel + trigger 阈值平衡）"

**川哥未答** → 暂停。明天主 session 从这个问题继续。

### 三种回应分支

| 川哥决定 | cc 行动 |
|---------|---------|
| **启 V0.3 P0-1** | 写 `docs/m0-w2-p0-1-daemon-v03-design.md`（user_triggered_mismatch channel）→ 派 codex 对抗审 → 川哥签 → 实现 |
| **先不启 / 等 24h watchdog 数据** | cc 主线低活跃，等 watchdog 跑稳 12-24h 后启；同时整理 PoC 16 memory 写 lessons-learned 综合 memory |
| **想其他事**（Demo / V2 验证 / 别的方向） | 按指示分支 |

---

## 4. M0 W1 待办（已 cc 自交付 P0-4，剩余川哥配合）

| 任务 | 状态 | 谁做 |
|------|------|------|
| watchdog 设计 + 脚本 + plist + --once 实测 | ✅ done (eac8a68) | cc |
| 川哥 install plist | ✅ done (19:40 PID 55269) | 川哥 |
| watchdog 24h 长跑监控 | 进行中（21:38 已跑 1.5h+，无误报） | 被动等 |
| supervisor 脚本（检测 RESTART_NEEDED + 重启 daemon） | ⏳ 待启 | cc 主线（需川哥 daemon 启动方式确认） |
| Demo 录屏 5-10min | ⏳ 待排期（M0 W1 第一周内） | 川哥 + cc |
| daemon V0.3 P0-1（user_triggered_mismatch） | ⏳ 待川哥决定启动时机 | cc → codex → 川哥 |
| feedback v2 P0-2（missing mask + (sid, fid) 强制） | ⏳ 待 V0.3 后 | cc → codex → 川哥 |
| vboss_self 真实反馈微调 P0-3 | ⏳ 待累积 ≥10 mismatch | cc 实现 |

---

## 5. 不可妥协约束（明天必须遵守）

1. **LNN-FIRST P-1**：所有改动走 LTC ODE forward + tau_drift（不准 fallback 到非 LNN 路径）
2. **评估同构**（cognitive.md §5）：AI 评估自己必须用产品本能（多模态信号），不准用问卷/弹窗
3. **BRO-PDCA O 阶段强制**：动手前先对齐目标 + 约束（缺 O2 是常见执行偏差根因）
4. **三道门**：每个 P0 模块都要走 cc 实现 → codex 对抗审 → 川哥签
5. **真实数据 < 25h 已不可达**：M0 W2 重新规划数据采集节奏
6. **不准 fallback 到合成数据微调**

## 6. 不准做（明天必须遵守）

- 不准动 `.venv-poc-fresh` 之外的 venv
- 不准从 cc Bash 工具启 daemon（macOS TCC 授权链限制）
- 不准把"反馈 < 10 mismatch"当作 Phase 2 启动条件
- 不准 fallback 到合成数据微调
- 不准在川哥未审 P0-X 设计前自启实现
- **不准主动重启 daemon**（TCC 限制 + 川哥未指示）

---

## 7. 关键路径速查

```
LnMem repo (origin):  Hahaxiong838383/LnMem
LnMem repo (远端):    git@github.com:Hahaxiong838383/LnMem.git
当前分支:             poc/lnn-validation
HEAD:                 90471af
主 session worktree:  /Users/rayman.chen/mycc/2-Projects/LnMem-poc
cc 14天 worktree:     /tmp/lnmem-fix/LnMem  ⚠️ 重启即丢，仅作 push 通道

mycc repo (origin):   Hahaxiong838383/mycc
当前分支:             main
PR #5:                log/lnmem-poc-completion-m0w1-start

daemon 启动方式:      川哥常用终端（TCC 授权链）
daemon PID:           15934 (uptime 9.52h)
watchdog PID:         55269 (launchctl com.lnmem.daemon-watchdog)

venv:                 .venv-poc-fresh
最新 ckpt:            poc/data/checkpoints/day12_with_self/checkpoint_day12.pt
反馈日志:             ~/.lnmem/feedback/queries_2026-05-*.jsonl
watchdog 日志:        ~/.lnmem/watchdog.log
```

## 8. 关键 mycc memory 索引（明天 cc 必读）

- `project_lnmem.md` — 项目主索引
- `feedback_lnmem_dev_harness.md` — BRO-PDCA × 目标驱动验收
- `feedback_lnmem_cognitive_spec.md` — 认知能力规范同构原则
- `feedback_lnmem_real_data_baseline.md` — vboss_001 ≠ 老板真实测评
- `feedback_lnmem_realtime_capture_pinned.md` — mediapipe 钉版本 + macOS TCC
- `feedback_lnmem_multi_ai_entry_pattern.md` — 多 AI 工具入口索引
- `reference_lnmem_vlm_api.md` — VLM API + Gemini 代理
- `feedback_codex_runtime_vs_exec_direct.md` — codex 直调强制（runtime 后台禁用）
- `feedback_global_view_before_local_fix.md` — 5min 全局扫描 SOP

---

## 9. 暂停时刻签名

**cc** | Claude Opus 4.7 (1M context) | 2026-05-10 21:38 CST

PoC 14 天交付完结 + M0 W1 P0-4 落地。明天主 session 接手时第一动作：拉 origin + 答 V0.3 启动问题。
