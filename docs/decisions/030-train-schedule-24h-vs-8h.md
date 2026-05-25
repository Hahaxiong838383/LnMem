# ADR-030: Studio nightly-train 调度选 24h 而非 8h/4h

**状态**：superseded by 川哥 5/25 12:18 决策 → **改 8h（00/08/16）+ 增量门 10 + 时间门 6h**
**日期**：2026-05-25（初次决策 12:08，川哥反转 12:18）
**决策者**：川哥（基于 cc/codex/gemini 三方评审 + 川哥最终拍板）
**触发**：川哥要求 "P0-3 的 studio 训练，能不能按照每 4 小时或者 8 小时一次"，cc 派 codex + gemini-3.5-flash 对抗审

---

## 背景

ADR-029 Sprint 2 初版 plist 设 Studio 每天 03:00 训一次 intent_head。M0 demo 临近，川哥希望加速反馈循环到 4h/8h，让"今天点的反馈今天生效"。

但反馈数据现实：
- 川哥白天点击节奏，1-2 天积累 ~10-30 条 intent_feedback
- 单次训练样本要 ≥50 才有 ML 意义（3 类均分 ≥17/类）
- 高频小数据集 BPTT 训练有灾难性遗忘风险

## 决策

**Studio nightly-train 维持每天 03:00 一次**（24h 周期），但加双门控：
1. 自上次成功训练以来**新增反馈 ≥ 30 条**（不是绝对总数）
2. **距上次训练 ≥ 20h**（防同一窗口连训）
3. 8h 备用 plist 落盘到 `scripts/launchd/com.lnmem.studio-nightly-train-8h.plist.alt`，川哥可随时 5 秒切换

## 选项与对比

| 选项 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| A: 24h (03:00) | 高质量批次 / Studio 深睡省电 / 训练稳定 | 反馈生效慢（最多 24h） | ⭐⭐⭐⭐⭐ |
| B: 8h (00/08/16) | 反馈循环 ≤8h | 多数窗口 <30 条反馈 → 实际 skip / 唤醒 Studio 频繁 | ⭐⭐ |
| C: 4h | 最快反馈 | 几乎全部窗口 skip / 高频唤醒磨损配件 | ⭐ |

## 为什么选这个（gemini-3.5-flash 对抗审钦定）

1. **灾难性遗忘**（最强论据）：极小数据集（<10 条）频繁 BPTT，LNN 权重快速向最近几次点击偏斜，历史特征崩塌
2. **统计学意义**：3 类均分要每类 ≥10 条才有梯度信号；30 条增量门控对应每类 ~10 条
3. **Studio 配件寿命**：M2 Max 24h 睡眠 vs 每 8h 唤醒的硬件磨损差异显著（gemini 提出）
4. **用户感知**：川哥实际感知不到 8h vs 24h 差异，但能感知"睡一觉变聪明"的确定性体验
5. **可恢复性**：备用 8h plist 已落盘，川哥试用 24h 一周后觉得太慢，5 秒切 8h，零成本

## 实现要点

`scripts/studio_nightly_train.sh`：
- 新增 `CURSOR_FILE` (`~/lnmem-train/.last_trained_feedback_count`)：训练成功后写 `${FEEDBACK_COUNT}:${TIMESTAMP}`
- 新增 `FEEDBACK_MIN_INCREMENT=30` (env: `LNMEM_FEEDBACK_MIN_INCREMENT`)
- 新增 `MIN_HOURS_BETWEEN_TRAINS=20` (env: `LNMEM_MIN_HOURS_BETWEEN_TRAINS`)
- symlink 改 `tmp+mv` 原子更新（codex 钦定，对齐 pull_ckpt_from_studio.sh）

`poc/src/realtime/daemon_main.py`：
- 新增 `_ckpt_watcher_loop()` thread (60s 周期)
- 监控 `~/.lnmem/ckpt/intent_head_latest.pt` symlink target
- 变化时调用 `inferencer._load_intent_head_ckpt(resolved)` 热加载

## 三方评审记录

| 引擎 | 立场 | 关键理由 |
|------|------|---------|
| **cc 初判** | 8h | 反馈循环快 |
| **codex (gpt-5.3-codex high)** | 中立，重点修实现 bug | FEEDBACK_MIN 数总行数 bug / symlink 非原子 / DST 时区缺失 |
| **gemini-3.5-flash high** | 强反对 8h，钦定 24h | 灾难性遗忘 / 统计学意义 / Studio 睡眠 / 用户感知差异不显著 |
| **cc 最终** | 改主意 → 采纳 gemini 24h | gemini 论据更扎实，codex 不反对此方向 |

## 后续观察指标

- 1 周后 retro：反馈生效延迟是否成为 demo 瓶颈
- 切 8h 触发条件：连续 3 天日均反馈 ≥30 → 重新评估
- 灾难性遗忘监测：每次训练记录 train_loss + val_loss + class distribution，发现 collapse 立刻拉长间隔

---

**关联 PR**：本 commit
**关联 ADR**：ADR-028（LNN 角色重定义）/ ADR-029（Sprint 2 spec）

---

## 川哥决策反转记录（5/25 12:18）

川哥在 cc 落 ADR-030（24h）+ commit b649815 之后明确指示："按照 8 小时间隔执行吧"。

**反转后实施**：
- 主 plist 从 24h（StartCalendarInterval Hour=3）→ 8h（00/08/16 三个 dict）
- 双门阈值：`LNMEM_FEEDBACK_MIN_INCREMENT=30→10`、`LNMEM_MIN_HOURS_BETWEEN_TRAINS=20→6`
- env vars 写到 plist 的 EnvironmentVariables（不依赖运行时 shell）
- 备用文件保留 24h 版本可回滚（见下）

**川哥决策依据（cc 推断 + 后续观察）**：
1. M0 demo 5/28-6/3 临近，**反馈循环速度**是 demo 叙事关键（"睡一觉 ≠ 几小时内迭代"差异显著）
2. 接受灾难性遗忘风险作为试点代价，**1 周后用 train_loss 曲线监测**，发现 collapse 再回滚
3. gemini 的论据偏向"长期最优"，川哥优先"短期 demo 体验" —— 价值排序由人决定

**回滚到 24h 的方法**（如观察到 catastrophic forgetting）：
```bash
# 1. 把当前 8h plist 备份为 .alt
cd /Users/fu-121/lnmem-poc/scripts/launchd
mv com.lnmem.studio-nightly-train.plist com.lnmem.studio-nightly-train-8h.plist.alt

# 2. 从 git 历史恢复 24h 版本
git show b649815:scripts/launchd/com.lnmem.studio-nightly-train.plist \
  > com.lnmem.studio-nightly-train.plist

# 3. 重装
./scripts/install_launchd_studio.sh
```

**监测指标（M0 demo 后第一次 retro）**：
- intent_head val_loss 训练曲线是否单调下降（监测 collapse）
- 每次训练的 class distribution 是否极度偏斜（监测过拟合最新批）
- Studio 系统日志 sleep/wake count 频率
- 川哥反馈生效感知主观打分（1-5）
