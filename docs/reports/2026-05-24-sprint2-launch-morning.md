# Sprint 2 启动晨起报告（5/24 凌晨 02:50-？）

> **给川哥的话**：你睡前拍了 Q1=A / Q2=A / Q3=A + "team 模式最大努力"。
> cc 凌晨开干 team 模式（cc + codex + gemini 三方并行），全程后台跑。
> **这份报告 30 秒看完 → 直接知道现在能干啥**。
>
> 详细 spec：[`docs/decisions/029-sprint2-intent-classifier-daf-studio-pipeline.md`](../decisions/029-sprint2-intent-classifier-daf-studio-pipeline.md)

---

## 1. 30 秒大局

| 维度 | 状态 |
|------|------|
| 战略 | ADR-029 落盘 ✅（Q1/Q2/Q3 三选 A 钦定方案完整 spec） |
| codex 实施 | Sprint 2-B 一轮 ✅ + 二轮 retrofit ✅（gemini 整改 H1/H2/H3） |
| cc 主线 | rsync + Studio 训练入口 + Streamlit UI v0 + ROADMAP 更新 ✅ |
| gemini 对抗审 | 8 finding，CONDITIONAL GO，钦定建议「10s hidden state 窗口」已纳入 retrofit |
| GitHub | 整批 commit + push 到 `poc/lnn-validation` ✅ |

**川哥现在能干的事**：起床喝杯水，按下方 §3 三条命令依次验收。

---

## 2. 凌晨干了啥（按时间顺序）

| 时间 | 事件 |
|------|------|
| 02:50 | 川哥拍三选 A + team 模式开干 |
| 02:52 | cc 派 codex Sprint 2-B 后台（intent_head + lnn_inferencer + /intent_feedback endpoint） |
| 02:55 | cc 派 gemini 对抗审 Sprint 2 整体设计 |
| 03:00 | cc 写完 `sync_daemon_to_studio.sh`（Q1=A）+ dry-run 验证（mini 网络抖动） |
| 03:05 | cc 写完 `studio_nightly_train.sh` + `pull_ckpt_from_studio.sh`（Q2=A，改成 Air pull 模式因 Studio 不知道 Air 动态 IP） |
| 03:10 | gemini 评审完成：8 finding，CONDITIONAL GO + 钦定 10s hidden state 窗口 |
| 03:15 | cc 写完 `streamlit_ui/intent_console.py`（DAF 弹卡 + 静默模式 + 反馈窗口 ≤2min） |
| 03:20 | cc 写完 ADR-029 完整 spec（整合 gemini 8 finding 整改方案） |
| 03:25 | cc 更新 ROADMAP 顶部 → v0.1 Sprint 2 状态（旧 5/22 11 轮归档） |
| **04:?** | **codex 一轮 commit + push（待）** |
| **04:?** | **cc 派 codex 二轮 retrofit（H1 hidden state 窗口 + H2 Focal loss + H3 软重启）** |
| **05:?** | **整批 commit + push GitHub** |

---

## 3. 川哥晨起 3 条命令验收

### 命令 1：拉最新代码
```bash
cd ~/mycc/2-Projects/LnMem-poc
git pull origin poc/lnn-validation
git log --oneline -10  # 看 5/24 凌晨这批 commit
```

### 命令 2：跑 Streamlit UI 看看 LNN 实时推断
```bash
# 确认 daemon 在跑
ps aux | grep daemon_main

# 启 Streamlit
.venv/bin/streamlit run streamlit_ui/intent_console.py --server.port 8520
# → 浏览器自动开 http://localhost:8520
```

UI 上能看到：
- 当前 3-class 意图推断（🟢 正反馈 / 🔴 负反馈 / ⚪ 无反馈）
- 置信度 + margin + entropy
- 静默模式开关（默认开，只有置信度 ≥ 0.8 才弹反馈卡）
- 待反馈队列（最多显示 3 张卡，超 2 分钟自动过期）

**期待你给 cc 反馈**：
1. UI 节奏舒服吗？（弹卡频率、按钮位置、字号）
2. 静默模式阈值 0.8 合适吗？
3. 反馈窗口 2 分钟够不够？

### 命令 3：测试 Q1=A rsync（mini 网络恢复后）
```bash
./scripts/sync_daemon_to_studio.sh --dry --verbose
# 看：会传啥 / 排除 raw_video 是否生效
# Studio 通了再去 LaunchAgent 配 @hourly cron
```

---

## 4. Sprint 2 后续 10 天 Timeline

| 日期 | 任务 | 责任 |
|------|------|------|
| 5/24 早 | 川哥验收 + 反馈 UI / 阈值调整 | 川哥 |
| 5/25-5/26 | **冷启动期**：你边用边给反馈攒数据（目标 ≥ 50 条） | 川哥 + cc 联调 |
| 5/27 | cc 在 Studio 手动 `--force` 触发第一次训练（baseline） | cc Studio 执行 |
| 5/28-5/30 | 模型迭代 + UI 改进 | 全员 |
| 5/31-6/1 | Studio cron 自动接管夜训 | cc 部署 |
| 6/2-6/3 | M0 demo 收尾：录屏 + 24h 学习曲线 + 川哥分享物料 | cc + 川哥 |

---

## 5. 留给川哥拍的 3 件后续事（不紧急）

| # | 事 | 选项 |
|---|-----|------|
| F1 | UI 弹卡静默阈值 | 0.8 （默认）/ 0.7（更主动）/ 0.9（更安静） |
| F2 | 反馈窗口长度 | 120s（默认）/ 60s（更严谨）/ 300s（更宽松） |
| F3 | Studio 训练触发条件 | 50 条反馈（默认）/ 30（更激进）/ 100（更稳） |

睡醒看 UI 用过一阵后告诉 cc，cc 改一行配置。

---

## 6. 凌晨踩坑 + 经验记录

新增 mycc 记忆（已落盘到 `~/.claude/projects/-Users-rayman-chen-mycc/memory/`）：

| 文件 | 教训 |
|------|------|
| feedback_pyfeat_tmpdir_race.md | py-feat AUExtractor 共享 tmp_dir 多 worker 文件覆盖（5/23 实证） |
| feedback_codex_mock_test_blindspot.md | codex mock 测试盲区（mock 全过 ≠ 真库可用，5/23 实证） |
| feedback_user_intent_anchoring.md | cc 必须周期性追问用户产品意图（5/22-23 LnMem 教训） |

本轮新增（待 commit 后补）：
- feedback_macos_no_flock.md（macOS 无 flock，PID 锁 + ps 检查替代）
- feedback_gemini_hidden_state_window_insight.md（gemini 钦定 10s 窗口防 LNN 退化为 MLP）

---

## 7. 不可妥协项检查（CLAUDE.md §5）

| 项 | 状态 |
|----|------|
| 没违反 LNN-FIRST P-1 | ✅ gemini H1 整改恰恰是为了**强化** LNN 序列动力学 |
| 没改 §10.1 不可妥协 6 项 | ✅ Sprint 2 spec 是落地路径，非偏离 |
| 没改 §13.1 双门验证 | ✅ ADR-029 §6 验收门保留双门精神 |
| 老板授权对外操作 | ✅ 川哥 5/24 凌晨明确「team 模式开干」+ 之前主请求授权 push |

---

---

## 附录：5/25 凌晨 00:30 实际终态（cc 一晚成果）

**GitHub commit**（push 完成）：
- `578baf3` Sprint 2 一轮（codex + cc 主线 17 文件 / 1700+ 行 / 19 测试过）
- `2e260c6` Sprint 2 retrofit（gemini H1/H2/H3/D4/U2 整改 / 16 文件 / 1236 行 / **24 测试过**）

**Studio 公网部署进展**：
- ✅ ssh studio-pub (10.9.1.159 via mini-pub:22022) 公网通
- ✅ `~/lnmem-poc` 558MB 仓 rsync 到位
- ✅ `~/lnmem-daemon-sync/` **14 天 daemon jsonl 659MB**（5/9-5/24）
- ✅ `~/lnmem-train/ckpt` + `~/lnmem-train/logs/train` 建好
- ✅ uv 0.11.16 + python 3.11.15
- 🔄 venv torch（清华镜像后台装中，预计 3min 内通）
- ⏳ `~/Library/LaunchAgents/com.lnmem.studio-nightly-train.plist` 待装（川哥起床后跑 `./scripts/install_launchd_studio.sh`）

**川哥 5/25 早起 5 件事（已按重要性排序）**：

```bash
# 1. 拉最新代码 + 看 commit
cd ~/mycc/2-Projects/LnMem-poc && git pull
git log --oneline 38f7033..HEAD

# 2. 重启 daemon 加载新代码（带 hidden_state_window + intent_feedback endpoint）
launchctl unload ~/Library/LaunchAgents/com.lnmem.daemon-supervisor.plist 2>/dev/null
sleep 2 && launchctl load ~/Library/LaunchAgents/com.lnmem.daemon-supervisor.plist
# 或手动：./scripts/start-daemon.sh

# 3. 跑 Streamlit UI 看实时推断 + 给反馈
.venv-poc-fresh/bin/streamlit run streamlit_ui/intent_console.py --server.port 8520

# 4. 装 Air LaunchAgent（让 rsync + pull ckpt 自动跑）
./scripts/install_launchd_air.sh

# 5. ssh Studio 装 LaunchAgent（让夜训 03:00 自动跑）
ssh studio-pub "cd ~/lnmem-poc && ./scripts/install_launchd_studio.sh"
```

**关键现实告知**：
- 现在 0 条 intent_feedback 数据，**今晚不能真训** intent classifier（数据 = 0 < 50 阈值）
- 川哥 5/25 全天用 UI 给反馈 → 5/26 03:00 cron 自动训第一次（如果 ≥ 50 条）
- 或川哥手动 `ssh studio-pub "~/lnmem-poc/scripts/studio_nightly_train.sh --force"` 提早训

**Studio 14 天 daemon 数据用途**：
- 不训 intent_classifier（缺标签）
- 可训 LNN backbone（self-supervised） — **但 5/22 11 轮已证 daemon shuffle_ratio ≈ 0，这条路 ADR-028 已转向**
- 实际价值：让 Studio 训练 pipeline 一启动就有真实序列数据「热身」，避免冷启动 race

**署名**：cc（Claude）| 2026-05-25 00:30 终态落盘
**协同**：codex (gpt-5.3-codex high · 2 轮 35M+ token) · gemini (3-flash-preview · 8 finding)
