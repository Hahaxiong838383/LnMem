---
audience: cc / codex / gemini / 工程师 / 川哥
purpose: LnMem PoC Day 11 末完整状态 + Day 12+ 剩余工作清单 + 验证目标缺口分析 + 兜底方案
status: 当前快照 2026-05-10 12:14（Day 11 收官时落档）
parent: poc/docs/poc-spec.md（PoC 完整规范）
---

# LnMem PoC Day 11 末状态 + Day 12+ Roadmap

> **本档作用**：所有 AI 工具（cc / codex / gemini）和川哥本人，开任何 LnMem 新 session 时**第一件事先读这个**。比读单条 commit 信息更有效率，比 grep 散落的 feedback 更聚焦。
>
> **更新规则**：每个 Day 收官时由 cc 主线刷新（增量 append "Day N 末状态"段），不删除历史段落（追溯链）。

---

## 0. 一句话现状

- **PoC 阶段**：Day 11 ✅ 收官，反馈闭环开闸 + daemon V0.2 长期运行（PID 15934 用户终端启动）
- **最大卡点**：真实数据时长约 **2-3h**，距 Day 12 末硬门槛 25-30h **缺 22+ h**
- **CC 工程债**：Phase 2 在线 LoRA + Day 13 微调 + Day 14 5 可视化 + 文档完整版（共 5-6.5 天）
- **预估真实完工**：从 2026-05-10 起 **8-10 天**（即 2026-05-18 ~ 2026-05-20，含川哥每天 4-6h 自用）

---

## 1. Day 0-11 已完成（按 poc-spec.md AC 严格对照）

| Day | AC | 实际 | 状态 |
|-----|----|------|------|
| 0 | 30 画像 + validator | 31 画像（30 vboss + 1 vboss_self）/ 217/217 PASS | ✅ |
| 1-2 | 30fps + 16kHz + 上下文 | smoke 4/4 PASS（25.5fps + 16kHz + osascript） | ✅ |
| 3-4 | 30 画像 × 30 天 = 7,191h 数据 | 900 LTC 演化 18-dim 信号 | ✅ |
| 5 | 状态识别 ≥30% / boundary F1 >0.5 / t-SNE 聚类 | val_acc **70.2%** → 6.5 升 **91.5%** / bnd_f1 **0.565** | ✅ |
| 6 | feature_dim 12→18 + boundary 调参 | feature_dim 18 + AU 6 维 / 22 tests PASS | ✅ |
| 6.5 | 真人微笑→positive 100% 触发 | 实际 100% + persona 个性化真人级显现（vboss_001 vs vboss_007 同帧 72% vs 60%） | ✅ |
| 7 | 5 阶段记忆 全部跑通 | **🟡 部分**：1 重要性标记 + 2 工作记忆 OK，3 事件分割 + 4 精炼 + 5 consolidation 仍是合成数据 stub | ⚠️ 工程债 |
| 7-8 | Realtime Daemon V0.1 | 6 模块 + HTTP 9876 + 双摄像头 20 AU 激活 | ✅ |
| 9 | VLM 标注上线 | Gemini 3.1 Flash Lite + MiniMax M2.7 优先链 / 1801 条 vlm_label | ✅ |
| 10 | Phase 1 反馈闭环 | V0.2 macOS 弹窗 6 模块 + codex 8 finding 4 修 + 第一条真实反馈样本（mismatch+distracted） | ✅ |
| **11** | FB-09 修复 + venv 重建 + daemon V0.2 持续 | 14 trigger 测试 PASS + .venv-poc-fresh 79 包 + daemon 449 帧 / 13 skipped / 第 3 条 reward 样本（match brow_burst） | ✅ |

**Why 这套对照**：poc-spec.md 是 Day 0 立的合同，每天收官按合同对一次，避免"感觉做了很多但 AC 没过"的偏差（BRO-PDCA 验收纪律）。

---

## 2. Day 12+ 验证目标缺口（poc-spec §0 一句话目标拆解）

> 目标："**LNN 真的能在多模态时序数据上学到主人特征**，且学习过程对老板和团队**可见、可解释、可治理**。"

| 关键词 | 当前进度 | 缺什么 |
|--------|---------|--------|
| LNN 真的能学到 | ✅ Day 6.5 合成数据 val_acc 91.5% | 已证 |
| 多模态时序数据 | ✅ 18 dim FusionFrame | 已证 |
| **学到主人特征** | ⚠️ 仅合成 30 主人有效；川哥真实数据 0 微调 | **Day 13 微调实验** |
| **可见** | ⚠️ 5 个核心可视化 0 出片 | **Day 14 录屏** |
| **可解释** | ⚠️ τ 偏置 + FiLM 真人证据有了，参数热力图未出 | 可视化 |
| **可治理** | ✅ Phase 1 V0.2 反馈闭环 + 3 条 reward 样本 | 雏形已立 |

---

## 3. 9 项剩余工作清单（按 poc-spec §2 严格对照）

| 项 | poc-spec AC | 当前 | 缺口 | cc 工时 | 川哥工时 |
|----|-------------|------|------|---------|---------|
| **A. 真实数据时长** | 25-30h | ~2-3h | **22+ h** | 0 | **4-5 天 × 4-6h** |
| **B. 反馈样本** | ≥30 触发 Phase 2 | 3 条 | 27 条 | 0（被动） | 川哥使用积累 |
| **C. 5 阶段记忆 consolidation** | 每晚自动跑通 | Day 7 stub | 工程缺口（事件分割/精炼/consolidation） | **1 天** | 0 |
| **D. Phase 2 在线 τ-bias 漂移 LoRA** | 反馈→reward→微调闭环 | 未启动 | online_lora_adapter.py 新建 | **1-2 天** | 0 |
| **E. vboss_self 入 ckpt** | 真人画像加 1 slot 微调 5-10 epoch | personalt-test v0.2 自评在 | 加 ckpt slot + 训练脚本 | **0.5 天** | 0 |
| **F. Day 13 微调对比报告** | 预训练 vs 微调 准确率 + t-SNE 距离 | 未启动 | 实验 + 对比图 | **0.5-1 天** | 0 |
| **G. Day 14 5 可视化** | 准确率曲线 / 参数热力图 / t-SNE / 消融 / AI 学习日志 | 0 出片 | 5 个 matplotlib + Streamlit | **1.5-2 天** | 0.5 天（脸出镜） |
| **H. demos/ 3 工作场景录屏** | 至少 1 个"AI 错被纠正"或"选择不介入" | 第 1 条 mismatch 已采，未录屏 | 录屏 + 剪辑 | **0.5 天** | 0.5 天（演员） |
| **I. limitations.md + migration-guide.md 完整版** | 完整诚实声明 + 主分支迁移建议 | 骨架 v0.1 | 补全 | **0.5 天** | 0.1 天（review） |

**总计**：cc **5-6.5 天工程**，川哥 **4-5 天使用 + 0.5-1 天演示配合**（可并行）

**Why 这个清单**：每项都对应 poc-spec.md 一条具体 AC，避免"做了但不算 PoC AC 完成"的工作浪费。

---

## 4. 兜底方案（按 poc-spec §2 + §6）

| 兜底层级 | 触发条件 | 行动 |
|---------|---------|------|
| **L1 时间延长** | Day 12 末数据 < 15h | 延长川哥自用 1-3 天（PoC 时间盒可弹性） |
| **L2 简化目标** | Day 13 微调没出明显效果 / 时间紧迫 | 仅"状态识别"单任务，去掉 boundary + 个性化（保至少 1 条 LNN 学习证据） |
| **L3 重启复盘** | 川哥对体验严重不满 | 立即停 + 复盘根因 + 决定是否重启（O-Q3 失败兜底） |

---

## 5. Day 12 P0 任务（已在另一 session 启动）

> P0 prompt 已发川哥，在另一 session 执行。本档不重复，简列任务名：

| 任务 | 责任 | 依赖 |
|------|------|------|
| daemon 稳定性观察 + reward 采集 | cc 监控 + 川哥使用 | 川哥每天 4-6h |
| Phase 2 在线 LoRA 工程骨架 | cc 主线 + codex 对抗审 | 反馈样本 ≥30 条 |
| vboss_self 入 ckpt | cc 主线 | personalt-test v0.2 已有（partial 字段⚠️ 标注） |
| Day 12 末 retro | cc 写 docs/day12-progress.md + project_lnmem.md 更新 | Day 12 收官时 |

---

## 6. 不可妥协约束（违反 = PoC 失败）

来自 `poc-spec.md §1` + `lnn-first.md` + `cognitive.md §5` + `development-harness.md`：

1. **诚实优于惊艳**：演示给团队的"AI 学习了你"，必须明说 LNN 30-40% / LLM 60-70% 协作占比
2. **架构形态优于性能数字**：不追求 90%+ 准确率，追求"完整流程跑通 + 学习趋势可见 + 个性化参数分化"
3. **可解释优于黑箱**：训练 / 个性化参数 / 长期记忆全部可视化
4. **用户参与优于自动化**：所有"AI 对主人新理解"必须老板确认才进长期记忆
5. **LNN-FIRST P-1**（lnn-first.md）：凡涉及"时间/演化/记忆/个性化"必须 LTC/CfC 微分方程，违反需 ADR + 川哥签字
6. **评估同构**（cognitive.md §5）：禁问卷弹窗评分，多模态信号自动聚合
7. **BRO-PDCA × 目标驱动**（development-harness.md）：O 阶段 AC + DoD 书面化，C 阶段附证据
8. **三道门**（CLAUDE.md §3）：cc 实现 → codex 对抗审 → 川哥签

---

## 7. 后续启动 session 必读清单

任何 cc / codex / gemini 新会话开 LnMem 工作，**第一件事**：

```bash
# 1. 项目定位（30s）
cat ~/.claude/projects/-Users-rayman-chen-mycc/memory/project_lnmem.md

# 2. 当前状态（30s）
cat /tmp/lnmem-fix/LnMem/docs/day11-end-status-and-day12-roadmap.md  # 本档

# 3. 仓库现状（1min）
cd /tmp/lnmem-fix/LnMem
git log --oneline -10
git status

# 4. 历史踩坑（2min，PoC 已积 17 条）
ls ~/.claude/projects/-Users-rayman-chen-mycc/memory/feedback_lnmem_*.md

# 5. Anchor 输出（30s）
echo "我做的任务在 PoC 大局位置：____"
```

完成才动手。详见 `feedback_global_view_before_local_fix.md`（mycc memory）。

**Why 这条规则**：2026-05-10 cc 一上来陷入细节修补 .venv-poc 1 小时，川哥喊停 2 次才纠正。"项目记忆是已经付过学费的资产，不读就是浪费。宁可慢 5 分钟扫描，不可丢 30 分钟在错误 hypothesis 上瞎修。"

---

## 8. 关键资产路径速查

| 资产 | 路径 | 用途 |
|------|------|------|
| daemon 二进制 | `.venv-poc-fresh/bin/python -m poc.src.realtime.daemon_main` | Day 12+ 启动入口 |
| daemon 日志 | `/tmp/lnmem-daemon-day11.log` | 排错 |
| 反馈样本 | `~/.lnmem/feedback/queries_*.jsonl` | Phase 2 reward 信号源 |
| 跳过事件 | `~/.lnmem/feedback/skipped_*.jsonl` | FB-09 修复证据 |
| signals 数据 | `~/.lnmem/data/signals_*.jsonl` | 训练原料 |
| API keys | `poc/.env`（gitignored） | GEMINI_PROXY_KEY + MINIMAX_API_KEY |
| 备份兜底 | `/tmp/lnmem-backup-2026-05-10/` | venv tar 1.1G + feat_resources 949M + .env |
| 旧 venv（fallback） | `.venv-poc/`（残破，**勿动**） | 不可恢复时 untar 回滚 |
| Day 0 规范 | `poc/docs/poc-spec.md` | AC + DoD 合同 |
| Day 6.5 进度 | `docs/day8-realtime-progress.md` | 范式跃迁记录 |
| Day 10 进度 | `docs/day10-feedback-loop-progress.md` | 反馈闭环开闸 |
| **Day 11 末进度（本档）** | `docs/day11-end-status-and-day12-roadmap.md` | Day 12 起步入口 |

---

## 9. 关联 mycc memory（17 条 LnMem feedback）

按相关度排序，cc 新会话**至少读前 5 条**：

1. `feedback_global_view_before_local_fix.md` — **元方法论**：5 分钟全局扫描 SOP
2. `feedback_lnmem_day11_venv_rebuild.md` — Day 11 venv 重建踩坑全记录
3. `feedback_lnmem_pyfeat_model_mgmt.md` — py-feat 6 铁律（949M 模型管理）
4. `feedback_lnmem_realtime_capture_pinned.md` — TCC 摄像头 + mediapipe 钉版本
5. `feedback_lnmem_phase1_review_loop.md` — codex 对抗审查 8 finding 模式

剩余 12 条覆盖：trigger 阈值校准 / OOD 数据稀释 / macOS 弹窗 / cognitive spec / dev harness / personas gitignore / poc kickoff / real data baseline / day65 pyfeat / multi-AI entry / haiku 子 agent 信任但验证 / state confirmation。

---

## 更新日志（本档增量）

- **2026-05-10 12:14** Day 11 末快照（cc opus 4.7 + codex gpt-5.3-codex/high + haiku 4.5 三路并行收官，FB-09 + venv 重建 + daemon V0.2 启动 5min DoD 全过 + 元方法论沉淀）
