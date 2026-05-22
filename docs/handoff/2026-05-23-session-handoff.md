---
title: 5/23 Session Handoff · 11 轮 Team 模式收尾后接手 prompt
date: 2026-05-22 23:00
status: ready
previous_session: 5/22 single-day 11 rounds (commits 8e38bde → 6187cec)
---

# 5/23 Session Handoff · LnMem M0 W3 → M0 W4

## 项目整体目标达成对账

LnMem = **伴侣型 AI 的记忆内核**，差异化押在 LNN（liquid neural network）+ AU 多模态 + per-user persona

### 不可妥协 6 项 P-1 达成度（5/22 收尾）

| 项 | 达成度 | 关键证据 |
|---|---|---|
| #1 **LNN-FIRST** | 95% ✅ | ADR-022 v3 τ 释放 + ADR-023 ODE-loss + ADR-024 数据驱动阈值。loss 直训 `dh/dt=(f+τ)·h` |
| #2 per-user τ-bias | 70% | spec 钦定 + ADR-022 v3 释放，N=10 还没 per-user 真实测试（待 N=124 / β 川哥本人） |
| #3 AU 多模态 | 80% | py-feat AU 抽取 work，schema 24+28 占位。还没接入文本/语音 |
| #4 离线优先 | 90% ✅ | 全流程 Studio 本地跑，daemon 长跑未实测 |
| #5 ODE 可解释 | 75% | CfC 半隐式 + τ-spread 多时间尺度。还没"给用户的可解释 UI" |
| #6 时序记忆 | 85% ✅ | Shuffle ratio 1.73 证实 LNN 真用时序。记忆长度（小时/天）未压测 |

**总分**：技术层 ~83%，产品层 ~20%，商业化 0%

### 阶段达成

| 阶段 | 计划 | 实际 |
|---|---|---|
| M0 W1-W3 Day 3 | Daemon + parser + harness + γ-1/2/3 | ✅ 完成 |
| M0 W3 Day 4 | 5 PC 落地 + 50ep 验证 | ⚠️ 50ep 不达 → 范式转换 ODE-loss + 200ep 全过 |
| **5/27 Phase γ Step 1** | N=124 / 4 天 | **🟢 GREEN LIGHT 准予启动**（gemini 14 APPROVE）|
| M0 W4 (5/27-31) | Phase γ Step 1 + β-self-test | ⏳ 待 |
| M0 收口 (6/10) | Phase δ 川哥真人数据微调 | ⏳ 待 |

---

## 5/22 11 轮 Team 模式关键产出（接手必读）

### 主分支 / commits
- HEAD: `6187cec`
- 关键链：`6187cec` ← `7d1fc5f` ← `e38d28f` ← `f3ee848` ← `8e38bde`

### 三大架构级 ADR
- **ADR-022 v3**（`docs/decisions/022-tau-bias-gradient-blackhole-fix.md` §7）— τ-bias 释放 1e-3 + DGB 单项控制
- **ADR-023**（`docs/decisions/023-ode-dynamics-loss-paradigm-shift.md`）— 范式转换 InfoNCE+L2 → Latent World Model Rollout
- **ADR-024**（`docs/decisions/024-au-data-effective-dim-physical-upper-bound.md`）— AU 数据物理上限 5.07 锁死 + 相对阈值

### v3.0 200ep 实测（全 5 启动门破阈值）

| 门 | 实测 | 阈值 |
|---|---|---|
| τ.std（多时间尺度真分化）| 0.577 | > 0.5 ✅ |
| effective_dim / input | 0.97 (4.92/5.07) | ≥ 0.80 ✅ +21% |
| Shuffle ratio（时序贡献） | 1.73 | ≥ 1.5 ✅ +15% |
| vs_naive baseline | 0.632 | < 0.7 ✅ |
| vs_persistence baseline | 0.50 | < 0.7 ✅ |
| trivial_recon_check_pass | True | True ✅ |

完整复盘：[`docs/reports/2026-05-22-team-mode-11-rounds-retrospective.md`](../reports/2026-05-22-team-mode-11-rounds-retrospective.md)

---

## 5/23+ 接手清单（按优先级）

### P0 立即（5/23）

1. **β-self-test 川哥本人 webcam + 10 句日常对话**（1-2h，需川哥配合）
   - 跑 `poc/training/extract_au.py` 抽川哥真人 AU
   - 跑 v3.0 inference 看 effective_dim / Shuffle 是否在 N=1 真人数据保持
   - 这是 #2 per-user τ-bias 第一次真实落地

2. **罕见 AU 重建监控加入 evaluate.py**（gemini 14 N=124 警告应对）
   - 按 AU 频率分位数（top-10 / median / bottom-10）独立算 L_world_model
   - 防 N=124 时模型刷主成分牺牲长尾

### P0 5/24-5/26

3. **5/24 daemon Air 后台长跑启动**（0.5h，→ Phase δ 真人数据累积）
   - 川哥本人 webcam 24h 录制
   - 验证 #4 离线优先 + 长跑稳定性

4. **优化 long training 上限**（1h）
   - v3.0 200ep work，跑 500/1000ep 看 metric 是否还涨
   - 找 plateau 信号定 final epochs for 5/27

### P0 5/27 启动 Phase γ Step 1

5. **N=124 真训练 4 天**（Studio M2 Max）
   - 启动前先跑 `measure_input_effective_dim` 看 N=124 是否真升到 7.5-9（gemini 14 预测）
   - 监控 effective_dim / Shuffle / vs_baseline 全 5 门
   - 罕见 AU 监控（gemini 14 警告）

### P1 5/28-6/9 M0 收口准备

6. **产品形态文档**（项目缺口 #4，没产品 spec）
   - API surface / 接入伴侣 AI 的最小契约 / input/output schema
7. **隐私边界协议**（缺口 #5，AU 数据生物特征敏感）
   - 存储边界、用户删除权、第三方调用契约
8. **β 用户画像**

### P2 6/10 M0 收口

9. **Phase δ 川哥真人数据微调**（per-user τ-bias 真实生效）

---

## 工作方式（CLAUDE.md 钦定）

### 三道门（§3）
1. 接口先行（Protocol + schema）
2. 测试先行（TDD）
3. 对抗审查（cc / codex / gemini）

### cc + gemini 冲突诊断（§3.1）
- 默认采纳 gemini 对抗审意见（保守优先）
- cc 不同意必须**明示分歧 + 数学/实测论证**
- ADR 落盘冲突案例

### 红线（§5）
- 不可妥协 §10.1 6 项变更必须问川哥
- 关键数据 rsync 必用 `--filter='P dir/'` 防误删（5/22 教训）
- AU 数据物理上限 = 5.07（N=10）/ 预测 7.5-9（N=124），**不要再追绝对 8.0**
- 5/27 启动前必须先 `measure_input_effective_dim` 跑 baseline

---

## 引擎 + 工具链状态

### AI 引擎
| 引擎 | 用途 | 5/22 状态 |
|---|---|---|
| cc (Claude) | 主线 | ✅ |
| codex (gpt-5.3-codex high) | 实施 / 对抗审 | ⚠️ 5/22 下午限额，5/23 应恢复 |
| gemini-3.5-flash + reasoning_effort=high | 对抗审 / 根因诊断 | ✅（数值预测 ÷ 2.5 sanity check） |

**调用引擎**：参考 mycc memory `reference_gemini_proxy.md` + `reference_codex_default_config.md` + `~/.claude/rules/engine-routing.md`

### Studio 公网访问
- 川哥不在家：`studio-ibro`（经 mini ProxyJump）
- 川哥在家：`studio-pub`（经 mini-pub ProxyJump）
- 数据：`~/lnmem-data/veatic/veatic_extracted/{rating_averaged,videos}`
- 代码：`~/lnmem-train/LnMem`（5/23 接手前先 git pull 同步到 `6187cec`）

### 实测命令模板

```bash
# Studio 同步代码（rsync 安全版）
rsync -avz --delete \
  --filter='- .venv-poc-fresh/' \
  --filter='- __pycache__/' \
  --filter='- *.pyc' \
  --filter='- .pytest_cache/' \
  --filter='P poc/data/' \
  --filter='P .venv-poc-fresh/' \
  ./ studio-ibro:~/lnmem-train/LnMem/

# Studio 跑 v3.0 (本机配置参考)
ssh studio-ibro "cd ~/lnmem-train/LnMem && HF_HUB_OFFLINE=1 PYTHONPATH=. .venv-poc-fresh/bin/python \
  -m poc.training.run_tiny_training \
  --au-cache poc/data/au_cache_tiny.npz \
  --rating-dir ~/lnmem-data/veatic/veatic_extracted/rating_averaged \
  --ckpt-out poc/data/checkpoints/p0_v3_xxx.pt \
  --loss infonce --seq-len 90 --batch-size 16 --hidden-dim 64 \
  --epochs 50 --seed 42"
```

---

## 6 大经验教训复用资产（mycc memory）

5/22 沉淀的跨项目复用资产：

| # | 文件 | 一句话 |
|---|---|---|
| 1 | `~/.claude/projects/-Users-rayman-chen-mycc/memory/feedback_threshold_must_have_data_baseline_first.md` | 阈值必须先有数据 baseline |
| 2 | `~/.claude/projects/-Users-rayman-chen-mycc/memory/feedback_gemini_numeric_prediction_optimistic_bias.md` | gemini 数值预测 ÷2.5 才能信 |
| 3 | `~/.claude/projects/-Users-rayman-chen-mycc/memory/feedback_cc_gemini_adversarial_review_power.md` | 双引擎对抗审 11 轮 6 次救项目 |
| 4 | `~/.claude/projects/-Users-rayman-chen-mycc/memory/feedback_project_principle_protects_from_industry_pattern.md` | LNN-FIRST 护身符拦 SOTA 范式 |
| 5 | `~/.claude/projects/-Users-rayman-chen-mycc/memory/feedback_long_training_vs_param_tuning.md` | long training 比调参快 18x |
| 6 | `~/.claude/projects/-Users-rayman-chen-mycc/memory/feedback_physical_prior_is_metric_ceiling.md` | 物理先验是 metric 硬天花板 |

---

## 第一句应该说

```
川哥早。LnMem 5/23 接手 — P0 列表里你打算先做 β-self-test 还是先做罕见 AU 监控？
```

---

## 接手 prompt（可直接粘贴到新 session）

```text
你是 cc，LnMem 项目协作主线。

必读 4 份文档（按顺序）：
1. /Users/rayman.chen/mycc/2-Projects/LnMem-poc/CLAUDE.md
2. /Users/rayman.chen/mycc/2-Projects/LnMem-poc/ROADMAP.md
3. /Users/rayman.chen/mycc/2-Projects/LnMem-poc/docs/handoff/2026-05-23-session-handoff.md（本文件）
4. /Users/rayman.chen/mycc/2-Projects/LnMem-poc/docs/reports/2026-05-22-team-mode-11-rounds-retrospective.md

主分支：poc/lnn-validation HEAD = 6187cec（5/22 11 轮 Team 模式收尾）

工作目录：/Users/rayman.chen/mycc/2-Projects/LnMem-poc

5/22 单日完成 13 commits / 3 ADR / 7 轮 gemini 对抗审 / 5 次 Studio 真训练，5/27 GREEN LIGHT 终判。

5/23+ 待办按 handoff §5 P0/P1/P2 列表。第一件事按川哥指令决定（β-self-test or 罕见 AU 监控）。

红线：
- 不可妥协 §10.1 6 项变更必须问川哥
- AU 数据物理上限 5.07（不要再追 8.0）
- gemini 数值预测 ÷2.5 sanity check
- 关键数据 rsync 必 --filter='P dir/'

引擎凭据查 mycc memory（reference_gemini_proxy.md / reference_codex_default_config.md）—— 本仓库不存任何 token。

立刻汇报第一件事接手计划。
```

---

**📝 维护者**：cc 主写（5/22 23:00）| **签发**：本仓 commit 6187cec 之后
