---
adr-id: 014
title: Day2 决策检查点 — base v2.1 微调结果出来前的状态锁定
status: accepted（2026-05-21 10:10，cc 主线钦定，川哥要求 "commit 前做好决策记录点万一效果不行能回滚"）
date: 2026-05-21
deciders: 川哥 + cc + codex (5.4 / 5.3) + gemini (3-pro / 3.5-flash)
supersedes: —
related: ADR-013 base v2 失败分析 / ADR-007 / ADR-012
---

## 上下文

Day2 上午（08:00-10:10）密集协作产出 5 个模块（codex 3 大实施 + cc 主线 ADR-013 + 状态记录），但 base v2.1 微调结果 **未知**（正在跑，预计 1-3h 完成）。

川哥要求：「做好决策记录点再 commit，万一实际效果不行还能回滚。」

本 ADR 是 commit 前的状态快照 + 回滚指南。

## 本次 commit 包含的内容

### 1. 文档（不可逆，但安全）
- `docs/decisions/013-base-v2-failure-and-finetune-plan.md` — ADR-013 失败分析 + DFAT 方案
- `docs/decisions/014-day2-checkpoint-pre-v2_1-results.md` — 本文件
- `docs/reports/2026-05-21-base-v2-ab-replay.md` — codex A/B replay 失败报告（v1=v2=100% mismatch）

### 2. base v2.1 微调基础设施（codex 5.4 high 实施）
- `scripts/finetune_base_v2_1.py`（894 行）— DFAT 微调脚本
- `scripts/ab_replay_ckpt_v1_v2.py`（493 行）— 三模型 replay 对比脚本
- `run_finetune_and_replay.sh` — 串联 shell
- `tests/test_finetune_base_v2_1.py`（3 测试）
- `tests/test_ab_replay_ckpt.py`（3 测试）

### 3. parser 真 LLM 接通（codex 5.4 high 实施）
- `universal_grammar_parser/llm_client.py` — GeminiProxyClient（OpenAI 兼容）
- `universal_grammar_parser/parser_v0_llm.py` 改 — 接入 LLMClient
- `universal_grammar_parser/__init__.py` 改 — export GeminiProxyClient
- `universal_grammar_parser/tests/fixtures/` — 10 句中文测试数据
- `universal_grammar_parser/tests/test_universal_grammar_parser_real.py` — 真实 smoke
- 4 SKIPPED 测试转 PASS

### 4. Streamlit demo UI（codex 5.3 medium 实施）
- `apps/demo_e2e/main.py` + `components.py` + `llm_config.py` + `README.md`
- `tests/test_demo_e2e_smoke.py`（2 测试）

### 5. 不进 commit（已在 .gitignore）
- `poc/data/checkpoints/base_v2/*.pt`（226KB-228KB ckpts，明早从 Studio 拉来的）
- `poc/data/checkpoints/m0w2_phase2/*.pt`（旧训练产物）

## 验收门状态

| 维度 | 状态 |
|------|------|
| **Verify**（target tests）| ✅ 15 passed + 1 skipped（real smoke 默认 skip）|
| **Guard**（poc/tests 不破）| ✅ 573 passed + 2 skipped |
| **代码 review**（cc 自查）| ✅ 通过（codex 5 翻车点已记录到 ADR-013）|
| **base v2.1 实际微调效果** | ⏳ **未知**（finetune PID 56631 跑中，1-3h ETA）|
| **A/B 三模型对比** | ⏳ 等 v2.1 ckpt 出来 |
| **demo 端到端浏览器跑通** | ⏳ 待 LLM 真实接通验证 |

## 决策（commit 与否的关键点）

✅ **当前 commit 是「基础设施 + 失败分析 + 微调脚手架」，不依赖 v2.1 实际效果**。

理由：
1. 即使 v2.1 微调失败 → ADR-013 仍记录了失败教训，本 commit 仍有价值
2. 即使 demo 整合卡 → parser/UI 代码仍可作 M1 起点
3. 即使所有今日产出都不上线 → 测试 + 工程基础设施已沉淀

**这次 commit 不锁定 v2.1 训练结果**，结果出来后单独 commit + 更新 ADR-013 验收段。

## 回滚路径

### 场景 A：v2.1 微调失败（坍缩 / 不收敛 / 效果反退）
- 不做任何回滚（本 commit 仍有价值）
- 更新 ADR-013 加 v2.1 失败复盘
- 写 ADR-015 计划 base v3（VEATIC + 川哥真实数据混合 + DANN 域对抗）
- daemon 继续跑 base v1（不切换）

### 场景 B：v2.1 微调成功但 daemon 切换失败
- daemon 切回 base v1（一行配置）
- 保留 v2.1 ckpt 在 ADR-013 提到的位置，留 M1 继续微调

### 场景 C：parser/UI 整合发现 bug
- demo 暂时不上线，但代码留 repo
- 不影响 base v2.1 路径

### 场景 D：所有产出都翻车
```bash
git revert HEAD  # 撤销本 commit（保留历史）
# 或者
git reset --hard HEAD^  # 强制回退（cc 不推荐，会丢历史）
```
- 撤回后 LnMem 仍可继续：base v1 daemon + 之前的 graphiti Phase A + 全部 Day1 commits

### 场景 E：发现微调脚本有 P0 bug 但已 commit
- 不 revert，直接补 fix commit
- ADR-013 加 "v2.1 实施 bug 复盘" 段

## 后续 commit 计划

| commit 序号 | 内容 | 触发条件 |
|-------------|------|----------|
| 本次（即将）| 基础设施 + ADR-013/014 | 现在 |
| 下次 | base v2.1 ckpt + finetune 报告 + ADR-013 验收段 | finetune 完成 + A/B 验收 |
| 再下次 | demo 端到端验收 + 截图 | 浏览器跑通后 |

## 关键文档引用

- ADR-013: `docs/decisions/013-base-v2-failure-and-finetune-plan.md`
- A/B replay 报告: `docs/reports/2026-05-21-base-v2-ab-replay.md`
- gemini 评审: `/tmp/lnmem-team-2026-05-20/gemini_base_v2_failure.md`
- cc spec: `/tmp/lnmem-team-2026-05-20/codex_finetune_v2_1_spec.md`
- 昨日复盘: `docs/retrospectives/2026-05-20-day1-team-mode-retro.md`

## 风险声明

cc 主线认知状态：连续 25h 作战（昨日 11.5h + 今早 4h），决策疲劳风险存在。**本 commit 经过：测试全过 + 代码 review + 文档落地 + 川哥确认**，应该不会引入新错误。但 v2.1 微调实测结果可能引出新的设计问题，留 ADR-015 余地。

## Summary

本 commit 锁定基础设施（5 模块 + 1645 行 + 17 测试），不锁定 v2.1 微调结果。
回滚路径已写：4 种失败场景 + 1 种 P0 bug 场景，均有应对。
finetune PID 56631 跑中，结果出来后单独 commit + 更新 ADR-013 验收段。

---

**模板版本：v0.1 · 2026-05-03**
