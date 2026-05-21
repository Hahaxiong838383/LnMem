---
title: "M0 W3 Day 2 (5/21) — Phase α + L4 三砖完工复盘"
date: 2026-05-21
status: closed
authors: [cc, 川哥]
related:
  - docs/decisions/015-vision-path-freeze-pivot-to-L4.md
  - docs/decisions/016-base-v3-multimodal-self-supervised-lnn.md
  - docs/decisions/017-phase-alpha-multimodal-demo-upgrade.md
  - docs/retrospectives/2026-05-20-day1-team-mode-retro.md
---

## 一日全景

| 时段 | 阶段 | 关键产出 |
|------|------|---------|
| 09:00-12:00 | base v2.1 微调 + 决策回归 | DFAT 没解 collapse → ADR-015 视觉冻结 → 川哥拉回 → ADR-016 撤回 |
| 12:00-13:30 | ADR-016 写作 + L4 砖 1 e2e demo | M0 端到端首次跑通（5 句 / parse 4.95s / P0 3.40） |
| 13:30-14:00 | L4 砖 2 CLS v0 | 5 events → 3 highlights / LLM 2.91s |
| 14:00-15:30 | L4 砖 3 + Phase α 合并升级 | Streamlit 4-in-1 + multimodal 模块 + ADR-017 |
| 15:30-16:00 | 川哥实测割裂 + 诊断 | 3 个 bug 定位（LLM thinking / CWD / streamlit deprecation） |
| 16:00-16:30 | 全部修复 + UI 友好化 | parse 45s→4.91s / HELP 字典 11 术语 / 4 步流程引导 |

**单日 commits（origin/poc/lnn-validation）：**
```
27b9361 fix(demo): 修 3 个割裂感 bug + UI 全面友好化
f4a3340 feat(L4-3+phase-α): Streamlit demo 4-in-1 升级
5efdd37 feat(L4-2): CLS v0 夜间整合最小可行
720732b docs(adr-016): 撤回 ADR-015 + 立 base v3 多模态 self-supervised LNN 路线
ed744d1 feat(L4-1): M0 端到端 demo CLI 跑通
df5002e docs(adr-015): 视觉感知路径阶段性冻结 + 战略转向 L4 自传记忆主线
795bef4 feat(adr-013/014): Day2 决策检查点 — base v2 失败分析 + v2.1 DFAT
```

7 commits · 3 ADR · 单日 ~1400 行 · 川哥 2 次关键纠偏。

## 重要决策链

### 决策 1：ADR-015 → ADR-016 范式撤回（13:55）

**起因**：cc 主线被 gemini「视觉权重 2/10」误导，准备写 ADR-015 视觉冻结。
**川哥关键纠偏**：
> 视觉感知在 LnMem 中也重要，我在输入「昨晚 11 点和小李在公司吃了越南粉」的时候，AI 得知道我是什么表情，我是什么意图，我是哭着说还是笑着说？我是疑问句还是反问句？LnMem 使用 LNN 神经网络，就是为了让 AI 懂并且能预测用户在输入需求或者沟通时的心情状态及心理活动。**不要用传统大模型的思路**来干这件事。

**复盘 cc 主线认知错误**：
- base v2/v2.1 全部走「**传统 LLM-centric supervised 分类**」思路（9-state CE + Russell margin + 硬标签 + mismatch rate）
- 这违反 LNN-FIRST P-1：LnMem 视觉的真正用途是「**让 LNN hidden state 学到主人多模态连续表征**」，不是分类
- gemini 评审用的是 LLM-centric 评价指标（focused acc 0% → 失败），没看到 LNN-FIRST 范式下表征学习的价值

**决策**：ADR-016 撤回 ADR-015，立 base v3 multimodal self-supervised LNN 范式（4 loss：Temporal Contrast / Cross-modal / Persona / Weakly-supervised Readout），9.8GB 数据全留。

### 决策 2：Phase α + L4 砖 3 合并实施（14:00-15:30）

合并 4-in-1：
1. Phase α webcam snapshot（ADR-016 多模态 input signal 预览）
2. L4 砖 3 记忆 UI v0.1 删除按钮（产品定义 §10.1 #6）
3. CLS 今日精华显示区（砖 2 接通 UI）
4. KG 卡片完整状态描述展开

**Gemini 评审 5 维度 + cc 3 处分歧记 ADR-017**（按 CLAUDE.md §3.1）：
- 分歧 1：删除按钮（gemini 砍 / cc 保留极简） → 川哥钦定 §10.1 #6
- 分歧 2：multimodal context 存储位置（gemini DB / cc 旁挂） → v1 schema 封版红线
- 分歧 3：20 槽 UI 命名（gemini 涌现 / cc 符号投影） → 分层不冲突

### 决策 3：川哥实测割裂 → 立即诊断修复（15:30-16:00）

川哥反馈"过程比较割裂"。log 诊断出 3 个 bug：
1. parser LLM 超时 45s（gemini-3-flash-preview thinking 默认开 + 没设 max_tokens/reasoning_effort）
2. CLS 面板瞎掉（streamlit nohup 启动 cwd 不在 repo，相对路径失锚）
3. streamlit 14+ deprecation 噪音（use_container_width → width='stretch'）

全部修复，parse 45s→4.91s（9x），CLS 真读到 24KB / 5 events。

同时 UI 全面友好化（川哥钦定）：HELP 字典 11 术语 / 4 步流程引导 / 鼠标悬停 tooltip / 友好错误提示 / 中英术语并列。

## 经验教训（可推广，已沉淀到 mycc memory）

### 教训 1：gemini-3.x 默认强 thinking → parser 必加 reasoning_effort=minimal

**踩坑现象**：完全相同的代码，cls_consolidate.py 跑通 2.91s，apps/demo_e2e/llm_config.py 跑超时 45s。
**根因**：cls_consolidate 显式设了 `reasoning_effort="minimal", max_tokens=4096`；llm_config 忘了同步这个 fix。
**铁律**：任何用 gemini-3.x 做 parser / 抽取 / 短回合任务的代码，必须显式设这两个字段。
**沉淀**：`mycc/memory/feedback_gemini_flash_thinking_default_trap.md`

### 教训 2：streamlit nohup 启动 cwd 失锚 → 必须用基于 __file__ 的绝对路径

**踩坑现象**：本地跑 streamlit 没事，cc 用 nohup 后台启动后 CLS 面板瞎掉。
**根因**：streamlit 进程继承父 shell 的 cwd（mycc 根，不是 LnMem-poc 根）。
**铁律**：任何长跑服务的文件路径用 `Path(__file__).resolve().parents[N]` 推 repo 根，禁止 `Path("data/xxx")` 相对 cwd。
**沉淀**：`mycc/memory/feedback_streamlit_nohup_cwd_trap.md`

### 教训 3：cc 二次过度信任 gemini 评审，必须由川哥拉回

**Day 1 教训**：DFAT 微调 cc 听 gemini "85% confidence" 全力推进，结果只解 collapse 89% 没解透。
**Day 2 教训**：视觉冻结 cc 听 gemini "视觉权重 2/10" 准备写 ADR-015，川哥拉回。
**铁律**：CLAUDE.md §3.1 升级 —— gemini 评审是学术外脑参考，**不是最终决策者**；任何对 LNN-FIRST P-1 / 产品定义 §10.1 不可妥协项的妥协建议，cc 必须停下问川哥。
**沉淀**：本仓 ADR-016 §gemini 评审吸收段 + mycc `feedback_lnn_first_universal_principle.md`（已有）

### 教训 4：UI 友好化范式（HELP dict + 数字步骤 + tooltip + 友好错误）

川哥要求 demo 体验"清晰易懂，专业术语和英文要有注释"。Phase α 修复版落地 4 件套范式：
1. 顶部 expander 4 步流程引导
2. HELP 字典集中术语 + tooltip 复用
3. 数字 emoji 1️⃣2️⃣3️⃣4️⃣ 步骤标号
4. 错误提示含 3 个原因 + 修法（不只是 stack trace）

**沉淀**：`mycc/memory/feedback_demo_ui_friendlification_pattern.md`，可作为 streamlit 项目 demo 上线 checklist。

### 教训 5：中文文档 Python 字符串嵌套 ASCII " 必炸

写 HELP 字典时 ASCII `"..."` 套 ASCII `"..."` → SyntaxError。
**铁律**：中文长字符串里强调内容用「」，不用 `"`。
**沉淀**：`mycc/memory/feedback_python_chinese_nested_quotes.md`

## 量化指标

| 维度 | Day 1 (5/20) | Day 2 (5/21) | 变化 |
|------|--------------|--------------|------|
| commits | 8 | 7 | -1 |
| 新增代码行 | 2710 | ~1400 | -48% |
| 新 ADR | 0 | 3（015/016/017） | +3 |
| 测试新增 | ? | 15（multimodal） | — |
| 全量测试通过 | ? | 30/1 skip | ✅ |
| LLM 调用延迟 | — | 4.91s（修复后） | 45s→4.91s（9x）|
| 川哥关键纠偏 | 1 次（B+A 并行） | 2 次（视觉范式 + UI 友好化）| +1 |

## 晚上扩展（17:00-21:30）：Phase β 5 砖 + γ-spec codex 4 轮严审

### 决策 4：Phase β 5 砖一气呵成（17:00-19:00）

cc 主线 + codex 5.3-codex high × 2 后台并行 3 路：
- A 路 cc 主线：β-1 EventTensor v1.1（ADR-018）
- B 路 codex A：β-2 parser intent_type 抽取
- C 路 codex B：β-4 graphiti SQLite 持久化

25 分钟完成原计划 2.5h。
真 LLM 端到端验证 5/5 = 100% intent_type 准确率（含反问/疑问边界）。
随后 cc 主线串行做 β-3 完整状态描述 JSON + β-5 multimodal 持久化（30min）。

### 决策 5：γ-spec base v3 实施 spec 双评审（19:00-19:30）

Agent 后台写 v0.1 初稿（534 行 / 7000 中文字 / 10 段齐备）。
派双评审：
- gemini 3.5 high 学术深审 → B 级 CONDITIONAL GO + 4 学术建议（VICReg / PHATE / Hierarchical CfC / TTA Tent）
- codex 5.4 high 工程对抗审 → 学术 6.5/10 + 工程 4.5/10 + 5 工程前置条件

川哥钦定原话：「能符合这个项目最初设定的目标，达到我的基于 LNN 的多模态识别与预测，就可以 go」。

cc 主线宏观裁决 GO 方案 A+（项目 4 维目标对账零妥协 + 修订项是目标护卫者）。

### 决策 6：codex 4 轮严审锁定 γ-spec v0.2.3（19:30-21:00）

| 轮 | spec | cc 自评 | codex 判定 | 抓 gap |
|----|------|---------|-----------|--------|
| 1 | v0.1 | — | CONDITIONAL GO | 5 学术 + 工程建议 |
| 2 | v0.2 | 5/5 全 A APPROVE | **⚠️ REVISE** | **6 条一致性矛盾** |
| 3 | v0.2.1 | 9/9 全修 APPROVE | **⚠️ REVISE** | **3 条新残留** |
| 4 | v0.2.2 | 残留 0 APPROVE | ✅ APPROVE | 0 阻塞 / 1 非阻塞 |

**4 轮共抓 14 处 gap，全部解决**。

cc 主线 reflection：CLAUDE.md §3.1「真正风险是 cc/老板懒得读对抗意见」实证 —— cc 三轮自评都过度乐观，幸亏 codex 多轮严审抓住。

### 教训 6：codex 多轮严审是真有效（mycc 新沉淀 5 feedback）

- `feedback_codex_multi_round_audit_value.md` — 4 轮 14 gap 实证
- `feedback_doc_consistency_gap_pattern.md` — 主体改了引用没改的盲区
- `feedback_pyfeat_062_api_pitfalls.md` — py-feat 3 bug 完整修
- `feedback_gemini_proxy_payload_size_limit.md` — gemini payload 过大失败
- `feedback_macro_goal_judgment_method.md` — 川哥宏观裁决法 SOP

## Day 3（5/22）启动条件

- [x] Phase α demo 跑通可用，川哥真人可体验
- [x] ADR-016 base v3 范式锁定（M1 启动 Step 1）
- [x] L4 W+ 三砖（demo / CLS / 记忆 UI）全部完工
- [ ] Phase β 启动：ADR-018 EventTensor schema 升级加 `multimodal_context` JSON 字段
- [ ] Phase β：LLM parser 加 intent_type 5 类抽取
- [ ] L4-bonus：UX Cmd+Shift+F 主动状态锚点

## Day 3 候选优先级

| 优先级 | 任务 | 工时 |
|-------|------|------|
| **P0** | Phase β schema 升级（ADR-018 + EventTensor + 持久化）| 0.5 day |
| **P0** | Phase β intent_type 抽取（5 类：陈述/疑问/反问/感叹/请求）| 0.5 day |
| **P1** | 收尾 Day 1-2 codex 限额时未做的事 | 待 codex 配额恢复 |
| **P1** | L4-bonus Cmd+Shift+F 锚点 | 0.5 day |
| **P2** | base v3 实施 spec 草稿（M1 启动 Step 1 前置）| 1 day |

## Day 2 团队致谢（全日 09:00-21:30 · 12h）

- **川哥**：4 次关键决策（视觉范式纠偏 + 方案 A+ 钦定 + 方案 X codex 多轮 + 项目目标宏观裁决法）
- **cc 主线**（Sonnet 4.6）：15 commits / 全部代码 + ADR + spec + retro / 5 轮自评认错复盘 / LNN-FIRST 守门员
- **codex 5.4 + 5.3-codex high**：γ-spec 4 轮严审 + Phase β β-2 / β-4 实施 + 抓 14 处 gap
- **gemini 3.5-flash high**：spec v0.1 学术深审 B 级 + 4 学术建议（含具体论文引用）
- **Agent 后台**：spec v0.1 初稿 534 行 / γ-spec 起草

---

**📝 复盘人**：cc | 2026-05-21 16:30
**📂 关联文件**：见 frontmatter `related` 字段
