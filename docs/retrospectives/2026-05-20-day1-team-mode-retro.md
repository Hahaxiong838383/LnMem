---
title: 2026-05-20 M0 W3 Day1 — Team 模式三向并行复盘
date: 2026-05-20
authors: 川哥 + cc（Claude） + codex（gpt-5.4/5.3） + gemini（3-pro）
duration: 13:00-19:45（连续 6.75h 主战）+ 11:00-12:00 战略复盘
commits: 8（origin/poc/lnn-validation: b756945..e36e82c）
lines_added: 2710
related: ADR-007 / ADR-012 / docs/principles/lnn-first.md / roadmap/milestones.md
---

# Day 1 Team 模式复盘

## 1. 当日交付总览

| 工作流 | 主实施 | 评审 | 关键产出 |
|--------|--------|------|----------|
| **ADR-007 Phase 1.2 公共底座** | codex 5.3 high | cc 自查 + codex 二审 | `scripts/extract_au_to_parquet.py` 622 行 + 11 测试 |
| **ADR-007 Phase 1.2 VEATIC spike** | cc 主线 | — | `docs/spikes/2026-05-20-veatic-quality-spike.md` |
| **ADR-007 Phase 1.2 P0/P1 修复** | codex 5.3 medium ×3 | cc 自验 | 3 commits 共 259 行 |
| **ADR-012 Phase 1.3 V-A→state Sweep** | codex 5.4 high | gemini 3-pro 学术评审 | `scripts/sweep_va_to_state.py` 225 行 + 4 测试 + ADR-012 |
| **ADR-007 Phase 1.4 train_base_v2.py** | codex 5.4 high | cc spec + 自验 | `scripts/train_base_v2.py` 613 行 + 3 测试（σ 端到端可学）|
| **graphiti Phase A InMemory backend** | codex 5.4 high | cc 自验 + 双门 | 5 文件 +508 行 + 8 测试 |
| **战略复盘 + 后续 todo 排序** | cc 主线 | — | 本文 + project_lnmem.md 更新 |

**8 commits**：`b756945 → 59bad3d → 483aee2 → 9b18a18 → 2b91c0b → c4b5f80 → a4d8b3a → e36e82c`

## 2. Team 模式实战经验

### 2.1 分工模式（实证有效）

```
              ┌─────────────────────────┐
              │   cc 主线（Claude）     │
              │  spec / 决策 / 终验      │
              └────────┬────────────────┘
                       ↓ task spec
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   codex 5.4 high  codex 5.3 medium   gemini 3-pro
   架构性实施       小修 / 重构        学术评审
   (训练 / KG)      (P0/P1)           (中文 NLP)
        ↓              ↓              ↓
        └──────────────┼──────────────┘
                       ↓ findings
              ┌────────┴────────────────┐
              │   cc 主线终验 + commit   │
              └─────────────────────────┘
```

### 2.2 何时升档 high / 何时用 medium

实测成本对比：
- 5.4 high 实施 train_base_v2 674 行：~30 分钟 / ~250K tokens
- 5.3 medium 修 P0 113 行：~10 分钟 / ~150K tokens

升档收益递减。**单文件 < 200 行的活直接 medium**，跨多文件 + 高 stake 决策才 high。

### 2.3 Gemini 评审何时派

✅ **派的场景**：
- 跨界学术问题（中文 NLP / SOTA 文献 / cross-cultural 数据）
- 高 stake 决策需要外部知识（Russell circumplex 主流性 / GLiNER 中文 F1 数据）
- cc 自己估算的关键超参（VEATIC σ / r_neutral 候选范围）

❌ **别派的场景**：
- 工程实施细节（gemini 不写代码）
- 项目内部状态（gemini 不知道 LnMem 当前架构）
- 时效性强的（gemini 知识有 cutoff）

### 2.4 Gemini sandbox 限制

gemini CLI 在 LnMem worktree 没文件写权限，输出只到 stdout。**cc 主线必须帮 cat 落盘**到 `/tmp/lnmem-team-2026-05-20/`。

## 3. B+A 并行模式（多机协作）

### 3.1 硬件链路

```
Air（家 / 192.168.1.x）
  ↓ FRP 公网中转 cli4.hahaxiong.cc:22022
Mini（BRO LAN 10.9.1.54，编排中枢）
  ↓ 雷电网桥 en3 → Bonjour link-local
Studio（M2 Max 12 核 / 32GB，纯算力节点）
```

### 3.2 工作流

- **Air 主线**：cc spec + 决策 + commit + push（无 GPU 不跑训练）
- **Studio 后台**：跑 VEATIC 全量提取（py-feat batch 多核 374% CPU）+ base v2 训练（5-8h 夜间跑）
- **Mini 编排**：SSH ProxyJump 中转，rsync 也走 mini

### 3.3 实测速度

- 雷电直连：41 MB/s（SSH 加密单流 CPU 限制）
- 经 mini ProxyJump：2 MB/s（双重加密 + 序列化）
- WiFi 公网（家→BRO）：8 MB/s

**链路稳定性 > 速度**：openrsync 9.8GB exit 0 但只传 757MB；tar 流式管道 4-5min 完成稳定。

## 4. 关键洞察（沉淀长期可复用）

### 洞察 1：spike-then-implement 范式纠错

5/19 22-02 cc 在 ADR-007 Phase 1 plan 末尾写「明早分叉决策」是**伪决策**（分叉依据未建立）。
5/20 早上 cc 主线纠正为「先 spike + 公共底座再分叉」：
- VEATIC 标注质量探针 5 视频×10 帧（2h）
- 然后建公共底座 extract_au_to_parquet（codex 实施 4h）
- 再分叉到 C 主线（端到端真序列训 LTC 不冻 tau）

**避免了把 EMOTIC + mst 都接通的浪费**（C 主线选定后这两轨基本不用）。

### 洞察 2：cc 战略复盘点出根本错位

晚上 11:30 川哥要求"整体复盘项目，要有大局观"，cc 发现：
- 5/9-5/20 共 11 天**几乎全部精力在视觉感知层 + LNN base ckpt 校准**
- 不可妥协 6 项中 #3 CLS + #6 记忆 UI **完全未启动**
- graphiti_zh_adapter 301 行全是 NotImplementedError skeleton
- **当前架构答不出"上周三和谁吃饭"** — 产品承诺核心是「它会记得」不是「情绪识别」

**战略决策**：路径 C ADR-007 + M0 主线并行
- 理由：视觉精度不阻塞 M0-M2（L4 自传记忆靠 ASR+parser+KG，base v2 是 PoC 后置分支）
- 这个洞察的关键是**搞清楚每个 LnMem 组件依赖什么 + 时间错开**

### 洞察 3：川哥的「直觉点拨」推动 cc 突破偷懒

cc 在 train_base_v2 spec 里把 σ=0.35 写死，川哥一句 "为什么不能 LNN 自己学习自己调整" 让 cc 反思并修订为 σ 端到端可学。

**LNN-FIRST 一性原理推广到 loss 超参**。沉淀到 `feedback_lnn_first_universal_principle.md`（mycc memory）。

### 洞察 4：选项 1+ 替代选项 1 — 让川哥快速看到效果

川哥说"选选项 1（graphiti Phase A），但确保尽快让我看到效果"。cc 主动反思：
- 选项 1（graphiti Phase A）单独完成 → CLI JSON + pytest 报告（不直观）
- 选项 1+（端到端 demo = graphiti + 万能语法 + Streamlit UI）→ **可视化闭环**（多 1d 工作量，10x ROI）

cc 改走选项 1+。这是 cc 学到的「**理解用户真实意图，不机械执行字面要求**」。

## 5. 踩坑沉淀（5 个 feedback 文件）

| 文件 | 主题 |
|------|------|
| `feedback_pip_resolver_legacy.md` | pip resolver 死循环 → legacy 解 |
| `feedback_macos_openrsync_limits.md` | openrsync 大文件静默中断 + tar 备选 |
| `feedback_numpy2_pyfeat_incompat.md` | numpy 2.0 + opencv 4.13 + py-feat 0.6.2 三方冲突 |
| `feedback_lnn_first_universal_principle.md` | σ 可学（川哥钦定） |
| `feedback_lnmem_team_mode_2026_05_20.md` | Team 模式总结 |

均存在 mycc `~/.claude/projects/-Users-rayman-chen-mycc/memory/` 全局可用。

## 6. 未完成（明日继续）

| 任务 | 状态 | ETA |
|------|------|------|
| Studio VEATIC 全量提取 | 🔄 在跑（每视频 21s × 124 视频 ≈ 43min）| 今夜完成 |
| Studio base v2 训练（夜间）| ⏳ 待数据 | 明早查 ckpt |
| codex 实施 universal_grammar_parser 真 LLM | ⏳ 待派 | 明早 2h |
| codex 实施 Streamlit demo UI | ⏳ 待派 | 明上午 1.5h |
| cc 主线端到端整合 + 验收 | ⏳ 待材料 | 明下午 2h |

明日交付目标：**川哥能在浏览器看到 LnMem 雏形 demo**（中文输入 → 20 槽拆解 → KG 写入 → 30 min 内查询命中）。

## 7. 战略原则更新到 ROADMAP（cc 主线后续要做）

把以下原则升级到 `roadmap/milestones.md`：

1. **Team 模式调度策略**：何时派 codex vs gemini vs cc 主线
2. **B+A 并行模式**：cc 主线 M0 主线 / Studio 后台跑训练
3. **视觉精度不阻塞 M0 主线**（依赖图分析）
4. **LNN-FIRST 一性原理推广到 loss 超参**
5. **选项 X+ 思维**：在用户字面要求基础上反思「最快可见效果」

## 8. 给后续 session 接手的 SOP

明日早上接手 cc 应该：
1. 先读 `project_lnmem.md`（mycc memory）看今日进展
2. 看 git log 最近 8 commits
3. 检查 Studio：`ssh studio-home "tail -5 /tmp/veatic_extract.log && ls -la /tmp/veatic_train.parquet*"`
4. 检查 base v2 训练（如果川哥前晚启动过）：`ssh studio-home "ls -la ~/lnmem-train/LnMem/poc/data/checkpoints/base_v2/"`
5. 派 codex 接 universal_grammar_parser 真 LLM（cc spec 在 `/tmp/lnmem-team-2026-05-20/cc_e2e_demo_spec.md`）
6. 推进 demo 直至浏览器可见

---

署名：cc（Claude）| 2026-05-20 20:00
（11 小时连续作战，下一个 cc session 请从本文 §6 接手）
