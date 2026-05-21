---
title: "LnMem 技术架构全景速览"
audience: AI 同事（cc/codex/gemini）+ 工程师 + 投资人技术 due diligence
purpose: 一进 GitHub repo 5 分钟掌握技术全景
last_updated: 2026-05-21
related:
  - product-definition.md
  - principles/lnn-first.md
  - project-status.md
  - specs/base-v3-training-spec.md
体裁: L3（半结构化技术文档）
---

> **5 分钟读完，掌握 LnMem 完整技术架构 + 数据流 + 当前状态。**
> 详细请读 [`product-definition.md`](product-definition.md) §3 / [`principles/lnn-first.md`](principles/lnn-first.md) / [`specs/base-v3-training-spec.md`](specs/base-v3-training-spec.md)。

---

## 1. 系统全景一图

```
┌──────────────────────────────────────────────────────────────────┐
│  用户输入侧                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌────────┐  │
│  │ 中文文本输入 │  │ 摄像头抓帧    │  │ 音频（M1+）│  │ 时间戳 │  │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  └────┬───┘  │
│         │                │                 │              │       │
│         ↓                ↓                 ↓              ↓       │
│  ┌──────────────┐ ┌─────────────────┐  ┌────────┐  ┌─────────┐  │
│  │ Universal    │ │ py-feat AU 提取  │  │ 占位    │  │ 归一化  │  │
│  │ Grammar      │ │ + V-A 投影 v3   │  │         │  │         │  │
│  │ Parser       │ │ + 9-state       │  │         │  │         │  │
│  │ (LLM 拆 20槽 │ │                 │  │         │  │         │  │
│  │  + intent_t) │ │                 │  │         │  │         │  │
│  └──────┬───────┘ └────────┬────────┘  └───┬────┘  └────┬────┘  │
│         │                  │                │            │        │
│         └──────────┬───────┴────────────────┴────────────┘        │
│                    ↓                                              │
│         ┌──────────────────────────┐                              │
│         │ Multi-modal Fusion 52dim │ (24视觉+20槽+5intent+2时间+6音频占位)
│         └──────────┬───────────────┘                              │
│                    ↓                                              │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  心智核心（LNN-FIRST）                                            │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ CfC LNN Backbone（多时间尺度 LTC/CfC 微分方程）          │    │
│  │   ┌────────────────┐  ┌────────────────────┐            │    │
│  │   │ Layer 1: τ短   │  │ Layer 2: τ长       │            │    │
│  │   │ 0.1s~10s 快    │  │ 10s~600s 段落级   │ ← Phase γ   │    │
│  │   └────────────────┘  └────────────────────┘  目标       │    │
│  │   + Persona Conditioning (τ-bias + FiLM)                 │    │
│  │                ↓                                           │    │
│  │        64-dim hidden_state (主输出)                       │    │
│  └──────────────────────────────────────────────────────────┘    │
│         ↓                                                          │
│  5 head readout（附属，弱监督，backbone frozen）                  │
│    Head1 8-state | Head2 V-A | Head3 intent | Head4 conf | Head5 64d│
│                    ↓                                              │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  记忆层                                                            │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ EventTensor v1.1 (schema_v1.1)                            │    │
│  │   - 20 槽万能语法                                          │    │
│  │   - intent_type (5 类陈述/疑问/反问/感叹/请求)            │    │
│  │   - multimodal_context (V-A + AU + 9-state)              │    │
│  │   - hidden_repr 64-dim (M1+ 加，base v3 训完后)          │    │
│  └──────────────────┬───────────────────────────────────────┘    │
│                     ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Graphiti bi-temporal KG (SQLite 持久化 Phase β)          │    │
│  │   - t_valid + t_invalid 双时态                            │    │
│  │   - 永不物理删除（invalidate 逻辑失效）                   │    │
│  └──────────────────┬───────────────────────────────────────┘    │
│                     ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ CLS 夜间整合（每日 02:00 cron）                          │    │
│  │   当天 events → LLM 提炼 → 3 条今日精华（事实/情绪/未完成） │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                     ↓
                查询 / 失效 / 看精华
                Streamlit demo (apps/demo_e2e/main.py)
```

## 2. 三层 LNN-FIRST 架构（设计哲学）

| 层 | 名字 | 职责 | 实现 |
|----|------|------|------|
| **输入层** | Multi-modal Fusion | 视觉/文本/音频/时间统一 52-dim 向量 | `apps/demo_e2e/multimodal.py` + `universal_grammar_parser/` |
| **心智层** | LNN Backbone | 用 CfC 微分方程学连续 hidden state 演化（不是分类）| `poc/src/lnn/` (M0) / Phase γ base v3 (5/27+) |
| **记忆层** | bi-temporal KG + CLS | 事件持久化 + 夜间整合 + 用户可见可改可删 | `graphiti_zh_adapter/` + `scripts/cls_consolidate.py` |

**核心哲学**：与「LLM + RAG」对立——传统方案在对话时检索过去；LnMem 在永远在线的连续时间状态里持有当下，通过事件分割主动写入，通过夜间巩固主动学习。

详细见 [`principles/lnn-first.md`](principles/lnn-first.md) P-1 准则。

## 3. 多模态融合（输入层细化）

### 52-dim Fusion Vector（v0.2.1 锁定）

| 维度 | 来源 | dim | 实现状态 |
|------|------|-----|---------|
| **视觉 AU** | py-feat 真识别 20 AU + head pose | 24 | ✅ Phase α 上线（commit fc69ccb 3 bug 修复）|
| **万能语法 20 槽** | LLM parser 拆中文文本 | 20 | ✅ M0 W1 上线 |
| **intent_type 5 类** | LLM parser 同次调用抽（陈述/疑问/反问/感叹/请求）| 5 | ✅ Phase β β-2 上线 |
| **音频占位** | M1+ 加（语速 / 音调 / 沉默时长）| 6 | ⏳ M1+ |
| **时间戳归一化** | [sin(2πt/T_day), cos(2πt/T_day)] | 2 | ✅ |

详细 schema：[`contracts/event_tensor.py`](../contracts/event_tensor.py) v1.1

### Russell V-A 9-state 投影（视觉部分）

```
V-A 平面 9-state（1 neutral + 8 sectors × 45°）：
   90° A↑
       │
       │ 4 tense  │ 2 excited
   ────┼──────────┼──── V→ 0°
       │ 3 alert  │ 1 happy
       │
       │ 5 sad    │ 8 relaxed
   180°│──────────┼──── 360°
       │ 6 depressed│ 7 calm
       │
   270° A↓

r < r_neutral=0.35 → neutral（圆心）
r ≥ 0.35 → 按 θ 分扇区
```

实现：`apps/demo_e2e/multimodal.py::map_va_to_state_label`

## 4. base v1 / v2 / v3 LNN 范式演化

| 版本 | 范式 | 训练数据 | 输出 | 状态 |
|------|------|---------|------|------|
| **base v1** | 30 虚拟主人 + 1 真人 supervised 训练 | 合成数据 + 川哥 daemon 早期 | 9-state 分类 | ✅ 上线（ckpt_sha8=10cc41c3）但川哥真人 OOD 0% acc |
| **base v2** | DFAT 微调（freeze backbone + restore FiLM）| VEATIC 6GB 子集 + AU parquet | 9-state CE + Russell margin | ❌ 64/64 全 collapse neutral（ADR-013）|
| **base v2.1** | DFAT + σ 可学 + class-balanced + 视觉冻结 | 同 v2 | 同 v2 | ❌ 89.4% mismatch（ADR-015）|
| **base v3（Phase γ）** | **Multi-modal self-supervised LNN + weakly-supervised readout** | VEATIC 6GB 主 + EMOTIC 3.4GB + mst 387MB | **64-dim hidden_state + 5 head readout** | ⏳ 5/27-6/10 训（ADR-016 + spec v0.2.3）|

**base v3 4 个 loss**（spec v0.2.3 codex 4 轮 APPROVE）：
- L1 Temporal Contrast（InfoNCE / VICReg γ-1 spike 决策）weight=1.0 主线
- L2 Cross-modal Contrast weight=0（M1+ 加）
- L3 Persona Contrast weight=0 默认关（VEATIC viewer ≠ user persona）
- L4 Weakly-supervised Readout weight=0.5（backbone frozen）

**评估主指标 v0.2.1 三联**（禁 accuracy 主指标）：
- M1 KG 召回 Top-5 ≥ 40%
- M3 时序一致性 ratio ≥ 1.5
- M4 Frozen-feature Linear Probe + M5 视频 ID Shortcut 反测 < 30%

## 5. Graphiti bi-temporal KG（记忆层）

### 双时态语义

```
KGNode {
  t_valid: "2026-05-21 11:00 UTC"   # 事件发生时间
  t_invalid: null                    # 失效时间（None=永远 valid）
  slots: { 20 槽 }
  intent_type: "陈述"                # v1.1 加
  multimodal_context: {...}          # v1.1 加
}
```

### Invalidate 不删数据

用户点 demo 删除按钮 → `adapter.invalidate(event_id, t_invalid=now)`
节点 `t_invalid` 字段更新，**节点本身保留**，查询时按 `valid_at` 时间窗自动过滤。

### Backend

| Phase | 实现 | 持久化 |
|-------|------|--------|
| Phase A (M0 W2) | InMemoryGraphitiBackend | 重启丢 |
| **Phase B (Phase β β-4) ✅** | **SQLiteGraphitiBackend** | **持久化** + WAL + thread-safe |
| Phase C (M1+) | graphiti-core 真接入 | 全图操作 |

## 6. CLS 夜间整合（互补学习）

每日 02:00 cron 跑 `scripts/cls_consolidate.py`：

```
events 表 (当天) → LLM gemini-3-flash-preview → 3 条今日精华:
  ┌─────────┐
  │ 事实    │ 「今日和小李在公司吃了越南粉」
  │ 情绪    │ 「整体情绪平稳但带轻微挫败感」
  │ 未完成  │ 「明天 10 点与王总评审会」
  └─────────┘
        ↓
daily_summaries 表（长期存储）
        ↓
demo 渲染（apps/demo_e2e/main.py::render_cls_panel）
```

## 7. demo 当前状态（Phase α + Phase β 上线版）

```
http://localhost:8501 (Streamlit)
├─ 顶部 📖 这个 demo 在干什么？(4 步流程 + 11 术语速查)
├─ 左侧
│   1️⃣ 中文文本输入
│   2️⃣ 📸 抓帧 (py-feat 真识别推荐 / mock 占位演示)
│   3️⃣ 🚀 解析 + 写 KG (~5s)
│      → 🎯 完整状态描述 JSON (6 维)
│        - text_raw
│        - universal_grammar_20_slots (P0-P3)
│        - intent_type (5 类)
│        - emotional_state (V-A + 9-state + 20 AU)
│        - lnn_hidden_state (Phase γ placeholder)
│        - follow_up_suggestions (M1+ placeholder)
└─ 右侧
    4️⃣ 🔎 查询 KG (按用户/时间/槽位匹配)
       → 命中卡片 (含多模态 state + 20 槽 + 🗑 失效按钮)
    📊 KG 节点总览 (SQLite 持久化)
    🌙 CLS 夜间整合 (今日精华 3 条)
```

## 8. 项目阶段与里程碑

| 阶段 | 日期 | 内容 | 状态 |
|------|------|------|------|
| M0 W1-W2 | 5/04-5/19 | 万能语法 / parser / KG skeleton / synthetic_data / base v1/v2 实验 | ✅ |
| M0 W3 Day 1 | 5/20 | Team 模式 + B+A 并行 + ADR-007 Phase 1.2-1.4 | ✅ |
| M0 W3 Day 2 | 5/21 | L4 三砖 + Phase α + Phase β 全完 + γ-spec v0.2.3 codex 4 轮 APPROVE | ✅ |
| **修订周** | 5/22-5/26 | γ-1/γ-2/γ-3 spike + 10-video preflight 11 项 | 🚀 明天启动 |
| **Phase γ** | 5/27-6/10 | base v3 self-supervised 预训练（Studio）| ⏳ |
| Phase δ | 6/11+ M1 起 | 川哥真人 daemon 60w 帧微调 + per-user 三件套 | ⏳ |
| M4 MVP α | 9-10 月 | 居家场景 Web/App + 10 名种子用户 | ⏳ |

详细路线：[`../roadmap/milestones.md`](../roadmap/milestones.md)

## 9. AI 同事协作纪律

| 场景 | 必做 |
|------|------|
| 写代码前 | 三道门（Protocol → 测试 → codex 对抗审）|
| 改 EventTensor schema | 红线，写 ADR + 川哥签字 |
| 任何 §10.1 不可妥协项妥协建议 | 停下问川哥 |
| 重要 spec / ADR | **派 codex 多轮严审**（实测 4 轮抓 14 处 gap）|
| 文档修订后 | grep 残留语义 + 派 codex 验收 |
| LLM 调用 timeout | reasoning_effort=minimal + max_tokens=4096 |
| 长跑服务路径 | `Path(__file__).resolve().parents[N]` 不用相对 cwd |

详细：[`../CLAUDE.md`](../CLAUDE.md) §3 三道门 + §3.1 cc/codex 冲突诊断

## 10. 一图速懂全栈

```
[Python venv: .venv-poc-fresh / Python 3.9]
  contracts/                  ← schema_v1 + EventTensor v1.1（红线封版）
  universal_grammar_parser/   ← LLM 拆 20 槽 + intent_type
  graphiti_zh_adapter/        ← bi-temporal KG (InMemory + SQLite)
  apps/demo_e2e/              ← Streamlit demo (Phase α/β)
  scripts/
    cls_consolidate.py        ← CLS 夜间整合
    e2e_demo.py               ← CLI 端到端测
    extract_au_to_parquet.py  ← py-feat AU 批量提取
    train_base_v2.py          ← base v2 训练（v3 即将替代）
    spike_lnn_realtime_infer.py ← LNN 实时推理 spike
  poc/                        ← LNN base ckpts + daemon
    src/lnn/                  ← CfC + τ-bias + FiLM (M0)
    data/checkpoints/         ← base v1 ckpt (10cc41c3)
  data/
    cls.sqlite                ← CLS daily_summaries（25KB）
    kg.sqlite                 ← Phase β-4 KG 持久化（占位）
  docs/                       ← 产品定义 + ADR + spec + retro
    decisions/                ← 19 ADR
    specs/                    ← base-v3 spec v0.2.3
    retrospectives/           ← Day 1/2 复盘
    principles/lnn-first.md   ← P-1 最高准则
```

---

**📝 维护者**：cc 主线（LNN-FIRST 守门员）| **生效**：2026-05-21 21:30
