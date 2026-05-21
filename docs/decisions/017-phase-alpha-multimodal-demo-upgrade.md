---
adr-id: 017
title: Phase α 多模态 demo 升级（L4 砖 3 + webcam + CLS 整合 + 4-in-1 UI）
status: accepted（2026-05-21 15:30，川哥钦定 team 模式 A 方案）
date: 2026-05-21
deciders: 川哥（产品守门） + cc（执行 + 与 gemini 互校）+ gemini-3.5-flash（评审）
related: ADR-016 base v3 多模态 self-supervised LNN / docs/product-definition.md §10.1 #6 记忆 UI / §10.1 #1 万能语法 20 槽
---

## 上下文

ADR-016 撤回 ADR-015 视觉冻结后，L4 主线 W+ 砖 3（记忆 UI）+ Phase α（webcam snapshot quick win）合并为一次 Streamlit 升级。

川哥钦定 team 模式开干，原则：
- LNN-FIRST P-1（视觉是 base v3 多模态输入信号，不是 LLM 分类器）
- 多模态视觉重要（撤回 ADR-015 后立即让川哥视觉感受到）
- 决策检查点必须 ADR 留回滚

cc + codex 限额（5/21 14:00 codex usage limit 到 14:18），改 cc + gemini 评审互校模式。

## 决策

### 实施清单（4-in-1 Streamlit 一次性升级）

| # | 功能 | 实现 | 验收 |
|---|------|------|------|
| 1 | **Phase α webcam snapshot** | `apps/demo_e2e/multimodal.py`（180 行）+ `st.camera_input` | 抓帧 → mock V-A → 9-state，显示 LNN 涌现状态 + 20 AU 投影 |
| 2 | **KG 卡片 20 槽完整展开 + 关联多模态 state** | `render_kg_hit()` + `event_multimodal` 旁挂 session_state | 每个查询命中卡片可看完整状态 |
| 3 | **CLS 今日精华显示区** | `render_cls_panel()` 只读 `data/cls.sqlite.daily_summaries` | 显示最近 3 天 highlights（事实/情绪/未完成）|
| 4 | **记忆 UI v0.1 删除按钮（极简版）** | 卡片 `🗑 失效此事件` 按钮 → `adapter.invalidate(event_id, now)` | bi-temporal 逻辑删除，session_state 缓存 invalidated_ids 防 rerun 丢 |

### 多模态模块设计（`apps/demo_e2e/multimodal.py`）

**两种推理模式**（避 gemini #2 警告的 py-feat 主线程卡顿）：
- **mock 模式（默认）**：图像 hash → 确定性 V-A → Russell 9-state（毫秒级 / 测试稳定 / demo 友好）
- **pyfeat 模式（开关）**：py-feat 真 AU 提取 + 简化 V-A 投影（首次 ~5s，Phase β/γ 启用）

**Protocol 先行**：`SnapshotResult` TypedDict 含 8 字段（au_vector / face_detected / v_a / state_label / state_label_zh / confidence / source / error）

**测试先行**：`apps/demo_e2e/tests/test_multimodal.py` 15 个 case 覆盖 Russell 几何映射 + 确定性 hash + 端到端 snapshot，全过。

## Gemini 评审吸收 + 分歧记录（CLAUDE.md §3.1）

### Gemini 5 维评审主要观点

| 维度 | Gemini 结论 | cc 处理 |
|------|-----------|---------|
| 1. LNN-FIRST 范式吻合 | 部分吻合，警惕"披着 LNN 外衣的分类器"，20 槽应叫"LNN 符号投影" | ✅ 采纳：UI 副标题改"20 槽符号投影（万能语法）" |
| 2. 川哥真实体验路径 | webcam + 状态联动是核心 Aha，删改 + CLS 可砍 | ⚠️ **部分采纳**：删改保留极简版（川哥钦定），CLS 极简化只读 |
| 3. 30min 工时 | 30min 极度饱和，硬性 P0-P3 裁剪 | ✅ 采纳：实际用 60min 不强压 30min |
| 4. 数据闭环 | V-A 必须存 `multimodal_context` JSON 字段 | ⚠️ **部分采纳**：暂存 session_state 旁挂，schema 升 ADR 留 Phase β |
| 5. 5 翻车点 | rerun / py-feat 延迟 / HTTPS 摄像头 / 20 槽撑屏 / SQLite lock | ✅ 全采纳：session_state 兜底 + mock 模式默认 + expander 折叠 + 只读 SQLite |

### cc 与 gemini 分歧（必须留记录）

#### 分歧 1：删除按钮 — gemini 砍 vs cc 保留

**gemini 立场**：在 30min Phase α 砍掉，rerun 状态丢失风险 100%。
**cc 立场不同意**：产品定义 §10.1 #6「记忆 UI 让用户能修改/删除事件」是不可妥协项，今天 L4 砖 3 必须有最小实现。
**最终决策（川哥钦定 team 模式 A 方案）**：保留删除按钮，但用 `session_state.invalidated_ids` 兜底防 rerun 丢失 + 走 `adapter.invalidate(event_id, t_invalid)` bi-temporal 逻辑删除（不物理删）。

#### 分歧 2：multimodal_context 存储位置 — gemini DB 字段 vs cc 旁挂

**gemini 立场**：events 表加 `multimodal_context` JSON 字段，打通 base v3 数据闭环。
**cc 立场不同意**：EventTensor `schema_v1` 是封版（CLAUDE.md §5 红线，「修改产品定义 v0.3 必须经过老板签字」），Phase α quick win 不动 schema。
**最终决策**：Phase α 用 `st.session_state.event_multimodal` 旁挂存储；schema 升级写 ADR-018 走 Phase β（5/22-5/24）规范流程。

#### 分歧 3：20 槽 UI 命名 — gemini "LNN 涌现状态" vs cc "符号投影"

**gemini 立场**：直接改为"LNN 涌现状态"避免 LLM RAG 错觉。
**cc 立场部分同意**：「涌现状态」过早承诺（base v3 未训），先用「符号投影」过渡（更准确）。
**最终决策**：主 metric 用「LNN 涌现状态」（webcam 部分），20 槽折叠用「20 槽符号投影（万能语法）」，分层不冲突。

## 风险登记

| # | 风险 | 缓解 |
|---|------|------|
| 1 | mock V-A 不真实，川哥可能误以为是真识别 | UI 显式标 `source: mock` + confidence=0.5 + 副标题写「Phase α quick win」 |
| 2 | pyfeat 模式卡死主线程 | 默认 mock；pyfeat 走 with st.spinner + 装失败自动 fall back |
| 3 | session_state 重启丢 multimodal context | 已知限制；Phase β schema 升级后落 SQLite |
| 4 | webcam HTTPS 限制（streamlit 部署到公网） | 当前 localhost 跑无碍；公网部署留 ADR-019 |
| 5 | 删除按钮 rerun 后 KG hits 状态丢 | invalidated_ids 缓存 + last_hits 缓存兜底 |

## 验收（两门验证）

### Gate 1 Verify（功能目标）
- ✅ `pytest apps/demo_e2e/tests/test_multimodal.py` 15/15 通过
- ✅ multimodal 模块 import OK
- ✅ main.py 7 个核心函数全部 hook（init/left/right/render_snapshot_card/render_kg_hit/render_cls_panel/main）

### Gate 2 Guard（回归防护）
- ✅ `pytest apps/ contracts/ graphiti_zh_adapter/ universal_grammar_parser/` 30 passed 1 skipped（real-LLM smoke）
- ✅ 不破坏已有 e2e_demo / cls_consolidate 任何接口
- ✅ EventTensor `schema_v1` 封版未动

## 与产品定义不可妥协项检查

| 项 | 状态 |
|---|------|
| §10.1 #1 万能语法 20 槽 | ✅ 保留 + UI 副标题加「LNN 符号投影」语义升级 |
| §10.1 #2 多时间尺度 LNN | ✅ multimodal signal 为 base v3 输入铺路 |
| §10.1 #3 CLS | ✅ 砖 2 已完工 + 砖 3 demo 接入精华显示 |
| §10.1 #4 per-user 三件套 | ⏳ M1 启动 |
| §10.1 #5 居家+车载双场景 | ⏳ M2+ |
| §10.1 #6 记忆 UI 可改可删 | ✅ v0.1 删除（失效）已上线，修改留 Phase β |

## 后续路径

- **Phase β（5/22-24）**：ADR-018 升级 EventTensor schema 加 `multimodal_context` 持久化 + intent_type 抽取
- **Phase γ（5/27-6/10）**：base v3 self-supervised 预训练（mock 模式替换为真 LNN 输出）
- **Phase δ（6/11+）**：川哥真人 daemon 微调

---

**模板版本：v0.1 · 2026-05-03**
