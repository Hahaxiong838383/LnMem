---
adr-id: 018
title: EventTensor schema v1→v1.1 升级 — 加 intent_type + multimodal_context
status: accepted（2026-05-21 17:55，川哥钦定 team 模式 + Phase β 启动）
date: 2026-05-21
deciders: 川哥（产品守门员）+ cc（主线）+ codex-A/B（执行）
supersedes: —
related: ADR-016 base v3 multimodal self-supervised LNN / ADR-017 Phase α demo 升级 / docs/product-definition.md §10.1 #1 万能语法
---

## 上下文

ADR-017 Phase α demo 把 webcam multimodal V-A + AU + 9-state 用 `st.session_state.event_multimodal` 旁挂存储（不污染 EventTensor schema_v1 封版字段）。**当时记 ADR-017 cc/gemini 分歧 2**：gemini 建议加 `multimodal_context` JSON 字段到数据库，cc 主线坚持不动 schema_v1 红线 → 留 Phase β 走 ADR-018 规范升级。

川哥 Day 2 关键纠偏：
> 我是疑问句还是反问句？我是哭着说还是笑着说？LnMem 使用 LNN 神经网络，就是为了让 AI 懂并且能预测用户在输入需求或者沟通时的心情状态及心理活动。

需要让 EventTensor 同时承载：
- **句式意图**（陈述/疑问/反问/感叹/请求）— 文本维度的 input signal
- **视觉情绪**（V-A + AU + 9-state）— 视觉维度的 input signal

两者都是 ADR-016 base v3 多模态 LNN 的输入维度，**不是 LLM 分类标签**。

## 决策

### v1.1 加 2 个 EventTensor 顶级字段（向后兼容）

```python
@dataclass
class EventTensor:
    # v1.0 已有字段保留不变...
    user_id: str
    timestamp: datetime
    slots: Dict[SlotID, SlotValue] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    valid_until: Optional[datetime] = None
    raw_modalities: Dict[str, bytes] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    # v1.1 新字段（向后兼容，旧代码不传默认值兜底）
    intent_type: IntentType = IntentType.DECLARATIVE
    multimodal_context: Optional[Dict[str, Any]] = None

    schema_version: str = "v1.1"
```

### intent_type 5 类（IntentType Enum，单一来源 contracts.event_tensor）

| Enum | 中文值 | 示例 |
|------|--------|------|
| DECLARATIVE | 陈述 | 「昨晚 11 点和小李在公司吃了越南粉」|
| QUESTION | 疑问 | 「今天几号」|
| RHETORICAL | 反问 | 「这能怪我吗」|
| EXCLAMATION | 感叹 | 「太累了！」|
| REQUEST | 请求 | 「帮我订一下票」|

### intent_type 与 SlotID.INTENT 的正交关系

| 维度 | 字段 | 例 |
|------|------|---|
| **句式层面**（v1.1 新）| `EventTensor.intent_type` | 「我饿了？」→ RHETORICAL（反问）|
| **语义层面**（v1.0 已有）| `SlotID.INTENT (slot 7)` | 「我饿了？」→ "情绪表达"（语义意图）|

**两个独立维度**，不替换不冲突。

### multimodal_context schema

```python
# 可选字典，包含 ADR-017 SnapshotResult 全部字段
multimodal_context = {
    "v_a": [valence, arousal],          # Russell 二维 [-1, 1]
    "state_label": "relaxed",            # 9-state 英文
    "state_label_zh": "放松",            # 9-state 中文
    "au_vector": {"AU06": 2.5, ...},     # 20 AU 强度 [0, 5]
    "source": "mock" | "pyfeat+...",     # 数据源
    "confidence": 0.5,                    # 置信度 [0, 1]
}
```

不抓帧时 = `None`。

### 实现分工（Team 模式 5/21 17:30-17:55）

| 路 | 引擎 | 任务 | 产出 |
|---|------|------|------|
| **A** | **cc 主线** | EventTensor + IntentType + schema_version v1.1 + 14 单测 + ADR-018 | `contracts/event_tensor.py` + `contracts/tests/test_event_tensor_v1_1.py` |
| **B** | **codex 5.3-codex high** | parser intent_type 抽取（单次 LLM 调用 + fallback）+ 6 单测 | `universal_grammar_parser/parser_v0_llm.py` + `universal_grammar_parser/types.py` + `tests/test_parser_intent_type.py` |
| **C** | **codex 5.3-codex high** | graphiti SQLite 持久化（接口等价 InMemory）+ 6 单测 | `graphiti_zh_adapter/sqlite_backend.py` + `adapter.py` + `tests/test_sqlite_backend.py` |

### IntentType 单一来源去重（cc/codex A 冲突解决）

codex A 在 `universal_grammar_parser/types.py` 独立定义了 `IntentType` Enum，与 cc 主线 `contracts/event_tensor.py` 的定义重复。

**冲突按 CLAUDE.md §3.1 解决**：
- 默认采纳 codex 意见 ✗（这次 cc 不同意）
- cc 立场：contracts 是单一来源，parser 应 import 不应 redefine
- **决策**：删 `universal_grammar_parser/types.py` 的 IntentType 定义，改 `from contracts.event_tensor import IntentType` re-export
- 实测 56/56 测试通过 → 单一来源不破坏 parser 功能

## 验收（两门验证）

### Gate 1 Verify（功能目标）

- ✅ `pytest contracts/tests/test_event_tensor_v1_1.py` 14/14 通过
- ✅ `pytest universal_grammar_parser/tests/test_parser_intent_type.py` 6/6 通过（codex A）
- ✅ `pytest graphiti_zh_adapter/tests/test_sqlite_backend.py` 6/6 通过（codex B）
- ✅ EventTensor 向后兼容：旧代码不传 intent_type / multimodal_context 仍可构造
- ✅ schema_version 从 "v1" → "v1.1"（向后兼容主版本前缀检查）

### Gate 2 Guard（回归防护）

- ✅ `pytest apps/ contracts/ graphiti_zh_adapter/ universal_grammar_parser/` 56 passed / 1 skipped
- ✅ 不破坏现有 demo / e2e / CLS 任何接口
- ✅ `to_dict()` 输出含新字段（schema_version "v1.1"）
- ✅ `set()` 不可变更新保留新字段 + deep copy

## Codex 对抗审查吸收（cc 主线归档）

### codex A 5 翻车点（intent_type 路径）

| # | 风险 | cc 处理 |
|---|------|---------|
| 1 | LLM 返回格式漂移（非 5 类字符串） | 已有 fallback "陈述" + IntentType 严格枚举 |
| 2 | 5 类边界模糊（反问 vs 疑问）| 测试用川哥例句锚定边界 |
| 3 | 默认值 "陈述" 污染统计 | 加 source 字段标"是 LLM 真识别还是 fallback"留 Phase β |
| 4 | prompt 太长影响延迟 | 监控 LLM 调用时长，超 10s 报警 |
| 5 | 测试 mock 不稳 | 已用 mock LLM client，独立于真 endpoint |

### codex B 5 翻车点（SQLite backend）

| # | 风险 | cc 处理 |
|---|------|---------|
| 1 | sqlite3 thread safety（streamlit 多线程）| `check_same_thread=False` + WAL 已开 |
| 2 | WAL 锁冲突 | 已 PRAGMA journal_mode=WAL |
| 3 | datetime 序列化丢 tz | 已统一 ISO 8601 + UTC |
| 4 | JSON 字段查询无索引 | slot_filter 查询走 Python 端过滤，性能 M1 评估 |
| 5 | streamlit rerun 重连 connection | 默认 InMemory 不切，sqlite 路径用 connection pool 留 Phase β 优化 |

## 风险登记

| # | 风险 | 缓解 |
|---|------|------|
| 1 | EventTensor 字段过度膨胀 | v1.1→v1.2 必须经 ADR，v2.x 才大改 schema_v1 文档冻结字段 |
| 2 | parser fallback "陈述" 占比过高 | 加 source 字段（"llm_explicit" / "fallback"）便于后续监控 |
| 3 | multimodal_context schema 不固定 | M1 启动 base v3 前再发 ADR-019 锁定 multimodal_context schema |
| 4 | SQLite 单文件锁导致并发瓶颈 | M1+ daemon 流量大后切 PostgreSQL（schema 设计上 SQL 标准兼容）|

## 与不可妥协 6 项检查

| 项 | 状态 |
|---|------|
| §10.1 #1 万能语法 20 槽 | ✅ schema_v1 字段未动，只在 EventTensor 顶层加可选元数据 |
| §10.1 #2 多时间尺度 LNN | ✅ multimodal_context 为 base v3 input signal 铺路（ADR-016）|
| §10.1 #3 CLS | ✅ CLS 不动 |
| §10.1 #4 per-user 个性化 | ⏳ M3 |
| §10.1 #5 居家+车载双场景 | ⏳ M6 |
| §10.1 #6 记忆 UI 可改可删 | ✅ KG SQLite 持久化让删除真正落地 |

## 后续

- **β-3 demo 输出完整状态描述 JSON**（用 v1.1 字段，立刻可做）
- **β-5 multimodal_context 持久化**（已经 unblock，可立刻做）
- **ADR-019**：M1 前锁定 multimodal_context schema（base v3 启动前的最后一次升级机会）

---

**模板版本：v0.1 · 2026-05-03**
