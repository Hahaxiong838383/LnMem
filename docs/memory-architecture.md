# LNN-Centric Memory Architecture (LnMem) · Technical Design v0.1

> **如果你是 Claude Code / Codex 第一次读这份文档**：先读 [`product-definition.md`](product-definition.md) §5（约 200 行），它给你"为什么这样做"的产品视角；然后回来读这份，它给你"怎么做"的工程视角。两份配合理解，你就有了完整骨架。
>
> 编制：cc + 老板 + claude code（vibecoding 协作） | 2026-05-03 | 配套产品定义 v0.3

---

## 0. 30 秒摘要

LnMem = **以液态神经网络（LNN）为心脏的三层记忆系统**。LNN 不是"模型之一"，是**贯穿所有时间尺度的中枢神经**——L1 感知（秒）→ L2 工作记忆（分钟）→ 通过事件边界门控写入 L2 情景记忆（Graphiti KG，小时-天）→ 夜间 CLS consolidation 上卷到 L3 语义/人格记忆（LoRA + 偏好向量，周-年）。万能语法 20 槽 schema 是横切所有层的**事件张量**统一格式。

**与传统"LLM + RAG"记忆方案的本质区别**：传统方案在对话时检索过去；LnMem 在**永远在线的连续时间状态**里持有当下，通过事件分割主动写入，通过夜间巩固主动学习。

---

## 1. 为什么 LNN 是记忆系统的"心脏"——而不是 LSTM / Transformer

记忆系统的工程需求有三个：(a) 永远在线、(b) 多时间尺度、(c) 端侧可部署。LNN（Liquid Time-Constant / Closed-form Continuous-time，详见研究 §1.1）相对其他选项的关键优势：

| 维度 | LSTM | Transformer | **LTC/CfC（LNN）** |
|------|------|-------------|------------------|
| **时间动力学** | 离散步进 | 离散步进 | **连续时间 ODE，τ 可学** |
| **多时间尺度** | 需多层堆叠 | 需多层堆叠 | **单元级 τ 学习，天然多尺度** |
| **端侧延迟** | 中 | 高（自注意力 O(n²)） | **低（Loihi-2 实测 15.2ms/帧）** |
| **小神经元稳定性** | 差（梯度消失） | 差（需大量参数） | **64+32 神经元稳定（ncps 实测）** |
| **记忆系统适配** | 隐状态需手动管理 | 全靠注意力 + KV cache | **隐状态本身就是"当下"的连续表示** |

**核心洞察**：LSTM/Transformer 把记忆当"上下文窗口里的 tokens"——要查就得检索。LNN 把记忆当"心智的连续状态"——它**就在那里**，无需检索，因为状态本身在演化。

**Why 这是产品差异化的算法基础**：永远在线 + 跨场景同一个"它"——这两个产品承诺都依赖"心智状态不中断"。LSTM/Transformer 关掉就清零；LNN 状态可以序列化保存、跨设备恢复、跨会话延续。

---

## 2. 整体数据流（LNN 视角）

```
┌─────────────────────────────────────────────────────────────┐
│  Multi-modal Sensors                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │
│  │ Audio   │  │ Vision  │  │ Text    │  shape: per modality │
│  │ Fun-ASR │  │ Qwen3-VL│  │ direct  │                      │
│  └────┬────┘  └────┬────┘  └────┬────┘                      │
│       └────────────┼────────────┘                            │
│                    ▼                                         │
│       multimodal_embedding [B, T, D=768]                     │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────────────┐
│  L1 LNN · 感知层 (τ ≈ 1s, 64 neurons)                       │
│  input: [B, T, 768] → LTC/CfC cell → hidden [B, T, 64]      │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────────────┐
│  L2 LNN · 工作记忆层 (τ ≈ 60s, 32 neurons)                  │
│  input: [B, T, 64] → LTC/CfC cell → hidden [B, T, 32]       │
└─────┬──────────────────────────┬────────────────────────────┘
      │                          │
      ▼                          ▼
┌──────────────┐       ┌────────────────────────┐
│ Linear Probe │       │ Event Boundary Gate    │
│ → 20 slots   │       │ → boundary? {0,1}      │
│   confidence │       └────────┬───────────────┘
└──────┬───────┘                │ if boundary=true
       │                        ▼
       │              ┌────────────────────────┐
       └─────────────►│ EventTensor 组装        │
                      │ + Overflow Capsule      │
                      └────────┬───────────────┘
                               ▼
                      ┌────────────────────────┐
                      │ L2 情景记忆 · Graphiti  │
                      │ KG (bi-temporal)        │
                      └────────┬───────────────┘
                               │
                               ▼ (夜间)
                      ┌────────────────────────┐
                      │ CLS Consolidation       │
                      │ 高强度事件 replay       │
                      └────────┬───────────────┘
                               ▼
                      ┌────────────────────────┐
                      │ L3 语义/人格记忆        │
                      │ LoRA + 偏好向量          │
                      └────────────────────────┘
```

**张量形状约定**（B=batch, T=time steps, D=feature dim）：
- 多模态输入：`[B, T, 768]`（统一编码到 768 维）
- L1 隐状态：`[B, T, 64]`
- L2 隐状态：`[B, T, 32]`
- EventTensor 槽位：`[B, 20, 4]`（20 槽 × {value_emb, confidence, position, timestamp}）

---

## 3. LNN 心智核心规格

### 3.1 L1 · 感知层（τ ≈ 1s, 64 neurons）

| 字段 | 值 |
|------|----|
| **库** | `ncps` (mlech26l/ncps) PyTorch backend |
| **类型** | CfC (Closed-form Continuous-time)，比 LTC 推理快 2-3× |
| **神经元数** | 64 |
| **τ 范围** | 0.5s – 5s（可学习） |
| **输入** | 多模态融合 embedding `[B, T, 768]` |
| **输出** | 隐状态序列 `[B, T, 64]` |
| **训练目标 v0** | next-event prediction（self-supervised） |
| **训练目标 v1** | + 槽位填充 auxiliary loss |
| **目标延迟** | < 50 ms（RTX 4090 / Jetson Orin Nano） |

**Why 64 neurons**：研究文档 §1.1 给出 Loihi-2 上 LTC 神经元数与端侧延迟的实测。64 是"够稳定 + 够小"的最低妥协点；32 不够稳，128 在 Jetson 上超 50ms。

### 3.2 L2 · 工作记忆层（τ ≈ 60s, 32 neurons）

| 字段 | 值 |
|------|----|
| **库** | `ncps` CfC + `mixed_memory=True` |
| **神经元数** | 32 |
| **τ 范围** | 30s – 600s（可学习） |
| **输入** | L1 隐状态 `[B, T, 64]` + 当前 EventTensor `[B, T, 80]`（20×4 flatten） |
| **输出** | 隐状态序列 `[B, T, 32]` |
| **训练目标** | 与 L1 联合训练（end-to-end），通过 L2 → linear probe → 槽位置信度向量 |

**Why 把当前 EventTensor 喂回 L2**：让工作记忆能"看到"自己刚才填了什么槽——这是"追问能跟住"产品承诺（v0.3 §5.5 M1）的算法基础。

### 3.3 训练数据来源（M0-M3）

| 来源 | 占比 | 用途 |
|------|------|------|
| 类型 1 合成对话（DeepSeek + Qwen + Claude + GPT 四模型联合） | 40-50% | LNN 主预训练 + 槽位监督 |
| 类型 2 长期记忆轨迹（50 虚拟主人 ×30+ 天） | 30-40% | L2 时间动力学 + CLS pipeline 测试 |
| 类型 3 真实交互（脱敏 + 老板一审） | ≤ 20% | 长上下文连续性金标准 |

**硬约束**：模型间交叉一致率 ≥ 75%（低于此线整批丢弃，详见 [`product-definition.md`](product-definition.md) §6.3）

---

## 4. LNN 隐状态 → 记忆系统的接口

LNN 的隐状态本身不可读——必须通过**三个 probing decoder** 把它翻译成系统其他部分能消费的信号。

### 4.1 槽位置信度 probe（L2 hidden → 20 slots）

```python
# Pseudocode
class SlotConfidenceProbe(nn.Module):
    def __init__(self, hidden_dim=32, num_slots=20, value_dim=64):
        self.confidence_head = nn.Linear(hidden_dim, num_slots)  # → [B, T, 20]
        self.value_head = nn.Linear(hidden_dim, num_slots * value_dim)
    def forward(self, l2_hidden):
        confidence = sigmoid(self.confidence_head(l2_hidden))  # 0-1
        values = self.value_head(l2_hidden).view(B, T, 20, 64)
        return confidence, values
```

### 4.2 事件边界门控（L2 hidden seq → boundary?）

```python
# v0：规则 + LLM 心跳
def detect_boundary_v0(l2_hidden_seq, llm_summary, last_boundary_t):
    # 隐状态突变 + 时间间隔 + LLM 判断
    delta = cosine_distance(l2_hidden_seq[-1], l2_hidden_seq[-30])  # 30s 前对比
    time_gap = current_t - last_boundary_t
    if delta > 0.3 or time_gap > 600 or llm_says_new_scene(llm_summary):
        return True, confidence=delta
    return False, confidence=0

# v1：On-GEBD-on-LNN（M2 末上线）
class BoundaryGateV1(nn.Module):
    def __init__(self): self.mlp = MLP(32, hidden=64, output=1)
    def forward(self, l2_hidden_seq): return sigmoid(self.mlp(l2_hidden_seq))
```

### 4.3 状态摘要 decoder（L2 hidden → 自然语言）

```python
# probing decoder：让 LNN 状态"可解释"
def state_summary_decoder(l2_hidden):
    """输出："主人最近 10 分钟在专注写代码，情绪平稳" """
    return llm.complete(prompt=f"Describe state: {l2_hidden.tolist()}")
```

**Why 三个 probe 并存**：分别服务三个下游——槽位 probe 服务 EventTensor 组装、boundary gate 服务 L2 情景写入、summary decoder 服务用户可解释面板（v0.3 §8 "可解释面板"能力）。

---

## 5. 三层记忆的工程接口

### 5.1 EventTensor 数据模型

```python
from dataclasses import dataclass
from datetime import datetime
import torch

@dataclass
class SlotValue:
    value_embedding: torch.Tensor  # [64]
    confidence: float              # 0-1
    position: int                  # slot index 0-19
    raw_text: Optional[str] = None # for debugging / user UI

@dataclass
class OverflowCapsule:
    """v0.3 §5.3：槽位填不下时的外溢通道"""
    event_id: str
    free_text_summary: str
    emotion_trajectory: List[float]
    evidence_refs: List[str]  # audio chunk paths / original text spans

@dataclass
class EventTensor:
    slots: Dict[int, SlotValue]          # 1..20
    timestamp: datetime
    valid_until: Optional[datetime]      # bi-temporal
    user_id: str
    raw_modalities: Dict[str, bytes]     # audio/video reference
    embedding: torch.Tensor              # [768] for vector retrieval
    overflow: Optional[OverflowCapsule] = None
```

### 5.2 三层记忆 API 协议

```python
class L1WorkingMemory(Protocol):
    """秒-分钟，活在 LNN 隐状态里"""
    def update(self, event: EventTensor) -> None: ...
    def get_state_summary(self) -> str: ...
    def reset_on_timeout(self, timeout_minutes: int = 30) -> None: ...

class L2EpisodicMemory(Protocol):
    """小时-天，Graphiti KG (bi-temporal)"""
    def add_episode(self, event: EventTensor) -> str:  # returns episode_id
        ...
    def search(
        self,
        query: str,
        time_filter: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 10,
    ) -> List[Tuple[EventTensor, float]]: ...
    def update(self, episode_id: str, patch: Dict) -> None: ...
    def soft_delete(self, episode_id: str, recoverable_hours: int = 24) -> None: ...
    def hard_delete(self, episode_id: str) -> AuditID: ...

class L3SemanticMemory(Protocol):
    """周-年，LoRA on LLM + 槽位偏好向量"""
    def get_user_adapter(self, user_id: str) -> LoRAAdapter: ...
    def get_slot_preference_vector(self, user_id: str) -> torch.Tensor:  # [20]
        ...
    def update_from_consolidation(self, user_id: str, consolidated_facts: List[ConsolidatedFact]) -> None: ...
```

### 5.3 写入门控的优先级语义（v0.3 §4.3）

| 槽位档 | 触发行为 |
|--------|---------|
| **P0**（句子类型/主语/谓语） | 未填 → 触发主动澄清；写入 L2 的硬约束 |
| **P1**（时间/地点/直宾/补语） | L2 检索的主排序键 |
| **P2**（时态） | bi-temporal `valid_until` 推理 |
| **P3**（修饰/连接词） | 仅向量化嵌入，不进 graph 路径 |

**写入门控伪代码**：

```python
def gate_l2_write(event: EventTensor) -> bool:
    p0_filled = all(event.slots[i].confidence > 0.7 for i in P0_INDICES)
    p1_threshold = sum(event.slots[i].confidence for i in P1_INDICES) >= 2.5
    boundary, _ = detect_boundary(...)
    user_explicit = event.metadata.get("user_says_remember", False)

    if user_explicit: return True
    if not p0_filled: return False  # P0 未填硬阻断
    if boundary and p1_threshold: return True
    return False
```

---

## 6. CLS Consolidation 算法

夜间巩固是"L2 → L3 的离线学习"。**不是数据库批处理，是产品仪式感**（v0.3 §5.4）。

### 6.1 流程伪代码

```python
def nightly_consolidation(user_id: str, day: datetime):
    # 1. 取出当天高强度事件（情绪强 / 重复出现 / 用户标 important）
    candidates = l2_memory.query_by_intensity(
        user_id=user_id,
        time_range=(day_start, day_end),
        min_intensity=0.6,
    )

    # 2. 聚类去重（避免相同事件多次写入 L3）
    clusters = cluster_episodes(candidates, similarity_threshold=0.8)

    # 3. 为每个簇生成"今日精华"候选（用户可见 = 「今天」）
    today_summary = []
    for cluster in clusters[:3]:  # 最多 3 条（v0.3 §5.4.1）
        summary = llm.summarize(
            cluster, schema="event-emotion-unfinished"
        )
        today_summary.append(summary)

    # 4. 提取偏好信号（用户可见 = 「我对你的新理解」）
    new_preferences = extract_preferences(clusters, existing_l3=l3.get(user_id))

    # 5. 等用户审核（不是自动应用！）
    user_decisions = await user_review_ui.show(today_summary, new_preferences)

    # 6. 应用确认的部分
    for fact in user_decisions.approved:
        l3_memory.update_from_consolidation(user_id, [fact])

    # 7. 训练 LoRA（per-user adapter，rank=8）
    if len(user_decisions.approved) >= 10:
        lora_trainer.train_increment(
            user_id=user_id,
            new_facts=user_decisions.approved,
            rank=8,
        )

    # 8. 记忆账本（用户可见 = 「记忆账本」）
    audit_log.write(user_id, day, user_decisions)
```

### 6.2 写入 L3 的硬约束

- **不能直接写**：必须经过用户审核（v0.3 §5.4.2）
- **置信度衰减**：与近期行为冲突时进入"待确认"状态
- **占比限制**：类型 3 数据训练贡献占比 ≤ 50%（v0.3 §13.3 老板偏差防护）

---

## 7. 个性化栈接入点

```python
class PersonalizationStack:
    """LoRA on LLM + Slot Preference Vector + BitFit on LTC"""

    def apply_to_inference(self, user_id: str, llm_input):
        # Layer 1: per-user 槽位偏好向量注入 LNN 输入侧 embedding
        pref_vec = self.l3.get_slot_preference_vector(user_id)  # [20]
        # 调制 EventTensor 的槽位权重
        modulated_input = llm_input * (1 + 0.1 * pref_vec.softmax())

        # Layer 2: per-user LoRA adapter 加载到 LLM
        adapter = self.l3.get_user_adapter(user_id)
        llm.load_adapter(adapter)

        # Layer 3: BitFit on LTC（仅 cell bias 可学，per-user 独立 bias）
        ltc_bias = self.l3.get_ltc_bias(user_id)
        l2_lnn.cell.bias = ltc_bias

        return llm.generate(modulated_input)
```

**Why 三件套并行而非选一**：
- 偏好向量低维（20）—— 快速启动、易冷启
- LoRA on LLM（rank=8）—— 中等容量、最显著效果
- BitFit on LTC —— 极轻量、研究探索（不在关键路径）

详细论证见研究文档 §3.3——为什么 State-offset Tuning 已确认不可行。

---

## 8. 工程实施图（M0-M12，按 v0.3 §7 时间线）

### 8.1 模块构建顺序

| 月 | 新增模块 | 关键文件 |
|----|---------|---------|
| M0 | `synthetic_data_generator/`、`universal_grammar_parser/`、`graphiti_zh_adapter/` | parser.py, schema_v1.py |
| M1 | `lnn_core/`（L1+L2 ncps 封装）、`slot_probe/` | lnn_core.py, probe.py |
| M2 | `boundary_gate/`、`event_tensor/`、`state_summary/` | gate_v0.py, tensor.py |
| M3 | `personalization/v0/`（偏好向量）、`memory_ui/`（读/删 v0） | pref_vec.py, ui.py |
| M4 | **MVP α 集成**：所有模块通过 `agent_core/` 编排 | agent.py |
| M5 | `cls_consolidation/`、`memory_ui/` 编辑功能、KG 分层启动 | consolidator.py |
| M6 | `automotive_adapter/`、`lfm2_offline/` | car.py |
| M7-M8 | `personalization/v1/`（LoRA + BitFit）、`active_inference/v0/` | lora_trainer.py |
| M9 | `proactive_advisor/`、记忆驱动主动建议 | advisor.py |
| M10 | `multi_device_sync/` | sync.py |
| M11 | `kg_governance/`（热/温/冷分层全量） | governance.py |
| M12 | `meta_cognition/`（L6）、`symbiosis_protocol/`（L7）启动 | l6.py, l7.py |

### 8.2 三色判定（v0.3 §7.3）应用到 LNN 训练

| 评估时点 | LNN 关键指标 | 红区 | 黄区 | 绿区 |
|---------|------------|------|------|------|
| M1 末 | L2 联合训练 loss / 推理延迟 | 训练发散 / >100ms | loss 不收敛但有趋势 / 60-100ms | 收敛 / <50ms |
| M2 末 | 状态摘要可读性（人评） / boundary F1 | <2/5 / <0.3 | 2-3/5 / 0.3-0.5 | ≥3/5 / ≥0.5 |
| M3 末 | 个性化偏好向量学习速度 | 不学到任何用户差异 | 弱差异 | 明显差异 |

---

## 9. 测试矩阵（验收标准）

| 层 | 单元测试 | 集成测试 | 用户感知 |
|----|---------|---------|---------|
| L1 LNN | hidden state 不发散 / 推理 < 50ms | 多模态输入跨模态对齐 IoU ≥ 0.6 | 不直接可见 |
| L2 LNN | 工作记忆 30 分钟连续不重置 | 与 LLM-only baseline 槽位 IoU ≥ 0.6 | "追问能跟住" |
| 槽位 probe | P0 召回 ≥ 90% / P1 ≥ 70% | 跨模态一致 | 复述准确 |
| boundary gate | F1 ≥ 0.5（v0）/ 0.7（v1） | 写入 KG 无重复噪声 | "回溯成片段故事" |
| L2 KG | 中文 entity F1 ≥ 0.6 / 检索 P95 < 300ms | LongMemEval-zh ≥ 65% | "上周三和谁吃饭"答得出 |
| CLS | 巩固后 L3 一致性 ≥ 0.8 | 第二天对话质量 A/B 提升 ≥ 1 分 | "今天总结更懂自己" |
| L3 个性化 | LoRA 收敛 / 偏好向量低维稳定 | per-user 风格差异显著 | "回复语气更一致" |

---

## 10. 给 AI 同事的工作启动指南

### 10.1 cc（Claude Code）任务模式
- 架构设计 / 接口定义 / 文档 / 测试 / 跨模块整合
- 主线 PR：在本仓库 `src/` 下按 §8.1 顺序建立模块骨架，每个模块先写 Protocol + 测试，再写实现
- 调用 codex 时机：单模块代码量 > 500 行 / 跨模块对抗审查 / 性能瓶颈深挖

### 10.2 codex 任务模式
- 工程规范深挖 / 边界条件 / 性能优化 / 对抗审查
- 输入要求：完整 Protocol + 上下游接口 + 测试样例
- 输出落盘：`/tmp/codex-<task>-output.md`，末尾必须打 `## Summary`

### 10.3 任何模块开始前的三道门
1. **接口先行**：先写 Protocol + 输入输出 schema，再写实现
2. **测试先行**：单元测试定义验收标准，TDD 节奏
3. **codex 对抗审查**：cc 写完立即让 codex 挑刺，再 commit

### 10.4 何时应该停下来问老板
- 任何对 §10.1（不可妥协 6 项）的妥协建议
- 三色判定连续两次进入红区
- 类型 3 数据使用边界的灰色地带
- 用户访谈反馈与 v0.3 §13.1 双门验证标准冲突

---

## 11. 链接到产品定义与研究底稿

| 文档 | 位置 | 用途 |
|------|------|------|
| 产品定义 v0.3（封版） | [`product-definition.md`](product-definition.md) | "为什么这样做" |
| 研究底稿 | [`source-research.md`](source-research.md) | 技术依据与决策链 |
| 术语字典 | [`glossary.md`](glossary.md) | 名词速查 |
| M0-M12 路线图 | [`../roadmap/milestones.md`](../roadmap/milestones.md) | 时间线 + 三色判定 |

---

**署名：cc + 老板 + claude code（vibecoding 协作） | 2026-05-03 | v0.1**
