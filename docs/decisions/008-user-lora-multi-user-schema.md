# ADR-008: User-LoRA + multi-user schema（替代原 hypernetwork 方案）

**状态**：accepted（M3 启动前必做，今日先锁路线）
**日期**：2026-05-16
**决策者**：川哥 + cc + codex（F-05 multi-user 路由证据）+ gemini（hypernetwork 否决）
**触发**：codex X 方案审查 F-05 "persona 多用户路由完全不支持" + gemini "hypernetwork 训练不稳，工业用 User-LoRA 替代"

---

## 背景

cc 原 X 方案 ADR-007 提议 persona_embedding hypernetwork（hypernetwork(persona_emb) → tau_drift）。
被 gemini 调研直接否决：
> "推荐 User-specific LoRA 替代 Hypernetwork。理由：128d 在传统用户画像足够，但预测网络权重的 Hypernetwork **训练极不稳定**，LoRA 才是当前工业界**低成本个性化的标配**。风险：动态生成权重的梯度爆炸风险高。"

加上 codex F-05 证据：当前 LnMem 多用户路由根本不支持：
- `n_personas=30` 写死在 LNNTrainConfig
- `feedback_to_batch` 假设所有 feedback 同一 persona_idx
- `fit_orchestrator` 单 persona 入参
- 没有 user_id → persona_id 路由层

## 决策

**用 User-LoRA 替代 Hypernetwork，并加 multi-user routing schema**。

每用户独立一个 LoRA adapter，共享 base ckpt v2（ADR-007 真人化后）。
LoRA rank=2-4，每用户 3-5KB 参数，符合 §10.1 第 4 项三件套约束。

## 选项与对比

| 选项 | 优点 | 缺点 | 推荐 |
|---|---|---|---|
| A. Hypernetwork(persona_emb) → tau_drift | 知识共享好 | **训练不稳，梯度爆炸**（gemini 否决） | ❌ |
| **B. User-LoRA per user** | **工业标配**，稳定，3KB/用户 | 用户多时存储 N × 3KB（仍轻量） | ✅ 选 |
| C. Continued tau_drift only | 现状，简单 | 不支持 multi-user 路由 | 现状 |
| D. Adapter-Fusion（多 LoRA 合并） | 高级 | 工程复杂 | M5+ |

## multi-user 数据 schema 改造（codex F-05 + F-08）

### 当前 schema 缺陷
- `~/.lnmem/feedback/queries_*.jsonl` 没有 user_id
- `~/.lnmem/data/signals_*.jsonl` 没有 user_id
- ckpt 假设单 persona
- fit_orchestrator 单 persona_idx 入参

### M3 启动前必做 schema 改造

**signals_*.jsonl 新增字段**：
```json
{
  "ts": "...",
  "frame_id": ...,
  "user_id": "vboss_self",     // 新增：业务 ID
  "persona_idx": 30,            // 新增：LNN 内部 slot
  "extractor_version": "py-feat-0.7.1",  // 新增：可追溯
  "signal_path": "raw",         // 已有（ADR-005）
  ...
}
```

**queries_*.jsonl 新增字段**：
```json
{
  ...
  "user_id": "vboss_self",
  "persona_idx": 30,
  "frame_hash": "sha256(...)",  // 新增：raw_au + frame_id 唯一标识
  "label_source": "user_active",  // 新增：user_active / user_passive_response / system_inferred
  "extractor_version": "py-feat-0.7.1",
  ...
}
```

**新增 user → persona_idx 路由层**：
- `~/.lnmem/users.json`：`{"vboss_self": 30, "user_alice": 31, ...}`
- daemon 启动时读 user_id env 变量 → 查 routing → 设 persona_idx
- LoRA per persona_idx 存于 ckpt：`ckpt["user_lora"]["vboss_self"]`

### fit_orchestrator 多 persona batch 改造

- 接受 `persona_id_filter` 参数（默认全部）
- 按 persona_idx 分组训练，每组独立 LoRA
- 共享 base ckpt 不动

## 实施分阶段

| 阶段 | 工作 | 工时 | 时间窗口 |
|---|---|---|---|
| ADR-008 落盘（本文档） | 锁定方案 | 30min ✅ | 今日 |
| schema 字段加 (C 阶段同步) | user_id / persona_idx / frame_hash / extractor_version / label_source | 1-1.5h | **今晚 Phase 3** |
| user → persona 路由层 | users.json + daemon 启动读取 | 2-3h | M3 启动前 |
| LoRA 工程实现 | OnlineLoRAAdapter 改造为多 user LoRA | 1 周 | M3 启动 |

## 后果

### 正面
- 符合 §10.1 第 4 项 "LNN 原生轻量个性化三件套" 3KB/用户承诺
- 工业标配方案（稳定 + 可观测）
- schema 改造现在做，避免 M3 时数据迁移痛苦

### 负面 / 风险
- 多用户共享 base ckpt，单用户的"灾难性 fine-tune"可能影响 base（应禁止，只动 LoRA）
- LoRA per user 推理时需要按 user_id 加载对应 LoRA（推理路径分支）

## 反对意见 / 不同看法

- cc 原 X 方案：hypernetwork → 被 gemini 否决（训练不稳）→ cc 改 LoRA
- codex F-05：multi-user 路由根本不支持 → cc 加 schema 改造章节

---

**末尾自检**：本 ADR 替换 X 方案 v0.1 的"hypernetwork"提议，采纳 gemini + codex 联合建议。
