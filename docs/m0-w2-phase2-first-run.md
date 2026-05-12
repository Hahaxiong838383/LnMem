---
name: M0 W2 P0-3 Phase 2 首次微调实战报告
description: vboss_self 在线微调首次跑通 + fit_orchestrator 工程入口 + dry-run 验证管线
type: report
phase: M0 W2
created: 2026-05-12
status: implementation 完成（force dry-run 跑通），待 mismatch ≥10 自然触发实战
related:
  - poc/src/lnn/fit_orchestrator.py
  - poc/src/lnn/online_trainer.py
  - scripts/run_fit_step.py
  - docs/m0-w2-tasks.md §P0-3
---

# Phase 2 P0-3 — vboss_self 首次微调实战

## 0. TL;DR

| 项 | 结果 |
|---|---|
| 工程链 | ✅ **打通**（CLI 一键跑通 force dry-run）|
| 首次 dry-run | ckpt fbc42ac6 → `m0w2_phase2/checkpoint_phase2_20260512T011645Z.pt` |
| batch_size | 23（27 candidate = 23 match + 4 mismatch；过滤 alignment 后 23）|
| loss | 0.2514 |
| drift_max_norm | 3.4e-6（1 步 SGD + clip=0.5）|
| 测试 | fit_orchestrator 8/8 + 全量 423/423 ✅ |
| codex 对抗审 | (gpt-5.3-codex high 后台跑中) |
| 三道门状态 | cc 实现 ✅ → codex 审中 ⏳ → 川哥签字待 |

## 1. 设计决策：不嵌 daemon，CLI + cron

### 1.1 为什么不嵌 daemon

- daemon 进程加载 torch + LNN 模型会拖慢启动 + 占内存（PoC 阶段 torch=1.x.x）
- daemon 单线程跑 fit_step 会阻塞 frame 推理（即使开后台 thread 也争 GIL）
- 解耦让 fit_step 失败不影响 daemon 持续采集

### 1.2 当前架构

```
daemon (PID 60663, Phase B)
  └─ /trigger_mismatch HTTP 接收用户主动反馈
     └─ 写 ~/.lnmem/feedback/queries_*.jsonl

scripts/run_fit_step.py (CLI + cron 调度)
  └─ poc.src.lnn.fit_orchestrator.try_fit_step()
     ├─ 阶段 1: trigger_if_ready（min_total≥10 ∧ mismatch≥10，--force 跳过）
     ├─ 阶段 2: load_ckpt (day12_with_self/checkpoint_day12.pt)
     ├─ 阶段 3: wrap OnlineLoRAAdapter（freeze base + trainable tau_drift）
     ├─ 阶段 4: build_training_batch (alignment 硬过滤 + (sid,fid) 复合键)
     ├─ 阶段 5: fill_match_labels_with_argmax (codex F1-02 hard assert 防 -1 占位)
     ├─ 阶段 6: fit_step 单步 SGD on tau_drift (reward-weighted REINFORCE)
     └─ 阶段 7: save ckpt with phase2_metadata
```

## 2. fit_orchestrator API

```python
from poc.src.lnn.fit_orchestrator import try_fit_step
result = try_fit_step(
    feedback_dir=Path.home() / ".lnmem" / "feedback",
    signals_dir=Path.home() / ".lnmem" / "data",
    ckpt_in_path=Path("poc/data/checkpoints/day12_with_self/checkpoint_day12.pt"),
    ckpt_out_dir=Path("poc/data/checkpoints/m0w2_phase2"),
    persona_idx=30,  # vboss_self
    force=False,
)
# result.triggered / counts / metrics / ckpt_out_path / error
```

**Fail-soft 全链路**：任何阶段失败返 `FitRunResult` 不抛异常，通过 `reason` 字段分支处理：

| reason | 触发条件 | 处理建议 |
|---|---|---|
| `fired` | fit 跑了 + ckpt 已存 | ✅ 成功 |
| `below_threshold` | trigger_if_ready 不满足 | cron 静默继续 |
| `no_feedback` | force=True 但无 events | 等数据 |
| `no_batch` | alignment / 滑窗过滤后 batch=0 | 数据质量问题 |
| `no_ckpt` | ckpt_in_path 不存在 | 配置错误 |
| `adapter_mismatch` | base_adapter 类型不对 | 模型架构变更 |
| `error` | 其他异常 | 日志排查 |

## 3. Dry-run 实战结果（2026-05-12 09:16 CST）

### 3.1 输入

- feedback_dir: `~/.lnmem/feedback/` (50 events: match=23, mismatch=4, skip=14, input=9)
- signals_dir: `~/.lnmem/data/` (3 天 79K 帧 ≈ 22h)
- ckpt: `day12_with_self/checkpoint_day12.pt` (sha8=fbc42ac6, 31 personas, n_state_classes=8, hidden_dim=64)
- persona_idx: 30 (vboss_self)
- force: True（mismatch=4 < 阈值 10，跳过自然触发条件）

### 3.2 输出

```json
{
  "triggered": true,
  "reason": "fired",
  "counts": {"match": 23, "mismatch": 4, "skip": 14, "input": 9, "total": 50},
  "metrics": {
    "loss": 0.2514,
    "drift_max_norm": 3.37e-06,
    "batch_size": 23
  },
  "ckpt_in_sha8": "fbc42ac6",
  "ckpt_out_path": "poc/data/checkpoints/m0w2_phase2/checkpoint_phase2_20260512T011645Z.pt"
}
```

### 3.3 解读

- **drift_max_norm = 3.4e-6**：1 步 SGD + lr=1e-3 + 0 初始 + clip=0.5 → 极小漂移，符合预期
- **batch_size=23**：27 candidate (23 match + 4 mismatch) - 4 (alignment / 滑窗过滤) = 23
- **loss=0.2514**：reward-weighted CE 量级合理（match 走 -log p(label)，mismatch 走 +log p(label)）
- **dry-run 价值**：工程链 100% 打通，未来 mismatch 到 10 时不再卡接线问题

## 4. 后续

### 4.1 自然触发等待期（预计 1-3 天）

mismatch 当前 4/10，按 Phase B 上线后预估每天新增 6-7 mismatch（弹窗多样化 + cooldown 5min/次），**今天晚上或明天 mismatch ≥10 自动触发**。

### 4.2 cron 调度建议

```bash
# 每小时检查一次，trigger 满足才跑（非 force）
0 * * * * cd ~/mycc/2-Projects/LnMem-poc && .venv-poc-fresh/bin/python3 scripts/run_fit_step.py >> ~/.lnmem/fit_step.log 2>&1
```

### 4.3 多步训练扩展

当前 fit_step 单步 SGD。`OnlineAdapterConfig.max_steps=50` 默认，但 orchestrator 只跑 1 步。
未来扩展：累积 ≥10 mismatch 时跑 max_steps 步 + loss 早停。

### 4.4 daemon 嵌入（P0-3 第二轮）

PoC 后续可考虑 daemon 内嵌后台 thread，每 N 分钟自动检查触发。**当前阶段不需要**（CLI + cron 已足）。

## 5. LNN-FIRST P-1 合规

- ✅ fit_orchestrator **只编排** 已有 `fit_step` 调用，未引入新建模逻辑
- ✅ tau_drift 仍是唯一可训练参数（OnlineLoRAAdapter freeze base）
- ✅ fit_step 内 state_model(x, persona_idx) 走完整 LTC ODE forward
- ✅ persona_idx=30 vboss_self 专属 drift，不影响 base 30 personas

## 6. 三道门状态

- ✅ **第一道：cc 实现**（初版 fit_orchestrator 260 + CLI 130 + 8 测试 380 + 本文档 220）
- ✅ **第二道一审：codex BLOCK 8 finding**（2 F1 + 3 F2 + 3 F3 + 1 F4）— cc 修 7 finding
  - F1-01 ckpt 兼容性：unwrap state_model.personalization=base + 单独存 online_tau_drift（schema=phase2_v1）
  - F1-02 state/boundary key 一致性：unwrap 副效应保持共享 personalization
  - F2-01 LNN-FIRST P-1 严格合规：freeze 整个 state_model 除 tau_drift
  - F2-02 fail-soft 一致性：fit_step try/except 改 Exception 含 RuntimeError
  - F2-03 mismatch gate：force=True + mismatch=0 → reason=no_mismatch（拒绝训练）
  - F3-01 metadata 完整：online_config 加 delta_weight_decay/max_steps/batch_size/seed/alignment_window_s
  - F3-02 sha8 audit：load 失败也带 sha8
  - F3-03 atomic write：tempfile + os.fsync + os.replace
  - F4-01 CLI cron 友好：not_fired 返 0（仅 error 返 2）
- ⏳ **第二道二审：codex 审修复**（gpt-5.3-codex medium，后台跑中）
- ⏳ **第三道：川哥签字**（待二审通过 + mismatch≥10 自然触发后正式 sign-off）

## 7. 改动文件

| 文件 | 改动 | 行数 |
|---|---|---|
| `poc/src/lnn/fit_orchestrator.py` | 新建（一站式入口） | +260 |
| `scripts/run_fit_step.py` | 新建（CLI） | +130 |
| `poc/tests/test_fit_orchestrator.py` | 新建（8 测试 case） | +220 |
| `docs/m0-w2-phase2-first-run.md` | 本文档（新建） | +220 |

**测试**：fit_orchestrator 8 (初版) + 7 (一审 finding 回归) = 15 + 全量 415→**430/430** ✅。

## 8. 反馈 → fit 的端到端流程（生产建议）

```
川哥日常工作
  ↓ daemon 弹窗（V0.3 Phase B 多通道，5.5min/次）
  ↓ 川哥回 mismatch + lnmem-tell（每天约 6-7 mismatch）
  ↓ 累积到 ~/.lnmem/feedback/queries_*.jsonl
  ↓
cron 每小时 scripts/run_fit_step.py（非 force）
  ↓
trigger_if_ready 检查 min_total≥10 ∧ mismatch≥10
  ↓ 满足
fit_orchestrator.try_fit_step (7 阶段流水线)
  ↓
新 ckpt 落盘 poc/data/checkpoints/m0w2_phase2/checkpoint_phase2_YYYYMMDDTHHMMSSZ.pt
  ↓
（未来）daemon 加载新 ckpt → 推理时使用更新后的 tau_drift（个性化）
```

预估首次自然触发：mismatch 4 → 10 约 1-3 天（Phase B 上线后弹窗多样化提升 mismatch 占比）。
