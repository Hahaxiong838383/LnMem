"""Phase 2 在线 LoRA τ-bias 漂移训练循环（cc 主线 Day 12 P0-2）。

设计要点
--------
1. **本文件不实现 events → tensor 转换**——signal 维度 4-key vs LNN feature_dim=18 的
   join 需要 signals jsonl 配合（feedback_dataset.py 之外的工作），由调用方组装好
   `(x, persona_idx, labels, rewards)` 喂入。本文件聚焦 reward-weighted 微调内核。
2. **LNN-FIRST P-1 合规**：fit_step 通过 `state_model(x, persona_idx)` 走完整 LTC ODE
   forward，loss.backward 仅触达 `online_adapter.tau_drift`（base 在 OnlineLoRAAdapter
   构造时已 freeze）。
3. **REINFORCE-style reward weighting**：
   - match (reward=+1)  → 标准 CE，pull p(label) up
   - mismatch (reward=-1) → 取 negation，push p(label) down
   - skip / input (reward=0) → 该样本对 loss 不贡献

下游消费
--------
- 实测 sanity check：mismatch 反传后同帧 logits[label] 应严格下降
- 反馈累积达到 trigger 时调用 fit_step（由 daemon 或 cron 编排）
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from poc.src.lnn.feedback_dataset import (
    count_by_label,
    count_mismatch,
    load_events,
)
from poc.src.lnn.online_adapter import OnlineLoRAAdapter
from poc.src.lnn.online_contracts import (
    FeedbackDataError,
    OnlineAdapterConfig,
    OnlineAdapterError,
)


# ---------- Trigger 决策 ----------


def trigger_if_ready(
    feedback_dir: Path,
    config: OnlineAdapterConfig,
) -> tuple[bool, dict[str, int]]:
    """检查反馈累积是否达到首次 / 增量微调触发条件。

    触发条件（AND）：
    1. 总反馈条数 ≥ `config.min_total_feedback`（默认 10，PoC 不可妥协最低门槛）
    2. mismatch 条数 ≥ `config.trigger_mismatch_threshold`（默认 10，理想触发量）

    Fail-soft（codex F3-03 修复）：
    - feedback_dir 不存在 → 返回 (False, zero counts)，不抛错
      （首次部署 / 目录尚未生成时，调度方应"暂无数据"而非"硬错"）

    Returns
    -------
    (should_trigger, counts) where counts contains keys
    {match, mismatch, skip, input, total}（任一类型 0 时也存在该 key 值为 0）
    """
    counts = {"match": 0, "mismatch": 0, "skip": 0, "input": 0, "total": 0}
    try:
        events = load_events(feedback_dir)
    except FeedbackDataError:
        # F3-03: feedback_dir missing → fail-soft 返零计数（不抛）
        return False, counts

    counts.update(count_by_label(events))
    counts["total"] = len(events)

    if counts["total"] < config.min_total_feedback:
        return False, counts
    return (
        count_mismatch(events) >= config.trigger_mismatch_threshold,
        counts,
    )


# ---------- Loss 内核 ----------


def compute_reward_loss(
    logits: Tensor,
    labels: Tensor,
    rewards: Tensor,
) -> Tensor:
    """Reward-weighted negative log-likelihood.

    Formula
    -------
    ``loss = -mean(reward * log_softmax(logits)[label])``

    - reward=+1 (match)    → ``-log p(label)``，标准 CE，最小化即 pull p(label) up
    - reward=-1 (mismatch) → ``+log p(label)``，最小化即 push p(label) down
    - reward= 0 (skip/input) → 该样本本身贡献 0，**但仍占 mean 分母**

    设计决策注解（codex F3-02）
    --------------------------
    skip/input 样本 reward=0 时其 grad 贡献为 0，但因 ``-weighted.mean()`` 仍
    包含在分母 B 中，等价于把 match/mismatch 样本的有效梯度按 ``B_eff/B`` 缩放。
    这是"按 batch 平均"的 REINFORCE 标准实现。如果未来需要"非零均值"语义
    （仅对 ``reward!=0`` 子集求平均），调用方应负责在 batching 阶段就过滤掉
    skip/input，让本函数收到的 batch 不含 0 reward 样本。

    Parameters
    ----------
    logits : Tensor  (B, n_classes)
        模型最后一帧的 raw logits（softmax 前）
    labels : Tensor  (B,) long
        每个样本的目标 class index
    rewards : Tensor  (B,) float
        +1.0 / -1.0 / 0.0

    Returns
    -------
    loss : Tensor  scalar
    """
    if logits.ndim != 2:
        raise OnlineAdapterError(
            f"expected logits (B, n_classes), got shape {tuple(logits.shape)}"
        )
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise OnlineAdapterError(
            f"expected labels (B,) matching logits batch, got {tuple(labels.shape)}"
        )
    if rewards.shape != labels.shape:
        raise OnlineAdapterError(
            f"expected rewards (B,) matching labels, got {tuple(rewards.shape)}"
        )

    log_probs = F.log_softmax(logits, dim=-1)
    # log p(label) per sample: (B,)
    log_p_label = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    weighted = rewards * log_p_label  # (B,)
    return -weighted.mean()


# ---------- 训练 step ----------


def fit_step(
    state_model: nn.Module,
    online_adapter: OnlineLoRAAdapter,
    optim: torch.optim.Optimizer,
    *,
    x: Tensor,
    persona_idx: Tensor,
    labels: Tensor,
    rewards: Tensor,
    config: OnlineAdapterConfig,
) -> dict[str, float]:
    """Single SGD step on `tau_drift` driven by reward-weighted loss.

    Parameters
    ----------
    state_model : nn.Module
        StateRecognitionLNN with `personalization` already replaced by `online_adapter`
    online_adapter : OnlineLoRAAdapter
    optim : torch.optim.Optimizer
        Should only manage `online_adapter.tau_drift`
    x : Tensor  (B, T, feature_dim)
    persona_idx : Tensor  (B,) long
    labels : Tensor  (B,) long
    rewards : Tensor  (B,) float
    config : OnlineAdapterConfig

    Returns
    -------
    metrics : dict
        {loss, drift_max_norm, batch_size}
    """
    if x.ndim != 3:
        raise OnlineAdapterError(f"expected x (B, T, F), got {tuple(x.shape)}")

    # F2-01 (codex): 校验 model-adapter 绑定，防 silent no-op
    # state_model._PersonaConditionedCfC.personalization 必须 IS online_adapter，
    # 否则 loss.backward() 不会把梯度落到 online_adapter.tau_drift 上。
    bound_personalization = getattr(state_model, "personalization", None)
    if bound_personalization is not online_adapter:
        raise OnlineAdapterError(
            "fit_step: state_model.personalization is not the passed online_adapter; "
            "this would silently no-op tau_drift updates. Replace state_model."
            "personalization = online_adapter before calling fit_step."
        )

    state_model.train()
    optim.zero_grad()

    # Forward — 完整 LTC ODE 路径（半隐式欧拉 + softplus 时间常数）
    logits_seq = state_model(x, persona_idx)  # (B, T, n_classes)
    # 取序列最后一帧的 logits 做 reward 加权（弹窗触发时 daemon 看的是当前帧）
    logits = logits_seq[:, -1, :]

    loss = compute_reward_loss(logits, labels, rewards)
    loss.backward()

    # 梯度裁剪（仅 tau_drift 有 grad，base frozen）
    torch.nn.utils.clip_grad_norm_(
        [online_adapter.tau_drift], max_norm=10.0
    )
    optim.step()

    # 后置 per-persona L2 范数裁剪（防 drift 漂太远）
    drift_max_norm = online_adapter.clip_drift_norm()

    return {
        "loss": float(loss.item()),
        "drift_max_norm": float(drift_max_norm),
        "batch_size": int(x.shape[0]),
    }


__all__ = [
    "trigger_if_ready",
    "compute_reward_loss",
    "fit_step",
]
