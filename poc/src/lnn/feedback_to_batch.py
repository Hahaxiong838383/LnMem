"""poc/src/lnn/feedback_to_batch.py — feedback × signals → 训练 batch.

把累积的 feedback events × signals jsonl × ckpt persona embedding 三方 join 成
fit_step 期望的 (x, persona_idx, labels, rewards) 四元组 tensor。

设计要点
--------
1. **frame_id 严格 join**：feedback.frame_id 必须能在 signals 中找到对应行；
   否则该 event 跳过（warning）+ 计入 skipped count。
2. **滑动窗口**：每个 event 取 [frame_id - window_size + 1, frame_id] 共
   window_size 帧的 signals，组成 (T, 18) 序列；不足 window 时前导 pad
   或丢弃（依 strategy）。
3. **label 选择策略**（PoC Day 13 - 简化版）：
   - mismatch: user_response.true_state 必须在 STATE_LABELS 中
     → idx，否则跳过
   - match: 用模型 forward 后 argmax 自一致 label（reward=+1 强化）
     当前实现：返回 -1 sentinel，调用方负责 model forward 后填充
   - skip / input: reward=0 但仍可入 batch（按 docstring F3-02 决策）
4. **persona_idx 来源**：daemon V0.2 不在 feedback jsonl 中写 persona_id，
   所以需要 caller 指定（PoC 阶段所有反馈来自 vboss_self → idx=30）

LNN-FIRST P-1 合规
-------------------
本模块仅做数据 join + tensor 组装；不做任何 reward 信号转换或 LLM 标签
推断（mismatch label 来自用户显式 true_state，match label 来自模型自身）。
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from poc.src.lnn.expression_dynamics import STATE_LABELS
from poc.src.lnn.feedback_dataset import (
    filter_for_training,
    load_events,
)
from poc.src.lnn.online_contracts import FeedbackEvent
from poc.src.lnn.signals_loader import (
    SignalsFrame,
    build_frame_id_index,
    load_signals,
)

_log = logging.getLogger(__name__)

# state name → idx 反查表
STATE_NAME_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(STATE_LABELS)}

MATCH_LABEL_PLACEHOLDER = -1
"""match 反馈尚未确定 label（待调用方 model.forward 后用 argmax 填充）。"""


@dataclass(frozen=True)
class TrainingBatch:
    """fit_step 期望的训练批 — 四元组 tensor。"""

    x: Tensor              # (B, T, 18) float32
    persona_idx: Tensor    # (B,) long
    labels: Tensor         # (B,) long, ∈ [0, n_state_classes) ∪ {-1}
    rewards: Tensor        # (B,) float32
    # metadata for debugging
    frame_ids: tuple[int, ...]
    label_sources: tuple[str, ...]  # "mismatch_true_state" / "match_argmax_pending" / etc
    skipped_count: int


def _label_for_event(
    event: FeedbackEvent,
) -> tuple[int, str] | None:
    """根据 event.label 决定 (label_idx, source_tag)；返 None 表示跳过。

    - mismatch + true_state 在 STATE_LABELS 中 → (idx, "mismatch_true_state")
    - match → (-1, "match_argmax_pending")，调用方填 argmax
    - skip / input → None（跳过，不入 batch）
    - mismatch 但 true_state 缺失或非法 → None（warning）
    """
    if event.label == "match":
        return MATCH_LABEL_PLACEHOLDER, "match_argmax_pending"
    if event.label == "mismatch":
        ts = event.true_state
        if not ts or ts not in STATE_NAME_TO_IDX:
            warnings.warn(
                f"feedback_to_batch: mismatch event {event.event_uuid or event.frame_id} "
                f"true_state={ts!r} not in STATE_LABELS, skipping",
                RuntimeWarning,
            )
            return None
        return STATE_NAME_TO_IDX[ts], "mismatch_true_state"
    # skip / input
    return None


def _slice_window(
    signals: tuple[SignalsFrame, ...],
    frame_id_index: dict[int, int],
    target_frame_id: int,
    window_size: int,
) -> Tensor | None:
    """取 [target - window+1, target] 共 window_size 帧的 (T, 18) 张量；不足返 None.

    PoC 阶段：不足 window_size 直接跳过（不做 padding，避免污染 LTC ODE 早期帧）。
    """
    if target_frame_id not in frame_id_index:
        return None
    end_pos = frame_id_index[target_frame_id]
    start_pos = end_pos - window_size + 1
    if start_pos < 0:
        return None
    rows = [list(signals[i].vector) for i in range(start_pos, end_pos + 1)]
    if len(rows) != window_size:
        return None
    return torch.tensor(rows, dtype=torch.float32)


def build_training_batch(
    feedback_dir: Path,
    signals_dir: Path,
    *,
    persona_idx: int,
    window_size: int = 60,
    date_prefix: str | None = None,
    include_skip_input: bool = False,
) -> TrainingBatch:
    """从磁盘加载 feedback + signals → 组装训练 batch。

    Parameters
    ----------
    feedback_dir : Path
        ``~/.lnmem/feedback/``
    signals_dir : Path
        ``~/.lnmem/data/``
    persona_idx : int
        所有 feedback 视为来自此 persona（PoC 阶段：vboss_self=30）
    window_size : int
        LNN 输入序列长度（默认 60，对应 LNNTrainConfig.sequence_length）
    date_prefix : str | None
        过滤前缀，None=全部
    include_skip_input : bool
        是否把 skip/input 也纳入 batch（reward=0，按 F3-02 决策默认 False）

    Returns
    -------
    batch : TrainingBatch

    Raises
    ------
    FeedbackDataError
        feedback / signals dir 不存在 或 无法组装任何样本
    """
    events = load_events(feedback_dir, date_prefix=date_prefix)
    if include_skip_input:
        usable_events = events  # 所有 4 类都纳入
    else:
        usable_events = filter_for_training(events)  # 仅 match + mismatch

    signals = load_signals(signals_dir, date_prefix=date_prefix)
    if not signals:
        from poc.src.lnn.online_contracts import FeedbackDataError as _FDE
        raise _FDE("no signals frames loaded")

    frame_index = build_frame_id_index(signals)

    xs: list[Tensor] = []
    labels: list[int] = []
    rewards_list: list[float] = []
    frame_ids: list[int] = []
    sources: list[str] = []
    skipped = 0

    for ev in usable_events:
        # label 选择
        label_pair = _label_for_event(ev)
        if label_pair is None:
            skipped += 1
            continue
        label_idx, source = label_pair

        # window 切片
        x_window = _slice_window(
            signals, frame_index, ev.frame_id, window_size
        )
        if x_window is None:
            warnings.warn(
                f"feedback_to_batch: frame_id={ev.frame_id} not enough signals "
                f"window (need {window_size}), skipping",
                RuntimeWarning,
            )
            skipped += 1
            continue

        xs.append(x_window)
        labels.append(label_idx)
        rewards_list.append(ev.reward())
        frame_ids.append(ev.frame_id)
        sources.append(source)

    if not xs:
        from poc.src.lnn.online_contracts import FeedbackDataError as _FDE
        raise _FDE(
            f"no usable training samples assembled "
            f"(events={len(usable_events)}, skipped={skipped})"
        )

    x_tensor = torch.stack(xs, dim=0)  # (B, T, 18)
    persona_t = torch.full(
        (len(xs),), fill_value=int(persona_idx), dtype=torch.long
    )
    label_t = torch.tensor(labels, dtype=torch.long)
    reward_t = torch.tensor(rewards_list, dtype=torch.float32)

    return TrainingBatch(
        x=x_tensor,
        persona_idx=persona_t,
        labels=label_t,
        rewards=reward_t,
        frame_ids=tuple(frame_ids),
        label_sources=tuple(sources),
        skipped_count=skipped,
    )


def fill_match_labels_with_argmax(
    batch: TrainingBatch,
    state_model: torch.nn.Module,
) -> TrainingBatch:
    """对 match 样本（label=MATCH_LABEL_PLACEHOLDER）用 model 当前 argmax 填 label.

    语义：match 反馈 reward=+1，希望强化"模型当前对该帧的预测"，所以 label
    取模型自身的 argmax（自一致 self-distill）。

    Returns
    -------
    new_batch : TrainingBatch
        新对象（immutable），labels 中 -1 已替换为 argmax
    """
    state_model.eval()
    with torch.no_grad():
        logits_seq = state_model(batch.x, batch.persona_idx)  # (B, T, n_classes)
        argmax_last = logits_seq[:, -1, :].argmax(dim=-1)  # (B,)

    new_labels = batch.labels.clone()
    mask = (new_labels == MATCH_LABEL_PLACEHOLDER)
    new_labels[mask] = argmax_last[mask]

    return TrainingBatch(
        x=batch.x,
        persona_idx=batch.persona_idx,
        labels=new_labels,
        rewards=batch.rewards,
        frame_ids=batch.frame_ids,
        label_sources=batch.label_sources,
        skipped_count=batch.skipped_count,
    )


__all__ = [
    "MATCH_LABEL_PLACEHOLDER",
    "STATE_NAME_TO_IDX",
    "TrainingBatch",
    "build_training_batch",
    "fill_match_labels_with_argmax",
]
