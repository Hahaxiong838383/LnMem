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
    build_session_frame_index,
    load_signals,
)

_log = logging.getLogger(__name__)

# state name → idx 反查表
STATE_NAME_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(STATE_LABELS)}

# F4-07 (codex): true_state 别名/变体 → 标准 STATE_LABELS 名
# 用户填 true_state 时常见非标准变体（连字符 / 下划线 / 缩写）应正确映射
TRUE_STATE_ALIASES: dict[str, str] = {
    # 常见 hyphen / underscore 变体（normalize 前已 lower+strip）
    "distracted-focused": "distracted",
    "distracted_focused": "distracted",
    "distractedfocused": "distracted",
    "low-energy": "low_energy",
    "lowenergy": "low_energy",
    "emotional-positive": "emotional_positive",
    "emotionalpositive": "emotional_positive",
    "positive": "emotional_positive",
    "emotional-negative": "emotional_negative",
    "emotionalnegative": "emotional_negative",
    "negative": "emotional_negative",
    # 缩写 / 同义词
    "tired": "fatigued",
    "concentrating": "focused",
    "concentrated": "focused",
    "transitioning_state": "transitioning",
}

MATCH_LABEL_PLACEHOLDER = -1
"""match 反馈尚未确定 label（待调用方 model.forward 后用 argmax 填充）。"""

# F2-03 (codex): match argmax 置信门槛（softmax 最大概率）
DEFAULT_MATCH_CONFIDENCE_THRESHOLD = 0.30
"""match argmax fill 时模型 softmax 最大概率必须 ≥ 此阈值；否则该样本 reward 改 0
不被强化（防 noise label 污染 tau_drift）。"""


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


def _normalize_true_state(raw: str | None) -> str | None:
    """F4-07 (codex): true_state alias/case normalize.

    步骤：raw → strip → lower → 直接命中 STATE_LABELS 即返；否则查 TRUE_STATE_ALIASES
    映射；都失败返 None。
    """
    if not raw:
        return None
    s = str(raw).strip().lower()
    if s in STATE_NAME_TO_IDX:
        return s
    if s in TRUE_STATE_ALIASES:
        return TRUE_STATE_ALIASES[s]
    return None


def _label_for_event(
    event: FeedbackEvent,
) -> tuple[int, str] | None:
    """根据 event.label 决定 (label_idx, source_tag)；返 None 表示跳过。

    - mismatch + true_state 在 STATE_LABELS 中（含 alias normalize）
      → (idx, "mismatch_true_state")
    - match → (-1, "match_argmax_pending")，调用方填 argmax
    - skip / input → None（跳过，不入 batch）
    - mismatch 但 true_state 缺失或非法 → None（warning）
    """
    if event.label == "match":
        return MATCH_LABEL_PLACEHOLDER, "match_argmax_pending"
    if event.label == "mismatch":
        # F4-07: normalize alias
        normalized = _normalize_true_state(event.true_state)
        if normalized is None:
            warnings.warn(
                f"feedback_to_batch: mismatch event {event.event_uuid or event.frame_id} "
                f"true_state={event.true_state!r} not in STATE_LABELS (after alias normalize), skipping",
                RuntimeWarning,
            )
            return None
        return STATE_NAME_TO_IDX[normalized], "mismatch_true_state"
    # skip / input
    return None


def _slice_window(
    signals: tuple[SignalsFrame, ...],
    session_frame_index: dict[tuple[str | None, int], int],
    target_session_id: str | None,
    target_frame_id: int,
    window_size: int,
) -> Tensor | None:
    """取 [target - window+1, target] 共 window_size 帧的 (T, 18) 张量；不足返 None.

    F1-01 修复：用 (session_id, frame_id) 复合键，跨天/跨 session 不再 silent
    overwrite。

    F1-01 兼容：V0.1 老 feedback 不写 session_id（None）→ fallback 在 signals
    中找该 frame_id 的所有 session 候选，多于一个时取最近的（list pos 最大）+
    warning（提示老数据语义不严谨）；零个时返 None。

    F3-05 修复：window 切片同时校验 frame_id 连续性（同 session 内 frame_id
    应严格 +1 step），不连续时发 warning（不阻断，因 daemon 偶尔丢帧合理）。

    PoC 阶段：不足 window_size 直接跳过（不做 padding，避免污染 LTC ODE 早期帧）。
    """
    # F1-01 兼容：target_session_id is None 时 fallback 找任一 session 的同 frame_id
    if target_session_id is None:
        candidates = [
            k for k in session_frame_index if k[1] == target_frame_id
        ]
        if not candidates:
            return None
        if len(candidates) > 1:
            warnings.warn(
                f"feedback_to_batch: V0.1 feedback w/o session_id frame_id={target_frame_id} "
                f"matches {len(candidates)} signals sessions; taking last (latest by file order)",
                RuntimeWarning,
            )
        # 取 list pos 最大的（最新）
        candidates.sort(key=lambda k: session_frame_index[k])
        key = candidates[-1]
    else:
        key = (target_session_id, target_frame_id)

    if key not in session_frame_index:
        return None
    end_pos = session_frame_index[key]
    start_pos = end_pos - window_size + 1
    if start_pos < 0:
        return None

    # F3-05: 校验 window 内同 session_id + frame_id 连续性
    window_frames = signals[start_pos:end_pos + 1]
    if len(window_frames) != window_size:
        return None
    sids = {f.session_id for f in window_frames}
    if len(sids) > 1:
        warnings.warn(
            f"feedback_to_batch: window crosses session boundary at "
            f"frame_id={target_frame_id} (sids={sids}), skipping",
            RuntimeWarning,
        )
        return None
    expected_first_fid = target_frame_id - window_size + 1
    actual_first_fid = window_frames[0].frame_id
    if actual_first_fid != expected_first_fid:
        warnings.warn(
            f"feedback_to_batch: window non-contiguous at frame_id={target_frame_id} "
            f"(expected first fid={expected_first_fid}, got {actual_first_fid}), "
            f"using as-is but signals may have gaps",
            RuntimeWarning,
        )
    rows = [list(f.vector) for f in window_frames]
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

    # F1-01 (codex): use (session_id, frame_id) 复合键防跨天 silent overwrite
    session_frame_index = build_session_frame_index(signals)

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

        # F1-01: window 切片用 (event.session_id, frame_id) 复合键
        x_window = _slice_window(
            signals, session_frame_index,
            target_session_id=ev.session_id,
            target_frame_id=ev.frame_id,
            window_size=window_size,
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

    # F3-06 (codex): clone 防御 in-place 篡改（虽然新建 tensor 已经独立，但保险）
    x_tensor = torch.stack(xs, dim=0).clone()  # (B, T, 18)
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
    *,
    confidence_threshold: float = DEFAULT_MATCH_CONFIDENCE_THRESHOLD,
) -> TrainingBatch:
    """对 match 样本（label=MATCH_LABEL_PLACEHOLDER）用 model 当前 argmax 填 label.

    语义：match 反馈 reward=+1，希望强化"模型当前对该帧的预测"，所以 label
    取模型自身的 argmax（自一致 self-distill）。

    F2-03 (codex): match argmax 加 confidence 门槛 — 若该样本最大 softmax 概率
    < threshold，模型在该帧 confidence 低（uniform-ish），argmax 是 noise。
    此时降级 reward=0（不强化噪声标签），label 仍填 argmax 但不影响 loss。

    F3-06 (codex): tensor clone 防御外部 in-place 篡改。

    Parameters
    ----------
    batch : TrainingBatch
    state_model : nn.Module (StateRecognitionLNN)
    confidence_threshold : float
        match argmax 最大 softmax 概率必须 ≥ 此阈值；否则该样本 reward 改 0

    Returns
    -------
    new_batch : TrainingBatch
        新对象，含克隆的 tensor + match label 已填 argmax + 低置信样本 reward 降为 0
    """
    state_model.eval()
    with torch.no_grad():
        logits_seq = state_model(batch.x, batch.persona_idx)  # (B, T, n_classes)
        last_logits = logits_seq[:, -1, :]
        probs = last_logits.softmax(dim=-1)  # (B, n_classes)
        max_probs, argmax_last = probs.max(dim=-1)  # (B,) each

    new_labels = batch.labels.clone()
    new_rewards = batch.rewards.clone()

    placeholder_mask = (new_labels == MATCH_LABEL_PLACEHOLDER)
    new_labels[placeholder_mask] = argmax_last[placeholder_mask]

    # F2-03: 低置信 match 样本 reward 降为 0（不强化噪声标签）
    low_conf_match_mask = placeholder_mask & (max_probs < confidence_threshold)
    if low_conf_match_mask.any():
        warnings.warn(
            f"feedback_to_batch: {int(low_conf_match_mask.sum().item())} match samples "
            f"argmax confidence < {confidence_threshold}, reward降为 0 (F2-03 防 noise)",
            RuntimeWarning,
        )
        new_rewards[low_conf_match_mask] = 0.0

    # F3-06: clone x / persona_idx 防御 in-place
    return TrainingBatch(
        x=batch.x.clone(),
        persona_idx=batch.persona_idx.clone(),
        labels=new_labels,
        rewards=new_rewards,
        frame_ids=batch.frame_ids,
        label_sources=batch.label_sources,
        skipped_count=batch.skipped_count,
    )


__all__ = [
    "MATCH_LABEL_PLACEHOLDER",
    "DEFAULT_MATCH_CONFIDENCE_THRESHOLD",
    "STATE_NAME_TO_IDX",
    "TRUE_STATE_ALIASES",
    "TrainingBatch",
    "build_training_batch",
    "fill_match_labels_with_argmax",
]
