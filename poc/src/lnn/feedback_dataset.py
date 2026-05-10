"""FeedbackDataset — 加载 ~/.lnmem/feedback/queries_*.jsonl，过滤 + 映射成 reward 训练样本。

输入文件格式（与 daemon V0.2 PromptChannel 写入对齐）
-----------------------------------------------------
每行一个 JSON 对象，字段对应 `online_contracts.FeedbackEvent`。

输出
----
- `load_events(...)` 返回 `tuple[FeedbackEvent, ...]`（frozen，便于复用）
- `filter_for_training(...)` 过滤 skip / input，仅保留 match + mismatch
- `count_mismatch(...)` 统计 mismatch 条数（决定是否触发微调）

零 PyTorch 依赖（pure python + json + pathlib）。
"""
from __future__ import annotations

import json
from pathlib import Path

from poc.src.lnn.online_contracts import (
    FeedbackDataError,
    FeedbackEvent,
)


def _parse_event(raw: object) -> FeedbackEvent | None:
    """Parse one jsonl row → FeedbackEvent；fail-soft，失败返 None 但记录 warning.

    F3-01 (codex): raw 可能不是 dict（可能是 list/number/string，json 行结构异常但语法合法），
    入口必须先 isinstance 检查 + 把 AttributeError 纳入 fail-soft 捕获。
    """
    if not isinstance(raw, dict):
        import warnings
        warnings.warn(
            f"FeedbackDataset: skip non-dict row (type={type(raw).__name__})",
            RuntimeWarning,
        )
        return None
    try:
        user_resp = raw.get("user_response") or {}
        # user_resp 也必须是 dict（防嵌套结构异常）
        if not isinstance(user_resp, dict):
            user_resp = {}
        label = user_resp.get("label")
        if label not in {"match", "mismatch", "skip", "input"}:
            # 老旧格式或损坏，跳过
            return None
        return FeedbackEvent(
            ts=raw["ts"],
            frame_id=int(raw.get("frame_id", -1)),
            reason=raw.get("reason", ""),
            signal=dict(raw.get("signal") or {}),
            vlm_label=raw.get("vlm_label"),  # 可能 None
            label=label,
            true_state=user_resp.get("true_state"),
            note=user_resp.get("note"),
            latency_s=float(raw.get("latency_s", 0.0)),
            session_id=raw.get("session_id"),
            event_uuid=raw.get("event_uuid"),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        # fail-soft: log via warning (避免 silent fail 隐藏 bug)
        # 按 ~/.claude/rules silent_fail_soft_hides_bugs，必须 console.warn
        import warnings
        warnings.warn(
            f"FeedbackDataset: skip malformed row ({type(e).__name__}: {e})",
            RuntimeWarning,
        )
        return None


def load_events(
    feedback_dir: Path,
    *,
    date_prefix: str | None = None,
) -> tuple[FeedbackEvent, ...]:
    """Load all feedback events from `feedback_dir/queries_*.jsonl`.

    Parameters
    ----------
    feedback_dir : Path
        反馈目录（e.g. `~/.lnmem/feedback/`）
    date_prefix : str | None
        过滤前缀（e.g. `2026-05-10`）。None = 全部

    Returns
    -------
    events : tuple[FeedbackEvent, ...]
        按文件名字典序 → 行序排列；frozen tuple 保证不可变

    Raises
    ------
    FeedbackDataError
        feedback_dir 不存在
    """
    feedback_dir = Path(feedback_dir)
    if not feedback_dir.exists():
        raise FeedbackDataError(f"feedback_dir does not exist: {feedback_dir}")

    pattern = "queries_*.jsonl"
    if date_prefix:
        pattern = f"queries_{date_prefix}*.jsonl"

    events: list[FeedbackEvent] = []
    for path in sorted(feedback_dir.glob(pattern)):
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as e:
                        import warnings
                        warnings.warn(
                            f"FeedbackDataset: bad jsonl line in {path.name}: {e}",
                            RuntimeWarning,
                        )
                        continue
                    ev = _parse_event(raw)
                    if ev is not None:
                        events.append(ev)
                        # __init__ frozen=True；append 前调 reward() 提前发现非法 label
        except OSError as e:
            raise FeedbackDataError(f"cannot read {path}: {e}") from e

    return tuple(events)


def filter_for_training(
    events: tuple[FeedbackEvent, ...],
) -> tuple[FeedbackEvent, ...]:
    """Keep only `match` + `mismatch` events（skip / input 不入训练 batch）."""
    return tuple(ev for ev in events if ev.label in {"match", "mismatch"})


def count_mismatch(events: tuple[FeedbackEvent, ...]) -> int:
    """统计 mismatch 条数（决定是否达到 OnlineAdapterConfig.trigger_mismatch_threshold）."""
    return sum(1 for ev in events if ev.label == "mismatch")


def count_by_label(events: tuple[FeedbackEvent, ...]) -> dict[str, int]:
    """按 label 分桶计数（debug / progress 用）."""
    out: dict[str, int] = {}
    for ev in events:
        out[ev.label] = out.get(ev.label, 0) + 1
    return out


__all__ = [
    "load_events",
    "filter_for_training",
    "count_mismatch",
    "count_by_label",
]
