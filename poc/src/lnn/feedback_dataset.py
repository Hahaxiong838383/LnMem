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
from datetime import datetime, timezone
from pathlib import Path

from poc.src.lnn.online_contracts import (
    FeedbackDataError,
    FeedbackEvent,
)


# F-06 fix (codex 三审 2026-05-16): ts 数值化比较，防仓内多格式（'Z' / '+00:00' / 本地偏移 / 小数秒）
# 字典序错位导致水位线误过滤/漏过滤。
def _parse_iso_ts_utc(ts: str | None) -> datetime | None:
    """Parse ISO8601 ts → tz-aware datetime in UTC. Returns None on parse failure.

    支持格式：
    - 标准带 Z: '2026-05-16T08:14:09Z'
    - +00:00 后缀: '2026-05-16T08:14:09+00:00'
    - 含微秒: '2026-05-16T07:25:08.124540Z'
    - 本地时区偏移: '2026-05-16T16:14:09+08:00' → 自动归一化到 UTC
    """
    if not ts or not isinstance(ts, str):
        return None
    try:
        # Python 3.9 fromisoformat 不支持 'Z' 后缀，先转 '+00:00'
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    # naive datetime（无 tz）视为 UTC（PoC 内部约定）
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # 归一化到 UTC
    return dt.astimezone(timezone.utc)


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
        # V0.3 字段（缺失时 None，向后兼容 V0.2）
        alignment_raw = raw.get("alignment_latency_s")
        try:
            alignment_v: float | None = (
                float(alignment_raw) if alignment_raw is not None else None
            )
        except (TypeError, ValueError):
            alignment_v = None
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
            alignment_latency_s=alignment_v,
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
    min_event_ts: str | None = None,
) -> tuple[FeedbackEvent, ...]:
    """Load all feedback events from `feedback_dir/queries_*.jsonl`.

    Parameters
    ----------
    feedback_dir : Path
        反馈目录（e.g. `~/.lnmem/feedback/`）
    date_prefix : str | None
        过滤前缀（e.g. `2026-05-10`）。None = 全部
    min_event_ts : str | None
        ⚠️ F-03 fix (codex 二审，2026-05-16) — 增量消费水位线
        如不 None：仅返回 `ev.ts > min_event_ts` 的 events（防 cron 重复训练同批数据）。
        值通常为上次 fit 成功后写入 ckpt phase2_metadata['fired_at_utc']。
        ISO8601 字符串字典序 ≡ 时间序，故直接 str 比较即可。

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

    # F-06 fix (codex 三审): 预解析 min_event_ts 到 UTC datetime，做数值比较
    min_dt: datetime | None = None
    if min_event_ts is not None:
        min_dt = _parse_iso_ts_utc(min_event_ts)
        if min_dt is None:
            import warnings
            warnings.warn(
                f"FeedbackDataset: min_event_ts={min_event_ts!r} unparseable, ignoring watermark",
                RuntimeWarning,
            )

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
                        # F-03 + F-06 fix: 增量水位线 — 用 UTC datetime 数值比较（防字符串字典序错位）
                        if min_dt is not None:
                            ev_dt = _parse_iso_ts_utc(ev.ts)
                            if ev_dt is not None and ev_dt <= min_dt:
                                continue
                            # ev.ts 不可解析时不过滤（fail-soft，让后续阶段决定是否跳过）
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
