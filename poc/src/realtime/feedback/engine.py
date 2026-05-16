"""ProactiveQueryEngine — 编排 trigger/rate_limiter/dnd/prompt/persistence。

调用入口：daemon_main._signal_router_loop 每帧调 engine.observe(item)。
内部流：
  observe(item)
    → trigger.observe → 没事件 ⇒ return
    → 有事件 ⇒ rate_limiter.can_query? → DND.should_suppress?
    → 都 OK ⇒ 弹窗收反馈 → record_query + append_query 落盘
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poc.src.realtime.feedback.dnd import DNDConfig, get_frontmost_app, should_suppress
from poc.src.realtime.feedback.persistence import append_query
from poc.src.realtime.feedback.prompt import UserResponse, run_three_layer_query
from poc.src.realtime.feedback.rate_limiter import (
    RateLimiterConfig,
    TokenBucketRateLimiter,
)
from poc.src.realtime.feedback.trigger import ProactiveQueryTrigger, TriggerEvent


@dataclass(frozen=True)
class EngineConfig:
    rate: RateLimiterConfig
    dnd: DNDConfig
    enable_prompt: bool = True  # False = 仅记录 trigger，不真弹窗（测试/dry-run）


_BROW_PATH_WHITELIST = frozenset({"raw", "calibrated"})
_DIALOG_FIELD_MAX_LEN = 60
# 控制字符（含换行/制表/回车/垂直/换页/NUL 等）—— 弹窗 UI 单行展示，必须剥
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]+")


def _safe_finite_float(x: Any, default: float = 0.0) -> float:
    """F1/F2 fix（codex V0.3.3 一审 P0）：脏数据防御 — 不可解析/非有限值回落 default。

    覆盖：None / NaN / Inf / bool（int 子类需显式拒）/ "abc" / dict / list / "nan" 字面量。
    与 trigger._safe_float 风格一致；本函数永远不抛，承担 UI 格式化前的最后一道防线。
    """
    if isinstance(x, bool):
        return default
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else default
    if isinstance(x, str):
        try:
            v = float(x)
        except (ValueError, TypeError):
            return default
        return v if math.isfinite(v) else default
    return default


def _normalize_brow_path(p: Any) -> str:
    """F3 fix（codex V0.3.3 一审 P1）：brow_used_path 白名单收敛。

    - 接受 {'raw', 'calibrated'}（大小写/前后空白容错：" RAW " → "raw"）
    - 其他（含 None / 数字 / 注入字符串 / 多行污染）→ "calibrated"（最保守，向后兼容默认）
    """
    if not isinstance(p, str):
        return "calibrated"
    canon = p.strip().lower()
    return canon if canon in _BROW_PATH_WHITELIST else "calibrated"


def _sanitize_dialog_text(s: Any, max_len: int = _DIALOG_FIELD_MAX_LEN) -> str:
    """F4 fix（codex V0.3.3 一审 P1）：UI 文本最小清洗 — 去控制字符 + 限长 + 防 None。

    用于 VLM au_description 等"自然语言来源"字段，防换行/制表破坏弹窗布局。
    """
    if s is None:
        return ""
    text = str(s)
    text = _CONTROL_CHAR_RE.sub(" ", text)
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def _format_question(event: TriggerEvent) -> tuple[str, str]:
    """生成通知短消息 + 弹窗详细消息。

    ADR-005 V0.3.3（2026-05-16）：弹窗皱眉显示用 trigger 决策实际依据的值
    （signal_snapshot.brow_used_value/_path），而不是 calibrated polarized 的 brow_furrow。
    raw 路径下附带 [cal=X.XX] 让用户直观看到 calibration polarization 现象。

    V0.3.4（2026-05-16 codex 对抗审查后）：所有 sig 字段经 _safe_finite_float 归一化，
    brow_path 经 _normalize_brow_path 白名单收敛，VLM 文本经 _sanitize_dialog_text 清洗。
    "向后兼容" 升级语义：字段缺失 OR 类型错 OR 非有限值 → 统一回落到 brow_furrow + calibrated。
    """
    sig = event.signal_snapshot if isinstance(event.signal_snapshot, dict) else {}
    short = f"看到你 {_short_reason(event.reason)}，能确认下吗？"

    brow_cal = _safe_finite_float(sig.get("brow_furrow"), default=0.0)
    # F2 fix：用"可解析且有限"判定回落，不再只看 is None
    brow_used_raw = sig.get("brow_used_value")
    brow_used_safe = _safe_finite_float(brow_used_raw, default=math.nan)
    has_brow_used = math.isfinite(brow_used_safe)
    # F3 fix：路径白名单
    brow_path = _normalize_brow_path(sig.get("brow_used_path"))
    brow_display = brow_used_safe if has_brow_used else brow_cal

    brow_line = f"信号: 皱眉={brow_display:.2f}({brow_path})"
    # raw 路径且 brow_used_value 真实可用 → 附带 calibrated 对照
    if brow_path == "raw" and has_brow_used:
        brow_line += f" [cal={brow_cal:.2f}]"
    brow_line += (
        f"  微笑={_safe_finite_float(sig.get('smile_intensity')):.2f}"
        f"  闭眼={_safe_finite_float(sig.get('eye_closure')):.2f}"
    )

    # F4 fix：reason 字段经白名单 + 兜底（防上游污染 event.reason）
    reason_str = _sanitize_dialog_text(event.reason, max_len=20) or "unknown"
    detail_lines = [
        f"# {int(event.frame_id) if isinstance(event.frame_id, (int, float)) and math.isfinite(float(event.frame_id)) else 0}  原因: {reason_str}",
        brow_line,
    ]
    vlm = event.vlm_label if isinstance(event.vlm_label, dict) else None
    if vlm:
        au_desc = _sanitize_dialog_text(vlm.get("au_description"))
        if au_desc:
            detail_lines.append(f"VLM: {au_desc}")
        valence = vlm.get("valence")
        if valence is not None:
            v_safe = _safe_finite_float(valence, default=math.nan)
            a_safe = _safe_finite_float(vlm.get("arousal"), default=math.nan)
            v_str = f"{v_safe:.2f}" if math.isfinite(v_safe) else "?"
            a_str = f"{a_safe:.2f}" if math.isfinite(a_safe) else "?"
            detail_lines.append(f"VLM valence={v_str}  arousal={a_str}")
    detail_lines.append("")
    detail_lines.append("这个描述准确吗？")
    return short, "\n".join(detail_lines)


def _short_reason(reason: Any) -> str:
    # F4 fix：reason 上游可能脏数据；只接已知字符串
    if not isinstance(reason, str):
        return "状态变化"
    return {
        "brow_burst": "皱眉了一会儿",
        "low_vlm_conf": "表情有点模糊",
        "pose_held": "盯着屏幕一会儿",  # V0.3 Phase B 新增通道，原默认显示 "状态变化"
    }.get(reason, "状态变化")


class ProactiveQueryEngine:
    def __init__(
        self,
        config: EngineConfig,
        trigger: ProactiveQueryTrigger | None = None,
        rate_limiter: TokenBucketRateLimiter | None = None,
        prompt_runner: Any | None = None,
        session_id: str = "",
    ) -> None:
        self._config = config
        self._trigger = trigger or ProactiveQueryTrigger()
        self._rate = rate_limiter or TokenBucketRateLimiter(config.rate)
        self._prompt_runner = prompt_runner
        self._session_id = session_id
        self._lock = threading.Lock()
        self._log = logging.getLogger("lnmem.feedback")
        # FB-01: 追踪后台 prompt 线程，stop() 时 join
        self._prompt_threads: list[threading.Thread] = []
        self._threads_lock = threading.Lock()
        self._stopped = False
        # Stats
        self._stats = {
            "observed": 0,
            "triggered": 0,
            "rate_blocked": 0,
            "dnd_blocked": 0,
            "queried": 0,
            "responses": 0,
        }

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def observe(self, item: dict[str, Any], home: Path | None = None) -> UserResponse | None:
        """主入口：每帧调用。返回 UserResponse 或 None。"""
        self._stats["observed"] += 1
        event = self._trigger.observe(item)
        if event is None:
            return None
        self._stats["triggered"] += 1

        with self._lock:
            now = time.time()
            if not self._rate.can_query(now):
                self._stats["rate_blocked"] += 1
                self._log.info("query suppressed by rate limit (frame %d)", event.frame_id)
                return None

            frontmost = get_frontmost_app()
            if should_suppress(self._config.dnd, frontmost):
                self._stats["dnd_blocked"] += 1
                self._log.info(
                    "query suppressed by DND (app=%s, frame %d)", frontmost, event.frame_id
                )
                return None

            self._rate.record_query(now)

        if not self._config.enable_prompt:
            self._stats["queried"] += 1
            self._record(event, response=UserResponse(label="skip", true_state=None, note="dry-run"), latency_s=0.0, home=home)
            return None

        # 同步弹窗（osascript 阻塞，daemon 主流不能等太久 → 放后台 thread）
        # FB-01: 每个 event 生成 event_uuid 作为 jsonl 关联键
        event_uuid = uuid.uuid4().hex
        thread = threading.Thread(
            target=self._prompt_and_record,
            args=(event, home, event_uuid),
            name=f"feedback-prompt-{event.frame_id}",
            daemon=False,  # FB-01: 改 non-daemon，stop() 时 join 保数据
        )
        with self._threads_lock:
            self._prompt_threads.append(thread)
        thread.start()
        self._stats["queried"] += 1
        return None  # 弹窗异步，主流不阻塞

    def _prompt_and_record(self, event: TriggerEvent, home: Path | None, event_uuid: str) -> None:
        try:
            short, detail = _format_question(event)
            t0 = time.time()
            response = run_three_layer_query(
                title="🤖 LnMem 反馈",
                short_message=short,
                detail_message=detail,
                runner=self._prompt_runner,
            )
            latency_s = time.time() - t0
            self._stats["responses"] += 1
            self._record(event, response=response, latency_s=latency_s, home=home, event_uuid=event_uuid)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("prompt thread failed: %s", exc)

    def _record(
        self,
        event: TriggerEvent,
        response: UserResponse,
        latency_s: float,
        home: Path | None,
        event_uuid: str = "",
    ) -> None:
        record = {
            "ts": event.ts or datetime.now(timezone.utc).astimezone().isoformat(),
            "frame_id": event.frame_id,
            "reason": event.reason,
            "signal": event.signal_snapshot,
            "vlm_label": event.vlm_label,
            "user_response": {
                "label": response.label,
                "true_state": response.true_state,
                "note": response.note,
            },
            "latency_s": round(latency_s, 2),
            # FB-06: 关联键，Phase 2 奖励回灌依赖
            "session_id": self._session_id,
            "event_uuid": event_uuid or uuid.uuid4().hex,
        }
        try:
            append_query(record, home=home)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("append_query failed: %s", exc)

    def stop(self, timeout_s: float = 5.0) -> None:
        """FB-01: daemon 停机时调用，限时 join 后台 prompt 线程保数据完整。"""
        if self._stopped:
            return
        self._stopped = True
        with self._threads_lock:
            pending = list(self._prompt_threads)
        deadline = time.time() + timeout_s
        for thread in pending:
            remaining = max(0.0, deadline - time.time())
            try:
                thread.join(timeout=remaining)
            except Exception as exc:  # noqa: BLE001
                self._log.warning("join prompt thread %s failed: %s", thread.name, exc)
        # 清理已完成线程
        with self._threads_lock:
            self._prompt_threads = [t for t in self._prompt_threads if t.is_alive()]
