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
import threading
import time
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


def _format_question(event: TriggerEvent) -> tuple[str, str]:
    """生成通知短消息 + 弹窗详细消息。"""
    sig = event.signal_snapshot
    short = f"看到你 {_short_reason(event.reason)}，能确认下吗？"
    detail_lines = [
        f"# {event.frame_id}  原因: {event.reason}",
        f"信号: 皱眉={sig.get('brow_furrow', 0):.2f}  "
        f"微笑={sig.get('smile_intensity', 0):.2f}  "
        f"闭眼={sig.get('eye_closure', 0):.2f}",
    ]
    if event.vlm_label:
        au_desc = event.vlm_label.get("au_description", "")
        valence = event.vlm_label.get("valence")
        if au_desc:
            detail_lines.append(f"VLM: {au_desc[:60]}")
        if valence is not None:
            detail_lines.append(
                f"VLM valence={valence}  arousal={event.vlm_label.get('arousal')}"
            )
    detail_lines.append("")
    detail_lines.append("这个描述准确吗？")
    return short, "\n".join(detail_lines)


def _short_reason(reason: str) -> str:
    return {
        "brow_burst": "皱眉了一会儿",
        "low_vlm_conf": "表情有点模糊",
    }.get(reason, "状态变化")


class ProactiveQueryEngine:
    def __init__(
        self,
        config: EngineConfig,
        trigger: ProactiveQueryTrigger | None = None,
        rate_limiter: TokenBucketRateLimiter | None = None,
        prompt_runner: Any | None = None,
    ) -> None:
        self._config = config
        self._trigger = trigger or ProactiveQueryTrigger()
        self._rate = rate_limiter or TokenBucketRateLimiter(config.rate)
        self._prompt_runner = prompt_runner
        self._lock = threading.Lock()
        self._log = logging.getLogger("lnmem.feedback")
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
        thread = threading.Thread(
            target=self._prompt_and_record,
            args=(event, home),
            name=f"feedback-prompt-{event.frame_id}",
            daemon=True,
        )
        thread.start()
        self._stats["queried"] += 1
        return None  # 弹窗异步，主流不阻塞

    def _prompt_and_record(self, event: TriggerEvent, home: Path | None) -> None:
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
            self._record(event, response=response, latency_s=latency_s, home=home)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("prompt thread failed: %s", exc)

    def _record(
        self,
        event: TriggerEvent,
        response: UserResponse,
        latency_s: float,
        home: Path | None,
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
        }
        try:
            append_query(record, home=home)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("append_query failed: %s", exc)
