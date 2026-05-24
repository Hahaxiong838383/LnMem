"""Realtime daemon entrypoint. / 启动采集线程、存储线程和本地 HTTP 状态端点。

M0 W2 P0-1 V0.3（2026-05-11）：加 POST /trigger_mismatch 用户主动反馈通道。
- 鉴权：X-LnMem-Token Header（共享 token，daemon 启动时读 ~/.lnmem/secret_token，
  chmod 600）
- rate-limit：1 次 / 分钟（防恶意进程批量投毒触发 fit_step）
- 写入：append_query 走 persistence 统一计数（trigger_types）
- 对齐：frame_id 可选指认 + client_action_ts 时间戳 + alignment_latency_s 计算

详细设计 docs/m0-w2-p0-1-daemon-v03-design.md。
"""

from __future__ import annotations

from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import queue
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .capture_worker import CaptureWorker, CapturedFrame
from .daemon_config import HTTP_HOST, HTTP_PORT, load_config, preflight_check
from .feedback import (
    DNDConfig,
    EngineConfig,
    ProactiveQueryEngine,
    ProactiveQueryTrigger,
    RateLimiterConfig,
    TokenBucketRateLimiter,
)
from .feedback.persistence import (
    append_query,
    get_trigger_types,
    get_user_triggered_metrics,
)
from .signal_pipeline import SignalPipelinePool, SignalPipelineWorker
from .storage_manager import StorageManager

# 合法的 user_response.true_state 枚举（与 STATE_LABELS 严格对齐）
# 仅在 daemon HTTP handler 校验时用；feedback_to_batch 还会走 alias normalize
# 来源：from poc.src.lnn.expression_dynamics import STATE_LABELS （8 个标签）
_VALID_TRUE_STATES = {
    "low_energy", "focused", "distracted", "fatigued",
    "emotional_positive", "emotional_negative", "neutral", "transitioning",
}

# user_triggered rate-limit 配置（响应 codex F6）
_USER_TRIGGERED_RATE_LIMIT_SECONDS = 60  # 1 次 / 分钟


class DaemonApp:
    def __init__(self, device: str | None = None, camera: int | None = None, config_path: str | None = None) -> None:
        self._config = load_config(config_path, device_override=device, camera_override=camera)
        self._preflight = preflight_check(self._config)
        self._stop_event = threading.Event()
        self._started_at = time.monotonic()
        self._session_id = str(uuid4())
        self._capture_out: "queue.Queue[CapturedFrame]" = queue.Queue(maxsize=2)
        self._pipeline_in: "queue.Queue[CapturedFrame]" = queue.Queue(maxsize=2)
        self._pipeline_out: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=64)
        self._storage = StorageManager(self._config.storage, self._stop_event)
        self._capture = CaptureWorker(self._config.capture, self._capture_out, self._stop_event, self._session_id)
        # M0 W3 Phase 4.2 (2026-05-16 ADR-005/008): 实时 LNN inferencer（fail-soft）
        self._lnn_inferencer = self._build_lnn_inferencer()
        # ADR-028 v0.1 (5/23 cc): pyfeat_workers > 1 用 SignalPipelinePool（共享 caliper + lock）
        if self._config.pipeline.pyfeat_workers > 1:
            self._pipeline = SignalPipelinePool(
                n_workers=self._config.pipeline.pyfeat_workers,
                config=self._config.pipeline,
                input_queue=self._pipeline_in,
                output_queue=self._pipeline_out,
                stop_event=self._stop_event,
                device_name=self._config.capture.device_name,
                lnn_inferencer=self._lnn_inferencer,
            )
        else:
            self._pipeline = SignalPipelineWorker(
                self._config.pipeline,
                self._pipeline_in,
                self._pipeline_out,
                self._stop_event,
                self._config.capture.device_name,
                lnn_inferencer=self._lnn_inferencer,
            )
        self._router_threads: list[threading.Thread] = []
        self._http_server: ThreadingHTTPServer | None = None
        self._feedback_engine = self._build_feedback_engine()
        # M0 W2 P0-1 V0.3: user_triggered 状态
        self._secret_token: str | None = self._load_secret_token()
        # rate-limit 时间戳（monotonic）：初始用一个"久远的过去"防首次启动 60s 内误 rate-limit
        # （monotonic clock 在 process start 时 ~0，若初始 0.0 → uptime=40s 时 elapsed=40<60 误报）
        self._last_user_triggered_ts: float = -1e6
        self._user_triggered_lock = threading.Lock()
        # 最近一帧时间戳缓存（POST 接收时拿 trigger_ref_frame_ts）
        self._recent_frame_ts: dict[int, str] = {}  # frame_id → ISO8601 ts
        self._recent_frame_ts_lock = threading.Lock()

    def start(self) -> None:
        logging.info("starting realtime daemon: %s", json.dumps(self._preflight, ensure_ascii=False))
        self._storage.start()
        self._capture.start()
        self._pipeline.start()
        self._router_threads = [
            threading.Thread(target=self._frame_router_loop, name="lnmem-frame-router", daemon=True),
            threading.Thread(target=self._signal_router_loop, name="lnmem-signal-router", daemon=True),
        ]
        for thread in self._router_threads:
            thread.start()
        self._start_http_server()

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        logging.info("stop requested for session %s", self._session_id)
        self._stop_event.set()
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        self._capture.join(timeout=5.0)
        self._pipeline.join(timeout=10.0)
        for thread in self._router_threads:
            thread.join(timeout=5.0)
        # FB-01: 等弹窗线程落盘完成（在线 prompt 可能还在等老板回复）
        self._feedback_engine.stop(timeout_s=5.0)
        self._storage.stop()

    def serve_forever(self) -> None:
        self.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(0.2)
        finally:
            self.stop()

    def feedback_stats(self) -> dict[str, int]:
        return self._feedback_engine.stats()

    def health(self) -> dict[str, Any]:
        # V0.3: feedback 字段加 trigger_types + user_triggered_metrics（响应 codex F7）
        feedback = self.feedback_stats()
        feedback["trigger_types"] = get_trigger_types()
        feedback["user_triggered_metrics"] = get_user_triggered_metrics()
        # M0 W3 Phase 4.2 + 4.6: LNN 推理状态暴露到 /health
        lnn_health: dict[str, Any] = {"enabled": False}
        if self._lnn_inferencer is not None:
            try:
                lnn_health = self._lnn_inferencer.health_dict()
            except Exception:
                lnn_health = {"enabled": False, "error": "health_dict_failed"}
        frames_processed = getattr(self._pipeline, "frames_processed", getattr(self._pipeline, "_frames_processed", 0))
        last_frame_ts = getattr(self._pipeline, "last_frame_ts", None)
        if last_frame_ts is None:
            last_frame_ts = getattr(self._pipeline, "_last_frame_ts", None)
        return {
            "status": "ok",
            "uptime_s": round(time.monotonic() - self._started_at, 3),
            "frames_processed": frames_processed,
            "last_frame_ts": last_frame_ts or self._capture.last_frame_ts,
            "feedback": feedback,
            "lnn": lnn_health,
        }

    def recent_signals(self, seconds: int) -> list[dict[str, Any]]:
        return self._storage.get_recent(seconds)

    def _frame_router_loop(self) -> None:
        while not self._stop_event.is_set() or not self._capture_out.empty():
            try:
                item = self._capture_out.get(timeout=0.2)
            except queue.Empty:
                continue
            self._storage.enqueue_frame(item)
            _offer(self._pipeline_in, item)

    def _signal_router_loop(self) -> None:
        while not self._stop_event.is_set() or not self._pipeline_out.empty():
            try:
                item = self._pipeline_out.get(timeout=0.2)
            except queue.Empty:
                continue
            self._storage.enqueue_signal(item)
            # V0.3: 缓存 frame_id → ts 映射，供 /trigger_mismatch 查 trigger_ref_frame_ts
            fid = item.get("frame_id") if isinstance(item, dict) else None
            ts = item.get("ts") if isinstance(item, dict) else None
            if isinstance(fid, int) and isinstance(ts, str):
                self._cache_recent_frame_ts(fid, ts)
            try:
                self._feedback_engine.observe(item)
            except Exception as exc:  # noqa: BLE001
                logging.warning("feedback_engine.observe failed: %s", exc)

    # ---------- M0 W2 P0-1 V0.3 user_triggered support ----------

    def _load_secret_token(self) -> str | None:
        """从 ~/.lnmem/secret_token 读 token。返 None 表示未配置 → /trigger_mismatch
        端点禁用。

        响应 codex F6：token 文件权限必须 ≤ 600（防泄漏给其他用户）。
        """
        token_path = Path.home() / ".lnmem" / "secret_token"
        if not token_path.exists():
            logging.warning(
                "secret_token 未配置（%s 不存在），/trigger_mismatch 端点禁用。"
                "运行 scripts/lnmem-init-token.sh 生成。",
                token_path,
            )
            return None
        # 权限检查
        mode = token_path.stat().st_mode & 0o777
        if mode > 0o600:
            logging.error(
                "secret_token 权限过松（mode=%o > 0o600），拒绝加载。"
                "chmod 600 %s 后重启 daemon。",
                mode, token_path,
            )
            return None
        token = token_path.read_text(encoding="utf-8").strip()
        if not token:
            logging.warning("secret_token 文件为空，端点禁用。")
            return None
        logging.info("secret_token loaded (length=%d), /trigger_mismatch enabled", len(token))
        return token

    def get_recent_frame_ts(self, frame_id: int) -> str | None:
        """供 HTTP handler 查 trigger_ref_frame_ts。"""
        with self._recent_frame_ts_lock:
            return self._recent_frame_ts.get(frame_id)

    def latest_frame_id(self) -> int:
        """daemon 最近一帧 frame_id（0 表示未推理过任何帧）。"""
        return getattr(self._pipeline, "_frames_processed", 0)

    def latest_pred_state(self) -> str | None:
        """daemon 最近一次状态推理结果（V0.2 暂未对外暴露，先返 None）。

        V0.3 Phase A 范围内 pred_state 不强求；后续 P0-3 vboss_self 自学习时再接。
        """
        return None

    def _cache_recent_frame_ts(self, frame_id: int, ts_iso: str) -> None:
        """signal_router_loop 内调用，缓存最近 window 帧的 ts（默认保留 60 条）。"""
        with self._recent_frame_ts_lock:
            self._recent_frame_ts[frame_id] = ts_iso
            # LRU 简易实现：超过 200 帧时丢最老的
            if len(self._recent_frame_ts) > 200:
                oldest_fid = min(self._recent_frame_ts)
                self._recent_frame_ts.pop(oldest_fid, None)

    def handle_trigger_mismatch(
        self, body: dict[str, Any], caller_pid: int | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """处理 POST /trigger_mismatch 请求逻辑（独立函数便于单测）。

        Parameters
        ----------
        body : dict
            JSON 解析后的请求 body
        caller_pid : int | None
            HTTP 连接来源 PID（仅审计，None 时填 -1）

        Returns
        -------
        (status_code, response_body) : tuple[int, dict]
        """
        # === 阶段 A: 字段快速校验（不动 cooldown 状态） ===
        true_state = body.get("true_state")
        if not isinstance(true_state, str) or true_state not in _VALID_TRUE_STATES:
            return 400, {
                "error": "invalid_true_state",
                "valid_states": sorted(_VALID_TRUE_STATES),
            }
        label = body.get("label", "mismatch")
        if label not in ("mismatch", "input"):
            return 400, {"error": "invalid_label", "valid": ["mismatch", "input"]}
        note = body.get("note")
        if note is not None and not isinstance(note, str):
            return 400, {"error": "invalid_note"}

        # === 阶段 B: cooldown 预检（lock 内，不占位） ===
        # codex daily check 2026-05-12: 原实现"先占位再校验" → 无效请求会消耗 60s 配额，
        # 让紧随的合法请求被 429。改两阶段 lock：
        #   B 预检 → 失败 429（不占位）
        #   C 深度校验（lock 外，开销大；失败直接 4xx 不污染配额）
        #   D double-check + 占位（lock 内，防 B→D 期间并发抢占）
        now_mono = time.monotonic()
        with self._user_triggered_lock:
            elapsed = now_mono - self._last_user_triggered_ts
            if elapsed < _USER_TRIGGERED_RATE_LIMIT_SECONDS:
                return 429, {
                    "error": "rate_limited",
                    "retry_after_s": round(_USER_TRIGGERED_RATE_LIMIT_SECONDS - elapsed, 1),
                }

        # === 阶段 C: 深度校验（lock 外，失败不污染配额） ===
        # 3. frame_id 处理（用户指认 OR fallback 最近帧）
        latest_fid = self.latest_frame_id()
        if latest_fid == 0:
            return 503, {"error": "no_frames_yet", "hint": "daemon 未推理过任何帧"}
        user_frame_id = body.get("frame_id")
        if user_frame_id is not None:
            if not isinstance(user_frame_id, int) or user_frame_id <= 0:
                return 400, {"error": "invalid_frame_id"}
            if self.get_recent_frame_ts(user_frame_id) is None:
                return 503, {
                    "error": "frame_not_in_range",
                    "hint": f"frame_id {user_frame_id} not in recent cache",
                    "latest_frame_id": latest_fid,
                }
            frame_id = user_frame_id
        else:
            frame_id = latest_fid

        trigger_ref_frame_ts = self.get_recent_frame_ts(frame_id)
        if trigger_ref_frame_ts is None:
            trigger_ref_frame_ts = datetime.now(timezone.utc).isoformat()

        # 4. client_action_ts 处理
        client_action_ts = body.get("client_action_ts")
        if client_action_ts is None:
            client_action_ts = datetime.now(timezone.utc).isoformat()
        elif not isinstance(client_action_ts, str):
            return 400, {"error": "invalid_client_action_ts"}

        # 5. 计算 alignment_latency_s
        try:
            ref_dt = datetime.fromisoformat(trigger_ref_frame_ts.replace("Z", "+00:00"))
            action_dt = datetime.fromisoformat(client_action_ts.replace("Z", "+00:00"))
            alignment_latency_s = (action_dt - ref_dt).total_seconds()
        except Exception:
            alignment_latency_s = 0.0

        # 6. warning（不阻断）
        warning = None
        if alignment_latency_s > self._config.feedback.__dict__.get("alignment_window_s", 5.0):
            warning = (
                f"alignment_latency_s={alignment_latency_s:.1f}s > 5.0s; "
                f"此样本写入但 feedback_to_batch 会硬过滤不入训练批"
            )

        # 7. 构造 record
        event_uuid = str(uuid4())
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "frame_id": frame_id,
            "trigger_ref_frame_ts": trigger_ref_frame_ts,
            "client_action_ts": client_action_ts,
            "alignment_latency_s": round(alignment_latency_s, 3),
            "reason": "user_triggered",
            "signal": None,
            "vlm_label": None,
            "user_response": {
                "label": label,
                "true_state": true_state,
                "note": note,
            },
            "latency_s": 0.0,
            "session_id": self._session_id,
            "event_uuid": event_uuid,
            "caller_source": {
                "pid": caller_pid if caller_pid is not None else -1,
                "cmdline": "user_triggered",
            },
        }

        # === 阶段 D: cooldown double-check + 占位（lock 内） ===
        # 阶段 B→C 之间并发请求可能抢占；double-check 防 TOCTOU
        now_mono2 = time.monotonic()
        with self._user_triggered_lock:
            elapsed2 = now_mono2 - self._last_user_triggered_ts
            if elapsed2 < _USER_TRIGGERED_RATE_LIMIT_SECONDS:
                return 429, {
                    "error": "rate_limited",
                    "retry_after_s": round(_USER_TRIGGERED_RATE_LIMIT_SECONDS - elapsed2, 1),
                }
            self._last_user_triggered_ts = now_mono2

        # === 阶段 E: 写 JSONL（lock 外，rate-limit 已占位） ===
        try:
            append_query(record)
        except Exception as exc:  # noqa: BLE001
            logging.error("trigger_mismatch append_query failed: %s", exc)
            return 500, {"error": "write_failed", "detail": str(exc)}

        return 200, {
            "ok": True,
            "event_uuid": event_uuid,
            "frame_id": frame_id,
            "trigger_ref_frame_ts": trigger_ref_frame_ts,
            "client_action_ts": client_action_ts,
            "alignment_latency_s": round(alignment_latency_s, 3),
            "pred_state": self.latest_pred_state(),
            "warning": warning,
        }

    def _build_lnn_inferencer(self):
        """M0 W3 Phase 4.2 (ADR-005/008): 实时 LNN inferencer 构造 + fail-soft

        优先级（codex F-04 fix: 显式 path > max-glob）：
        1. env LNMEM_LNN_CKPT 指定路径
        2. 最新 m0w2_phase2/*.pt（fallback）
        3. 失败 → 返 None（daemon 仍可启动）
        """
        from pathlib import Path
        from poc.src.realtime.lnn_inferencer import RealtimeLNNInferencer
        # F5 (P1): 用 resolve_runtime_config() 而非 import-time 快照
        from poc.src.realtime.feedback.trigger import resolve_runtime_config

        ckpt_env = os.environ.get("LNMEM_LNN_CKPT")
        ckpt_path: Path | None = None
        if ckpt_env:
            ckpt_path = Path(os.path.expanduser(ckpt_env))
        else:
            ckpt_dir = Path(__file__).resolve().parent.parent.parent.parent / "poc" / "data" / "checkpoints" / "m0w2_phase2"
            try:
                ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if ckpts:
                    ckpt_path = ckpts[0]
            except Exception:
                ckpt_path = None

        if ckpt_path is None or not ckpt_path.exists():
            logging.getLogger(__name__).warning(
                "no LNN ckpt found (env LNMEM_LNN_CKPT=%s, fallback dir empty) → inferencer disabled",
                ckpt_env or "<unset>",
            )
            return RealtimeLNNInferencer(ckpt_path=None)

        return RealtimeLNNInferencer(
            ckpt_path=ckpt_path,
            persona_idx=int(os.environ.get("LNMEM_PERSONA_IDX", "30")),
            signal_path=resolve_runtime_config()["signal_path"],
        )

    def _build_feedback_engine(self) -> ProactiveQueryEngine:
        from pathlib import Path

        fb = self._config.feedback
        rate_state = Path.home() / ".lnmem" / "feedback" / "rate_state.json"
        rate = RateLimiterConfig(
            min_per_hour=fb.query_min_per_hour,
            max_per_hour=fb.query_max_per_hour,
            state_path=rate_state,
        )
        dnd = DNDConfig(apps=fb.dnd_apps)
        cfg = EngineConfig(rate=rate, dnd=dnd, enable_prompt=True)
        # FB-03: 把 daemon.toml 的 query_confidence_threshold 接到 trigger
        # ADR-005 F-08 fix: daemon 启动时根据 signal_path 显式传 brow 阈值 + signal_path 实例属性
        # TODO-1 (ADR-010): raw 路径下默认启用 Schmitt + EMA，显式传 brow_thr_high/low + α
        # F5 (P1) ADR-010 codex review (2026-05-17)：
        #   每次构建 trigger 时调 resolve_runtime_config()，**不再** 读 module-level
        #   _SIGNAL_PATH / _BROW_*_RAW 等 import-time 快照常量。
        #   这样运行时切 ~/.lnmem/trigger_mode 或测试 monkeypatch env 都能立刻反映。
        from poc.src.realtime.feedback.trigger import resolve_runtime_config
        _cfg = resolve_runtime_config()
        _sp = _cfg["signal_path"]
        _mode = _cfg["mode"]
        _brow_thr = (
            _cfg["brow_burst_threshold_raw"] if _sp == "raw"
            else _cfg["brow_burst_threshold_cal"]
        )
        # raw 路径才启用 Schmitt+EMA；calibrated 仍走 legacy strict-deque（向后兼容）
        if _sp == "raw":
            trigger = ProactiveQueryTrigger(
                vlm_conf_threshold=fb.query_confidence_threshold,
                brow_threshold=_brow_thr,
                brow_window=_cfg["brow_burst_window"],
                pose_held_window=_cfg["pose_held_window"],
                pose_held_stddev_threshold_deg=_cfg["pose_held_stddev_threshold_deg"],
                pose_outlier_abs_threshold_deg=_cfg["pose_outlier_abs_threshold_deg"],
                cooldown_frames_brow=_cfg["cooldown_frames_brow"],
                cooldown_frames_pose_held=_cfg["cooldown_frames_pose_held"],
                cooldown_frames_low_vlm_conf=_cfg["cooldown_frames_low_vlm_conf"],
                signal_path=_sp,
                use_schmitt_ema=True,
                brow_ema_alpha=_cfg["brow_ema_alpha_raw"],
                brow_thr_high=_cfg["brow_thr_high_raw"],
                brow_thr_low=_cfg["brow_thr_low_raw"],
            )
            logging.getLogger(__name__).info(
                "trigger init: mode=%s signal_path=%s schmitt_ema=ON alpha=%.2f thr_high=%.2f thr_low=%.2f cooldown=%d",
                _mode, _sp,
                _cfg["brow_ema_alpha_raw"], _cfg["brow_thr_high_raw"], _cfg["brow_thr_low_raw"],
                _cfg["cooldown_frames_brow"],
            )
        else:
            trigger = ProactiveQueryTrigger(
                vlm_conf_threshold=fb.query_confidence_threshold,
                brow_threshold=_brow_thr,
                brow_window=_cfg["brow_burst_window"],
                pose_held_window=_cfg["pose_held_window"],
                pose_held_stddev_threshold_deg=_cfg["pose_held_stddev_threshold_deg"],
                pose_outlier_abs_threshold_deg=_cfg["pose_outlier_abs_threshold_deg"],
                cooldown_frames_brow=_cfg["cooldown_frames_brow"],
                cooldown_frames_pose_held=_cfg["cooldown_frames_pose_held"],
                cooldown_frames_low_vlm_conf=_cfg["cooldown_frames_low_vlm_conf"],
                signal_path=_sp,
            )
            logging.getLogger(__name__).info(
                "trigger init: mode=%s signal_path=%s brow_threshold=%.2f (legacy strict-deque)",
                _mode, _sp, _brow_thr,
            )
        return ProactiveQueryEngine(
            cfg,
            trigger=trigger,
            rate_limiter=TokenBucketRateLimiter(rate),
            session_id=self._session_id,  # FB-06
        )

    def _start_http_server(self) -> None:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    self._write_json(200, app.health())
                    return
                if parsed.path == "/feedback/stats":
                    self._write_json(200, app.feedback_stats())
                    return
                if parsed.path == "/signals/recent":
                    seconds = int(parse_qs(parsed.query).get("seconds", ["300"])[0])
                    self._write_json(200, app.recent_signals(seconds))
                    return
                self._write_json(404, {"error": "not_found"})

            def do_POST(self) -> None:  # noqa: N802
                if self.path == "/control/stop":
                    threading.Thread(target=app.stop, name="lnmem-stop", daemon=True).start()
                    self._write_json(202, {"status": "stopping"})
                    return
                if self.path == "/trigger_mismatch":
                    self._handle_trigger_mismatch()
                    return
                self._write_json(404, {"error": "not_found"})

            def _handle_trigger_mismatch(self) -> None:
                """V0.3: POST /trigger_mismatch — 用户主动反馈通道（响应 codex F1-F8）。"""
                # 鉴权（响应 codex F6）
                if app._secret_token is None:
                    self._write_json(503, {
                        "error": "endpoint_disabled",
                        "hint": "secret_token 未配置，daemon 启动时未加载",
                    })
                    return
                supplied = self.headers.get("X-LnMem-Token", "")
                if not supplied:
                    self._write_json(401, {"error": "missing_token"})
                    return
                # 等长 + timing-safe 比较（防 timing attack）
                import hmac
                if not hmac.compare_digest(supplied, app._secret_token):
                    self._write_json(401, {"error": "invalid_token"})
                    return

                # 读 body
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._write_json(400, {"error": "invalid_content_length"})
                    return
                if content_length <= 0 or content_length > 8192:
                    self._write_json(400, {"error": "body_size_out_of_range"})
                    return
                raw = self.rfile.read(content_length)
                try:
                    body = json.loads(raw.decode("utf-8"))
                except Exception:
                    self._write_json(400, {"error": "invalid_json"})
                    return
                if not isinstance(body, dict):
                    self._write_json(400, {"error": "body_must_be_object"})
                    return

                # 委托给 DaemonApp.handle_trigger_mismatch
                # caller_pid 暂用 -1（HTTP 连接很难拿到真实 PID）
                code, resp = app.handle_trigger_mismatch(body, caller_pid=-1)
                self._write_json(code, resp)

            def log_message(self, _fmt: str, *_args: Any) -> None:
                logging.info("http %s", self.path)

            def _write_json(self, code: int, payload: Any) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._http_server = ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), Handler)
        threading.Thread(target=self._http_server.serve_forever, name="lnmem-http", daemon=True).start()


def _request_camera_auth_macos(camera_index: int) -> bool:
    """在主线程触发 macOS TCC 摄像头授权对话框，返回是否授权成功。"""
    import os
    import sys
    if sys.platform != "darwin":
        return True
    # 先不跳过 auth，让系统弹出 TCC 对话框
    os.environ.pop("OPENCV_AVFOUNDATION_SKIP_AUTH", None)
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(camera_index)
        authorized = cap.isOpened()
        cap.release()
        if not authorized:
            logging.warning(
                "camera TCC not authorized. "
                "Grant permission in System Settings → Privacy & Security → Camera, "
                "then restart daemon."
            )
        return authorized
    except Exception as exc:
        logging.warning("camera auth check failed: %s", exc)
        return False


def _load_dotenv() -> None:
    """自动加载项目根目录 poc/.env（含 VLM API key）。不报错，没文件就跳过。"""
    import pathlib
    env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_path, override=False)
            logging.info("daemon: loaded %s", env_path)
        except ImportError:
            # dotenv 未安装时手动解析
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        import os
                        os.environ.setdefault(k, v)
            logging.info("daemon: loaded %s (manual parse)", env_path)


def main() -> None:
    parser = ArgumentParser(description="LnMem realtime capture daemon")
    parser.add_argument("--device", default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-camera-auth", action="store_true",
                        help="跳过主线程 TCC 授权预检（已确认权限时用）")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_dotenv()  # 加载 poc/.env → GEMINI_PROXY_KEY / MINIMAX_API_KEY
    if not args.skip_camera_auth:
        cam_idx = args.camera if args.camera is not None else 0
        _request_camera_auth_macos(cam_idx)
    app = DaemonApp(device=args.device, camera=args.camera, config_path=args.config)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _signum, _frame: app.stop())

    # W2-P0-bugfix（2026-05-16）：SIGUSR1 → faulthandler dump 所有线程栈
    # 用于"daemon HTTP 假死但进程还在"的事后复盘。watchdog 重启前会 kill -USR1。
    #
    # ⚠️ 已知 issue（2026-05-16 实测）：daemon 进程内 faulthandler.register 注册成功
    # （lsof 显示 fd 持有 dump 文件）但 SIGUSR1 触发后文件不增长。
    # 单文件 sanity test（同 venv python）工作正常。
    # 推测：OpenCV / threading 库可能 hook SIGUSR1 抢占，或 GIL/C 扩展 block 影响。
    # 待深挖根因（>1h，已 backlog）。
    #
    # 不影响 watchdog 核心 RESTART 功能：watchdog 用 mtime+size 双校验，
    # dump 未写入会 log "NOT OBSERVED" WARN，但 RESTART_NEEDED 仍会被 mark。
    # 即"SIGUSR1 dump 是 nice-to-have，不是 watchdog 重启路径的依赖项"。
    import faulthandler as _fh
    _dump_path = os.path.expanduser(os.environ.get(
        "DAEMON_TRACEBACK_DUMP", "~/.lnmem/daemon_traceback.log"
    ))
    try:
        os.makedirs(os.path.dirname(_dump_path), exist_ok=True)
        # 用 O_APPEND 打开常驻文件供 faulthandler 写入（不可关闭，必须保持引用）
        _dump_file_handle = open(_dump_path, "ab", buffering=0)
        _fh.register(signal.SIGUSR1, file=_dump_file_handle, all_threads=True, chain=False)
        logging.getLogger(__name__).info(
            "SIGUSR1 faulthandler registered → dump path=%s (known issue: dump may not write in daemon ctx, see comment)",
            _dump_path,
        )
    except Exception as _reg_exc:
        # 注册失败不阻止 daemon 启动，但要明确 log
        logging.getLogger(__name__).warning(
            "SIGUSR1 faulthandler.register failed: %s — dump will be unavailable", _reg_exc
        )
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        app.stop()


def _offer(target: queue.Queue[Any], item: Any) -> None:
    try:
        target.put_nowait(item)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        target.put_nowait(item)


__all__ = ["DaemonApp", "main"]


if __name__ == "__main__":
    main()
