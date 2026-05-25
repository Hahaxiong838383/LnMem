"""Signal pipeline worker. / py-feat 信号提取线程（LNN-FIRST）。

LNN-FIRST 原则：py-feat 是唯一面部信号来源。
AUExtractor.extract() 同时返回 AU / 情绪 / 头部姿态（img2pose），
不再引入 MediaPipe 作为备用头姿估计路径。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
import logging
import queue
import threading
import time
from typing import Any

from .capture_worker import CapturedFrame
from .daemon_config import PipelineConfig
from .pyfeat_mp_pool import PyfeatMpPool


class SignalPipelineWorker(threading.Thread):
    # VLM 标注每 N 帧触发一次（1 FPS × 10 = 约 10 秒/次）
    _VLM_INTERVAL_FRAMES = 10

    def __init__(
        self,
        config: PipelineConfig,
        input_queue: "queue.Queue[CapturedFrame]",
        output_queue: "queue.Queue[dict[str, Any]]",
        stop_event: threading.Event,
        device_name: str,
        lnn_inferencer: Any | None = None,  # M0 W3 Phase 4.2: optional realtime LNN forward
        # ADR-028 v0.1 (5/23 cc): 2 worker py-feat pool 支持
        name: str = "lnmem-signal",
        shared_caliper: Any | None = None,  # 多 worker 共享同一 AUCaliper（per-user 校准状态不能分裂）
        shared_caliper_lock: threading.Lock | None = None,  # 配合 shared_caliper 防并发 mutation
        shared_lnn_lock: threading.Lock | None = None,  # 配合 shared lnn_inferencer 防 hidden state 错位
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._config = config
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._device_name = device_name
        self._frames_processed = 0
        self._last_frame_ts: str | None = None
        self._au_extractor: Any | None = None
        # caliper: shared 模式（worker pool） 或 own 模式（单 worker 向后兼容）
        self._caliper: Any | None = shared_caliper
        self._caliper_lock = shared_caliper_lock  # None 表示单 worker 无需 lock
        self._vlm_annotator: Any | None = None
        self._lnn_inferencer = lnn_inferencer  # M0 W3 Phase 4.2
        self._lnn_lock = shared_lnn_lock  # None 表示无 lock（单 worker 或 lnn_inferencer 内部线程安全）

    def run(self) -> None:
        while not self._stop_event.is_set() or not self._input_queue.empty():
            try:
                item = self._input_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                result = self._process(item)
                _offer(self._output_queue, result)
                self._frames_processed += 1
                self._last_frame_ts = result["ts"]
            except Exception:
                logging.exception("signal pipeline failed for frame_id=%s", item.frame_id)
        self._close_resources()

    def _process(self, item: CapturedFrame) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ts": item.ts,
            "device": self._device_name,
            "frame_id": item.frame_id,
            "raw_video_offset_ms": item.raw_video_offset_ms,
            "face_detected": False,
            "raw_au": {},
            "calibrated_au": {},
            "expression": {},
            # head_pose: filled by py-feat img2pose below (yaw/pitch/roll in degrees)
            "head_pose": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            "session_id": item.session_id,
        }

        if self._config.enable_pyfeat:
            raw_au, _emo, pose = self._extract_pyfeat(item.frame)
            # ADR-011 V0.3.8: py-feat img2pose 失败时 raw_au 非空但全 NaN，
            # head_pose 返上次缓存值。早期判定为"无脸"，防 LNN 训练数据 + queries 污染。
            if raw_au and _has_finite_value(raw_au):
                result["face_detected"] = True
                result["raw_au"] = _round_map(raw_au)
                result["head_pose"] = {k: round(v, 3) for k, v in pose.items()}
                caliper = self._ensure_caliper()
                # ADR-028 v0.1: shared caliper 必须加锁（per-user baseline EMA mutation 不可并发）
                if self._caliper_lock is not None:
                    with self._caliper_lock:
                        calibrated = caliper.step(raw_au)
                        expression = caliper.to_lnn_expression(calibrated)
                else:
                    calibrated = caliper.step(raw_au)
                    expression = caliper.to_lnn_expression(calibrated)
                result["calibrated_au"] = _round_map(calibrated)
                result["expression"] = _round_map(expression)

        # VLM 标注：每 _VLM_INTERVAL_FRAMES 帧触发一次（面朝检测到时才标注）
        if (
            result["face_detected"]
            and self._frames_processed % self._VLM_INTERVAL_FRAMES == 0
        ):
            vlm_label = self._annotate_vlm(item.frame)
            if vlm_label is not None:
                result["vlm_label"] = vlm_label

        # M0 W3 Phase 4.2: 实时 LNN forward（F-06 fail-soft，独立 try/except）
        # 失败仅清空 lnn 字段，不影响基础信号出队（codex F-06 修复）
        # ADR-028 v0.1: shared lnn_inferencer 必须加锁（LNN hidden state 有 history，不能并发更新）
        if self._lnn_inferencer is not None and result["face_detected"]:
            try:
                if self._lnn_lock is not None:
                    with self._lnn_lock:
                        infer_result = self._lnn_inferencer.infer(result)
                else:
                    infer_result = self._lnn_inferencer.infer(result)
                result["lnn"] = {
                    "status": infer_result.lnn_status,
                    "predicted_state": infer_result.predicted_state,
                    "state_probs": infer_result.state_probs,  # 5/25 cc: 暴露完整 8 类概率给 UI stacked area
                    "top_prob": infer_result.top_prob,
                    "margin": infer_result.margin,
                    "entropy": infer_result.entropy,
                    "intent_probs": infer_result.intent_probs,
                    "intent_pred": infer_result.intent_pred,
                    "intent_confidence": infer_result.intent_confidence,
                    "ckpt_sha8": infer_result.ckpt_sha8,
                    "signal_path": infer_result.signal_path_used,
                    "error": infer_result.error,
                }
            except Exception:
                logging.exception("lnn_inferencer.infer failed (continuing without lnn)")
                result["lnn"] = {"status": "error", "error": "exception_in_signal_pipeline"}

        return result

    def _extract_pyfeat(
        self, frame: Any
    ) -> tuple[Mapping[str, float], dict[str, float], dict[str, float]]:
        """Call AUExtractor; returns (raw_au, emo, pose). / 调用 AUExtractor，返回三元组。"""
        if self._au_extractor is None:
            from poc.src.vision.au_extractor import AUExtractor

            self._au_extractor = AUExtractor()
        raw_au, emo, pose = self._au_extractor.extract(frame)
        return raw_au, emo, pose

    def _ensure_caliper(self) -> Any:
        if self._caliper is None:
            from poc.src.vision.feature_calibrator import AUCaliper

            self._caliper = AUCaliper()
        return self._caliper

    def _annotate_vlm(self, frame: Any) -> "dict[str, Any] | None":
        """VLM 标注（同步，阻塞约 1-3s，PoC 可接受）。"""
        try:
            if self._vlm_annotator is None:
                from poc.src.vision.vlm_annotator import VLMAnnotator
                self._vlm_annotator = VLMAnnotator.from_env()
            label = self._vlm_annotator.annotate(frame)
            return label.to_dict() if label else None
        except Exception:
            logging.exception("VLM annotation failed")
            return None

    def _close_resources(self) -> None:
        self._au_extractor = None
        self._caliper = None
        self._vlm_annotator = None


def _round_map(values: Mapping[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in values.items()}


def _has_finite_value(values: Mapping[str, Any]) -> bool:
    """ADR-011 V0.3.8: True 若 mapping 至少含一个有限数（防 py-feat 假阳性脸全 NaN）。

    codex 一审 P0-1：用 numbers.Real + float() 转换，统一覆盖 Python/numpy/Decimal
    数值类型（裸 isinstance(v,(int,float)) 对 numpy.float32 返 False → 误杀真脸）。
    保守判定：缺/空 dict 由上层 `if raw_au` 处理，本函数只在非空时被调用。

    注：本函数与 trigger.py:_is_finite_number 是相同语义的双层兜底（pipeline 治本 +
    trigger 兜底）。未来可抽到 poc.src.common.numeric_utils 共享（ADR-011 P2 提及）。
    """
    from numbers import Real
    for v in values.values():
        if isinstance(v, bool):
            continue
        if not isinstance(v, Real):
            continue
        try:
            if math.isfinite(float(v)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _offer(target: "queue.Queue[dict[str, Any]]", item: dict[str, Any]) -> None:
    try:
        target.put_nowait(item)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        target.put_nowait(item)


class SignalPipelineMpDispatcher(threading.Thread):
    """Dispatch captured frames from thread queue to multiprocessing input queue."""

    def __init__(
        self,
        mp_pool: PyfeatMpPool,
        input_queue: "queue.Queue[CapturedFrame]",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="lnmem-mp-dispatcher", daemon=True)
        self._mp_pool = mp_pool
        self._input_queue = input_queue
        self._stop_event = stop_event
        self._frames_dispatched = 0

    def run(self) -> None:
        while not self._stop_event.is_set() or not self._input_queue.empty():
            try:
                captured = self._input_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            submitted = self._mp_pool.submit(captured, block=True, timeout=0.2)
            if submitted:
                self._frames_dispatched += 1
            else:
                logging.warning(
                    "mp dispatcher drop frame_id=%s (input queue full)", captured.frame_id
                )

    @property
    def frames_dispatched(self) -> int:
        return self._frames_dispatched


class SignalPipelineMpRouter(threading.Thread):
    """Main-process router: consume mp results, run caliper/LNN, then push output queue."""

    def __init__(
        self,
        mp_pool: PyfeatMpPool,
        output_queue: "queue.Queue[dict[str, Any]]",
        stop_event: threading.Event,
        device_name: str,
        lnn_inferencer: Any | None = None,
    ) -> None:
        super().__init__(name="lnmem-mp-router", daemon=True)
        self._mp_pool = mp_pool
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._device_name = device_name
        self._lnn_inferencer = lnn_inferencer
        self._caliper: Any | None = None
        self._frames_processed = 0
        self._last_frame_ts: str | None = None

    def run(self) -> None:
        while not self._stop_event.is_set() or self._mp_pool.is_alive():
            result = self._mp_pool.get_result(timeout=0.2)
            if result is None:
                continue
            try:
                processed = self._post_process(result)
                _offer(self._output_queue, processed)
                self._frames_processed += 1
                self._last_frame_ts = processed["ts"]
            except Exception:
                logging.exception("mp router post_process failed")

    def _post_process(self, res: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ts": res["ts"],
            "device": self._device_name,
            "frame_id": res["frame_id"],
            "raw_video_offset_ms": res["raw_video_offset_ms"],
            "face_detected": False,
            "raw_au": {},
            "calibrated_au": {},
            "expression": {},
            "head_pose": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            "session_id": res["session_id"],
        }

        raw_au = res.get("raw_au") or {}
        pose = res.get("head_pose") or {}
        if raw_au and _has_finite_value(raw_au):
            au_sum = sum(
                abs(float(value))
                for value in raw_au.values()
                if isinstance(value, (int, float))
            )
            if au_sum > 0.001:
                out["face_detected"] = True
                out["raw_au"] = _round_map(raw_au)
                if pose:
                    out["head_pose"] = {
                        key: round(float(value), 3)
                        for key, value in pose.items()
                    }
                caliper = self._ensure_caliper()
                calibrated = caliper.step(raw_au)
                out["calibrated_au"] = _round_map(calibrated)
                out["expression"] = _round_map(caliper.to_lnn_expression(calibrated))

        if self._lnn_inferencer is not None and out["face_detected"]:
            try:
                infer_result = self._lnn_inferencer.infer(out)
                out["lnn"] = {
                    "status": infer_result.lnn_status,
                    "predicted_state": infer_result.predicted_state,
                    "state_probs": infer_result.state_probs,  # 5/25 cc: 8 类完整概率
                    "top_prob": infer_result.top_prob,
                    "margin": infer_result.margin,
                    "entropy": infer_result.entropy,
                    "intent_probs": infer_result.intent_probs,
                    "intent_pred": infer_result.intent_pred,
                    "intent_confidence": infer_result.intent_confidence,
                    "ckpt_sha8": infer_result.ckpt_sha8,
                    "signal_path": infer_result.signal_path_used,
                    "error": infer_result.error,
                }
            except Exception:
                logging.exception("lnn_inferencer.infer failed")
                out["lnn"] = {"status": "error", "error": "exception_in_router"}
        return out

    def _ensure_caliper(self) -> Any:
        if self._caliper is None:
            from poc.src.vision.feature_calibrator import AUCaliper

            self._caliper = AUCaliper()
        return self._caliper

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def last_frame_ts(self) -> str | None:
        return self._last_frame_ts


class SignalPipelinePool:
    """ADR-028 v0.1 (5/23 cc 写框架 + codex 评审兜底): N worker py-feat 并行池.

    设计要点：
    - **共享 caliper**：per-user EMA baseline 状态不能分裂，必须 shared + lock 保护
    - **共享 lnn_inferencer**：LNN hidden state 有 history，并发推理会错位 → lock 保护
    - **独立 AUExtractor**：py-feat 模型加载后是 read-only 推理；独立加载避免并发风险，
      代价是 N × ~2GB RAM（M5 Air 16GB 完全够 2 worker = 4GB）
    - **input/output queue 共享**：queue.Queue 线程安全，N worker 同时 get/put OK
    - **out-of-order 风险**：worker 处理速度不一致 → output 可能不按 frame_id 顺序
      验收方案：jsonl 写入端按 frame_id sort buffer（已在 storage_manager 处理；如未处理需补）

    单 worker 向后兼容：DaemonApp 检查 pyfeat_workers，==1 时仍用 SignalPipelineWorker 直挂.
    """

    def __init__(
        self,
        n_workers: int,
        config: PipelineConfig,
        input_queue: "queue.Queue[CapturedFrame]",
        output_queue: "queue.Queue[dict[str, Any]]",
        stop_event: threading.Event,
        device_name: str,
        lnn_inferencer: Any | None = None,
    ) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        if n_workers > 8:
            raise ValueError(f"n_workers must be <= 8 (M5 Air cores), got {n_workers}")
        self._n_workers = n_workers
        self._shared_caliper_lock = threading.Lock() if n_workers > 1 else None
        self._shared_lnn_lock = threading.Lock() if (n_workers > 1 and lnn_inferencer is not None) else None
        # 共享 caliper：N workers 同一 AUCaliper 实例（per-user baseline 状态一致性）
        # 延迟创建（避免 import 副作用）；worker 首次访问时 _ensure_caliper 触发
        self._shared_caliper_holder: list[Any] = [None]  # mutable container
        self._shared_caliper_init_lock = threading.Lock()

        def _ensure_shared_caliper() -> Any:
            with self._shared_caliper_init_lock:
                if self._shared_caliper_holder[0] is None:
                    from poc.src.vision.feature_calibrator import AUCaliper
                    self._shared_caliper_holder[0] = AUCaliper()
                return self._shared_caliper_holder[0]

        # 预创建共享 caliper（保证所有 worker 拿到同一个）
        _ensure_shared_caliper()
        shared_caliper = self._shared_caliper_holder[0]

        self._workers: list[SignalPipelineWorker] = [
            SignalPipelineWorker(
                config=config,
                input_queue=input_queue,
                output_queue=output_queue,
                stop_event=stop_event,
                device_name=device_name,
                lnn_inferencer=lnn_inferencer,
                name=f"lnmem-signal-{i}",
                shared_caliper=shared_caliper,
                shared_caliper_lock=self._shared_caliper_lock,
                shared_lnn_lock=self._shared_lnn_lock,
            )
            for i in range(n_workers)
        ]

    def start(self) -> None:
        for w in self._workers:
            w.start()

    def join(self, timeout: float | None = None) -> None:
        if timeout is None:
            for w in self._workers:
                w.join()
            return
        # total timeout budget: keep API semantics aligned with threading.Thread.join
        deadline = time.monotonic() + timeout
        for w in self._workers:
            remain = max(0.0, deadline - time.monotonic())
            w.join(remain)

    def is_alive(self) -> bool:
        return any(w.is_alive() for w in self._workers)

    @property
    def frames_processed(self) -> int:
        return sum(w._frames_processed for w in self._workers)

    @property
    def last_frame_ts(self) -> str | None:
        candidates = [w._last_frame_ts for w in self._workers if w._last_frame_ts]
        return max(candidates) if candidates else None

    @property
    def n_workers(self) -> int:
        return self._n_workers


__all__ = [
    "SignalPipelineMpDispatcher",
    "SignalPipelineMpRouter",
    "SignalPipelinePool",
    "SignalPipelineWorker",
]
