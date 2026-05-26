"""RealtimeLNNInferencer with temporal windows and smooth checkpoint reload.

封装 LNN ckpt 加载 + 实时 forward + margin/entropy 计算，给 signal_pipeline 用。
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from poc.src.fusion import MultimodalFusion
from poc.src.lnn.expression_dynamics import STATE_LABELS
from poc.src.lnn.intent_head import INTENT_LABELS, IntentClassifierHead
from poc.src.lnn.model import DefaultLNNFactory
from poc.src.lnn.online_adapter import OnlineLoRAAdapter
from poc.src.lnn.online_contracts import OnlineAdapterConfig
from poc.src.lnn.signals_loader import _signals_row_to_frame_dict

# 5/25 cc: 60 帧 × 1.07fps = 56s 暖机太慢（川哥实测「做笑脸几十秒没反馈」）。
# 改 8 帧 ≈ 8s 暖机（phase2 ckpt seq_len=8 训练，匹配）。可被 env LNMEM_INFER_WINDOW 覆盖。
_DEFAULT_WINDOW = int(os.environ.get("LNMEM_INFER_WINDOW", "8"))
_RECENT_BUFFER_MAXLEN = 300
_TRANSITION_FRAMES = 30


@dataclass(frozen=True)
class InferenceResult:
    """LNN 单帧推理结果（不可变）"""

    lnn_status: str  # "ok" | "warmup" | "error" | "disabled"
    predicted_state: str | None = None
    state_probs: list[float] | None = None
    top_prob: float | None = None
    margin: float | None = None
    entropy: float | None = None
    error: str | None = None
    ckpt_sha8: str | None = None
    signal_path_used: str | None = None
    intent_probs: list[float] | None = None
    intent_pred: str | None = None
    intent_confidence: float | None = None


class RealtimeLNNInferencer:
    """Realtime LNN forward 封装。daemon 内单线程使用（signal_pipeline thread）。"""

    def __init__(
        self,
        ckpt_path: Path | None = None,
        intent_head_ckpt: Path | None = None,
        persona_idx: int = 30,
        window: int = _DEFAULT_WINDOW,
        signal_path: str = "calibrated",
    ) -> None:
        self._persona_idx = persona_idx
        self._window = window
        self._signal_path = signal_path
        self._fusion = MultimodalFusion()
        self._history: deque[list[float]] = deque(maxlen=window)
        self._recent_temporal: deque[tuple[int, list[float], list[float]]] = deque(
            maxlen=_RECENT_BUFFER_MAXLEN
        )

        self._state_model = None
        self._ckpt_sha8: str | None = None
        self._intent_head: IntentClassifierHead | None = None
        self._intent_head_ckpt_sha8: str | None = None

        self._transition_old_state_model = None
        self._transition_old_intent_head: IntentClassifierHead | None = None
        self._transition_progress_frames = 0
        self._transition_total_frames = _TRANSITION_FRAMES
        self._in_transition = False

        self._enabled = False
        self._fail_count = 0
        self._fail_count_circuit_break = 100
        self._logger = logging.getLogger(__name__)

        # ADR-037：state 投影对齐 + intent EMA 平滑
        # 缓存最新 infer 的 state_probs/intent_probs，供 predict_future 第 0 步对齐
        self._last_state_probs: list[float] | None = None
        self._last_intent_probs: list[float] | None = None
        # intent EMA 平滑：新值权重 α，老值 (1-α)
        # 修复川哥反馈"intent 1 秒翻 3 次"（gemini 钦定"快变量欠阻尼"）
        try:
            self._intent_ema_alpha = float(os.environ.get("LNMEM_INTENT_EMA_ALPHA", "0.3"))
        except (TypeError, ValueError):
            self._intent_ema_alpha = 0.3
        self._intent_ema_alpha = max(0.05, min(1.0, self._intent_ema_alpha))
        self._intent_ema_probs: list[float] | None = None

        if ckpt_path is None:
            self._logger.warning("RealtimeLNNInferencer: ckpt_path=None → disabled")
            return

        try:
            self._load_ckpt(ckpt_path)
            if intent_head_ckpt is not None:
                self._load_intent_head_ckpt(intent_head_ckpt)
            self._enabled = True
            self._logger.info(
                "RealtimeLNNInferencer ready: ckpt=%s sha8=%s intent=%s persona_idx=%d signal_path=%s",
                ckpt_path.name,
                self._ckpt_sha8,
                "on" if self._intent_head is not None else "off",
                persona_idx,
                signal_path,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "RealtimeLNNInferencer init failed: %s — running in disabled mode", exc
            )
            self._state_model = None
            self._enabled = False

    def _load_ckpt_artifacts(self, ckpt_path: Path) -> tuple[Any, str]:
        """加载 ckpt 并返回 state_model 与 sha8（供 init/reload 共用）。"""
        import hashlib

        ckpt_sha8 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:8]
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        factory = DefaultLNNFactory()
        sm = factory.create_state_model(ckpt["config"])
        sm.load_state_dict(ckpt["state_model"])
        sm.eval()

        if ckpt.get("online_adapter_schema") == "phase2_v1" and "online_tau_drift" in ckpt:
            base = getattr(sm, "personalization", None)
            if base is not None:
                online = OnlineLoRAAdapter(base_adapter=base, online_config=OnlineAdapterConfig())
                online.tau_drift.data.copy_(ckpt["online_tau_drift"])
                sm.personalization = online
                self._logger.info(
                    "  OnlineLoRAAdapter wrapped: tau_drift shape=%s",
                    list(ckpt["online_tau_drift"].shape),
                )
        return sm, ckpt_sha8

    def _load_ckpt(self, ckpt_path: Path) -> None:
        state_model, ckpt_sha8 = self._load_ckpt_artifacts(ckpt_path)
        self._state_model = state_model
        self._ckpt_sha8 = ckpt_sha8

    def reload_ckpt(self, new_ckpt_path: Path) -> None:
        """热重载 state ckpt，并在 30 帧内平滑过渡输出。"""
        old_state_model = self._state_model
        old_intent_head = self._intent_head
        new_state_model, new_sha8 = self._load_ckpt_artifacts(new_ckpt_path)

        self._state_model = new_state_model
        self._ckpt_sha8 = new_sha8
        self._enabled = True
        self._fail_count = 0

        if old_state_model is not None:
            self._transition_old_state_model = old_state_model
            self._transition_old_intent_head = old_intent_head
            self._transition_progress_frames = 0
            self._transition_total_frames = _TRANSITION_FRAMES
            self._in_transition = True
        else:
            self._transition_old_state_model = None
            self._transition_old_intent_head = None
            self._transition_progress_frames = 0
            self._in_transition = False

        self._logger.info(
            "RealtimeLNNInferencer reloaded ckpt=%s sha8=%s transition=%s",
            new_ckpt_path.name,
            new_sha8,
            self._in_transition,
        )

    def _load_intent_head_ckpt(self, ckpt_path: Path) -> None:
        """加载 intent head，失败时仅禁用 intent 输出，不影响 state 推理。"""
        import hashlib

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            state_dict = payload.get("intent_head") or payload.get("state_dict") or payload
        else:
            raise ValueError("intent head checkpoint must be a dict payload")

        hidden_dim = int(getattr(self._state_model, "hidden_dim", 64))
        head = IntentClassifierHead(hidden_dim=hidden_dim)
        head.load_state_dict(state_dict)
        head.eval()

        self._intent_head = head
        self._intent_head_ckpt_sha8 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:8]

    def _forward_last_hidden_for_model(self, state_model: Any, x: Tensor, persona_idx: Tensor) -> Tensor:
        """复用 state model 内部 CfC cell 计算最后时刻 hidden（给 intent head / H1）。"""
        sm = state_model
        features = sm.input_projection(x.float())
        persona_ids = persona_idx.to(device=features.device, dtype=torch.long)
        tau_bias = sm.personalization.get_tau_bias(persona_ids).to(features.device, features.dtype)
        gamma, beta = sm.personalization.get_film_params(persona_ids)
        gamma = gamma.to(features.device, features.dtype)
        beta = beta.to(features.device, features.dtype)
        timespans = sm._timespans_from_tau(tau_bias)

        batch, steps, _ = features.shape
        hidden = torch.zeros(batch, sm.hidden_dim, device=features.device, dtype=features.dtype)
        for step in range(steps):
            hidden, _ = sm.backbone.rnn_cell(features[:, step], hidden, timespans)
            hidden = sm._apply_film(hidden, gamma, beta)
        return hidden

    def _run_model_forward(
        self,
        state_model: Any,
        intent_head: IntentClassifierHead | None,
        x: Tensor,
        pid: Tensor,
    ) -> dict[str, Any]:
        """对给定模型执行一次前向，返回 state/intent + hidden。"""
        with torch.no_grad():
            logits = state_model(x, pid)
        last_logits = logits[0, -1, :]
        probs_tensor = torch.softmax(last_logits, dim=-1)
        probs = probs_tensor.tolist()
        sorted_probs = sorted(probs, reverse=True)
        top_idx = int(last_logits.argmax().item())

        last_hidden = self._forward_last_hidden_for_model(state_model, x, pid)
        hidden = last_hidden[0].detach().cpu().tolist()

        intent_probs: list[float] | None = None
        intent_pred: str | None = None
        intent_confidence: float | None = None
        if intent_head is not None:
            with torch.no_grad():
                intent_logits = intent_head(last_hidden)
                intent_tensor = torch.softmax(intent_logits[0], dim=-1)
            intent_probs = intent_tensor.tolist()
            intent_idx = int(intent_tensor.argmax().item())
            intent_pred = INTENT_LABELS[intent_idx]
            intent_confidence = float(intent_tensor[intent_idx].item())

        return {
            "state_probs": probs,
            "predicted_state": STATE_LABELS[top_idx],
            "top_prob": float(sorted_probs[0]),
            "margin": float(sorted_probs[0] - sorted_probs[1]),
            "entropy": float(-sum(p * math.log2(max(p, 1e-12)) for p in probs)),
            "hidden": hidden,
            "intent_probs": intent_probs,
            "intent_pred": intent_pred,
            "intent_confidence": intent_confidence,
        }

    def _append_recent_temporal(self, ts_ns: int, hidden_state: list[float], au_features: list[float]) -> None:
        self._recent_temporal.append((ts_ns, list(hidden_state), list(au_features)))

    def _extract_ts_ns(self, signal_row: dict[str, Any]) -> int:
        value = signal_row.get("ts_ns")
        if isinstance(value, int) and value > 0:
            return value
        ts = signal_row.get("ts")
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1_000_000_000)
        return int(datetime.now(tz=timezone.utc).timestamp() * 1_000_000_000)

    def _transition_alpha(self) -> float:
        if self._transition_total_frames <= 1:
            return 1.0
        alpha = self._transition_progress_frames / float(self._transition_total_frames - 1)
        return max(0.0, min(1.0, alpha))

    def _advance_transition(self) -> None:
        if not self._in_transition:
            return
        self._transition_progress_frames += 1
        if self._transition_progress_frames >= self._transition_total_frames:
            self._in_transition = False
            self._transition_old_state_model = None
            self._transition_old_intent_head = None
            self._transition_progress_frames = 0

    @staticmethod
    def _blend_vectors(old_values: list[float], new_values: list[float], alpha: float) -> list[float]:
        return [alpha * n + (1.0 - alpha) * o for o, n in zip(old_values, new_values)]

    def _blend_optional_vectors(
        self,
        old_values: list[float] | None,
        new_values: list[float] | None,
        alpha: float,
    ) -> list[float] | None:
        if old_values is None and new_values is None:
            return None
        if old_values is None:
            return new_values
        if new_values is None:
            return old_values
        return self._blend_vectors(old_values, new_values, alpha)

    def _state_from_probs(self, probs: list[float]) -> tuple[str, float, float, float]:
        sorted_probs = sorted(probs, reverse=True)
        top_idx = int(max(range(len(probs)), key=probs.__getitem__))
        entropy = -sum(p * math.log2(max(p, 1e-12)) for p in probs)
        return STATE_LABELS[top_idx], float(sorted_probs[0]), float(sorted_probs[0] - sorted_probs[1]), float(entropy)

    def get_recent_window(self, seconds: int = 10) -> list[dict[str, Any]]:
        """返回最近 N 秒 temporal 窗口（H1），包含 hidden + au。"""
        if seconds <= 0 or not self._recent_temporal:
            return []
        latest_ts = self._recent_temporal[-1][0]
        cutoff_ns = latest_ts - int(seconds * 1_000_000_000)
        window = [
            {"ts_ns": ts_ns, "h": hidden, "au": au}
            for ts_ns, hidden, au in self._recent_temporal
            if ts_ns >= cutoff_ns
        ]
        return window

    def reset(self) -> None:
        """清空滑窗（如 daemon session 切换）"""
        self._history.clear()
        self._recent_temporal.clear()
        self._transition_old_state_model = None
        self._transition_old_intent_head = None
        self._transition_progress_frames = 0
        self._in_transition = False

    def infer(self, signal_row: dict[str, Any]) -> InferenceResult:
        """单帧推理 + 维护滑窗。"""
        if not self._enabled or self._state_model is None:
            return InferenceResult(
                lnn_status="disabled",
                ckpt_sha8=self._ckpt_sha8,
                signal_path_used=self._signal_path,
            )

        try:
            frame_dict = _signals_row_to_frame_dict(signal_row)
            fused = self._fusion.fuse(frame_dict)
            if len(fused.vector) != 18:
                return InferenceResult(
                    lnn_status="error",
                    error=f"fused vector dim {len(fused.vector)} != 18",
                    ckpt_sha8=self._ckpt_sha8,
                    signal_path_used=self._signal_path,
                )
            self._history.append(fused.vector)
        except Exception as exc:  # noqa: BLE001
            self._record_fail()
            return InferenceResult(
                lnn_status="error",
                error=f"fusion failed: {type(exc).__name__}: {exc}",
                ckpt_sha8=self._ckpt_sha8,
                signal_path_used=self._signal_path,
            )

        ts_ns = self._extract_ts_ns(signal_row)
        try:
            x = torch.tensor(list(self._history), dtype=torch.float32).unsqueeze(0)
            pid = torch.tensor([self._persona_idx], dtype=torch.long)

            transition_ready = self._in_transition and self._transition_old_state_model is not None
            alpha = self._transition_alpha() if transition_ready else 1.0

            if len(self._history) < self._window:
                hidden_new = self._forward_last_hidden_for_model(self._state_model, x, pid)[0].tolist()
                if transition_ready:
                    hidden_old = self._forward_last_hidden_for_model(
                        self._transition_old_state_model,
                        x,
                        pid,
                    )[0].tolist()
                    hidden_for_store = self._blend_vectors(hidden_old, hidden_new, alpha)
                    self._advance_transition()
                else:
                    hidden_for_store = hidden_new
                self._append_recent_temporal(ts_ns, hidden_for_store, fused.vector)
                return InferenceResult(
                    lnn_status="warmup",
                    ckpt_sha8=self._ckpt_sha8,
                    signal_path_used=self._signal_path,
                )

            new_out = self._run_model_forward(self._state_model, self._intent_head, x, pid)
            if transition_ready:
                old_out = self._run_model_forward(
                    self._transition_old_state_model,
                    self._transition_old_intent_head,
                    x,
                    pid,
                )
                state_probs = self._blend_vectors(old_out["state_probs"], new_out["state_probs"], alpha)
                intent_probs = self._blend_optional_vectors(
                    old_out["intent_probs"],
                    new_out["intent_probs"],
                    alpha,
                )
                hidden_for_store = self._blend_vectors(old_out["hidden"], new_out["hidden"], alpha)
                predicted_state, top_prob, margin, entropy = self._state_from_probs(state_probs)
                self._advance_transition()
            else:
                state_probs = new_out["state_probs"]
                intent_probs = new_out["intent_probs"]
                hidden_for_store = new_out["hidden"]
                predicted_state = str(new_out["predicted_state"])
                top_prob = float(new_out["top_prob"])
                margin = float(new_out["margin"])
                entropy = float(new_out["entropy"])

            self._append_recent_temporal(ts_ns, hidden_for_store, fused.vector)

            # ADR-037: intent EMA 平滑（修"intent 1 秒翻 3 次"快变量欠阻尼）
            if intent_probs is not None:
                if self._intent_ema_probs is None or len(self._intent_ema_probs) != len(intent_probs):
                    self._intent_ema_probs = list(intent_probs)  # 冷启动
                else:
                    a = self._intent_ema_alpha
                    self._intent_ema_probs = [
                        a * new + (1.0 - a) * old
                        for new, old in zip(intent_probs, self._intent_ema_probs)
                    ]
                # 归一化（防 EMA 数值漂移）
                _s = sum(self._intent_ema_probs) or 1.0
                self._intent_ema_probs = [v / _s for v in self._intent_ema_probs]
                intent_probs = list(self._intent_ema_probs)

            # ADR-037: 缓存最新 frame 输出，供 predict_future step=0 对齐
            self._last_state_probs = list(state_probs)
            self._last_intent_probs = list(intent_probs) if intent_probs is not None else None

            intent_pred: str | None = None
            intent_confidence: float | None = None
            if intent_probs is not None:
                best_idx = int(max(range(len(intent_probs)), key=intent_probs.__getitem__))
                intent_pred = INTENT_LABELS[best_idx]
                intent_confidence = float(intent_probs[best_idx])

            self._fail_count = 0
            return InferenceResult(
                lnn_status="ok",
                predicted_state=predicted_state,
                state_probs=state_probs,
                top_prob=top_prob,
                margin=margin,
                entropy=entropy,
                ckpt_sha8=self._ckpt_sha8,
                signal_path_used=self._signal_path,
                intent_probs=intent_probs,
                intent_pred=intent_pred,
                intent_confidence=intent_confidence,
            )
        except Exception as exc:  # noqa: BLE001
            self._record_fail()
            return InferenceResult(
                lnn_status="error",
                error=f"forward failed: {type(exc).__name__}: {exc}",
                ckpt_sha8=self._ckpt_sha8,
                signal_path_used=self._signal_path,
            )

    def predict_future(self, steps: int) -> list[dict[str, Any]]:
        """Project future frames with LTC/CfC ODE + A+B 混合 input 外推。

        ADR-036 (2026-05-26)：放弃 ADR-033 的纯 persistence assumption（input 常数），
        改为 input trend 1阶导数外推 + decay 回 baseline 混合：
          input[t+1] = input[t] + α·velocity - β·(input[t] - baseline)
          velocity *= 0.8  # 衰减防爆炸

        - velocity = (history[-1] - history[-VELOCITY_WINDOW]) / (VELOCITY_WINDOW-1)
        - baseline = mean(history)
        - α=0.5（trend 保持系数），β=0.1（decay 拉回系数）

        修复 cc 实测发现：persistence 让 LTC 第 2 步就收敛到吸引子，10 步 state_probs
        变化 < 0.4%（gemini 3.5 钦定根因 1：ODE 自治系统在常数输入下趋向稳态）。
        现在 input 有动态变化 → hidden 也有动态变化 → 预测曲线不再是直线。
        """
        if not self._enabled or self._state_model is None:
            return [{"error": "disabled"}]
        if steps < 0:
            return [{"error": "steps must be >= 0"}]
        if not self._history:
            return [{"error": "no_recent_input"}]

        # A+B 混合外推系数（ADR-036 §决策）
        AUTOREG_ALPHA = 0.5         # trend 保持
        AUTOREG_BETA = 0.1          # decay 回 baseline
        AUTOREG_VELOCITY_DECAY = 0.8  # velocity 每步衰减
        VELOCITY_WINDOW = 5         # 用最近 5 帧算 trend

        try:
            sm = self._state_model
            history = [list(row) for row in self._history]
            last_input_values = list(history[-1])
            if len(last_input_values) != 18:
                return [{"error": f"last input dim {len(last_input_values)} != 18"}]

            # 算 baseline (mean of history) 和 velocity (recent trend)
            import numpy as _np
            hist_arr = _np.array(history, dtype=_np.float32)  # (T, 18)
            baseline_np = hist_arr.mean(axis=0)               # (18,)
            if len(history) >= VELOCITY_WINDOW:
                velocity_np = (hist_arr[-1] - hist_arr[-VELOCITY_WINDOW]) / (VELOCITY_WINDOW - 1)
            else:
                velocity_np = _np.zeros(18, dtype=_np.float32)
            projected_input_np = _np.array(last_input_values, dtype=_np.float32)

            x = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
            pid = torch.tensor([self._persona_idx], dtype=torch.long)

            with torch.no_grad():
                hidden = self._forward_last_hidden_for_model(sm, x, pid)

                persona_ids = pid.to(device=hidden.device, dtype=torch.long)
                tau_bias = sm.personalization.get_tau_bias(persona_ids).to(hidden.device, hidden.dtype)
                gamma, beta = sm.personalization.get_film_params(persona_ids)
                gamma = gamma.to(hidden.device, hidden.dtype)
                beta = beta.to(hidden.device, hidden.dtype)
                timespans = sm._timespans_from_tau(tau_bias)

                predictions: list[dict[str, Any]] = []
                # ADR-037: state 投影对齐 ——
                # step=0 用最新一帧 frame state (跟 UI 虚线左侧严格连续)
                # step=1..N-1 才用 LTC ODE + A+B 外推
                # 修复 gemini 钦定的"算法不对称"根因：虚线左侧 frame infer vs 右侧 LTC forward 主导 state 翻转
                ALIGN_STEP0_WITH_FRAME = (
                    self._last_state_probs is not None
                    and len(self._last_state_probs) == 8
                )

                for step in range(steps):
                    if step == 0 and ALIGN_STEP0_WITH_FRAME:
                        # 短路：用 frame infer 的 state_probs/intent_probs（保证 UI 虚线两侧连续）
                        state_probs = list(self._last_state_probs)
                        predicted_state, top_prob, margin, entropy = self._state_from_probs(state_probs)
                        intent_probs = list(self._last_intent_probs) if self._last_intent_probs else None
                        intent_pred = None
                        intent_confidence = None
                        if intent_probs is not None:
                            ii = int(max(range(len(intent_probs)), key=intent_probs.__getitem__))
                            intent_pred = INTENT_LABELS[ii]
                            intent_confidence = float(intent_probs[ii])
                        predictions.append({
                            "step": 0,
                            "predicted_state": predicted_state,
                            "state_probs": state_probs,
                            "top_prob": top_prob,
                            "margin": margin,
                            "entropy": entropy,
                            "intent_probs": intent_probs,
                            "intent_pred": intent_pred,
                            "intent_confidence": intent_confidence,
                            "aligned_with_frame": True,
                        })
                        continue

                    # step >= 1：走 A+B 外推 + LTC forward（原 ADR-036 逻辑）
                    projected_input_np = (
                        projected_input_np
                        + AUTOREG_ALPHA * velocity_np
                        - AUTOREG_BETA * (projected_input_np - baseline_np)
                    )
                    velocity_np = velocity_np * AUTOREG_VELOCITY_DECAY

                    last_input = torch.tensor(projected_input_np, dtype=torch.float32).unsqueeze(0)
                    last_input = last_input.to(device=hidden.device, dtype=hidden.dtype)
                    features = sm.input_projection(last_input.float()).to(hidden.device, hidden.dtype)
                    hidden, _ = sm.backbone.rnn_cell(features, hidden, timespans)
                    hidden = sm._apply_film(hidden, gamma, beta)

                    logits = sm.readout(hidden)[0]
                    state_tensor = torch.softmax(logits, dim=-1)
                    state_probs = [float(value) for value in state_tensor.detach().cpu().tolist()]
                    predicted_state, top_prob, margin, entropy = self._state_from_probs(state_probs)

                    intent_probs: list[float] | None = None
                    intent_pred: str | None = None
                    intent_confidence: float | None = None
                    if self._intent_head is not None:
                        intent_logits = self._intent_head(hidden)[0]
                        intent_tensor = torch.softmax(intent_logits, dim=-1)
                        intent_probs = [float(value) for value in intent_tensor.detach().cpu().tolist()]
                        intent_idx = int(max(range(len(intent_probs)), key=intent_probs.__getitem__))
                        intent_pred = INTENT_LABELS[intent_idx]
                        intent_confidence = float(intent_probs[intent_idx])

                    predictions.append(
                        {
                            "step": step,
                            "predicted_state": predicted_state,
                            "state_probs": state_probs,
                            "top_prob": top_prob,
                            "margin": margin,
                            "entropy": entropy,
                            "intent_probs": intent_probs,
                            "intent_pred": intent_pred,
                            "intent_confidence": intent_confidence,
                        }
                    )
            return predictions
        except Exception as exc:  # noqa: BLE001
            self._record_fail()
            return [{"error": f"predict_future failed: {type(exc).__name__}: {exc}"}]

    def _record_fail(self) -> None:
        """累计失败计数，超过阈值永久 disable（fail-soft 熔断）。"""
        self._fail_count += 1
        if self._fail_count >= self._fail_count_circuit_break:
            self._logger.warning(
                "RealtimeLNNInferencer: %d consecutive failures, disabling for this session",
                self._fail_count,
            )
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def ckpt_sha8(self) -> str | None:
        return self._ckpt_sha8

    def health_dict(self) -> dict[str, Any]:
        """暴露给 daemon /health endpoint。"""
        return {
            "enabled": self._enabled,
            "ckpt_sha8": self._ckpt_sha8,
            "intent_head_enabled": self._intent_head is not None,
            "intent_head_ckpt_sha8": self._intent_head_ckpt_sha8,
            "persona_idx": self._persona_idx,
            "signal_path": self._signal_path,
            "history_len": len(self._history),
            "fail_count": self._fail_count,
            "in_transition": self._in_transition,
            "recent_temporal_len": len(self._recent_temporal),
        }


__all__ = ["RealtimeLNNInferencer", "InferenceResult"]
