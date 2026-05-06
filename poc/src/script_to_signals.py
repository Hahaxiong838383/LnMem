"""Day 3 Lane C — 剧本 → 12-dim FusionFrame 时间序列转换器（v0.1 / 2026-05-06）。

输入：``poc.src.data.event_schema.Script``（含 ScriptEvent 序列 + MultimodalHint）。
输出：JSONL 信号流，每行一帧，向量长度 = ``poc.src.fusion.FEATURE_DIM`` (12)。

转换流程：
1. 遍历 Script.events
2. 每个 event 按 ``frame_rate_hz``（默认 1Hz）展开为 ``ceil(duration_s)`` 帧
3. 每帧用 event.multimodal_hint 构造 fusion 输入 dict → ``MultimodalFusion.fuse()``
   → 12-dim FusionFrame
4. 叠加高斯扰动 (mu=0, sigma=noise_sigma) + clamp 到 [0,1]，保证可重现 (seed)
5. 落盘 JSONL：``{ts, frame_idx, vector, missing, ground_truth_state, persona_id,
   source_event_id, day_index, activity}``

下游消费方：
- ``poc/src/lnn/`` (Day 5-6) 的 LNN 训练 — vector + ground_truth_state
- ``poc/src/dataset_report.py`` (Day 3 Lane D) — 多脚本聚合统计

设计原则：
- frozen dataclass + tuple/dict immutability（coding-style.md）
- 零外部依赖（无 numpy）— PoC 阶段用 ``random.gauss``
- 与 ``fusion.py`` 协议一致（FEATURE_DIM=12, missing 标注）
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

from .data.event_schema import Script, ScriptEvent
from .fusion import FEATURE_DIM, FusionConfig, MultimodalFusion


# ---------- 输出帧 ----------


@dataclass(frozen=True)
class SignalFrame:
    """单帧信号（带 ground truth + 溯源信息）。"""

    ts: str
    frame_idx: int                 # 全 Script 内的全局帧序号
    vector: tuple[float, ...]      # 长度 = FEATURE_DIM (12)
    missing: tuple[str, ...]       # 由 fusion.py 标注，PoC 通常为空
    ground_truth_state: str        # 来自 source event.state.value
    activity: str                  # 来自 source event.activity.value
    persona_id: str
    source_event_id: str
    day_index: int

    def to_jsonl_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "frame_idx": self.frame_idx,
            "vector": list(self.vector),
            "missing": list(self.missing),
            "ground_truth_state": self.ground_truth_state,
            "activity": self.activity,
            "persona_id": self.persona_id,
            "source_event_id": self.source_event_id,
            "day_index": self.day_index,
        }


# ---------- 内部：MultimodalHint → fusion 输入 dict ----------


def _hint_to_fusion_input(event: ScriptEvent, ts_iso: str, frame_idx: int) -> dict[str, Any]:
    """把 ScriptEvent.multimodal_hint 还原成 fusion.fuse() 需要的 frame dict。

    映射规则（与 fusion.py 子模态 schema 对齐）：
    - hint.eye_closure → face.eye_closure
    - hint.head_pitch_norm × 45 → face.head_pitch_deg（FusionConfig.head_pitch_max_deg）
    - hint.head_yaw_norm × 60 → face.head_yaw_deg
    - hint.gaze_stability → face.gaze_stability
    - landmarks_n 默认 478（Day 1 实测值）
    - hint.voice_present 1.0 → audio 段（duration ≥ 0.2s + 占位 transcript）
    - hint.utterance_density × 4.0 → audio.duration_s (FusionConfig.audio_long_utterance_s)
    - hint.active_app → context.active_app
    - hint.app_switch_count_5min → context.app_switch_count_5min
    """
    hint = event.multimodal_hint

    face: dict[str, Any] = {
        "eye_closure": hint.eye_closure,
        "head_pitch_deg": hint.head_pitch_norm * FusionConfig().head_pitch_max_deg,
        "head_yaw_deg": hint.head_yaw_norm * FusionConfig().head_yaw_max_deg,
        "gaze_stability": hint.gaze_stability,
        "landmarks_n": 478,
    }

    if hint.voice_present > 0.0:
        audio: dict[str, Any] = {
            "transcript": "[hint:voice_present]",
            "lang": "zh",
            "duration_s": hint.utterance_density * FusionConfig().audio_long_utterance_s + 0.3,
        }
    else:
        audio = {"transcript": "", "lang": "zh", "duration_s": 0.0}

    context: dict[str, Any] = {
        "active_app": hint.active_app,
        "app_switch_count_5min": hint.app_switch_count_5min,
    }

    return {
        "ts": ts_iso,
        "frame_idx": frame_idx,
        "face": face,
        "audio": audio,
        "context": context,
    }


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _add_noise(
    vector: tuple[float, ...], rng: random.Random, sigma: float
) -> tuple[float, ...]:
    """高斯噪声 + clamp 到 [0,1]。sigma=0 时直接返回原值。"""
    if sigma <= 0:
        return vector
    return tuple(_clamp_unit(v + rng.gauss(0.0, sigma)) for v in vector)


# ---------- 主入口 ----------


def script_to_signals(
    script: Script,
    frame_rate_hz: float = 1.0,
    noise_sigma: float = 0.05,
    seed: int = 0,
) -> Iterator[SignalFrame]:
    """生成器：把 Script 展开为 SignalFrame 序列。

    参数：
        script: 来自 generate_scripts.py 或 Script.from_json
        frame_rate_hz: 帧率（默认 1Hz，对应 fusion.py PoC 取样率）
        noise_sigma: 高斯扰动标准差（建议 0.03-0.10）
        seed: 随机种子（同 seed → 可重现序列）
    """
    if frame_rate_hz <= 0:
        raise ValueError(f"frame_rate_hz must be > 0, got {frame_rate_hz}")
    if noise_sigma < 0:
        raise ValueError(f"noise_sigma must be >= 0, got {noise_sigma}")

    fusion = MultimodalFusion()
    rng = random.Random(seed)
    period_s = 1.0 / frame_rate_hz
    global_idx = 0

    for event in script.events:
        n_frames = max(1, math.ceil(event.duration_s * frame_rate_hz))
        try:
            event_start = datetime.fromisoformat(event.ts)
        except ValueError:
            continue  # 时间戳坏掉的 event 跳过（validator 应已报错）

        for i in range(n_frames):
            frame_ts = event_start + timedelta(seconds=i * period_s)
            ts_iso = frame_ts.isoformat()
            fusion_input = _hint_to_fusion_input(event, ts_iso, global_idx)
            fused = fusion.fuse(fusion_input)
            noisy_vec = _add_noise(fused.vector, rng, noise_sigma)

            yield SignalFrame(
                ts=ts_iso,
                frame_idx=global_idx,
                vector=noisy_vec,
                missing=fused.missing,
                ground_truth_state=event.state.value,
                activity=event.activity.value,
                persona_id=event.persona_id,
                source_event_id=event.event_id,
                day_index=event.day_index,
            )
            global_idx += 1


def write_signals_jsonl(
    frames: Iterable[SignalFrame], out_path: Path
) -> int:
    """落盘 JSONL，返回写入条目数。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for frame in frames:
            fh.write(json.dumps(frame.to_jsonl_dict(), ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def script_file_to_signals_file(
    script_path: Path,
    out_path: Path,
    frame_rate_hz: float = 1.0,
    noise_sigma: float = 0.05,
    seed: int = 0,
) -> int:
    """便捷端到端入口：JSON 剧本 → JSONL 信号流。"""
    script = Script.from_json(script_path.read_text(encoding="utf-8"))
    return write_signals_jsonl(
        script_to_signals(
            script, frame_rate_hz=frame_rate_hz, noise_sigma=noise_sigma, seed=seed
        ),
        out_path,
    )


__all__ = [
    "SignalFrame",
    "script_to_signals",
    "write_signals_jsonl",
    "script_file_to_signals_file",
    "FEATURE_DIM",
]
