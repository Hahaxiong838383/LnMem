"""多模态融合层 — Day 6 特征扩展版（v1.0 / 2026-05-08）

把 `poc/src/capture/multimodal_capture.MultimodalCapture.capture_frame()` 输出的单帧
（face/audio/context 三段 dict）融合为下游 LNN / 记忆系统可消费的统一信号向量。

上游契约（poc/src/capture/multimodal_capture.py + face_signals.py 已定）::

    {"ts": "ISO8601", "frame_idx": int,
     "face": {"eye_closure": 0-1, "head_pitch_deg": float,
              "head_yaw_deg": float, "gaze_stability": 0-1,
              "smile_intensity": 0-1, "brow_furrow": 0-1,
              "cheek_raise": 0-1, "lip_corner_depress": 0-1,
              "jaw_drop": 0-1, "mouth_open_ratio": 0-1,
              "landmarks_n": int},
     "audio": {"transcript": str, "lang": str, "duration_s": float},
     "context": {"active_app": str, "app_switch_count_5min": int}}

下游契约（FusionFrame）::

    FusionFrame(
      ts: str,
      frame_idx: int,
      vector: tuple[float, ...],   # 长度=FEATURE_DIM=18，全部 [0,1] 归一化
      missing: tuple[str, ...],    # 哪些子模态缺失（face/audio/context）
      meta: dict,                  # 调试用：app/transcript_words/lang
    )

Day 6 扩展（v1.0）：
- FEATURE_DIM 12→18，新增 6 维 AU 表情特征
- 视觉 4 维 / 疲劳-注意 4 维 / **表情 6 维** / 语音 2 维 / 上下文 2 维
- 解决 Day 5 真人验证暴露的核心缺陷：12 维全认知、无表情 → 不能检测安静的积极

设计原则：
1. **缺模态不崩**：face/audio/context 任一缺失 → 用 PoC-spec §6 默认值填补，
   并在 ``missing`` 字段标注（下游 LNN 可见，便于 attention mask）。
2. **零外部依赖**：纯 Python + 标准库；测试在 sandbox 里也能跑。
3. **特征解释力优先**：18 维向量按认知 + 表情 + 语音 + 上下文四大类分组。
4. **阈值参数化**：所有阈值集中在 ``FusionConfig``，便于 Day 5+ LNN 训练时按
   主人差异化调参（per-user τ 偏置同源思路）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

FEATURE_DIM = 18

# ---------- 配置 ----------


@dataclass(frozen=True)
class FusionConfig:
    """阈值与归一化参数（主人差异化时按 per-user 覆盖）。"""

    # 视觉阈值（face 子模态）
    eye_closure_fatigue_threshold: float = 0.35  # > 视为眨眼/疲劳
    head_pitch_max_deg: float = 45.0  # 用于 / 归一到 [0,1]
    head_yaw_max_deg: float = 60.0
    # 语音阈值
    audio_voice_min_duration_s: float = 0.20
    audio_long_utterance_s: float = 4.0
    # 上下文阈值
    context_high_switch_count: int = 8  # 5min 内切换 > 8 次视为分心高
    # 缺失模态默认值（中性 0.5，避免 0/1 极端污染 LNN）
    missing_default: float = 0.5


DEFAULT_CONFIG = FusionConfig()


# ---------- 输出帧 ----------


@dataclass(frozen=True)
class FusionFrame:
    """融合后的单帧信号（immutable，符合 coding-style.md immutability）。"""

    ts: str
    frame_idx: int
    vector: tuple[float, ...]
    missing: tuple[str, ...]
    meta: dict[str, Any] = field(default_factory=dict)


# ---------- 子模态特征提取 ----------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _face_features(face: dict[str, Any] | None, cfg: FusionConfig) -> tuple[float, ...]:
    """4 维：eye_closure / pitch_norm / yaw_norm / gaze_stability。

    缺失 → 全部填 cfg.missing_default。
    """
    if not face:
        return (cfg.missing_default,) * 4
    eye = _clamp(float(face.get("eye_closure", cfg.missing_default)))
    pitch = _clamp(abs(float(face.get("head_pitch_deg", 0.0))) / cfg.head_pitch_max_deg)
    yaw = _clamp(abs(float(face.get("head_yaw_deg", 0.0))) / cfg.head_yaw_max_deg)
    gaze = _clamp(float(face.get("gaze_stability", cfg.missing_default)))
    return (eye, pitch, yaw, gaze)


def _fatigue_attention_features(
    face: dict[str, Any] | None, cfg: FusionConfig
) -> tuple[float, ...]:
    """4 维派生信号：fatigue_score / attention_score / pose_steadiness / blink_burst。

    PoC 阶段用 face 子模态线性组合做近似；Day 8+ 真实采集后换 LNN 学习的非线性权重。
    """
    if not face:
        return (cfg.missing_default,) * 4
    eye = _clamp(float(face.get("eye_closure", cfg.missing_default)))
    gaze = _clamp(float(face.get("gaze_stability", cfg.missing_default)))
    pitch_abs = abs(float(face.get("head_pitch_deg", 0.0))) / cfg.head_pitch_max_deg
    yaw_abs = abs(float(face.get("head_yaw_deg", 0.0))) / cfg.head_yaw_max_deg

    fatigue = _clamp(0.6 * eye + 0.4 * (1.0 - gaze))
    attention = _clamp(0.7 * gaze + 0.3 * (1.0 - max(pitch_abs, yaw_abs)))
    pose_steadiness = _clamp(1.0 - 0.5 * (pitch_abs + yaw_abs))
    blink_burst = 1.0 if eye > cfg.eye_closure_fatigue_threshold else 0.0
    return (fatigue, attention, pose_steadiness, blink_burst)


def _audio_features(audio: dict[str, Any] | None, cfg: FusionConfig) -> tuple[float, ...]:
    """2 维：voice_present / utterance_density。

    voice_present: 0/1 — 是否有有效语音帧
    utterance_density: 0-1 — 当前 transcript 长度 vs 长发言阈值
    """
    if not audio:
        return (cfg.missing_default, cfg.missing_default)
    duration = float(audio.get("duration_s", 0.0))
    transcript = str(audio.get("transcript", "")).strip()
    voice_present = 1.0 if duration >= cfg.audio_voice_min_duration_s and transcript else 0.0
    density = _clamp(duration / cfg.audio_long_utterance_s)
    return (voice_present, density)


def _context_features(
    context: dict[str, Any] | None, cfg: FusionConfig
) -> tuple[float, ...]:
    """2 维：context_known / focus_stability。"""
    if not context:
        return (cfg.missing_default, cfg.missing_default)
    app = str(context.get("active_app", "")).strip()
    switches = int(context.get("app_switch_count_5min", 0))
    context_known = 1.0 if app else 0.0
    # switches 越多 → focus_stability 越低
    focus = _clamp(1.0 - switches / max(cfg.context_high_switch_count, 1))
    return (context_known, focus)


def _expression_features(
    face: dict[str, Any] | None, cfg: FusionConfig
) -> tuple[float, ...]:
    """6 维 AU 表情特征（Day 6 新增）。

    smile_intensity / brow_furrow / cheek_raise /
    lip_corner_depress / jaw_drop / mouth_open_ratio

    来源：face_signals.py capture_signals() 新增的 6 个字段，
    或合成数据中 MultimodalHint 的 6 个表情字段。
    缺失 → 全部填 cfg.missing_default。
    """
    if not face:
        return (cfg.missing_default,) * 6
    smile = _clamp(float(face.get("smile_intensity", 0.0)))
    brow = _clamp(float(face.get("brow_furrow", 0.0)))
    cheek = _clamp(float(face.get("cheek_raise", 0.0)))
    lip_down = _clamp(float(face.get("lip_corner_depress", 0.0)))
    jaw = _clamp(float(face.get("jaw_drop", 0.0)))
    mouth = _clamp(float(face.get("mouth_open_ratio", 0.0)))
    return (smile, brow, cheek, lip_down, jaw, mouth)


# ---------- 融合主入口 ----------


class MultimodalFusion:
    """把 capture_frame() 输出融合为 FusionFrame。

    用法::

        fusion = MultimodalFusion()
        frame_dict = capture.capture_frame()  # 来自 capture/
        fused = fusion.fuse(frame_dict)
        assert len(fused.vector) == FEATURE_DIM
    """

    def __init__(self, config: FusionConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def fuse(self, frame: dict[str, Any]) -> FusionFrame:
        """单帧融合。input frame 必须含 ts + frame_idx，其余子模态可缺。"""
        cfg = self.config
        missing: list[str] = []

        face = frame.get("face")
        audio = frame.get("audio")
        context = frame.get("context")
        if not face:
            missing.append("face")
        if not audio:
            missing.append("audio")
        if not context:
            missing.append("context")

        face_feats = _face_features(face, cfg)          # 4 dim
        derived_feats = _fatigue_attention_features(face, cfg)  # 4 dim
        expr_feats = _expression_features(face, cfg)   # 6 dim (Day 6)
        audio_feats = _audio_features(audio, cfg)      # 2 dim
        context_feats = _context_features(context, cfg)  # 2 dim

        vector = face_feats + derived_feats + expr_feats + audio_feats + context_feats
        assert len(vector) == FEATURE_DIM, f"FEATURE_DIM mismatch: {len(vector)} != {FEATURE_DIM}"

        meta = {
            "active_app": (context or {}).get("active_app", ""),
            "transcript_words": len(str((audio or {}).get("transcript", "")).split()),
            "lang": (audio or {}).get("lang", ""),
        }

        return FusionFrame(
            ts=str(frame.get("ts", "")),
            frame_idx=int(frame.get("frame_idx", 0)),
            vector=vector,
            missing=tuple(missing),
            meta=meta,
        )

    def fuse_batch(
        self, frames: Iterable[dict[str, Any]]
    ) -> list[FusionFrame]:
        """批量融合（5min 滑窗用）。"""
        return [self.fuse(f) for f in frames]


__all__ = [
    "FEATURE_DIM",
    "FusionConfig",
    "DEFAULT_CONFIG",
    "FusionFrame",
    "MultimodalFusion",
]
