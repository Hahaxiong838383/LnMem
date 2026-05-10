"""poc/src/lnn/signals_loader.py — 解析 ~/.lnmem/data/signals_*.jsonl → 18 维 fusion vector。

桥接 daemon V0.2 实采的 signals jsonl 格式与 MultimodalFusion 18 维输出。

输入 jsonl 行字段（daemon V0.2 写入）
-------------------------------------
- ts, frame_id, face_detected, device, raw_video_offset_ms
- raw_au + calibrated_au: 20 AU 各
- expression: 6 维 (smile_intensity / brow_furrow / cheek_raise /
                   lip_corner_depress / jaw_drop / mouth_open_ratio)
- head_pose: pitch_deg / roll_deg / yaw_deg
- session_id (V0.2)
- vlm_label (可选): au_description / valence / arousal / confidence / model_used

输出 18 维 fusion vector（fusion.MultimodalFusion.fuse 公契约）
---------------------------------------------------------------
- 4 face: eye_closure / pitch_norm / yaw_norm / gaze_stability
- 4 derived: fatigue / attention / pose_steadiness / blink_burst
- 6 expression: smile / brow / cheek / lip_down / jaw / mouth_open
- 2 audio: voice_present / utterance_density (missing → 0.5)
- 2 context: context_known / focus_stability (missing → 0.5)

注意事项
--------
- daemon V0.2 不采 audio + context → 这两段走 fusion missing_default=0.5
- eye_closure 来自 calibrated_au["AU43"]（眨眼/眼闭单元）
- gaze_stability 当前 daemon 不采 → 走 missing_default=0.5
- face_detected=False 的行也保留（fusion 会用 missing_default 处理）
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from poc.src.fusion import FEATURE_DIM, MultimodalFusion
from poc.src.lnn.online_contracts import FeedbackDataError

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalsFrame:
    """单帧 signals → fusion 后的 18 维向量（immutable）。"""

    ts: str
    frame_id: int
    vector: tuple[float, ...]
    face_detected: bool
    expression: dict[str, float]
    session_id: str | None = None  # F1-01: V0.2 起 daemon 写入；老格式 None


def _signals_row_to_frame_dict(row: dict) -> dict:
    """把 signals jsonl 单行 → fusion.fuse() 期望的 frame dict（face/audio/context）.

    F2-04 (codex)：signals 字段缺失时记录 warning（不抛错以保持 fail-soft），
    让上层数据质量监控可见。PoC 阶段缺失语义仍以 fusion.missing_default=0.5 + 0.0
    填充（M0 W2 引入显式 missing mask）。

    Returns
    -------
    frame : dict
        {ts, frame_idx, face: {...18-dim raw fields...}, audio: None, context: None}
    """
    expression_raw = row.get("expression")
    head_pose_raw = row.get("head_pose")
    calibrated_au_raw = row.get("calibrated_au")
    frame_id = row.get("frame_id", "?")

    # F2-04: 整段缺失时 warning（说明 daemon V0.1 老格式或解析失败）
    if expression_raw is None:
        warnings.warn(
            f"signals_loader: frame_id={frame_id} 'expression' field missing → "
            f"6 维表情走 0.0（PoC 阶段；M0 W2 改 missing mask）",
            RuntimeWarning,
        )
    if head_pose_raw is None:
        warnings.warn(
            f"signals_loader: frame_id={frame_id} 'head_pose' field missing → "
            f"pitch/yaw 走 0.0（中性姿态语义错位风险）",
            RuntimeWarning,
        )
    if calibrated_au_raw is None:
        warnings.warn(
            f"signals_loader: frame_id={frame_id} 'calibrated_au' field missing → "
            f"AU43 (eye_closure) 走 0.5（半闭语义错位风险）",
            RuntimeWarning,
        )

    expression = expression_raw or {}
    head_pose = head_pose_raw or {}
    calibrated_au = calibrated_au_raw or {}

    # eye_closure 从 AU43（眼闭单元）取
    eye_closure = float(calibrated_au.get("AU43", 0.5))
    pitch = float(head_pose.get("pitch_deg", 0.0))
    yaw = float(head_pose.get("yaw_deg", 0.0))

    face = {
        # face_feats 4 dim（pitch / yaw 用度数原始值，fusion 内部归一化）
        "eye_closure": eye_closure,
        "head_pitch_deg": pitch,
        "head_yaw_deg": yaw,
        "gaze_stability": 0.5,  # daemon V0.2 不采 → missing_default
        # expression 6 dim 直接拷
        "smile_intensity": float(expression.get("smile_intensity", 0.0)),
        "brow_furrow": float(expression.get("brow_furrow", 0.0)),
        "cheek_raise": float(expression.get("cheek_raise", 0.0)),
        "lip_corner_depress": float(expression.get("lip_corner_depress", 0.0)),
        "jaw_drop": float(expression.get("jaw_drop", 0.0)),
        "mouth_open_ratio": float(expression.get("mouth_open_ratio", 0.0)),
    }
    return {
        "ts": row.get("ts", ""),
        "frame_idx": int(row.get("frame_id", 0)),
        "face": face,
        "audio": None,  # daemon V0.2 不采
        "context": None,  # daemon V0.2 不采
    }


def parse_signals_row(row: object) -> SignalsFrame | None:
    """parse 单行 signals jsonl → SignalsFrame；fail-soft 返 None.

    F3-01 风格：raw 必须 dict，否则 warning + skip。
    """
    if not isinstance(row, dict):
        warnings.warn(
            f"SignalsLoader: skip non-dict row (type={type(row).__name__})",
            RuntimeWarning,
        )
        return None
    try:
        frame_dict = _signals_row_to_frame_dict(row)
        fusion = MultimodalFusion()
        fused = fusion.fuse(frame_dict)
        if len(fused.vector) != FEATURE_DIM:
            warnings.warn(
                f"SignalsLoader: fused vector dim {len(fused.vector)} != {FEATURE_DIM}",
                RuntimeWarning,
            )
            return None
        return SignalsFrame(
            ts=str(row.get("ts", "")),
            frame_id=int(row.get("frame_id", 0)),
            vector=tuple(fused.vector),
            face_detected=bool(row.get("face_detected", True)),
            expression=dict(row.get("expression") or {}),
            session_id=row.get("session_id"),  # F1-01: V0.2 daemon 写入；老格式 None
        )
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        warnings.warn(
            f"SignalsLoader: skip malformed row ({type(e).__name__}: {e})",
            RuntimeWarning,
        )
        return None


def load_signals(
    signals_dir: Path,
    *,
    date_prefix: str | None = None,
) -> tuple[SignalsFrame, ...]:
    """Load all signals from `signals_dir/signals_*.jsonl`.

    Parameters
    ----------
    signals_dir : Path
        e.g. ``~/.lnmem/data/``
    date_prefix : str | None
        过滤前缀（e.g. ``2026-05-10``）

    Returns
    -------
    frames : tuple[SignalsFrame, ...]
        按文件名字典序 → 行序排列；frozen tuple
    """
    signals_dir = Path(signals_dir)
    if not signals_dir.exists():
        raise FeedbackDataError(f"signals_dir does not exist: {signals_dir}")

    pattern = "signals_*.jsonl"
    if date_prefix:
        pattern = f"signals_{date_prefix}*.jsonl"

    frames: list[SignalsFrame] = []
    for path in sorted(signals_dir.glob(pattern)):
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as e:
                        warnings.warn(
                            f"SignalsLoader: bad jsonl in {path.name}: {e}",
                            RuntimeWarning,
                        )
                        continue
                    sf = parse_signals_row(raw)
                    if sf is not None:
                        frames.append(sf)
        except OSError as e:
            raise FeedbackDataError(f"cannot read {path}: {e}") from e

    return tuple(frames)


def build_frame_id_index(
    frames: tuple[SignalsFrame, ...],
) -> dict[int, int]:
    """[deprecated F1-01] frame_id → list-index 反查表。

    daemon V0.2 frame_id 在单天内单调递增（每天 daemon 重启会重置 0）；
    跨天时同一 frame_id 可能出现两次 → silent overwrite，**会导致跨天反馈静默
    错配**（codex F1-01）。

    保留此函数仅用于向后兼容（无 session_id 的老测试）；新代码必须用
    `build_session_frame_index`。

    Returns
    -------
    index : dict[int, int]
        frame_id → frames 元组下标（重复 frame_id 取后到）
    """
    out: dict[int, int] = {}
    duplicates = 0
    for i, f in enumerate(frames):
        if f.frame_id in out:
            duplicates += 1
        out[f.frame_id] = i
    if duplicates > 0:
        warnings.warn(
            f"build_frame_id_index: {duplicates} duplicate frame_id silently overwritten "
            f"(F1-01 risk); use build_session_frame_index for safety",
            RuntimeWarning,
        )
    return out


def build_session_frame_index(
    frames: tuple[SignalsFrame, ...],
) -> dict[tuple[str | None, int], int]:
    """[F1-01 修复] (session_id, frame_id) → list-index 复合键反查表。

    每个 daemon session（重启即新 session_id）独立索引，跨天 frame_id 重复不再
    silent overwrite，确保 feedback × signals join 语义正确。

    向后兼容
    --------
    SignalsFrame 当前不含 session_id 字段（M0 W2 待加）；本函数从 frames 元数据
    无法直接拿，调用方需要外部传入。临时方案：对 SignalsFrame 增加可选 session_id
    属性，**但不破坏向后兼容**——本函数允许 session_id=None 的 frame，等价于
    单 session 模式。

    Returns
    -------
    index : dict[(session_id, frame_id), int]
    """
    out: dict[tuple[str | None, int], int] = {}
    duplicates = 0
    for i, f in enumerate(frames):
        sid = getattr(f, "session_id", None)
        key = (sid, f.frame_id)
        if key in out:
            duplicates += 1
        out[key] = i
    if duplicates > 0:
        warnings.warn(
            f"build_session_frame_index: {duplicates} (session_id, frame_id) duplicates "
            f"overwritten — same session same frame_id is unexpected",
            RuntimeWarning,
        )
    return out


def signals_to_tensor(
    frames: tuple[SignalsFrame, ...],
) -> torch.Tensor:
    """tuple[SignalsFrame] → Tensor (n_frames, 18) float32.

    Raises
    ------
    FeedbackDataError
        frames 为空
    """
    if not frames:
        raise FeedbackDataError("cannot build tensor from empty frames")
    rows = [list(f.vector) for f in frames]
    return torch.tensor(rows, dtype=torch.float32)


__all__ = [
    "SignalsFrame",
    "parse_signals_row",
    "load_signals",
    "build_frame_id_index",
    "build_session_frame_index",
    "signals_to_tensor",
]
