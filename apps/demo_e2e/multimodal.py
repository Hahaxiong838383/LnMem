"""Phase α 多模态 snapshot 模块（ADR-016 base v3 input signal）

设计原则（LNN-FIRST P-1）：
- 视觉**不是**当 LLM 分类器
- AU/V-A 是 ADR-016 base v3 多模态融合的输入信号
- 当前 demo 用 Russell 几何映射 9-state 让川哥可视（base v1 真推理留接口）

两种模式：
- mock 模式（默认）：图像 hash → 确定性 V-A → 9-state（无依赖、毫秒级、测试稳定）
- pyfeat 模式（开关）：py-feat 提 AU → 简化 V-A 投影（首次模型下载慢，gemini 评审 #2）

guard:
- ADR-010 假脸：人脸面积阈值
- ADR-011 AU NaN 容错
"""
from __future__ import annotations

import hashlib
import math
from typing import Literal, TypedDict

# 20 AU 维度（与 scripts/extract_au_to_parquet.py 对齐）
AU_NAMES_20 = (
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43",
)

# Russell 9-state 标签（与 sweep_va_to_state.py 对齐：1 neutral + 8 sectors × 45°）
STATE_LABELS_9 = (
    "neutral",       # 0：圆心 r < r_neutral
    "happy",         # 1：sector 0°-45° （高 V 中 A）
    "excited",       # 2：sector 45°-90° （中高 V 高 A）
    "alert",         # 3：sector 90°-135°（中 V 高 A 偏负）
    "tense",         # 4：sector 135°-180°（低 V 高 A）
    "sad",           # 5：sector 180°-225°（低 V 中 A）
    "depressed",     # 6：sector 225°-270°（低 V 低 A）
    "calm",          # 7：sector 270°-315°（中 V 低 A）
    "relaxed",       # 8：sector 315°-360°（高 V 低 A）
)

# 中文映射（UI 展示）
STATE_LABELS_ZH = {
    "neutral": "平静",
    "happy": "开心",
    "excited": "兴奋",
    "alert": "警觉",
    "tense": "紧张",
    "sad": "悲伤",
    "depressed": "沮丧",
    "calm": "安宁",
    "relaxed": "放松",
}


class SnapshotResult(TypedDict):
    """多模态 snapshot 端到端结果。

    LNN-FIRST 语义：au_vector + v_a 是 base v3 input signal；
    state_label 是几何投影（不是 LLM 分类）。
    """
    au_vector: dict[str, float]      # 20 AU → [0, 5]
    face_detected: bool
    v_a: tuple[float, float]          # Russell (V, A) ∈ [-1, 1]
    state_label: str                  # STATE_LABELS_9 之一
    state_label_zh: str
    confidence: float                 # [0, 1]
    source: str                       # "mock" | "pyfeat+base_v1" | "error"
    error: str | None


# ──────────────────── Russell 几何映射（与 sweep_va_to_state 同构） ────────────────────

def map_va_to_state_label(
    v: float, a: float,
    r_neutral: float = 0.35,
    theta_offset_deg: float = 0.0,
) -> str:
    """单点 V-A → 9-state label（Russell 圆形几何）。"""
    radius = math.sqrt(v * v + a * a)
    if radius < r_neutral:
        return STATE_LABELS_9[0]  # neutral
    theta = (math.degrees(math.atan2(a, v)) - theta_offset_deg) % 360.0
    sector = int(theta // 45.0) + 1
    sector = max(1, min(8, sector))
    return STATE_LABELS_9[sector]


# ──────────────────── Mock 模式（默认，毫秒级） ────────────────────

def _deterministic_va_from_bytes(image_bytes: bytes) -> tuple[float, float]:
    """图像 hash → 确定性 V-A（测试稳定 / demo 可复现）。"""
    if not image_bytes:
        return 0.0, 0.0
    h = hashlib.sha256(image_bytes).digest()
    # 取前 4 字节 → V，后 4 字节 → A，映射到 [-0.8, 0.8]（避开边缘）
    v_raw = int.from_bytes(h[:4], "big") / (2**32 - 1)  # [0, 1]
    a_raw = int.from_bytes(h[4:8], "big") / (2**32 - 1)
    v = (v_raw - 0.5) * 1.6  # [-0.8, 0.8]
    a = (a_raw - 0.5) * 1.6
    return round(v, 3), round(a, 3)


def _mock_au_vector(v: float, a: float) -> dict[str, float]:
    """从 V-A 反推 mock AU vector（让 UI 展示有意义、可解释）。

    简化规则（不是真模型，仅 demo 友好）：
    - V > 0 → AU06 (cheek raiser) + AU12 (lip corner) 强（笑）
    - V < 0 → AU04 (brow lowerer) + AU15 (lip corner depressor) 强（皱眉）
    - A > 0 → AU01 + AU02 (inner/outer brow raiser) 强（警觉）
    - A < 0 → AU43 (eyes closed) 强（疲惫）
    """
    au = {name: 0.0 for name in AU_NAMES_20}
    # Valence influence
    if v > 0:
        au["AU06"] = round(min(5.0, abs(v) * 4.0), 2)
        au["AU12"] = round(min(5.0, abs(v) * 4.5), 2)
    else:
        au["AU04"] = round(min(5.0, abs(v) * 4.0), 2)
        au["AU15"] = round(min(5.0, abs(v) * 3.5), 2)
    # Arousal influence
    if a > 0:
        au["AU01"] = round(min(5.0, abs(a) * 3.0), 2)
        au["AU02"] = round(min(5.0, abs(a) * 2.8), 2)
        au["AU05"] = round(min(5.0, abs(a) * 3.5), 2)
    else:
        au["AU43"] = round(min(5.0, abs(a) * 4.0), 2)
    return au


# ──────────────────── 真 py-feat 推理（可选开关，Phase β 启用） ────────────────────

def extract_au_from_image(image_bytes: bytes) -> dict[str, float] | None:
    """py-feat 提 AU。失败返回 None（不抛异常，由上层兜底）。

    注意：py-feat 首次推理需下载模型（~150MB），CPU 一帧 2-5s。
    Phase α 默认走 Mock 路径，本函数留作 Phase β/γ 启用。
    """
    if not image_bytes:
        return None
    try:
        # 延迟 import 避免 streamlit 启动阻塞
        from feat import Detector  # type: ignore
        import io
        from PIL import Image
        import numpy as np

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_arr = np.array(img)
        detector = Detector(device="cpu")
        out = detector.detect_image(img_arr)
        if out is None or out.empty:
            return None
        # py-feat 列名是 AU01_r / AU02_r 等
        row = out.iloc[0]
        au = {}
        for name in AU_NAMES_20:
            col = f"{name}_r"
            v = float(row.get(col, 0.0))
            if math.isnan(v):
                v = 0.0  # ADR-011 NaN 容错
            au[name] = round(max(0.0, min(5.0, v)), 2)
        return au
    except Exception:
        return None


# ──────────────────── 端到端入口 ────────────────────

def snapshot_to_state(
    image_bytes: bytes | None,
    mode: Literal["mock", "pyfeat"] = "mock",
) -> SnapshotResult:
    """端到端：bytes → AU → V-A → state。

    Args:
        image_bytes: 摄像头抓帧 bytes（None / 空 → no face）
        mode: "mock" 毫秒级确定性 / "pyfeat" 真推理（慢）

    Returns:
        SnapshotResult dict，永不抛异常（失败走 error 字段）。
    """
    if not image_bytes:
        return SnapshotResult(
            au_vector={name: 0.0 for name in AU_NAMES_20},
            face_detected=False,
            v_a=(0.0, 0.0),
            state_label="neutral",
            state_label_zh=STATE_LABELS_ZH["neutral"],
            confidence=0.0,
            source="error",
            error="empty image bytes",
        )

    if mode == "pyfeat":
        au = extract_au_from_image(image_bytes)
        if au is None:
            # 降级到 mock 兜底
            return _mock_snapshot(image_bytes, fallback_reason="pyfeat extract failed")
        # 真 AU → 简化 V-A 投影（Phase α 占位算法；Phase γ 由 base v3 学）
        v_pos = (au.get("AU06", 0.0) + au.get("AU12", 0.0)) / 10.0
        v_neg = (au.get("AU04", 0.0) + au.get("AU15", 0.0)) / 10.0
        a_pos = (au.get("AU01", 0.0) + au.get("AU02", 0.0) + au.get("AU05", 0.0)) / 15.0
        a_neg = au.get("AU43", 0.0) / 5.0
        v = round(max(-1.0, min(1.0, v_pos - v_neg)), 3)
        a = round(max(-1.0, min(1.0, a_pos - a_neg)), 3)
        label = map_va_to_state_label(v, a)
        return SnapshotResult(
            au_vector=au,
            face_detected=True,
            v_a=(v, a),
            state_label=label,
            state_label_zh=STATE_LABELS_ZH[label],
            confidence=0.65,  # py-feat 占位 V-A 投影置信度
            source="pyfeat+heuristic",
            error=None,
        )

    # mock 模式（默认）
    return _mock_snapshot(image_bytes)


def _mock_snapshot(image_bytes: bytes, fallback_reason: str | None = None) -> SnapshotResult:
    v, a = _deterministic_va_from_bytes(image_bytes)
    au = _mock_au_vector(v, a)
    label = map_va_to_state_label(v, a)
    return SnapshotResult(
        au_vector=au,
        face_detected=True,
        v_a=(v, a),
        state_label=label,
        state_label_zh=STATE_LABELS_ZH[label],
        confidence=0.50,  # mock 模式诚实标低
        source="mock" if fallback_reason is None else f"mock(fallback:{fallback_reason})",
        error=None,
    )
