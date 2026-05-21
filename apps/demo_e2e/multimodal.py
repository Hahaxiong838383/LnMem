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
    au_vector: dict[str, float]      # 20 AU → [0, 1] (py-feat XGB 概率) 或 mock 反推
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
    # Valence influence (mock，范围 [0, 1] 跟 py-feat 对齐)
    if v > 0:
        au["AU06"] = round(min(1.0, abs(v) * 0.8), 3)
        au["AU12"] = round(min(1.0, abs(v) * 0.9), 3)
    else:
        au["AU04"] = round(min(1.0, abs(v) * 0.8), 3)
        au["AU15"] = round(min(1.0, abs(v) * 0.7), 3)
    # Arousal influence
    if a > 0:
        au["AU01"] = round(min(1.0, abs(a) * 0.6), 3)
        au["AU02"] = round(min(1.0, abs(a) * 0.55), 3)
        au["AU05"] = round(min(1.0, abs(a) * 0.7), 3)
    else:
        au["AU43"] = round(min(1.0, abs(a) * 0.8), 3)
    return au


# ──────────────────── 真 py-feat 推理（可选开关，Phase β 启用） ────────────────────

def _au_to_valence_arousal(au: dict[str, float]) -> tuple[float, float]:
    """11 AU → V-A 启发投影 v3（py-feat 0.6.2 输出 [0, 1] 概率适配）。

    Valence 维度（情感极性）—— 笑脸 AU 减 皱眉/愤怒 AU：
    - 正：AU06 (cheek raiser=Duchenne smile) + AU12 (lip corner pull) + AU25 (lips part 微笑)
    - 负：AU04 (brow lowerer=皱眉) + AU07 (lid tighten=愤怒) + AU09 (nose wrinkle=厌恶) +
          AU15 (lip corner depress=悲伤) + AU17 (chin raise=不满) + AU23 (lip tighten=紧张)

    Arousal 维度（唤起度）—— 兴奋/惊讶 AU 减 疲惫 AU：
    - 正：AU01 (inner brow up=担忧) + AU02 (outer brow up=惊讶) + AU05 (upper lid raise=警觉) +
          AU20 (lip stretch=惊吓) + AU26 (jaw drop=张嘴惊讶)
    - 负：AU43 (eyes closed=疲惫/困)

    py-feat 输出 AU 是 [0, 1] 概率（XGB 分类器），不是 0-5 强度。
    """
    def s(k): return au.get(k, 0.0)

    # Valence（AU 已是 [0, 1]，加权求和直接是 [0, 1] 量级）
    v_pos = s("AU06") * 0.35 + s("AU12") * 0.45 + s("AU25") * 0.20
    v_neg = (s("AU04") * 0.25 + s("AU07") * 0.15 + s("AU09") * 0.15
             + s("AU15") * 0.20 + s("AU17") * 0.10 + s("AU23") * 0.15)
    # v_pos / v_neg 各自 ∈ [0, 1]，差值 ∈ [-1, 1]
    v = max(-1.0, min(1.0, v_pos - v_neg))

    # Arousal
    a_pos = (s("AU01") * 0.20 + s("AU02") * 0.20 + s("AU05") * 0.25
             + s("AU20") * 0.15 + s("AU26") * 0.20)
    a_neg = s("AU43")
    a = max(-1.0, min(1.0, a_pos - a_neg))

    return round(v, 3), round(a, 3)


def extract_au_from_image(image_bytes: bytes) -> dict[str, float] | None:
    """py-feat 真识别 20 AU。失败返回 None（不抛异常，由上层兜底）。

    py-feat 0.6.2 实测（2026-05-21 调试）：
    - detect_image(np.ndarray) → AttributeError（API 只接受文件路径）
    - 列名是 AU01 / AU02 等（不带 _r 后缀）
    - 输出值 [0, 1] 概率（不是 0-5 强度）
    - 首次推理 ~5s（模型已缓存），后续 2-3s

    首次推理需下载模型（~150MB，已通过 self-test 缓存）。
    """
    if not image_bytes:
        return None

    tmp_path = None
    try:
        # 延迟 import 避免 streamlit 启动阻塞
        import tempfile
        from feat import Detector  # type: ignore

        # py-feat 0.6.2 detect_image 只接受文件路径，bytes 必须先落盘
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name

        detector = Detector(device="cpu")
        out = detector.detect_image(tmp_path)
        if out is None or len(out) == 0:
            return None

        # py-feat 列名是 AU01 / AU02 等（不带 _r 后缀）
        row = out.iloc[0]
        au = {}
        for name in AU_NAMES_20:
            v = float(row.get(name, 0.0))
            if math.isnan(v):
                v = 0.0  # ADR-011 NaN 容错
            # py-feat XGB 输出 [0, 1]，clamp 到合法范围
            au[name] = round(max(0.0, min(1.0, v)), 3)
        return au
    except Exception:
        return None
    finally:
        # 清理临时文件
        if tmp_path:
            try:
                import os
                os.unlink(tmp_path)
            except OSError:
                pass


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
        v, a = _au_to_valence_arousal(au)
        label = map_va_to_state_label(v, a)
        return SnapshotResult(
            au_vector=au,
            face_detected=True,
            v_a=(v, a),
            state_label=label,
            state_label_zh=STATE_LABELS_ZH[label],
            confidence=0.65,  # py-feat 真 AU + 启发投影，base v3 替换后升 0.9+
            source="pyfeat+heuristic_v2",
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
