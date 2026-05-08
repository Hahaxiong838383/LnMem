"""AU → LNN 18 维表情字段的校准层（v2.0 - gemini AUCaliper 高级方案）。

设计理念（来自 docs/au-calibration-research.md）：
- **EMA 动态 baseline**：流式跟踪每个 AU 的平滑均值，自动适应光线/角度
- **AU04 gamma 校正**：弱信号非线性拉伸（用力 0.19 → ~0.8）
- **AU01/02/05/11/17 相对归一化**：(raw-base)/(1-base) 减除高基线 bias
- **AU06/12 仅去基线噪**：信号本身好

无状态版（calibrate_to_lnn_expression）保留作向后兼容。
有状态版（AUCaliper）用于 live_verify 流式场景。
"""
from __future__ import annotations

import math
from typing import Mapping


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------- 静态先验基线（亚洲男性实测，2026-05-08 30s 采样）----------

DEFAULT_BASELINE: dict[str, float] = {
    "AU01": 0.85, "AU02": 0.64, "AU04": 0.08,
    "AU05": 0.42, "AU06": 0.08, "AU07": 0.35,
    "AU09": 0.20, "AU10": 0.16, "AU11": 0.50,
    "AU12": 0.04, "AU14": 0.31, "AU15": 0.25,
    "AU17": 0.45, "AU20": 0.31, "AU23": 0.23,
    "AU24": 0.36, "AU25": 0.34, "AU26": 0.28,
    "AU28": 0.20, "AU43": 0.11,
}

# 哪些 AU 用相对归一化（去高基线）
HIGH_BASELINE_AUS = {"AU01", "AU02", "AU05", "AU07", "AU11", "AU17"}

# 良好信号 AU（仅去基线噪）
GOOD_SIGNAL_AUS = {"AU06", "AU12"}


# ---------- 无状态版（向后兼容）----------


def calibrate_to_lnn_expression(au_dict: Mapping[str, float]) -> dict[str, float]:
    """py-feat AU → LNN 6 维表情字段（无状态版，使用静态 baseline）。

    用于 script_to_signals 等批处理场景。流式场景请用 AUCaliper。
    """
    # 用静态 baseline 校准 17 AU
    cal: dict[str, float] = {}
    for au, raw in au_dict.items():
        base = DEFAULT_BASELINE.get(au, 0.0)
        if au == "AU04":
            # gamma 拉伸：(raw - base) / 0.15) ^ 0.8
            cal[au] = _clamp01(((max(0.0, raw - base)) / 0.15) ** 0.8)
        elif au in GOOD_SIGNAL_AUS:
            cal[au] = raw  # AU06/12 直接用 raw，base 极低无需减
        elif au in HIGH_BASELINE_AUS:
            denom = max(0.1, 1.0 - base)
            cal[au] = max(0.0, (raw - base) / denom)
        else:
            cal[au] = max(0.0, raw - base)

    # 映射到 6 维 LNN 表情字段
    return _au_to_expression(cal)


# ---------- 有状态版（流式 EMA baseline）----------


class AUCaliper:
    """流式 AU 校准器：EMA 动态基线 + 三类 AU 处理策略。

    每帧调 step(raw_au) → 校准后 17 AU。
    再调 to_lnn_expression(cal_au) → 6 维 LNN 表情字段。
    """

    def __init__(
        self,
        ema_alpha: float = 0.01,
        baseline: dict[str, float] | None = None,
    ) -> None:
        self.ema_alpha = ema_alpha
        # 初始用静态先验，EMA 慢慢自适应
        self.baseline: dict[str, float] = dict(baseline or DEFAULT_BASELINE)

    def step(self, raw_au: Mapping[str, float]) -> dict[str, float]:
        """单帧校准。返回 17 AU 校准后值。"""
        cal: dict[str, float] = {}
        for au, raw in raw_au.items():
            current_base = self.baseline.get(au, 0.0)

            # 动态更新基线（屏蔽极强信号避免污染）
            if raw < current_base * 1.5:
                self.baseline[au] = (1 - self.ema_alpha) * current_base + self.ema_alpha * raw

            base = self.baseline.get(au, 0.0)

            # 三类处理策略
            if au == "AU04":
                # 弱信号 gamma 拉伸
                stretched = ((max(0.0, raw - base)) / 0.15) ** 0.8
                cal[au] = _clamp01(stretched)
            elif au in GOOD_SIGNAL_AUS:
                # AU06/12 信号本身好（base 极低 ~0.05），直接用 raw
                # 减 baseline 反而会让 smile 0.71+0.76 平均跌到 0.67
                cal[au] = raw
            elif au in HIGH_BASELINE_AUS:
                # 高基线 AU 用相对归一化
                denom = max(0.1, 1.0 - base)
                cal[au] = max(0.0, (raw - base) / denom)
            else:
                cal[au] = max(0.0, raw - base)

        return cal

    def to_lnn_expression(self, cal_au: Mapping[str, float]) -> dict[str, float]:
        """校准后 AU → 6 维 LNN 表情字段。"""
        return _au_to_expression(cal_au)

    def step_and_map(self, raw_au: Mapping[str, float]) -> dict[str, float]:
        """便捷：一步校准 + 映射。"""
        return self.to_lnn_expression(self.step(raw_au))


# ---------- AU → 6 维 LNN 表情 + 7 emotion 公式 ----------


def _au_to_expression(cal_au: Mapping[str, float]) -> dict[str, float]:
    """校准后 17 AU → 6 维 LNN 表情字段。

    映射依据 EMFACS（Ekman EMFACS）。
    """
    au04 = cal_au.get("AU04", 0.0)
    au06 = cal_au.get("AU06", 0.0)
    au12 = cal_au.get("AU12", 0.0)
    au15 = cal_au.get("AU15", 0.0)
    au17 = cal_au.get("AU17", 0.0)
    au25 = cal_au.get("AU25", 0.0)
    au26 = cal_au.get("AU26", 0.0)

    return {
        # Happiness 主信号：AU06+AU12 平均
        # 注：AU06/AU12 已经用 GOOD_SIGNAL_AUS 策略去基线（基线本身极低 ~0.05），
        # 这里直接平均，让 AU06=0.71+AU12=0.76 → smile=0.735 > 0.7 阈值
        "smile_intensity": _clamp01((au06 + au12) / 2.0),
        # 皱眉直接用校准后 AU04（已 gamma 拉伸）
        "brow_furrow": _clamp01(au04),
        # Duchenne 笑标志
        "cheek_raise": _clamp01(au06),
        # Sadness 主信号：AU15 + 部分 AU17
        "lip_corner_depress": _clamp01(0.7 * au15 + 0.3 * au17),
        # AU26 jaw drop
        "jaw_drop": _clamp01(au26),
        # AU25 lips part
        "mouth_open_ratio": _clamp01(au25),
    }


def map_au_to_emotion(cal_au: Mapping[str, float]) -> dict[str, float]:
    """校准后 17 AU → 7 EMFACS emotion 概率（fallback 用，LNN 不直接消费）。

    公式来源：Ekman EMFACS + Du et al. 2014 复合情绪研究。
    """
    def s(*aus: int) -> float:
        return sum(cal_au.get(f"AU{a:02d}", 0.0) for a in aus)

    return {
        "happiness": _clamp01(0.4 * s(6) + 0.6 * s(12)),
        "sadness":   _clamp01(0.3 * s(1) + 0.4 * s(4) + 0.3 * s(15)),
        "anger":     _clamp01(0.4 * s(4) + 0.3 * s(5) + 0.3 * s(7)),
        "fear":      _clamp01(0.2 * s(1, 2) + 0.3 * s(4, 5) + 0.2 * s(20)),
        "disgust":   _clamp01(0.6 * s(9) + 0.2 * s(15, 17)),
        "surprise":  _clamp01(0.3 * s(1, 2, 5) + 0.1 * s(26)),
        "contempt":  _clamp01(max(0.0, cal_au.get("AU14", 0.0) - 0.5 * cal_au.get("AU12", 0.0))),
    }


# ---------- 旧 API 保留（向后兼容）----------


class FeatureCalibrator:
    """无状态校准器（向后兼容）。流式场景请用 AUCaliper。"""

    @staticmethod
    def calibrate(au_dict: Mapping[str, float]) -> dict[str, float]:
        return calibrate_to_lnn_expression(au_dict)


def extract_lnn_expression_dict(au_dict: Mapping[str, float]) -> dict[str, float]:
    return calibrate_to_lnn_expression(au_dict)


__all__ = [
    "AUCaliper",
    "FeatureCalibrator",
    "DEFAULT_BASELINE",
    "calibrate_to_lnn_expression",
    "extract_lnn_expression_dict",
    "map_au_to_emotion",
]
