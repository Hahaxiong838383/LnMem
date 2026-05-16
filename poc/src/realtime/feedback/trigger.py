"""ProactiveQueryTrigger — 监听信号判断是否触发询问。

触发条件（V0.3 Phase B 多通道版）：
1. brow_burst: 连续 N 帧 brow_furrow > threshold（强烈皱眉，V0.2→V0.3 调严）
2. low_vlm_conf: vlm_label.confidence < threshold（VLM 拿不准）
3. **pose_held**（V0.3 新通道）: 滑窗 N 帧 head_pose stddev < threshold（长时间专注盯屏）

每个通道独立 cooldown（按 frame_id 计），防同通道连续触发，保证弹窗 reason 多样化。

触发后：emit TriggerEvent，清窗口 + 设 cooldown deadline。

FB-09 fix (V0.2)：
VLM 可能自报"未检测到人脸"等，此时 confidence=0 不应触发弹窗。
加双重过滤：au_description 命中关键词 AND confidence < 0.1 -> 跳过。

V0.3 Phase B 数据驱动决策（基于 2026-05-09~11 共 77979 帧实测）：
- brow P75=0.936 → V0.2 0.85+10 帧每 2.5min 触发，过敏 → V0.3 默认 0.95+15 帧（每 8min）
- smile_intensity / cheek_raise 49% 饱和到 1.0（信号病态），absolute threshold 失效 → 放弃
- pose_held std<2.5°/60 帧每 2.3min 触发，加 300s cooldown 后 ~5min 一次合理
"""
from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# V0.3 Phase B 阈值（数据驱动收紧）：
# - brow window 10→15 帧 (15s @ 1fps)，过滤瞬间皱眉
# - brow threshold 0.85→0.95，过滤老板日常 brow=0.85-0.93（P75）
#
# V0.3.1（2026-05-16 川哥钦定 dual-mode）：LNMEM_TRIGGER_MODE 切 demo / work
# V0.3.2（2026-05-16 ADR-005 川哥钦定 dual signal_path）：
#   - 5/16 D-1 数据分析发现 expression.brow_furrow 被 calibration 函数 polarized
#     川哥 raw_au.AU04 P75=0.395, P91=0.55, P99=0.757，calibration gamma 拉伸把 raw>0.19 全 clamp 到 1.0
#   - 新增 LNMEM_SIGNAL_PATH = raw / calibrated 切换
#     * calibrated（默认，向后兼容）：brow_burst 用 expression.brow_furrow（旧阈值 0.95）
#     * raw（v0.3.2 推荐）：brow_burst 用 raw_au.AU04（新数据驱动阈值 0.55 = P91）
#   - signal_path 用 ~/.lnmem/signal_path 文件 + env 变量双源（与 trigger_mode 同范式）
import os as _os

def _detect_setting(env_key: str, file_path: str, valid: set, default: str) -> str:
    """优先级：env 变量 > 文件 > default"""
    env_val = _os.environ.get(env_key, "").strip().lower()
    if env_val in valid:
        return env_val
    file_full = _os.path.expanduser(file_path)
    try:
        if _os.path.exists(file_full):
            with open(file_full) as _f:
                file_val = _f.read().strip().lower()
            if file_val in valid:
                return file_val
    except Exception:
        pass
    return default

_TRIGGER_MODE = _detect_setting(
    "LNMEM_TRIGGER_MODE", "~/.lnmem/trigger_mode", {"demo", "work"}, "work"
)
_SIGNAL_PATH = _detect_setting(
    "LNMEM_SIGNAL_PATH", "~/.lnmem/signal_path", {"raw", "calibrated"}, "calibrated"
)

if _TRIGGER_MODE == "demo":
    _BROW_BURST_WINDOW = 5
    _BROW_BURST_THRESHOLD_CAL = 0.80
    _BROW_BURST_THRESHOLD_RAW = 0.40  # raw 路径下 demo 阈值（≈ P76）
    _LOW_VLM_CONF_THRESHOLD = 0.65
    _POSE_HELD_WINDOW = 20
    _POSE_HELD_STDDEV_THRESHOLD_DEG = 3.5
    _POSE_OUTLIER_ABS_THRESHOLD_DEG = 60.0
    _COOLDOWN_FRAMES_BROW = 30
    _COOLDOWN_FRAMES_POSE_HELD = 30
    _COOLDOWN_FRAMES_LOW_VLM_CONF = 30
    # TODO-1 (ADR-010) demo 模式 Schmitt+EMA 参数（sweep_brow_threshold_quantile 2026-05-16）：
    #   α=0.40 平滑掉脉冲信号的 1 帧 dropout；
    #   thr_high=0.42 ≈ P85 of EMA(raw_AU04)，比 work 模式更敏感；
    #   thr_low=0.22 留 ~50% 滞回带防震荡。
    #   demo session (638 帧) 模拟触发 5 次 ≥ ADR-010 验收阈 2 次。
    _BROW_EMA_ALPHA_RAW = 0.40
    _BROW_THR_HIGH_RAW = 0.42
    _BROW_THR_LOW_RAW = 0.22
else:  # work (default)
    _BROW_BURST_WINDOW = 15
    _BROW_BURST_THRESHOLD_CAL = 0.95
    # ADR-005 数据驱动: raw_au.AU04 P91=0.55，每天 ~5-6 次触发（与 work 节奏一致）
    _BROW_BURST_THRESHOLD_RAW = 0.55
    _LOW_VLM_CONF_THRESHOLD = 0.65
    _POSE_HELD_WINDOW = 60
    _POSE_HELD_STDDEV_THRESHOLD_DEG = 2.5
    _POSE_OUTLIER_ABS_THRESHOLD_DEG = 60.0
    _COOLDOWN_FRAMES_BROW = 300
    _COOLDOWN_FRAMES_POSE_HELD = 300
    _COOLDOWN_FRAMES_LOW_VLM_CONF = 120
    # TODO-1 (ADR-010) work 模式 Schmitt+EMA 参数（sweep_brow_threshold_quantile 2026-05-16）：
    #   α=0.40 + thr_high=0.55 (≈P95 of EMA) + thr_low=0.30；
    #   demo session 触发 2 次 / 7 日全量 ~405 次（~60/日，~每 12 min 一次，匹配 work 节奏）。
    _BROW_EMA_ALPHA_RAW = 0.40
    _BROW_THR_HIGH_RAW = 0.55
    _BROW_THR_LOW_RAW = 0.30

# F-08 fix (codex v0.3 二审 2026-05-16)：
# _BROW_BURST_THRESHOLD 永远 = _BROW_BURST_THRESHOLD_CAL（calibrated 默认 0.95）
# observe() 内部按 _SIGNAL_PATH 选用 _RAW 还是 _CAL，不让模块默认值随运行时环境变化。
# 这样所有现有测试（断言默认 0.95）保持稳定 ≡ "测试环境无关"
_BROW_BURST_THRESHOLD = _BROW_BURST_THRESHOLD_CAL

# FB-09: VLM 自报无脸的关键词清单（中文 + 英文）
# 中英双语必须都覆盖：gemini-3.1-pro-preview 在 prompt 不强制中文时会偶尔回英文
# （5/10 04:32 实证：au_description="The image does not contain a face." 漏过）
# 匹配采用 lower-case 包含语义，故英文项必须全部小写。
VLM_NO_FACE_KEYWORDS_ZH = (
    "未检测到人脸",
    "图片中未出现人脸",
    "图中未出现人脸",
    "图片中未包含人脸",
    "图中未包含人脸",
    "无法识别",
    "无法分析",
    "无法检测",
    "无法进行",
)
VLM_NO_FACE_KEYWORDS_EN = (
    "does not contain a face",
    "does not contain any face",
    "no face detected",
    "no face is detected",
    "no face visible",
    "cannot detect a face",
    "cannot detect any face",
    "unable to detect",
    "unable to identify",
    "unable to analyze",
    "unable to analyse",
    "image does not contain",
    "no human face",
)
# FB-09 v3: VLM 偶尔返字面量 sentinel（gemini-3.1-pro-preview quirk）。
# 实证 5/10 14:16 那条：au_description="None" + confidence=0.0 漏过 keyword 列表
# 进入弹窗。这些值与 confidence=0 配合即视为无脸。
# 必须全小写 + strip 后对照（避免 "Hello, none here" 类正常描述误命中——只匹配
# au_description 整体等于 sentinel，不做子串）。
VLM_NO_FACE_DESC_SENTINELS = frozenset({
    "",
    "none",
    "null",
    "n/a",
    "na",
    "no description",
    "no_description",
    "unknown",
})
# Public alias 保持向后兼容（旧测试可能 import VLM_NO_FACE_KEYWORDS）
VLM_NO_FACE_KEYWORDS = VLM_NO_FACE_KEYWORDS_ZH + VLM_NO_FACE_KEYWORDS_EN
VLM_NO_FACE_CONF_THRESHOLD = 0.1

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriggerEvent:
    """触发询问事件（不可变）。"""
    ts: str
    frame_id: int
    reason: str  # "brow_burst" | "low_vlm_conf" | "pose_held"
    signal_snapshot: dict[str, float]
    vlm_label: dict[str, Any] | None


class ProactiveQueryTrigger:
    """监听 signal pipeline 输出 dict，判断是否应触发询问。

    V0.3 Phase B：三通道 + per-channel cooldown。
    """

    def __init__(
        self,
        brow_threshold: float = _BROW_BURST_THRESHOLD,
        brow_window: int = _BROW_BURST_WINDOW,
        vlm_conf_threshold: float = _LOW_VLM_CONF_THRESHOLD,
        pose_held_window: int = _POSE_HELD_WINDOW,
        pose_held_stddev_threshold_deg: float = _POSE_HELD_STDDEV_THRESHOLD_DEG,
        pose_outlier_abs_threshold_deg: float = _POSE_OUTLIER_ABS_THRESHOLD_DEG,
        cooldown_frames_brow: int = _COOLDOWN_FRAMES_BROW,
        cooldown_frames_pose_held: int = _COOLDOWN_FRAMES_POSE_HELD,
        cooldown_frames_low_vlm_conf: int = _COOLDOWN_FRAMES_LOW_VLM_CONF,
        enable_pose_held: bool = True,
        home: Path | None = None,
        signal_path: str = "calibrated",  # ADR-005 F-08 fix: 实例属性，不读 module-level
        # TODO-1 (ADR-010) Schmitt + EMA opt-in mode（默认关，daemon raw 路径下打开）：
        #   use_schmitt_ema=True 时 brow_burst 走 EMA 平滑 + Schmitt 滞回判定，
        #   而非 strict-deque all(b > thr) — 容忍 1 帧 dropout 的脉冲信号。
        use_schmitt_ema: bool = False,
        brow_ema_alpha: float = 0.40,
        brow_thr_high: float | None = None,  # None → 回落 brow_threshold（向后兼容）
        brow_thr_low: float | None = None,   # None → 回落 brow_threshold * 0.55
    ) -> None:
        self._brow_threshold = brow_threshold
        self._signal_path = signal_path if signal_path in ("raw", "calibrated") else "calibrated"
        self._brow_window_size = brow_window
        self._vlm_conf_threshold = vlm_conf_threshold
        self._brow_history: deque[float] = deque(maxlen=brow_window)
        # TODO-1 (ADR-010) Schmitt + EMA 状态：
        #   _brow_ema：EMA 平滑后的 brow 值（None=首帧前 / face 丢失后清空）
        #   _brow_above_high：Schmitt 状态机（True=已在 thr_high 之上，等 thr_low 解禁）
        #   _brow_samples_since_reset：face-lost 后样本计数，需 ≥ brow_window 才允许触发
        self._use_schmitt_ema = bool(use_schmitt_ema)
        # α 必须在 (0, 1]，越界回落 0.40
        if isinstance(brow_ema_alpha, (int, float)) and 0.0 < float(brow_ema_alpha) <= 1.0:
            self._brow_ema_alpha = float(brow_ema_alpha)
        else:
            self._brow_ema_alpha = 0.40
        _hi = float(brow_thr_high) if isinstance(brow_thr_high, (int, float)) else float(brow_threshold)
        _lo = float(brow_thr_low) if isinstance(brow_thr_low, (int, float)) else float(brow_threshold) * 0.55
        # cc 自审 F6: 防 thr_high ≤ 0 让滞回退化为"永远 above"死锁——回落到 brow_threshold 或最小值
        if _hi <= 0.0:
            _hi = max(float(brow_threshold), 0.1)
        # 保证 thr_low ≤ thr_high（防参数手滑）；同值退化为非滞回阈值器
        if _lo > _hi:
            _lo = _hi * 0.55
        # 同步 thr_low ≥ 0（避免负值在 ema=0 时永不下降沿）
        if _lo < 0.0:
            _lo = 0.0
        self._brow_thr_high = _hi
        self._brow_thr_low = _lo
        self._brow_ema: float | None = None
        self._brow_above_high: bool = False
        self._brow_samples_since_reset: int = 0
        # V0.3 pose_held 状态：分别滑窗存 pitch / yaw / roll
        # F2-04: enable_pose_held=False 时全程不维护 pose 滑窗（V0.2 真实禁用语义）
        self._enable_pose_held = enable_pose_held
        self._pose_window_size = pose_held_window
        self._pose_stddev_threshold = pose_held_stddev_threshold_deg
        self._pose_outlier_threshold = pose_outlier_abs_threshold_deg
        self._pose_pitch: deque[float] = deque(maxlen=pose_held_window)
        self._pose_yaw: deque[float] = deque(maxlen=pose_held_window)
        self._pose_roll: deque[float] = deque(maxlen=pose_held_window)
        # V0.3 per-channel cooldown：last_trigger_frame_id（None=从未触发）
        self._cooldown_frames = {
            "brow_burst": cooldown_frames_brow,
            "pose_held": cooldown_frames_pose_held,
            "low_vlm_conf": cooldown_frames_low_vlm_conf,
        }
        self._last_trigger_frame: dict[str, int | None] = {
            "brow_burst": None,
            "pose_held": None,
            "low_vlm_conf": None,
        }
        # F1-01: 跟踪上一帧 frame_id，用于检测 stream 重启/跨天回退
        self._last_seen_frame_id: int | None = None
        self._home = home or Path.home()

    def _in_cooldown(self, channel: str, frame_id: int) -> bool:
        """True 若 channel 距上次触发不足 cooldown_frames。"""
        last = self._last_trigger_frame.get(channel)
        if last is None:
            return False
        delta = frame_id - last
        if delta < 0:
            # frame_id 回退（防御保险，正常已被 observe() 入口的单调性检查捕获）
            self._last_trigger_frame[channel] = None
            return False
        cd = self._cooldown_frames[channel]
        return delta < cd

    def _mark_trigger(self, channel: str, frame_id: int) -> None:
        self._last_trigger_frame[channel] = frame_id
        self._last_seen_frame_id = frame_id

    def _detect_and_handle_frame_rollback(self, frame_id: int) -> None:
        """F1-01 (codex 一审)：frame_id 跨天会被重置（实测 5/9 fid=801→5/10 fid=194）。

        在 observe() 入口主动检测：若 frame_id 比上次见过的小，视为 stream 重启
        → 一次性 reset：
        (a) 所有 channel cooldown 状态
        (b) NF-01 (codex 二审)：brow_history + pose 滑窗 evidence
            —— 防跨 session 旧 evidence 在 F3-06 修复下立即触发新 session 首帧
        """
        if self._last_seen_frame_id is not None and frame_id < self._last_seen_frame_id:
            for ch in self._last_trigger_frame:
                self._last_trigger_frame[ch] = None
            # NF-01: rollback 必须清 evidence，否则 F3-06 "cooldown 内保留 evidence"
            # 会让旧 session 的高 brow 窗口在新 session 首帧立即触发
            self._brow_history.clear()
            # TODO-1 (ADR-010): Schmitt+EMA 状态同步清空，避免跨 session evidence 残留
            self._brow_ema = None
            self._brow_above_high = False
            self._brow_samples_since_reset = 0
            self._clear_pose_windows()
        self._last_seen_frame_id = frame_id

    def _is_vlm_self_report_no_face(self, vlm_label: dict[str, Any] | None) -> bool:
        """判断 VLM 是否自报"未检测到人脸"。

        两种 confidence=0 语义：
        (a) VLM 看到脸但拿不准状态 → 应该问用户（correct）
        (b) VLM 自报未检测到人脸 → 不该问用户（无效弹窗）

        双重过滤：au_description 命中关键词 AND confidence < 0.1
        """
        if not vlm_label:
            return False

        conf = vlm_label.get("confidence", 1.0)
        if not isinstance(conf, (int, float)) or conf >= VLM_NO_FACE_CONF_THRESHOLD:
            return False

        desc_raw = vlm_label.get("au_description")
        # FB-09 v3: au_description 可能是字面量字符串 "None" 而非 Python None
        if desc_raw is None:
            desc = ""
        else:
            desc = str(desc_raw).strip()

        # FB-09 v3 sentinel：au_description 整体等于已知占位字面量（含空串）→ 视为无脸
        if desc.lower() in VLM_NO_FACE_DESC_SENTINELS:
            return True

        # 中文 keyword 直接匹配（保持原语义）；英文 keyword 走 lower-case 包含
        if any(kw in desc for kw in VLM_NO_FACE_KEYWORDS_ZH):
            return True
        desc_lower = desc.lower()
        return any(kw in desc_lower for kw in VLM_NO_FACE_KEYWORDS_EN)

    def _log_vlm_no_face_skip(self, item: dict[str, Any], vlm_label: dict[str, Any] | None) -> None:
        """记录 VLM 自报无脸导致的跳过事件到 ~/.lnmem/feedback/skipped_<date>.jsonl。"""
        try:
            feedback_dir = self._home / ".lnmem" / "feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)

            day = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
            path = feedback_dir / f"skipped_{day}.jsonl"

            record = {
                "ts": str(item.get("ts", "")),
                "frame_id": int(item.get("frame_id", 0)),
                "reason": "vlm_no_face_skip",
                "vlm_label": vlm_label,
                "au_description": (vlm_label or {}).get("au_description", ""),
            }

            line = json.dumps(record, ensure_ascii=False) + "\n"
            with path.open("a", encoding="utf-8") as f:
                f.write(line)

        except Exception as e:
            _log.warning("Failed to log vlm_no_face_skip: %s", e)

    @staticmethod
    def _safe_float(x: Any) -> float | None:
        """F2-03 (codex 一审)：脏数据防御 — None/str/NaN/inf 均返 None。

        注：bool 是 int 的子类，需显式排除（True 会被误当 1.0）。
        """
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)) and math.isfinite(x):
            return float(x)
        return None

    def _clear_pose_windows(self) -> None:
        self._pose_pitch.clear()
        self._pose_yaw.clear()
        self._pose_roll.clear()

    def _update_pose_window(self, head_pose: dict[str, Any] | None) -> None:
        """更新 pose 滑窗。

        F2-03: head_pose 字段任意值通过 _safe_float 防御性 parse。
        F3-05: roll 字段可选（schema 不固定，py-feat 偶缺）；缺则**仅不入 roll 滑窗**，
              pitch/yaw 仍参与统计；不再用 0.0 兜底污染 std。
        outlier (>60°) 任一轴触发 → 整窗作废。
        """
        if not self._enable_pose_held:
            return  # F2-04: V0.2 真实禁用语义

        if not isinstance(head_pose, dict):
            self._clear_pose_windows()
            return

        pitch = self._safe_float(head_pose.get("pitch_deg"))
        yaw = self._safe_float(head_pose.get("yaw_deg"))
        roll = self._safe_float(head_pose.get("roll_deg"))  # 可缺

        # 必须的轴（pitch/yaw）缺则清窗
        if pitch is None or yaw is None:
            self._clear_pose_windows()
            return

        # outlier：任一可用轴超阈值 → 整窗作废
        out = self._pose_outlier_threshold
        if abs(pitch) > out or abs(yaw) > out or (roll is not None and abs(roll) > out):
            self._clear_pose_windows()
            return

        self._pose_pitch.append(pitch)
        self._pose_yaw.append(yaw)
        if roll is not None:
            self._pose_roll.append(roll)
        else:
            # F3-05 corner case (实测 7.5% 帧无 roll)：
            # 若 roll deque 已满（来自历史帧）而当前帧缺 roll，
            # 不清 roll deque 会让 roll std 计算的窗口与 pitch/yaw 不对齐
            # （roll 内容滞后）。清空 roll deque 强制后续仅 pitch/yaw 参与 max。
            self._pose_roll.clear()

    def _pose_held_max_stddev(self) -> float | None:
        """返回滑窗内 max(pitch_std, yaw_std, [roll_std])，窗未满返 None。

        F3-05: roll 滑窗满时才参与 max；不满（roll 字段间歇缺）时仅用 pitch/yaw。
        """
        if len(self._pose_pitch) < self._pose_window_size:
            return None
        n = self._pose_window_size

        def _std(buf: deque[float]) -> float:
            mean = sum(buf) / n
            return math.sqrt(sum((v - mean) ** 2 for v in buf) / n)

        candidates = [_std(self._pose_pitch), _std(self._pose_yaw)]
        if len(self._pose_roll) == n:
            candidates.append(_std(self._pose_roll))
        return max(candidates)

    def observe(self, item: dict[str, Any]) -> TriggerEvent | None:
        """处理一条 signal pipeline 输出，返回 TriggerEvent 或 None。

        V0.3 Phase B 通道优先级（per-channel cooldown 独立）：
        1. low_vlm_conf（VLM 描述具体，先看）
        2. brow_burst（强烈情绪）
        3. pose_held（专注盯屏）
        """
        frame_id = int(item.get("frame_id", 0))
        ts = str(item.get("ts", ""))

        # F1-01: 跨天/daemon 重启会让 frame_id 回退；检测到回退则 reset cooldown
        self._detect_and_handle_frame_rollback(frame_id)

        if not item.get("face_detected"):
            self._brow_history.clear()
            # TODO-1 (ADR-010): face 丢失等同于 evidence 重置，Schmitt+EMA 状态同步清
            self._brow_ema = None
            self._brow_above_high = False
            self._brow_samples_since_reset = 0
            self._clear_pose_windows()
            return None

        # ADR-005 dual signal_path（2026-05-16）：
        #   - raw 模式：brow = raw_au.AU04（py-feat 原始皱眉强度，避免 calibration polarization）
        #   - calibrated 模式（默认，向后兼容）：brow = expression.brow_furrow
        # F-08 fix: signal_path 是实例属性 self._signal_path（构造时传入），不读 module-level
        expression = item.get("expression") or {}
        raw_au = item.get("raw_au") or {}
        # V0.3.6 NaN 防御（ADR-010 cc 二审 2026-05-16 23:58）：
        # py-feat 偶发返 NaN/Inf（实测 14:30-14:59 demo session 1714 帧中 6 个 NaN）。
        # 之前 float(raw_au.AU04) 直通 → ema 一旦遇 NaN 永久卡 NaN（NaN op any = NaN），
        # Schmitt 后续永远不再触发。NaN brow 视为"本帧无信号"：不入 history/ema/samples。
        _raw_brow = raw_au.get("AU04", 0.0) if self._signal_path == "raw" else expression.get("brow_furrow", 0.0)
        brow_safe = self._safe_float(_raw_brow)
        brow = brow_safe if brow_safe is not None else 0.0  # 兜底（snapshot 显示用 0）
        if brow_safe is not None:
            self._brow_history.append(brow_safe)
            # TODO-1 (ADR-010) Schmitt+EMA：每帧更新 EMA + 样本计数
            if self._brow_ema is None:
                self._brow_ema = brow_safe
            else:
                a = self._brow_ema_alpha
                self._brow_ema = a * brow_safe + (1.0 - a) * self._brow_ema
            self._brow_samples_since_reset += 1
        # NaN brow 不入 history/ema/samples，等下一帧；above 状态保留

        self._update_pose_window(item.get("head_pose"))

        # signal_snapshot 同时记 raw 与 calibrated 两路，便于 A/B 离线分析
        # TODO-1: 新增 brow_ema / brow_above_high 字段，供 daemon log + 离线 replay 诊断
        # V0.3.6 snapshot 字段 NaN 防御（同 brow 路径）
        _brow_furrow_safe = self._safe_float(expression.get("brow_furrow", 0.0))
        _raw_au04_safe = self._safe_float(raw_au.get("AU04", 0.0))
        signal_snapshot = {
            "brow_furrow": _brow_furrow_safe if _brow_furrow_safe is not None else 0.0,
            "raw_au_AU04": _raw_au04_safe if _raw_au04_safe is not None else 0.0,
            "brow_used_path": self._signal_path,
            "brow_used_value": brow,
            # _brow_ema 可能因首帧 NaN 仍是 None；snapshot 用 0.0 兜底（不影响决策）
            "brow_ema": float(self._brow_ema) if self._brow_ema is not None else 0.0,
            "brow_above_high": bool(self._brow_above_high),
            "smile_intensity": float(expression.get("smile_intensity", 0.0)),
            "eye_closure": float(expression.get("eye_closure", 0.0)),
            "cheek_raise": float(expression.get("cheek_raise", 0.0)),
        }
        vlm_label = item.get("vlm_label")

        # 优先级 1: low_vlm_conf（有 VLM 描述更具体）
        # F2-02 (codex 一审): no-face 自报**只**屏蔽 low_vlm_conf 通道，
        # 仍允许 brow/pose 通道在同一帧评估（多通道独立性）
        if isinstance(vlm_label, dict):
            conf = vlm_label.get("confidence")
            if isinstance(conf, (int, float)) and conf < self._vlm_conf_threshold:
                if self._is_vlm_self_report_no_face(vlm_label):
                    self._log_vlm_no_face_skip(item, dict(vlm_label))
                    # 不 return — 继续看 brow_burst / pose_held
                elif not self._in_cooldown("low_vlm_conf", frame_id):
                    self._mark_trigger("low_vlm_conf", frame_id)
                    return TriggerEvent(
                        ts=ts,
                        frame_id=frame_id,
                        reason="low_vlm_conf",
                        signal_snapshot=signal_snapshot,
                        vlm_label=dict(vlm_label),
                    )

        # 优先级 2: brow_burst
        # ──────────────────────────────────────────────────────────────────────
        # TODO-1 (ADR-010) Schmitt + EMA 分支（opt-in）：
        #   - 上升沿 ema ≥ thr_high 且 _brow_above_high=False → 触发候选
        #   - 下降沿 ema ≤ thr_low → 退出 above 状态（重新允许下次触发）
        #   - thr_low < thr_high 间为滞回带，不切状态（防边缘抖动）
        #   - 仍需 _brow_samples_since_reset ≥ brow_window 保证"窗口稳定性"
        #   - 触发后清 _brow_samples_since_reset，cooldown 内保留 above 状态防同段重触
        # 否则走 legacy strict-deque 分支（V0.3 Phase B 原语义，向后兼容所有现有测试）。
        # ──────────────────────────────────────────────────────────────────────
        if self._use_schmitt_ema:
            ema = self._brow_ema  # 入 observe 时已更新（face_detected=True 分支）
            window_stable = self._brow_samples_since_reset >= self._brow_window_size

            # Schmitt 下降沿：始终允许解除 above 状态（防"高位 dropout 进入低位"后死锁）
            if self._brow_above_high and ema is not None and ema <= self._brow_thr_low:
                self._brow_above_high = False

            # Schmitt 上升沿：cooldown 不阻挡 latch，只阻挡 fire
            # ──────────────────────────────────────────────────────────────────
            # V0.3.6 bugfix（ADR-010 cc 一审 2026-05-16 23:55）：
            # 之前 cooldown 检查嵌在 latch 内部 → cooldown 期间"上升沿"事件被全部
            # 吞掉，above 一直停留在 False，cooldown 解禁后"上次有过高位"信息全丢，
            # 即使整段 cooldown 内 ema 多次过 thr_high 也只 fire 一次（trace 实证：
            # frame 524-535 ema 5 次过 0.42 但 above 不 latch，cooldown 解禁帧
            # ema 已跌 < thr_high，永久错过）。
            #
            # 修：把 cooldown 检查从 "latch 内部" 提到 "fire 内部"。Schmitt 状态机
            # 照常 update（latch above + reset samples），但 cooldown 内不 fire。
            # 下次 ema 跌到 thr_low 重新解锁 + 再次 ≥ thr_high + cooldown 已解禁
            # → 才 fire。物理语义：cooldown 只控制"两次 fire 间隔"，不应该让
            # Schmitt 状态机失忆。
            # ──────────────────────────────────────────────────────────────────
            just_latched = False
            if (
                not self._brow_above_high
                and ema is not None
                and ema >= self._brow_thr_high
                and window_stable
            ):
                self._brow_above_high = True
                just_latched = True

            # Fire 条件：刚 latch 这一帧 + cooldown 已解禁
            # （pending_fire 不需要单独跟踪：cooldown 内 latch 后 ema 必须先跌到
            # thr_low 解除 above，再重新爬升才会再次 just_latched，期间已自然
            # 等过 cooldown）
            if just_latched and not self._in_cooldown("brow_burst", frame_id):
                # snapshot 反映本帧 latch 后的 above 状态（便于离线复盘）
                signal_snapshot["brow_above_high"] = True
                # 触发后重置样本计数器（防同一段平台期反复触发）
                self._brow_samples_since_reset = 0
                self._mark_trigger("brow_burst", frame_id)
                return TriggerEvent(
                    ts=ts,
                    frame_id=frame_id,
                    reason="brow_burst",
                    signal_snapshot=signal_snapshot,
                    vlm_label=dict(vlm_label) if isinstance(vlm_label, dict) else None,
                )
        else:
            # F3-06 (codex 一审): cooldown 内**不清窗**，evidence 保留供 cooldown 解禁后立即触发
            if (
                len(self._brow_history) >= self._brow_window_size
                and all(b > self._brow_threshold for b in self._brow_history)
            ):
                if not self._in_cooldown("brow_burst", frame_id):
                    self._brow_history.clear()  # 仅触发时清窗，防同段 evidence 反复触发
                    self._mark_trigger("brow_burst", frame_id)
                    return TriggerEvent(
                        ts=ts,
                        frame_id=frame_id,
                        reason="brow_burst",
                        signal_snapshot=signal_snapshot,
                        vlm_label=dict(vlm_label) if isinstance(vlm_label, dict) else None,
                    )
                # cooldown 内：保留 brow_history，待 cooldown 解禁后下一帧立即触发

        # 优先级 3: pose_held（V0.3 Phase B 新增）
        # F2-04: enable_pose_held=False 时 _pose_held_max_stddev 永返 None（pitch deque 空）
        max_std = self._pose_held_max_stddev()
        if max_std is not None and max_std < self._pose_stddev_threshold:
            if not self._in_cooldown("pose_held", frame_id):
                self._mark_trigger("pose_held", frame_id)
                # F2-03: pose snapshot 用 _safe_float，缺/脏返 None
                head_pose = item.get("head_pose") or {}
                signal_snapshot_with_pose = {
                    **signal_snapshot,
                    "pose_pitch_deg": self._safe_float(head_pose.get("pitch_deg")),
                    "pose_yaw_deg": self._safe_float(head_pose.get("yaw_deg")),
                    "pose_roll_deg": self._safe_float(head_pose.get("roll_deg")),
                    "pose_max_stddev_deg": float(max_std),
                }
                # 清窗防同一段静止反复触发
                self._clear_pose_windows()
                return TriggerEvent(
                    ts=ts,
                    frame_id=frame_id,
                    reason="pose_held",
                    signal_snapshot=signal_snapshot_with_pose,
                    vlm_label=dict(vlm_label) if isinstance(vlm_label, dict) else None,
                )

        return None
