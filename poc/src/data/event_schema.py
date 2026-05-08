"""Day 3 ScriptEvent schema — LLM 剧本生成的标准事件契约（v0.1 / 2026-05-06）。

ScriptEvent 是 LNN 训练数据的"原子事件"——LLM 根据画像（HEXACO + 依恋 + Layer 4
if-then 情境 + Layer 6 多模态签名）生成一段时间的事件序列，每条事件含：

- 时间戳 + 时长（秒）
- 活动类型（Activity 枚举，8 类）
- 内部状态 ground truth（State 枚举，8 类，对齐 LNN StateRecognition 训练目标）
- 多模态信号提示（face/audio/context 三段，给 script_to_signals.py 转 12-dim 向量）
- 自由元数据（LLM 写入的画面描述、对话片段等）

下游消费方：
- ``script_validator.py``      — 时间连续性 + 状态分布 + 必填字段
- ``script_to_signals.py``     — 转换为 12-dim FusionFrame 序列
- ``dataset_report.py``        — 多脚本聚合报告

设计原则（与 fusion.py 同源）：
1. **frozen dataclass**：immutable，符合 ~/.claude/rules/coding-style.md
2. **State / Activity 用 Enum**：避免拼写错误污染下游
3. **JSONL 序列化**：``to_jsonl_dict()`` 输出纯 dict（Enum → str），便于 LLM 写盘
4. **零外部依赖**：纯标准库
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------- 枚举：活动 + 状态 ----------


class Activity(str, Enum):
    """8 类活动（剧本骨架）。

    覆盖 30 画像 × 8h/天的常见生活/工作切片。LLM 必须从此枚举里选一个。
    """

    WORK_FOCUS = "work_focus"          # 单人深度工作（写代码 / 写报告）
    WORK_MEETING = "work_meeting"      # 多人会议（线上或线下）
    BREAK_SHORT = "break_short"        # 短休（< 10min，喝水 / 站起活动）
    BREAK_LONG = "break_long"          # 长休（午饭 / 散步 / 运动）
    COMMUTE = "commute"                # 通勤（地铁 / 步行 / 开车）
    LEISURE = "leisure"                # 休闲娱乐（社交 / 看视频 / 阅读）
    SLEEP = "sleep"                    # 睡眠（PoC 阶段不重点采）
    TRANSITION = "transition"          # 状态切换（开会切回写代码的过渡 ≤ 60s）


class State(str, Enum):
    """8 类内部状态——与 LNN StateRecognition 训练目标 1:1 对齐。

    详见 ``poc/docs/poc-spec.md`` §4 训练任务 1。
    """

    LOW_ENERGY = "low_energy"
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    FATIGUED = "fatigued"
    EMOTIONAL_POSITIVE = "emotional_positive"
    EMOTIONAL_NEGATIVE = "emotional_negative"
    NEUTRAL = "neutral"
    TRANSITIONING = "transitioning"


# ---------- 多模态提示 ----------


@dataclass(frozen=True)
class MultimodalHint:
    """LLM 写给 script_to_signals.py 的多模态画面提示。

    字段全部归一到 [0,1] 区间——下游转换器直接喂 fusion.py 各子模态特征。
    """

    # face 子模态（4 维 face_features 的"目标值"）
    eye_closure: float = 0.15        # 0=睁圆，1=闭眼；疲劳态高
    head_pitch_norm: float = 0.0     # 头低头抬头幅度归一（PoC 看 abs(deg)/45）
    head_yaw_norm: float = 0.0       # 左右转头幅度归一
    gaze_stability: float = 0.7      # 0=游离，1=稳定注视

    # expression 子模态（Day 6 新增 6 维 AU 表情特征）
    smile_intensity: float = 0.0     # AU6+AU12 微笑强度 [0,1]
    brow_furrow: float = 0.0         # AU4 皱眉 [0,1]（1=紧皱）
    cheek_raise: float = 0.0         # AU6 脸颊抬升 [0,1]（Duchenne 标志）
    lip_corner_depress: float = 0.0  # AU15 嘴角下垂 [0,1]
    jaw_drop: float = 0.0           # AU26 下巴张开 [0,1]
    mouth_open_ratio: float = 0.0    # 嘴巴纵横比 [0,1]

    # audio 子模态
    voice_present: float = 0.0       # 0=无声，1=持续讲话
    utterance_density: float = 0.0   # 单段发言密度归一

    # context 子模态
    active_app: str = ""             # 前台 app（IDE / Slack / Chrome / "" ）
    app_switch_count_5min: int = 0   # 5min 切换次数

    def to_dict(self) -> dict[str, Any]:
        return {
            "eye_closure": self.eye_closure,
            "head_pitch_norm": self.head_pitch_norm,
            "head_yaw_norm": self.head_yaw_norm,
            "gaze_stability": self.gaze_stability,
            "smile_intensity": self.smile_intensity,
            "brow_furrow": self.brow_furrow,
            "cheek_raise": self.cheek_raise,
            "lip_corner_depress": self.lip_corner_depress,
            "jaw_drop": self.jaw_drop,
            "mouth_open_ratio": self.mouth_open_ratio,
            "voice_present": self.voice_present,
            "utterance_density": self.utterance_density,
            "active_app": self.active_app,
            "app_switch_count_5min": self.app_switch_count_5min,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MultimodalHint":
        return cls(
            eye_closure=float(d.get("eye_closure", 0.15)),
            head_pitch_norm=float(d.get("head_pitch_norm", 0.0)),
            head_yaw_norm=float(d.get("head_yaw_norm", 0.0)),
            gaze_stability=float(d.get("gaze_stability", 0.7)),
            smile_intensity=float(d.get("smile_intensity", 0.0)),
            brow_furrow=float(d.get("brow_furrow", 0.0)),
            cheek_raise=float(d.get("cheek_raise", 0.0)),
            lip_corner_depress=float(d.get("lip_corner_depress", 0.0)),
            jaw_drop=float(d.get("jaw_drop", 0.0)),
            mouth_open_ratio=float(d.get("mouth_open_ratio", 0.0)),
            voice_present=float(d.get("voice_present", 0.0)),
            utterance_density=float(d.get("utterance_density", 0.0)),
            active_app=str(d.get("active_app", "")),
            app_switch_count_5min=int(d.get("app_switch_count_5min", 0)),
        )


# ---------- ScriptEvent ----------


@dataclass(frozen=True)
class ScriptEvent:
    """剧本中的单条事件。

    LLM 输出每天 60-100 条 ScriptEvent → JSON list，下游消费方挑字段用。
    """

    event_id: str                          # e.g. "vboss_001_d0_e007"
    ts: str                                # ISO8601, e.g. "2026-05-06T09:32:15"
    persona_id: str                        # vboss_001 ~ vboss_030 / vboss_self
    day_index: int                         # 0..N，剧本日序
    duration_s: float                      # 事件时长（秒），范围 [5, 3600]
    activity: Activity
    state: State
    multimodal_hint: MultimodalHint
    description: str = ""                  # LLM 自由描述（一句话 ≤ 30 字）
    meta: dict[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> dict[str, Any]:
        """序列化为 JSONL 友好 dict（Enum → str，frozen dataclass → dict）。"""
        return {
            "event_id": self.event_id,
            "ts": self.ts,
            "persona_id": self.persona_id,
            "day_index": self.day_index,
            "duration_s": self.duration_s,
            "activity": self.activity.value,
            "state": self.state.value,
            "multimodal_hint": self.multimodal_hint.to_dict(),
            "description": self.description,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_jsonl_dict(cls, d: dict[str, Any]) -> "ScriptEvent":
        return cls(
            event_id=str(d["event_id"]),
            ts=str(d["ts"]),
            persona_id=str(d["persona_id"]),
            day_index=int(d["day_index"]),
            duration_s=float(d["duration_s"]),
            activity=Activity(d["activity"]),
            state=State(d["state"]),
            multimodal_hint=MultimodalHint.from_dict(d["multimodal_hint"]),
            description=str(d.get("description", "")),
            meta=dict(d.get("meta", {})),
        )


# ---------- Script 容器 ----------


@dataclass(frozen=True)
class Script:
    """单画像 × 单天的完整剧本。

    存盘格式：JSON 文件，结构 ``{persona_id, day_index, generated_at, events: [...]}``。
    """

    persona_id: str
    day_index: int
    generated_at: str           # ISO8601
    events: tuple[ScriptEvent, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "day_index": self.day_index,
            "generated_at": self.generated_at,
            "events": [e.to_jsonl_dict() for e in self.events],
            "meta": dict(self.meta),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Script":
        return cls(
            persona_id=str(d["persona_id"]),
            day_index=int(d["day_index"]),
            generated_at=str(d["generated_at"]),
            events=tuple(ScriptEvent.from_jsonl_dict(e) for e in d.get("events", [])),
            meta=dict(d.get("meta", {})),
        )

    @classmethod
    def from_json(cls, s: str) -> "Script":
        return cls.from_dict(json.loads(s))


# ---------- 校验常量 ----------

DURATION_MIN_S = 5.0
DURATION_MAX_S = 3600.0


__all__ = [
    "Activity",
    "State",
    "MultimodalHint",
    "ScriptEvent",
    "Script",
    "DURATION_MIN_S",
    "DURATION_MAX_S",
]
