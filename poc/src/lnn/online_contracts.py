"""Phase 2 在线学习契约 — frozen dataclass，零 PyTorch 依赖。

设计目标
--------
1. **接口先行**：本文件冻结 reward 标签集 / FeedbackEvent / OnlineAdapterConfig，
   下游 `online_adapter.py`（torch 实现）+ `feedback_dataset.py`（pure python）
   各自 import，不互相依赖。
2. **immutability**（~/.claude/rules/coding-style.md）：所有 dataclass `frozen=True`。
3. **LNN-FIRST P-1**：reward → τ-bias delta 的转换在 adapter 实现端，本文件仅定义类型。

下游消费方
----------
- `online_adapter.py` (torch) — OnlineLoRAAdapter
- `feedback_dataset.py` (pure python) — FeedbackDataset
- `test_online_adapter.py` / `test_feedback_dataset.py` — 单测
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------- Reward 信号映射 ----------

# user_response.label 取值（与 daemon V0.2 弹窗 channel 对齐）
FeedbackLabel = Literal["match", "mismatch", "skip", "input"]

# Reward 信号（Phase 2 在线训练 loss 权重）
# match    → +1.0  当前 τ-bias 正确，强化（最大化 lnn_score 当前预测分布）
# mismatch → -1.0  τ-bias 错误，下次同帧 lnn_score 应下降
# skip     →  0.0  中性，不入 batch
# input    → stub 0.0  free-form text NLU 待 Phase 2.x 接入
REWARD_MAP: dict[str, float] = {
    "match": 1.0,
    "mismatch": -1.0,
    "skip": 0.0,
    "input": 0.0,
}


# ---------- 反馈事件 ----------


@dataclass(frozen=True)
class FeedbackEvent:
    """单条用户反馈样本（来自 ~/.lnmem/feedback/queries_*.jsonl）。

    Attributes
    ----------
    ts : str
        ISO8601 UTC 时间戳（事件触发时间）
    frame_id : int
        daemon 内部 frame 序号（与 signals jsonl 对齐）
    reason : str
        触发原因（low_vlm_conf / brow_burst / smile_burst / ...）
    signal : dict[str, float]
        当帧 4-key 信号（brow_furrow / smile_intensity / eye_closure / cheek_raise）
    vlm_label : dict | None
        VLM 标注（可能 None，FB-09 修复后无脸帧不再写 jsonl）
    label : FeedbackLabel
        用户响应标签
    true_state : str | None
        用户额外指定的真实状态（仅 mismatch 时可能填）
    note : str | None
        free-form 文本（input label 时主要载体）
    latency_s : float
        弹窗到响应的延迟秒数
    session_id : str | None
        Phase 1.V0.2 起加，UUID4 字符串
    event_uuid : str | None
        Phase 1.V0.2 起加，单条事件唯一 ID
    """

    ts: str
    frame_id: int
    reason: str
    signal: dict[str, float]
    vlm_label: dict | None
    label: FeedbackLabel
    true_state: str | None = None
    note: str | None = None
    latency_s: float = 0.0
    session_id: str | None = None
    event_uuid: str | None = None

    def reward(self) -> float:
        """Map self.label → scalar reward（REWARD_MAP）."""
        return REWARD_MAP.get(self.label, 0.0)


# ---------- 在线适配器配置 ----------


@dataclass(frozen=True)
class OnlineAdapterConfig:
    """OnlineLoRAAdapter 超参 — Phase 2 在线 τ-bias 漂移。

    设计原则（LNN-FIRST P-1 合规）
    -----------------------------
    - 仅在 τ 维度加 per-persona delta（不动 FiLM、不动 backbone）
    - delta 默认初始化 0（包装首日行为 == base adapter）
    - 触发条件：累积 mismatch ≥ trigger_mismatch_threshold（默认 10）
    - 微调粒度：每 trigger 跑 max_steps 步 SGD（默认 50）
    """

    # 触发条件
    trigger_mismatch_threshold: int = 10  # mismatch 累积条数，达此触发一次微调
    min_total_feedback: int = 10           # 总反馈条数最低门槛（不可妥协）

    # 微调超参
    delta_lr: float = 1e-3
    delta_weight_decay: float = 1e-4       # L2 防 delta 漂移过远
    max_steps: int = 50                    # 单次触发训练步数
    batch_size: int = 16

    # 数值稳定
    delta_clip_norm: float = 0.5           # delta L2 范数上限（防爆）

    # IO
    feedback_dir: Path = field(
        default_factory=lambda: Path.home() / ".lnmem" / "feedback"
    )
    base_ckpt_path: Path = field(
        default_factory=lambda: Path("poc/data/checkpoints/day7_v2/checkpoint_day5.pt")
    )
    output_dir: Path = field(
        default_factory=lambda: Path("poc/data/checkpoints/day12_online_lora")
    )

    # 复现
    seed: int = 42


# ---------- 异常 ----------


class OnlineAdapterError(RuntimeError):
    """OnlineLoRAAdapter 加载/微调错误。"""


class FeedbackDataError(ValueError):
    """FeedbackDataset 解析/过滤错误。"""


__all__ = [
    "FeedbackLabel",
    "REWARD_MAP",
    "FeedbackEvent",
    "OnlineAdapterConfig",
    "OnlineAdapterError",
    "FeedbackDataError",
]
