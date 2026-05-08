"""Day 5-6 LNN 训练公共契约 — agent team 跨模块的唯一共享 API 表面。

设计目标
--------
1. **接口先行**：cc 主线先冻结这份 contracts，再并行派 codex（model + personalization）+
   gemini（dataset + viz）。各模块内部表示自由，对外只通过本文件的类型交换。
2. **immutability**：所有 dataclass `frozen=True`，符合 ~/.claude/rules/coding-style.md。
3. **零 PyTorch 依赖**：本文件不导入 torch（`Module` 用 ``Any`` 占位），避免 import 链。
4. **PyTorch 类型由实现端 import**：model.py / dataset.py 自己 import torch，本文件不强绑定。

下游消费方
----------
- ``train.py`` 主线训练入口
- ``test_lnn_contracts.py`` Protocol 兼容性单测

参考
----
- contracts/README.md（主分支 M0 W2 接口先行哲学）
- feedback_codex_multi_field_contract_gate.md（多 lane 字段漂移踩坑）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    # 仅用于类型提示，运行期不依赖 torch
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset


# ---------- 训练配置 ----------


@dataclass(frozen=True)
class LNNTrainConfig:
    """LNN 预训练所有超参——一份配置贯穿 dataset/model/train/viz。"""

    # 数据维度（与 fusion.py FEATURE_DIM=18 + ScriptEvent.State 8 类对齐）
    # Day 6: 12→18，新增 6 维 AU 表情特征（smile/cheek/brow/lip/jaw/mouth）
    feature_dim: int = 18
    n_state_classes: int = 8
    n_personas: int = 30           # vboss_001..vboss_030
    persona_embed_dim: int = 8     # FiLM 适配器输入维度

    # 序列化窗口（1Hz × seconds = frames per window）
    sequence_length: int = 60      # 60s 窗口（每窗口 60 帧）
    sequence_stride: int = 30      # 30s 步幅，50% overlap
    sequence_min_frames: int = 30  # 短于此的窗口丢弃

    # 模型超参
    hidden_dim: int = 64
    cfc_mode: str = "default"      # default / pure / no_gate（ncps CfC 三种）
    use_tau_bias: bool = True
    use_film: bool = True

    # 训练超参
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 30
    train_val_split: float = 0.8   # 80% train / 20% val（按 (persona,day) tuple 切分）
    grad_clip_norm: float = 1.0

    # 边界检测（Day 5 第二任务，可选）
    boundary_pos_weight: float = 5.0  # 边界稀疏，正样本加权

    # 设备 & 复现
    device: str = "auto"           # auto / cpu / mps / cuda
    seed: int = 42

    # 数据路径
    signals_dir: Path = field(default_factory=lambda: Path("poc/data/synthetic/llm_v2"))
    out_dir: Path = field(default_factory=lambda: Path("poc/data/checkpoints/day5"))

    def resolved_device(self) -> str:
        """auto → mps（macOS）/ cuda / cpu 顺序尝试，否则原样返回。"""
        if self.device != "auto":
            return self.device
        try:
            import torch  # noqa: WPS433
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"


# ---------- Checkpoint ----------


@dataclass(frozen=True)
class EpochMetric:
    """单个 epoch 的训练 + 验证指标。"""

    epoch: int
    train_loss: float
    val_loss: float
    train_state_acc: float
    val_state_acc: float
    val_boundary_f1: float = 0.0
    notes: str = ""


@dataclass(frozen=True)
class LNNCheckpoint:
    """训练结果——可序列化（state_dict + 元数据），与具体模型解耦。"""

    config: LNNTrainConfig
    persona_to_idx: dict[str, int]
    state_model_state_dict: dict[str, Any]   # torch state_dict（运行期为 OrderedDict[str, Tensor]）
    boundary_model_state_dict: dict[str, Any] | None = None
    epoch_metrics: tuple[EpochMetric, ...] = field(default_factory=tuple)
    final_state_acc: float = 0.0
    final_boundary_f1: float = 0.0


# ---------- Protocol：gemini 实现 ----------


class SignalsLoader(Protocol):
    """gemini 实现 — 从 signals_*.jsonl 读取并切窗，生成 train/val DataLoader。

    输出契约：
    - 每个 batch tensor 形状: ``(B, T, feature_dim)``，dtype=float32
    - 每个 batch label state 形状: ``(B, T)``，dtype=long，值 ∈ [0, n_state_classes)
    - 每个 batch persona_idx 形状: ``(B,)``，dtype=long，值 ∈ [0, n_personas)
    - 每个 batch boundary 形状: ``(B, T)``，dtype=float32，值 ∈ {0, 1}（事件起点 = 1）
    """

    def build_loaders(
        self,
        config: LNNTrainConfig,
    ) -> tuple["DataLoader", "DataLoader", dict[str, int]]:
        """Returns (train_loader, val_loader, persona_to_idx)."""
        ...


# ---------- Protocol：codex 实现 ----------


class LNNModelFactory(Protocol):
    """codex 实现 — 创建 LNN 模型实例。

    StateRecognitionLNN 输入 ``(B, T, feature_dim) + (B,) persona_idx``，
    输出 ``(B, T, n_state_classes)`` logits。

    BoundaryDetectionLNN 同输入，输出 ``(B, T, 1)`` logits（sigmoid 后为边界概率）。
    """

    def create_state_model(self, config: LNNTrainConfig) -> "nn.Module":
        """返回 StateRecognitionLNN 实例（forward(x, persona_idx) -> logits）。"""
        ...

    def create_boundary_model(self, config: LNNTrainConfig) -> "nn.Module":
        """返回 BoundaryDetectionLNN 实例（forward(x, persona_idx) -> logits）。"""
        ...


class PersonalizationAdapter(Protocol):
    """codex 实现 — τ 偏置 + FiLM 个性化适配器。

    每个 persona 一个 embedding 向量，通过 ``persona_embedding(persona_idx)`` 拿到，
    再投影成 (a) τ 偏置加到 CfC cell 时间常数；(b) FiLM gamma/beta 调制 hidden state。
    """

    def get_tau_bias(self, persona_idx: "torch.Tensor") -> "torch.Tensor":
        """Returns ``(B, hidden_dim)`` tau offset tensor。"""
        ...

    def get_film_params(self, persona_idx: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Returns ``(gamma, beta)``，各为 ``(B, hidden_dim)``。"""
        ...


# ---------- Protocol：gemini 实现 ----------


class TrainingViz(Protocol):
    """gemini 实现 — 训练后可视化。

    所有方法落盘 PNG，不返回图片对象（避免 matplotlib 全局状态污染）。
    """

    def plot_loss_curve(
        self,
        epoch_metrics: tuple[EpochMetric, ...],
        out_path: Path,
    ) -> None:
        """train/val loss + state accuracy 双 Y 轴图。"""
        ...

    def plot_tsne_personalization(
        self,
        persona_embedding_weights: "torch.Tensor",  # (n_personas, persona_embed_dim)
        persona_to_idx: dict[str, int],
        out_path: Path,
    ) -> None:
        """30 画像 persona embedding t-SNE 2D 散点图。"""
        ...


# ---------- 异常 ----------


class LNNDataError(ValueError):
    """数据加载/切窗错误（gemini dataset.py 抛出）。"""


class LNNModelError(RuntimeError):
    """模型构建/forward 错误（codex model.py 抛出）。"""


__all__ = [
    "LNNTrainConfig",
    "EpochMetric",
    "LNNCheckpoint",
    "SignalsLoader",
    "LNNModelFactory",
    "PersonalizationAdapter",
    "TrainingViz",
    "LNNDataError",
    "LNNModelError",
]
