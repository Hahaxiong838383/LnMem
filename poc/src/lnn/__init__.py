"""LnMem PoC Day 5-6 LNN 训练包。

模块边界（agent team 并行实现）：
- ``contracts.py`` (cc 主线): Protocol 接口 + LNNTrainConfig + LNNCheckpoint
- ``model.py``    (codex):    StateRecognitionLNN + BoundaryDetectionLNN（CfC + τ 偏置 + FiLM）
- ``personalization.py`` (codex): per-persona τ 偏置 + FiLM gamma/beta 适配器
- ``dataset.py``  (gemini):   SignalsDataset / SignalsLoader（jsonl → (T, 12) tensor + state label）
- ``viz.py``      (gemini):   plot_loss_curve + plot_tsne_personalization
- ``train.py``    (cc 主线): 端到端入口，集成上述模块
"""

from poc.src.lnn.intent_head import (
    INTENT_LABELS,
    INTENT_TO_INDEX,
    IntentClassifierHead,
    cost_sensitive_cross_entropy,
    default_intent_loss,
    dynamic_focal_loss,
)

__all__ = [
    "INTENT_LABELS",
    "INTENT_TO_INDEX",
    "IntentClassifierHead",
    "dynamic_focal_loss",
    "cost_sensitive_cross_entropy",
    "default_intent_loss",
]
