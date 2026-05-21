"""Tests for base_v2.1 DFAT helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "finetune_base_v2_1.py"
_SPEC = importlib.util.spec_from_file_location("finetune_base_v2_1", _SCRIPT)
assert _SPEC and _SPEC.loader
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

_BASE_V2 = Path("poc/data/checkpoints/base_v2/checkpoint_base_v2_best.pt")
_BASE_V1 = Path("poc/data/checkpoints/day12_with_self/checkpoint_day12.pt")


def test_backbone_frozen() -> None:
    """CfC backbone + input projection must be frozen and ratio <10%."""
    model = mod.build_model_from_v2_ckpt(_BASE_V2, _BASE_V1, persona_idx=30)
    stats = mod.freeze_backbone_for_dfat(model)
    assert stats["ratio"] < 0.10
    for name, param in model.named_parameters():
        if "backbone" in name or "input_projection" in name:
            assert not param.requires_grad, name


def test_persona_embedding_loaded_from_v1() -> None:
    """Row[30] should exactly match base_v1 vboss_self embedding."""
    model = mod.build_model_from_v2_ckpt(_BASE_V2, _BASE_V1, persona_idx=30)
    base_v1 = torch.load(_BASE_V1, map_location="cpu", weights_only=False)
    expected = base_v1["state_model"]["personalization.embedding.weight"][30]
    actual = model.cfc_block.personalization.embedding.weight[30].detach()
    assert torch.allclose(actual, expected)


def test_supervised_loss_with_cost_matrix() -> None:
    """Focused->negative should receive 5x the plain weighted CE."""
    logits = torch.tensor([[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([mod._class_index_from_state("focused")], dtype=torch.long)
    class_weights = torch.ones((9,), dtype=torch.float32)
    cost_matrix = mod.build_cost_matrix("cpu")
    weighted = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    penalized = mod.cost_sensitive_ce(logits, labels, class_weights, cost_matrix)
    assert torch.isclose(penalized, weighted.mean() * 5.0)
