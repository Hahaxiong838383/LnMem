"""Day 5 LNN model tests for CfC forward, backward, and personalization."""
from __future__ import annotations

from dataclasses import replace

import torch

from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.model import BoundaryDetectionLNN, DefaultLNNFactory, StateRecognitionLNN
from poc.src.lnn.personalization import DefaultPersonalizationAdapter


def _sample_batch(cfg: LNNTrainConfig, batch: int = 4, steps: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, steps, cfg.feature_dim)
    persona_idx = torch.arange(batch, dtype=torch.long) % cfg.n_personas
    return x, persona_idx


def test_factory_shares_one_personalization_adapter() -> None:
    """Factory should reuse one adapter across state and boundary models."""
    cfg = LNNTrainConfig()
    factory = DefaultLNNFactory()
    state_model = factory.create_state_model(cfg)
    boundary_model = factory.create_boundary_model(cfg)
    assert state_model.personalization is boundary_model.personalization


def test_state_model_forward_shape() -> None:
    """StateRecognitionLNN returns per-frame state logits."""
    cfg = LNNTrainConfig()
    model = StateRecognitionLNN(cfg, DefaultPersonalizationAdapter(cfg))
    x, persona_idx = _sample_batch(cfg)
    logits = model(x, persona_idx)
    assert logits.shape == (x.shape[0], x.shape[1], cfg.n_state_classes)


def test_boundary_model_forward_shape() -> None:
    """BoundaryDetectionLNN returns per-frame boundary logits."""
    cfg = LNNTrainConfig()
    model = BoundaryDetectionLNN(cfg, DefaultPersonalizationAdapter(cfg))
    x, persona_idx = _sample_batch(cfg)
    logits = model(x, persona_idx)
    assert logits.shape == (x.shape[0], x.shape[1], 1)


def test_backward_reaches_shared_personalization_parameters() -> None:
    """Loss backward should populate gradients on shared persona embeddings."""
    cfg = LNNTrainConfig()
    factory = DefaultLNNFactory()
    state_model = factory.create_state_model(cfg)
    boundary_model = factory.create_boundary_model(cfg)
    x, persona_idx = _sample_batch(cfg)
    loss = state_model(x, persona_idx).sum() + boundary_model(x, persona_idx).sum()
    loss.backward()
    grad = state_model.personalization.embedding.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad).item() > 0


def test_persona_changes_logits_when_personalization_enabled() -> None:
    """Different personas should change outputs for identical input."""
    cfg = LNNTrainConfig()
    model = StateRecognitionLNN(cfg, DefaultPersonalizationAdapter(cfg))
    x, _ = _sample_batch(cfg)
    out_a = model(x, torch.zeros(x.shape[0], dtype=torch.long))
    out_b = model(x, torch.ones(x.shape[0], dtype=torch.long))
    assert not torch.allclose(out_a, out_b)


def test_tau_bias_returns_zeros_when_disabled() -> None:
    """Tau branch should hard-disable to zero offsets."""
    cfg = replace(LNNTrainConfig(), use_tau_bias=False)
    adapter = DefaultPersonalizationAdapter(cfg)
    persona_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    tau_bias = adapter.get_tau_bias(persona_idx)
    assert tau_bias.shape == (3, cfg.hidden_dim)
    assert torch.allclose(tau_bias, torch.zeros_like(tau_bias))


def test_film_params_return_zeros_when_disabled() -> None:
    """FiLM branch should hard-disable to identity parameters."""
    cfg = replace(LNNTrainConfig(), use_film=False)
    adapter = DefaultPersonalizationAdapter(cfg)
    persona_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    gamma, beta = adapter.get_film_params(persona_idx)
    assert gamma.shape == (3, cfg.hidden_dim)
    assert beta.shape == (3, cfg.hidden_dim)
    assert torch.allclose(gamma, torch.zeros_like(gamma))
    assert torch.allclose(beta, torch.zeros_like(beta))


def test_outputs_match_when_all_personalization_is_disabled() -> None:
    """Same input should yield identical logits once tau and FiLM are both off."""
    cfg = replace(LNNTrainConfig(), use_tau_bias=False, use_film=False)
    model = StateRecognitionLNN(cfg, DefaultPersonalizationAdapter(cfg))
    x, _ = _sample_batch(cfg)
    out_a = model(x, torch.zeros(x.shape[0], dtype=torch.long))
    out_b = model(x, torch.ones(x.shape[0], dtype=torch.long))
    assert torch.allclose(out_a, out_b)
