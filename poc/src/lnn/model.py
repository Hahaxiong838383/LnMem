"""CfC-based Day 5 LNN models with shared persona conditioning."""
from __future__ import annotations

import torch
from ncps.torch import CfC
from torch import Tensor, nn

from poc.src.lnn.contracts import LNNModelError, LNNTrainConfig
from poc.src.lnn.personalization import DefaultPersonalizationAdapter


class _PersonaConditionedCfC(nn.Module):
    """Run CfC step-by-step so tau bias and FiLM can modulate each frame."""

    def __init__(
        self,
        config: LNNTrainConfig,
        personalization: DefaultPersonalizationAdapter,
        out_dim: int,
    ) -> None:
        super().__init__()
        if config.cfc_mode not in {"default", "pure", "no_gate"}:
            raise LNNModelError(f"unsupported CfC mode: {config.cfc_mode}")
        self.config = config
        self.personalization = personalization
        self.hidden_dim = config.hidden_dim
        self.input_projection: nn.Module
        if config.feature_dim == config.hidden_dim:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(config.feature_dim, config.hidden_dim)
        self.backbone = CfC(
            config.hidden_dim,
            config.hidden_dim,
            return_sequences=True,
            batch_first=True,
            mode=config.cfc_mode,
        )
        self.readout = nn.Linear(config.hidden_dim, out_dim)

    def _timespans_from_tau(self, tau_bias: Tensor) -> Tensor:
        import torch.nn.functional as F
        return 0.5 + F.softplus(tau_bias)

    def _apply_film(self, hidden: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        scale = 1.0 + 0.1 * torch.tanh(gamma)
        shift = 0.1 * torch.tanh(beta)
        return hidden * scale + shift

    def forward(self, x: Tensor, persona_idx: Tensor) -> Tensor:
        """Return per-frame logits for the requested task."""
        if x.ndim != 3:
            raise LNNModelError(f"expected x shape (B, T, F), got {tuple(x.shape)}")
        if persona_idx.ndim != 1 or persona_idx.shape[0] != x.shape[0]:
            raise LNNModelError(
                f"expected persona_idx shape ({x.shape[0]},), got {tuple(persona_idx.shape)}",
            )
        if x.shape[-1] != self.config.feature_dim:
            raise LNNModelError(
                f"expected feature_dim={self.config.feature_dim}, got {x.shape[-1]}",
            )

        features = self.input_projection(x.float())
        persona_ids = persona_idx.to(device=features.device, dtype=torch.long)
        tau_bias = self.personalization.get_tau_bias(persona_ids).to(features.device, features.dtype)
        gamma, beta = self.personalization.get_film_params(persona_ids)
        gamma = gamma.to(features.device, features.dtype)
        beta = beta.to(features.device, features.dtype)
        timespans = self._timespans_from_tau(tau_bias)

        batch, steps, _ = features.shape
        hidden = torch.zeros(batch, self.hidden_dim, device=features.device, dtype=features.dtype)
        outputs: list[Tensor] = []
        for step in range(steps):
            hidden, _ = self.backbone.rnn_cell(features[:, step], hidden, timespans)
            hidden = self._apply_film(hidden, gamma, beta)
            outputs.append(self.readout(hidden))
        return torch.stack(outputs, dim=1)


class StateRecognitionLNN(_PersonaConditionedCfC):
    """Predict one of eight scripted states for every frame."""

    def __init__(self, config: LNNTrainConfig, personalization: DefaultPersonalizationAdapter) -> None:
        super().__init__(config, personalization, out_dim=config.n_state_classes)


class BoundaryDetectionLNN(_PersonaConditionedCfC):
    """Predict event-boundary logits for every frame."""

    def __init__(self, config: LNNTrainConfig, personalization: DefaultPersonalizationAdapter) -> None:
        super().__init__(config, personalization, out_dim=1)


class DefaultLNNFactory:
    """Create state and boundary models that share one personalization adapter."""

    def __init__(self) -> None:
        self._adapter: DefaultPersonalizationAdapter | None = None
        self._config: LNNTrainConfig | None = None

    def _shared_adapter(self, config: LNNTrainConfig) -> DefaultPersonalizationAdapter:
        if self._adapter is None or self._config != config:
            self._adapter = DefaultPersonalizationAdapter(config)
            self._config = config
        return self._adapter

    def create_state_model(self, config: LNNTrainConfig) -> StateRecognitionLNN:
        """Return the Day 5 state-recognition model."""
        return StateRecognitionLNN(config, self._shared_adapter(config))

    def create_boundary_model(self, config: LNNTrainConfig) -> BoundaryDetectionLNN:
        """Return the Day 5 boundary-detection model."""
        return BoundaryDetectionLNN(config, self._shared_adapter(config))


__all__ = [
    "StateRecognitionLNN",
    "BoundaryDetectionLNN",
    "DefaultLNNFactory",
]
