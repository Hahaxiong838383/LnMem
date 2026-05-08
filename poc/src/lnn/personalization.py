"""Persona-conditioned tau and FiLM adapters for Day 5 LNN models."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from poc.src.lnn.contracts import LNNTrainConfig


class DefaultPersonalizationAdapter(nn.Module):
    """Project persona embeddings into tau bias and FiLM parameters."""

    def __init__(self, config: LNNTrainConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.embedding = nn.Embedding(config.n_personas, config.persona_embed_dim)
        self.tau_projection = nn.Linear(config.persona_embed_dim, config.hidden_dim)
        self.gamma_projection = nn.Linear(config.persona_embed_dim, config.hidden_dim)
        self.beta_projection = nn.Linear(config.persona_embed_dim, config.hidden_dim)

    def _persona_embedding(self, persona_idx: Tensor) -> Tensor:
        persona_ids = persona_idx.to(device=self.embedding.weight.device, dtype=torch.long)
        return self.embedding(persona_ids)

    def get_tau_bias(self, persona_idx: Tensor) -> Tensor:
        """Return ``(B, hidden_dim)`` tau offsets; zeros when disabled."""
        if not self.config.use_tau_bias:
            batch = persona_idx.shape[0]
            return torch.zeros(batch, self.hidden_dim, device=persona_idx.device)
        return self.tau_projection(self._persona_embedding(persona_idx))

    def get_film_params(self, persona_idx: Tensor) -> tuple[Tensor, Tensor]:
        """Return FiLM ``(gamma, beta)``; zeros when disabled."""
        if not self.config.use_film:
            batch = persona_idx.shape[0]
            zeros = torch.zeros(batch, self.hidden_dim, device=persona_idx.device)
            return zeros, zeros
        embedding = self._persona_embedding(persona_idx)
        return self.gamma_projection(embedding), self.beta_projection(embedding)


__all__ = ["DefaultPersonalizationAdapter"]
