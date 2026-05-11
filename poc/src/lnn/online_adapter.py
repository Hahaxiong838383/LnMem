"""OnlineLoRAAdapter — Phase 2 在线 τ-bias 漂移（LNN-FIRST P-1 合规）。

设计要点
--------
1. **包装** `DefaultPersonalizationAdapter`，base 完全冻结（requires_grad=False）。
2. **唯一可训练参数**：per-persona `tau_drift` 张量，形状 `(n_personas, hidden_dim)`，初始 0。
3. **加法叠加**：`get_tau_bias()` 返回 `base.get_tau_bias(idx) + tau_drift[idx]`，
   下游 `_PersonaConditionedCfC._timespans_from_tau` 仍走 `0.5 + softplus(...)` 强制正性。
4. **FiLM pass-through**：`get_film_params()` 直接转发 base，不学 FiLM 维度。
5. **LNN-FIRST P-1**：reward 反传只调 `tau_drift`，不动 backbone / readout / FiLM 投影。

接口契约（与 `DefaultPersonalizationAdapter` 完全兼容）
------------------------------------------------------
- `get_tau_bias(persona_idx) -> Tensor (B, hidden_dim)`
- `get_film_params(persona_idx) -> tuple[Tensor (B, hidden_dim), Tensor (B, hidden_dim)]`

使用方式
--------
```python
base = DefaultPersonalizationAdapter(config)
# load base ckpt state_dict 到 base
online = OnlineLoRAAdapter(base, online_config)
# 把 _PersonaConditionedCfC.personalization 替换为 online
state_model.personalization = online
boundary_model.personalization = online
# 训练时只 online.tau_drift 有 grad
optim = torch.optim.SGD([online.tau_drift], lr=online_config.delta_lr, ...)
```
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from poc.src.lnn.online_contracts import OnlineAdapterConfig, OnlineAdapterError
from poc.src.lnn.personalization import DefaultPersonalizationAdapter


class OnlineLoRAAdapter(nn.Module):
    """Wrap a frozen `DefaultPersonalizationAdapter` with reward-driven τ-bias delta.

    Parameters
    ----------
    base_adapter : DefaultPersonalizationAdapter
        已加载 base ckpt 权重的 adapter，本类 __init__ 内会 freeze 其所有参数。
    online_config : OnlineAdapterConfig
        在线学习超参（trigger 阈值、lr、clip_norm）。
    """

    def __init__(
        self,
        base_adapter: DefaultPersonalizationAdapter,
        online_config: OnlineAdapterConfig,
    ) -> None:
        super().__init__()
        if not isinstance(base_adapter, DefaultPersonalizationAdapter):
            raise OnlineAdapterError(
                f"base_adapter must be DefaultPersonalizationAdapter, got {type(base_adapter).__name__}"
            )
        self.base = base_adapter
        self.online_config = online_config

        # Freeze base parameters — LoRA 风格冻结底层
        for param in self.base.parameters():
            param.requires_grad_(False)

        # Trainable τ drift: per-persona offset
        n_personas = base_adapter.config.n_personas
        hidden_dim = base_adapter.hidden_dim
        # 初始 0 → 包装后行为完全等同 base
        self.tau_drift = nn.Parameter(
            torch.zeros(n_personas, hidden_dim),
            requires_grad=True,
        )

    # ---------- 兼容 PersonalizationAdapter Protocol ----------

    def get_tau_bias(self, persona_idx: Tensor) -> Tensor:
        """Return `base.get_tau_bias + tau_drift[persona_idx]`。

        与 base 形状一致：`(B, hidden_dim)`。
        """
        base_bias = self.base.get_tau_bias(persona_idx)
        # 索引 drift；persona_idx 必须 long
        idx = persona_idx.to(device=self.tau_drift.device, dtype=torch.long)
        delta = self.tau_drift.index_select(0, idx)
        return base_bias + delta.to(base_bias.device, base_bias.dtype)

    def get_film_params(self, persona_idx: Tensor) -> tuple[Tensor, Tensor]:
        """Pass-through to base — Phase 2 不学 FiLM。"""
        return self.base.get_film_params(persona_idx)

    # ---------- 数值稳定 ----------

    @torch.no_grad()
    def clip_drift_norm(self) -> float:
        """Clip per-persona `tau_drift` L2 norm to `online_config.delta_clip_norm`。

        Returns
        -------
        max_norm : float
            裁剪前的最大 per-persona L2 norm（监控用）
        """
        max_norm = float(self.online_config.delta_clip_norm)
        # per-persona 范数
        norms = self.tau_drift.norm(p=2, dim=1, keepdim=True)  # (n_personas, 1)
        max_observed = float(norms.max().item())
        # 缩放系数：超过 max_norm 的行按 max_norm/norm 缩放
        scale = torch.clamp(max_norm / norms.clamp(min=1e-12), max=1.0)
        self.tau_drift.mul_(scale)
        return max_observed

    # ---------- 序列化 ----------

    def drift_state_dict(self) -> dict[str, Tensor]:
        """Return only the trainable delta（exclude frozen base）— ckpt 紧凑保存。"""
        return {"tau_drift": self.tau_drift.detach().cpu().clone()}

    def load_drift_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load only the delta；base 由调用方独立 load。"""
        if "tau_drift" not in state:
            raise OnlineAdapterError(
                "drift state_dict must contain 'tau_drift' key"
            )
        loaded = state["tau_drift"]
        if loaded.shape != self.tau_drift.shape:
            raise OnlineAdapterError(
                f"tau_drift shape mismatch: expected {self.tau_drift.shape}, got {loaded.shape}"
            )
        with torch.no_grad():
            self.tau_drift.copy_(loaded.to(self.tau_drift.device, self.tau_drift.dtype))


__all__ = ["OnlineLoRAAdapter"]
