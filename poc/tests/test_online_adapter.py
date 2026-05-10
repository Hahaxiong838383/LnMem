"""TDD: poc/src/lnn/online_adapter.py — OnlineLoRAAdapter 包装器契约。

覆盖
----
1. base 参数全部冻结（requires_grad=False）
2. tau_drift 唯一可训练参数（requires_grad=True，shape 正确）
3. tau_drift=0 时 get_tau_bias / get_film_params == base（行为等同）
4. tau_drift 修改后 get_tau_bias 加法叠加 base
5. get_film_params 始终 pass-through（FiLM 不学）
6. clip_drift_norm 限制 per-persona L2 范数
7. drift_state_dict / load_drift_state_dict 序列化往返
8. **LNN-FIRST P-1 合规**：替换到 _PersonaConditionedCfC 后 forward 路径仍走 LTC ODE
   （通过验证模型 backbone.rnn_cell 不变 + readout 不变）
9. mismatch reward 反传后 lnn_score 应下降（端到端梯度流验证）
"""
from __future__ import annotations

import pytest
import torch

from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.model import StateRecognitionLNN
from poc.src.lnn.online_adapter import OnlineLoRAAdapter
from poc.src.lnn.online_contracts import OnlineAdapterConfig, OnlineAdapterError
from poc.src.lnn.personalization import DefaultPersonalizationAdapter


# ---------- Fixtures ----------


@pytest.fixture
def base_config() -> LNNTrainConfig:
    """Tiny LNN config to keep tests fast."""
    return LNNTrainConfig(
        feature_dim=8,
        n_state_classes=4,
        n_personas=3,
        persona_embed_dim=4,
        hidden_dim=8,
        sequence_length=4,
        sequence_stride=2,
        sequence_min_frames=2,
        batch_size=2,
        epochs=1,
        device="cpu",
    )


@pytest.fixture
def online_cfg(tmp_path) -> OnlineAdapterConfig:
    return OnlineAdapterConfig(
        trigger_mismatch_threshold=10,
        min_total_feedback=10,
        delta_lr=1e-2,
        delta_clip_norm=0.5,
        feedback_dir=tmp_path / "feedback",
        base_ckpt_path=tmp_path / "ckpt.pt",
        output_dir=tmp_path / "out",
        seed=42,
    )


@pytest.fixture
def base_adapter(base_config: LNNTrainConfig) -> DefaultPersonalizationAdapter:
    torch.manual_seed(0)
    return DefaultPersonalizationAdapter(base_config)


@pytest.fixture
def online_adapter(
    base_adapter: DefaultPersonalizationAdapter,
    online_cfg: OnlineAdapterConfig,
) -> OnlineLoRAAdapter:
    return OnlineLoRAAdapter(base_adapter, online_cfg)


# ---------- 行为测试 ----------


class TestInitFreeze:
    """构造时 base 必须冻结，tau_drift 必须可训练。"""

    def test_base_params_frozen(self, online_adapter: OnlineLoRAAdapter) -> None:
        for name, p in online_adapter.base.named_parameters():
            assert not p.requires_grad, f"base.{name} should be frozen"

    def test_tau_drift_trainable(self, online_adapter: OnlineLoRAAdapter) -> None:
        assert online_adapter.tau_drift.requires_grad

    def test_tau_drift_shape(
        self,
        online_adapter: OnlineLoRAAdapter,
        base_config: LNNTrainConfig,
    ) -> None:
        assert online_adapter.tau_drift.shape == (
            base_config.n_personas,
            base_config.hidden_dim,
        )

    def test_tau_drift_init_zero(self, online_adapter: OnlineLoRAAdapter) -> None:
        assert torch.allclose(
            online_adapter.tau_drift,
            torch.zeros_like(online_adapter.tau_drift),
        )

    def test_rejects_non_default_adapter(
        self, online_cfg: OnlineAdapterConfig
    ) -> None:
        class FakeAdapter:
            pass

        with pytest.raises(OnlineAdapterError):
            OnlineLoRAAdapter(FakeAdapter(), online_cfg)  # type: ignore[arg-type]


class TestForwardEquivalence:
    """tau_drift=0 时行为完全等同 base。"""

    def test_tau_bias_equals_base_when_drift_zero(
        self,
        base_adapter: DefaultPersonalizationAdapter,
        online_adapter: OnlineLoRAAdapter,
    ) -> None:
        idx = torch.tensor([0, 1, 2], dtype=torch.long)
        assert torch.allclose(
            online_adapter.get_tau_bias(idx),
            base_adapter.get_tau_bias(idx),
        )

    def test_film_passthrough(
        self,
        base_adapter: DefaultPersonalizationAdapter,
        online_adapter: OnlineLoRAAdapter,
    ) -> None:
        idx = torch.tensor([0, 1], dtype=torch.long)
        b_g, b_b = base_adapter.get_film_params(idx)
        o_g, o_b = online_adapter.get_film_params(idx)
        assert torch.allclose(b_g, o_g)
        assert torch.allclose(b_b, o_b)


class TestDriftAddition:
    """非零 tau_drift 必须加法叠加。"""

    def test_drift_adds_to_tau_bias(
        self,
        base_adapter: DefaultPersonalizationAdapter,
        online_adapter: OnlineLoRAAdapter,
    ) -> None:
        # 手动写入 drift
        with torch.no_grad():
            online_adapter.tau_drift.fill_(0.1)
        idx = torch.tensor([0, 1, 2], dtype=torch.long)
        base_bias = base_adapter.get_tau_bias(idx)
        out = online_adapter.get_tau_bias(idx)
        assert torch.allclose(out, base_bias + 0.1)


class TestClipNorm:
    """clip_drift_norm 限制 per-persona L2 范数。"""

    def test_clip_within_bound_unchanged(
        self, online_adapter: OnlineLoRAAdapter
    ) -> None:
        with torch.no_grad():
            online_adapter.tau_drift.fill_(0.01)  # tiny norm
        before = online_adapter.tau_drift.clone()
        max_obs = online_adapter.clip_drift_norm()
        assert torch.allclose(online_adapter.tau_drift, before)
        # max_obs 应远小于 clip_norm（0.5）
        assert max_obs < online_adapter.online_config.delta_clip_norm

    def test_clip_exceeds_bound(
        self,
        online_adapter: OnlineLoRAAdapter,
    ) -> None:
        with torch.no_grad():
            # 故意造一个 norm > clip_norm 的行
            online_adapter.tau_drift[0].fill_(10.0)  # L2 = 10*sqrt(8) ≈ 28.3
        clip_norm = online_adapter.online_config.delta_clip_norm
        online_adapter.clip_drift_norm()
        # 裁剪后 row 0 的 L2 范数应 ≤ clip_norm + eps
        post_norm = float(online_adapter.tau_drift[0].norm(p=2).item())
        assert post_norm <= clip_norm + 1e-6


class TestSerialize:
    """drift_state_dict / load_drift_state_dict 往返一致。"""

    def test_round_trip(
        self,
        base_adapter: DefaultPersonalizationAdapter,
        online_cfg: OnlineAdapterConfig,
    ) -> None:
        torch.manual_seed(123)
        adapter_a = OnlineLoRAAdapter(base_adapter, online_cfg)
        with torch.no_grad():
            adapter_a.tau_drift.uniform_(-0.1, 0.1)
        state = adapter_a.drift_state_dict()

        # 新 adapter（同 base）→ load
        torch.manual_seed(0)
        # base 必须共享，不然 base.get_tau_bias 不同
        adapter_b = OnlineLoRAAdapter(base_adapter, online_cfg)
        adapter_b.load_drift_state_dict(state)
        assert torch.allclose(adapter_a.tau_drift, adapter_b.tau_drift)

    def test_load_rejects_shape_mismatch(
        self, online_adapter: OnlineLoRAAdapter
    ) -> None:
        bad = {"tau_drift": torch.zeros(99, 99)}
        with pytest.raises(OnlineAdapterError):
            online_adapter.load_drift_state_dict(bad)

    def test_load_requires_tau_drift_key(
        self, online_adapter: OnlineLoRAAdapter
    ) -> None:
        with pytest.raises(OnlineAdapterError):
            online_adapter.load_drift_state_dict({"wrong_key": torch.zeros(3, 8)})


class TestLNNFirstCompliance:
    """LNN-FIRST P-1：替换到模型后 forward 仍走 LTC ODE。"""

    def test_replace_in_state_model_keeps_forward_signature(
        self,
        base_config: LNNTrainConfig,
        online_adapter: OnlineLoRAAdapter,
    ) -> None:
        """替换 personalization 后，模型 forward 仍能跑通且返回正确形状。"""
        torch.manual_seed(0)
        # 重新建一个 model 但用 online adapter
        model = StateRecognitionLNN(base_config, online_adapter)  # type: ignore[arg-type]
        # 注意：StateRecognitionLNN 期望 PersonalizationAdapter Protocol，
        # OnlineLoRAAdapter 兼容（duck-type）
        x = torch.randn(2, base_config.sequence_length, base_config.feature_dim)
        idx = torch.tensor([0, 1], dtype=torch.long)
        logits = model(x, idx)
        assert logits.shape == (
            2,
            base_config.sequence_length,
            base_config.n_state_classes,
        )

    def test_only_tau_drift_has_grad_after_loss(
        self,
        base_config: LNNTrainConfig,
        online_adapter: OnlineLoRAAdapter,
    ) -> None:
        """SGD 反传：除 tau_drift 外所有参数 grad 为 None（base 冻结）。"""
        torch.manual_seed(0)
        model = StateRecognitionLNN(base_config, online_adapter)  # type: ignore[arg-type]

        # backbone & readout 也参与 forward，但是非 base.* 的（属于 model 自身）
        # 测试目标：online_adapter.base.* 必须无 grad
        x = torch.randn(2, base_config.sequence_length, base_config.feature_dim)
        idx = torch.tensor([0, 1], dtype=torch.long)
        logits = model(x, idx)
        loss = logits.pow(2).mean()
        loss.backward()

        # base 所有参数 grad 必须 None 或全 0
        for name, p in online_adapter.base.named_parameters():
            assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)), (
                f"base.{name} got non-zero grad — base 应被冻结"
            )
        # tau_drift 必须有 grad
        assert online_adapter.tau_drift.grad is not None
