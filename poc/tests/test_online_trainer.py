"""TDD: poc/src/lnn/online_trainer.py — Phase 2 在线训练循环。

覆盖
----
1. trigger_if_ready：min_total_feedback gate / trigger_mismatch_threshold gate / 计数完整
2. compute_reward_loss：match=+CE / mismatch=-CE / skip=0 贡献 / 形状错误抛
3. fit_step：
   - loss 有限 + tau_drift 改变（base 不变）
   - **关键 sanity**：mismatch 反馈反传后同帧 logits[label] 严格下降（LNN-FIRST P-1 实测）
   - 仅 tau_drift 有 grad（base 全冻结）
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.model import StateRecognitionLNN
from poc.src.lnn.online_adapter import OnlineLoRAAdapter
from poc.src.lnn.online_contracts import OnlineAdapterConfig, OnlineAdapterError
from poc.src.lnn.personalization import DefaultPersonalizationAdapter
from poc.src.lnn.online_trainer import (
    compute_reward_loss,
    fit_step,
    trigger_if_ready,
)
from poc.src.lnn.personalization import DefaultPersonalizationAdapter


# ---------- Fixtures ----------


@pytest.fixture
def base_config() -> LNNTrainConfig:
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
def online_cfg(tmp_path: Path) -> OnlineAdapterConfig:
    return OnlineAdapterConfig(
        trigger_mismatch_threshold=10,
        min_total_feedback=10,
        delta_lr=1e-1,  # 大一点便于看到 step effect
        delta_clip_norm=1.0,
        feedback_dir=tmp_path / "feedback",
        base_ckpt_path=tmp_path / "ckpt.pt",
        output_dir=tmp_path / "out",
        seed=42,
    )


@pytest.fixture
def setup(base_config: LNNTrainConfig, online_cfg: OnlineAdapterConfig):
    """Tiny end-to-end setup: state_model + online_adapter + optim。"""
    torch.manual_seed(0)
    base = DefaultPersonalizationAdapter(base_config)
    online = OnlineLoRAAdapter(base, online_cfg)
    model = StateRecognitionLNN(base_config, online)  # type: ignore[arg-type]
    optim = torch.optim.SGD([online.tau_drift], lr=online_cfg.delta_lr)
    return model, online, optim


def _write_feedback_jsonl(
    feedback_dir: Path,
    rows: list[dict],
    *,
    date: str = "2026-05-10",
) -> None:
    feedback_dir.mkdir(parents=True, exist_ok=True)
    path = feedback_dir / f"queries_{date}.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _row(label: str, frame_id: int = 0) -> dict:
    return {
        "ts": f"2026-05-10T00:00:{frame_id:02d}Z",
        "frame_id": frame_id,
        "reason": "brow_burst",
        "signal": {},
        "vlm_label": None,
        "user_response": {"label": label, "true_state": None, "note": None},
        "latency_s": 0.0,
    }


# ---------- trigger_if_ready ----------


class TestTriggerIfReady:
    def test_below_min_total_returns_false(
        self, online_cfg: OnlineAdapterConfig
    ) -> None:
        # 9 mismatch < min_total_feedback=10
        rows = [_row("mismatch", i) for i in range(9)]
        _write_feedback_jsonl(online_cfg.feedback_dir, rows)
        ready, counts = trigger_if_ready(online_cfg.feedback_dir, online_cfg)
        assert not ready
        assert counts["total"] == 9
        assert counts["mismatch"] == 9

    def test_total_ok_but_mismatch_short(
        self, online_cfg: OnlineAdapterConfig
    ) -> None:
        # 12 total, 5 mismatch < trigger_mismatch_threshold=10
        rows = [_row("mismatch", i) for i in range(5)] + [
            _row("match", i + 5) for i in range(7)
        ]
        _write_feedback_jsonl(online_cfg.feedback_dir, rows)
        ready, counts = trigger_if_ready(online_cfg.feedback_dir, online_cfg)
        assert not ready
        assert counts["total"] == 12
        assert counts["mismatch"] == 5

    def test_both_thresholds_met_triggers(
        self, online_cfg: OnlineAdapterConfig
    ) -> None:
        rows = [_row("mismatch", i) for i in range(10)] + [
            _row("match", i + 10) for i in range(2)
        ]
        _write_feedback_jsonl(online_cfg.feedback_dir, rows)
        ready, counts = trigger_if_ready(online_cfg.feedback_dir, online_cfg)
        assert ready
        assert counts["mismatch"] == 10

    def test_failsoft_when_dir_missing(
        self, online_cfg: OnlineAdapterConfig
    ) -> None:
        """F3-03 (codex): feedback_dir 不存在 → 返 (False, zero counts)，不抛错。"""
        # 未写入任何 jsonl，dir 不存在
        ready, counts = trigger_if_ready(online_cfg.feedback_dir, online_cfg)
        assert not ready
        assert counts["total"] == 0
        assert counts["match"] == 0
        assert counts["mismatch"] == 0

    def test_counts_include_zeros_for_missing_labels(
        self, online_cfg: OnlineAdapterConfig
    ) -> None:
        rows = [_row("match", i) for i in range(3)]
        _write_feedback_jsonl(online_cfg.feedback_dir, rows)
        _, counts = trigger_if_ready(online_cfg.feedback_dir, online_cfg)
        assert counts["match"] == 3
        assert counts["mismatch"] == 0
        assert counts["skip"] == 0
        assert counts["input"] == 0
        assert counts["total"] == 3


# ---------- compute_reward_loss ----------


class TestComputeRewardLoss:
    def test_match_reward_equals_standard_ce(self) -> None:
        """reward=+1 时 loss 等于标准 NLLLoss(log_softmax(logits), labels)."""
        logits = torch.zeros(1, 3)  # uniform
        labels = torch.tensor([0])
        rewards = torch.tensor([1.0])
        loss = compute_reward_loss(logits, labels, rewards)
        # uniform: log p(any) = log(1/3); loss = -mean(1.0 * log(1/3)) = +log(3)
        expected = -torch.log(torch.tensor(1.0 / 3.0))
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_mismatch_reward_negates_ce(self) -> None:
        """reward=-1 时 loss 等于 +log p(label)（取相反符号），用于 push down。"""
        logits = torch.zeros(1, 3)
        labels = torch.tensor([0])
        rewards = torch.tensor([-1.0])
        loss = compute_reward_loss(logits, labels, rewards)
        # -mean(-1 * log(1/3)) = -mean(log(3)) ≈ -1.0986
        expected = torch.log(torch.tensor(1.0 / 3.0))  # negative
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_skip_reward_zero_contribution(self) -> None:
        """reward=0 样本对 mean loss 贡献 0（但仍占一份分母）。"""
        logits = torch.zeros(2, 3)
        labels = torch.tensor([0, 1])
        rewards_only_first = torch.tensor([1.0, 0.0])
        rewards_both = torch.tensor([1.0, 1.0])
        loss_only_first = compute_reward_loss(logits, labels, rewards_only_first)
        loss_both = compute_reward_loss(logits, labels, rewards_both)
        # uniform 时单样本 loss = log(3)；mean=log(3)/2 vs log(3)；后者大
        assert loss_only_first < loss_both

    def test_shape_error_logits_wrong_dim(self) -> None:
        with pytest.raises(OnlineAdapterError):
            compute_reward_loss(
                torch.zeros(2, 3, 4), torch.zeros(2, dtype=torch.long), torch.zeros(2)
            )

    def test_shape_error_label_mismatch(self) -> None:
        with pytest.raises(OnlineAdapterError):
            compute_reward_loss(
                torch.zeros(2, 3), torch.zeros(3, dtype=torch.long), torch.zeros(2)
            )

    def test_shape_error_reward_mismatch(self) -> None:
        with pytest.raises(OnlineAdapterError):
            compute_reward_loss(
                torch.zeros(2, 3), torch.zeros(2, dtype=torch.long), torch.zeros(3)
            )


# ---------- fit_step ----------


class TestFitStep:
    def test_returns_finite_loss(self, setup, base_config, online_cfg) -> None:
        model, online, optim = setup
        x = torch.randn(2, base_config.sequence_length, base_config.feature_dim)
        idx = torch.tensor([0, 1], dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)
        rewards = torch.tensor([1.0, -1.0])
        metrics = fit_step(
            model, online, optim,
            x=x, persona_idx=idx, labels=labels, rewards=rewards,
            config=online_cfg,
        )
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert metrics["batch_size"] == 2

    def test_only_tau_drift_changes(self, setup, base_config, online_cfg) -> None:
        """fit_step 后 base.* 参数不变，仅 tau_drift 变。"""
        model, online, optim = setup

        # Snapshot all base parameters
        base_snap = {
            name: p.detach().clone()
            for name, p in online.base.named_parameters()
        }
        drift_before = online.tau_drift.detach().clone()

        x = torch.randn(2, base_config.sequence_length, base_config.feature_dim)
        idx = torch.tensor([0, 1], dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)
        rewards = torch.tensor([1.0, -1.0])  # 至少一个非零让 drift 动
        fit_step(
            model, online, optim,
            x=x, persona_idx=idx, labels=labels, rewards=rewards,
            config=online_cfg,
        )

        # Base 全部参数不变
        for name, p in online.base.named_parameters():
            assert torch.allclose(p.detach(), base_snap[name]), (
                f"base.{name} 不应改变（应被冻结）"
            )
        # tau_drift 应变（即使值小）
        assert not torch.allclose(online.tau_drift.detach(), drift_before), (
            "tau_drift 应被 SGD 更新"
        )

    def test_mismatch_reward_pushes_down_target_logit(
        self, setup, base_config, online_cfg
    ) -> None:
        """**关键 LNN-FIRST sanity**：mismatch reward 反传后，同帧 logits[label] 严格下降。

        语义：用户说"模型预测错了 label=0"，反传后下次同帧 model(x)[label=0] 应变小。
        """
        model, online, optim = setup
        x = torch.randn(1, base_config.sequence_length, base_config.feature_dim)
        idx = torch.tensor([0], dtype=torch.long)
        target_label = torch.tensor([0], dtype=torch.long)

        # baseline: 无训练，记录 logits[0] 当前帧最后一步
        model.eval()
        with torch.no_grad():
            baseline_logits = model(x, idx)[:, -1, :]  # (1, n_classes)
        baseline_logit_0 = float(baseline_logits[0, 0].item())

        # 训练：mismatch (reward=-1) on label=0
        rewards = torch.tensor([-1.0])
        # 多跑几步累积变化（lr=0.1 单步可能太小）
        for _ in range(5):
            fit_step(
                model, online, optim,
                x=x, persona_idx=idx, labels=target_label, rewards=rewards,
                config=online_cfg,
            )

        # 再 forward 一次
        model.eval()
        with torch.no_grad():
            after_logits = model(x, idx)[:, -1, :]
        after_logit_0 = float(after_logits[0, 0].item())

        assert after_logit_0 < baseline_logit_0, (
            f"mismatch reward 反传后 logits[label=0] 应下降，"
            f"实际 baseline={baseline_logit_0:.4f} -> after={after_logit_0:.4f}"
        )

    def test_match_reward_pushes_up_target_logit(
        self, setup, base_config, online_cfg
    ) -> None:
        """对偶验证：match reward 反传后同帧 logits[label] 严格上升。"""
        model, online, optim = setup
        x = torch.randn(1, base_config.sequence_length, base_config.feature_dim)
        idx = torch.tensor([0], dtype=torch.long)
        target_label = torch.tensor([2], dtype=torch.long)

        model.eval()
        with torch.no_grad():
            baseline = model(x, idx)[:, -1, 2].item()

        rewards = torch.tensor([1.0])
        for _ in range(5):
            fit_step(
                model, online, optim,
                x=x, persona_idx=idx, labels=target_label, rewards=rewards,
                config=online_cfg,
            )

        model.eval()
        with torch.no_grad():
            after = model(x, idx)[:, -1, 2].item()

        assert after > baseline, (
            f"match reward 反传后 logits[label=2] 应上升，"
            f"实际 baseline={baseline:.4f} -> after={after:.4f}"
        )

    def test_skip_reward_no_drift_change(self, setup, base_config, online_cfg) -> None:
        """全 reward=0 时 tau_drift 应不变（loss=0 即 grad=0）。"""
        model, online, optim = setup
        drift_before = online.tau_drift.detach().clone()

        x = torch.randn(2, base_config.sequence_length, base_config.feature_dim)
        idx = torch.tensor([0, 1], dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)
        rewards = torch.zeros(2)
        fit_step(
            model, online, optim,
            x=x, persona_idx=idx, labels=labels, rewards=rewards,
            config=online_cfg,
        )
        assert torch.allclose(online.tau_drift.detach(), drift_before)

    def test_silent_no_op_guard_when_model_not_bound(
        self, setup, base_config, online_cfg
    ) -> None:
        """F2-01 (codex): fit_step 必须校验 model.personalization is online_adapter，
        否则 silent no-op 风险。错误绑定时抛 OnlineAdapterError。
        """
        model, online, optim = setup
        # 故意构造一个未替换 personalization 的模型
        torch.manual_seed(0)
        rogue_base = DefaultPersonalizationAdapter(base_config)
        rogue_model = StateRecognitionLNN(base_config, rogue_base)

        x = torch.randn(1, base_config.sequence_length, base_config.feature_dim)
        idx = torch.tensor([0], dtype=torch.long)
        labels = torch.tensor([0], dtype=torch.long)
        rewards = torch.tensor([1.0])

        with pytest.raises(OnlineAdapterError, match="not the passed online_adapter"):
            fit_step(
                rogue_model, online, optim,
                x=x, persona_idx=idx, labels=labels, rewards=rewards,
                config=online_cfg,
            )

    def test_drift_norm_reported_and_clipped(
        self, setup, base_config, online_cfg
    ) -> None:
        """fit_step 返回 drift_max_norm，且 ≤ delta_clip_norm（数值稳定）。"""
        model, online, optim = setup
        # Pre-charge tau_drift to large value
        with torch.no_grad():
            online.tau_drift.fill_(2.0)  # huge norm

        x = torch.randn(1, base_config.sequence_length, base_config.feature_dim)
        idx = torch.tensor([0], dtype=torch.long)
        labels = torch.tensor([0], dtype=torch.long)
        rewards = torch.tensor([1.0])
        metrics = fit_step(
            model, online, optim,
            x=x, persona_idx=idx, labels=labels, rewards=rewards,
            config=online_cfg,
        )
        # Returned max_norm is BEFORE clip; post-clip the actual norm should be ≤ clip_norm
        post_max_row_norm = float(
            online.tau_drift.norm(p=2, dim=1).max().item()
        )
        assert post_max_row_norm <= online_cfg.delta_clip_norm + 1e-6
        assert "drift_max_norm" in metrics
