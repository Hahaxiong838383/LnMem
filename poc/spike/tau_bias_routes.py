"""γ-1.2 ncps τ-bias 三路径 spike

验证 spec v0.2.4 §2.1 + §9 #13：τ-bias 真进 ODE，不是 ncps 外层后处理「假 LNN-FIRST」。

三路径设计（基于 ncps 1.0.1 实测内部结构）：
- **Path A**: 用 `forward(timespans=...)` 外部时间步缩放 (per-persona timescale)
- **Path B**: hook `CfCCell.time_a.weight` + persona-conditioned bias (per-neuron τ-bias)
- **Path C**: 占位（手写 CfCCell 单步隐式欧拉），γ-2 必要时才写

验收 (spec §11.1 preflight #12)：
- baseline vs 各 path 的 hidden state 差异显著性 paired t-test p < 0.05
- 不显著则说明该 path 对 ODE 内核没真实影响 → 走 Path C 兜底

Usage:
    PYTHONPATH=. .venv-poc-fresh/bin/python poc/spike/tau_bias_routes.py
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable

import torch
import torch.nn as nn
from scipy import stats

from ncps.torch import CfC


# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SpikeConfig:
    """spike 配置，对应 spec §2.1 / §2.2."""
    input_dim: int = 52       # 24 AU + 20 槽 + 5 intent + 2 时间 + 6 音频占位
    hidden_dim: int = 64
    seq_len: int = 30         # 1s @ 30fps
    batch_size: int = 16
    num_personas: int = 4     # spike: 4 个虚拟 persona
    seed: int = 42

    @property
    def expected_T_seconds(self) -> float:
        return self.seq_len / 30.0


# -----------------------------------------------------------------------------
# Path A: forward(timespans=) 标量 dt 缩放（验证 ncps 是否真接受外部 dt 控制）
# -----------------------------------------------------------------------------
def path_a_status() -> str:
    """Path A: ncps timespans API 接口实测有 broadcast bug（5/21 23:30 实测 1.0.1）

    根因：
        ncps cfc.py:173-178 取 `ts = timespans[:, t].squeeze()`，
        ts shape 变成 (batch,) 1-D，
        但 cfc_cell.py:162 `t_a * ts + t_b` 里 t_a 是 (batch, hidden=64)，
        PyTorch 右对齐 broadcast 失败 → RuntimeError size mismatch.

    工程含义：
        - Path A 标量 dt 也不可用（除非 batch_size=1）
        - per-sample 时间步外部注入彻底走不通
        - 必须走 Path B（hook time_a）或 Path C（手写 CfCCell）才能让 τ 真进 ODE

    Phase γ 决策：放弃 Path A，主用 Path B；若 Path B 也无法稳定通过
    p<0.05 显著性，γ-2 启动 Path C 手写 cell（+1 天工时，spec §8 已注明）。

    （未来修复）若 ncps 上游修了 squeeze bug 或我们提 PR，Path A 重新可用。
    """
    return "ABANDONED · ncps 1.0.1 timespans broadcast bug 实测确认"


# -----------------------------------------------------------------------------
# Path B: hook time_a.weight 加 persona-conditioned bias
# -----------------------------------------------------------------------------
class PathBWrapper(nn.Module):
    """Path B: per-neuron τ-bias via time_a hook

    spec §2.1 解释：CfC closed-form τ = sigmoid(time_a(x) * t + time_b(x))
                我们给 time_a 的输出加 persona-conditioned bias
                让不同 persona 的 hidden neuron 学到不同时间常数
    """

    def __init__(self, cfg: SpikeConfig, base_cfc: CfC):
        super().__init__()
        self.cfg = cfg
        self.cfc = base_cfc
        # persona → hidden 维度的 τ-bias offset
        self.persona_tau_bias = nn.Embedding(cfg.num_personas, cfg.hidden_dim)
        nn.init.zeros_(self.persona_tau_bias.weight)
        # 故意给 persona 0 和 3 不同的 bias 信号，验证扰动确实传到 hidden
        with torch.no_grad():
            self.persona_tau_bias.weight[0] = 0.5
            self.persona_tau_bias.weight[3] = -0.5

    def forward(self, x: torch.Tensor, persona_idx: torch.Tensor) -> torch.Tensor:
        """注入 persona τ-bias 到 ncps CfCCell.time_a 输出

        Note: ncps 1.0.1 的 CfC.forward 内部循环 cell.forward(x_t, h, t)，
              我们 monkey-patch cell.time_a.forward 临时把 bias 加到输出上
        """
        cells = self._get_cells()
        bias = self.persona_tau_bias(persona_idx)  # (batch, hidden_dim)

        # monkey-patch: 临时改写 time_a.forward，加 persona bias
        original_forwards = []
        for cell in cells:
            orig = cell.time_a.forward
            original_forwards.append(orig)

            def new_forward(input_x, _orig=orig, _bias=bias):
                # input_x shape: (batch, fused_dim)
                out = _orig(input_x)  # (batch, hidden_dim)
                return out + _bias

            cell.time_a.forward = new_forward

        try:
            y, _ = self.cfc(x)
        finally:
            # 还原，防止污染后续 forward
            for cell, orig in zip(cells, original_forwards):
                cell.time_a.forward = orig
        return y

    def _get_cells(self) -> list:
        """ncps CfC 内部 cells 列表（实测 1.0.1 用 ModuleList）"""
        if hasattr(self.cfc, "rnn_cell"):
            return [self.cfc.rnn_cell]
        if hasattr(self.cfc, "cells"):
            return list(self.cfc.cells)
        # fallback: find all CfCCell descendants
        from ncps.torch.cfc_cell import CfCCell
        return [m for m in self.cfc.modules() if isinstance(m, CfCCell)]


# -----------------------------------------------------------------------------
# Path C: 占位（手写 CfCCell 单步隐式欧拉）
# -----------------------------------------------------------------------------
def path_c_placeholder() -> str:
    """Path C: 手写 CfCCell 单步隐式欧拉求解器（占位）

    γ-2 若 Path A+B 都不能让 τ 真进 ODE，则启用此路径。
    实现思路：
        τ_i = τ_base * exp(persona_bias_i)  # spec §2.1 公式
        h_t = h_{t-1} + dt * (-h_{t-1}/τ + f(x_t, h_{t-1}))  # 单步隐式欧拉
    工时：+1 天（spec γ-2 任务表注明）
    """
    return "PLACEHOLDER · γ-2 才需要"


# -----------------------------------------------------------------------------
# 验证脚本：跑三路径 + paired t-test
# -----------------------------------------------------------------------------
def run_spike(cfg: SpikeConfig | None = None) -> dict:
    cfg = cfg or SpikeConfig()
    torch.manual_seed(cfg.seed)

    # 输入：固定一份 x 和 persona_idx，让对比公平
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.input_dim)
    persona_idx = torch.arange(cfg.batch_size) % cfg.num_personas
    # persona_timescale: persona 0 慢, 3 快, 中间梯度
    persona_timescale = torch.tensor([2.0, 1.2, 0.8, 0.5])

    # ---------- baseline: 普通 CfC，dt=1.0 (ncps 默认) ----------
    base = CfC(cfg.input_dim, cfg.hidden_dim, return_sequences=True)
    with torch.no_grad():
        y_base, _ = base(x)

    # ---------- Path A: ABANDONED（ncps timespans broadcast bug 实测确认）----------
    # 详见 path_a_status() docstring

    # ---------- Path B: per-sample persona τ-bias (核心路径) ----------
    wrapper_b = PathBWrapper(cfg, base)
    with torch.no_grad():
        y_b = wrapper_b(x, persona_idx)

    # ---------- 差异显著性 ----------
    diff_b = (y_b - y_base).pow(2).sum(dim=(1, 2)).sqrt().numpy()
    zeros = torch.zeros(cfg.batch_size).numpy()
    t_b, p_b = stats.ttest_rel(diff_b, zeros)
    mean_b = float(diff_b.mean())

    # persona separability: 不同 persona 的 hidden 距离应大于同 persona 内
    def persona_separation(y: torch.Tensor) -> float:
        """计算 inter-persona / intra-persona 距离比，> 1 表示 persona 真生效"""
        flat = y.reshape(cfg.batch_size, -1)
        same = 0.0; same_n = 0
        diff = 0.0; diff_n = 0
        for i in range(cfg.batch_size):
            for j in range(i + 1, cfg.batch_size):
                d = (flat[i] - flat[j]).pow(2).sum().sqrt().item()
                if persona_idx[i] == persona_idx[j]:
                    same += d; same_n += 1
                else:
                    diff += d; diff_n += 1
        return (diff / max(diff_n, 1)) / max(same / max(same_n, 1), 1e-6)

    # Path B 核心：persona separation 必须 > 1.1（不同 persona 间距 > 同 persona 内）
    sep_b = persona_separation(y_b)

    return {
        "config": dataclasses.asdict(cfg),
        "path_a": {
            "status": path_a_status(),
            "verdict": "ABANDONED",
        },
        "path_b": {
            "mean_hidden_diff_vs_baseline": mean_b,
            "paired_t_test_t": float(t_b),
            "paired_t_test_p": float(p_b),
            "persona_separation_ratio": sep_b,
            "verdict": "REAL" if p_b < 0.05 and sep_b > 1.1 else "WEAK",
        },
        "path_c": {"status": path_c_placeholder()},
    }


def _print_report(result: dict) -> None:
    print("=" * 70)
    print("γ-1.2 ncps τ-bias 三路径 spike 报告")
    print("=" * 70)
    print(f"配置: {result['config']}")
    print()
    print(f"[PATH_A] verdict={result['path_a']['verdict']}")
    print(f"  {result['path_a']['status']}")
    print()
    info_b = result["path_b"]
    print(f"[PATH_B] verdict={info_b['verdict']}")
    print(f"  hidden diff vs baseline: {info_b['mean_hidden_diff_vs_baseline']:.4f}")
    print(f"  paired t-test p:         {info_b['paired_t_test_p']:.6e}")
    print(f"  persona separation:      {info_b['persona_separation_ratio']:.3f}x")
    print()
    print(f"[PATH_C] {result['path_c']['status']}")
    print()
    b_ok = info_b["verdict"] == "REAL"
    if b_ok:
        print("✅ Path B 通过 preflight #12（τ 真进 ODE，p<0.05 + sep>1.1）")
    else:
        print("❌ Path B WEAK → γ-2 启动 Path C 手写 CfCCell（+1 天工时）")


if __name__ == "__main__":
    result = run_spike()
    _print_report(result)
    # 同时落盘 JSON
    import json
    from pathlib import Path
    out = Path(__file__).resolve().parents[2] / "poc" / "spike" / "tau_bias_routes_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n报告落盘: {out}")
