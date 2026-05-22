"""罕见 AU 重建监控（gemini round 14 + round 2 对抗审改良）.

防 N=124 时模型刷主成分指标而牺牲长尾罕见 AU（AU07 / AU10 类）的重建.

ROADMAP P0-2 (5/23+) 落地：按 AU 频率分桶 (top-7 / mid-6 / bot-7) 独立算
L_world_model + model_vs_persistence_ratio，输出 green / yellow / red verdict.

接口设计原则：
- 纯函数，无副作用，零侵入 evaluate.py 主流程
- AU 段约束 = event_tensor[:, :20]（py-feat 真 AU，剔除 head pose padding）
- V0.3 已用川哥 14 天 + VEATIC 10 vid 双数据集校准，N=124 全量数据准备好后还需再校准一次

合约见 docs/decisions/025-rare-au-longtail-monitor.md + 026-rare-au-monitor-v03-cross-dataset-validation.md.
"""
from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# AU 段约束：py-feat 实际输出 20 AU，event_tensor[20:24] 是 head pose 零 padding
# -----------------------------------------------------------------------------
AU_DIM_START = 0
AU_DIM_END = 20  # exclusive，真 AU 段

# gemini D3 Finding 3: 0.5 阈值会让 AU43（眨眼，常 0.2-0.4）激活率算成 0 被误归类
# py-feat 0.6.2 AU 输出范围 [0,1]，长尾微表情强度天生弱，用 0.1 才能捕捉
AU_ACTIVATION_THRESHOLD = 0.1

# gemini D3 Finding 1: 分母用 max(pers, 1e-9) 会让恒 0 AU 列 ratio 数值爆表 → false red
# 0.001 = 工程 "感知无关阈值"，AU 强度差 ≤ 0.001 视作 indistinguishable from noise
RATIO_DENOM_SMOOTHING = 1e-3

# gemini D3 Finding 2: spread 检查只在罕见桶本身表现不佳时才触发
# 模型在罕见 AU 上比 trivial 强一倍以上（ratio_bottom < 0.5）说明长尾未被牺牲，不应警告
SPREAD_GATE_RATIO_BOTTOM = 0.5

# V0.3 D'（gemini round 2 改良）：跨数据集共识罕见 AU07/AU10 作为固定 bottom 候选
# py-feat 0.6.2 PYFEAT_AU_COLUMNS 列序：AU07 = index 5, AU10 = index 7
DEFAULT_FIXED_BOTTOM_INDICES: tuple[int, ...] = (5, 7)

# V0.4 cc Y2 实证发现：当 ratio_top > TOP_COLLAPSE_RATIO（模型在高频桶也比 trivial 差）
# 说明模型在当前数据集上整体崩盘，"长尾被牺牲"语义不适用，verdict 应判 N/A 而非 green
# 触发场景：训练分布与评估分布严重偏移（如 VEATIC 30fps 训练 → daemon 1fps 评估）
TOP_COLLAPSE_RATIO = 1.0


@dataclasses.dataclass(frozen=True)
class AuBuckets:
    """按激活频率分桶的 AU 索引（相对 event_tensor[:, :20] 的列索引）."""

    top: tuple[int, ...]      # 高频 7 个
    median: tuple[int, ...]   # 中频 6 个
    bottom: tuple[int, ...]   # 罕见 7 个（防 N=124 时被牺牲的目标）
    activation_freq: tuple[float, ...]  # 长度 20，原始激活频率
    evicted_fixed: tuple[int, ...] = ()  # 被踢出的 fixed（因在当前数据 top-K）

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "top": list(self.top),
            "median": list(self.median),
            "bottom": list(self.bottom),
            "evicted_fixed": list(self.evicted_fixed),
        }


@dataclasses.dataclass(frozen=True)
class RareAuVerdict:
    """长尾监控判定结果."""

    # per-bucket world model loss (MSE on rollout)
    L_world_top: float
    L_world_median: float
    L_world_bottom: float

    # per-bucket vs persistence baseline ratio（< 1 表示模型比 trivial 强）
    ratio_top: float
    ratio_median: float
    ratio_bottom: float

    # 长尾倾斜程度：bottom_ratio − top_ratio
    # > 0 表示罕见 AU 改善差于高频 AU（不健康）
    ratio_spread: float

    # green / yellow / red
    verdict: str

    # 触发的判定原因（便于追因）
    reason: str

    # 分桶时的频率分布（debug）
    buckets: AuBuckets

    n_samples: int  # 评估的 window 数


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def compute_au_activation_frequency(
    dataset,
    au_start: int = AU_DIM_START,
    au_end: int = AU_DIM_END,
    threshold: float = AU_ACTIVATION_THRESHOLD,
) -> np.ndarray:
    """计算每个 AU 维度的激活频率 = (au_value > threshold).mean() over all frames.

    Args:
        dataset: VEATICDataset，每个 sample 含 event_tensor (T, 52)
        au_start: AU 段起始列（默认 0）
        au_end: AU 段终止列 exclusive（默认 20，剔除 head pose padding）
        threshold: 激活阈值（py-feat AU 输出 [0,1]）

    Returns:
        shape (au_end - au_start,) 的激活频率向量，每个值 ∈ [0, 1]
    """
    all_au = []
    for i in range(len(dataset)):
        sample = dataset[i]
        et = sample["event_tensor"]
        au = et[:, au_start:au_end].detach().cpu().numpy()
        all_au.append(au)
    mat = np.concatenate(all_au, axis=0)  # (total_frames, au_dim)
    return (mat > threshold).mean(axis=0).astype(np.float64)


def partition_au_by_frequency(
    activation_freq: np.ndarray,
    top_k: int = 7,
    bottom_k: int = 7,
    fixed_bottom_indices: tuple[int, ...] = DEFAULT_FIXED_BOTTOM_INDICES,
) -> AuBuckets:
    """按激活频率把 AU 索引划分 top / median / bottom 三桶.

    V0.3 D'：fixed 仅作为候选，若当前数据里该 AU 已进入 top-K，则自动踢出（evict），
    bottom 槽位改由纯频率低位补齐，防止把高频 AU 强塞进罕见桶.

    Args:
        activation_freq: shape (au_dim,)
        top_k: 高频桶大小（默认 7）
        bottom_k: 罕见桶大小（默认 7）
        fixed_bottom_indices: 优先落入 bottom 桶的 AU 索引（跨数据集先验）.
            按 PYFEAT_AU_COLUMNS 列序，默认 (5=AU07, 7=AU10).
            传 () 退化为纯按频率分桶.

    Returns:
        AuBuckets，索引按"原 AU 列号"排序（便于阅读）
    """
    au_dim = len(activation_freq)
    if top_k + bottom_k > au_dim:
        raise ValueError(
            f"top_k({top_k}) + bottom_k({bottom_k}) > au_dim({au_dim})"
        )
    fixed_set = set(fixed_bottom_indices)
    if any(i < 0 or i >= au_dim for i in fixed_set):
        raise ValueError(
            f"fixed_bottom_indices {fixed_bottom_indices} 越界 au_dim={au_dim}"
        )
    if len(fixed_set) > bottom_k:
        raise ValueError(
            f"fixed_bottom_indices 数量 {len(fixed_set)} > bottom_k {bottom_k}"
        )

    # 先做全量 rank，检查 fixed 是否已进入 top-K（高频就踢出）
    order = np.argsort(-activation_freq)
    freq_rank = np.empty(au_dim, dtype=np.int64)
    freq_rank[order] = np.arange(1, au_dim + 1, dtype=np.int64)
    evicted_fixed = sorted(i for i in fixed_set if int(freq_rank[i]) <= top_k)
    effective_fixed = fixed_set - set(evicted_fixed)

    # 对剩余索引按频率降序分 top，按低频补 bottom
    remaining = [i for i in range(au_dim) if i not in effective_fixed]
    remaining_sorted = sorted(remaining, key=lambda i: -activation_freq[i])

    top_idx = sorted(remaining_sorted[:top_k])
    bottom_remaining_slots = bottom_k - len(effective_fixed)
    if bottom_remaining_slots > 0:
        bottom_from_freq = remaining_sorted[-bottom_remaining_slots:]
    else:
        bottom_from_freq = []
    bottom_idx = sorted(list(effective_fixed) + list(bottom_from_freq))

    used = set(top_idx) | set(bottom_idx)
    median_idx = sorted(i for i in range(au_dim) if i not in used)

    return AuBuckets(
        top=tuple(top_idx),
        median=tuple(median_idx),
        bottom=tuple(bottom_idx),
        evicted_fixed=tuple(evicted_fixed),
        activation_freq=tuple(activation_freq.tolist()),
    )


def compute_per_bucket_mse(
    x_pred: torch.Tensor,
    x_target: torch.Tensor,
    buckets: AuBuckets,
    au_start: int = AU_DIM_START,
) -> dict[str, float]:
    """对每个 bucket 单独算 MSE.

    Args:
        x_pred: (B, T_roll, D) 模型 rollout 预测
        x_target: (B, T_roll, D) ground truth
        buckets: AuBuckets，索引相对 AU 段（[au_start:au_end]）
        au_start: AU 段起始列，用于把 bucket 相对索引偏移到 event_tensor 绝对索引

    Returns:
        {"top": float, "median": float, "bottom": float}
    """
    if x_pred.shape != x_target.shape:
        raise ValueError(f"x_pred {tuple(x_pred.shape)} != x_target {tuple(x_target.shape)}")

    out: dict[str, float] = {}
    for name, idx_rel in [("top", buckets.top), ("median", buckets.median), ("bottom", buckets.bottom)]:
        if len(idx_rel) == 0:
            out[name] = float("nan")
            continue
        idx_abs = [i + au_start for i in idx_rel]
        cols = torch.tensor(idx_abs, dtype=torch.long, device=x_pred.device)
        pred_sub = x_pred.index_select(dim=-1, index=cols)
        tgt_sub = x_target.index_select(dim=-1, index=cols)
        out[name] = float(F.mse_loss(pred_sub, tgt_sub).item())
    return out


def evaluate_rare_au_drop(
    world_losses: dict[str, float],
    persistence_losses: dict[str, float],
    n_samples: int,
    buckets: AuBuckets,
    spread_yellow: float = 0.3,
    spread_red: float = 0.5,
    ratio_red: float = 1.0,
) -> RareAuVerdict:
    """综合 per-bucket world loss + persistence baseline ratio 得出 verdict.

    判定规则（V0.1 起步，gemini 14 + N=124 baseline 后必须迭代）：

    - ratio_X = L_world_X / max(L_persistence_X, eps)
    - 模型在 bucket X 上比 persistence 强 ⇔ ratio_X < 1
    - ratio_spread = ratio_bottom - ratio_top

    | verdict | 触发条件 |
    |---|---|
    | red    | ratio_bottom ≥ ratio_red (罕见 AU 比 trivial 还差) |
    | red    | ratio_bottom ≥ spread_gate AND spread ≥ spread_red |
    | yellow | ratio_bottom ≥ spread_gate AND spread ≥ spread_yellow |
    | green  | 其他（罕见 AU 改善与高频相当，或罕见 AU 已强于 trivial 2x+） |

    gemini D3 Finding 1 (smoothing): ratio = world / (pers + 1e-3) 防恒 0 AU 分母塌
    gemini D3 Finding 2 (gate): spread 只在 ratio_bottom ≥ 0.5 时才检查，避免"优秀惩罚"

    Args:
        world_losses: {"top","median","bottom"} from compute_per_bucket_mse on pred/target
        persistence_losses: 同上，pred 用 prefix 最后一帧反复
        n_samples: 评估窗口数（便于追因）
        buckets: AuBuckets，写进 verdict 便于复盘
        spread_yellow: spread 黄区阈值，V0.2=0.3
        spread_red: spread 红区阈值，V0.2=0.5
        ratio_red: 罕见 bucket ratio 红线，V0.2=1.0（≥1 = 比 trivial 差）
    """

    def _smoothed_ratio(num: float, den: float) -> float:
        return float(num / (den + RATIO_DENOM_SMOOTHING))

    r_top = _smoothed_ratio(world_losses["top"], persistence_losses["top"])
    r_mid = _smoothed_ratio(world_losses["median"], persistence_losses["median"])
    r_bot = _smoothed_ratio(world_losses["bottom"], persistence_losses["bottom"])
    spread = r_bot - r_top
    spread_gate_active = r_bot >= SPREAD_GATE_RATIO_BOTTOM

    # V0.4 cc Y2 finding：整体崩盘检查（top 桶比 trivial 差 → 长尾监控无意义）
    if r_top >= TOP_COLLAPSE_RATIO:
        verdict = "n/a"
        reason = (
            f"top ratio {r_top:.3f} >= top_collapse {TOP_COLLAPSE_RATIO}"
            f"（模型在高频桶也比 trivial 差，整体崩盘，长尾监控不适用）"
        )
    elif r_bot >= ratio_red:
        verdict = "red"
        reason = f"bottom ratio {r_bot:.3f} >= ratio_red {ratio_red}（罕见 AU 比 trivial 还差）"
    elif spread_gate_active and spread >= spread_red:
        verdict = "red"
        reason = (
            f"spread {spread:.3f} >= spread_red {spread_red} "
            f"且 bottom ratio {r_bot:.3f} >= gate {SPREAD_GATE_RATIO_BOTTOM}"
            f"（长尾被严重牺牲）"
        )
    elif spread_gate_active and spread >= spread_yellow:
        verdict = "yellow"
        reason = (
            f"spread {spread:.3f} >= spread_yellow {spread_yellow} "
            f"且 bottom ratio {r_bot:.3f} >= gate {SPREAD_GATE_RATIO_BOTTOM}"
            f"（轻度倾斜）"
        )
    elif spread_gate_active:
        verdict = "green"
        reason = (
            f"spread {spread:.3f} < spread_yellow {spread_yellow}（长尾未被牺牲）"
        )
    else:
        verdict = "green"
        reason = (
            f"bottom ratio {r_bot:.3f} < gate {SPREAD_GATE_RATIO_BOTTOM} "
            f"（罕见 AU 已强于 trivial 2x+，spread 检查跳过）"
        )

    return RareAuVerdict(
        L_world_top=world_losses["top"],
        L_world_median=world_losses["median"],
        L_world_bottom=world_losses["bottom"],
        ratio_top=r_top,
        ratio_median=r_mid,
        ratio_bottom=r_bot,
        ratio_spread=spread,
        verdict=verdict,
        reason=reason,
        buckets=buckets,
        n_samples=n_samples,
    )


def verdict_to_dict(v: RareAuVerdict) -> dict:
    """便于 merge 进 run_evaluation 的 result dict（去掉 dataclass）."""
    return {
        "L_world_top": v.L_world_top,
        "L_world_median": v.L_world_median,
        "L_world_bottom": v.L_world_bottom,
        "ratio_top": v.ratio_top,
        "ratio_median": v.ratio_median,
        "ratio_bottom": v.ratio_bottom,
        "ratio_spread": v.ratio_spread,
        "rare_au_verdict": v.verdict,
        "rare_au_reason": v.reason,
        "rare_au_buckets": v.buckets.as_dict(),
        "rare_au_activation_freq": list(v.buckets.activation_freq),
        "rare_au_evicted_fixed": list(v.buckets.evicted_fixed),
        "rare_au_n_samples": v.n_samples,
    }
