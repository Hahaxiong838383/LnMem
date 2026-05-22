"""γ-3 Step 4: 评估 M1-V Visual-Only + M5 video_id Shortcut + collapse 四联

spec §6.1 + §11.2 验收门:
- M1-V: 25d 文本特征置零后 KG retrieval Top-5 ≥ 20%（spike 用 cosine sim 模拟 KG）
- M5: hidden → video_id linear probe acc < 30% green / < 50% yellow / > 50% red
- collapse 四联: var > 0.01 trained / dist > 0.1 / linear_probe > 1.5x random / KG retrieval > base v1
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .rare_au_monitor import (
    compute_au_activation_frequency,
    compute_per_bucket_mse,
    evaluate_rare_au_drop,
    partition_au_by_frequency,
    verdict_to_dict,
)


# -----------------------------------------------------------------------------
# M1-V Visual-Only mask helper
# -----------------------------------------------------------------------------
def mask_text_features(event_tensor: torch.Tensor) -> torch.Tensor:
    """spec §6.1 M1-V: 把 20d 文本（万能语法槽含 intent）置 0，只留视觉 + 时间 + 音频占位.

    Args:
        event_tensor: (..., 52)
            - [0:24] AU 视觉 ← 保留
            - [24:44] 20 槽含 intent ← **置 0**
            - [44:46] 2 时间戳 ← 保留
            - [46:52] 6 音频占位 ← 保留

    Returns:
        masked (..., 52) 同形状但 [24:44] 全 0（20 维文本特征置 0）
    """
    out = event_tensor.clone()
    out[..., 24:44] = 0.0
    return out


# -----------------------------------------------------------------------------
# M5 video_id Linear Probe（防视频背景 shortcut）
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class VideoIDProbeResult:
    train_acc: float
    test_acc: float
    n_videos: int
    n_train_samples: int
    n_test_samples: int
    verdict: str  # "green" / "yellow" / "red"


def evaluate_video_id_shortcut(
    hidden_states: np.ndarray,
    video_ids: list[str],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> VideoIDProbeResult:
    """spec §6.1 M5: linear probe video_id from hidden state.

    Args:
        hidden_states: (N, hidden_dim) 提取的 hidden 表示
        video_ids: 长度 N 的视频 ID 列表

    Returns:
        VideoIDProbeResult, 含 train/test acc + green/yellow/red verdict
    """
    from sklearn.model_selection import train_test_split
    from collections import Counter
    n_videos = len(set(video_ids))
    rng = np.random.RandomState(seed)
    vid_to_int = {vid: i for i, vid in enumerate(sorted(set(video_ids)))}
    y = np.array([vid_to_int[v] for v in video_ids])

    # stratify 需要每类至少 2 个；sample 不足时退化为非 stratified split
    counts = Counter(y.tolist())
    min_per_class = min(counts.values())
    can_stratify = (n_videos > 1) and (min_per_class >= 2)

    X_train, X_test, y_train, y_test = train_test_split(
        hidden_states, y, test_size=test_ratio, random_state=seed,
        stratify=y if can_stratify else None,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, random_state=seed, multi_class="multinomial")
    clf.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_s))
    test_acc = accuracy_score(y_test, clf.predict(X_test_s))

    # spec §6.1 三层梯度
    if test_acc < 0.30:
        verdict = "green"
    elif test_acc < 0.50:
        verdict = "yellow"
    else:
        verdict = "red"

    return VideoIDProbeResult(
        train_acc=float(train_acc),
        test_acc=float(test_acc),
        n_videos=n_videos,
        n_train_samples=len(X_train),
        n_test_samples=len(X_test),
        verdict=verdict,
    )


# -----------------------------------------------------------------------------
# M1-V KG Retrieval（spike 阶段用 cosine sim 模拟 KG Top-K）
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class RetrievalResult:
    top1_acc: float
    top5_acc: float
    n_queries: int
    method: str  # "full_input" / "visual_only"


def evaluate_retrieval_topk(
    query_hidden: np.ndarray,
    doc_hidden: np.ndarray,
    query_video_ids: list[str],
    doc_video_ids: list[str],
    k: int = 5,
) -> RetrievalResult:
    """KG retrieval 模拟：用 hidden cosine sim 找 Top-K，检验是否同视频.

    spike 阶段简化：每个 query 找 Top-K doc，如果同视频在 Top-K 内即命中.
    """
    # L2 normalize 后内积 = cosine
    q = query_hidden / (np.linalg.norm(query_hidden, axis=1, keepdims=True) + 1e-9)
    d = doc_hidden / (np.linalg.norm(doc_hidden, axis=1, keepdims=True) + 1e-9)

    sim = q @ d.T  # (n_q, n_d)
    # 排除 self-retrieval
    np.fill_diagonal(sim, -np.inf)
    # Top-K
    topk_indices = np.argsort(-sim, axis=1)[:, :k]

    top1_hits, topk_hits = 0, 0
    for i, q_vid in enumerate(query_video_ids):
        retrieved_vids = [doc_video_ids[j] for j in topk_indices[i]]
        if retrieved_vids[0] == q_vid:
            top1_hits += 1
        if q_vid in retrieved_vids:
            topk_hits += 1

    return RetrievalResult(
        top1_acc=top1_hits / len(query_video_ids),
        top5_acc=topk_hits / len(query_video_ids),
        n_queries=len(query_video_ids),
        method="full_input",
    )


# -----------------------------------------------------------------------------
# Full evaluation pipeline
# -----------------------------------------------------------------------------
def measure_input_effective_dim(dataset) -> float:
    """ADR-024: 测量 AU 输入数据本身的 effective_dim 物理上限.

    11 轮血泪教训：5/27 阈值必须基于 input baseline 的相对值，不能拍 absolute。
    AU 24 维但 FACS 文献早就说高度相关，实测 ~5.07 (10 视频)。
    N=124 时预计升到 6-8。

    Returns:
        effective_dim = sum(s) / s[0]，AU 段 (first 24 dims of event_tensor)
    """
    all_au = []
    for i in range(len(dataset)):
        sample = dataset[i]
        et = sample["event_tensor"]
        # AU schema: first 24 dims are AU values, rest is placeholder
        au = et[:, :24].detach().cpu().numpy()
        all_au.append(au)
    mat = np.concatenate(all_au, axis=0)
    centered = mat - mat.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    return float(s.sum() / max(s[0], 1e-9))


def shuffle_frames_in_window(et: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """硬门 #2 Shuffle Test: 打乱 window 内帧顺序（保持 batch 维度不变）.

    Args:
        et: (B, T, D) 或 (T, D)
    Returns:
        same shape, T 维度被打乱
    """
    if et.dim() == 2:
        T = et.size(0)
        perm = torch.randperm(T, generator=torch.Generator().manual_seed(seed))
        return et[perm]
    else:
        T = et.size(1)
        perm = torch.randperm(T, generator=torch.Generator().manual_seed(seed))
        return et[:, perm, :]


def run_evaluation(
    trainer,  # TinyTrainer instance
    dataset,  # VEATICDataset
    sample_per_video: int = 5,
    include_shuffle_test: bool = True,    # 硬门 #2 默认开启
    spike_phase: bool = True,             # 硬门 #4: spike 阶段 M1-V N/A
) -> dict:
    """跑完整 γ-3 evaluation: M1 + M1-V + M5 + collapse + Shuffle Test.

    Args:
        trainer: 训练好的 TinyTrainer
        dataset: VEATICDataset
        sample_per_video: 每视频抽多少窗口做评估
        include_shuffle_test: 硬门 #2 gemini 第 5 轮钦定
        spike_phase: True=spike 阶段 M1-V 标 N/A（gemini 硬门 #4）
    """
    from .dataset import split_to_windows

    if getattr(trainer.cfg, "use_ode_loss", False):
        world_losses_by_vid: dict[str, list[float]] = {}
        world_losses_all: list[float] = []
        world_losses_shuffled: list[float] = []
        prefix_hidden_rows: list[np.ndarray] = []
        world_pred_rollouts: list[torch.Tensor] = []
        world_target_rollouts: list[torch.Tensor] = []
        persistence_pred_rollouts: list[torch.Tensor] = []
        # M1 baseline buffers
        naive_baseline_losses: list[float] = []
        persistence_baseline_losses: list[float] = []
        activation_freq = compute_au_activation_frequency(dataset)
        buckets = partition_au_by_frequency(activation_freq)

        trainer.fusion.eval()
        trainer.backbone.eval()
        persona_idx = torch.zeros(1, dtype=torch.long, device=trainer.device)
        input_stride = int(getattr(trainer.ode_loss.cfg, "input_stride", 1))

        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                windows = split_to_windows(sample, seq_len=trainer.cfg.seq_len, stride=trainer.cfg.seq_len)
                for w_idx, w in enumerate(windows[:sample_per_video]):
                    et = w["event_tensor"].unsqueeze(0).to(trainer.device)
                    out = trainer.ode_loss(
                        x=et,
                        backbone_step_fn=trainer.backbone_step,
                        persona_idx=persona_idx,
                        init_hidden_fn=lambda b, dev: [
                            cell.init_hidden(b, dev) for cell in trainer.backbone.cells
                        ],
                    )
                    world_loss = float(out["L_world_model"].item())
                    vid = str(w["video_id"])
                    world_losses_by_vid.setdefault(vid, []).append(world_loss)
                    world_losses_all.append(world_loss)
                    prefix_hidden_rows.append(out["h_seq_prefix"][0].detach().cpu().numpy())

                    # M1 (cc 第 11 轮整改): 加 trivial baseline 对比，暴露均值/persistence 重构捷径
                    pf_len = int(out["prefix_len"])
                    rl_len = int(out["rollout_len"])
                    et_strided = et[:, ::input_stride, :] if input_stride > 1 else et
                    x_rollout_target_t = et_strided[:, pf_len : pf_len + rl_len, :]
                    x_prefix_t = et_strided[:, :pf_len, :]
                    x_pred_rollout_t = out["x_pred_rollout"]
                    world_pred_rollouts.append(x_pred_rollout_t.detach())
                    world_target_rollouts.append(x_rollout_target_t.detach())
                    # baseline 1: 用 prefix 的 per-dim mean 预测 rollout
                    naive_pred = x_prefix_t.mean(dim=1, keepdim=True).expand_as(x_rollout_target_t)
                    naive_mse = float(F.mse_loss(naive_pred, x_rollout_target_t).item())
                    naive_baseline_losses.append(naive_mse)
                    # baseline 2: persistence — 用 prefix 最后一帧反复预测 rollout
                    last_frame = et_strided[:, pf_len - 1 : pf_len, :].expand_as(x_rollout_target_t)
                    persistence_pred_rollouts.append(last_frame.detach())
                    pers_mse = float(F.mse_loss(last_frame, x_rollout_target_t).item())
                    persistence_baseline_losses.append(pers_mse)

                    if include_shuffle_test:
                        et_shuf = shuffle_frames_in_window(et, seed=42 + w_idx)
                        out_shuf = trainer.ode_loss(
                            x=et_shuf,
                            backbone_step_fn=trainer.backbone_step,
                            persona_idx=persona_idx,
                            init_hidden_fn=lambda b, dev: [
                                cell.init_hidden(b, dev) for cell in trainer.backbone.cells
                            ],
                        )
                        world_losses_shuffled.append(float(out_shuf["L_world_model"].item()))

        prefix_mat = np.concatenate(prefix_hidden_rows, axis=0) if prefix_hidden_rows else np.zeros((1, 1))
        prefix_centered = prefix_mat - prefix_mat.mean(axis=0, keepdims=True)
        try:
            s = np.linalg.svd(prefix_centered, compute_uv=False)
            effective_dim = float(s.sum() / max(s[0], 1e-9))
        except Exception:
            effective_dim = -1.0

        vids = sorted(world_losses_by_vid.keys())
        split = max(1, len(vids) // 2)
        train_vids = set(vids[:split])
        unseen_vids = set(vids[split:])
        train_losses = [x for vid, vals in world_losses_by_vid.items() if vid in train_vids for x in vals]
        unseen_losses = [x for vid, vals in world_losses_by_vid.items() if vid in unseen_vids for x in vals]

        result = {
            "n_samples": len(world_losses_all),
            "L_world_model_train": float(np.mean(train_losses)) if train_losses else float("nan"),
            "L_world_model_unseen": float(np.mean(unseen_losses)) if unseen_losses else float("nan"),
            "L_world_model_all": float(np.mean(world_losses_all)) if world_losses_all else float("nan"),
            "effective_dim": effective_dim,
        }
        # M1 整改：暴露 trivial 重构对比指标
        if naive_baseline_losses:
            naive_mean = float(np.mean(naive_baseline_losses))
            pers_mean = float(np.mean(persistence_baseline_losses))
            result["L_naive_mean_baseline"] = naive_mean
            result["L_persistence_baseline"] = pers_mean
            base_world = result["L_world_model_all"]
            # ratio < 1 → model 比 naive 强；ratio ≥ 1 → trivial 路径 win
            result["model_vs_naive_ratio"] = base_world / max(naive_mean, 1e-9)
            result["model_vs_persistence_ratio"] = base_world / max(pers_mean, 1e-9)
            # 验收门：model 必须比 naive 显著强 (ratio < 0.7)
            result["trivial_recon_check_pass"] = (
                result["model_vs_naive_ratio"] < 0.7 and result["model_vs_persistence_ratio"] < 0.7
            )
        if world_pred_rollouts and world_target_rollouts and persistence_pred_rollouts:
            x_world_pred = torch.cat(world_pred_rollouts, dim=0)
            x_world_target = torch.cat(world_target_rollouts, dim=0)
            x_pers_pred = torch.cat(persistence_pred_rollouts, dim=0)
            world_buckets = compute_per_bucket_mse(x_world_pred, x_world_target, buckets)
            pers_buckets = compute_per_bucket_mse(x_pers_pred, x_world_target, buckets)
            rare_au_verdict = evaluate_rare_au_drop(
                world_buckets,
                pers_buckets,
                n_samples=len(world_losses_all),
                buckets=buckets,
            )
            result.update(verdict_to_dict(rare_au_verdict))
        if include_shuffle_test and world_losses_shuffled:
            shuffled_mean = float(np.mean(world_losses_shuffled))
            base_mean = result["L_world_model_all"]
            result["L_world_model_shuffled"] = shuffled_mean
            result["shuffle_increase_ratio"] = (shuffled_mean - base_mean) / max(abs(base_mean), 1e-9)
            result["shuffle_test_pass"] = shuffled_mean >= base_mean * 1.5
        return result

    hidden_full = []
    hidden_visual = []
    hidden_shuffled = []  # 硬门 #2: 打乱帧顺序后 hidden
    video_ids = []

    trainer.fusion.eval()
    trainer.backbone.eval()
    persona_idx = torch.zeros(1, dtype=torch.long, device=trainer.device)

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            windows = split_to_windows(sample, seq_len=trainer.cfg.seq_len, stride=trainer.cfg.seq_len)
            for w_idx, w in enumerate(windows[:sample_per_video]):
                et = w["event_tensor"].unsqueeze(0).to(trainer.device)
                # Full input
                h_full = trainer.encode(et, persona_idx)[0, -1, :]
                hidden_full.append(h_full.cpu().numpy())
                # Visual-only mask
                et_masked = mask_text_features(et)
                h_vis = trainer.encode(et_masked, persona_idx)[0, -1, :]
                hidden_visual.append(h_vis.cpu().numpy())
                # 硬门 #2 Shuffle Test
                if include_shuffle_test:
                    et_shuf = shuffle_frames_in_window(et, seed=42 + w_idx)
                    h_shuf = trainer.encode(et_shuf, persona_idx)[0, -1, :]
                    hidden_shuffled.append(h_shuf.cpu().numpy())
                video_ids.append(w["video_id"])

    hidden_full = np.stack(hidden_full)
    hidden_visual = np.stack(hidden_visual)

    # M5 video_id shortcut
    m5_result = evaluate_video_id_shortcut(hidden_full, video_ids)

    # M1 retrieval
    m1_full = evaluate_retrieval_topk(hidden_full, hidden_full, video_ids, video_ids, k=5)
    m1_visual = evaluate_retrieval_topk(hidden_visual, hidden_visual, video_ids, video_ids, k=5)
    m1_visual.method = "visual_only"

    # collapse 四联
    hidden_var = float(np.var(hidden_full))
    pairwise = np.linalg.norm(
        hidden_full[:, None, :] - hidden_full[None, :, :], axis=-1
    )
    pairwise_dist = float(pairwise[np.triu_indices_from(pairwise, k=1)].mean())

    # 维度坍塌检测（gemini 第 5 轮盲区 #1）
    # SVD 看奇异值衰减，effective_dim = ratio of sum(s) / s[0]（粗糙估算）
    try:
        s = np.linalg.svd(hidden_full - hidden_full.mean(axis=0), compute_uv=False)
        effective_dim = float((s.sum() / max(s[0], 1e-9)))
        s_decay_ratio = float(s[0] / max(s[min(len(s)-1, 10)], 1e-9))
    except Exception:
        effective_dim = -1.0
        s_decay_ratio = -1.0

    result = {
        "n_samples": len(hidden_full),
        "M5_video_id_shortcut": dataclasses.asdict(m5_result),
        "M1_full_retrieval": dataclasses.asdict(m1_full),
        "M1V_visual_only_retrieval": dataclasses.asdict(m1_visual),
        # 硬门 #4: Spike 阶段 M1-V 标 N/A（文本 0 时 M1-V ≡ M1 trivial pass 无意义）
        "M1V_pass": "N/A (Spike: text=0, M1-V≡M1)" if spike_phase
                    else (m1_visual.top5_acc >= 0.20),
        "collapse_var": hidden_var,
        "collapse_pairwise_dist": pairwise_dist,
        "collapse_var_pass": hidden_var > 0.01,
        # 维度坍塌（gemini 第 5 轮盲区 #1）
        "effective_dim": effective_dim,
        "singular_value_decay_top0_vs_10": s_decay_ratio,
    }

    # 硬门 #2 Shuffle Test
    if include_shuffle_test and hidden_shuffled:
        hidden_shuf = np.stack(hidden_shuffled)
        m1_shuf = evaluate_retrieval_topk(hidden_shuf, hidden_shuf, video_ids, video_ids, k=5)
        m1_shuf.method = "shuffled_frames"
        m1_drop = m1_full.top5_acc - m1_shuf.top5_acc
        result["M1_shuffled_retrieval"] = dataclasses.asdict(m1_shuf)
        result["shuffle_test_drop"] = m1_drop
        # gemini 硬门 #2: M1 必须降 > 5%
        result["shuffle_test_pass"] = m1_drop > 0.05

    return result
