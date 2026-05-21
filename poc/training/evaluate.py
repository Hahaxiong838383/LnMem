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
    n_videos = len(set(video_ids))
    rng = np.random.RandomState(seed)
    # 数字编码 video_id
    vid_to_int = {vid: i for i, vid in enumerate(sorted(set(video_ids)))}
    y = np.array([vid_to_int[v] for v in video_ids])

    X_train, X_test, y_train, y_test = train_test_split(
        hidden_states, y, test_size=test_ratio, random_state=seed, stratify=y if n_videos > 1 else None
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
def run_evaluation(
    trainer,  # TinyTrainer instance
    dataset,  # VEATICDataset
    sample_per_video: int = 5,
) -> dict:
    """跑完整 γ-3 evaluation: M1-V + M5 + collapse.

    Args:
        trainer: 训练好的 TinyTrainer
        dataset: VEATICDataset
        sample_per_video: 每视频抽多少窗口做评估
    """
    from .dataset import split_to_windows

    # 收集 hidden states
    hidden_full = []     # full input
    hidden_visual = []   # 25d 文本 mask 后
    video_ids = []

    trainer.fusion.eval()
    trainer.backbone.eval()
    persona_idx = torch.zeros(1, dtype=torch.long, device=trainer.device)

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            windows = split_to_windows(sample, seq_len=trainer.cfg.seq_len, stride=trainer.cfg.seq_len)
            # 每视频抽 sample_per_video 窗口
            for w in windows[:sample_per_video]:
                et = w["event_tensor"].unsqueeze(0).to(trainer.device)
                # Full input
                h_full = trainer.encode(et, persona_idx)[0, -1, :]
                hidden_full.append(h_full.cpu().numpy())
                # Visual-only mask
                et_masked = mask_text_features(et)
                h_vis = trainer.encode(et_masked, persona_idx)[0, -1, :]
                hidden_visual.append(h_vis.cpu().numpy())
                video_ids.append(w["video_id"])

    hidden_full = np.stack(hidden_full)
    hidden_visual = np.stack(hidden_visual)

    # M5 video_id shortcut
    m5_result = evaluate_video_id_shortcut(hidden_full, video_ids)

    # M1 retrieval: full vs visual-only
    m1_full = evaluate_retrieval_topk(hidden_full, hidden_full, video_ids, video_ids, k=5)
    m1_visual = evaluate_retrieval_topk(hidden_visual, hidden_visual, video_ids, video_ids, k=5)
    m1_visual.method = "visual_only"

    # collapse 四联
    hidden_var = float(np.var(hidden_full))
    pairwise = np.linalg.norm(
        hidden_full[:, None, :] - hidden_full[None, :, :], axis=-1
    )
    pairwise_dist = float(pairwise[np.triu_indices_from(pairwise, k=1)].mean())

    return {
        "n_samples": len(hidden_full),
        "M5_video_id_shortcut": dataclasses.asdict(m5_result),
        "M1_full_retrieval": dataclasses.asdict(m1_full),
        "M1V_visual_only_retrieval": dataclasses.asdict(m1_visual),
        "M1V_pass": m1_visual.top5_acc >= 0.20,  # spec §6.1 M1-V 门槛
        "collapse_var": hidden_var,
        "collapse_pairwise_dist": pairwise_dist,
        "collapse_var_pass": hidden_var > 0.01,  # spec §6.1 trained 阈值
    }
