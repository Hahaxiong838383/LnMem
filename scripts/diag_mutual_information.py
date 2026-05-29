"""ADR-040 Phase 2 中立裁判: 信息论互信息诊断 v2 (泄露鲁棒 + 同维对齐)

回答 "intent 任务到底需不需要 >10s 历史" —— 不碰那个短 τ 模型, 直接算原始 AU
特征和川哥真标签 (corrected) 之间的信息量, 从根上绕开 gemini 推翻 lag ablation 的
循环论证 (用短 τ 模型测不需要长 τ)。

## v2 修订 (gemini-3.5-high 对抗审 2026-05-29 + cc 实测复核)
gemini 抓到 v1 的泄露/维度/校准三雷, cc 实测复核后采纳如下 (实测复核见 §下):
  R1 时间泄露: gemini 说"10s 窗重叠90%"→ cc 实测相邻标注 median 间隔 204s,
     仅 4.2% 对 <10s, 窗口重叠泄露很小; 但 session-block 非独立性真实存在
     (10 个 ui_session, 同 session 共享人物/光照/当天基线) → 采纳 GroupKFold。
  R2 维度混淆: 0s(18D) vs 10s(72D) 不公平 → PCA 把三者对齐到同维再比。
  R3 负 I_lower: 高维过拟合推高 CE → PCA 降维 (主治) + 强 L2 (辅) 消负值伪影。
判读以 **GroupKFold(by ui_session) + 同维 PCA 下的置换 p** 为第一裁判
(置换对过拟合伪影鲁棒, GroupKFold 对 session-block 泄露鲁棒)。

三层时程对照 (gemini 二审 R5.2 + ADR-040 Phase 2a/2c):
  - 0s  Reactive : 仅最近一帧 au_window[-1]
  - 10s Recent   : 0s + 最近 10s 窗口动态 (mean/slope/std)
  - long Cond    : 10s + 10-120s 稀疏长程 au_window_long (互补不重叠)

用法: PYTHONPATH=. .venv-poc-fresh/bin/python scripts/diag_mutual_information.py
输出: /tmp/mi_diagnosis_result.md + stdout
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import log_loss, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEEDBACK_GLOB = os.path.expanduser("~/.lnmem/data/intent_feedback/*.jsonl")
AU_DIM = 18
INTENT_CLASSES = ("none", "negative", "positive")
RNG_SEED = 42
N_PERM = 200
N_BOOT = 100
MIN_LONG_FRAMES = 3
PCA_DIM = 12          # 同维对齐目标 (< 0s 的 18D, 三层都压到这个维度公平比)
C_REG = 0.3           # L2 (gemini R3: 比 v1 的 0.5 更强, 配 PCA 消过拟合)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    y: str
    group: str                              # ui_session, GroupKFold 分组键
    recent: list[list[float]]
    long: list[list[float]] = field(default_factory=list)


def _clean_au(frames) -> list[list[float]]:
    out = []
    for fr in frames or []:
        au = fr.get("au")
        if isinstance(au, list) and len(au) == AU_DIM:
            out.append([float(x) for x in au])
    return out


def load_samples() -> list[Sample]:
    samples: list[Sample] = []
    for path in sorted(glob.glob(FEEDBACK_GLOB)):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                y = rec.get("corrected")
                recent = _clean_au(rec.get("au_window"))
                if y not in INTENT_CLASSES or not recent:
                    continue
                samples.append(Sample(
                    y=y,
                    group=str(rec.get("ui_session") or "none"),
                    recent=recent,
                    long=_clean_au(rec.get("au_window_long")),
                ))
    return samples


# ---------------------------------------------------------------------------
# 特征工程 (per-AU 统计, 不裸拼)
# ---------------------------------------------------------------------------
def _snapshot(seq: list[list[float]]) -> np.ndarray:
    return np.asarray(seq[-1], dtype=np.float64)


def _window_dynamics(seq: list[list[float]]) -> np.ndarray:
    """per-AU mean / slope / std → 54 维。MI 不受斜率符号影响, 时序方向反了也不改结论。"""
    arr = np.asarray(seq, dtype=np.float64)
    t = arr.shape[0]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    if t >= 2:
        idx = np.arange(t, dtype=np.float64) - (t - 1) / 2.0
        denom = float((idx ** 2).sum())
        slope = (idx[:, None] * (arr - mean)).sum(axis=0) / denom if denom > 0 else np.zeros(AU_DIM)
    else:
        slope = np.zeros(AU_DIM)
    return np.concatenate([mean, slope, std])


def feat_0s(s: Sample) -> np.ndarray:
    return _snapshot(s.recent)


def feat_10s(s: Sample) -> np.ndarray:
    return np.concatenate([_snapshot(s.recent), _window_dynamics(s.recent)])


def feat_long_block(s: Sample) -> np.ndarray:
    if not s.long:
        return np.zeros(AU_DIM * 3)
    return _window_dynamics(s.long)


# ---------------------------------------------------------------------------
# 互信息估计 (GroupKFold + PCA 同维对齐)
# ---------------------------------------------------------------------------
def _entropy_bits(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _clf(balanced: bool, pca_dim: int | None):
    """标准化 (+ 可选 PCA 同维对齐) + 正则 logistic。"""
    steps = [StandardScaler()]
    if pca_dim is not None:
        steps.append(PCA(n_components=pca_dim, random_state=RNG_SEED))
    cw = "balanced" if balanced else None
    steps.append(LogisticRegression(max_iter=3000, C=C_REG, class_weight=cw))
    return make_pipeline(*steps)


def _n_groups(groups: np.ndarray) -> int:
    return len(np.unique(groups))


def _cv_proba(X, y, groups, balanced, pca_dim):
    """StratifiedGroupKFold out-of-fold predict_proba (同 session 不跨集, 防 block 泄露)。"""
    n_splits = max(2, min(5, _n_groups(groups)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
    pca = pca_dim if (pca_dim is not None and pca_dim < X.shape[1]) else None
    return cross_val_predict(_clf(balanced, pca), X, y, groups=groups, cv=sgkf,
                             method="predict_proba")


def _mi_only(X, y, groups, pca_dim=None) -> float:
    proba = _cv_proba(X, y, groups, balanced=False, pca_dim=pca_dim)
    ce_bits = log_loss(y, proba, labels=list(INTENT_CLASSES)) / math.log(2)
    return _entropy_bits(y) - ce_bits


def mi_lower_bits(X, y, groups, pca_dim=None) -> tuple[float, float, dict]:
    """I_lower = H(Y) - CE_cv (bits) + balanced Macro-F1。"""
    labels = list(INTENT_CLASSES)
    proba_mi = _cv_proba(X, y, groups, balanced=False, pca_dim=pca_dim)
    ce_bits = log_loss(y, proba_mi, labels=labels) / math.log(2)
    h_bits = _entropy_bits(y)
    mi = h_bits - ce_bits
    proba_f1 = _cv_proba(X, y, groups, balanced=True, pca_dim=pca_dim)
    pred = np.array(labels)[proba_f1.argmax(axis=1)]
    macro_f1 = f1_score(y, pred, labels=labels, average="macro", zero_division=0)
    return mi, macro_f1, {"H_bits": h_bits, "CE_bits": ce_bits}


def perm_pvalue(X, y, groups, observed_mi, pca_dim=None) -> float:
    """置换检验: shuffle Y 建零分布 (保 group 结构), p = P(null >= observed)。"""
    rng = np.random.default_rng(RNG_SEED)
    null = np.array([_mi_only(X, rng.permutation(y), groups, pca_dim) for _ in range(N_PERM)])
    return float((np.sum(null >= observed_mi) + 1) / (N_PERM + 1))


def cond_perm_pvalue(X_short, X_long, y, groups, observed_gain, pca_dim=None) -> float:
    """条件增益置换: 仅打乱 long 块, 保 (X_short, y, group)。H0: X_long ⟂ Y | X_short。"""
    rng = np.random.default_rng(RNG_SEED + 1)
    base = _mi_only(X_short, y, groups, pca_dim)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        joint = np.hstack([X_short, X_long[rng.permutation(len(X_long))]])
        null[i] = _mi_only(joint, y, groups, pca_dim) - base
    return float((np.sum(null >= observed_gain) + 1) / (N_PERM + 1))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _stack(samples, featfn):
    X = np.vstack([featfn(s) for s in samples])
    y = np.array([s.y for s in samples])
    g = np.array([s.group for s in samples])
    return X, y, g


def main() -> None:
    samples = load_samples()
    y_all = np.array([s.y for s in samples])
    g_all = np.array([s.group for s in samples])
    long_samples = [s for s in samples if len(s.long) >= MIN_LONG_FRAMES]
    L = ["# Phase 2 信息论互信息诊断 v2 (GroupKFold 泄露鲁棒 + PCA 同维对齐)\n",
         "_v1→v2: 采纳 gemini 对抗审 (GroupKFold by ui_session / PCA 同维 / 强 L2), cc 实测复核窗口重叠泄露仅 4.2% (相邻标注 median 间隔 204s), 真问题是 session-block 非独立性。_\n",
         f"- 总样本: **{len(samples)}** / ui_session 组数: {_n_groups(g_all)}",
         f"- 标签分布 (Y=corrected): " + ", ".join(f"{c}={int((y_all==c).sum())}" for c in INTENT_CLASSES),
         f"- H(Y) = {_entropy_bits(y_all):.4f} bits; 多数类基线 acc={float((y_all=='none').mean()):.3f}",
         f"- 长程子集 (long ≥ {MIN_LONG_FRAMES} 帧): **{len(long_samples)}**",
         f"- 估计器: 模型无关 Ross kNN(per-AU) + I_lower=H−CE_cv; CV=StratifiedGroupKFold; PCA→{PCA_DIM}D 同维; L2 C={C_REG}",
         f"- 置换 {N_PERM} 次, seed={RNG_SEED}; 第一裁判=GroupKFold 下置换 p\n"]
    print("\n".join(L))

    # ---------- 实验 1 · per-AU 模型无关 MI ----------
    X0, _, _ = _stack(samples, feat_0s)
    mi_per_au = mutual_info_classif(X0, y_all, discrete_features=False, random_state=RNG_SEED)
    order = np.argsort(mi_per_au)[::-1]
    L += ["\n## 实验 1 · per-AU 模型无关 MI (0s 单帧, Ross kNN)\n",
          "_每个 AU 通道单独携带多少意图信息 (非联合 MI)。_\n",
          "| 排名 | AU | MI (nats) | 占 H(Y) |", "|---|---|---|---|"]
    for rank, i in enumerate(order[:6], 1):
        L.append(f"| {rank} | AU[{i}] | {mi_per_au[i]:.4f} | {mi_per_au[i]/math.log(2)/_entropy_bits(y_all)*100:.1f}% |")
    print(f"[实验1] per-AU MI top3: " + ", ".join(f"AU[{i}]={mi_per_au[i]:.3f}" for i in order[:3]))

    # ---------- 实验 2 · 0s vs 10s, 原维 + PCA 同维双跑 (GroupKFold) ----------
    def run_pair(tag, pca_dim):
        rows = []
        L.append(f"\n### {tag}\n")
        L.append("| 时程 | I_lower (bits) | 占H(Y) | Macro-F1 | 置换 p(GroupKFold) |")
        L.append("|---|---|---|---|---|")
        for name, featfn in [("0s 单帧", feat_0s), ("10s 窗口", feat_10s)]:
            X, y, g = _stack(samples, featfn)
            mi, mf1, det = mi_lower_bits(X, y, g, pca_dim)
            p = perm_pvalue(X, y, g, mi, pca_dim)
            frac = mi / det["H_bits"] if det["H_bits"] > 0 else 0.0
            rows.append((name, mi, frac, mf1, p))
            L.append(f"| {name} | {mi:.4f} | {frac*100:.1f}% | {mf1:.3f} | {p:.4f} |")
            print(f"[实验2·{tag}] {name}: I_lower={mi:.4f} ({frac*100:.1f}%) MacroF1={mf1:.3f} p={p:.4f}")
        return rows

    L.append("\n## 实验 2 · 0s vs 10s 信息下界 (全量 n={}, GroupKFold)\n".format(len(samples)))
    rows_raw = run_pair("2A 原始维度 (0s=18D, 10s=72D)", None)
    rows_pca = run_pair(f"2B PCA 同维对齐 ({PCA_DIM}D, 排除维度混淆)", PCA_DIM)
    L.append(f"\n- **判读锚点**: 看 2B(同维)的置换 p。若 10s p<0.05 且 0s p≥0.05 → 时序动态真携带信号;")
    L.append(f"  若 2B 下 10s p 也不显著 → v1 的'10s显著'是维度红利/session泄露伪影。")

    # ---------- 实验 3 · 长程条件 (judgment 挂起) ----------
    L += ["\n## 实验 3 · 长程条件增益 I(X_long;Y|X_short) (n_long={}, ⚠️欠功率挂起)\n".format(len(long_samples))]
    lg = np.array([s.group for s in long_samples])
    if len(long_samples) < 10 or len(set(s.y for s in long_samples)) < len(INTENT_CLASSES) or _n_groups(lg) < 2:
        L.append("- ⚠️ 长程子集样本/类别/分组不足, 跳过。继续积累 au_window_long。")
        print("[实验3] 长程子集不足, 跳过")
    else:
        Xs = np.vstack([feat_10s(s) for s in long_samples]); yl = np.array([s.y for s in long_samples])
        Xl = np.vstack([feat_long_block(s) for s in long_samples])
        pca_l = min(PCA_DIM, Xs.shape[1] - 1)
        mi_s, f1_s, _ = mi_lower_bits(Xs, yl, lg, pca_l)
        mi_j, f1_j, _ = mi_lower_bits(np.hstack([Xs, Xl]), yl, lg, pca_l)
        gain = mi_j - mi_s
        p_cond = cond_perm_pvalue(Xs, Xl, yl, lg, gain, pca_l)
        L += [f"- 长程子集标签: " + ", ".join(f"{c}={int((yl==c).sum())}" for c in INTENT_CLASSES) + f"; 组数={_n_groups(lg)}",
              f"- 仅10s I_lower={mi_s:.4f} (F1={f1_s:.3f}); 10s+长程 I_lower={mi_j:.4f} (F1={f1_j:.3f})",
              f"- **长程条件增益** = {gain:+.4f} bits, 条件置换 p={p_cond:.4f}",
              "\n> ⚠️ **JUDGMENT 挂起**: gemini EPV 估算长程判决需每少数类 ≥ ~15×D/p_min 样本",
              f">   (即便 PCA 到 {PCA_DIM}D, 也需 ~1000+ 条带完整长程窗的标注)。n_long={len(long_samples)} 远不够。",
              "> 下负结论会重犯欠功率假阴性 (10秒鱼塘没捞到鲸鱼)。真判决门控在数据量。"]
        print(f"[实验3] 长程条件增益={gain:+.4f} p={p_cond:.4f} (n_long={len(long_samples)}, 挂起)")

    out = os.path.expanduser("/tmp/mi_diagnosis_result.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"\n结果已写 {out}")


if __name__ == "__main__":
    main()
