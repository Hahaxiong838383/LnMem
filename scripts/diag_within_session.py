"""ADR-040 Phase 2 判别实验: Within-Session CV (gemini 确认审 R2/R3 钦定)

gemini 指出 v2 的 GroupKFold 崩塌有两种相反解释, 必须用域内 CV 判别:
  (a) 真无信号        → 域内也不显著 → 转 FiLM 是缘木求鱼, 该审 AU 提取/标注对齐
  (b) 有信号+域漂移    → 域内显著但跨 session 崩 → 意图 per-person → FiLM 是对的杠杆

做法: 对最大的几个 ui_session 各自独立做 StratifiedKFold(5) 域内 CV (无跨 session
泛化压力, 窗口重叠泄露实测仅 4.2% 可忽略), 对比该 session 自身多数类基线 + 置换 p。

用法: PYTHONPATH=. .venv-poc-fresh/bin/python scripts/diag_within_session.py
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.diag_mutual_information import (
    INTENT_CLASSES, RNG_SEED, feat_0s, feat_10s, load_samples, _entropy_bits,
)

N_PERM = 300
TOP_SESSIONS = 3
MIN_PER_CLASS = 5  # 域内每类至少这么多才跑 5-fold


def _clf(balanced):
    cw = "balanced" if balanced else None
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, C=0.3, class_weight=cw))


def _cv_proba(X, y, balanced):
    min_c = int(np.unique(y, return_counts=True)[1].min())
    k = max(2, min(5, min_c))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RNG_SEED)
    return cross_val_predict(_clf(balanced), X, y, cv=skf, method="predict_proba")


def _mi(X, y):
    proba = _cv_proba(X, y, balanced=False)
    ce = log_loss(y, proba, labels=list(INTENT_CLASSES)) / math.log(2)
    return _entropy_bits(y) - ce


def _evaluate(X, y, tag):
    labels = list(INTENT_CLASSES)
    mi = _mi(X, y)
    proba_f1 = _cv_proba(X, y, balanced=True)
    pred = np.array(labels)[proba_f1.argmax(axis=1)]
    mf1 = f1_score(y, pred, labels=labels, average="macro", zero_division=0)
    rng = np.random.default_rng(RNG_SEED)
    null = np.array([_mi(X, rng.permutation(y)) for _ in range(N_PERM)])
    p = float((np.sum(null >= mi) + 1) / (N_PERM + 1))
    # 该 session 的平凡 Macro-F1 基线 (恒预测多数类)
    maj = Counter(y).most_common(1)[0][0]
    triv = f1_score(y, np.array([maj] * len(y)), labels=labels, average="macro", zero_division=0)
    print(f"  [{tag}] I_lower={mi:+.4f} bits | Macro-F1={mf1:.3f} (平凡基线={triv:.3f}) | 置换p={p:.4f} "
          f"| {'✅显著>基线' if (p < 0.05 and mf1 > triv + 0.02) else '❌不显著'}")
    return {"tag": tag, "mi": mi, "mf1": mf1, "triv": triv, "p": p}


def main():
    samples = load_samples()
    by_sess = {}
    for s in samples:
        by_sess.setdefault(s.group, []).append(s)
    ranked = sorted(by_sess.items(), key=lambda kv: len(kv[1]), reverse=True)
    print(f"共 {len(samples)} 样本, {len(by_sess)} sessions. 域内判别 top{TOP_SESSIONS} 大 session:\n")
    results = []
    for sess, items in ranked[:TOP_SESSIONS]:
        ydist = Counter(s.y for s in items)
        min_c = min(ydist.get(c, 0) for c in INTENT_CLASSES)
        head = f"session {sess[:32]} (n={len(items)}, {dict(ydist)})"
        if min_c < MIN_PER_CLASS:
            print(f"{head}\n  少数类<{MIN_PER_CLASS}, 跳过域内 CV\n")
            continue
        print(head)
        X0 = np.vstack([feat_0s(s) for s in items])
        X10 = np.vstack([feat_10s(s) for s in items])
        y = np.array([s.y for s in items])
        results.append(_evaluate(X0, y, "0s 单帧"))
        results.append(_evaluate(X10, y, "10s 窗口"))
        print()

    # ---- 判别结论 ----
    any_sig = any(r["p"] < 0.05 and r["mf1"] > r["triv"] + 0.02 for r in results)
    print("=" * 60)
    if not results:
        print("判别: 无足够大且类别齐全的 session, 需积累更均衡数据")
    elif any_sig:
        print("判别: 🟡 域内有信号 + 跨session(v2)崩 → DOMAIN SHIFT (意图 per-person)")
        print("       → 支持转向 FiLM/个性化对齐; τ-horizon 仍 moot 但信号存在")
    else:
        print("判别: 🔴 域内也不显著 → 真无信号 (AU→意图标签无线性解释力)")
        print("       → FiLM 缘木求鱼; 优先审计 AU 提取算法 / 标注-窗口时刻对齐 / 标签噪声")
    print("=" * 60)


if __name__ == "__main__":
    main()
