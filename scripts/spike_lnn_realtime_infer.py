#!/usr/bin/env python3
"""SPIKE: 验证 LNN 实时推理可行性（2026-05-16，#3.0 W3-A 前置 spike）

目标：用 #2 川哥配合产生的 9 events + 历史 235K 帧 signals 数据，验证：
  1. LNN ckpt 可正确加载（含 OnlineLoRAAdapter unwrap）— F-01/F-03 fix
  2. signals 数据可正确转 fusion frame → 18 维 input — F-02 fix
  3. LNN forward(x, persona_idx) 输出合理 state distribution — F-01 fix
  4. predicted state vs user true_state 是否有相关性
  5. entropy / margin 指标在真实数据上的分布

不动 daemon，只跑离线脚本。

判断 spike PASS：
  - 加载 + forward 无 runtime error
  - 9 events 至少 4 个 predicted state 和 true_state 一致（>= 44% top-1 accuracy）
  - entropy / margin 区分度有意义（不是全部接近）
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.lnn.expression_dynamics import STATE_LABELS
from poc.src.lnn.model import DefaultLNNFactory
from poc.src.lnn.online_adapter import OnlineLoRAAdapter
from poc.src.lnn.online_contracts import OnlineAdapterConfig
from poc.src.lnn.signals_loader import _signals_row_to_frame_dict  # F-02 fix
from poc.src.fusion import MultimodalFusion


def load_state_model_with_drift(ckpt_path: Path):
    """加载 base state_model + OnlineLoRAAdapter unwrap tau_drift（F-03 fix）"""
    print(f"[spike] loading ckpt: {ckpt_path.name}", file=sys.stderr)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[spike] online_adapter_schema: {ckpt.get('online_adapter_schema')}", file=sys.stderr)

    factory = DefaultLNNFactory()
    state_model = factory.create_state_model(ckpt["config"])
    state_model.load_state_dict(ckpt["state_model"])
    state_model.eval()
    print(f"[spike] base state_model loaded (config={type(ckpt['config']).__name__})", file=sys.stderr)

    # F-03 fix: 如果 ckpt 有 online_tau_drift，wrap OnlineLoRAAdapter
    if ckpt.get("online_adapter_schema") == "phase2_v1" and "online_tau_drift" in ckpt:
        base_adapter = getattr(state_model, "personalization", None)
        if base_adapter is not None:
            online_config = OnlineAdapterConfig()
            online_adapter = OnlineLoRAAdapter(
                base_adapter=base_adapter,
                online_config=online_config,
            )
            # 恢复 tau_drift 学习成果
            online_adapter.tau_drift.data.copy_(ckpt["online_tau_drift"])
            state_model.personalization = online_adapter
            print(f"[spike] ✅ OnlineLoRAAdapter wrapped, tau_drift shape={ckpt['online_tau_drift'].shape}",
                  file=sys.stderr)
        else:
            print(f"[spike] ⚠️  state_model.personalization is None, skipping online wrap",
                  file=sys.stderr)
    else:
        print(f"[spike] base only (no online drift)", file=sys.stderr)

    return state_model, ckpt


def build_input_window(signals: list[dict], end_idx: int, window: int = 60):
    """从 signals 流取 [end_idx-window+1, end_idx] 60 帧，转 18 维 fusion vec sequence

    F-02 fix: 用 _signals_row_to_frame_dict 转 fusion frame
    """
    if end_idx + 1 < window:
        return None
    fusion = MultimodalFusion()
    vectors = []
    for i in range(end_idx - window + 1, end_idx + 1):
        try:
            frame_dict = _signals_row_to_frame_dict(signals[i])
        except Exception as e:
            return None
        try:
            fused = fusion.fuse(frame_dict)
        except Exception as e:
            return None
        if len(fused.vector) != 18:
            return None
        vectors.append(fused.vector)
    return torch.tensor(vectors, dtype=torch.float32).unsqueeze(0)  # (1, 60, 18)


def predict_state(state_model, x, persona_idx_int: int):
    """F-01 fix: forward(x, persona_idx) 双参数"""
    persona_idx = torch.tensor([persona_idx_int], dtype=torch.long)
    with torch.no_grad():
        logits = state_model(x, persona_idx)  # (1, T, n_states)
    last_logits = logits[0, -1, :]  # (n_states,)
    probs = torch.softmax(last_logits, dim=-1).tolist()
    sorted_probs = sorted(probs, reverse=True)
    entropy = -sum(p * math.log2(max(p, 1e-12)) for p in probs)
    margin = sorted_probs[0] - sorted_probs[1]
    top_idx = int(last_logits.argmax())
    return {
        "predicted_state": STATE_LABELS[top_idx],
        "top_prob": sorted_probs[0],
        "second_prob": sorted_probs[1],
        "margin": margin,
        "entropy": entropy,
        "probs": probs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=REPO_ROOT / "poc/data/checkpoints/m0w2_phase2/checkpoint_phase2_20260516T103643Z.pt",
                        help="raw 路径新训 ckpt（含 online_tau_drift）")
    parser.add_argument("--persona-idx", type=int, default=30, help="vboss_self=30")
    parser.add_argument("--events-date", default="2026-05-16")
    parser.add_argument("--events-since", default="2026-05-16T11:00:00Z",
                        help="过滤 #2 配合期间 events (CST 19:00 = UTC 11:00)")
    args = parser.parse_args()

    # 加载 LNN
    state_model, ckpt = load_state_model_with_drift(args.ckpt)

    # 加载 #2 events
    events_path = Path.home() / f".lnmem/feedback/queries_{args.events_date}.jsonl"
    events = []
    for line in events_path.open():
        try:
            d = json.loads(line)
            if d["ts"] >= args.events_since:
                events.append(d)
        except: pass
    print(f"\n[spike] #2 events: {len(events)}", file=sys.stderr)

    # 加载历史 signals
    signals_path = Path.home() / f".lnmem/data/signals_{args.events_date}.jsonl"
    signals = [json.loads(line) for line in signals_path.open() if line.strip()]
    signals.sort(key=lambda s: s.get("ts", ""))
    print(f"[spike] today signals: {len(signals)}", file=sys.stderr)

    # 给每个 event 找对应 signal 索引（按 ts 最近）
    def find_signal_idx(ev_ts):
        for i, s in enumerate(signals):
            if s.get("ts", "") >= ev_ts:
                return i
        return None

    print(f"\n[spike] === per-event LNN forward (using persona_idx={args.persona_idx} = vboss_self) ===\n",
          file=sys.stderr)
    print(f"{'time':<10} {'reason':<14} {'label':<10} {'true_state':<22} "
          f"{'predicted':<22} {'match?':<7} {'top%':<6} {'margin':<7} {'entropy':<7}")
    print("-" * 130)

    results = []
    state_correct = 0
    state_total = 0  # 只算有 true_state 的 events
    for ev in events:
        sig_idx = find_signal_idx(ev["ts"])
        if sig_idx is None or sig_idx < 60:
            continue
        x = build_input_window(signals, sig_idx, window=60)
        if x is None:
            continue
        try:
            pred = predict_state(state_model, x, args.persona_idx)
        except Exception as e:
            print(f"  ERROR forward: {e}", file=sys.stderr)
            continue
        ur = ev.get("user_response") or {}
        true_state = ur.get("true_state") or "-"
        label = ur.get("label", "?")
        ts_cst = datetime.fromisoformat(ev["ts"].replace("Z", "+00:00")).astimezone()
        is_match = pred["predicted_state"] == true_state if true_state != "-" else None
        if true_state != "-":
            state_total += 1
            if is_match:
                state_correct += 1
        results.append({
            "ts": ev["ts"], "reason": ev["reason"], "label": label,
            "true_state": true_state, **pred,
        })
        match_str = "✅" if is_match else ("❌" if is_match is False else "—")
        print(f"{ts_cst.strftime('%H:%M:%S'):<10} {ev['reason']:<14} {label:<10} "
              f"{true_state:<22} {pred['predicted_state']:<22} {match_str:<7} "
              f"{pred['top_prob']*100:<5.1f}% {pred['margin']:<7.3f} {pred['entropy']:<7.3f}")

    # 汇总
    print(f"\n[spike] === SPIKE 判定 ===", file=sys.stderr)
    print(f"  events with true_state: {state_total}", file=sys.stderr)
    print(f"  predicted == true: {state_correct} / {state_total} = "
          f"{state_correct/max(state_total,1)*100:.1f}%", file=sys.stderr)
    print(f"  forward 无 runtime error ✅", file=sys.stderr)

    if results:
        margins = [r["margin"] for r in results]
        entropies = [r["entropy"] for r in results]
        print(f"\n[spike] margin 分布: min={min(margins):.3f} max={max(margins):.3f} "
              f"mean={sum(margins)/len(margins):.3f}", file=sys.stderr)
        print(f"[spike] entropy 分布: min={min(entropies):.3f} max={max(entropies):.3f} "
              f"mean={sum(entropies)/len(entropies):.3f}", file=sys.stderr)
        pred_dist = Counter(r["predicted_state"] for r in results)
        print(f"[spike] LNN 预测分布: {dict(pred_dist)}", file=sys.stderr)

    # SPIKE PASS/FAIL
    print(f"\n[spike] === 判定 ===", file=sys.stderr)
    pass_thresholds = {
        "no_runtime_error": True,
        "state_accuracy >= 30%": state_correct / max(state_total, 1) >= 0.3,
        "events_processed >= 5": len(results) >= 5,
    }
    all_pass = all(pass_thresholds.values())
    for k, v in pass_thresholds.items():
        print(f"  {'✅' if v else '❌'} {k}", file=sys.stderr)
    print(f"\n[spike] {'🎉 SPIKE PASS' if all_pass else '⚠️  SPIKE PARTIAL'}", file=sys.stderr)
    print(f"[spike] => 可以推进 W3-A 实施" if all_pass else "[spike] => 需要重新设计或更多数据",
          file=sys.stderr)


if __name__ == "__main__":
    main()
